//
//  GLM4MOE.swift
//  LLM
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm4_moe.py
//  Created by Ronald Mannak on 2025/1/7.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class GLM4MoEAttention: Module {
    let args: GLM4MoEConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPELayer

    init(_ args: GLM4MoEConfiguration) {
        self.args = args

        let headDim = args.headDim > 0 ? args.headDim : args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(args.attentionHeads * headDim, args.hiddenSize, bias: false)

        if args.useQkNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        }

        self.rope = initializeRope(
            dims: Int(Float(headDim) * args.partialRotaryFactor),
            base: args.ropeTheta,
            traditional: false, scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1)
        keys = keys.reshaped(B, L, args.kvHeads, -1)

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class GLM4MoEMLP: Module, UnaryLayer {
    let hiddenSize: Int
    let intermediateSize: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GLM4MoEConfiguration, hiddenSize: Int? = nil, intermediateSize: Int? = nil) {
        self.hiddenSize = hiddenSize ?? config.hiddenSize
        self.intermediateSize = intermediateSize ?? config.intermediateSize

        _gateProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(self.hiddenSize, self.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(self.intermediateSize, self.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class GLM4MoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int
    let scoringFunc: String

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: GLM4MoEConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoEGate requires nRoutedExperts")
        }

        precondition(config.topkMethod == "noaux_tc", "Unsupported topk method.")

        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nRoutedExperts = nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.scoringFunc = config.scoringFunc

        _weight.wrappedValue = zeros([nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = zeros([nRoutedExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let hiddenStates = x.matmul(weight.T)
        var scores: MLXArray
        if scoringFunc == "sigmoid" {
            scores = sigmoid(hiddenStates.asType(.float32))
        } else {
            scores = softmax(hiddenStates.asType(.float32), axis: -1)
        }

        let originalScores = scores
        var selectionScores = scores + eScoreCorrectionBias

        if nGroup > 1 {
            selectionScores = unflatten(selectionScores, axis: -1, shape: [nGroup, -1])
            let groupScores = top(selectionScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            selectionScores = putAlong(
                selectionScores, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
            selectionScores = flattened(selectionScores, start: -2, end: -1)
        }

        let k = topK
        let inds = argPartition(-selectionScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var selectedScores = takeAlong(originalScores, inds, axis: -1)

        if topK > 1, normTopkProb {
            let denominator = selectedScores.sum(axis: -1, keepDims: true)
            selectedScores = selectedScores / denominator
        }
        selectedScores = selectedScores * routedScalingFactor

        return (inds, selectedScores)
    }
}

class GLM4MoE: Module, UnaryLayer {
    let numExpertsPerTok: Int
    let gate: GLM4MoEGate

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: GLM4MoEMLP?

    init(_ config: GLM4MoEConfiguration) {
        guard let nRoutedExperts = config.nRoutedExperts else {
            fatalError("GLM4MoE requires nRoutedExperts")
        }

        self.numExpertsPerTok = config.numExpertsPerTok
        self.gate = GLM4MoEGate(config)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: nRoutedExperts
        )

        if let shared = config.nSharedExperts, shared > 0 {
            let intermediateSize = config.moeIntermediateSize * shared
            _sharedExperts.wrappedValue = GLM4MoEMLP(
                config, intermediateSize: intermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = weightedExpertSum(y, scores).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class GLM4MoEDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: GLM4MoEAttention
    let mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: GLM4MoEConfiguration, layerIdx: Int) {
        _attention.wrappedValue = GLM4MoEAttention(args)

        if args.nRoutedExperts != nil && layerIdx >= args.firstKDenseReplace {
            self.mlp = GLM4MoE(args)
        } else {
            self.mlp = GLM4MoEMLP(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class GLM4MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    // Mutable + optional so `pipeline(group:boundary:)` can nil the non-local layers in
    // place. MLX reflection drops `.none` array elements but PRESERVES their index, so the
    // surviving layers keep their original numbers for weight loading (mirrors Python
    // `self.layers[:start] = [None]*start`). Single-machine runs keep every element non-nil.
    var layers: [GLM4MoEDecoderLayer?]
    let norm: RMSNorm

    // Pipeline-parallel state. `pipelineGroup == nil` (or size 1) == single-machine — the
    // default path, byte-for-byte unchanged. These are plain stored props (reflected as
    // `.other`, ignored by parameter/module filters).
    var pipelineGroup: DistributedGroup?
    var startIdx = 0
    var endIdx: Int

    init(_ args: GLM4MoEConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { idx in
                GLM4MoEDecoderLayer(args, layerIdx: idx)
            }
        self.endIdx = args.hiddenLayers
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    /// This rank's local (non-nil) layers, in order — Python `pipeline_layers`.
    var pipelineLayers: [GLM4MoEDecoderLayer] {
        layers[startIdx ..< endIdx].compactMap { $0 }
    }

    /// Decoder layers resident on this rank (== hiddenLayers when not sharded).
    var localLayerCount: Int { endIdx - startIdx }

    /// Split the layers across the ring in reverse (rank 0 = LAST layers; the highest rank
    /// embeds). Port of mlx_lm `PipelineMixin.pipeline` + the uneven-boundary patch
    /// (glm_dist/patch.py). MUST run before any `parameters()`/`update()`/forward.
    func pipeline(group: DistributedGroup, boundary: Int?) {
        self.pipelineGroup = group
        let r = group.rank
        let n = group.size
        let total = layers.count
        if let b = boundary, n == 2 {
            // Asymmetric pair: the last rank (the memory-constrained peer) holds the FIRST b.
            if r == n - 1 {
                startIdx = 0
                endIdx = b
            } else {
                startIdx = b
                endIdx = total
            }
        } else {
            // Stock even split, reverse order so rank 0 gets the last layers.
            var perRank = total / n
            let extra = total - perRank * n
            if r < extra { perRank += 1 }
            startIdx = (n - r - 1) * perRank
            endIdx = startIdx + perRank
        }
        // Truncate to end_idx and nil the leading [0, start_idx) — same None-masking as
        // stock, keeping layer indices stable for weight loading.
        layers = Array(layers[0 ..< endIdx])
        for i in 0 ..< startIdx { layers[i] = nil }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        guard let g = pipelineGroup, g.size > 1 else {
            // Single-machine path (unchanged behavior).
            for (i, layer) in layers.enumerated() {
                if let layer { h = layer(h, mask: mask, cache: cache?[i]) }
            }
            return norm(h)
        }

        // Pipeline-parallel path. Collectives run on the CPU stream — the default (GPU)
        // stream parks a cross-machine wait inside a Metal command buffer and trips the
        // ~45 s GPU watchdog (glm_dist/patch.py). Greedy decode keeps ranks lock-step.
        let pr = g.rank
        let ps = g.size
        let batch = inputs.dim(0)

        // Receive the activation from the next-higher rank (which holds the earlier layers).
        if pr < ps - 1 {
            h = try! g.recvLike(h, from: pr + 1, stream: .cpu)
        }

        // The cache is local-length, indexed from 0 (Python zip(pipeline_layers, cache)).
        for (i, layer) in pipelineLayers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        // Forward to the previous rank, tying the send into the last local cache so the lazy
        // graph can't drop/reorder it relative to the KV write (glm_dist/patch.py:56-57).
        if pr != 0 {
            let token = try! g.send(h, to: (pr - 1) % ps, stream: .cpu)
            if let last = cache?.last as? KVCacheSimple {
                last.tieKeys(dependingOn: token)
            }
            h = token
        }

        // Broadcast rank 0's final hidden to every rank: all_gather concatenates in rank
        // order, so the first `batch` rows are rank 0's (the LAST layers' output). Every rank
        // then runs the same norm + lm_head → identical logits → greedy stays in lockstep.
        h = g.allGather(h, stream: .cpu)[0 ..< batch]
        return norm(h)
    }
}

public class GLM4MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: GLM4MoEModelInner
    let configuration: GLM4MoEConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: GLM4MoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = GLM4MoEModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// One cache per LOCAL layer (== hiddenLayers when not pipelined). Overrides the
    /// `KVCacheDimensionProvider` default (which would size to all `kvHeads`) so a
    /// pipeline-sharded rank's cache matches `pipelineLayers`, indexed from 0.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let n = model.localLayerCount
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< n).map { _ in RotatingKVCache(maxSize: maxKVSize, keep: 4) }
        }
        return (0 ..< n).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for n in ["gate_proj", "down_proj", "up_proj"] {
                for k in ["weight", "scales", "biases"] {
                    let key = "\(prefix).mlp.experts.0.\(n).\(k)"
                    if sanitized[key] != nil, let nRoutedExperts = configuration.nRoutedExperts {
                        let toJoin = (0 ..< nRoutedExperts).map { e in
                            sanitized.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(n).\(k)")!
                        }
                        sanitized["\(prefix).mlp.switch_mlp.\(n).\(k)"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        let mptLayerPrefix = "model.layers.\(configuration.hiddenLayers)"
        sanitized = sanitized.filter { !($0.key.hasPrefix(mptLayerPrefix)) }

        return sanitized
    }
}

public struct GLM4MoEConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var moeIntermediateSize: Int
    var normTopkProb: Bool
    var attentionHeads: Int
    var nGroup: Int
    var headDim: Int
    var topkGroup: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var numExpertsPerTok: Int
    var firstKDenseReplace: Int
    var hiddenLayers: Int
    var kvHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var useQkNorm: Bool
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var partialRotaryFactor: Float
    var scoringFunc: String = "sigmoid"
    var topkMethod: String = "noaux_tc"

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case attentionHeads = "num_attention_heads"
        case nGroup = "n_group"
        case headDim = "head_dim"
        case topkGroup = "topk_group"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case useQkNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case partialRotaryFactor = "partial_rotary_factor"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
    }

    public init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<GLM4MoEConfiguration.CodingKeys> =
            try decoder.container(keyedBy: GLM4MoEConfiguration.CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.normTopkProb = try container.decode(Bool.self, forKey: .normTopkProb)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.nGroup = try container.decode(Int.self, forKey: .nGroup)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.topkGroup = try container.decode(Int.self, forKey: .topkGroup)
        self.nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        self.nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        self.routedScalingFactor = try container.decode(Float.self, forKey: .routedScalingFactor)
        self.numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        self.firstKDenseReplace = try container.decode(Int.self, forKey: .firstKDenseReplace)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.useQkNorm = try container.decode(Bool.self, forKey: .useQkNorm)
        self.tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)
        self.attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        self.partialRotaryFactor = try container.decode(Float.self, forKey: .partialRotaryFactor)
        self.scoringFunc =
            try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sigmoid"
        self.topkMethod =
            try container.decodeIfPresent(String.self, forKey: .topkMethod) ?? "noaux_tc"
    }
}

// MARK: - LoRA

extension GLM4MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.pipelineLayers
    }
}

// MARK: - Pipeline parallelism

extension GLM4MoEModel: PipelineParallel {
    public func pipeline(group: DistributedGroup, boundary: Int?) {
        model.pipeline(group: group, boundary: boundary)
    }

    public func localParameterKeys() -> [String] {
        // After pipeline(), nil layers reflect as `.none` and drop out — so this is the
        // union of local layers + the always-resident embed_tokens / norm / lm_head.
        parameters().flattened().map { $0.0 }
    }

    public func filterLocalWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let lo = model.startIdx
        let hi = model.endIdx
        let prefix = "model.layers."
        return weights.filter { key, _ in
            guard key.hasPrefix(prefix) else { return true }  // keep embed/norm/lm_head/etc.
            let rest = key.dropFirst(prefix.count)
            guard let dot = rest.firstIndex(of: "."),
                let idx = Int(rest[rest.startIndex ..< dot])
            else { return true }
            return idx >= lo && idx < hi
        }
    }

    public func warmup() {
        // LOCAL forward (no collectives) through this rank's own layers, per-layer eval, to
        // fault the resident shard + compile kernels — the GPU-watchdog fix (run.py:51-57).
        let dummy = MLXArray((1 ... 4).map { Int32($0) }).reshaped(1, 4)
        var wx = model.embedTokens(dummy)
        eval(wx)
        let mask = createAttentionMask(h: wx, cache: nil as KVCache?)
        for layer in model.pipelineLayers {
            wx = layer(wx, mask: mask, cache: nil)
            eval(wx)
        }
        wx = model.norm(wx)
        eval(wx)
        if let lmHead {
            eval(lmHead(wx))
        } else {
            eval(model.embedTokens.asLinear(wx))
        }
    }
}
