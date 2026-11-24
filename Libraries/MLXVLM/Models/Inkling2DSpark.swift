// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The DSpark draft of an Inkling 2 checkpoint: its `mtp_config`, with the vocabulary of the text
/// configuration it drafts for.
public struct Inkling2DSparkConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDimensions: Int
    let intermediateSize: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let localLayers: Set<Int>
    let slidingWindow: Int
    /// The anchor and the masks after it, one draft per mask.
    let blockSize: Int
    let markovRank: Int
    let markovHeadType: String
    /// The layer visits whose features the draft reads; the last one is the final hidden state.
    let targetLayerIDs: [Int]
    let vocabularySize: Int
    let finalLogitSoftcap: Float

    private enum CodingKeys: String, CodingKey {
        case draft = "mtp_config"
        case text = "text_config"
    }

    private enum DraftKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case localLayers = "local_layer_ids"
        case slidingWindow = "sliding_window_size"
        case blockSize = "block_size"
        case markovRank = "markov_rank"
        case markovHeadType = "markov_head_type"
        case targetLayerIDs = "target_layer_ids"
    }

    private enum TextKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case finalLogitSoftcap = "final_logit_softcapping"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let draft = try container.nestedContainer(keyedBy: DraftKeys.self, forKey: .draft)
        hiddenSize = try draft.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try draft.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try draft.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try draft.decode(Int.self, forKey: .kvHeads)
        headDimensions = try draft.decode(Int.self, forKey: .headDimensions)
        intermediateSize = try draft.decode(Int.self, forKey: .intermediateSize)
        rmsNormEps = try draft.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try draft.decode(Float.self, forKey: .ropeTheta)
        localLayers = try draft.decode(Set<Int>.self, forKey: .localLayers)
        slidingWindow = try draft.decode(Int.self, forKey: .slidingWindow)
        blockSize = try draft.decode(Int.self, forKey: .blockSize)
        markovRank = try draft.decode(Int.self, forKey: .markovRank)
        markovHeadType = try draft.decode(String.self, forKey: .markovHeadType)
        targetLayerIDs = try draft.decode([Int].self, forKey: .targetLayerIDs)
        let text = try container.nestedContainer(keyedBy: TextKeys.self, forKey: .text)
        vocabularySize = try text.decode(Int.self, forKey: .vocabularySize)
        finalLogitSoftcap = try text.decode(Float.self, forKey: .finalLogitSoftcap)
    }
}

extension Inkling2DSparkConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard markovHeadType == "rnn_hidden" else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 DSpark needs the recurrent Markov head, not '\(markovHeadType)'.")
        }
        guard hiddenSize > 0, hiddenLayers > 0, headDimensions > 0, kvHeads > 0,
            attentionHeads.isMultiple(of: kvHeads), intermediateSize > 0, markovRank > 0,
            blockSize > 1, slidingWindow > 0, finalLogitSoftcap > 0,
            targetLayerIDs.count > 1, targetLayerIDs == targetLayerIDs.sorted(),
            Set(targetLayerIDs).count == targetLayerIDs.count
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 DSpark dimensions must be positive and its feature visits sorted.")
        }
    }
}

// MARK: - Draft layers

private final class Inkling2DSparkAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "k_proj_ctx") var contextKeyProjection: Linear
    @ModuleInfo(key: "v_proj_ctx") var contextValueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "k_norm_ctx") var contextKeyNorm: RMSNorm

    let heads: Int
    let kvHeads: Int
    /// Context positions a local layer sees before the block; `nil` on a global layer.
    let window: Int?
    let scale: Float
    let rope: RoPE

    init(_ configuration: Inkling2DSparkConfiguration, isLocal: Bool) {
        let (d, h, kv) = (
            configuration.hiddenSize, configuration.headDimensions, configuration.kvHeads
        )
        heads = configuration.attentionHeads
        kvHeads = kv
        window = isLocal ? configuration.slidingWindow : nil
        scale = 1 / Float(h).squareRoot()
        rope = RoPE(dimensions: h, traditional: false, base: configuration.ropeTheta)
        _queryProjection.wrappedValue = Linear(d, heads * h, bias: false)
        _keyProjection.wrappedValue = Linear(d, kv * h, bias: false)
        _valueProjection.wrappedValue = Linear(d, kv * h, bias: false)
        _contextKeyProjection.wrappedValue = Linear(d, kv * h, bias: false)
        _contextValueProjection.wrappedValue = Linear(d, kv * h, bias: false)
        _outputProjection.wrappedValue = Linear(heads * h, d, bias: false)
        _queryNorm.wrappedValue = RMSNorm(dimensions: h, eps: configuration.rmsNormEps)
        _keyNorm.wrappedValue = RMSNorm(dimensions: h, eps: configuration.rmsNormEps)
        _contextKeyNorm.wrappedValue = RMSNorm(dimensions: h, eps: configuration.rmsNormEps)
    }

    private func heads(_ x: MLXArray, _ count: Int) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1), count, -1)
    }

    /// Appends the K/V of mixed target features at the cache's next positions.
    func appendContext(_ context: MLXArray, cache: KVCache) {
        let keys = contextKeyNorm(heads(contextKeyProjection(context), kvHeads))
        let values = heads(contextValueProjection(context), kvHeads)
        _ = cache.update(
            keys: rope(keys.transposed(0, 2, 1, 3), offset: cache.offset),
            values: values.transposed(0, 2, 1, 3))
    }

    /// The block attends to the context before it and to all of itself.
    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let offset = cache.offset
        let queries = queryNorm(heads(queryProjection(x), heads)).transposed(0, 2, 1, 3)
        let keys = keyNorm(heads(keyProjection(x), kvHeads)).transposed(0, 2, 1, 3)
        let values = heads(valueProjection(x), kvHeads).transposed(0, 2, 1, 3)
        var (allKeys, allValues) = cache.update(
            keys: rope(keys, offset: offset), values: values)
        if let window, allKeys.dim(2) > window + l {
            let start = allKeys.dim(2) - window - l
            allKeys = allKeys[0..., 0..., start...]
            allValues = allValues[0..., 0..., start...]
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: rope(queries, offset: offset), keys: allKeys, values: allValues,
            scale: scale, mask: .none)
        return outputProjection(output.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

private final class Inkling2DSparkMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ configuration: Inkling2DSparkConfiguration) {
        let (d, i) = (configuration.hiddenSize, configuration.intermediateSize)
        _gate.wrappedValue = Linear(d, i, bias: false)
        _up.wrappedValue = Linear(d, i, bias: false)
        _down.wrappedValue = Linear(i, d, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class Inkling2DSparkLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2DSparkAttention
    @ModuleInfo var mlp: Inkling2DSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    /// Normalizes this layer's mix of the target features.
    @ModuleInfo(key: "dflare_norm") var featureNorm: RMSNorm

    init(_ configuration: Inkling2DSparkConfiguration, isLocal: Bool) {
        let (d, eps) = (configuration.hiddenSize, configuration.rmsNormEps)
        _attention.wrappedValue = Inkling2DSparkAttention(configuration, isLocal: isLocal)
        _mlp.wrappedValue = Inkling2DSparkMLP(configuration)
        _inputNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _featureNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache)
        return h + mlp(postAttentionNorm(h))
    }
}

/// A recurrent correction of the draft's logits: each draft feeds the next through a gated state.
///
/// `hidden_out` and the target's vocabulary head are folded into one rank-`R` projection when
/// the checkpoint loads.
private final class Inkling2DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var previousToken: Embedding
    @ModuleInfo(key: "rnn_proj") var recurrence: Linear
    @ModuleInfo(key: "fold") var fold: Linear
    /// Normalizes the head's mix of the last committed token's features.
    @ModuleInfo(key: "dflare_norm") var featureNorm: RMSNorm

    let rank: Int
    private var preparedRecurrence:
        (stepWeightT: MLXArray, contextWeightT: MLXArray, bias: MLXArray)?

    init(_ configuration: Inkling2DSparkConfiguration) {
        let (d, r, v) = (
            configuration.hiddenSize, configuration.markovRank, configuration.vocabularySize
        )
        rank = r
        _previousToken.wrappedValue = Embedding(embeddingCount: v, dimensions: r)
        _recurrence.wrappedValue = Linear(2 * r + 2 * d, 3 * r, bias: true)
        _fold.wrappedValue = Linear(r, v, bias: true)
        _featureNorm.wrappedValue = RMSNorm(dimensions: d, eps: configuration.rmsNormEps)
    }

    @discardableResult
    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate,
        path: [String] = [], modulePath: [String] = []
    ) throws -> Self {
        let replacesRecurrence = parameters.flattened().contains { key, _ in
            key.hasPrefix("rnn_proj.")
        }
        defer {
            if replacesRecurrence {
                preparedRecurrence = nil
            }
        }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        defer {
            if key == "rnn_proj" {
                preparedRecurrence = nil
            }
        }
        try super.updateModule(key: key, value)
    }

    func prepareRecurrenceWeights() {
        let resolved = computeRecurrenceWeights()
        eval(resolved.stepWeightT, resolved.contextWeightT, resolved.bias)
        preparedRecurrence = resolved
    }

    private func computeRecurrenceWeights() -> (
        stepWeightT: MLXArray, contextWeightT: MLXArray, bias: MLXArray
    ) {
        let denseWeight: MLXArray
        if let quantized = recurrence as? QuantizedLinear {
            denseWeight = dequantized(
                quantized.weight,
                scales: quantized.scales,
                biases: quantized.biases,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode
            ).asType(.float32)
        } else {
            denseWeight = recurrence.weight.asType(.float32)
        }
        let bias =
            (recurrence.bias ?? MLXArray.zeros([3 * rank], dtype: .float32)).asType(.float32)
        return (
            stepWeightT: contiguous(denseWeight[0..., ..<(2 * rank)].T),
            contextWeightT: contiguous(denseWeight[0..., (2 * rank)...].T),
            bias: bias
        )
    }

    /// The float32 bias and `[hidden, features]` part of the recurrence, for every draft.
    func context(hidden: MLXArray, features: MLXArray) -> MLXArray {
        let inputs = concatenated(
            [hidden, broadcast(features, to: hidden.shape)], axis: -1
        ).asType(.float32)
        let (_, contextWeightT, bias) = preparedRecurrence ?? computeRecurrenceWeights()
        return matmul(inputs, contextWeightT) + bias
    }

    /// One step from the float32 `context` of a draft: the next state and the logit correction.
    func step(state: MLXArray, previous: MLXArray, context: MLXArray) -> (MLXArray, MLXArray) {
        let inputs = concatenated([state, previousToken(previous)], axis: -1).asType(.float32)
        let (stepWeightT, _, _) = preparedRecurrence ?? computeRecurrenceWeights()
        let raw = (context + matmul(inputs, stepWeightT)).asType(state.dtype)
        let gate = sigmoid(raw[.ellipsis, ..<rank])
        let candidate = tanh(raw[.ellipsis, rank ..< (2 * rank)])
        let next = gate * state + (1 - gate) * candidate
        return (next, fold(tanh(raw[.ellipsis, (2 * rank)...])))
    }
}

// MARK: - Draft model

/// Thinking Machines' DSpark drafter for Inkling 2, from the checkpoint's `mtp.safetensors`
/// or `mtp-4bit.safetensors`.
///
/// An anchor and `blockSize - 1` masks run through three dense layers that attend to K/V
/// projected from four target features: partial blocks of three layer visits and the final hidden
/// state. A recurrent Markov head then picks the drafts one after another. The token embeddings
/// and the vocabulary head are the target's.
public final class Inkling2DSparkDraftModel: Module, StatefulMTPDrafterModel,
    AdditionalWeightFilesProviding
{
    public let configuration: Inkling2DSparkConfiguration
    public var maximumBlockSize: Int? { configuration.blockSize }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    public var targetLayerIds: [Int]? { configuration.targetLayerIDs }
    /// Inkling 2's 128-expert top-8 looped MoE target fans out across ~4-5x more active experts
    /// during multi-token verify than single-token decode (~41% break-even); step down below 36%.
    public var minimumRollingAcceptanceRate: Double? { 0.36 }

    /// The draft ships beside the target's weights (`mtp.safetensors` or 4-bit `mtp-4bit.safetensors`),
    /// reading the target's vocabulary head only to fold `markov_head.hidden_out`.
    public var additionalWeightFiles: [String] { ["mtp.safetensors", "mtp-4bit.safetensors"] }
    public var excludedCheckpointNamespaces: [String] { Inkling2Checkpoint.namespacesBesideHead }

    @ModuleInfo fileprivate var layers: [Inkling2DSparkLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "markov_head") fileprivate var markovHead: Inkling2DSparkMarkovHead
    /// `[layers + 1, features]`: the mix of the features of each layer, then of the head.
    @ParameterInfo(key: "fuse_weight") var fuseWeight: MLXArray
    @ParameterInfo(key: "mask_embed") var maskEmbedding: MLXArray

    public init(_ configuration: Inkling2DSparkConfiguration) {
        self.configuration = configuration
        let d = configuration.hiddenSize
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map {
            Inkling2DSparkLayer(configuration, isLocal: configuration.localLayers.contains($0))
        }
        _norm.wrappedValue = RMSNorm(dimensions: d, eps: configuration.rmsNormEps)
        _markovHead.wrappedValue = Inkling2DSparkMarkovHead(configuration)
        _fuseWeight.wrappedValue = MLXArray.zeros(
            [configuration.hiddenLayers + 1, configuration.targetLayerIDs.count])
        _maskEmbedding.wrappedValue = MLXArray.zeros([d])
        super.init()
    }

    public func makeState(parameters _: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: layers.map { _ in KVCacheSimple() })
    }

    public func prepareDrafterState(
        target: any LanguageModel, promptTokens _: MLXArray, targetHidden: MLXArray,
        firstBonus _: MLXArray, positionDeltas _: MLXArray?, state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        precondition(isCompatible(with: target), "Inkling 2 DSpark draft and target differ")
        appendContext(targetHidden, state: &state)
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
        queryOffset _: Int, blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: [:],
            positionDeltas: nil, queryOffset: 0, blockSize: blockSize, state: &state,
            sampler: sampler)
    }

    /// Always drafts the whole trained block and keeps the first `blockSize - 1` drafts: the masks
    /// attend to each other, so a shorter block would change every draft.
    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)], positionDeltas _: MLXArray?,
        queryOffset _: Int, blockSize: Int, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        let target = inkling(target)
        let drafts = blockSize - 1
        let block = configuration.blockSize
        precondition((1 ..< block).contains(drafts), "Inkling 2 DSpark drafts 1 to \(block - 1)")

        let anchor = normalizedMTPColumn(lastToken)
        let rows = anchor.dim(0)
        let anchorEmbedding = target.embedding(anchor)
        var x = concatenated(
            [
                anchorEmbedding,
                broadcast(
                    maskEmbedding.asType(anchorEmbedding.dtype).reshaped(1, 1, -1),
                    to: [rows, block - 1, configuration.hiddenSize]),
            ], axis: 1)
        for (layer, cache) in zip(layers, state.cache) {
            x = layer(x, cache: cache)
        }
        // The block is transient: only the target features stay in the cache.
        for cache in state.cache { cache.trim(block) }

        let hidden = norm(x)[0..., 1 ... drafts]
        let baseLogits = target.vocabularyLogits(hidden)
        let context = markovHead.context(
            hidden: hidden, features: mix(lastHidden, row: layers.count, markovHead.featureNorm))
        let softcap = configuration.finalLogitSoftcap
        var markovState = MLXArray.zeros([rows, 1, markovHead.rank], dtype: hidden.dtype)
        var previous = anchor
        var proposals = [MLXArray]()
        for index in 0 ..< drafts {
            let (next, correction) = markovHead.step(
                state: markovState, previous: previous,
                context: context[0..., index ..< (index + 1)])
            markovState = next
            let logits = (baseLogits[0..., index ..< (index + 1)] + correction).asType(.float32)
            let capped = tanh(logits / softcap) * softcap
            previous = normalizedMTPColumn(
                sampler.sample(logits: capped[0..., 0, ..<target.vocabularySize]))
            proposals.append(previous)
        }
        return concatenated(proposals, axis: 1)
    }

    public func commitDrafterState(
        target _: any LanguageModel, targetHidden: MLXArray, draftTokens _: MLXArray,
        acceptedCount: Int, finalToken _: MLXArray, positionDeltas _: MLXArray?,
        state: inout MTPDrafterState, sampler _: any LogitSampler
    ) {
        let committed = acceptedCount + 1
        precondition(
            targetHidden.dim(1) >= committed, "the target did not emit every committed feature")
        appendContext(targetHidden[0..., ..<committed], state: &state)
    }

    /// Each layer's K/V of the committed rows' features `[B, L, features * D]`.
    private func appendContext(_ features: MLXArray, state: inout MTPDrafterState) {
        for (index, (layer, cache)) in zip(layers, state.cache).enumerated() {
            layer.attention.appendContext(
                mix(features, row: index, layer.featureNorm), cache: cache)
        }
        state.nextPosition += features.dim(1)
    }

    /// The RMS norm of the features weighted by the softmax of `fuse_weight[row]`, in float32.
    private func mix(_ features: MLXArray, row: Int, _ norm: RMSNorm) -> MLXArray {
        let taps = configuration.targetLayerIDs.count
        let shape = features.shape.dropLast() + [taps, configuration.hiddenSize]
        let alpha = softmax(fuseWeight[row].asType(.float32), axis: -1).reshaped(taps, 1)
        let mixed = (features.reshaped(Array(shape)).asType(.float32) * alpha).sum(axis: -2)
        return MLXFast.rmsNorm(mixed, weight: norm.weight.asType(.float32), eps: norm.eps)
            .asType(features.dtype)
    }

    private func inkling(_ target: any LanguageModel) -> Inkling2 {
        guard let target = target as? Inkling2, isCompatible(with: target) else {
            preconditionFailure("Inkling 2 DSpark needs its Inkling 2 target")
        }
        return target
    }

    public func validateCompatibility(with target: any LanguageModel) throws {
        guard isCompatible(with: target) else {
            throw MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Inkling2DSparkDraftModel",
                expected: "an Inkling 2 target with \(configuration.targetLayerIDs.last ?? 0) "
                    + "layer visits and hidden size \(configuration.hiddenSize)",
                actual: String(describing: type(of: target)))
        }
    }

    public func isCompatible(with target: any LanguageModel) -> Bool {
        guard let target = target as? Inkling2 else { return false }
        let text = target.config.textConfiguration
        return text.hiddenSize == configuration.hiddenSize
            && text.vocabularySize == configuration.vocabularySize
            && text.layerVisits.count == configuration.targetLayerIDs.last
    }

    public func prepare() throws {
        markovHead.prepareRecurrenceWeights()
    }

    /// Folds `hidden_out` into the target's vocabulary head, which is read for this alone, and
    /// infers per-layer quantization for pre-quantized `mtp-4bit.safetensors` companions.
    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        var weights = [String: MLXArray](minimumCapacity: checkpoint.weights.count)
        let hasTopLevelDraftKeys =
            checkpoint.weights["fuse_weight"] != nil || checkpoint.weights["mask_embed"] != nil
        for (rawKey, value) in checkpoint.weights {
            var key = rawKey
            if !hasTopLevelDraftKeys {
                if key.hasPrefix("model.mtp.") {
                    key = String(key.dropFirst("model.mtp.".count))
                } else if key.hasPrefix("mtp.") {
                    key = String(key.dropFirst("mtp.".count))
                }
            }
            weights[key] = value
        }

        guard
            let head = weights.removeValue(forKey: "lm_head.weight")
                ?? weights.removeValue(forKey: "\(Inkling2Checkpoint.originalHead).weight"),
            let rawProjection = weights.removeValue(forKey: "markov_head.hidden_out.weight"),
            let bias = weights.removeValue(forKey: "markov_head.hidden_out.bias")
        else {
            throw ModelFactoryError.invalidConfiguration(
                "Inkling 2 DSpark needs mtp.safetensors or mtp-4bit.safetensors beside the model's vocabulary head."
            )
        }
        let scales =
            weights.removeValue(forKey: "lm_head.scales")
            ?? weights.removeValue(forKey: "\(Inkling2Checkpoint.originalHead).weight_scale")
        let biases = weights.removeValue(forKey: "lm_head.biases")
        let headQuantization =
            checkpoint.perLayerQuantization?.quantization(layer: "lm_head")
            ?? scales.flatMap {
                Self.inferAffineQuantization(
                    packedWeight: head, scales: $0, inputDimensions: configuration.hiddenSize)
            }

        let hiddenOutScales = weights.removeValue(forKey: "markov_head.hidden_out.scales")
        let hiddenOutBiases = weights.removeValue(forKey: "markov_head.hidden_out.biases")
        let projection: MLXArray
        if let hiddenOutScales {
            let q =
                checkpoint.perLayerQuantization?.quantization(layer: "markov_head.hidden_out")
                ?? Self.inferAffineQuantization(
                    packedWeight: rawProjection,
                    scales: hiddenOutScales,
                    inputDimensions: configuration.markovRank)
            if let q {
                projection = dequantized(
                    rawProjection, scales: hiddenOutScales, biases: hiddenOutBiases,
                    groupSize: q.groupSize, bits: q.bits, mode: q.mode)
            } else {
                projection = rawProjection
            }
        } else {
            projection = rawProjection
        }

        var foldWeights = [MLXArray]()
        var foldBiases = [MLXArray]()
        for start in stride(from: 0, to: head.dim(0), by: 16384) {
            let rows = start ..< min(start + 16384, head.dim(0))
            var chunk = head[rows]
            if let scales, let headQuantization {
                chunk = dequantized(
                    chunk, scales: scales[rows], biases: biases?[rows],
                    groupSize: headQuantization.groupSize, bits: headQuantization.bits,
                    mode: headQuantization.mode)
            }
            let rowsF32 = chunk.asType(.float32)
            foldWeights.append(matmul(rowsF32, projection.asType(.float32)).asType(.bfloat16))
            foldBiases.append(matmul(rowsF32, bias.asType(.float32)).asType(.bfloat16))
            eval(foldWeights.last!, foldBiases.last!)
        }
        weights["markov_head.fold.weight"] = concatenated(foldWeights)
        weights["markov_head.fold.bias"] = concatenated(foldBiases)

        // Populate per-layer quantization entries for any pre-quantized companion tensors when
        // the target's `config.json` has no matching quantization metadata (e.g. NVFP4/BF16 target
        // paired with a 4-bit affine `mtp-4bit.safetensors` companion).
        var inferredPerLayer = checkpoint.perLayerQuantization?.perLayerQuantization ?? [:]
        var inferredAny = false
        for (path, module) in namedModules() where !(module is any Quantized) {
            guard let layerScales = weights["\(path).scales"],
                let packedWeight = weights["\(path).weight"]
            else { continue }
            let inputDims: Int? =
                if let linear = module as? Linear {
                    linear.weight.dim(1)
                } else if let embedding = module as? Embedding {
                    embedding.weight.dim(1)
                } else {
                    nil
                }
            guard let inputDims,
                let inferred = Self.inferAffineQuantization(
                    packedWeight: packedWeight, scales: layerScales, inputDimensions: inputDims)
            else { continue }
            let existing = checkpoint.perLayerQuantization?.quantization(layer: path)
            if existing == nil || existing?.groupSize != inferred.groupSize
                || existing?.bits != inferred.bits
                || (weights["\(path).biases"] != nil && existing?.mode != .affine)
            {
                inferredPerLayer[path] = .quantize(inferred)
                inferredAny = true
            }
        }
        if inferredAny {
            checkpoint.perLayerQuantization = .init(
                quantization: checkpoint.perLayerQuantization?.quantization,
                perLayerQuantization: inferredPerLayer)
        }

        checkpoint.weights = weights
        return checkpoint
    }

    private static func inferAffineQuantization(
        packedWeight: MLXArray, scales: MLXArray, inputDimensions: Int
    ) -> BaseConfiguration.Quantization? {
        guard inputDimensions > 0, scales.dim(-1) > 0, packedWeight.dim(-1) > 0,
            inputDimensions.isMultiple(of: scales.dim(-1))
        else {
            return nil
        }
        let groupSize = inputDimensions / scales.dim(-1)
        let bits = (packedWeight.dim(-1) * 32) / inputDimensions
        guard bits == 2 || bits == 3 || bits == 4 || bits == 6 || bits == 8 else { return nil }
        return .init(groupSize: groupSize, bits: bits)
    }
}
