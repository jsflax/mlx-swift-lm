// Copyright © 2026 Apple Inc.

import Foundation

/// Configuration types for Thinking Machines Inkling 2 (`inkling2` and `inkling2_loop`).
public struct Inkling2Configuration: Codable, Sendable {

    public struct LoopConfig: Codable, Sendable, Equatable {
        public var coreStart: Int
        public var coreEnd: Int
        public var numPasses: Int
        public var gateRank: Int
        public var isVarloop: Bool

        public var coreRange: Range<Int> {
            coreStart ..< coreEnd
        }

        enum CodingKeys: String, CodingKey {
            case coreStart = "core_start"
            case coreEnd = "core_end"
            case numPasses = "num_passes"
            case gateRank = "gate_rank"
            case isVarloop = "is_varloop"
        }

        public init(
            coreStart: Int = 6,
            coreEnd: Int = 18,
            numPasses: Int = 3,
            gateRank: Int = 64,
            isVarloop: Bool = false
        ) {
            self.coreStart = coreStart
            self.coreEnd = coreEnd
            self.numPasses = numPasses
            self.gateRank = gateRank
            self.isVarloop = isVarloop
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.coreStart = try container.decode(Int.self, forKey: .coreStart)
            self.coreEnd = try container.decode(Int.self, forKey: .coreEnd)
            self.numPasses = try container.decodeIfPresent(Int.self, forKey: .numPasses) ?? 3
            self.gateRank = try container.decodeIfPresent(Int.self, forKey: .gateRank) ?? 64
            self.isVarloop = try container.decodeIfPresent(Bool.self, forKey: .isVarloop) ?? false
        }
    }

    public struct TextConfig: Codable, Sendable {
        public var modelType: String?
        public var modelMaxLength: Int?
        public var hiddenSize: Int
        public var numHiddenLayers: Int
        public var vocabSize: Int
        public var unpaddedVocabSize: Int
        public var numAttentionHeads: Int
        public var numKeyValueHeads: Int
        public var headDim: Int
        public var swaNumAttentionHeads: Int
        public var swaNumKeyValueHeads: Int
        public var swaHeadDim: Int
        public var slidingWindowSize: Int
        public var localLayerIds: [Int]
        public var dRel: Int
        public var relExtent: Int
        public var logScalingNFloor: Int
        public var logScalingAlpha: Float
        public var rmsNormEps: Float
        public var qkL2ClipThreshold: Float
        public var relL2ClipThreshold: Float
        public var sconvKernelSize: Int
        public var nLayersPerAttnResBlock: Int
        public var nRoutedExperts: Int
        public var numExpertsPerTok: Int
        public var intermediateSize: Int
        public var mlpGateSoftcap: Float
        public var mlpUpSoftcap: Float
        public var routeScale: Float
        public var finalLogitSoftcapping: Float
        public var loopConfig: LoopConfig?

        public var loopCore: Range<Int> {
            loopConfig?.coreRange ?? (0 ..< 0)
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case modelMaxLength = "model_max_length"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case vocabSize = "vocab_size"
            case unpaddedVocabSize = "unpadded_vocab_size"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case swaNumAttentionHeads = "swa_num_attention_heads"
            case swaNumKeyValueHeads = "swa_num_key_value_heads"
            case swaHeadDim = "swa_head_dim"
            case slidingWindowSize = "sliding_window_size"
            case localLayerIds = "local_layer_ids"
            case dRel = "d_rel"
            case relExtent = "rel_extent"
            case logScalingNFloor = "log_scaling_n_floor"
            case logScalingAlpha = "log_scaling_alpha"
            case rmsNormEps = "rms_norm_eps"
            case qkL2ClipThreshold = "qk_l2_clip_threshold"
            case relL2ClipThreshold = "rel_l2_clip_threshold"
            case sconvKernelSize = "sconv_kernel_size"
            case nLayersPerAttnResBlock = "n_layers_per_attn_res_block"
            case nRoutedExperts = "n_routed_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case intermediateSize = "intermediate_size"
            case mlpGateSoftcap = "mlp_gate_softcap"
            case mlpUpSoftcap = "mlp_up_softcap"
            case routeScale = "route_scale"
            case finalLogitSoftcapping = "final_logit_softcapping"
            case loopConfig = "loop_config"
        }

        public init(
            modelType: String? = nil,
            modelMaxLength: Int? = 1_048_576,
            hiddenSize: Int = 2304,
            numHiddenLayers: Int = 24,
            vocabSize: Int = 201_024,
            unpaddedVocabSize: Int = 200_058,
            numAttentionHeads: Int = 24,
            numKeyValueHeads: Int = 8,
            headDim: Int = 128,
            swaNumAttentionHeads: Int? = nil,
            swaNumKeyValueHeads: Int? = nil,
            swaHeadDim: Int? = nil,
            slidingWindowSize: Int = 1024,
            localLayerIds: [Int] = [],
            dRel: Int = 16,
            relExtent: Int = 1024,
            logScalingNFloor: Int = 8192,
            logScalingAlpha: Float = 0.1,
            rmsNormEps: Float = 1e-6,
            qkL2ClipThreshold: Float = 22.627416997969522,
            relL2ClipThreshold: Float = 8.0,
            sconvKernelSize: Int = 4,
            nLayersPerAttnResBlock: Int = 6,
            nRoutedExperts: Int = 128,
            numExpertsPerTok: Int = 8,
            intermediateSize: Int = 1152,
            mlpGateSoftcap: Float = 4.0,
            mlpUpSoftcap: Float = 25.0,
            routeScale: Float = 8.0,
            finalLogitSoftcapping: Float = 30.0,
            loopConfig: LoopConfig? = nil
        ) {
            self.modelType = modelType
            self.modelMaxLength = modelMaxLength
            self.hiddenSize = hiddenSize
            self.numHiddenLayers = numHiddenLayers
            self.vocabSize = vocabSize
            self.unpaddedVocabSize = unpaddedVocabSize
            self.numAttentionHeads = numAttentionHeads
            self.numKeyValueHeads = numKeyValueHeads
            self.headDim = headDim
            self.swaNumAttentionHeads = swaNumAttentionHeads ?? numAttentionHeads
            self.swaNumKeyValueHeads = swaNumKeyValueHeads ?? numKeyValueHeads
            self.swaHeadDim = swaHeadDim ?? headDim
            self.slidingWindowSize = slidingWindowSize
            self.localLayerIds = localLayerIds
            self.dRel = dRel
            self.relExtent = relExtent
            self.logScalingNFloor = logScalingNFloor
            self.logScalingAlpha = logScalingAlpha
            self.rmsNormEps = rmsNormEps
            self.qkL2ClipThreshold = qkL2ClipThreshold
            self.relL2ClipThreshold = relL2ClipThreshold
            self.sconvKernelSize = sconvKernelSize
            self.nLayersPerAttnResBlock = nLayersPerAttnResBlock
            self.nRoutedExperts = nRoutedExperts
            self.numExpertsPerTok = numExpertsPerTok
            self.intermediateSize = intermediateSize
            self.mlpGateSoftcap = mlpGateSoftcap
            self.mlpUpSoftcap = mlpUpSoftcap
            self.routeScale = routeScale
            self.finalLogitSoftcapping = finalLogitSoftcapping
            self.loopConfig = loopConfig
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType)
            self.modelMaxLength = try c.decodeIfPresent(Int.self, forKey: .modelMaxLength)
            self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            self.numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
            let vocabSize = try c.decode(Int.self, forKey: .vocabSize)
            self.vocabSize = vocabSize
            self.unpaddedVocabSize =
                try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabSize) ?? vocabSize
            let numHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
            let numKVHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
            let headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (self.hiddenSize / numHeads)
            self.numAttentionHeads = numHeads
            self.numKeyValueHeads = numKVHeads
            self.headDim = headDim
            self.swaNumAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .swaNumAttentionHeads) ?? numHeads
            self.swaNumKeyValueHeads =
                try c.decodeIfPresent(Int.self, forKey: .swaNumKeyValueHeads) ?? numKVHeads
            self.swaHeadDim = try c.decodeIfPresent(Int.self, forKey: .swaHeadDim) ?? headDim
            self.slidingWindowSize =
                try c.decodeIfPresent(Int.self, forKey: .slidingWindowSize) ?? 1024
            self.localLayerIds = try c.decodeIfPresent([Int].self, forKey: .localLayerIds) ?? []
            self.dRel = try c.decodeIfPresent(Int.self, forKey: .dRel) ?? 16
            self.relExtent = try c.decodeIfPresent(Int.self, forKey: .relExtent) ?? 1024
            self.logScalingNFloor =
                try c.decodeIfPresent(Int.self, forKey: .logScalingNFloor) ?? 8192
            self.logScalingAlpha =
                try c.decodeIfPresent(Float.self, forKey: .logScalingAlpha) ?? 0.1
            self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.qkL2ClipThreshold =
                try c.decodeIfPresent(Float.self, forKey: .qkL2ClipThreshold) ?? 22.627416997969522
            self.relL2ClipThreshold =
                try c.decodeIfPresent(Float.self, forKey: .relL2ClipThreshold) ?? 8.0
            self.sconvKernelSize = try c.decodeIfPresent(Int.self, forKey: .sconvKernelSize) ?? 4
            self.nLayersPerAttnResBlock =
                try c.decodeIfPresent(Int.self, forKey: .nLayersPerAttnResBlock) ?? 6
            self.nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 128
            self.numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
            self.intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1152
            self.mlpGateSoftcap =
                try c.decodeIfPresent(Float.self, forKey: .mlpGateSoftcap) ?? 4.0
            self.mlpUpSoftcap = try c.decodeIfPresent(Float.self, forKey: .mlpUpSoftcap) ?? 25.0
            self.routeScale = try c.decodeIfPresent(Float.self, forKey: .routeScale) ?? 8.0
            self.finalLogitSoftcapping =
                try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
            self.loopConfig = try c.decodeIfPresent(LoopConfig.self, forKey: .loopConfig)
        }
    }

    public struct VisionConfig: Codable, Sendable {
        public var hiddenSize: Int
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var numChannels: Int
        public var nativePatchSize: Int
        public var adapterMergeSize: Int
        public var posembGridSize: Int
        public var projectionHiddenDim: Int
        public var projectorBias: Bool
        public var mlpRatio: Float
        public var layerNormEps: Float
        public var ropeTheta: Float

        public var headDim: Int {
            hiddenSize / numAttentionHeads
        }

        public var intermediateSize: Int {
            Int(Float(hiddenSize) * mlpRatio)
        }

        public var blockSize: Int {
            nativePatchSize * adapterMergeSize
        }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numChannels = "num_channels"
            case nativePatchSize = "native_patch_size"
            case adapterMergeSize = "adapter_merge_size"
            case posembGridSize = "posemb_grid_size"
            case projectionHiddenDim = "projection_hidden_dim"
            case projectorBias = "projector_bias"
            case mlpRatio = "mlp_ratio"
            case layerNormEps = "layer_norm_eps"
            case ropeTheta = "rope_theta"
        }

        public init(
            hiddenSize: Int = 1024,
            numHiddenLayers: Int = 23,
            numAttentionHeads: Int = 16,
            numChannels: Int = 3,
            nativePatchSize: Int = 20,
            adapterMergeSize: Int = 2,
            posembGridSize: Int = 32,
            projectionHiddenDim: Int = 4096,
            projectorBias: Bool = false,
            mlpRatio: Float = 4.0,
            layerNormEps: Float = 1e-5,
            ropeTheta: Float = 10000.0
        ) {
            self.hiddenSize = hiddenSize
            self.numHiddenLayers = numHiddenLayers
            self.numAttentionHeads = numAttentionHeads
            self.numChannels = numChannels
            self.nativePatchSize = nativePatchSize
            self.adapterMergeSize = adapterMergeSize
            self.posembGridSize = posembGridSize
            self.projectionHiddenDim = projectionHiddenDim
            self.projectorBias = projectorBias
            self.mlpRatio = mlpRatio
            self.layerNormEps = layerNormEps
            self.ropeTheta = ropeTheta
        }

        public init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 23
            self.numAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
            self.numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
            self.nativePatchSize = try c.decodeIfPresent(Int.self, forKey: .nativePatchSize) ?? 20
            self.adapterMergeSize =
                try c.decodeIfPresent(Int.self, forKey: .adapterMergeSize) ?? 2
            self.posembGridSize = try c.decodeIfPresent(Int.self, forKey: .posembGridSize) ?? 32
            self.projectionHiddenDim =
                try c.decodeIfPresent(Int.self, forKey: .projectionHiddenDim) ?? 4096
            self.projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? false
            self.mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4.0
            self.layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
            self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        }
    }

    public var modelType: String
    public var textConfig: TextConfig
    public var visionConfig: VisionConfig?
    public var imageTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
    }

    public init(
        modelType: String = "inkling2_loop",
        textConfig: TextConfig,
        visionConfig: VisionConfig? = nil,
        imageTokenId: Int = 200_054
    ) {
        self.modelType = modelType
        self.textConfig = textConfig
        self.visionConfig = visionConfig
        self.imageTokenId = imageTokenId
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "inkling2"
        if let nestedText = try c.decodeIfPresent(TextConfig.self, forKey: .textConfig) {
            self.textConfig = nestedText
        } else {
            self.textConfig = try TextConfig(from: decoder)
        }
        self.visionConfig = try c.decodeIfPresent(VisionConfig.self, forKey: .visionConfig)
        self.imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200_054
    }
}
