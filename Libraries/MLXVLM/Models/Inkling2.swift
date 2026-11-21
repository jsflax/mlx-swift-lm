// Copyright © 2026 Apple Inc.

import CoreGraphics
@preconcurrency import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Inkling 2 Vision Tower

public enum Inkling2Vision {

    fileprivate static func preciseGELU(_ x: MLXArray) -> MLXArray {
        MLXNN.gelu(x.asType(.float32)).asType(x.dtype)
    }

    fileprivate static func gridPositions(
        gridH: Int,
        gridW: Int,
        merge: Int
    ) -> (MLXArray, MLXArray) {
        let i = MLXArray(0 ..< (gridH * gridW))
        let blocksW = gridW / merge
        let mergeSq = merge * merge
        let row = i.floorDivide(mergeSq * blocksW) * merge + (i.floorDivide(merge) % merge)
        let col = (i.floorDivide(mergeSq) % blocksW) * merge + (i % merge)
        return (row, col)
    }

    fileprivate static func bilinearTaps(
        index: MLXArray,
        size: Int,
        side: Int
    ) -> (MLXArray, MLXArray) {
        let src = (index.asType(.float32) + Float(0.5)) * Float(side) / Float(size) - Float(0.5)
        let taps = floor(src)[0..., .newAxis] + MLXArray(0 ..< 2).asType(.float32)
        let weights = maximum(Float(1.0) - abs(src[0..., .newAxis] - taps), Float(0.0))
        return (clip(taps, min: 0, max: side - 1).asType(.int32), weights)
    }

    public final class VisionEmbeddings: Module {
        let mergeSize: Int
        let gridSize: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ParameterInfo(key: "cls_position_embedding") var clsPositionEmbedding: MLXArray

        public init(_ args: Inkling2Configuration.VisionConfig) {
            self.mergeSize = args.adapterMergeSize
            self.gridSize = args.posembGridSize
            let patchDims = args.numChannels * args.nativePatchSize * args.nativePatchSize
            self._patchEmbedding.wrappedValue = Linear(patchDims, args.hiddenSize, bias: false)
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: args.posembGridSize * args.posembGridSize,
                dimensions: args.hiddenSize
            )
            self._classEmbedding.wrappedValue = MLXArray.zeros([args.hiddenSize])
            self._clsPositionEmbedding.wrappedValue = MLXArray.zeros([args.hiddenSize])
            super.init()
        }

        public func callAsFunction(
            _ pixelPatches: MLXArray,
            gridH: Int,
            gridW: Int
        ) -> MLXArray {
            var patchEmbeds = patchEmbedding(pixelPatches)

            let side = gridSize
            let (row, col) = gridPositions(gridH: gridH, gridW: gridW, merge: mergeSize)
            let (rowTaps, rowWeights) = bilinearTaps(index: row, size: gridH, side: side)
            let (colTaps, colWeights) = bilinearTaps(index: col, size: gridW, side: side)

            let interpIndices =
                rowTaps[0..., 0..., .newAxis] * Int32(side) + colTaps[0..., .newAxis, 0...]
            let interpWeights =
                rowWeights[0..., 0..., .newAxis] * colWeights[0..., .newAxis, 0...]
            let posEmbeds =
                positionEmbedding(interpIndices.reshaped(-1, 4)).asType(.float32)
                * interpWeights.reshaped(-1, 4, 1)
            patchEmbeds = patchEmbeds + posEmbeds.sum(axis: 1).asType(patchEmbeds.dtype)

            let clsEmbed = (classEmbedding + clsPositionEmbedding).asType(patchEmbeds.dtype)
            return concatenated([clsEmbed[.newAxis], patchEmbeds], axis: 0)
        }
    }

    /// Plain struct (not `Module`) so `invFreq` is excluded from `Module.parameters()`
    /// and `verify: [.all]` weight checks.
    struct VisionRotaryEmbedding {
        let invFreq: MLXArray

        init(_ args: Inkling2Configuration.VisionConfig) {
            let dims = args.headDim / 2
            let exponents =
                MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims)
            self.invFreq = Float(1.0) / pow(MLXArray(args.ropeTheta), exponents)
        }

        func callAsFunction(row: MLXArray, col: MLXArray) -> (MLXArray, MLXArray) {
            let freqsH = row[0..., .newAxis].asType(.float32) * invFreq
            let freqsW = col[0..., .newAxis].asType(.float32) * invFreq
            let freqs = repeated(concatenated([freqsW, freqsH], axis: -1), count: 2, axis: -1)
            return (cos(freqs), sin(freqs))
        }
    }

    fileprivate static func applyRotaryPosEmbVision(
        _ hiddenStates: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> MLXArray {
        let states = hiddenStates.asType(.float32)
        let pairs = states.reshaped(Array(states.shape.dropLast()) + [-1, 2])
        let rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1)
            .reshaped(states.shape)
        return (states * cos + rotated * sin).asType(hiddenStates.dtype)
    }

    public final class VisionAttention: Module {
        let numHeads: Int
        let scaling: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        public init(_ args: Inkling2Configuration.VisionConfig) {
            let dims = args.hiddenSize
            self.numHeads = args.numAttentionHeads
            self.scaling = 1.0 / sqrt(Float(args.headDim))
            self._qProj.wrappedValue = Linear(dims, dims, bias: true)
            self._kProj.wrappedValue = Linear(dims, dims, bias: true)
            self._vProj.wrappedValue = Linear(dims, dims, bias: true)
            self._outProj.wrappedValue = Linear(dims, dims, bias: true)
            super.init()
        }

        public func callAsFunction(
            _ hiddenStates: MLXArray,
            cos: MLXArray,
            sin: MLXArray
        ) -> MLXArray {
            let N = hiddenStates.dim(0)
            let cosExp = cos[0..., .newAxis, 0...]
            let sinExp = sin[0..., .newAxis, 0...]

            var q = qProj(hiddenStates).reshaped(N, numHeads, -1)
            var k = kProj(hiddenStates).reshaped(N, numHeads, -1)
            let v = vProj(hiddenStates).reshaped(N, numHeads, -1)

            q = applyRotaryPosEmbVision(q, cos: cosExp, sin: sinExp)
            k = applyRotaryPosEmbVision(k, cos: cosExp, sin: sinExp)

            let qB = q.transposed(1, 0, 2)[.newAxis]
            let kB = k.transposed(1, 0, 2)[.newAxis]
            let vB = v.transposed(1, 0, 2)[.newAxis]

            let attnOut = MLXFast.scaledDotProductAttention(
                queries: qB,
                keys: kB,
                values: vB,
                scale: scaling,
                mask: .none
            )
            return outProj(attnOut[0].transposed(1, 0, 2).reshaped(N, -1))
        }
    }

    public final class VisionMLP: Module {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear

        public init(_ args: Inkling2Configuration.VisionConfig) {
            self._fc1.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: true)
            self._fc2.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: true)
            super.init()
        }

        public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            fc2(preciseGELU(fc1(hiddenStates)))
        }
    }

    public final class VisionEncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: VisionAttention
        @ModuleInfo(key: "mlp") var mlp: VisionMLP
        @ModuleInfo(key: "layernorm_before") var layerNormBefore: LayerNorm
        @ModuleInfo(key: "layernorm_after") var layerNormAfter: LayerNorm
        @ParameterInfo(key: "lambda_1") var lambda1: MLXArray
        @ParameterInfo(key: "lambda_2") var lambda2: MLXArray

        public init(_ args: Inkling2Configuration.VisionConfig) {
            let dims = args.hiddenSize
            let eps = args.layerNormEps
            self._selfAttn.wrappedValue = VisionAttention(args)
            self._mlp.wrappedValue = VisionMLP(args)
            self._layerNormBefore.wrappedValue = LayerNorm(dimensions: dims, eps: eps)
            self._layerNormAfter.wrappedValue = LayerNorm(dimensions: dims, eps: eps)
            self._lambda1.wrappedValue = MLXArray.ones([dims])
            self._lambda2.wrappedValue = MLXArray.ones([dims])
            super.init()
        }

        public func callAsFunction(
            _ hiddenStates: MLXArray,
            cos: MLXArray,
            sin: MLXArray
        ) -> MLXArray {
            var h = hiddenStates + lambda1 * selfAttn(layerNormBefore(hiddenStates), cos: cos, sin: sin)
            h = h + lambda2 * mlp(layerNormAfter(h))
            return h
        }
    }

    public final class VisionMerger: Module {
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear

        public init(_ args: Inkling2Configuration.VisionConfig, textHiddenSize: Int) {
            let inDims = args.adapterMergeSize * args.adapterMergeSize * args.hiddenSize
            self._linear1.wrappedValue = Linear(
                inDims, args.projectionHiddenDim, bias: args.projectorBias)
            self._linear2.wrappedValue = Linear(
                args.projectionHiddenDim, textHiddenSize, bias: args.projectorBias)
            super.init()
        }

        public func callAsFunction(_ imageFeatures: MLXArray) -> MLXArray {
            linear2(preciseGELU(linear1(imageFeatures)))
        }
    }

    public final class VisionModel: Module {
        public let args: Inkling2Configuration.VisionConfig
        let rotaryEmb: VisionRotaryEmbedding

        @ModuleInfo(key: "embeddings") var embeddings: VisionEmbeddings
        @ModuleInfo(key: "pre_layernorm") var preLayerNorm: LayerNorm
        @ModuleInfo(key: "layers") var layers: [VisionEncoderLayer]
        @ModuleInfo(key: "merger") var merger: VisionMerger

        public init(_ args: Inkling2Configuration.VisionConfig, textHiddenSize: Int) {
            self.args = args
            self.rotaryEmb = VisionRotaryEmbedding(args)
            self._embeddings.wrappedValue = VisionEmbeddings(args)
            self._preLayerNorm.wrappedValue = LayerNorm(
                dimensions: args.hiddenSize, eps: args.layerNormEps)
            self._layers.wrappedValue = (0 ..< args.numHiddenLayers).map { _ in
                VisionEncoderLayer(args)
            }
            self._merger.wrappedValue = VisionMerger(args, textHiddenSize: textHiddenSize)
            super.init()
        }

        /// Encodes a single planar image `[C, H, W]` where `H` and `W` are multiples of
        /// `nativePatchSize * adapterMergeSize` (`40`).
        public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            let C = pixelValues.dim(0)
            let H = pixelValues.dim(1)
            let W = pixelValues.dim(2)
            let p = args.nativePatchSize
            let m = args.adapterMergeSize
            let block = p * m
            precondition(
                H % block == 0 && W % block == 0,
                "Inkling2 image size \(H)x\(W) must be a multiple of \(block)."
            )
            let rows = H / block
            let cols = W / block
            let patches = pixelValues
                .transposed(1, 2, 0)
                .reshaped(rows, m, p, cols, m, p, C)
                .transposed(0, 3, 1, 4, 2, 5, 6)
                .reshaped(-1, p * p * C)
            return encodePatches(patches, rows: rows, cols: cols)
        }

        /// Encodes a single image packed as `[rows * cols, C, block, block]` tiles.
        public func encodeTiles(_ tiles: MLXArray, rows: Int, cols: Int) -> MLXArray {
            let C = tiles.dim(1)
            let p = args.nativePatchSize
            let m = args.adapterMergeSize
            let patches = tiles
                .transposed(0, 2, 3, 1)
                .reshaped(rows * cols, m, p, m, p, C)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(-1, p * p * C)
            return encodePatches(patches, rows: rows, cols: cols)
        }

        private func encodePatches(_ patches: MLXArray, rows: Int, cols: Int) -> MLXArray {
            let m = args.adapterMergeSize
            let dtype = embeddings.patchEmbedding.weight.dtype
            var hiddenStates = embeddings(patches.asType(dtype), gridH: rows * m, gridW: cols * m)
            hiddenStates = preLayerNorm(hiddenStates)

            let (row, col) = gridPositions(gridH: rows * m, gridW: cols * m, merge: m)
            let zero = MLXArray.zeros([1], dtype: row.dtype)
            let (cos, sin) = rotaryEmb(
                row: concatenated([zero, row + Int32(1)], axis: 0),
                col: concatenated([zero, col + Int32(1)], axis: 0)
            )
            for layer in layers {
                hiddenStates = layer(hiddenStates, cos: cos, sin: sin)
            }
            let patchStates = hiddenStates[1...].reshaped(rows * cols, -1)
            return merger(patchStates)
        }
    }
}

// MARK: - Inkling 2 Multimodal VLM Model

final class Inkling2InnerModel: Module {
    @ModuleInfo(key: "language_model") var languageModel: Inkling2TextModel
    @ModuleInfo(key: "vision_tower") var visionTower: Inkling2Vision.VisionModel?

    init(_ config: Inkling2Configuration) {
        self._languageModel.wrappedValue = Inkling2TextModel(config.textConfig)
        if let visionConfig = config.visionConfig {
            self._visionTower.wrappedValue = Inkling2Vision.VisionModel(
                visionConfig, textHiddenSize: config.textConfig.hiddenSize)
        }
        super.init()
    }
}

public class Inkling2: Module, VLMModel {
    public let config: Inkling2Configuration

    @ModuleInfo(key: "model") var model: Inkling2InnerModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    private let logitProjection: Inkling2LogitProjection
    private var shardingGroup: DistributedGroup? = nil

    public var vocabularySize: Int {
        config.textConfig.unpaddedVocabSize
    }

    /// Looped models (`inkling2_loop`) execute middle layers 3 times per token with shared
    /// weights, so LoRA adapters are only exposed on non-looped `inkling2` configurations.
    public var loraLayers: [Module] {
        config.textConfig.loopConfig == nil ? model.languageModel.pipelineLayers : []
    }

    public required init(_ config: Inkling2Configuration) {
        self.config = config
        self._model.wrappedValue = Inkling2InnerModel(config)
        self._lmHead.wrappedValue = Linear(
            config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        self.logitProjection = Inkling2LogitProjection(config.textConfig)
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        // Sliding visits are always pinned to `slidingWindowSize` (1024) regardless of `parameters`.
        _ = parameters
        return model.languageModel.makeCache()
    }

    public func shard(group: DistributedGroup? = nil) throws {
        let resolvedGroup = group ?? DistributedGroup()
        let shardedHead = try Inkling2TensorParallel.shard(
            textModel: model.languageModel,
            lmHead: lmHead,
            args: config.textConfig,
            group: resolvedGroup
        )
        if resolvedGroup.size > 1 {
            self.update(modules: ModuleChildren.unflattened([("lm_head", shardedHead)]))
            self.shardingGroup = resolvedGroup
        }
    }

    public func pipeline(group: DistributedGroup, boundary: Int? = nil) throws {
        try model.languageModel.pipeline(group: group, boundary: boundary)
    }

    private func encodeProcessedImages(_ image: LMInput.ProcessedImage) throws -> MLXArray {
        guard let visionTower = model.visionTower else {
            throw VLMError.imageProcessingFailure("Inkling2 model has no vision_tower configured.")
        }
        let pixels = image.pixels
        if let frames = image.frames, !frames.isEmpty {
            var features: [MLXArray] = []
            features.reserveCapacity(frames.count)
            var offset = 0
            for thw in frames {
                let count = thw.product
                let slice = pixels[offset ..< (offset + count)]
                features.append(visionTower.encodeTiles(slice, rows: thw.h, cols: thw.w))
                offset += count
            }
            return features.count == 1 ? features[0] : concatenated(features, axis: 0)
        }
        if pixels.ndim == 3 {
            return visionTower(pixels)
        }
        throw VLMError.imageProcessingFailure(
            "Expected either 3D [C, H, W] pixels or 4D tiled pixels with THW frames, got \(pixels.shape)."
        )
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        let convertedCache: [KVCache]? = cache.isEmpty ? nil : cache.map { $0 }
        let step = max(1, windowSize ?? 256)

        if let processedImage = input.image {
            let imageFeatures = try encodeProcessedImages(processedImage)
            let expectedTokens = (tokens .== Int32(config.imageTokenId))
                .asType(.int32).sum().item(Int.self)
            guard expectedTokens == imageFeatures.dim(0) else {
                throw VLMError.processing(
                    "Image token count (\(expectedTokens)) does not match encoded vision features (\(imageFeatures.dim(0)))."
                )
            }

            let mergedEmbeddings = inkling2Splice(
                embeddings: model.languageModel.embed(tokens),
                inputs: tokens,
                rows: imageFeatures,
                tokenId: config.imageTokenId
            )

            // Prefill prefix chunks through `languageModel` ONLY (no 201k-vocab `lmHead` or FP32
            // softcap allocation on discarded prefix tokens), evaluating cache state per chunk.
            let totalPositions = mergedEmbeddings.dim(1)
            var processed = 0
            while totalPositions - processed > 1 {
                let chunkLen = min(step, totalPositions - processed - 1)
                let range = processed ..< (processed + chunkLen)
                _ = model.languageModel(
                    nil,
                    cache: convertedCache,
                    inputEmbeddings: mergedEmbeddings[0..., range, 0...]
                )
                if let convertedCache {
                    let stateArrays = convertedCache.flatMap { $0.innerState() }
                    if !stateArrays.isEmpty {
                        eval(stateArrays)
                    }
                }
                processed += chunkLen
            }

            let lastHidden = model.languageModel(
                nil,
                cache: convertedCache,
                inputEmbeddings: mergedEmbeddings[0..., processed..., 0...]
            )
            let logits = logitProjection.project(
                hiddenStates: lastHidden, lmHead: lmHead, shardingGroup: shardingGroup)
            return .logits(LMOutput(logits: logits))
        }

        let totalPositions = tokens.dim(1)
        var processed = 0
        while totalPositions - processed > 1 {
            let chunkLen = min(step, totalPositions - processed - 1)
            let range = processed ..< (processed + chunkLen)
            _ = model.languageModel(tokens[0..., range], cache: convertedCache)
            if let convertedCache {
                let stateArrays = convertedCache.flatMap { $0.innerState() }
                if !stateArrays.isEmpty {
                    eval(stateArrays)
                }
            }
            processed += chunkLen
        }

        let lastHidden = model.languageModel(tokens[0..., processed...], cache: convertedCache)
        let logits = logitProjection.project(
            hiddenStates: lastHidden, lmHead: lmHead, shardingGroup: shardingGroup)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [any KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        _ = state
        let convertedCache = cache?.map { $0 }
        var hidden = model.languageModel(input.tokens, cache: convertedCache)
        if hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        let logits = logitProjection.project(
            hiddenStates: hidden, lmHead: lmHead, shardingGroup: shardingGroup)
        return LMOutput(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]? = nil) -> MLXArray {
        let hidden = model.languageModel(inputs, cache: cache?.map { $0 })
        return logitProjection.project(
            hiddenStates: hidden, lmHead: lmHead, shardingGroup: shardingGroup)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Inkling2Checkpoint.sanitize(
            model: self,
            weights: weights,
            hasVision: model.visionTower != nil
        )
    }
}

// MARK: - Inkling 2 Processor Configuration & Processor

public struct Inkling2ProcessorConfiguration: Codable, Sendable {
    public struct SizeSpec: Codable, Sendable {
        public var height: Int
        public var width: Int

        public init(height: Int = 40, width: Int = 40) {
            self.height = height
            self.width = width
        }
    }

    public struct ImageProcessorSpec: Codable, Sendable {
        public var size: SizeSpec?
        public var imageMean: [Float]?
        public var imageStd: [Float]?
        public var rescaleFactor: Float?

        enum CodingKeys: String, CodingKey {
            case size
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case rescaleFactor = "rescale_factor"
        }
    }

    public var patchHeight: Int
    public var patchWidth: Int
    public var imageMean: [Float]
    public var imageStd: [Float]
    public var rescaleFactor: Float
    public var imageToken: String

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
        case imageToken = "image_token"
        case size
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case rescaleFactor = "rescale_factor"
        case visionConfig = "vision_config"
    }

    public init(
        patchHeight: Int = 40,
        patchWidth: Int = 40,
        imageMean: [Float] = [0.5, 0.5, 0.5],
        imageStd: [Float] = [0.5, 0.5, 0.5],
        rescaleFactor: Float = 1.0 / 255.0,
        imageToken: String = "<|unused_200054|>"
    ) {
        self.patchHeight = patchHeight
        self.patchWidth = patchWidth
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.rescaleFactor = rescaleFactor
        self.imageToken = imageToken
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(ImageProcessorSpec.self, forKey: .imageProcessor)
        let vision = try c.decodeIfPresent(
            Inkling2Configuration.VisionConfig.self, forKey: .visionConfig)
        let directSize = try c.decodeIfPresent(SizeSpec.self, forKey: .size)

        let defaultBlock = vision?.blockSize ?? 40
        let resolvedSize = nested?.size ?? directSize
        self.patchHeight = resolvedSize?.height ?? defaultBlock
        self.patchWidth = resolvedSize?.width ?? defaultBlock
        self.imageMean =
            try nested?.imageMean
            ?? c.decodeIfPresent([Float].self, forKey: .imageMean)
            ?? [0.5, 0.5, 0.5]
        self.imageStd =
            try nested?.imageStd
            ?? c.decodeIfPresent([Float].self, forKey: .imageStd)
            ?? [0.5, 0.5, 0.5]
        self.rescaleFactor =
            try nested?.rescaleFactor
            ?? c.decodeIfPresent(Float.self, forKey: .rescaleFactor)
            ?? (1.0 / 255.0)
        self.imageToken =
            try c.decodeIfPresent(String.self, forKey: .imageToken)
            ?? "<|unused_200054|>"
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(SizeSpec(height: patchHeight, width: patchWidth), forKey: .size)
        try c.encode(imageMean, forKey: .imageMean)
        try c.encode(imageStd, forKey: .imageStd)
        try c.encode(rescaleFactor, forKey: .rescaleFactor)
        try c.encode(imageToken, forKey: .imageToken)
    }
}

private let inkling2CIContext = CIContext(options: [.cacheIntermediates: false])

public struct Inkling2Processor: UserInputProcessor {
    public static let defaultMaxPixels = 4_194_304  // 2048 * 2048

    private let config: Inkling2ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Inkling2ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// Preprocesses one `CIImage` into `[rows * cols, 3, patchHeight, patchWidth]` float32 tiles
    /// and `THW(1, rows, cols)`:
    /// - Translates non-zero `extent.origin` to `.zero`
    /// - Applies `processing.resize` and proportional bicubic `maxPixels` (`4_194_304` default)
    /// - Calls `unpremultiplyingAlpha()` before sRGB `RGBA8` rasterization so semi-transparent
    ///   pixels are straight-alpha composited onto the 8×8 `(204, 153)` gray checkerboard
    /// - Pads right/bottom with `-1.0` on the `[0, 255]` scale before `(x / 255 - mean) / std`
    public func preprocessImage(
        _ rawImage: CIImage,
        processing: UserInput.Processing? = nil
    ) throws -> (MLXArray, THW) {
        var image = rawImage
        if image.extent.origin != .zero {
            image = image.transformed(
                by: CGAffineTransform(
                    translationX: -image.extent.origin.x,
                    y: -image.extent.origin.y
                ))
        }

        if let resize = processing?.resize {
            let target = CGSize(
                width: max(1, round(resize.width)),
                height: max(1, round(resize.height))
            )
            image = MediaProcessing.resampleBicubic(image, to: target)
        }

        let maxPixels = processing?.maxPixels ?? Self.defaultMaxPixels
        let curW = max(1, Int(image.extent.width.rounded()))
        let curH = max(1, Int(image.extent.height.rounded()))
        if maxPixels > 0 && curW * curH > maxPixels {
            let scale = sqrt(Double(maxPixels) / Double(curW * curH))
            let cappedW = max(1, Int(floor(Double(curW) * scale)))
            let cappedH = max(1, Int(floor(Double(curH) * scale)))
            image = MediaProcessing.resampleBicubic(
                image, to: CGSize(width: cappedW, height: cappedH))
        }

        let width = max(1, Int(image.extent.width.rounded()))
        let height = max(1, Int(image.extent.height.rounded()))
        guard let sRGB = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw VLMError.imageProcessingFailure("Unable to create sRGB color space.")
        }

        // Unpremultiply alpha prior to RGBA8 export so PIL's straight-alpha checkerboard blend
        // `(rgb * alpha + bg * (255 - alpha) + 127) / 255` does not multiply RGB by alpha twice.
        let straightImage = image.unpremultiplyingAlpha()
        let rowBytes = width * 4
        var rgbaBytes = [UInt8](repeating: 0, count: width * height * 4)
        rgbaBytes.withUnsafeMutableBytes { buf in
            inkling2CIContext.render(
                straightImage,
                toBitmap: buf.baseAddress!,
                rowBytes: rowBytes,
                bounds: CGRect(x: 0, y: 0, width: width, height: height),
                format: .RGBA8,
                colorSpace: sRGB
            )
            inkling2CIContext.clearCaches()
        }

        let ph = config.patchHeight
        let pw = config.patchWidth
        let rows = (height + ph - 1) / ph
        let cols = (width + pw - 1) / pw

        let meanR = config.imageMean.count > 0 ? config.imageMean[0] : 0.5
        let meanG = config.imageMean.count > 1 ? config.imageMean[1] : 0.5
        let meanB = config.imageMean.count > 2 ? config.imageMean[2] : 0.5
        let stdR = config.imageStd.count > 0 ? config.imageStd[0] : 0.5
        let stdG = config.imageStd.count > 1 ? config.imageStd[1] : 0.5
        let stdB = config.imageStd.count > 2 ? config.imageStd[2] : 0.5
        let rescale = config.rescaleFactor

        let padValR = (-1.0 * rescale - meanR) / stdR
        let padValG = (-1.0 * rescale - meanG) / stdG
        let padValB = (-1.0 * rescale - meanB) / stdB

        // Directly populate `[rows * cols, 3, ph, pw]` in row-major tile order.
        let tilePlane = ph * pw
        let tileStride = 3 * tilePlane
        var tiledFloats = [Float](repeating: 0, count: rows * cols * tileStride)

        for tileR in 0 ..< rows {
            for tileC in 0 ..< cols {
                let tileBase = (tileR * cols + tileC) * tileStride
                for localY in 0 ..< ph {
                    let y = tileR * ph + localY
                    for localX in 0 ..< pw {
                        let x = tileC * pw + localX
                        let pixelIdx = localY * pw + localX
                        if y < height && x < width {
                            let src = (y * width + x) * 4
                            let r0 = Int(rgbaBytes[src])
                            let g0 = Int(rgbaBytes[src + 1])
                            let b0 = Int(rgbaBytes[src + 2])
                            let a = Int(rgbaBytes[src + 3])

                            let r: Int
                            let g: Int
                            let b: Int
                            if a == 255 {
                                r = r0
                                g = g0
                                b = b0
                            } else {
                                let bg = (((x / 8) + (y / 8)) % 2 == 0) ? 204 : 153
                                let invA = 255 - a
                                r = (r0 * a + bg * invA + 127) / 255
                                g = (g0 * a + bg * invA + 127) / 255
                                b = (b0 * a + bg * invA + 127) / 255
                            }

                            tiledFloats[tileBase + pixelIdx] = (Float(r) * rescale - meanR) / stdR
                            tiledFloats[tileBase + tilePlane + pixelIdx] =
                                (Float(g) * rescale - meanG) / stdG
                            tiledFloats[tileBase + 2 * tilePlane + pixelIdx] =
                                (Float(b) * rescale - meanB) / stdB
                        } else {
                            tiledFloats[tileBase + pixelIdx] = padValR
                            tiledFloats[tileBase + tilePlane + pixelIdx] = padValG
                            tiledFloats[tileBase + 2 * tilePlane + pixelIdx] = padValB
                        }
                    }
                }
            }
        }

        let array = MLXArray(tiledFloats, [rows * cols, 3, ph, pw])
        return (array, THW(1, rows, cols))
    }

    private func buildMessages(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .messages(let messages):
            return messages

        case .text(let text):
            if input.images.isEmpty || text.contains(config.imageToken) {
                return [["role": "user", "content": text]]
            }
            var parts: [[String: any Sendable]] = input.images.map { _ in
                ["type": "image"]
            }
            if !text.isEmpty {
                parts.append(["type": "text", "text": text])
            }
            return [["role": "user", "content": parts]]

        case .chat(let chatMessages):
            return chatMessages.map { msg in
                if msg.images.isEmpty || msg.content.contains(config.imageToken) {
                    return [
                        "role": msg.role.rawValue,
                        "content": msg.content,
                    ]
                }
                var parts: [[String: any Sendable]] = msg.images.map { _ in
                    ["type": "image"]
                }
                if !msg.content.isEmpty {
                    parts.append(["type": "text", "text": msg.content])
                }
                return [
                    "role": msg.role.rawValue,
                    "content": parts,
                ]
            }
        }
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = buildMessages(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )

        if input.images.isEmpty {
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            // Keep `mask: nil` so multi-round sessions preserve cross-turn KV prefix reuse.
            return LMInput(text: .init(tokens: promptArray, mask: nil))
        }

        let imageTokenId = tokenizer.convertTokenToId(config.imageToken) ?? 200_054
        let encodedImages = try input.images.map { img in
            try preprocessImage(try img.asCIImage(), processing: input.processing)
        }
        let frames = encodedImages.map { $0.1 }
        let pixels =
            encodedImages.count == 1
            ? encodedImages[0].0
            : concatenated(encodedImages.map { $0.0 }, axis: 0)

        var expandedTokens: [Int] = []
        expandedTokens.reserveCapacity(
            promptTokens.count + frames.reduce(0) { $0 + max(0, $1.product - 1) }
        )
        var imageIndex = 0
        for tok in promptTokens {
            if tok == imageTokenId {
                guard imageIndex < frames.count else {
                    throw VLMError.processing(
                        "Prompt has more \(config.imageToken) placeholders than images (\(frames.count))."
                    )
                }
                let count = frames[imageIndex].product
                expandedTokens.append(contentsOf: repeatElement(imageTokenId, count: count))
                imageIndex += 1
            } else {
                expandedTokens.append(tok)
            }
        }
        guard imageIndex == frames.count else {
            throw VLMError.processing(
                "Prompt has \(imageIndex) \(config.imageToken) placeholders for \(frames.count) images."
            )
        }

        let promptArray = MLXArray(expandedTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: nil),
            image: .init(pixels: pixels, frames: frames)
        )
    }
}
