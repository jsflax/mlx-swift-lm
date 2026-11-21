// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Inkling2TextModelContainer: Module {
    @ModuleInfo(key: "language_model") var languageModel: Inkling2TextModel

    init(_ textConfig: Inkling2Configuration.TextConfig) {
        self._languageModel.wrappedValue = Inkling2TextModel(textConfig)
        super.init()
    }
}

/// Text-only `LLMModel` wrapper for Thinking Machines Inkling 2 (`inkling2` / `inkling2_loop`).
public class Inkling2Model: Module, LLMModel {
    public let config: Inkling2Configuration

    @ModuleInfo(key: "model") var model: Inkling2TextModelContainer
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    private let logitProjection: Inkling2LogitProjection
    private var shardingGroup: DistributedGroup? = nil

    public var vocabularySize: Int {
        config.textConfig.unpaddedVocabSize
    }

    public var loraLayers: [Module] {
        config.textConfig.loopConfig == nil ? model.languageModel.pipelineLayers : []
    }

    public required init(_ config: Inkling2Configuration) {
        self.config = config
        self._model.wrappedValue = Inkling2TextModelContainer(config.textConfig)
        self._lmHead.wrappedValue = Linear(
            config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        self.logitProjection = Inkling2LogitProjection(config.textConfig)
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
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

    public func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        let convertedCache: [KVCache]? = cache.isEmpty ? nil : cache
        let step = max(1, windowSize ?? 256)
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
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        _ = state
        var hidden = model.languageModel(input.tokens, cache: cache)
        if hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        let logits = logitProjection.project(
            hiddenStates: hidden, lmHead: lmHead, shardingGroup: shardingGroup)
        return LMOutput(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = model.languageModel(inputs, cache: cache)
        return logitProjection.project(
            hiddenStates: hidden, lmHead: lmHead, shardingGroup: shardingGroup)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Inkling2Checkpoint.sanitize(model: self, weights: weights, hasVision: false)
    }
}
