// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

package final class Inkling2TextModel: Module {
    package let args: Inkling2Configuration.TextConfig

    @ModuleInfo(key: "embed_tokens") package var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") package var embedNorm: RMSNorm
    @ModuleInfo(key: "layers") package var layers: [Inkling2DecoderLayer?]
    @ModuleInfo(key: "final_input") package var finalInput: Inkling2BlockAttentionResidual
    @ModuleInfo(key: "norm") package var norm: RMSNorm
    @ModuleInfo(key: "message_input") package var messageInput: Inkling2MessageInput?

    package var pipelineRank: Int = 0
    package var pipelineSize: Int = 1
    package var startIdx: Int = 0
    package var endIdx: Int

    package init(_ args: Inkling2Configuration.TextConfig) {
        self.args = args
        self.endIdx = args.numHiddenLayers

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self._embedNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        let localSet = Set(args.localLayerIds)
        self._layers.wrappedValue = (0 ..< args.numHiddenLayers).map { i in
            if localSet.contains(i) {
                return Inkling2SlidingDecoderLayer(args, layerIdx: i)
            } else {
                return Inkling2FullDecoderLayer(args, layerIdx: i)
            }
        }
        self._finalInput.wrappedValue = Inkling2BlockAttentionResidual(
            dims: args.hiddenSize, numQueries: 1)
        self._norm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if let loop = args.loopConfig, !loop.coreRange.isEmpty {
            self._messageInput.wrappedValue = Inkling2MessageInput(
                dims: args.hiddenSize, numLoopedLayers: loop.coreRange.count)
        }
        super.init()
    }

    package func embed(_ inputs: MLXArray) -> MLXArray {
        embedNorm(embedTokens(inputs))
    }

    package var pipelineLayers: [Inkling2DecoderLayer] {
        layers[startIdx ..< endIdx].compactMap { $0 }
    }

    /// Ordered sequence of layer visits per forward pass (`48` visits for `inkling2_loop`:
    /// `6` prelude + `3 × 12` middle core + `6` coda).
    package var layerVisits: [Inkling2DecoderLayer] {
        guard let loop = args.loopConfig else {
            return pipelineLayers
        }
        let core = loop.coreRange
        let prelude = layers[..<core.lowerBound].compactMap { $0 }
        let coreSlice = layers[core].compactMap { $0 }
        let coda = layers[core.upperBound...].compactMap { $0 }
        var visits: [Inkling2DecoderLayer] = []
        visits.reserveCapacity(prelude.count + coreSlice.count * loop.numPasses + coda.count)
        visits.append(contentsOf: prelude)
        for _ in 0 ..< loop.numPasses {
            visits.append(contentsOf: coreSlice)
        }
        visits.append(contentsOf: coda)
        return visits
    }

    /// Constructs the 48-visit composite cache (`40` `RotatingKVCache(maxSize: slidingWindowSize)`
    /// + `8` `KVCacheSimple`, each paired with `ArraysCache(size: 4)` for the 4 causal `ShortConv`
    /// states). Sliding visits are always pinned to `args.slidingWindowSize` (`1024`) regardless of
    /// caller `GenerateParameters.maxKVSize`.
    package func makeCache() -> [KVCache] {
        let window = args.slidingWindowSize
        return layerVisits.map { layer in
            let kv: KVCache =
                if layer is Inkling2SlidingDecoderLayer {
                    RotatingKVCache(maxSize: window, keep: 0)
                } else {
                    KVCacheSimple()
                }
            return CacheList(kv, ArraysCache(size: 4))
        }
    }

    package func pipeline(group: DistributedGroup, boundary: Int? = nil) throws {
        if args.loopConfig != nil {
            throw PipelineParallelError.unsupportedModel(
                "Pipeline parallel does not support looped models (inkling2_loop); use tensor-parallel shard(group:) instead."
            )
        }
        pipelineRank = group.rank
        pipelineSize = group.size
        let total = layers.count
        if pipelineSize == 2, let boundary, boundary > 0, boundary < total {
            // Rank 0 holds the last layers; rank 1 holds the leading `boundary` layers.
            if pipelineRank == 0 {
                startIdx = boundary
                endIdx = total
            } else {
                startIdx = 0
                endIdx = boundary
            }
        } else {
            let perRank = (total + pipelineSize - 1) / pipelineSize
            let revRank = pipelineSize - 1 - pipelineRank
            startIdx = min(total, revRank * perRank)
            endIdx = min(total, startIdx + perRank)
        }
        for idx in 0 ..< total where idx < startIdx || idx >= endIdx {
            layers[idx] = nil
        }
        // Rebuild Module's reflection cache after nil-ing non-local layers.
        _ = self.update(modules: ModuleChildren())
    }

    private func runLoop(
        _ hiddenStates: MLXArray,
        cache: [KVCache?],
        convMask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let loop = args.loopConfig!
        let core = loop.coreRange
        var cacheIdx = 0

        func runSlice(
            _ slice: ArraySlice<Inkling2DecoderLayer?>,
            _ hIn: MLXArray,
            _ blocksIn: MLXArray?,
            bank: (MLXArray, MLXArray)? = nil
        ) -> (MLXArray, MLXArray?) {
            var h = hIn
            var blocks = blocksIn
            for (i, layerOpt) in slice.enumerated() {
                let layer = layerOpt!
                let msgs = bank.map { messageInput!(bank: $0, loopLayerIdx: i) }
                let c = cacheIdx < cache.count ? (cache[cacheIdx] as? CacheList) : nil
                cacheIdx += 1
                let (nextH, nextBlocks) = layer(
                    h,
                    blockStates: blocks,
                    cache: c,
                    convMask: convMask,
                    messages: msgs
                )
                h = nextH
                blocks = nextBlocks
            }
            return (h, blocks)
        }

        let prelude = runSlice(layers[..<core.lowerBound], hiddenStates, nil)
        var h = prelude.0
        var blockStates = prelude.1
        let preludeBlockCount = prelude.1?.dim(0) ?? 0

        for pass in 0 ..< loop.numPasses {
            var bank: (MLXArray, MLXArray)? = nil
            if pass > 0, let currentBlocks = blockStates {
                bank = (
                    currentBlocks[(preludeBlockCount + 1)...],
                    inkling2UnweightedRMSNorm(h.asType(.float32), eps: args.rmsNormEps)
                )
                h = prelude.0
                blockStates = prelude.1
            }
            (h, blockStates) = runSlice(layers[core], h, blockStates, bank: bank)
        }

        let coda = runSlice(layers[core.upperBound...], h, blockStates)
        return (coda.0, coda.1!)
    }

    package func callAsFunction(
        _ inputs: MLXArray?,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbeddings {
            h = inputEmbeddings
        } else if var tokens = inputs {
            if tokens.ndim == 1 {
                tokens = tokens.expandedDimensions(axis: 0)
            }
            h = embed(tokens)
        } else {
            preconditionFailure("Inkling2TextModel requires either inputs or inputEmbeddings")
        }

        if h.ndim == 2 {
            h = h.expandedDimensions(axis: 0)
        }
        let origDType = h.dtype

        let activeVisitsCount = layerVisits.count
        let cacheSlots: [KVCache?] =
            if let cache, !cache.isEmpty {
                cache.map { Optional($0) }
            } else {
                Array(repeating: nil, count: activeVisitsCount)
            }

        var convMask: MLXArray? = nil
        if let firstList = cacheSlots.first ?? nil,
            let list = firstList as? CacheList,
            let arrays = list[1] as? ArraysCache
        {
            convMask = arrays.makeMask(N: h.dim(1))
        }

        var blockStates: MLXArray? = nil
        if args.loopConfig == nil {
            for (idx, layer) in pipelineLayers.enumerated() {
                let c = idx < cacheSlots.count ? (cacheSlots[idx] as? CacheList) : nil
                let (nextH, nextBlocks) = layer(
                    h, blockStates: blockStates, cache: c, convMask: convMask)
                h = nextH
                blockStates = nextBlocks
            }
        } else {
            let (loopH, loopBlocks) = runLoop(h, cache: cacheSlots, convMask: convMask)
            h = loopH
            blockStates = loopBlocks
        }

        let normedStates = inkling2UnweightedRMSNorm(h.asType(.float32), eps: args.rmsNormEps)
        h = finalInput(blockStates!, normedStates)[.ellipsis, 0, 0...]
        return norm(h).asType(origDType)
    }
}

// MARK: - Logit Projection & Tensor-Parallel Sharding

package struct Inkling2LogitProjection {
    let unpaddedVocabSize: Int
    private let softcapFn: @Sendable (MLXArray) -> MLXArray

    package init(_ args: Inkling2Configuration.TextConfig) {
        let cap = args.finalLogitSoftcapping
        self.unpaddedVocabSize = args.unpaddedVocabSize
        self.softcapFn = compile(shapeless: true) { rawLogits in
            let fp32 = rawLogits.asType(.float32)
            return tanh(fp32 / cap) * cap
        }
    }

    package func project(
        hiddenStates: MLXArray,
        lmHead: Linear,
        shardingGroup: DistributedGroup?
    ) -> MLXArray {
        var logits = lmHead(hiddenStates)
        if let group = shardingGroup, group.size > 1 {
            logits = inkling2GatherLastAxis(logits, group: group)
        }
        let capped = softcapFn(logits)
        return capped[.ellipsis, ..<unpaddedVocabSize]
    }
}

package enum Inkling2TensorParallel {
    private static func shardSlice(_ w: MLXArray, axis: Int, rank: Int, size: Int) -> MLXArray {
        let ax = axis < 0 ? w.ndim + axis : axis
        let chunk = w.dim(ax) / size
        let lo = rank * chunk
        let hi = lo + chunk
        var indices = Array(repeating: MLXArrayIndex.ellipsis, count: 0)
        _ = indices
        let splitParts = split(w, parts: size, axis: ax)
        let part = contiguous(splitParts[rank])
        _ = (lo, hi)
        return part
    }

    private static func shardAllToSharded(_ linear: Linear, rank: Int, size: Int) -> Linear {
        if let q = linear as? QuantizedLinear {
            return QuantizedLinear(
                weight: shardSlice(q.weight, axis: 0, rank: rank, size: size),
                bias: q.bias.map { shardSlice($0, axis: 0, rank: rank, size: size) },
                scales: shardSlice(q.scales, axis: 0, rank: rank, size: size),
                biases: q.biases.map { shardSlice($0, axis: 0, rank: rank, size: size) },
                groupSize: q.groupSize,
                bits: q.bits,
                mode: q.mode
            )
        }
        return Linear(
            weight: shardSlice(linear.weight, axis: 0, rank: rank, size: size),
            bias: linear.bias.map { shardSlice($0, axis: 0, rank: rank, size: size) }
        )
    }

    private static func shardShardedToAll(
        _ linear: Linear,
        group: DistributedGroup
    ) -> Linear {
        let rank = group.rank
        let size = group.size
        if let q = linear as? QuantizedLinear {
            return Inkling2QuantizedShardedToAllLinear(
                weight: shardSlice(q.weight, axis: 1, rank: rank, size: size),
                bias: q.bias,
                scales: shardSlice(q.scales, axis: 1, rank: rank, size: size),
                biases: q.biases.map { shardSlice($0, axis: 1, rank: rank, size: size) },
                groupSize: q.groupSize,
                bits: q.bits,
                mode: q.mode,
                group: group
            )
        }
        return Inkling2ShardedToAllLinear(
            weight: shardSlice(linear.weight, axis: 1, rank: rank, size: size),
            bias: linear.bias,
            group: group
        )
    }

    private static func shardExperts(
        _ layer: SwitchLinear,
        allToSharded: Bool,
        rank: Int,
        size: Int
    ) -> SwitchLinear {
        let axis = allToSharded ? 1 : 2
        let newInputDims = allToSharded ? layer.inputDims : (layer.inputDims / size)
        let newOutputDims = allToSharded ? (layer.outputDims / size) : layer.outputDims

        if let q = layer as? QuantizedSwitchLinear {
            let newWeight = shardSlice(q.weight, axis: axis, rank: rank, size: size)
            let newScales = shardSlice(q.scales, axis: axis, rank: rank, size: size)
            let newBiases = q.biases.map { shardSlice($0, axis: axis, rank: rank, size: size) }
            let newBias =
                allToSharded
                ? q.bias.map { shardSlice($0, axis: 1, rank: rank, size: size) }
                : q.bias
            // 1D per-expert NVFP4 `globalScale` (`[numExperts]`) is shared across the expert matrix
            // and intentionally remains unsharded.
            return QuantizedSwitchLinear(
                inputDims: newInputDims,
                outputDims: newOutputDims,
                numExperts: q.numExperts,
                weight: newWeight,
                scales: newScales,
                biases: newBiases,
                globalScale: q.globalScale,
                bias: newBias,
                groupSize: q.groupSize,
                bits: q.bits,
                mode: q.mode
            )
        }

        let newWeight = shardSlice(layer.weight, axis: axis, rank: rank, size: size)
        let newBias =
            allToSharded
            ? layer.bias.map { shardSlice($0, axis: 1, rank: rank, size: size) }
            : layer.bias
        return SwitchLinear(
            inputDims: newInputDims,
            outputDims: newOutputDims,
            numExperts: layer.numExperts,
            weight: newWeight,
            bias: newBias
        )
    }

    /// Shards `textModel` in-place and returns the sharded `lmHead` for `group`.
    package static func shard(
        textModel: Inkling2TextModel,
        lmHead: Linear,
        args: Inkling2Configuration.TextConfig,
        group: DistributedGroup
    ) throws -> Linear {
        let size = group.size
        let rank = group.rank
        guard size > 1 else { return lmHead }

        guard args.numKeyValueHeads % size == 0, args.swaNumKeyValueHeads % size == 0 else {
            throw DistributedError.invalidConfiguration(
                "Group size \(size) must divide num_key_value_heads (\(args.numKeyValueHeads), \(args.swaNumKeyValueHeads))."
            )
        }
        guard args.vocabSize % size == 0 else {
            throw DistributedError.invalidConfiguration(
                "Group size \(size) must divide vocab_size (\(args.vocabSize))."
            )
        }
        guard args.intermediateSize % size == 0 else {
            throw DistributedError.invalidConfiguration(
                "Group size \(size) must divide intermediate_size (\(args.intermediateSize))."
            )
        }

        let shardedHead = shardAllToSharded(lmHead, rank: rank, size: size)

        for layer in textModel.pipelineLayers {
            let attn = layer.attentionModule
            var attnModuleUpdates: [(String, Module)] = [
                ("q_proj", shardAllToSharded(attn.qProj, rank: rank, size: size)),
                ("k_proj", shardAllToSharded(attn.kProj, rank: rank, size: size)),
                ("v_proj", shardAllToSharded(attn.vProj, rank: rank, size: size)),
                ("r_proj", shardAllToSharded(attn.rProj, rank: rank, size: size)),
                ("gate_proj", shardAllToSharded(attn.gateProj, rank: rank, size: size)),
                ("sink_proj", shardAllToSharded(attn.sinkProj, rank: rank, size: size)),
                ("o_proj", shardShardedToAll(attn.oProj, group: group)),
            ]

            attn.qL2Clip.update(
                parameters: ModuleParameters.unflattened([
                    "scale": shardSlice(attn.qL2Clip.scale, axis: 0, rank: rank, size: size)
                ]))
            attn.relL2Clip.update(
                parameters: ModuleParameters.unflattened([
                    "scale": shardSlice(attn.relL2Clip.scale, axis: 0, rank: rank, size: size)
                ]))

            if let sliding = attn as? Inkling2SlidingAttention {
                attnModuleUpdates.append(
                    (
                        "k_sconv",
                        Inkling2ShortConvolution(
                            shardedWeight: shardSlice(
                                sliding.kSconv.conv1d.weight, axis: 0, rank: rank, size: size),
                            kernelSize: sliding.kSconv.kernelSize
                        )
                    ))
                attnModuleUpdates.append(
                    (
                        "v_sconv",
                        Inkling2ShortConvolution(
                            shardedWeight: shardSlice(
                                sliding.vSconv.conv1d.weight, axis: 0, rank: rank, size: size),
                            kernelSize: sliding.vSconv.kernelSize
                        )
                    ))
            } else if let full = attn as? Inkling2FullAttention {
                for conv in [full.kSconv, full.vSconv] {
                    conv.update(
                        parameters: ModuleParameters.unflattened([
                            "static_weight": shardSlice(
                                conv.staticWeight, axis: 1, rank: rank, size: size)
                        ]))
                    conv.update(
                        modules: ModuleChildren.unflattened([
                            (
                                "weight_proj",
                                shardAllToSharded(conv.weightProj, rank: rank, size: size)
                            )
                        ]))
                    conv.numHeads /= size
                }
            }

            attn.update(modules: ModuleChildren.unflattened(attnModuleUpdates))
            attn.numHeads /= size
            attn.numKeyValueHeads /= size

            layer.mlp.shardingGroup = group
            let experts = layer.mlp.experts
            experts.update(
                modules: ModuleChildren.unflattened([
                    (
                        "gate_up_proj",
                        shardExperts(
                            experts.gateUpProj, allToSharded: true, rank: rank, size: size)
                    ),
                    (
                        "down_proj",
                        shardExperts(
                            experts.downProj, allToSharded: false, rank: rank, size: size)
                    ),
                ]))
        }

        return shardedHead
    }
}
