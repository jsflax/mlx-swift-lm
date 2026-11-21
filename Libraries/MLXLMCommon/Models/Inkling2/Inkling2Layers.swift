// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Normalization & L2 Clipping

package func inkling2UnweightedRMSNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x, weight: nil, eps: eps)
}

package final class Inkling2L2Clip: Module {
    let threshold: Float
    private let clipFn: @Sendable (MLXArray) -> MLXArray

    package init(threshold: Float) {
        self.threshold = threshold
        self.clipFn = compile(shapeless: true) { hiddenStates in
            let origDType = hiddenStates.dtype
            let states = hiddenStates.asType(.float32)
            let norm = sqrt((states * states).sum(axis: -1, keepDims: true))
            let scaled = states * minimum(threshold / (norm + Float(1e-6)), Float(1.0))
            return scaled.asType(origDType)
        }
        super.init()
    }

    package func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        clipFn(hiddenStates)
    }
}

package final class Inkling2ScaledL2Clip: Module {
    let threshold: Float
    @ParameterInfo(key: "scale") package var scale: MLXArray
    private let clipFn: @Sendable (MLXArray, MLXArray) -> MLXArray

    package init(numHeads: Int, headDim: Int, threshold: Float) {
        self.threshold = threshold
        self._scale.wrappedValue = MLXArray.ones([numHeads, headDim])
        self.clipFn = compile(shapeless: true) { hiddenStates, scale in
            let origDType = hiddenStates.dtype
            let states = hiddenStates.asType(.float32) * scale.asType(.float32)
            let norm = sqrt((states * states).sum(axis: -1, keepDims: true))
            let scaled = states * minimum(threshold / (norm + Float(1e-6)), Float(1.0))
            return scaled.asType(origDType)
        }
        super.init()
    }

    package func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        clipFn(hiddenStates, scale)
    }
}

package final class Inkling2RelativeLogits: Module {
    @ParameterInfo(key: "proj") package var proj: MLXArray

    package init(dRel: Int, relExtent: Int) {
        self._proj.wrappedValue = MLXArray.zeros([dRel, relExtent])
        super.init()
    }

    package func callAsFunction(_ relativeStates: MLXArray) -> MLXArray {
        matmul(relativeStates, proj.asType(relativeStates.dtype))
    }
}

// MARK: - Causal Short Convolutions

private func inkling2ConvInput(
    _ hiddenStates: MLXArray,
    convState: MLXArray?,
    convMask: MLXArray?,
    kernelSize: Int
) -> MLXArray {
    var states = hiddenStates.asType(.float32)
    if let convMask {
        states = which(convMask[.ellipsis, .newAxis], states, MLXArray(Float(0.0)))
    }
    let prefix =
        convState
        ?? MLXArray.zeros([states.dim(0), kernelSize - 1, states.dim(-1)], dtype: .float32)
    return concatenated([prefix, states], axis: 1)
}

package func inkling2ConvState(
    window: MLXArray,
    lengths: MLXArray?,
    n: Int
) -> MLXArray {
    let width = window.dim(1)
    let pos: MLXArray
    if let lengths {
        let ends = clip(lengths, min: 0, max: width - n)
        pos = (ends[0..., .newAxis] + MLXArray(0 ..< n))[.ellipsis, .newAxis]
    } else {
        pos = MLXArray((width - n) ..< width)[.newAxis, 0..., .newAxis]
    }
    return takeAlong(window, pos, axis: 1)
}

package final class Inkling2ShortConvolution: Module {
    package let kernelSize: Int
    private let invScale: Float
    @ModuleInfo(key: "conv1d") package var conv1d: Conv1d

    package init(dims: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self.invScale = 1.0 / sqrt(Float(kernelSize + 1))
        self._conv1d.wrappedValue = Conv1d(
            inputChannels: dims,
            outputChannels: dims,
            kernelSize: kernelSize,
            groups: dims,
            bias: false
        )
        super.init()
    }

    package init(shardedWeight: MLXArray, kernelSize: Int) {
        let dims = shardedWeight.dim(0)
        self.kernelSize = kernelSize
        self.invScale = 1.0 / sqrt(Float(kernelSize + 1))
        let layer = Conv1d(
            inputChannels: dims,
            outputChannels: dims,
            kernelSize: kernelSize,
            groups: dims,
            bias: false
        )
        layer.update(parameters: ModuleParameters.unflattened(["weight": shardedWeight]))
        self._conv1d.wrappedValue = layer
        super.init()
    }

    package func callAsFunction(
        _ hiddenStates: MLXArray,
        convState: MLXArray?,
        convMask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let origDType = hiddenStates.dtype
        let residual = hiddenStates.asType(.float32)
        let window = inkling2ConvInput(
            hiddenStates, convState: convState, convMask: convMask, kernelSize: kernelSize)
        let out = conv1d(window.asType(conv1d.weight.dtype))
        let combined = (out.asType(.float32) + residual) * invScale
        return (combined.asType(origDType), window)
    }
}

package final class Inkling2DynamicShortConvolution: Module {
    package let kernelSize: Int
    package var numHeads: Int
    private let invScale: Float
    @ParameterInfo(key: "static_weight") package var staticWeight: MLXArray
    @ModuleInfo(key: "weight_proj") package var weightProj: Linear

    package init(dims: Int, inputDims: Int, numHeads: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self.numHeads = numHeads
        self.invScale = 1.0 / sqrt(Float(2 * kernelSize + 1))
        self._staticWeight.wrappedValue = MLXArray.zeros([kernelSize, dims])
        self._weightProj.wrappedValue = Linear(inputDims, numHeads * kernelSize, bias: false)
        super.init()
    }

    package func callAsFunction(
        _ hiddenStates: MLXArray,
        dynamicInput: MLXArray,
        convState: MLXArray?,
        convMask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let origDType = hiddenStates.dtype
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)
        let C = hiddenStates.dim(2)
        let K = kernelSize
        let H = numHeads

        let dynamicWeights = weightProj(dynamicInput).asType(.float32).reshaped(B, L, H, 1, K)
        let staticWeights = staticWeight.asType(.float32).T.reshaped(H, -1, K)
        let weights = staticWeights + dynamicWeights

        let window = inkling2ConvInput(
            hiddenStates, convState: convState, convMask: convMask, kernelSize: K)
        let lags = (0 ..< K).map { lag in
            let start = K - 1 - lag
            return window[0..., start ..< (start + L), 0...]
        }
        let windows = stacked(lags, axis: -1).reshaped(B, L, H, -1, K)
        let convOut = (windows * weights).sum(axis: -1).reshaped(B, L, C)
        let out = (convOut + hiddenStates.asType(.float32)) * invScale
        return (out.asType(origDType), window)
    }
}

// MARK: - Tensor-Parallel Sharded Linear Wrappers

package final class Inkling2ShardedToAllLinear: Linear {
    let group: DistributedGroup

    package init(weight: MLXArray, bias: MLXArray?, group: DistributedGroup) {
        self.group = group
        super.init(weight: weight, bias: bias)
    }

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = matmul(x, weight.T)
        if group.size > 1 {
            out = group.allSum(out)
        }
        if let bias {
            out = out + bias
        }
        return out
    }
}

package final class Inkling2QuantizedShardedToAllLinear: QuantizedLinear {
    let group: DistributedGroup

    package init(
        weight: MLXArray,
        bias: MLXArray?,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        group: DistributedGroup
    ) {
        self.group = group
        super.init(
            weight: weight,
            bias: bias,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        self.freeze()
    }

    override public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        if group.size > 1 {
            out = group.allSum(out)
        }
        if let bias {
            out = out + bias
        }
        return out
    }
}

// MARK: - Relative & Sink Attention

private func inkling2Positions(offset: Int, length: Int) -> MLXArray {
    MLXArray(0 ..< length) + Int32(offset)
}

private func inkling2KeyPositions(
    cache: KVCache?,
    numKeys: Int,
    numQueries: Int,
    end: Int
) -> MLXArray {
    var k = MLXArray(0 ..< numKeys)
    if numQueries == 1, let rotating = cache as? RotatingKVCache, numKeys > 0 {
        // Add `numKeys` before `% numKeys` so the unrotated ring-buffer index is always >= 0.
        let rotIdx = rotating.rotationIndex
        k = (k - Int32(rotIdx) + Int32(numKeys)) % Int32(numKeys)
    }
    return k + Int32(end - numKeys)
}

package func inkling2AttentionReference(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    positionBias: MLXArray,
    sinks: MLXArray,
    scale: Float,
    window: Int,
    queryPositions: MLXArray,
    keyPositions: MLXArray
) -> MLXArray {
    let B = query.dim(0)
    let Hq = query.dim(1)
    let L = query.dim(2)
    let D = query.dim(3)
    let Hkv = key.dim(1)
    let S = key.dim(2)
    let R = positionBias.dim(-1)

    var distance = queryPositions[.ellipsis, 0..., .newAxis] - keyPositions[.ellipsis, .newAxis, 0...]
    var visible = (distance .>= 0) & (keyPositions[.ellipsis, .newAxis, 0...] .>= 0)
    if window > 0 {
        visible = visible & (distance .< window)
    }
    if distance.ndim == 3 {
        distance = distance[0..., .newAxis, 0..., 0...]
        visible = visible[0..., .newAxis, 0..., 0...]
    }

    let gatheredBias = takeAlong(
        positionBias.transposed(0, 2, 1, 3),
        broadcast(clip(distance, min: 0, max: R - 1), to: [B, Hq, L, S]),
        axis: -1
    )
    let bias = which(distance .< R, gatheredBias.asType(.float32), MLXArray(Float(0.0)))
    let attnMask = which(visible, bias, MLXArray(-Float.infinity)).reshaped(B, Hkv, -1, L, S)

    let q32 = query.asType(.float32).reshaped(B, Hkv, -1, L, D)
    let k32 = key.asType(.float32)[0..., 0..., .newAxis, 0..., 0...]
    let logits = matmul(q32, k32.swappedAxes(-1, -2)) * scale + attnMask
    let sinkLogits = sinks.asType(.float32).transposed(0, 2, 1).reshaped(B, Hkv, -1, L, 1)
    let combined = concatenated([logits, sinkLogits], axis: -1)
    let probs = softmax(combined, axis: -1)[.ellipsis, ..<S]
    let vExpanded = value[0..., 0..., .newAxis, 0..., 0...]
    let out = matmul(probs.asType(value.dtype), vExpanded)
    return out.reshaped(B, Hq, L, D).transposed(0, 2, 1, 3)
}

package class Inkling2Attention: Module {
    package var numHeads: Int
    package var numKeyValueHeads: Int
    package let headDim: Int
    package let slidingWindow: Int
    package let scaling: Float
    package let eps: Float

    @ModuleInfo(key: "q_proj") package var qProj: Linear
    @ModuleInfo(key: "k_proj") package var kProj: Linear
    @ModuleInfo(key: "v_proj") package var vProj: Linear
    @ModuleInfo(key: "r_proj") package var rProj: Linear
    @ModuleInfo(key: "o_proj") package var oProj: Linear
    @ModuleInfo(key: "gate_proj") package var gateProj: Linear
    @ModuleInfo(key: "sink_proj") package var sinkProj: Linear
    @ModuleInfo(key: "rel_logits_proj") package var relLogitsProj: Inkling2RelativeLogits
    @ModuleInfo(key: "q_l2_clip") package var qL2Clip: Inkling2ScaledL2Clip
    @ModuleInfo(key: "k_l2_clip") package var kL2Clip: Inkling2L2Clip
    @ModuleInfo(key: "rel_l2_clip") package var relL2Clip: Inkling2ScaledL2Clip

    package init(_ args: Inkling2Configuration.TextConfig, sliding: Bool) {
        let relExtent: Int
        if sliding {
            self.numHeads = args.swaNumAttentionHeads
            self.numKeyValueHeads = args.swaNumKeyValueHeads
            self.headDim = args.swaHeadDim
            self.slidingWindow = args.slidingWindowSize
            relExtent = args.slidingWindowSize
        } else {
            self.numHeads = args.numAttentionHeads
            self.numKeyValueHeads = args.numKeyValueHeads
            self.headDim = args.headDim
            self.slidingWindow = 0
            relExtent = args.relExtent
        }
        self.scaling = 1.0 / sqrt(Float(self.headDim))
        self.eps = args.rmsNormEps

        let dim = args.hiddenSize
        let qDim = self.numHeads * self.headDim
        let kvDim = self.numKeyValueHeads * self.headDim

        self._qProj.wrappedValue = Linear(dim, qDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, kvDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, kvDim, bias: false)
        self._rProj.wrappedValue = Linear(dim, self.numHeads * args.dRel, bias: false)
        self._oProj.wrappedValue = Linear(qDim, dim, bias: false)
        self._gateProj.wrappedValue = Linear(dim, qDim, bias: false)
        self._sinkProj.wrappedValue = Linear(dim, self.numHeads, bias: false)
        self._relLogitsProj.wrappedValue = Inkling2RelativeLogits(
            dRel: args.dRel, relExtent: relExtent)
        self._qL2Clip.wrappedValue = Inkling2ScaledL2Clip(
            numHeads: self.numHeads, headDim: self.headDim, threshold: args.qkL2ClipThreshold)
        self._kL2Clip.wrappedValue = Inkling2L2Clip(threshold: args.qkL2ClipThreshold)
        self._relL2Clip.wrappedValue = Inkling2ScaledL2Clip(
            numHeads: self.numHeads, headDim: args.dRel, threshold: args.relL2ClipThreshold)
        super.init()
    }

    package func attend(
        hiddenStates: MLXArray,
        queryStates: MLXArray,
        keyStates: MLXArray,
        valueStates: MLXArray,
        relativeStates: MLXArray,
        cache: KVCache?,
        tau: MLXArray? = nil
    ) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)
        let gate = gateProj(hiddenStates)
        let sinks = sinkProj(hiddenStates)

        var q = qL2Clip(queryStates.reshaped(B, L, numHeads, -1))
        var k = kL2Clip(keyStates.reshaped(B, L, numKeyValueHeads, -1))
        var v = inkling2UnweightedRMSNorm(
            valueStates.reshaped(B, L, numKeyValueHeads, -1).asType(.float32), eps: eps
        ).asType(valueStates.dtype)
        var rel = relL2Clip(relativeStates.reshaped(B, L, numHeads, -1))

        if let tau {
            q = (q.asType(.float32) * tau).asType(q.dtype)
            rel = (rel.asType(.float32) * tau).asType(rel.dtype)
        }
        let positionBias = relLogitsProj(rel)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let attnOut = computeAttention(
            query: q, key: k, value: v, positionBias: positionBias, sinks: sinks, cache: cache)
        let gated =
            attnOut.reshaped(B, L, -1).asType(.float32) * sigmoid(gate.asType(.float32))
        return oProj(gated.asType(gate.dtype))
    }

    private func computeAttention(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        positionBias: MLXArray,
        sinks: MLXArray,
        cache: KVCache?
    ) -> MLXArray {
        let B = query.dim(0)
        let Hq = query.dim(1)
        let L = query.dim(2)
        let D = query.dim(3)
        let Hkv = key.dim(1)
        let S = key.dim(2)
        let end = cache?.offset ?? L

        let isBF16 =
            query.dtype == .bfloat16 && key.dtype == .bfloat16 && value.dtype == .bfloat16
        if L > 1 && D >= 8 && D % 8 == 0 && Hq % Hkv == 0 && inkling2UseKernels() && isBF16 {
            let keyStart = broadcast(MLXArray(Int32(S - end)), to: [B])
            return inkling2RelativeAttention(
                q: query,
                k: key,
                v: value,
                rel: positionBias,
                sinks: sinks,
                scale: scaling,
                window: slidingWindow,
                keyStart: keyStart
            )
        }

        let queryPositions = inkling2Positions(offset: end - L, length: L)
        let keyPositions = inkling2KeyPositions(cache: cache, numKeys: S, numQueries: L, end: end)

        if L == 1 && B == 1 {
            let distance = queryPositions.reshaped(1) - keyPositions.reshaped(S)
            let R = positionBias.dim(-1)
            var bias = take(positionBias[0, 0], clip(distance, min: 0, max: R - 1), axis: -1)
            bias = which(distance .< R, bias, MLXArray.zeros(like: bias))
            var visible = (distance .>= 0) & (keyPositions.reshaped(S) .>= 0)
            if slidingWindow > 0 {
                visible = visible & (distance .< slidingWindow)
            }
            let mask = which(visible, bias, MLXArray(-Float.infinity)).asType(query.dtype)
            let attnOut = MLXFast.scaledDotProductAttention(
                queries: query,
                keys: key,
                values: value,
                scale: scaling,
                mask: .array(mask[.newAxis, 0..., .newAxis, 0...]),
                sinks: sinks[0, 0].asType(query.dtype)
            )
            return attnOut.transposed(0, 2, 1, 3)
        }

        return inkling2AttentionReference(
            query: query,
            key: key,
            value: value,
            positionBias: positionBias,
            sinks: sinks,
            scale: scaling,
            window: slidingWindow,
            queryPositions: queryPositions,
            keyPositions: keyPositions
        )
    }
}

package final class Inkling2SlidingAttention: Inkling2Attention {
    @ModuleInfo(key: "k_sconv") package var kSconv: Inkling2ShortConvolution
    @ModuleInfo(key: "v_sconv") package var vSconv: Inkling2ShortConvolution

    package init(_ args: Inkling2Configuration.TextConfig) {
        let dims = args.swaNumKeyValueHeads * args.swaHeadDim
        self._kSconv.wrappedValue = Inkling2ShortConvolution(
            dims: dims, kernelSize: args.sconvKernelSize)
        self._vSconv.wrappedValue = Inkling2ShortConvolution(
            dims: dims, kernelSize: args.sconvKernelSize)
        super.init(args, sliding: true)
    }

    package func callAsFunction(
        hiddenStates: MLXArray,
        keyHiddenStates: MLXArray,
        valueHiddenStates: MLXArray,
        cache: CacheList?,
        convStates: inout [MLXArray?],
        convMask: MLXArray?
    ) -> MLXArray {
        let kvCache = cache?[0]
        let (kStates, kWin) = kSconv(
            kProj(keyHiddenStates), convState: convStates[0], convMask: convMask)
        convStates[0] = kWin
        let (vStates, vWin) = vSconv(
            vProj(valueHiddenStates), convState: convStates[1], convMask: convMask)
        convStates[1] = vWin
        return attend(
            hiddenStates: hiddenStates,
            queryStates: qProj(hiddenStates),
            keyStates: kStates,
            valueStates: vStates,
            relativeStates: rProj(hiddenStates),
            cache: kvCache
        )
    }
}

package final class Inkling2FullAttention: Inkling2Attention {
    @ModuleInfo(key: "k_sconv") package var kSconv: Inkling2DynamicShortConvolution
    @ModuleInfo(key: "v_sconv") package var vSconv: Inkling2DynamicShortConvolution
    let logScalingNFloor: Int
    let logScalingAlpha: Float

    package init(_ args: Inkling2Configuration.TextConfig) {
        let dims = args.numKeyValueHeads * args.headDim
        self._kSconv.wrappedValue = Inkling2DynamicShortConvolution(
            dims: dims,
            inputDims: args.hiddenSize,
            numHeads: args.numKeyValueHeads,
            kernelSize: args.sconvKernelSize
        )
        self._vSconv.wrappedValue = Inkling2DynamicShortConvolution(
            dims: dims,
            inputDims: args.hiddenSize,
            numHeads: args.numKeyValueHeads,
            kernelSize: args.sconvKernelSize
        )
        self.logScalingNFloor = args.logScalingNFloor
        self.logScalingAlpha = args.logScalingAlpha
        super.init(args, sliding: false)
    }

    package func callAsFunction(
        hiddenStates: MLXArray,
        keyHiddenStates: MLXArray,
        valueHiddenStates: MLXArray,
        cache: CacheList?,
        convStates: inout [MLXArray?],
        convMask: MLXArray?
    ) -> MLXArray {
        let kvCache = cache?[0]
        let (kStates, kWin) = kSconv(
            kProj(keyHiddenStates),
            dynamicInput: keyHiddenStates,
            convState: convStates[0],
            convMask: convMask
        )
        convStates[0] = kWin
        let (vStates, vWin) = vSconv(
            vProj(valueHiddenStates),
            dynamicInput: valueHiddenStates,
            convState: convStates[1],
            convMask: convMask
        )
        convStates[1] = vWin

        let offset = kvCache?.offset ?? 0
        let posIds = inkling2Positions(offset: offset, length: hiddenStates.dim(1)).asType(.float32)
        let ratio = maximum((posIds + Float(1.0)) / Float(logScalingNFloor), Float(1.0))
        let tau = (Float(1.0) + logScalingAlpha * log(ratio))[.ellipsis, .newAxis, .newAxis]

        return attend(
            hiddenStates: hiddenStates,
            queryStates: qProj(hiddenStates),
            keyStates: kStates,
            valueStates: vStates,
            relativeStates: rProj(hiddenStates),
            cache: kvCache,
            tau: tau
        )
    }
}

// MARK: - Softcapped SwiGLU MoE

package final class Inkling2SoftcappedSwiGLU: Module {
    let gateSoftcap: Float
    let upSoftcap: Float
    private let actFn: @Sendable (MLXArray, MLXArray) -> MLXArray

    package init(_ args: Inkling2Configuration.TextConfig) {
        let gateCap = args.mlpGateSoftcap
        let upCap = args.mlpUpSoftcap
        self.gateSoftcap = gateCap
        self.upSoftcap = upCap
        self.actFn = compile(shapeless: true) { gate, up in
            let origDType = gate.dtype
            let g = gate.asType(.float32)
            let u = up.asType(.float32)
            let gCapped = tanh(g / gateCap) * gateCap
            let uCapped = tanh(u / upCap) * upCap
            return (gCapped * sigmoid(g) * uCapped).asType(origDType)
        }
        super.init()
    }

    package func callAsFunction(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
        actFn(gate, up)
    }
}

package final class Inkling2Experts: Module {
    @ModuleInfo(key: "gate_up_proj") package var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") package var downProj: SwitchLinear
    @ModuleInfo(key: "act_fn") package var actFn: Inkling2SoftcappedSwiGLU

    package init(_ args: Inkling2Configuration.TextConfig) {
        let dims = args.hiddenSize
        let hidden = args.intermediateSize
        let numExperts = args.nRoutedExperts
        // Checkpoint alternates even/odd rows between gate and up projections.
        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: dims, outputDims: 2 * hidden, numExperts: numExperts, bias: false)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hidden, outputDims: dims, numExperts: numExperts, bias: false)
        self._actFn.wrappedValue = Inkling2SoftcappedSwiGLU(args)
        super.init()
    }

    package func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let detachedIndices = stopGradient(indices)
        let doSort = detachedIndices.size >= 64
        var idx = detachedIndices
        var invOrder = MLXArray()
        if doSort {
            (expanded, idx, invOrder) = gatherSort(x: expanded, indices: detachedIndices)
        }
        let gateUp = gateUpProj(expanded, idx, sortedIndices: doSort)
        let gate = gateUp[.ellipsis, .stride(by: 2)]
        let up = gateUp[.ellipsis, .stride(from: 1, by: 2)]
        var out = downProj(actFn(gate, up), idx, sortedIndices: doSort)
        if doSort {
            out = scatterUnsort(x: out, invOrder: invOrder, shape: detachedIndices.shape)
        }
        return MLX.squeezed(out, axis: -2)
    }
}

package final class Inkling2TopKRouter: Module {
    let topK: Int
    let routedScalingFactor: Float
    @ParameterInfo(key: "weight") package var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") package var scoreCorrectionBias: MLXArray
    private let compiledFallback: @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray)

    package init(_ args: Inkling2Configuration.TextConfig) {
        let k = args.numExpertsPerTok
        let scale = args.routeScale
        self.topK = k
        self.routedScalingFactor = scale
        self._weight.wrappedValue = MLXArray.zeros(
            [args.nRoutedExperts, args.hiddenSize], dtype: .float32)
        self._scoreCorrectionBias.wrappedValue = MLXArray.zeros(
            [args.nRoutedExperts], dtype: .float32)
        self.compiledFallback = compile(shapeless: true) { logits, bias in
            inkling2ExpertRoutingReference(
                logits: logits, bias: bias, topK: k, routeScale: scale)
        }
        super.init()
    }

    package func callAsFunction(_ hiddenStates: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(hiddenStates.asType(.float32), weight.asType(.float32).T)
        if inkling2UseKernels() {
            return inkling2ExpertRouting(
                logits: logits,
                bias: scoreCorrectionBias,
                topK: topK,
                routeScale: routedScalingFactor
            )
        }
        return compiledFallback(logits, scoreCorrectionBias)
    }
}

package final class Inkling2SparseMoeBlock: Module {
    @ModuleInfo(key: "gate") package var gate: Inkling2TopKRouter
    @ModuleInfo(key: "experts") package var experts: Inkling2Experts
    package var shardingGroup: DistributedGroup? = nil

    package init(_ args: Inkling2Configuration.TextConfig) {
        self._gate.wrappedValue = Inkling2TopKRouter(args)
        self._experts.wrappedValue = Inkling2Experts(args)
        super.init()
    }

    package func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let origDType = hiddenStates.dtype
        let (selectedExperts, routingWeights) = gate(hiddenStates)
        let expertStates = experts(hiddenStates, selectedExperts)
        var finalStates =
            (expertStates.asType(.float32) * routingWeights[.ellipsis, .newAxis]).sum(axis: -2)
        if let group = shardingGroup, group.size > 1 {
            finalStates = group.allSum(finalStates)
        }
        return finalStates.asType(origDType)
    }
}

// MARK: - Block Attention Residual & Loop Message Gates

package final class Inkling2BlockAttentionResidual: Module {
    @ParameterInfo(key: "query") package var query: MLXArray

    package init(dims: Int, numQueries: Int = 1) {
        let shape = numQueries == 1 ? [dims] : [numQueries, dims]
        self._query.wrappedValue = MLXArray.zeros(shape, dtype: .float32)
        super.init()
    }

    package func callAsFunction(
        _ blockStates: MLXArray,
        _ partialBlockStates: MLXArray
    ) -> MLXArray {
        let dims = partialBlockStates.dim(-1)
        return inkling2DepthAttention(
            blocks: blockStates,
            partial: partialBlockStates,
            queries: query.reshaped(-1, dims)
        )
    }
}

package final class Inkling2MessageGate: Module {
    let eps: Float
    @ModuleInfo(key: "down_proj") package var downProj: Linear
    @ModuleInfo(key: "forget_gate_proj") package var forgetGateProj: Linear
    @ModuleInfo(key: "message_gate_proj") package var messageGateProj: Linear

    package init(dims: Int, rank: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._downProj.wrappedValue = Linear(2 * dims, rank, bias: false)
        self._forgetGateProj.wrappedValue = Linear(rank, dims, bias: false)
        self._messageGateProj.wrappedValue = Linear(rank, dims, bias: false)
        super.init()
    }

    package func callAsFunction(_ hiddenStates: MLXArray, _ message: MLXArray) -> MLXArray {
        let joined = concatenated(
            [
                inkling2UnweightedRMSNorm(hiddenStates, eps: eps),
                inkling2UnweightedRMSNorm(message, eps: eps),
            ],
            axis: -1
        )
        let down = downProj(joined.asType(downProj.weight.dtype)).asType(.float32)
        let forgetGate = tanh(forgetGateProj(down))
        let messageGate = tanh(messageGateProj(down))
        return (Float(1.0) - forgetGate) * hiddenStates + messageGate * message
    }
}

package final class Inkling2MessageInput: Module {
    @ParameterInfo(key: "query") package var query: MLXArray

    package init(dims: Int, numLoopedLayers: Int) {
        // [num_looped_layers, 4 (q, k, v, mlp), dims]
        self._query.wrappedValue = MLXArray.zeros([numLoopedLayers, 4, dims], dtype: .float32)
        super.init()
    }

    package func callAsFunction(
        bank: (MLXArray, MLXArray),
        loopLayerIdx: Int
    ) -> MLXArray {
        inkling2DepthAttention(
            blocks: bank.0,
            partial: bank.1,
            queries: query[loopLayerIdx]
        )
    }
}

// MARK: - Decoder Layers

package class Inkling2DecoderLayer: Module {
    package let isBlockBoundary: Bool
    package let eps: Float

    @ModuleInfo(key: "attention_input") package var attentionInput: Inkling2BlockAttentionResidual?
    @ModuleInfo(key: "mlp_input") package var mlpInput: Inkling2BlockAttentionResidual
    @ModuleInfo(key: "mlp") package var mlp: Inkling2SparseMoeBlock
    @ModuleInfo(key: "post_attention_layernorm") package var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_convolution") package var postAttentionConvolution:
        Inkling2ShortConvolution
    @ModuleInfo(key: "post_mlp_convolution") package var postMlpConvolution:
        Inkling2ShortConvolution

    @ModuleInfo(key: "q_message_gate") package var qMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "k_message_gate") package var kMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "v_message_gate") package var vMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "mlp_message_gate") package var mlpMessageGate: Inkling2MessageGate?

    package init(_ args: Inkling2Configuration.TextConfig, layerIdx: Int, numQueries: Int) {
        let dims = args.hiddenSize
        self.eps = args.rmsNormEps
        self.isBlockBoundary = (layerIdx % args.nLayersPerAttnResBlock) == 0
        if layerIdx > 0 {
            self._attentionInput.wrappedValue = Inkling2BlockAttentionResidual(
                dims: dims, numQueries: numQueries)
        }
        self._mlpInput.wrappedValue = Inkling2BlockAttentionResidual(dims: dims, numQueries: 1)
        self._mlp.wrappedValue = Inkling2SparseMoeBlock(args)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: dims, eps: args.rmsNormEps)
        self._postAttentionConvolution.wrappedValue = Inkling2ShortConvolution(
            dims: dims, kernelSize: args.sconvKernelSize)
        self._postMlpConvolution.wrappedValue = Inkling2ShortConvolution(
            dims: dims, kernelSize: args.sconvKernelSize)

        if let loop = args.loopConfig, loop.coreRange.contains(layerIdx) {
            let rank = loop.gateRank
            self._qMessageGate.wrappedValue = Inkling2MessageGate(
                dims: dims, rank: rank, eps: args.rmsNormEps)
            self._kMessageGate.wrappedValue = Inkling2MessageGate(
                dims: dims, rank: rank, eps: args.rmsNormEps)
            self._vMessageGate.wrappedValue = Inkling2MessageGate(
                dims: dims, rank: rank, eps: args.rmsNormEps)
            self._mlpMessageGate.wrappedValue = Inkling2MessageGate(
                dims: dims, rank: rank, eps: args.rmsNormEps)
        }
        super.init()
    }

    fileprivate func mixMessages(
        _ q: MLXArray,
        _ k: MLXArray,
        _ v: MLXArray,
        messages: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        (
            qMessageGate!(q, messages[.ellipsis, 0, 0...]),
            kMessageGate!(k, messages[.ellipsis, 1, 0...]),
            vMessageGate!(v, messages[.ellipsis, 2, 0...])
        )
    }

    package func runSelfAttention(
        hiddenStates: MLXArray,
        attentionInputs: MLXArray?,
        cache: CacheList?,
        convStates: inout [MLXArray?],
        convMask: MLXArray?,
        messages: MLXArray?
    ) -> MLXArray {
        fatalError("Subclasses must override runSelfAttention")
    }

    package var attentionModule: Inkling2Attention {
        fatalError("Subclasses must override attentionModule")
    }

    package func callAsFunction(
        _ hiddenStates: MLXArray,
        blockStates: MLXArray?,
        cache: CacheList? = nil,
        convMask: MLXArray? = nil,
        messages: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        let origDType = hiddenStates.dtype
        let convCache = cache?[1] as? ArraysCache
        var convStates: [MLXArray?] =
            if let convCache {
                (0 ..< 4).map { convCache[$0] }
            } else {
                [nil, nil, nil, nil]
            }

        let blockResidual = hiddenStates
        let normedResidual = inkling2UnweightedRMSNorm(
            blockResidual.asType(.float32), eps: eps)

        var attnInputs: MLXArray? = nil
        if let attentionInput, let blockStates {
            attnInputs = attentionInput(blockStates, normedResidual)
        }

        var h = runSelfAttention(
            hiddenStates: hiddenStates,
            attentionInputs: attnInputs,
            cache: cache,
            convStates: &convStates,
            convMask: convMask,
            messages: messages
        )
        let (postAttn, postAttnWin) = postAttentionConvolution(
            h, convState: convStates[2], convMask: convMask)
        h = postAttn
        convStates[2] = postAttnWin

        var updatedBlocks: MLXArray
        var residual = h.asType(.float32)
        h = residual
        if isBlockBoundary {
            let block = normedResidual[.newAxis]
            if let blockStates {
                updatedBlocks = concatenated([blockStates, block], axis: 0)
            } else {
                updatedBlocks = block
            }
        } else {
            updatedBlocks = blockStates!
            residual = blockResidual.asType(.float32) + residual
            h = residual
        }

        let normedStates = inkling2UnweightedRMSNorm(h, eps: eps)
        h = mlpInput(updatedBlocks, normedStates)[.ellipsis, 0, 0...]
        if let messages, let mlpMessageGate {
            h = mlpMessageGate(h, messages[.ellipsis, 3, 0...])
        }
        h = mlp(postAttentionLayerNorm(h).asType(origDType))
        let (postMlp, postMlpWin) = postMlpConvolution(
            h, convState: convStates[3], convMask: convMask)
        h = postMlp
        convStates[3] = postMlpWin

        if let convCache {
            let keepLen = postMlpConvolution.kernelSize - 1
            let lengths = convCache.currentLengths
            for i in 0 ..< 4 {
                if let win = convStates[i] {
                    convCache[i] = inkling2ConvState(window: win, lengths: lengths, n: keepLen)
                }
            }
            convCache.advance(blockResidual.dim(1))
        }

        let out = (residual + h.asType(.float32)).asType(origDType)
        return (out, updatedBlocks)
    }
}

package final class Inkling2SlidingDecoderLayer: Inkling2DecoderLayer {
    @ModuleInfo(key: "self_attn") package var selfAttn: Inkling2SlidingAttention
    @ModuleInfo(key: "input_layernorm") package var inputLayerNorm: RMSNorm

    package init(_ args: Inkling2Configuration.TextConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = Inkling2SlidingAttention(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init(args, layerIdx: layerIdx, numQueries: 1)
    }

    override package var attentionModule: Inkling2Attention { selfAttn }

    override package func runSelfAttention(
        hiddenStates: MLXArray,
        attentionInputs: MLXArray?,
        cache: CacheList?,
        convStates: inout [MLXArray?],
        convMask: MLXArray?,
        messages: MLXArray?
    ) -> MLXArray {
        let dtype = hiddenStates.dtype
        let base = attentionInputs.map { $0[.ellipsis, 0, 0...] } ?? hiddenStates
        let qIn: MLXArray
        let kIn: MLXArray
        let vIn: MLXArray
        if let messages {
            let (qM, kM, vM) = mixMessages(base, base, base, messages: messages)
            qIn = inputLayerNorm(qM).asType(dtype)
            kIn = inputLayerNorm(kM).asType(dtype)
            vIn = inputLayerNorm(vM).asType(dtype)
        } else {
            let normed = inputLayerNorm(base).asType(dtype)
            qIn = normed
            kIn = normed
            vIn = normed
        }
        return selfAttn(
            hiddenStates: qIn,
            keyHiddenStates: kIn,
            valueHiddenStates: vIn,
            cache: cache,
            convStates: &convStates,
            convMask: convMask
        )
    }
}

package final class Inkling2FullDecoderLayer: Inkling2DecoderLayer {
    @ModuleInfo(key: "self_attn") package var selfAttn: Inkling2FullAttention
    @ModuleInfo(key: "q_input_layernorm") package var qInputLayerNorm: RMSNorm
    @ModuleInfo(key: "k_input_layernorm") package var kInputLayerNorm: RMSNorm
    @ModuleInfo(key: "v_input_layernorm") package var vInputLayerNorm: RMSNorm

    package init(_ args: Inkling2Configuration.TextConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = Inkling2FullAttention(args)
        self._qInputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._kInputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._vInputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init(args, layerIdx: layerIdx, numQueries: 3)
    }

    override package var attentionModule: Inkling2Attention { selfAttn }

    override package func runSelfAttention(
        hiddenStates: MLXArray,
        attentionInputs: MLXArray?,
        cache: CacheList?,
        convStates: inout [MLXArray?],
        convMask: MLXArray?,
        messages: MLXArray?
    ) -> MLXArray {
        let dtype = hiddenStates.dtype
        var qS: MLXArray
        var kS: MLXArray
        var vS: MLXArray
        if let attentionInputs {
            qS = attentionInputs[.ellipsis, 0, 0...]
            kS = attentionInputs[.ellipsis, 1, 0...]
            vS = attentionInputs[.ellipsis, 2, 0...]
        } else {
            qS = hiddenStates
            kS = hiddenStates
            vS = hiddenStates
        }
        if let messages {
            (qS, kS, vS) = mixMessages(qS, kS, vS, messages: messages)
        }
        return selfAttn(
            hiddenStates: qInputLayerNorm(qS).asType(dtype),
            keyHiddenStates: kInputLayerNorm(kS).asType(dtype),
            valueHiddenStates: vInputLayerNorm(vS).asType(dtype),
            cache: cache,
            convStates: &convStates,
            convMask: convMask
        )
    }
}
