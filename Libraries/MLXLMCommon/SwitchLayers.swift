import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    MLXNN.silu(gate) * up
}

public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        let activated =
            if let activationProduct {
                activationProduct(xGate, xUp)
            } else {
                activation(xGate) * xUp
            }
        x = downProj(
            activated,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - FusedGateUpSwitchGLU

/// SwitchGLU variant for models that ship a single fused `gate_up_proj` weight
/// of shape `[numExperts, 2*hiddenDims, inputDims]` instead of separate
/// `gate_proj` / `up_proj`. Used by Gemma 4 26B MoE.
public class FusedGateUpSwitchGLU: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let gateUp = gateUpProj(x, idx, sortedIndices: doSort)
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        let activated =
            if let activationProduct {
                activationProduct(parts[0], parts[1])
            } else {
                activation(parts[0]) * parts[1]
            }
        x = downProj(
            activated,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

private let compiledApplyNVFP4GlobalScale: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { out, selectedGlobalScale in
    // MLX NVFP4 convention: inv_scale_enc = global_scale / (448 * 6) = global_scale / 2688.0
    let factor = (selectedGlobalScale.asType(.float32) * Float(1.0 / 2688.0))
        .expandedDimensions(axes: [-2, -1])
    return (out.asType(.float32) * factor).asType(out.dtype)
}

// MARK: - SwitchLinear

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") public var weight: MLXArray
    @ModuleInfo(key: "bias") public var bias: MLXArray?

    public let inputDims: Int
    public let outputDims: Int
    public let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") public var scales: MLXArray
    @ModuleInfo(key: "biases") public var biases: MLXArray?
    @ModuleInfo(key: "global_scale") public var globalScale: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray? = nil,
        globalScale: MLXArray? = nil,
        bias: MLXArray? = nil,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        self._globalScale.wrappedValue = globalScale
        super.init(
            inputDims: inputDims, outputDims: outputDims, numExperts: numExperts,
            weight: weight, bias: bias)
        self.freeze()
    }

    public init(
        _ other: SwitchLinear,
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine,
        useGlobalScale: Bool = false
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        var sourceWeight = other.weight
        var computedGlobalScale: MLXArray? = nil
        if useGlobalScale && mode == .nvfp4 {
            let amax = maximum(
                abs(other.weight.asType(.float32)).max(axes: [-2, -1]),
                MLXArray(Float(1e-12))
            )
            let invScaleEnc = (amax * Float(1.0 / 2688.0)).expandedDimensions(axes: [-2, -1])
            sourceWeight = (other.weight.asType(.float32) / invScaleEnc).asType(other.weight.dtype)
            computedGlobalScale = amax
        }

        let (quantizedWeight, scales, biases) = MLX.quantized(
            sourceWeight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        self._globalScale.wrappedValue = computedGlobalScale

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let origDType = x.dtype
        // Promote FP16 inputs to BF16 when NVFP4 global_scale is active so unscaled
        // E4M3*E2M1 products (<= 2688) cannot overflow FP16 (65504) before post-scaling,
        // while keeping BF16 inputs on the hardware NAX fast path.
        let computeX =
            (self.globalScale != nil && origDType == .float16) ? x.asType(.bfloat16) : x
        var result = MLX.gatherQuantizedMM(
            computeX,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let globalScale = self.globalScale {
            result = compiledApplyNVFP4GlobalScale(result, globalScale[indices])
            if origDType == .float16 {
                result = result.asType(.float16)
            }
        }

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
