// Copyright © 2026 Orbital. Pipeline-parallel distributed inference.
//
// Companion to ``loadWeights`` for models too large to fit one machine. The model
// is sharded across a ``DistributedGroup`` ring (rank 0 = LAST decoder layers; the
// last rank embeds), each rank loads ONLY its shard's weight files, and a per-rank
// forward warmup faults the resident weights + compiles kernels before the first
// real turn (without it the first forward stalls ~59 s wiring weights into RAM and
// the peer parked at a pipeline collective trips the macOS GPU watchdog).
//
// Ported faithfully from the proven Python in /private/tmp/glm_dist/{patch.py,run.py}
// + stock mlx_lm `PipelineMixin` / `sharded_load`. Orbital stays agnostic that two
// machines are involved — this all lives in the SwiftLM/MLX layer.

import Foundation
import MLX
import MLXNN

/// A language model whose decoder layers can be sharded across a pipeline ring.
///
/// Conformers split their transformer layers in reverse (rank 0 holds the LAST
/// layers, the highest rank holds the FIRST), nil-ing the non-local layers so MLX
/// reflection drops them from `parameters()` while preserving layer indices.
public protocol PipelineParallel: AnyObject {
    /// Shard the decoder layers across `group`. Rank 0 = last layers; the last rank
    /// embeds. `boundary` (only honored for a 2-rank group) is the number of leading
    /// layers placed on the LAST rank (the peer) — pass a smaller value to give a
    /// memory-constrained peer fewer layers; `nil` = the stock even split. MUST be
    /// called before any `parameters()`/`update()`/forward, and identically on every
    /// rank (a mismatched boundary silently mis-places layers → fluent garbage).
    func pipeline(group: DistributedGroup, boundary: Int?)

    /// Flattened parameter keys (dotted, safetensors-style) that survive on THIS rank
    /// after `pipeline(group:boundary:)` — the union of local decoder layers plus the
    /// always-resident `embed_tokens` / `norm` / `lm_head`. Used to pick shard files.
    func localParameterKeys() -> [String]

    /// Drop weights for decoder layers NOT owned by this rank. A safetensors shard can
    /// straddle the rank boundary (e.g. contain both layer 45 and 46); loading the whole
    /// file then brings foreign-layer tensors that would collide with the model's nil
    /// holes on `update` (`(.none, .dictionary)` → throw). Keeps non-layer keys.
    func filterLocalWeights(_ weights: [String: MLXArray]) -> [String: MLXArray]

    /// Per-rank forward warmup: a LOCAL (no-collective) pass through this rank's own
    /// layers, per-layer `eval`, to fault weights resident + compile kernels. Runs AFTER
    /// the weights are loaded + quantized. NOT the distributed forward (that would block
    /// on the peer's collective).
    func warmup()

    /// The active pipeline ring group when this model is sharded across ≥2 machines, else nil.
    /// The decode loop uses it to broadcast the per-step sampled token from rank 0 to every rank
    /// (so a sampling — temp>0 — turn stays in lockstep despite per-rank RNG; for greedy it's a
    /// no-op since the all-gathered logits are already identical).
    var ringGroup: DistributedGroup? { get }
}

/// Process-global hook so the standard factory weight-load (``loadWeights``) transparently
/// routes to ``loadWeightsDistributed`` when a caller has set up a pipeline ring — without
/// threading a ``DistributedGroup`` through the whole `ModelFactory` API. Model loads are
/// serialized (one model per process), and this mirrors MLX's own process-global distributed
/// singleton, so a process-global is the right shape. Set it (with `defer` to clear) around a
/// factory load; ``loadWeights`` picks it up if the built model is ``PipelineParallel``.
public enum PipelineLoad {
    /// Active ring config for the in-flight load, or `nil` for a normal single-machine load.
    /// `@unchecked Sendable`: `DistributedGroup` is an opaque non-Sendable handle, but loads are
    /// serialized so there is no concurrent access.
    public final class Config: @unchecked Sendable {
        public let group: DistributedGroup
        public let boundary: Int?
        public init(group: DistributedGroup, boundary: Int?) {
            self.group = group
            self.boundary = boundary
        }
    }

    // nonisolated(unsafe): a plain process-global, safe because loads never overlap.
    nonisolated(unsafe) public static var active: Config?
}

/// Load weights for a pipeline-sharded model — the distributed analogue of ``loadWeights``.
///
/// Mirrors `mlx_lm.utils.sharded_load(pipeline_group=…)`: shard the model first, then
/// derive the local shard's weight files from the surviving `model.parameters()` via the
/// `model.safetensors.index.json` weight map, load ONLY those files, sanitize/quantize,
/// apply with `verify: .none` (boundary shards bring extra keys; non-local layers leave
/// holes), and finally warm up with a real forward.
public func loadWeightsDistributed(
    modelDirectory: URL,
    model: BaseLanguageModel,
    group: DistributedGroup,
    boundary: Int? = nil,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    guard let pp = model as? PipelineParallel else {
        throw PipelineParallelError.unsupportedModel(String(describing: type(of: model)))
    }

    // 1. Shard: nil the non-local layers (must precede parameters()/update()).
    pp.pipeline(group: group, boundary: boundary)

    // 2. Derive this rank's weight files from the surviving params + the index map.
    let localKeys = pp.localParameterKeys()
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    var shardFiles = Set<String>()
    if FileManager.default.fileExists(atPath: indexURL.path) {
        let data = try Data(contentsOf: indexURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let weightMap = (json?["weight_map"] as? [String: String]) ?? [:]
        for key in localKeys {
            // A quantized layer's .scales/.biases live in the SAME shard as its .weight,
            // so mapping the surviving (pre-quantization) .weight keys is sufficient.
            if let file = weightMap[key] { shardFiles.insert(file) }
        }
        if shardFiles.isEmpty {
            throw PipelineParallelError.noLocalShard(
                "no weight_map entries matched the local parameter keys "
                    + "(is this an MLX-converted model?)")
        }
    } else {
        // Single-file model: everything is in model.safetensors.
        shardFiles.insert("model.safetensors")
    }

    // 3. Load ONLY the local shard files.
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for file in shardFiles {
        let url = modelDirectory.appendingPathComponent(file)
        let (w, m) = try loadArraysAndMetadata(url: url)
        for (key, value) in w { weights[key] = value }
        if metadata.isEmpty { metadata = m }
    }

    // 4. Drop foreign-layer tensors carried in by boundary-straddling shards.
    weights = pp.filterLocalWeights(weights)

    // 5. Per-model cleanup (expert fusion is a no-op on already-fused MLX models;
    //    still filters the MTP layer + handles tied embeddings).
    weights = model.sanitize(weights: weights, metadata: metadata)

    // 6. Quantize if needed (only layers whose .scales were actually loaded).
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, _ in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // 7. Apply. verify: .none — non-local layers are intentionally absent (holes) and
    //    we already dropped straddling foreign tensors in step 4.
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: .none)
    eval(model)

    // 8. Warmup forward — faults the resident shard + compiles kernels (the watchdog fix).
    let distDbg = ProcessInfo.processInfo.environment["SWIFTLM_DIST_DEBUG"] != nil
    if distDbg { FileHandle.standardError.write(Data("[dist-load r\(group.rank)] warmup forward START (faulting shard) …\n".utf8)) }
    pp.warmup()
    if distDbg { FileHandle.standardError.write(Data("[dist-load r\(group.rank)] warmup forward DONE\n".utf8)) }

    // 9. Post-warmup barrier (Track A run.py:67): every rank waits HERE on the CPU stream (no GPU
    //    watchdog) until ALL ranks finish load+warmup. Without it the lighter-sharded rank (fewer
    //    layers) races into the collective-heavy turn while a heavier rank is still faulting ~100 GB
    //    of weights — its first forward then holds a GPU command buffer open past the ~45 s Metal
    //    watchdog waiting on the slow rank's `recv`, reproducing the original "GPU Timeout". A scalar
    //    all-sum is the cheapest barrier; eval forces it to actually round-trip.
    if group.size > 1 {
        let token = group.allSum(MLXArray(Float(1)), stream: .cpu)
        eval(token)
    }
}

public enum PipelineParallelError: LocalizedError {
    case unsupportedModel(String)
    case noLocalShard(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let type):
            "Model \(type) does not conform to PipelineParallel; cannot pipeline-shard it."
        case .noLocalShard(let message):
            "Pipeline shard selection failed: \(message)"
        }
    }
}
