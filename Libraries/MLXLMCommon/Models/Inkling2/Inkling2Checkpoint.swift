// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Checkpoint sanitization supporting both raw Thinking Machines checkpoints
/// (`thinkingmachines/Inkling-2-Mini` with `model.llm.*` / `model.visual.*`, NVFP4
/// `_amax`/`_outer_scale_denominator`, MXFP8, and bundled `mtp.safetensors`) and
/// pre-converted MLX checkpoints (`mlx-community/Inkling-2-Mini-4bit` with 4-bit affine
/// dense layers + NVFP4/MXFP8 MoE experts).
package enum Inkling2Checkpoint {
    private static let nvfp4MaxProduct: Float = 6.0 * 448.0  // 2688.0

    private static let attnQueryRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: #"^model\.llm\.attn_res\.attn_queries\.(\d+)$"#)
    }()

    private static let rawExpertsRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: #"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$"#)
    }()

    private static let renameRules: [(NSRegularExpression, String)] = {
        let pairs: [(String, String)] = [
            (#"^model\.llm\.unembed\."#, "lm_head."),
            (#"^model\.llm\.embed\."#, "model.language_model.embed_tokens."),
            (#"^model\.llm\."#, "model.language_model."),
            (#"\.attn_res\.mlp_queries\.(\d+)$"#, ".layers.$1.mlp_input.query"),
            (#"\.attn_res\.output_query$"#, ".final_input.query"),
            (#"\.attn\.wq_du\."#, ".self_attn.q_proj."),
            (#"\.attn\.wk_dv\."#, ".self_attn.k_proj."),
            (#"\.attn\.wv_dv\."#, ".self_attn.v_proj."),
            (#"\.attn\.wr_du\."#, ".self_attn.r_proj."),
            (#"\.attn\.wo_ud\."#, ".self_attn.o_proj."),
            (#"\.attn\.gate_attn\."#, ".self_attn.gate_proj."),
            (#"\.attn\.attn_sink_proj\."#, ".self_attn.sink_proj."),
            (#"\.controllers\.attn\."#, ".q_message_gate."),
            (#"\.controllers\.(k|v|mlp)\."#, ".$1_message_gate."),
            (#"_message_gate\.down_weight$"#, "_message_gate.down_proj.weight"),
            (#"_message_gate\.forget_weight$"#, "_message_gate.forget_gate_proj.weight"),
            (#"_message_gate\.gain_weight$"#, "_message_gate.message_gate_proj.weight"),
            (#"\.attn\."#, ".self_attn."),
            (#"\.attn_sconv\."#, ".post_attention_convolution."),
            (#"\.mlp_sconv\."#, ".post_mlp_convolution."),
            (#"sconv\.weight$"#, "sconv.conv1d.weight"),
            (#"convolution\.weight$"#, "convolution.conv1d.weight"),
            (#"sconv\.static_w$"#, "sconv.static_weight"),
            (#"sconv\.weight_proj$"#, "sconv.weight_proj.weight"),
            (#"\.mlp_norm\."#, ".post_attention_layernorm."),
            (#"\.attn_norm\."#, ".input_layernorm."),
            (#"\.(q|k|v)_input_norm\."#, ".$1_input_layernorm."),
            (#"\.mlp\.gate\.bias$"#, ".mlp.gate.e_score_correction_bias"),
            (#"^model\.visual\.backbone\.ln_pre\."#, "model.vision_tower.pre_layernorm."),
            (#"^model\.visual\.backbone\."#, "model.vision_tower."),
            (#"^model\.visual\.proj_in\."#, "model.vision_tower.merger.linear_1."),
            (#"^model\.visual\.proj_out\."#, "model.vision_tower.merger.linear_2."),
            (#"embeddings\.position_embedding$"#, "embeddings.position_embedding.weight"),
            (#"(vision_tower\.layers\.\d+)\.ln_1\."#, "$1.layernorm_before."),
            (#"(vision_tower\.layers\.\d+)\.ln_2\."#, "$1.layernorm_after."),
            (#"(vision_tower\.layers\.\d+)\.mlp\.c_fc\."#, "$1.mlp.fc1."),
            (#"(vision_tower\.layers\.\d+)\.mlp\.c_proj\."#, "$1.mlp.fc2."),
            (#"(vision_tower\.layers\.\d+)\.ls_(\d)\.gamma$"#, "$1.lambda_$2"),
        ]
        return pairs.map { pattern, replacement in
            (try! NSRegularExpression(pattern: pattern), replacement)
        }
    }()

    private static func isMTPKey(_ key: String) -> Bool {
        key == "fuse_weight"
            || key == "mask_embed"
            || key == "norm.weight"
            || key.hasPrefix("layers.")
            || key.hasPrefix("markov_head.")
            || key.hasPrefix("mtp.")
            || key.hasPrefix("model.mtp.")
    }

    private static func renameRawKey(_ key: String) -> String {
        let fullRange = NSRange(key.startIndex ..< key.endIndex, in: key)
        if let match = attnQueryRegex.firstMatch(in: key, range: fullRange),
            let idxRange = Range(match.range(at: 1), in: key),
            let idx = Int(key[idxRange])
        {
            // Layer 0 has no attention_input, so attn_queries.i belongs to layer i + 1.
            return "model.language_model.layers.\(idx + 1).attention_input.query"
        }

        var current = key
        for (regex, replacement) in renameRules {
            let range = NSRange(current.startIndex ..< current.endIndex, in: current)
            current = regex.stringByReplacingMatches(
                in: current, range: range, withTemplate: replacement)
        }
        return current
    }

    private static func requiresFP32(_ key: String) -> Bool {
        key.hasSuffix(".query")
            || key.hasSuffix(".mlp.gate.weight")
            || key.hasSuffix(".mlp.gate.e_score_correction_bias")
            || key.hasSuffix(".forget_gate_proj.weight")
            || key.hasSuffix(".message_gate_proj.weight")
            || key.hasSuffix(".global_scale")
    }

    package static func sanitize(
        model: Module,
        weights: [String: MLXArray],
        hasVision: Bool
    ) -> [String: MLXArray] {
        var amax: [String: MLXArray] = [:]
        var denominator: [String: MLXArray] = [:]
        var messageQueriesQ: MLXArray? = nil
        var messageQueriesKV: MLXArray? = nil
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (rawKey, rawValue) in weights {
            if isMTPKey(rawKey) {
                continue
            }

            var k = rawKey
            var v = rawValue

            if k.hasPrefix("model.llm.") || k.hasPrefix("model.visual.") {
                k = renameRawKey(k)
                if k.hasSuffix("conv1d.weight") {
                    v = v.swappedAxes(1, 2)
                }
            } else if k.hasPrefix("model.embed_tokens.")
                || k.hasPrefix("model.embed_norm.")
                || k.hasPrefix("model.layers.")
                || k.hasPrefix("model.final_input.")
                || k.hasPrefix("model.norm.")
                || k.hasPrefix("model.message_input.")
            {
                k = "model.language_model." + k.dropFirst("model.".count)
            }

            if !hasVision && k.hasPrefix("model.vision_tower.") {
                continue
            }

            if k.hasSuffix("cross_pass_kv_queries") {
                messageQueriesKV = v
                continue
            }
            if k.hasSuffix("cross_pass_queries") {
                messageQueriesQ = v
                continue
            }

            let nsRange = NSRange(k.startIndex ..< k.endIndex, in: k)
            if let match = rawExpertsRegex.firstMatch(in: k, range: nsRange),
                let prefixRange = Range(match.range(at: 1), in: k),
                let whichRange = Range(match.range(at: 2), in: k),
                let suffixRange = Range(match.range(at: 3), in: k)
            {
                let prefix = String(k[prefixRange])
                let which = String(k[whichRange])
                let suffix = String(k[suffixRange])
                let proj = which == "w13" ? "gate_up_proj" : "down_proj"
                let name = "\(prefix).\(proj)"

                switch suffix {
                case "_amax":
                    amax[name] = v
                case "_outer_scale_denominator":
                    denominator[name] = v
                case "_scale":
                    sanitized["\(name).scales"] = v
                case "":
                    sanitized["\(name).weight"] =
                        (v.dtype == .uint8) ? v.view(dtype: .uint32) : v
                default:
                    sanitized["\(name)\(suffix)"] = v
                }
                continue
            }

            if requiresFP32(k) && v.dtype != .float32 {
                v = v.asType(.float32)
            }
            sanitized[k] = v
        }

        if let qRaw = messageQueriesQ, let kvRaw = messageQueriesKV {
            let dims = qRaw.dim(-1)
            let q = qRaw.asType(.float32).reshaped(-1, 2, dims)
            let kv = kvRaw.asType(.float32).reshaped(-1, 2, dims)
            let combined = concatenated(
                [q[0..., ..<1, 0...], kv, q[0..., 1..., 0...]],
                axis: 1
            )
            sanitized["model.language_model.message_input.query"] = combined
        }

        for (name, amaxVal) in amax {
            guard let denomVal = denominator[name] else { continue }
            let scale = nvfp4MaxProduct / denomVal.asType(.float32)
            sanitized["\(name).global_scale"] = amaxVal.asType(.float32) * scale
        }

        // Convert FP-quantized SwitchLinear experts (NVFP4 / MXFP4 / MXFP8) in-place before
        // Load.swift runs affine `quantize(model:)`. Already-Quantized modules are skipped by
        // `quantizeSingle`, preserving mixed NVFP4 + MXFP8 experts alongside 4-bit affine dense layers.
        var moduleUpdates: [(String, Module)] = []
        for (path, module) in model.namedModules() {
            guard let switchLinear = module as? SwitchLinear,
                !(module is QuantizedSwitchLinear),
                let scales = sanitized["\(path).scales"],
                scales.dtype == .uint8,
                let qWeight = sanitized["\(path).weight"]
            else {
                continue
            }

            let inDims = switchLinear.inputDims
            guard scales.dim(-1) > 0, inDims % scales.dim(-1) == 0 else { continue }
            let groupSize = inDims / scales.dim(-1)
            let bits = (qWeight.dim(-1) * 32) / inDims

            let mode: QuantizationMode
            switch (bits, groupSize) {
            case (4, 16):
                mode = .nvfp4
            case (4, 32):
                mode = .mxfp4
            case (8, 32):
                mode = .mxfp8
            default:
                continue
            }

            let quantizedLayer = QuantizedSwitchLinear(
                inputDims: switchLinear.inputDims,
                outputDims: switchLinear.outputDims,
                numExperts: switchLinear.numExperts,
                weight: qWeight,
                scales: scales,
                biases: sanitized["\(path).biases"],
                globalScale: sanitized["\(path).global_scale"],
                bias: switchLinear.bias,
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
            moduleUpdates.append((path, quantizedLayer))
        }

        if !moduleUpdates.isEmpty {
            model.update(modules: ModuleChildren.unflattened(moduleUpdates))
        }

        return sanitized
    }
}
