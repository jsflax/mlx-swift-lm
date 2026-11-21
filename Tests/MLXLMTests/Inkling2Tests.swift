// Copyright © 2026 Apple Inc.

import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

struct Inkling2Tests {

    // MARK: - Synthetic Configurations (intermediateSize >= 8, hiddenSize >= 16)

    private static func makeBaseConfig(looped: Bool = false) -> Inkling2Configuration {
        let loop: Inkling2Configuration.LoopConfig? =
            looped
            ? .init(coreStart: 3, coreEnd: 6, numPasses: 3, gateRank: 8, isVarloop: false)
            : nil
        let text = Inkling2Configuration.TextConfig(
            modelType: looped ? "inkling2_loop_text" : "inkling2_text",
            modelMaxLength: 4096,
            hiddenSize: 64,
            numHiddenLayers: 8,
            vocabSize: 128,
            unpaddedVocabSize: 120,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 16,
            swaNumAttentionHeads: 4,
            swaNumKeyValueHeads: 2,
            swaHeadDim: 16,
            slidingWindowSize: 8,
            localLayerIds: [0, 1, 2, 3, 4, 6],
            dRel: 8,
            relExtent: 8,
            logScalingNFloor: 4,
            logScalingAlpha: 0.1,
            rmsNormEps: 1e-6,
            qkL2ClipThreshold: 3.0,
            relL2ClipThreshold: 2.0,
            sconvKernelSize: 4,
            nLayersPerAttnResBlock: 3,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            intermediateSize: 64,
            mlpGateSoftcap: 4.0,
            mlpUpSoftcap: 25.0,
            routeScale: 2.0,
            finalLogitSoftcapping: 30.0,
            loopConfig: loop
        )
        let vision = Inkling2Configuration.VisionConfig(
            hiddenSize: 32,
            numHiddenLayers: 2,
            numAttentionHeads: 2,
            numChannels: 3,
            nativePatchSize: 2,
            adapterMergeSize: 2,
            posembGridSize: 4,
            projectionHiddenDim: 48,
            projectorBias: false,
            mlpRatio: 2.0,
            layerNormEps: 1e-5,
            ropeTheta: 10000.0
        )
        return Inkling2Configuration(
            modelType: looped ? "inkling2_loop" : "inkling2",
            textConfig: text,
            visionConfig: vision,
            imageTokenId: 119
        )
    }

    private static func seedWeights<M: Module>(_ model: M) {
        MLXRandom.seed(42)
        var initialized: [(String, MLXArray)] = []
        for (key, value) in model.parameters().flattened() {
            let v: MLXArray
            if key.hasSuffix("norm.weight")
                || key.hasSuffix("l2_clip.scale")
                || key.hasSuffix("lambda_1")
                || key.hasSuffix("lambda_2")
            {
                v = (Float(1.0) + Float(0.1) * MLXRandom.normal(value.shape)).asType(value.dtype)
            } else {
                let fanIn = Float(value.ndim > 1 ? value.dim(-1) : 16)
                v = (MLXRandom.normal(value.shape) * (Float(1.0) / sqrt(fanIn))).asType(value.dtype)
            }
            initialized.append((key, v))
        }
        model.update(parameters: ModuleParameters.unflattened(initialized))
        eval(model)
    }

    // MARK: - Layer 1: Unweighted MLXFast.rmsNorm

    @Test("MLXFast.rmsNorm supports nil weight for unweighted RMS normalization")
    func testUnweightedRMSNorm() {
        MLXRandom.seed(7)
        let x = MLXRandom.normal([2, 5, 64]).asType(.float32)
        let eps: Float = 1e-6
        let fast = MLXFast.rmsNorm(x, weight: nil, eps: eps)
        let manual = x * rsqrt((x * x).mean(axis: -1, keepDims: true) + eps)
        eval(fast, manual)
        #expect(allClose(fast, manual, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    }

    // MARK: - Custom Metal Kernels vs Reference Parity

    @Test("Inkling 2 Metal kernels match reference ops (depth, routing, relative attention)")
    func testCustomKernelsMatchReference() {
        guard inkling2UseKernels() else { return }
        MLXRandom.seed(11)

        // 1. Depth attention (Q = 1 and Q = 4)
        for qCount in [1, 4] {
            let blocks = MLXRandom.normal([3, 2, 5, 64]).asType(.float32)
            let partial = MLXRandom.normal([2, 5, 64]).asType(.float32)
            let queries = MLXRandom.normal([qCount, 64]).asType(.float32)
            let gpu = inkling2DepthAttention(
                blocks: blocks, partial: partial, queries: queries, useKernel: true)
            let ref = inkling2DepthAttentionReference(
                blocks: blocks, partial: partial, queries: queries)
            eval(gpu, ref)
            #expect(allClose(gpu, ref, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        }

        // 2. Expert routing (top-8 of 128)
        let logits = MLXRandom.normal([2, 5, 128]).asType(.float32)
        let bias = MLXRandom.normal([128]).asType(.float32) * Float(0.1)
        let (gpuInds, gpuWeights) = inkling2ExpertRouting(
            logits: logits, bias: bias, topK: 8, routeScale: 8.0, useKernel: true)
        let (refInds, refWeights) = inkling2ExpertRoutingReference(
            logits: logits, bias: bias, topK: 8, routeScale: 8.0)
        // Sort both along last axis since argPartition order within top-k is unspecified
        let gpuOrder = argSort(gpuInds, axis: -1)
        let refOrder = argSort(refInds.asType(.uint32), axis: -1)
        let sortedGpuInds = takeAlong(gpuInds, gpuOrder, axis: -1)
        let sortedRefInds = takeAlong(refInds.asType(.uint32), refOrder, axis: -1)
        let sortedGpuW = takeAlong(gpuWeights, gpuOrder, axis: -1)
        let sortedRefW = takeAlong(refWeights, refOrder, axis: -1)
        eval(sortedGpuInds, sortedRefInds, sortedGpuW, sortedRefW)
        #expect(arrayEqual(sortedGpuInds, sortedRefInds).item(Bool.self))
        #expect(allClose(sortedGpuW, sortedRefW, rtol: 1e-4, atol: 1e-4).item(Bool.self))

        // 3. Causal relative flash attention with sinks (sliding and global)
        for window in [0, 8] {
            let B = 2, Hq = 4, Hkv = 2, L = 12, S = 16, D = 16, R = 8
            let q = MLXRandom.normal([B, Hq, L, D]).asType(.bfloat16)
            let k = MLXRandom.normal([B, Hkv, S, D]).asType(.bfloat16)
            let v = MLXRandom.normal([B, Hkv, S, D]).asType(.bfloat16)
            let rel = (MLXRandom.normal([B, L, Hq, R]) * Float(0.2)).asType(.bfloat16)
            let sinks = (MLXRandom.normal([B, L, Hq]) * Float(0.5)).asType(.bfloat16)
            let scale: Float = 1.0 / sqrt(Float(D))
            let keyStart = MLXArray([Int32(0), Int32(2)])
            let gpuAttn = inkling2RelativeAttention(
                q: q, k: k, v: v, rel: rel, sinks: sinks,
                scale: scale, window: window, keyStart: keyStart
            )
            let qPos = MLXArray((S - L) ..< S)[.newAxis, 0...] - keyStart[0..., .newAxis]
            let kPos = MLXArray(0 ..< S)[.newAxis, 0...] - keyStart[0..., .newAxis]
            let refAttn = inkling2AttentionReference(
                query: q, key: k, value: v, positionBias: rel, sinks: sinks,
                scale: scale, window: window, queryPositions: qPos, keyPositions: kPos
            )
            eval(gpuAttn, refAttn)
            #expect(
                allClose(
                    gpuAttn.asType(.float32),
                    refAttn.asType(.float32),
                    rtol: 2e-2,
                    atol: 2e-2
                ).item(Bool.self))
        }
    }

    // MARK: - NVFP4 Post-gatherQuantizedMM global_scale & FP16 Overflow Guard

    @Test("QuantizedSwitchLinear applies NVFP4 global_scale accurately in BF16 and FP16")
    func testNVFP4GlobalScaleAndFP16Guard() {
        MLXRandom.seed(19)
        // Use N = 64 >= 8 and large weights (amax ~ 120) so FP16 would overflow without BF16 guard.
        let dense = SwitchLinear(inputDims: 64, outputDims: 64, numExperts: 4, bias: false)
        let scaledWeights = dense.weight.asType(.float32) * Float(400.0)
        dense.update(parameters: ModuleParameters.unflattened(["weight": scaledWeights]))

        let nvfp4 = QuantizedSwitchLinear(
            dense, groupSize: 16, bits: 4, mode: .nvfp4, useGlobalScale: true)
        #expect(nvfp4.globalScale != nil)

        let indices = MLXArray([UInt32(0), UInt32(3), UInt32(1), UInt32(2)]).reshaped(2, 1, 2)
        let x32 = MLXRandom.normal([2, 1, 1, 64]).asType(.float32) * Float(4.0)
        let expected = dense(x32, indices)

        for dtype in [DType.bfloat16, DType.float16] {
            let actual = nvfp4(x32.asType(dtype), indices)
            eval(expected, actual)
            #expect(actual.dtype == dtype)
            let actual32 = actual.asType(.float32)
            #expect(all(isFinite(actual32)).item(Bool.self))
            let relErr =
                abs(actual32 - expected).mean().item(Float.self)
                / max(abs(expected).mean().item(Float.self), 1e-5)
            #expect(relErr < 0.12)
        }
    }

    // MARK: - 3-Pass Loop, RotatingKVCache Wrap-Around & Chunked Prefill Parity

    @Test("Inkling2 3-pass looped forward matches chunked prefill + RotatingKVCache wrap decode")
    func testLoopedForwardCacheWrapAndPrefillSlice() throws {
        let cfg = Self.makeBaseConfig(looped: true)
        let model = Inkling2Model(cfg)
        Self.seedWeights(model)

        // 3 prelude + 3 * 3 core + 2 coda = 14 visits
        let params = GenerateParameters(maxKVSize: 4096)
        let cache = model.newCache(parameters: params)
        #expect(cache.count == 14)

        // Verify HIGH-4: sliding visits stay pinned to slidingWindowSize (8), ignoring maxKVSize (4096).
        let firstList = try #require(cache[0] as? CacheList)
        #expect(firstList.children.count == 2)
        #expect(firstList.children[0] is RotatingKVCache)
        #expect(firstList.children[0].maxSize == 8)
        let fullVisitList = try #require(cache[9] as? CacheList)  // layer 5 in pass 2
        #expect(fullVisitList.children[0] is KVCacheSimple)

        // Sequence length 20 > slidingWindowSize (8) exercises RotatingKVCache ring wrap-around!
        // Test B = 1 (exercises single-row fused SDPA + `(k - rotIdx + S) % S` path) and B = 2.
        for batchSize in [1, 2] {
            MLXRandom.seed(99 + UInt64(batchSize))
            let tokens = MLXRandom.randInt(low: 0, high: 120, [batchSize, 20])
            let fullLogits = model(tokens)
            eval(fullLogits)
            #expect(fullLogits.shape == [batchSize, 20, 120])
            #expect(fullLogits.dtype == .float32)

            let runCache = model.newCache(parameters: nil)
            var chunks: [MLXArray] = [
                model(tokens[0..., ..<6], cache: runCache),
                model(tokens[0..., 6 ..< 11], cache: runCache),
            ]
            for pos in 11 ..< 20 {
                chunks.append(model(tokens[0..., pos ..< (pos + 1)], cache: runCache))
            }
            let steppedLogits = concatenated(chunks, axis: 1)
            eval(steppedLogits)
            #expect(runCache[0].offset == 20)
            #expect(KVCacheResidency.residentBytes(in: runCache) != nil)
            #expect(allClose(steppedLogits, fullLogits, rtol: 1e-3, atol: 1e-3).item(Bool.self))

            if batchSize == 1 {
                // Verify CRITICAL-4: `prepare` chunks without prefix lmHead and returns [1, 1, V]
                let prepCache = model.newCache(parameters: nil)
                let prepResult = try model.prepare(
                    LMInput(tokens: tokens), cache: prepCache, windowSize: 5)
                guard case .logits(let out) = prepResult else {
                    Issue.record("Expected .logits from Inkling2Model.prepare")
                    return
                }
                eval(out.logits)
                #expect(out.logits.shape == [1, 1, 120])
                #expect(
                    allClose(
                        out.logits,
                        fullLogits[0..., 19..., 0...],
                        rtol: 1e-3,
                        atol: 1e-3
                    ).item(Bool.self))

                // Verify `callAsFunction(_ input: LMInput.Text, ...)` slices to last token on L > 1
                let lmOut = model(LMInput.Text(tokens: tokens), cache: nil, state: nil)
                eval(lmOut.logits)
                #expect(lmOut.logits.shape == [1, 1, 120])
            }
        }

        // Verify pipeline-parallel guard throws cleanly on looped models
        #expect(throws: PipelineParallelError.self) {
            try model.pipeline(group: DistributedGroup())
        }
    }

    // MARK: - Checkpoint Sanitization & mtp.safetensors Skip

    @Test("Inkling2Checkpoint sanitizes raw NVFP4+MXFP8+MTP and 4-bit MLX checkpoints")
    func testCheckpointSanitizeAndMTPSkip() throws {
        let cfg = Self.makeBaseConfig(looped: true)
        let reference = Inkling2(cfg)
        Self.seedWeights(reference)

        // Quantize layers 0..6 experts as NVFP4 with global_scale and layer 7 as MXFP8
        var expertUpdates: [(String, Module)] = []
        for (path, mod) in reference.namedModules() {
            guard let sw = mod as? SwitchLinear else { continue }
            if path.contains(".layers.7.") {
                expertUpdates.append(
                    (path, QuantizedSwitchLinear(sw, groupSize: 32, bits: 8, mode: .mxfp8)))
            } else {
                expertUpdates.append(
                    (
                        path,
                        QuantizedSwitchLinear(
                            sw, groupSize: 16, bits: 4, mode: .nvfp4, useGlobalScale: true)
                    ))
            }
        }
        reference.update(modules: ModuleChildren.unflattened(expertUpdates))
        eval(reference)

        let expectedParams = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())

        // Convert to raw Thinking Machines (`model.llm.*` / `model.visual.*`) checkpoint + MTP keys
        var rawCkpt: [String: MLXArray] = [
            "fuse_weight": MLXArray.ones([4, 4]),
            "mask_embed": MLXArray.zeros([64]),
            "norm.weight": MLXArray.ones([64]),
            "layers.0.self_attn.q_proj.weight": MLXArray.zeros([64, 64]),
            "markov_head.hidden_out.weight": MLXArray.zeros([16, 64]),
        ]
        let denom = MLXArray([Float(7.5), Float(7.5), Float(7.5), Float(7.5)])

        let inverseRenames: [(String, String)] = [
            (#"^lm_head\."#, "model.llm.unembed."),
            (#"^model\.language_model\.embed_tokens\."#, "model.llm.embed."),
            (#"^model\.language_model\."#, "model.llm."),
            (#"\.final_input\.query$"#, ".attn_res.output_query"),
            (#"\.layers\.(\d+)\.mlp_input\.query$"#, ".attn_res.mlp_queries.$1"),
            (#"\.self_attn\.q_proj\."#, ".attn.wq_du."),
            (#"\.self_attn\.k_proj\."#, ".attn.wk_dv."),
            (#"\.self_attn\.v_proj\."#, ".attn.wv_dv."),
            (#"\.self_attn\.r_proj\."#, ".attn.wr_du."),
            (#"\.self_attn\.o_proj\."#, ".attn.wo_ud."),
            (#"\.self_attn\.gate_proj\."#, ".attn.gate_attn."),
            (#"\.self_attn\.sink_proj\."#, ".attn.attn_sink_proj."),
            (#"\.self_attn\."#, ".attn."),
            (#"\.q_message_gate\."#, ".controllers.attn."),
            (#"\.(k|v|mlp)_message_gate\."#, ".controllers.$1."),
            (#"(\.controllers\.\w+)\.down_proj\.weight$"#, "$1.down_weight"),
            (#"(\.controllers\.\w+)\.forget_gate_proj\.weight$"#, "$1.forget_weight"),
            (#"(\.controllers\.\w+)\.message_gate_proj\.weight$"#, "$1.gain_weight"),
            (#"\.post_attention_convolution\."#, ".attn_sconv."),
            (#"\.post_mlp_convolution\."#, ".mlp_sconv."),
            (#"\.conv1d\.weight$"#, ".weight"),
            (#"sconv\.static_weight$"#, "sconv.static_w"),
            (#"sconv\.weight_proj\.weight$"#, "sconv.weight_proj"),
            (#"\.post_attention_layernorm\."#, ".mlp_norm."),
            (#"\.input_layernorm\."#, ".attn_norm."),
            (#"\.(q|k|v)_input_layernorm\."#, ".$1_input_norm."),
            (#"\.mlp\.gate\.e_score_correction_bias$"#, ".mlp.gate.bias"),
            (#"^model\.vision_tower\.pre_layernorm\."#, "model.visual.backbone.ln_pre."),
            (#"^model\.vision_tower\.merger\.linear_1\."#, "model.visual.proj_in."),
            (#"^model\.vision_tower\.merger\.linear_2\."#, "model.visual.proj_out."),
            (#"^model\.vision_tower\."#, "model.visual.backbone."),
            (#"position_embedding\.weight$"#, "position_embedding"),
            (#"\.layernorm_before\."#, ".ln_1."),
            (#"\.layernorm_after\."#, ".ln_2."),
            (#"\.mlp\.fc1\."#, ".mlp.c_fc."),
            (#"\.mlp\.fc2\."#, ".mlp.c_proj."),
            (#"\.lambda_(\d)$"#, ".ls_$1.gamma"),
        ]

        let expertsMatch = try NSRegularExpression(
            pattern: #"^(.+\.mlp\.experts)\.(gate_up|down)_proj\.(\w+)$"#)
        let attnQueryMatch = try NSRegularExpression(
            pattern: #"^.+\.layers\.(\d+)\.attention_input\.query$"#)

        for (k, v) in expectedParams {
            if k.hasSuffix(".message_input.query") {
                let q = concatenated([v[0..., ..<1, 0...], v[0..., 3..., 0...]], axis: 1)
                let kv = v[0..., 1 ..< 3, 0...]
                rawCkpt["model.llm.middle_cycle_cross_pass_queries"] = flatten(
                    q, startAxis: 0, endAxis: 1)
                rawCkpt["model.llm.middle_cycle_cross_pass_kv_queries"] = flatten(
                    kv, startAxis: 0, endAxis: 1)
                continue
            }
            let ns = NSRange(k.startIndex ..< k.endIndex, in: k)
            if let m = expertsMatch.firstMatch(in: k, range: ns),
                let pRange = Range(m.range(at: 1), in: k),
                let nRange = Range(m.range(at: 2), in: k),
                let paramRange = Range(m.range(at: 3), in: k)
            {
                let prefix = String(k[pRange]).replacingOccurrences(
                    of: "model.language_model.", with: "model.llm.")
                let which = k[nRange] == "gate_up" ? "w13" : "w2"
                let base = "\(prefix).\(which)_weight"
                switch String(k[paramRange]) {
                case "weight":
                    rawCkpt[base] = v.view(dtype: .uint8)
                case "scales":
                    rawCkpt["\(base)_scale"] = v
                case "global_scale":
                    rawCkpt["\(base)_amax"] = v * denom / Float(2688.0)
                    rawCkpt["\(base)_outer_scale_denominator"] = denom
                default:
                    break
                }
                continue
            }
            if let m = attnQueryMatch.firstMatch(in: k, range: ns),
                let iRange = Range(m.range(at: 1), in: k),
                let layerIdx = Int(k[iRange])
            {
                rawCkpt["model.llm.attn_res.attn_queries.\(layerIdx - 1)"] = v
                continue
            }

            var rawKey = k
            for (pat, rep) in inverseRenames {
                let regex = try NSRegularExpression(pattern: pat)
                let r = NSRange(rawKey.startIndex ..< rawKey.endIndex, in: rawKey)
                rawKey = regex.stringByReplacingMatches(in: rawKey, range: r, withTemplate: rep)
            }
            var rawVal = v
            if rawKey.hasSuffix("sconv.weight") {
                rawVal = rawVal.swappedAxes(1, 2)
            }
            rawCkpt[rawKey] = rawVal
        }

        // Write `model.safetensors` AND `mtp.safetensors` to temp dir and verify `loadWeights`
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2_ckpt_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        try MLX.save(arrays: rawCkpt, url: tempDir.appendingPathComponent("model.safetensors"))
        try MLX.save(
            arrays: ["fuse_weight": MLXArray.ones([4, 4])],
            url: tempDir.appendingPathComponent("mtp.safetensors")
        )

        let loaded = Inkling2(cfg)
        try loadWeights(modelDirectory: tempDir, model: loaded)

        let testTokens = MLXArray([Int32(1), Int32(5), Int32(9)]).reshaped(1, 3)
        let refOut = reference(testTokens)
        let loadedOut = loaded(testTokens)
        eval(refOut, loadedOut)
        #expect(allClose(refOut, loadedOut, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    // MARK: - Vision Processor & End-to-End Multimodal Prepare

    @Test("Inkling2Processor handles -1 padding, unpremultiplied checkerboard alpha, maxPixels & vision prepare")
    func testVisionProcessorAndMultimodalPrepare() async throws {
        let procConfig = Inkling2ProcessorConfiguration(
            patchHeight: 4,
            patchWidth: 4,
            imageMean: [0.5, 0.5, 0.5],
            imageStd: [0.5, 0.5, 0.5],
            rescaleFactor: 1.0 / 255.0,
            imageToken: "<|unused_200054|>"
        )

        struct MockInklingTokenizer: Tokenizer {
            func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2, 3] }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
            func convertTokenToId(_ token: String) -> Int? {
                token == "<|unused_200054|>" ? 119 : nil
            }
            func convertIdToToken(_ id: Int) -> String? {
                id == 119 ? "<|unused_200054|>" : "tok\(id)"
            }
            var bosToken: String? { nil }
            var eosToken: String? { nil }
            var unknownToken: String? { nil }
            func applyChatTemplate(
                messages: [[String: any Sendable]],
                tools: [[String: any Sendable]]?,
                additionalContext: [String: any Sendable]?
            ) throws -> [Int] {
                // One placeholder per image part
                var out = [10]
                for msg in messages {
                    if let parts = msg["content"] as? [[String: any Sendable]] {
                        for part in parts where (part["type"] as? String) == "image" {
                            out.append(119)
                        }
                    }
                }
                out.append(11)
                return out
            }
        }

        let processor = Inkling2Processor(procConfig, tokenizer: MockInklingTokenizer())

        // Create a 3x5 premultiplied RGBA8 bitmap:
        // - (0, 0): fully transparent (alpha = 0) -> checkerboard tile (0, 0) = 204
        // - (0, 1): opaque white (255, 255, 255, 255) -> +1.0
        // - padded right/bottom pixels (x >= 5 or y >= 3) -> (-1/255 - 0.5) / 0.5 = -1.0078431
        let w = 5, h = 3
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        // (0, 1) opaque white
        bytes[4] = 255
        bytes[5] = 255
        bytes[6] = 255
        bytes[7] = 255
        let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
        let ciImage = CIImage(
            bitmapData: Data(bytes),
            bytesPerRow: w * 4,
            size: CGSize(width: w, height: h),
            format: .RGBA8,
            colorSpace: sRGB
        )

        let (tiles, thw) = try processor.preprocessImage(ciImage)
        eval(tiles)
        #expect(thw.values == (1, 1, 2))
        #expect(tiles.shape == [2, 3, 4, 4])

        let tileFloats = tiles.asArray(Float.self)
        // Pixel (0, 0) in tile 0 -> checkerboard 204 -> (204/255 - 0.5) / 0.5 = 0.6
        #expect(abs(tileFloats[0] - 0.6) < 1e-4)
        // Pixel (0, 1) in tile 0 -> opaque 255 -> 1.0
        #expect(abs(tileFloats[1] - 1.0) < 1e-4)
        // Row y = 3 (padded row) -> (-1/255 - 0.5) / 0.5 = -1.0078431
        let expectedPad: Float = (-1.0 / 255.0 - 0.5) / 0.5
        #expect(abs(tileFloats[3 * 4 + 0] - expectedPad) < 1e-5)

        // Verify maxPixels cap scales oversized image proportionally
        let bigImage = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 40, height: 40))
        let (_, cappedTHW) = try processor.preprocessImage(
            bigImage, processing: .init(maxPixels: 64))
        #expect(cappedTHW.h <= 2 && cappedTHW.w <= 2)

        // Verify prepare(input:) keeps mask == nil, expands 119 -> 2 tokens, and runs VLM prepare
        let userInput = UserInput(prompt: "Describe", images: [.ciImage(ciImage)])
        let lmInput = try await processor.prepare(input: userInput)
        #expect(lmInput.text.mask == nil)
        #expect(lmInput.text.tokens.shape == [1, 4])  // [10, 119, 119, 11]

        let vlm = Inkling2(Self.makeBaseConfig(looped: true))
        Self.seedWeights(vlm)
        let vlmCache = vlm.newCache(parameters: nil)
        let prepOut = try vlm.prepare(lmInput, cache: vlmCache, windowSize: 2)
        guard case .logits(let result) = prepOut else {
            Issue.record("Expected .logits from Inkling2.prepare")
            return
        }
        eval(result.logits)
        #expect(result.logits.shape == [1, 1, 120])
    }

    // MARK: - ChatTemplateSanitizer (CRITICAL-1)

    @Test("ChatTemplateSanitizer rewrites balanced tojson kwargs with brace-safe macro")
    func testChatTemplateSanitizer() throws {
        let rawTemplate = #"""
{{- tool_state.specs | tojson(sort_keys=true, separators=(",", ":")) -}}
{{- '{"name":' ~ (fn.name | tojson(sort_keys=true, separators=(",", ":"))) ~ ',"args":' -}}
{{- (key | tojson) ~ ":" ~ (value | tojson(separators=(",", ":"))) -}}
"""#
        let sanitized = try #require(ChatTemplateSanitizer.sanitize(rawTemplate))
        #expect(sanitized.contains(#"{{- "{" -}}"#))
        #expect(sanitized.contains(#"{{- "}" -}}"#))
        #expect(sanitized.contains(#"{{- "[" -}}"#))
        #expect(sanitized.contains(#"{{- "]" -}}"#))
        #expect(sanitized.contains("_mlx_compact_json(tool_state.specs, true)"))
        #expect(sanitized.contains("(_mlx_compact_json(fn.name, true))"))
        #expect(sanitized.contains("(_mlx_compact_json(value, false))"))
        #expect(sanitized.contains("(key | tojson)"))

        // Plain templates without sort_keys / separators return nil
        #expect(ChatTemplateSanitizer.sanitize("{{ x | tojson }}") == nil)
        #expect(ChatTemplateSanitizer.sanitize("{{ x | tojson(indent=2) }}") == nil)
    }

    // MARK: - .inkling2 ToolCallFormat, Parser & Streaming Envelope Hold

    @Test("Inkling2ToolCallParser & ToolCallProcessor suppress <fn_name> leak and parse multi-tool turns")
    func testInkling2ToolCallStreamingAndMultiParse() throws {
        #expect(ToolCallFormat.infer(from: "inkling2_loop") == .inkling2)
        #expect(ToolCallFormat.infer(from: "inkling2") == .inkling2)

        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "parameters": ["type": "object"],
                ] as [String: any Sendable],
            ],
            [
                "type": "function",
                "function": [
                    "name": "get_time",
                    "parameters": ["type": "object"],
                ] as [String: any Sendable],
            ],
        ]

        // 1. Character-by-character streaming of thinking + 2 parallel tool calls
        let streamText =
            "<|message_model|><|content_thinking|>Need weather & time.<|end_message|>"
            + "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Tokyo\"}}<|end_message|>"
            + "<|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":{\"tz\":\"JST\"}}<|end_message|>"

        let processor = ToolCallProcessor(format: .inkling2, tools: tools)
        var visible = ""
        for ch in streamText {
            if let emitted = processor.processChunk(String(ch)) {
                visible += emitted
            }
        }
        if let tail = processor.processEOS(returnBufferedText: true) {
            visible += tail
        }

        #expect(visible == "<|message_model|><|content_thinking|>Need weather & time.<|end_message|>")
        #expect(!visible.contains("get_weather"))
        #expect(!visible.contains("get_time"))
        #expect(processor.toolCalls.count == 2)
        #expect(processor.toolCalls[0].function.name == "get_weather")
        #expect(processor.toolCalls[0].function.arguments["city"] == .string("Tokyo"))
        #expect(processor.toolCalls[1].function.name == "get_time")
        #expect(processor.toolCalls[1].function.arguments["tz"] == .string("JST"))

        // 2. Direct `Inkling2ToolCallParser.parseAll` contract
        let parser = Inkling2ToolCallParser()
        let allCalls = parser.parseAll(content: streamText, tools: tools)
        #expect(allCalls.count == 2)
        #expect(allCalls[0].function.name == "get_weather")
        #expect(allCalls[1].function.name == "get_time")
    }
}
