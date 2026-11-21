// Copyright © 2026 Apple Inc.

import Foundation
import MLX

package func inkling2UseKernels() -> Bool {
    #if canImport(Metal)
        return Device.defaultDevice().deviceType == .gpu
    #else
        return false
    #endif
}

// MARK: - Depth Attention Metal Kernel

private let depthAttentionSource = """
    constexpr int N = K + 1;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    float acc[N * Q];
    for (int i = 0; i < N * Q; i++) {
        acc[i] = 0.0f;
    }
    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float qv = queries[j * D + d];
            for (int i = 0; i < N; i++) {
                acc[i * Q + j] += x[i] * qv;
            }
        }
    }

    threadgroup float buf[N * Q][TG / 32];
    for (int i = 0; i < N * Q; i++) {
        float v = simd_sum(acc[i]);
        if (lane == 0) {
            buf[i][sg] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float w[N * Q];
    for (int i = 0; i < N * Q; i++) {
        float t = 0.0f;
        for (int s = 0; s < TG / 32; s++) {
            t += buf[i][s];
        }
        w[i] = t * scale;
    }
    for (int j = 0; j < Q; j++) {
        float m = -INFINITY;
        for (int i = 0; i < N; i++) {
            m = max(m, w[i * Q + j]);
        }
        float z = 0.0f;
        for (int i = 0; i < N; i++) {
            w[i * Q + j] = metal::exp(w[i * Q + j] - m);
            z += w[i * Q + j];
        }
        for (int i = 0; i < N; i++) {
            w[i * Q + j] /= z;
        }
    }

    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float o = 0.0f;
            for (int i = 0; i < N; i++) {
                o += w[i * Q + j] * x[i];
            }
            out[(size_t(row) * Q + j) * D + d] = o;
        }
    }
"""

private let depthAttentionKernel = MLXFast.metalKernel(
    name: "inkling2_depth_attention",
    inputNames: ["blocks", "partial", "queries", "scale"],
    outputNames: ["out"],
    source: depthAttentionSource
)

package func inkling2DepthAttentionReference(
    blocks: MLXArray,
    partial: MLXArray,
    queries: MLXArray
) -> MLXArray {
    let b = blocks.asType(.float32)
    let p = partial.asType(.float32)
    let q = queries.asType(.float32)
    let scale = Float(1.0) / sqrt(Float(p.dim(-1)))
    let stacked = movedAxis(concatenated([b, p[.newAxis]], axis: 0), source: 0, destination: -2)
    let weights = softmax(matmul(stacked, q.T) * scale, axis: -2, precise: true)
    return matmul(weights.swappedAxes(-1, -2), stacked)
}

package func inkling2DepthAttention(
    blocks: MLXArray,
    partial: MLXArray,
    queries: MLXArray,
    useKernel: Bool = inkling2UseKernels()
) -> MLXArray {
    guard useKernel else {
        return inkling2DepthAttentionReference(blocks: blocks, partial: partial, queries: queries)
    }
    let b = blocks.asType(.float32)
    let p = partial.asType(.float32)
    let q = queries.asType(.float32)
    let D = p.dim(-1)
    let Q = q.dim(0)
    let scale = Float(1.0) / sqrt(Float(D))
    let tg = min(1024, max(32, (D / 8 / 32) * 32))
    let outShape = Array(p.shape.dropLast()) + [Q, D]
    return depthAttentionKernel(
        [b, p, q, MLXArray(scale)],
        template: [
            ("K", b.dim(0)),
            ("D", D),
            ("Q", Q),
            ("TG", tg),
        ],
        grid: ((p.size / D) * tg, 1, 1),
        threadGroup: (tg, 1, 1),
        outputShapes: [outShape],
        outputDTypes: [.float32]
    )[0]
}

// MARK: - Causal Relative Flash Attention Metal Kernel

private let relativeAttentionBQ = 32
private let relativeAttentionTG = 128

private let relativeAttentionSource = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < S) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""

private let relativeAttentionKernel = MLXFast.metalKernel(
    name: "inkling2_relative_attention",
    inputNames: ["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
    outputNames: ["out"],
    source: relativeAttentionSource,
    ensureRowContiguous: false
)

package func inkling2RelativeAttention(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    rel: MLXArray,
    sinks: MLXArray,
    scale: Float,
    window: Int,
    keyStart: MLXArray
) -> MLXArray {
    let B = q.dim(0)
    let Hq = q.dim(1)
    let L = q.dim(2)
    let D = q.dim(3)
    let Hkv = k.dim(1)
    let S = k.dim(2)
    let R = rel.dim(-1)
    let sizes = MLXArray([Int32(L), Int32(S)])
    return relativeAttentionKernel(
        [
            q,
            k,
            v,
            contiguous(rel),
            contiguous(sinks.asType(.float32)),
            contiguous(keyStart.asType(.int32)),
            sizes,
            MLXArray(scale),
        ],
        template: [
            ("T", q.dtype),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("R", R),
            ("WINDOW", window),
            ("BQ", relativeAttentionBQ),
            ("TG", relativeAttentionTG),
        ],
        grid: (((L + relativeAttentionBQ - 1) / relativeAttentionBQ) * relativeAttentionTG, Hq, B),
        threadGroup: (relativeAttentionTG, 1, 1),
        outputShapes: [[B, L, Hq, D]],
        outputDTypes: [q.dtype]
    )[0]
}

// MARK: - Top-K Expert Routing Metal Kernel

private let expertRoutingSource = """
    constexpr int R = (E + 31) / 32;
    uint token = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    auto row = logits + size_t(token) * E;

    float score[R];
    float biased[R];
    for (int i = 0; i < R; i++) {
        int e = lane + 32 * i;
        score[i] = 0.0f;
        biased[i] = -INFINITY;
        if (e < E) {
            float x = row[e] * multiplier;
            float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
            score[i] = x < 0 ? y : 1.0f - y;
            biased[i] = score[i] + bias[e];
        }
    }

    float w[K];
    for (int k = 0; k < K; k++) {
        float best = -INFINITY;
        int best_e = E;
        for (int i = 0; i < R; i++) {
            int e = lane + 32 * i;
            if (biased[i] > best || (biased[i] == best && e < best_e)) {
                best = biased[i];
                best_e = e;
            }
        }
        float top = simd_max(best);
        int winner = simd_min(best == top ? best_e : E);
        float s = 0.0f;
        for (int i = 0; i < R; i++) {
            if (lane + 32 * i == winner) {
                s = score[i];
                biased[i] = -INFINITY;
            }
        }
        w[k] = simd_sum(s);
        if (lane == 0) {
            inds[size_t(token) * K + k] = winner;
        }
    }

    float total = 0.0f;
    for (int k = 0; k < K; k++) {
        total = w[k] + total;
    }
    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            weights[size_t(token) * K + k] = w[k] / (total + eps) * scale;
        }
    }
"""

private let expertRoutingKernel = MLXFast.metalKernel(
    name: "inkling2_expert_routing",
    inputNames: ["logits", "bias", "multiplier", "eps", "scale"],
    outputNames: ["inds", "weights"],
    source: expertRoutingSource
)

package func inkling2ExpertRoutingReference(
    logits: MLXArray,
    bias: MLXArray,
    topK: Int,
    routeScale: Float,
    scoreEps: Float = 1e-20
) -> (MLXArray, MLXArray) {
    let scores = sigmoid(logits.asType(.float32))
    let biased = scores + bias.asType(.float32)
    let indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    let weights = takeAlong(scores, indices, axis: -1)
    let normalized = (weights / (weights.sum(axis: -1, keepDims: true) + scoreEps)) * routeScale
    return (indices, normalized)
}

package func inkling2ExpertRouting(
    logits: MLXArray,
    bias: MLXArray,
    topK: Int,
    routeScale: Float,
    scoreEps: Float = 1e-20,
    useKernel: Bool = inkling2UseKernels()
) -> (MLXArray, MLXArray) {
    guard useKernel else {
        return inkling2ExpertRoutingReference(
            logits: logits, bias: bias, topK: topK, routeScale: routeScale, scoreEps: scoreEps)
    }
    let E = logits.dim(-1)
    let outShape = Array(logits.shape.dropLast()) + [topK]
    let outputs = expertRoutingKernel(
        [
            logits.asType(.float32),
            bias.asType(.float32),
            MLXArray(Float(1.0)),
            MLXArray(scoreEps),
            MLXArray(routeScale),
        ],
        template: [
            ("E", E),
            ("K", topK),
        ],
        grid: ((logits.size / E) * 32, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [outShape, outShape],
        outputDTypes: [.uint32, .float32]
    )
    return (outputs[0], outputs[1])
}

// MARK: - Multimodal Splice & Distributed Gather Helpers

/// Replaces embeddings at positions where `inputs == tokenId` with consecutive rows of `rows`
/// without any host synchronization.
package func inkling2Splice(
    embeddings: MLXArray,
    inputs: MLXArray,
    rows: MLXArray,
    tokenId: Int
) -> MLXArray {
    guard rows.dim(0) > 0 else { return embeddings }
    let mask = inputs .== Int32(tokenId)
    let flatOrder = cumsum(mask.flattened().asType(.int32), axis: 0) - Int32(1)
    let clamped = clip(flatOrder, min: 0, max: max(0, rows.dim(0) - 1))
    let gathered = rows.asType(embeddings.dtype)[clamped].reshaped(embeddings.shape)
    return which(mask[.ellipsis, .newAxis], gathered, embeddings)
}

/// Gathers sharded last-axis slices across `group` into `[..., group.size * shardWidth]`.
package func inkling2GatherLastAxis(_ x: MLXArray, group: DistributedGroup) -> MLXArray {
    let n = group.size
    guard n > 1 else { return x }
    let all = group.allGather(x)
    let reshaped = all.reshaped([n] + x.shape)
    let moved = movedAxis(reshaped, source: 0, destination: -2)
    return flatten(moved, startAxis: -2, endAxis: -1)
}
