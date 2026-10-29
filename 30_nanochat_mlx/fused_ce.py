"""
Fused logit-softcap + cross-entropy (efficiency candidate #1 from efficiency/codex_audit.md).

capped_ce(z, y) == fast.cross_entropy(15*tanh(z.astype(f32)/15), y) but reads the raw
(compute-dtype, e.g. bf16) logits directly: one Metal kernel for the forward (per-row
logsumexp, fp32 math) and one for the backward (writes d/dz in z's dtype). The fp32 capped
logits tensor (1 GiB at 32x1024x8192) is never materialised.

Only used when GPT.ce_mode == "kernel" (default "ref").
"""
import mlx.core as mx

CAP = 15.0
_TG = 256  # threads per row

_fwd = None
_bwd = None


def _kernels():
    global _fwd, _bwd
    if _fwd is None:
        _fwd = mx.fast.metal_kernel(
            name="capped_ce_fwd",
            input_names=["z", "y"],
            output_names=["loss"],
            source=r"""
                uint r = threadgroup_position_in_grid.x;
                uint t = thread_position_in_threadgroup.x;
                threadgroup float partial[8];
                const device auto* zr = z + (size_t)r * V;
                float s = 0.0f;
                for (uint j = t; j < V; j += 256) {
                    float u = 15.0f * metal::precise::tanh(float(zr[j]) / 15.0f);
                    s += metal::precise::exp(u - 15.0f);
                }
                s = metal::simd_sum(s);
                if ((t & 31) == 0) partial[t >> 5] = s;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (t < 32) {
                    float s2 = metal::simd_sum(t < 8 ? partial[t] : 0.0f);
                    if (t == 0) {
                        float target = 15.0f * metal::precise::tanh(float(zr[y[r]]) / 15.0f);
                        loss[r] = metal::precise::log(s2) + (15.0f - target);
                    }
                }
            """,
        )
        _bwd = mx.fast.metal_kernel(
            name="capped_ce_bwd",
            input_names=["z", "y", "loss", "cot"],
            output_names=["dz"],
            source=r"""
                uint r = threadgroup_position_in_grid.x;
                uint t = thread_position_in_threadgroup.x;
                const device auto* zr = z + (size_t)r * V;
                uint yr = uint(y[r]);
                float target = 15.0f * metal::precise::tanh(float(zr[yr]) / 15.0f);
                float lse = loss[r] + target;
                float c = cot[r];
                for (uint j = t; j < V; j += 256) {
                    float a = metal::precise::tanh(float(zr[j]) / 15.0f);
                    float p = metal::precise::exp(15.0f * a - lse);
                    dz[(size_t)r * V + j] = static_cast<OT>(c * (p - (j == yr ? 1.0f : 0.0f)) * (1.0f - a * a));
                }
            """,
        )
    return _fwd, _bwd


def _ref(z, y):
    return mx.fast.cross_entropy(CAP * mx.tanh(z.astype(mx.float32) / CAP), y)


@mx.custom_function
def _capped_ce_gpu(z, y):
    fwd, _ = _kernels()
    return fwd(inputs=[z, y], template=[("V", z.shape[-1])], grid=(y.size * _TG, 1, 1),
               threadgroup=(_TG, 1, 1), output_shapes=[y.shape], output_dtypes=[mx.float32])[0]


@_capped_ce_gpu.vjp
def _capped_ce_vjp(primals, cotangent, output):
    z, y = primals
    _, bwd = _kernels()
    cot = cotangent.astype(mx.float32)
    dz = bwd(inputs=[z, y, output, cot], template=[("V", z.shape[-1]), ("OT", z.dtype)], grid=(y.size * _TG, 1, 1),
             threadgroup=(_TG, 1, 1), output_shapes=[z.shape], output_dtypes=[z.dtype])[0]
    return dz, mx.zeros_like(y)


def capped_ce(z, y):
    """Per-token CE (fp32, shape of y) of softcapped logits 15*tanh(z/15)."""
    if mx.default_device() != mx.gpu:
        return _ref(z, y)
    return _capped_ce_gpu(z, y.astype(mx.int32))
