"""
int8 quantization for the shipped vector index.

Each vector gets its own float32 scale, so the dot product of two quantized
vectors reconstructs as
    dot(a, b) ~= (a_i8 . b_i8) * scale_a * scale_b
Storing a per-vector scale rather than one global scale costs 4 bytes per row
and keeps short vectors from losing their precision to long ones.
"""
import numpy as np

I8_MAX = 127.0


def quantize(vectors: np.ndarray):
    """(n, d) float32 -> (int8 codes, float32 scales)."""
    v = np.asarray(vectors, dtype=np.float32)
    peak = np.max(np.abs(v), axis=1)
    peak = np.maximum(peak, 1e-12)
    scale = (peak / I8_MAX).astype(np.float32)
    codes = np.rint(v / scale[:, None]).clip(-127, 127).astype(np.int8)
    return codes, scale


def dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * scale[:, None]
