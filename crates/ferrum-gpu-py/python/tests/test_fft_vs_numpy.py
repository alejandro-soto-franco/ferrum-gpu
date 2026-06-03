"""GPU FFT vs numpy.fft.fft on 8 cases x 2 device modes."""

import numpy as np
import pytest

import ferrum_gpu as fgpu


@pytest.fixture(scope="module")
def device():
    return fgpu.cuda.Device(0)


def _input_signal(n: int, batch: int) -> np.ndarray:
    out = np.empty(batch * n, dtype=np.complex64)
    for b in range(batch):
        for i in range(n):
            out[b * n + i] = complex(np.sin(i), np.cos(i))
    return out


def _reference(arr: np.ndarray, n: int, batch: int, direction: str, normalize: bool) -> np.ndarray:
    out = np.empty_like(arr)
    for b in range(batch):
        lane = arr[b * n:(b + 1) * n]
        if direction == "forward":
            out[b * n:(b + 1) * n] = np.fft.fft(lane).astype(np.complex64)
        else:
            # numpy.fft.ifft already divides by N; undo when normalize=False.
            ref = np.fft.ifft(lane)
            if not normalize:
                ref = ref * n
            out[b * n:(b + 1) * n] = ref.astype(np.complex64)
    return out


CASES = [
    (2,  1, "forward", False, "N=4 fwd"),
    (3,  1, "forward", False, "N=8 fwd"),
    (6,  1, "forward", False, "N=64 fwd"),
    (8,  1, "forward", False, "N=256 fwd"),
    (10, 1, "forward", False, "N=1024 fwd"),
    (12, 1, "forward", False, "N=4096 fwd"),
    (8,  8, "forward", False, "N=256 fwd batch=8"),
    (8,  1, "inverse", True,  "N=256 inv normalize"),
]


@pytest.mark.parametrize("log_n,batch,direction,normalize,label", CASES)
@pytest.mark.parametrize("use_device", [False, True])
def test_gpu_fft_matches_numpy(device, log_n, batch, direction, normalize, label, use_device):
    n = 1 << log_n
    inp = _input_signal(n, batch)
    expected = _reference(inp, n, batch, direction, normalize)
    kwargs = {"log_n": log_n, "direction": direction, "normalize": normalize}
    if use_device:
        kwargs["device"] = device
    out = fgpu.fft.fft_1d_c2c_pow2(inp, **kwargs)
    assert out.shape == inp.shape, f"{label} (use_device={use_device}): shape mismatch"

    diff = out - expected
    err = float(np.max(np.abs(diff))) / max(1e-9, float(np.max(np.abs(expected))))
    assert err < 1e-4, f"{label} (use_device={use_device}): max rel err {err} >= 1e-4"


# Batch >= ferrum_gpu_fft::R16S_MIN_BATCH (4096) routes N=256 and N=4096 to the
# scalarised radix-16 + u64-coalesced kernels (fft_c2c_256_r16s /
# fft_c2c_4096_r16s) that reach cuFFT parity / 1.74x. These cases exercise that
# path through the Python API and check it stays bit-faithful to numpy. Inputs
# and the reference are vectorised so the larger transforms stay fast.
R16S_CASES = [
    (8,  4096, "forward", False, "N=256 fwd batch=4096 (r16s)"),
    (8,  4096, "inverse", True,  "N=256 inv batch=4096 (r16s)"),
    (12, 4096, "forward", False, "N=4096 fwd batch=4096 (r16s)"),
]


def _input_signal_vec(n: int, batch: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32)
    lane = (np.sin(i) + 1j * np.cos(i)).astype(np.complex64)
    return np.tile(lane, batch)


@pytest.mark.parametrize("log_n,batch,direction,normalize,label", R16S_CASES)
@pytest.mark.parametrize("use_device", [False, True])
def test_r16s_large_batch_matches_numpy(device, log_n, batch, direction, normalize, label, use_device):
    n = 1 << log_n
    inp = _input_signal_vec(n, batch)
    lanes = inp.reshape(batch, n)
    if direction == "forward":
        ref = np.fft.fft(lanes, axis=1)
    else:
        ref = np.fft.ifft(lanes, axis=1)
        if not normalize:
            ref = ref * n
    expected = ref.astype(np.complex64).reshape(-1)

    kwargs = {"log_n": log_n, "direction": direction, "normalize": normalize}
    if use_device:
        kwargs["device"] = device
    out = fgpu.fft.fft_1d_c2c_pow2(inp, **kwargs)
    assert out.shape == inp.shape, f"{label} (use_device={use_device}): shape mismatch"

    err = float(np.max(np.abs(out - expected))) / max(1e-9, float(np.max(np.abs(expected))))
    # Radix-16 does more f32 arithmetic per output than radix-2; 1e-3 matches
    # the bench's r16s-vs-radix-2 correctness gate.
    assert err < 1e-3, f"{label} (use_device={use_device}): max rel err {err} >= 1e-3"
