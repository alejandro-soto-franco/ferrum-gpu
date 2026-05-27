"""GPU FFT vs numpy.fft.fft on 8 cases x 2 device modes."""

import numpy as np
import pytest

import ferrum_gpu as fg


@pytest.fixture(scope="module")
def device():
    return fg.cuda.Device(0)


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
    out = fg.fft.fft_1d_c2c_pow2(inp, **kwargs)
    assert out.shape == inp.shape, f"{label} (use_device={use_device}): shape mismatch"

    diff = out - expected
    err = float(np.max(np.abs(diff))) / max(1e-9, float(np.max(np.abs(expected))))
    assert err < 1e-4, f"{label} (use_device={use_device}): max rel err {err} >= 1e-4"
