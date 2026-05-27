"""GPU 2D FFT vs numpy.fft.fft2 on size + device matrix."""

import numpy as np
import pytest

import ferrum_gpu as fg


@pytest.fixture(scope="module")
def device():
    return fg.cuda.Device(0)


def _input_2d(n: int) -> np.ndarray:
    base = np.arange(n * n, dtype=np.float32).reshape(n, n)
    return (np.sin(base * 0.1) + 1j * np.cos(base * 0.07)).astype(np.complex64)


@pytest.mark.parametrize(
    "log_n,label",
    [
        (4, "N=16"),
        (5, "N=32"),
        (6, "N=64"),
        (8, "N=256"),
        (10, "N=1024"),
    ],
)
@pytest.mark.parametrize("use_device", [False, True])
def test_gpu_fft2d_matches_numpy(device, log_n, label, use_device):
    n = 1 << log_n
    arr = _input_2d(n)
    expected = np.fft.fft2(arr).astype(np.complex64)
    kwargs = {"log_n": log_n, "direction": "forward", "normalize": False}
    if use_device:
        kwargs["device"] = device
    out = fg.fft.fft_2d_c2c_pow2(arr, **kwargs)
    assert out.shape == arr.shape, f"{label}: shape mismatch"
    diff = out - expected
    err = float(np.max(np.abs(diff))) / max(1e-9, float(np.max(np.abs(expected))))
    assert err < 1e-3, f"{label} use_device={use_device}: max rel err {err}"


@pytest.mark.parametrize(
    "log_n,label",
    [
        (4, "N=16"),
        (6, "N=64"),
        (8, "N=256"),
    ],
)
def test_gpu_fft2d_inverse_roundtrip(device, log_n, label):
    n = 1 << log_n
    arr = _input_2d(n)
    fwd = fg.fft.fft_2d_c2c_pow2(arr, log_n=log_n, direction="forward", device=device)
    inv = fg.fft.fft_2d_c2c_pow2(
        fwd, log_n=log_n, direction="inverse", normalize=True, device=device
    )
    err = float(np.max(np.abs(arr - inv))) / max(1e-9, float(np.max(np.abs(arr))))
    assert err < 1e-3, f"{label} roundtrip err {err}"
