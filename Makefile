.PHONY: check test test-gpu example-vector-add example-vector-add-oxide example-fft verify-all fmt clippy clean

check:
	cargo check --workspace

test:
	cargo test --workspace --exclude vector-add --exclude vector-add-cuda-oxide --exclude fft-1d-c2c

test-gpu:
	FERRUM_GPU_HAS_CUDA=1 cargo test --workspace --exclude vector-add --exclude vector-add-cuda-oxide --exclude fft-1d-c2c

example-vector-add:
	cargo run --release -p vector-add

example-vector-add-oxide:
	cargo oxide run vector-add-cuda-oxide --bin vector-add-cuda-oxide

example-fft:
	cargo oxide run fft-1d-c2c --bin fft-1d-c2c

verify-all: check test-gpu example-vector-add example-vector-add-oxide example-fft
	@echo
	@echo "=== ALL CHECKS PASSED ==="

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets --exclude vector-add-cuda-oxide --exclude fft-1d-c2c -- -D warnings

clean:
	cargo clean
