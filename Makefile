.PHONY: check test test-gpu example-vector-add example-vector-add-oxide verify-all fmt clippy clean

check:
	cargo check --workspace

test:
	cargo test --workspace --exclude vector-add --exclude vector-add-cuda-oxide

test-gpu:
	FERRUM_GPU_HAS_CUDA=1 cargo test --workspace --exclude vector-add --exclude vector-add-cuda-oxide

example-vector-add:
	cargo run --release -p vector-add

example-vector-add-oxide:
	cargo oxide run vector-add-cuda-oxide --bin vector-add-cuda-oxide

verify-all: check test-gpu example-vector-add example-vector-add-oxide
	@echo
	@echo "=== ALL CHECKS PASSED ==="

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets --exclude vector-add-cuda-oxide -- -D warnings

clean:
	cargo clean
