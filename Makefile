.PHONY: check test test-gpu example-vector-add fmt clippy clean

check:
	cargo check --workspace

test:
	cargo test --workspace --exclude vector-add

test-gpu:
	FERRUM_GPU_HAS_CUDA=1 cargo test --workspace --exclude vector-add

example-vector-add:
	cargo run --release -p vector-add

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

clean:
	cargo clean
