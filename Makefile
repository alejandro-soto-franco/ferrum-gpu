.PHONY: check test test-gpu example-vector-add example-vector-add-oxide fmt clippy clean

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

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

clean:
	cargo clean
