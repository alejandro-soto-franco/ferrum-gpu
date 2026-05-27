.PHONY: check test test-gpu example-vector-add example-vector-add-oxide example-fft bench develop wheel pytest verify-all fmt clippy clean

# cuda-oxide-codegen-backend RUSTFLAGS used by maturin when building the
# Python cdylib. Mirrors what cargo-oxide sets internally for `cargo oxide run/build`.
FERRUM_GPU_RUSTFLAGS = -Z codegen-backend=$(HOME)/.cargo/cuda-oxide/librustc_codegen_cuda.so -C opt-level=3 -C debug-assertions=off -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0

# Activate the Plan 4 venv before any python invocation.
VENV = $(HOME)/.venvs/ferrum-gpu

check:
	cargo check --workspace --exclude ferrum-gpu-py

test:
	cargo test --workspace --exclude vector-add --exclude vector-add-cuda-oxide --exclude fft-1d-c2c --exclude ferrum-gpu-py

test-gpu:
	FERRUM_GPU_HAS_CUDA=1 cargo test --workspace --exclude vector-add --exclude vector-add-cuda-oxide --exclude fft-1d-c2c --exclude ferrum-gpu-py

example-vector-add:
	cargo run --release -p vector-add

example-vector-add-oxide:
	cargo oxide run vector-add-cuda-oxide --bin vector-add-cuda-oxide

example-fft:
	cargo oxide run fft-1d-c2c --bin fft-1d-c2c

bench:
	cargo oxide run ferrum-gpu-bench --bin ferrum-gpu-bench

develop:
	# Wipe stale release artifacts so cargo always rebuilds ferrum-gpu-py
	# with the cuda-oxide RUSTFLAGS. `cargo clean -p ferrum-gpu-py` alone
	# leaves a stale `libferrum_gpu.so` next to the fingerprint dir, which
	# confuses maturin into thinking the build is up-to-date.
	rm -rf /home/cargo-targets/ferrum-gpu/release/.fingerprint/ferrum-gpu-py-* \
	       /home/cargo-targets/ferrum-gpu/release/libferrum_gpu.so \
	       /home/cargo-targets/ferrum-gpu/release/libferrum_gpu.d \
	       /home/cargo-targets/ferrum-gpu/release/deps/libferrum_gpu-*.so \
	       /home/cargo-targets/ferrum-gpu/maturin
	cd crates/ferrum-gpu-py && \
	  VIRTUAL_ENV=$(VENV) PATH=$(VENV)/bin:$$PATH \
	  RUSTFLAGS='$(FERRUM_GPU_RUSTFLAGS)' \
	  $(VENV)/bin/maturin develop --release

wheel:
	cd crates/ferrum-gpu-py && \
	  VIRTUAL_ENV=$(VENV) PATH=$(VENV)/bin:$$PATH \
	  RUSTFLAGS='$(FERRUM_GPU_RUSTFLAGS)' \
	  $(VENV)/bin/maturin build --release

wheel-manylinux:
	docker build -f crates/ferrum-gpu-py/Dockerfile.manylinux \
	    -t ferrum-gpu-builder:latest crates/ferrum-gpu-py
	docker run --rm -v $(PWD):/work -w /work --gpus all \
	    ferrum-gpu-builder:latest \
	    /work/crates/ferrum-gpu-py/build-wheel.sh

pytest: develop
	cd crates/ferrum-gpu-py && \
	  VIRTUAL_ENV=$(VENV) PATH=$(VENV)/bin:$$PATH \
	  $(VENV)/bin/pytest python/tests -v

verify-all: check test-gpu example-vector-add example-vector-add-oxide example-fft pytest
	@echo
	@echo "=== ALL CHECKS PASSED ==="

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets --exclude vector-add-cuda-oxide --exclude fft-1d-c2c --exclude ferrum-gpu-py -- -D warnings

clean:
	cargo clean
