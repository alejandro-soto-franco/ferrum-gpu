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
	# Same stale-artifact wipe as `develop`: cargo's fingerprint cache
	# doesn't track RUSTFLAGS the way it should for the cuda-oxide
	# codegen-backend, leaving a 0-byte .so in target/release/ that
	# breaks maturin's manylinux compliance check.
	rm -rf /home/cargo-targets/ferrum-gpu/release/.fingerprint/ferrum-gpu-py-* \
	       /home/cargo-targets/ferrum-gpu/release/libferrum_gpu.so \
	       /home/cargo-targets/ferrum-gpu/release/libferrum_gpu.d \
	       /home/cargo-targets/ferrum-gpu/release/deps/libferrum_gpu-*.so \
	       /home/cargo-targets/ferrum-gpu/maturin
	cd crates/ferrum-gpu-py && \
	  VIRTUAL_ENV=$(VENV) PATH=$(VENV)/bin:$$PATH \
	  RUSTFLAGS='$(FERRUM_GPU_RUSTFLAGS)' \
	  $(VENV)/bin/maturin build --release --auditwheel skip

wheel-manylinux:
	docker build -f crates/ferrum-gpu-py/Dockerfile.manylinux \
	    -t ferrum-gpu-builder:latest crates/ferrum-gpu-py
	docker run --rm -v $(PWD):/work -w /work \
	    ferrum-gpu-builder:latest \
	    /work/crates/ferrum-gpu-py/build-wheel.sh

pytest: develop
	cd crates/ferrum-gpu-py && \
	  VIRTUAL_ENV=$(VENV) PATH=$(VENV)/bin:$$PATH \
	  $(VENV)/bin/pytest python/tests -v

verify-all: check test-gpu example-vector-add example-vector-add-oxide example-fft pytest bench
	@echo
	@echo "=== ALL CHECKS PASSED ==="

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets --exclude vector-add-cuda-oxide --exclude fft-1d-c2c --exclude ferrum-gpu-py -- -D warnings

.PHONY: ptx-cusimd-dump ptx-radix8-regreport phase0
ptx-cusimd-dump:
	cargo build --release --bin cusimd-ptx-dump
	@find /home/cargo-targets/ferrum-gpu -name 'cusimd_ptx_dump*.ptx' -exec cat {} \;

ptx-radix8-regreport:
	RUSTFLAGS="-C link-arg=-Xptxas=-v" cargo build --release --bin radix8-regreport 2>&1 | grep -E "registers|spill|stack"

phase0:
	cd crates/ferrum-gpu-bench && cargo oxide run --bin cufft-ncu-trace && cd ../..
	cd crates/ferrum-gpu-bench && cargo oxide run --bin cusimd-ptx-dump && cd ../..
	cd crates/ferrum-gpu-bench && cargo oxide run --bin launch-overhead-microbench && cd ../..
	cd crates/ferrum-gpu-bench && cargo oxide run --bin smem-bank-conflict-probe && cd ../..
	@echo
	@echo "Note: Task 0.1 step 5 requires: ncu --set full --csv target/release/cufft-ncu-trace"

.PHONY: gpu-lock bench-locked perf-gate
gpu-lock:
	./tools/bench-gpu-lock.sh

bench-locked:
	./tools/bench-gpu-lock.sh make bench

perf-gate:
	./tools/bench-gpu-lock.sh sh -c 'cd crates/ferrum-gpu-bench && cargo oxide run --bin perf-gate'

clean:
	cargo clean
