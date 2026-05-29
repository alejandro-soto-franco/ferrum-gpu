// Ceiling experiment for fft_c2c_4096 (Task 3.5c).
//
// Compiles the SAME radix-8 Stockham algorithm as the in-tree cuda-oxide
// kernel, but through nvcc -O3, and times it against cuFFT with the same
// alternating CUDA-event scheme the Rust perf-gate uses. Answers: is the
// ~3.8x gap to cuFFT a cuda-oxide codegen limitation (nvcc gets close ->
// build the hand/nvcc-PTX bridge) or algorithmic (nvcc also ~3.8x off ->
// relax the gate)?
//
// Build: nvcc -O3 -arch=sm_120 tools/radix8_ceiling.cu -lcufft -o /tmp/radix8_ceiling
// Run:   /tmp/radix8_ceiling

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("cuda err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

static const int N = 4096;
static const int BATCH = 256;
static const int STAGES = 4;
static const int THREADS = 512;
static const int NR = N / 8;
static const int WARMUP = 10;
static const int TRIALS = 100;

// Same dft8 as the Rust dft8! macro (DIT radix-2 flow graph, W_8 folded).
__device__ __forceinline__ void dft8(float* x) {
    const float C = 0.70710678f;
    float x0r=x[0],x0i=x[1],x1r=x[2],x1i=x[3],x2r=x[4],x2i=x[5],x3r=x[6],x3i=x[7];
    float x4r=x[8],x4i=x[9],x5r=x[10],x5i=x[11],x6r=x[12],x6i=x[13],x7r=x[14],x7i=x[15];
    float a0r=x0r+x4r,a0i=x0i+x4i,a1r=x0r-x4r,a1i=x0i-x4i;
    float b0r=x2r+x6r,b0i=x2i+x6i,b1r=x2r-x6r,b1i=x2i-x6i;
    float e0r=a0r+b0r,e0i=a0i+b0i,e2r=a0r-b0r,e2i=a0i-b0i;
    float e1r=a1r+b1i,e1i=a1i-b1r,e3r=a1r-b1i,e3i=a1i+b1r;
    float c0r=x1r+x5r,c0i=x1i+x5i,c1r=x1r-x5r,c1i=x1i-x5i;
    float d0r=x3r+x7r,d0i=x3i+x7i,d1r=x3r-x7r,d1i=x3i-x7i;
    float o0r=c0r+d0r,o0i=c0i+d0i,o2r=c0r-d0r,o2i=c0i-d0i;
    float o1r=c1r+d1i,o1i=c1i-d1r,o3r=c1r-d1i,o3i=c1i+d1r;
    float w1r=C*(o1r+o1i),w1i=C*(o1i-o1r);
    float w3r=C*(o3i-o3r),w3i=-C*(o3r+o3i);
    x[0]=e0r+o0r;  x[1]=e0i+o0i;
    x[2]=e1r+w1r;  x[3]=e1i+w1i;
    x[4]=e2r+o2i;  x[5]=e2i-o2r;
    x[6]=e3r+w3r;  x[7]=e3i+w3i;
    x[8]=e0r-o0r;  x[9]=e0i-o0i;
    x[10]=e1r-w1r; x[11]=e1i-w1i;
    x[12]=e2r-o2i; x[13]=e2i+o2r;
    x[14]=e3r-w3r; x[15]=e3i-w3i;
}

__global__ void fft_c2c_4096(const float* __restrict__ in_data,
                             const float* __restrict__ tw,
                             float* __restrict__ out_data) {
    __shared__ float BUF[8192];
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int lane_off = block * N * 2;
    for (int t = tid; t < N; t += THREADS) {
        BUF[2*t]   = in_data[lane_off + 2*t];
        BUF[2*t+1] = in_data[lane_off + 2*t+1];
    }
    __syncthreads();

    int b = tid;
    int m_r = 1, stage_off = 0;
    for (int stage = 0; stage < STAGES; ++stage) {
        int m = m_r * 8;
        int j = b / m_r;
        int k = b - j * m_r;
        int src_base = j * m_r + k;
        int tw_base = 2 * (stage_off + 8 * k);

        float x[16];
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            int si = 2 * (src_base + p * NR);
            float re = BUF[si], im = BUF[si+1];
            if (p == 0) { x[0]=re; x[1]=im; }
            else {
                float wr = tw[tw_base + 2*p], wi = tw[tw_base + 2*p + 1];
                x[2*p]   = re*wr - im*wi;
                x[2*p+1] = re*wi + im*wr;
            }
        }
        dft8(x);
        __syncthreads();
        int dst_base = j * m + k;
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            int di = 2 * (dst_base + q * m_r);
            BUF[di]   = x[2*q];
            BUF[di+1] = x[2*q+1];
        }
        __syncthreads();
        stage_off += 8 * m_r;
        m_r = m;
    }
    for (int t = tid; t < N; t += THREADS) {
        out_data[lane_off + 2*t]   = BUF[2*t];
        out_data[lane_off + 2*t+1] = BUF[2*t+1];
    }
}

// Radix-8 input-twiddle table, same layout as twiddles_radix8(12).
static std::vector<float> make_tw() {
    std::vector<float> out;
    int m_r = 1;
    for (int s = 1; s <= STAGES; ++s) {
        int m = m_r * 8;
        double scale = -2.0 * M_PI / (double)m;
        for (int k = 0; k < m_r; ++k)
            for (int p = 0; p < 8; ++p) {
                int e = (p * k) % m;
                double th = scale * e;
                out.push_back((float)cos(th));
                out.push_back((float)sin(th));
            }
        m_r = m;
    }
    return out;
}

static float median(std::vector<float>& v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

int main() {
    int total = N * BATCH;
    std::vector<float> h_in(total*2);
    for (int i = 0; i < total*2; ++i) h_in[i] = sinf(i*0.001f);
    auto h_tw = make_tw();

    float *d_in,*d_out,*d_tw;
    CK(cudaMalloc(&d_in,  total*2*sizeof(float)));
    CK(cudaMalloc(&d_out, total*2*sizeof(float)));
    CK(cudaMalloc(&d_tw,  h_tw.size()*sizeof(float)));
    CK(cudaMemcpy(d_in, h_in.data(), total*2*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tw, h_tw.data(), h_tw.size()*sizeof(float), cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, BATCH);
    cufftComplex *c_in,*c_out;
    CK(cudaMalloc(&c_in,  total*sizeof(cufftComplex)));
    CK(cudaMalloc(&c_out, total*sizeof(cufftComplex)));
    CK(cudaMemcpy(c_in, h_in.data(), total*sizeof(cufftComplex), cudaMemcpyHostToDevice));

    dim3 grid(BATCH), blk(THREADS);
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);

    for (int i=0;i<WARMUP;i++){
        fft_c2c_4096<<<grid,blk>>>(d_in,d_tw,d_out);
        cufftExecC2C(plan,c_in,c_out,CUFFT_FORWARD);
    }
    CK(cudaDeviceSynchronize());

    std::vector<float> ours, cu;
    for (int i=0;i<TRIALS;i++){
        cudaEventRecord(s); fft_c2c_4096<<<grid,blk>>>(d_in,d_tw,d_out); cudaEventRecord(e);
        cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e); ours.push_back(ms);
        cudaEventRecord(s); cufftExecC2C(plan,c_in,c_out,CUFFT_FORWARD); cudaEventRecord(e);
        cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); cu.push_back(ms);
    }
    CK(cudaDeviceSynchronize());

    float o = median(ours), c = median(cu);
    // per-FFT microseconds
    float o_us = o*1e3f/BATCH, c_us = c*1e3f/BATCH;
    printf("nvcc radix-8 : %.4f us/FFT\n", o_us);
    printf("cuFFT        : %.4f us/FFT\n", c_us);
    printf("ratio (ours/cuFFT): %.3f  (gate wants <= 0.9)\n", o_us/c_us);
    return 0;
}
