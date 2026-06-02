// N=4096 radix-16, scalarised, u64-coalesced IO — generalises the N=256 win.
// 4096 = 16^3, exactly 3 Stockham radix-16 stages (vs radix-8's 4 stages).
// 256 threads/block (= N/16 butterflies), ALL active (no idle lanes), one FFT
// per block. Same recipe that reached cuFFT parity at N=256: fewer shared
// round-trips (higher radix) + u64-coalesced global IO (cuda-oxide emits scalar
// generic ld.b32 for &[f32], ~22% sector util; one u64/complex -> ~100%).
//
// dft16 = two scalar dft8 + 8 W_16 combines (named registers; the array form is
// rejected by cuda-oxide). Twiddle via a W_4096 table indexed by p*k*(4096/m).
// Built only via `cargo oxide`, not the shipped wheel.

#[::cuda_host::cuda_module]
mod fft4096_r16s {
    use ::cuda_device::{DisjointSlice, SharedArray, kernel, thread};

    /// 1D forward C2C FFT, N=4096, scalarised radix-16, u64-coalesced IO.
    /// `w4096`: `W_4096^e`, e in 0..4096. Launch: grid=(batch,1,1), block=(256,1,1).
    #[kernel]
    pub fn fft_c2c_4096_r16s(in_data: &[f32], w4096: &[f32], mut out_data: DisjointSlice<f32>) {
        static mut BUF: SharedArray<f32, 8192> = SharedArray::UNINIT; // 4096 complex

        const N: usize = 4096;
        const THREADS: usize = 256; // = N/16 butterflies
        const NR: usize = N / 16; // 256 stride between gathered inputs

        let blk = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane_off = blk * N * 2;
        let in_ptr = in_data.as_ptr();

        // Coalesced load: one u64 (re|im) per complex.
        {
            let mut t = tid;
            while t < N {
                let c: u64 = unsafe { *(in_ptr.add(lane_off + 2 * t) as *const u64) };
                unsafe {
                    BUF[2 * t] = f32::from_bits(c as u32);
                    BUF[2 * t + 1] = f32::from_bits((c >> 32) as u32);
                }
                t += THREADS;
            }
        }
        thread::sync_threads();

        macro_rules! dft8 {
            ($x0:expr,$x1:expr,$x2:expr,$x3:expr,$x4:expr,$x5:expr,$x6:expr,$x7:expr,
             $x8:expr,$x9:expr,$x10:expr,$x11:expr,$x12:expr,$x13:expr,$x14:expr,$x15:expr) => {{
                const C: f32 = 0.70710678_f32;
                let (x0r,x0i)=($x0,$x1); let (x1r,x1i)=($x2,$x3);
                let (x2r,x2i)=($x4,$x5); let (x3r,x3i)=($x6,$x7);
                let (x4r,x4i)=($x8,$x9); let (x5r,x5i)=($x10,$x11);
                let (x6r,x6i)=($x12,$x13); let (x7r,x7i)=($x14,$x15);
                let (a0r,a0i)=(x0r+x4r,x0i+x4i); let (a1r,a1i)=(x0r-x4r,x0i-x4i);
                let (b0r,b0i)=(x2r+x6r,x2i+x6i); let (b1r,b1i)=(x2r-x6r,x2i-x6i);
                let (e0r,e0i)=(a0r+b0r,a0i+b0i); let (e2r,e2i)=(a0r-b0r,a0i-b0i);
                let (e1r,e1i)=(a1r+b1i,a1i-b1r); let (e3r,e3i)=(a1r-b1i,a1i+b1r);
                let (c0r,c0i)=(x1r+x5r,x1i+x5i); let (c1r,c1i)=(x1r-x5r,x1i-x5i);
                let (d0r,d0i)=(x3r+x7r,x3i+x7i); let (d1r,d1i)=(x3r-x7r,x3i-x7i);
                let (o0r,o0i)=(c0r+d0r,c0i+d0i); let (o2r,o2i)=(c0r-d0r,c0i-d0i);
                let (o1r,o1i)=(c1r+d1i,c1i-d1r); let (o3r,o3i)=(c1r-d1i,c1i+d1r);
                let (w1r,w1i)=(C*(o1r+o1i),C*(o1i-o1r));
                let (w3r,w3i)=(C*(o3i-o3r),-C*(o3r+o3i));
                (e0r+o0r,e0i+o0i, e1r+w1r,e1i+w1i, e2r+o2i,e2i-o2r, e3r+w3r,e3i+w3i,
                 e0r-o0r,e0i-o0i, e1r-w1r,e1i-w1i, e2r-o2i,e2i+o2r, e3r-w3r,e3i-w3i)
            }};
        }

        // One radix-16 stage. `fac` = 4096/m so the twiddle is W_4096^(p*k*fac).
        macro_rules! stage {
            ($m_r:expr, $fac:expr, $do_tw:expr) => {{
                let b: usize = tid;
                let m_r: usize = $m_r;
                let m = m_r * 16;
                let j = b / m_r;
                let k = b - j * m_r;
                let src_base = j * m_r + k;
                macro_rules! g { ($p:expr) => {{
                    let s = 2*(src_base + ($p)*NR);
                    unsafe { (BUF[s], BUF[s+1]) }
                }}; }
                macro_rules! tw { ($p:expr,$re:expr,$im:expr) => {{
                    if $do_tw && $p != 0 {
                        let e = 2*((($p)*k*$fac) & 4095);
                        let wr = w4096[e]; let wi = w4096[e+1];
                        ($re*wr - $im*wi, $re*wi + $im*wr)
                    } else { ($re,$im) }
                }}; }
                let (g0r,g0i)=g!(0);
                let (mut p1r,mut p1i)=g!(1); let (mut p2r,mut p2i)=g!(2);
                let (mut p3r,mut p3i)=g!(3); let (mut p4r,mut p4i)=g!(4);
                let (mut p5r,mut p5i)=g!(5); let (mut p6r,mut p6i)=g!(6);
                let (mut p7r,mut p7i)=g!(7); let (mut p8r,mut p8i)=g!(8);
                let (mut p9r,mut p9i)=g!(9); let (mut p10r,mut p10i)=g!(10);
                let (mut p11r,mut p11i)=g!(11); let (mut p12r,mut p12i)=g!(12);
                let (mut p13r,mut p13i)=g!(13); let (mut p14r,mut p14i)=g!(14);
                let (mut p15r,mut p15i)=g!(15);
                let r=tw!(1,p1r,p1i); p1r=r.0; p1i=r.1;
                let r=tw!(2,p2r,p2i); p2r=r.0; p2i=r.1;
                let r=tw!(3,p3r,p3i); p3r=r.0; p3i=r.1;
                let r=tw!(4,p4r,p4i); p4r=r.0; p4i=r.1;
                let r=tw!(5,p5r,p5i); p5r=r.0; p5i=r.1;
                let r=tw!(6,p6r,p6i); p6r=r.0; p6i=r.1;
                let r=tw!(7,p7r,p7i); p7r=r.0; p7i=r.1;
                let r=tw!(8,p8r,p8i); p8r=r.0; p8i=r.1;
                let r=tw!(9,p9r,p9i); p9r=r.0; p9i=r.1;
                let r=tw!(10,p10r,p10i); p10r=r.0; p10i=r.1;
                let r=tw!(11,p11r,p11i); p11r=r.0; p11i=r.1;
                let r=tw!(12,p12r,p12i); p12r=r.0; p12i=r.1;
                let r=tw!(13,p13r,p13i); p13r=r.0; p13i=r.1;
                let r=tw!(14,p14r,p14i); p14r=r.0; p14i=r.1;
                let r=tw!(15,p15r,p15i); p15r=r.0; p15i=r.1;
                let ev = dft8!(g0r,g0i, p2r,p2i, p4r,p4i, p6r,p6i, p8r,p8i, p10r,p10i, p12r,p12i, p14r,p14i);
                let od = dft8!(p1r,p1i, p3r,p3i, p5r,p5i, p7r,p7i, p9r,p9i, p11r,p11i, p13r,p13i, p15r,p15i);
                const W: [(f32,f32);8] = [
                    (1.0,0.0),(0.92387953,-0.38268343),(0.70710678,-0.70710678),
                    (0.38268343,-0.92387953),(0.0,-1.0),(-0.38268343,-0.92387953),
                    (-0.70710678,-0.70710678),(-0.92387953,-0.38268343)];
                macro_rules! cb { ($c:expr, $er:expr,$ei:expr,$or:expr,$oi:expr) => {{
                    let (wr,wi)=W[$c];
                    let tr=$or*wr - $oi*wi; let ti=$or*wi + $oi*wr;
                    ($er+tr, $ei+ti, $er-tr, $ei-ti)
                }}; }
                let (x0r,x0i,x8r,x8i)=cb!(0, ev.0,ev.1, od.0,od.1);
                let (x1r,x1i,x9r,x9i)=cb!(1, ev.2,ev.3, od.2,od.3);
                let (x2r,x2i,x10r,x10i)=cb!(2, ev.4,ev.5, od.4,od.5);
                let (x3r,x3i,x11r,x11i)=cb!(3, ev.6,ev.7, od.6,od.7);
                let (x4r,x4i,x12r,x12i)=cb!(4, ev.8,ev.9, od.8,od.9);
                let (x5r,x5i,x13r,x13i)=cb!(5, ev.10,ev.11, od.10,od.11);
                let (x6r,x6i,x14r,x14i)=cb!(6, ev.12,ev.13, od.12,od.13);
                let (x7r,x7i,x15r,x15i)=cb!(7, ev.14,ev.15, od.14,od.15);
                let dst_base = j*m + k;
                macro_rules! s { ($q:expr,$re:expr,$im:expr) => {{
                    let d = 2*(dst_base + ($q)*m_r);
                    unsafe { BUF[d]=$re; BUF[d+1]=$im; }
                }}; }
                s!(0,x0r,x0i); s!(1,x1r,x1i); s!(2,x2r,x2i); s!(3,x3r,x3i);
                s!(4,x4r,x4i); s!(5,x5r,x5i); s!(6,x6r,x6i); s!(7,x7r,x7i);
                s!(8,x8r,x8i); s!(9,x9r,x9i); s!(10,x10r,x10i); s!(11,x11r,x11i);
                s!(12,x12r,x12i); s!(13,x13r,x13i); s!(14,x14r,x14i); s!(15,x15r,x15i);
            }};
        }

        stage!(1, 256, false);
        thread::sync_threads();
        stage!(16, 16, true);
        thread::sync_threads();
        stage!(256, 1, true);
        thread::sync_threads();

        {
            let mut t = tid;
            while t < N {
                let (re, im) = unsafe { (BUF[2 * t], BUF[2 * t + 1]) };
                let packed: u64 = (re.to_bits() as u64) | ((im.to_bits() as u64) << 32);
                let p = unsafe { out_data.get_unchecked_mut(lane_off + 2 * t) as *mut f32 as *mut u64 };
                unsafe { *p = packed; }
                t += THREADS;
            }
        }
    }
}
