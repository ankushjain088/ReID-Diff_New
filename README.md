# Benchmarks: Relighting + Face Hallucination

This is a Google Colab notebook (exported as a `.py` file) that benchmarks a proposed image restoration pipeline against some standard baselines. The goal is to take dark, low-resolution face photos and make them look good again — and see if the proposed approach beats the existing methods.

---

## What it does

The pipeline takes a clean face image, intentionally degrades it (dark + blurry + noisy), then tries to restore it using different methods and compares the results.

**Degradation stuff:**
- Simulates realistic low-light camera captures — thinks about exposure reduction, sensor noise (Poisson + Gaussian), camera response curves, color temperature shifts, downsampling, and optional JPEG compression. It's trying to be physically accurate about how bad photos actually happen.

**The restoration methods being compared:**
1. **Zero-DCE + Bicubic** — basic relighting, then just upscale with bicubic
2. **Zero-DCE + Real-ESRGAN** — relight, then use a super-resolution model
3. **Zero-DCE + GFPGAN** — relight, then use a face restoration model
4. **Zero-DCE + SUPIR** — relight, then throw a big diffusion model at it
5. **Ours (EFE + Proposed)** — their own relighting model (EFE), run CodeFormer on it, inject edge structure, then SUPIR

**Metrics used** (all at 1024×1024):
- PSNR ↑ (higher is better)
- FSIM ↑ (higher is better)
- DISTS ↓ (lower is better)
- LPIPS ↓ (lower is better)
- Identity similarity via ArcFace (for face-specific evaluation)

---

## Models / repos it pulls in

- [EG3D](https://github.com/NVlabs/eg3d) — NVIDIA's 3D-aware GAN
- [GOAE](https://github.com/jiangyzy/GOAE) — encoder for GAN inversion
- [RetinexFormer](https://github.com/caiyuanhao1998/RetinexFormer) — low-light enhancement baseline
- [SwinIR](https://github.com/JingyunLiang/SwinIR) — super-resolution baseline
- [Lite2Relight](https://github.com/prraoo/lite2relight) — relighting model
- CodeFormer, GFPGAN, Real-ESRGAN, SUPIR — various face/image restoration tools
- Zero-DCE — lightweight low-light enhancement

---

## Datasets used

Downloaded from Kaggle:
- `lol-v1` — standard low-light benchmark dataset
- `flickrfaceshq-dataset-nvidia-resized-256px` — FFHQ faces at 256px
- `retinexformer-weights` — pretrained RetinexFormer checkpoint

---

## How to run it

This was designed for **Google Colab** (GPU required). You'd need:
1. A Kaggle API key (`kaggle.json`) — it uploads this at the start
2. GPU enabled in Colab (it checks for this)
3. CUDA 11.8 compatible environment

Just run the cells in order. It sets up the environment, downloads everything, trains/loads the models, runs the benchmark loop, and finally saves a comparison grid image.

---

## Output

A big visualization grid saved to `/content/ffhq_1024_benchmark_grid_zerodce.png` showing side-by-side results for each method, with PSNR/LPIPS/DISTS scores printed under each image.

A summary table also prints to console after each low-light severity level is tested.

---

## Notes

- The file is ~9400 lines long because it's a whole Colab notebook smashed into one `.py`. It contains model definitions, training loops, the degradation pipeline, benchmark runners, and visualization all in one place.
- Some checkpoints (like the EG3D one from NVIDIA NGC) might fail to download automatically and need to be grabbed manually.
- Needs a decent amount of VRAM — it's running SUPIR which is not small.
