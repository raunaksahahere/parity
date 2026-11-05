# Baseline noise of local targets

Measured with `scripts/baseline_noise.py` (raw output: `data/baseline-noise.json`).
Model: Qwen3-0.6B. Reference: HF transformers fp32 on CPU at revision `c1899de`.
Targets: llama.cpp b12128 serving GGUFs built from one BF16 file. GPU runs use the RX 6500M
through Vulkan (offload to 29/29 layers asserted from the server log).

Corpus: the 25 documents of `core-v1`, each truncated to 64 tokens: 1437 teacher-forced
positions. For each position the log-probability of the *actual next token* is compared
with the reference. The server is asked for its top-100; positions where the forced token
is outside the returned top-100 (about 7%) have no log-prob and are excluded (`miss`), which
biases these numbers toward high-probability tokens.

## Each deployment vs the HF fp32 reference (nats)

| deployment | n | miss | mean Δ | mean \|Δ\| | p99 \|Δ\| | max \|Δ\| |
|---|---:|---:|---:|---:|---:|---:|
| BF16, GPU, default | 1335 | 102 | 0.00024 | 0.00112 | 0.00654 | 0.01643 |
| BF16, GPU, default (repeat) | 1335 | 102 | 0.00024 | 0.00112 | 0.00654 | 0.01643 |
| BF16, GPU, `-b 64 -ub 32` | 1335 | 102 | 0.00024 | 0.00112 | 0.00654 | 0.01643 |
| BF16, GPU, 4 parallel slots, 4 concurrent requests | 1335 | 102 | 0.00022 | 0.00116 | 0.00664 | 0.01960 |
| BF16, CPU | 1333 | 104 | -0.00045 | 0.00675 | 0.04154 | 0.06020 |
| Q8_0, GPU | 1333 | 104 | 0.00257 | 0.03388 | 0.19807 | 0.55308 |

## Each deployment vs the default BF16 GPU run (nats)

| deployment | mean \|Δ\| | p99 \|Δ\| | max \|Δ\| |
|---|---:|---:|---:|
| BF16, GPU, repeat | 0 | 0 | 0 |
| BF16, GPU, `-b 64 -ub 32` | 0 | 0 | 0 |
| BF16, GPU, 4 parallel slots | 0.00103 | 0.00630 | 0.00991 |
| BF16, CPU | 0.00684 | 0.04220 | 0.06048 |
| Q8_0, GPU | 0.03379 | 0.19781 | 0.54934 |

## What this tells us

- **Repeats on the same configuration are bit-identical** on the GPU, so single-configuration
  repeat noise is zero here. A noise floor built only from repeats would be zero and would
  over-flag. Calibration therefore has to include benign *configuration* changes.
- `-b`/`-ub` made no difference **in this measurement** because teacher forcing sends one new
  token per request, so the batch never gets larger than one token. This is a limit of the
  measurement, not evidence that batch size never matters; concurrent slots do change the numbers.
- **Backend is a real source of benign variation:** CPU BF16 is about 6x further from the
  reference than GPU BF16 (mean |Δ| 0.0068 vs 0.0011). Comparing a CPU deployment to a GPU
  reference (or the reverse) would otherwise be read as a deviation.
- **Q8_0 is separable from benign noise on this model:** mean |Δ| is about 30x the GPU noise
  floor and 5x the CPU-vs-GPU difference. Whether Parity reports it as a deviation depends
  on the practical threshold, which is the subject of the validation study.
- Wall-clock on this machine for 1437 requests: GPU 22-40 s (Q8_0 37 s, BF16 40 s,
  parallel slots 22 s), CPU 131 s.
