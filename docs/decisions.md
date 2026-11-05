# Decisions

Non-obvious choices, recorded as they are made.

## D1. Repo identity and licensing
- Local git identity `raunaksahahere <raunaksahahere@users.noreply.github.com>`, set repo-locally only.
- License: Apache-2.0. Commits stay local; nothing is pushed.
- No third-party provider API keys; validation is local only. Secrets are read
  from environment variables only if a user supplies them.

## D2. Test model and hardware
- Hardware: 12 CPU threads, ~7 GB RAM, AMD RX 6500M (4 GB VRAM, RDNA2) plus an
  integrated Vega GPU. No CUDA; ROCm does not officially support gfx1034.
- Primary study model: Qwen3-0.6B (reference Q8_0; Q4_K_M and Q2_K as
  quantization faults). Optional Qwen3-1.7B smoke check if memory allows.
- Backend: the user chose to run on the GPU. llama.cpp's Vulkan backend is the
  primary backend, with reference and targets always on the same backend so
  backend numerics are never mistaken for a fault. CPU is the fallback if Vulkan
  is unusable, and a CPU-reference vs GPU-target run is kept as an explicit control.
