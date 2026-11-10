#!/usr/bin/env bash
# Create the Ollama models used to validate the Ollama adapter (v0.2.0, docs/validation.md).
#
# Weights: the same verified GGUF files used with llama.cpp. Chat template, parameters and
# system prompt: the Ollama library's own layers for qwen3:0.6b and smollm2:360m, fetched by
# pinned digest and checked. Variants:
#   parity-<m>-libtpl       correct deployment as Ollama ships it (GPU when available)
#   parity-<m>-libtpl-cpu   same, forced onto the CPU (num_gpu 0): a benign variant
#   parity-<m>-libtpl-ctx8k same, num_ctx 8192: a held-out benign variant
#   parity-smollm2-q4km-libtpl   Q4_K_M weights with the library template: a real fault
#   parity-smollm2-f16-notpl     F16 weights, no TEMPLATE: Ollama's fallback, a real fault
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
work=${OLLAMA_WORK:-$root/runs/ollama-models}
mkdir -p "$work"
reg=https://registry.ollama.ai/v2/library

fetch() {  # fetch REPO DIGEST OUT
  curl -sfL "$reg/$1/blobs/$2" -o "$3"
  echo "${2#sha256:}  $3" | sha256sum -c --quiet
}
fetch qwen3 sha256:ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4 "$work/qwen3.template"
fetch qwen3 sha256:cff3f395ef3756ab63e58b0ad1b32bb6f802905cae1472e6a12034e4246fbbdb "$work/qwen3.params"
fetch smollm2 sha256:d502d55c1d609104ae6127aee92eb940e51e15c56dfb26dbd067e2771ee746f1 "$work/smollm2.template"
fetch smollm2 sha256:f02dd72bb2423204352eabc5637b44d79d17f109fdb510a7c51455892aa2d216 "$work/smollm2.params"
fetch smollm2 sha256:fbacade46b4da804e0398c339c64b944d4b954452adf77cf050b49420116129e "$work/smollm2.system"

modelfile() {  # modelfile NAME GGUF FAMILY [extra PARAMETER lines...]
  local name=$1 gguf=$2 fam=$3; shift 3
  python3 - "$work" "$fam" "$gguf" "$@" > "$work/$name.Modelfile" <<'EOF'
import json, sys
from pathlib import Path
work, fam, gguf, *extra = sys.argv[1:]
w = Path(work)
print(f"FROM {gguf}")
if fam != "none":
    print('TEMPLATE """' + (w / f"{fam}.template").read_text() + '"""')
    if (w / f"{fam}.system").exists():
        print('SYSTEM """' + (w / f"{fam}.system").read_text().strip() + '"""')
    for k, v in json.loads((w / f"{fam}.params").read_text()).items():
        for item in v if isinstance(v, list) else [v]:
            print(f"PARAMETER {k} {json.dumps(item) if isinstance(item, str) else item}")
for line in extra:
    print(f"PARAMETER {line}")
EOF
  ollama create "$name" -f "$work/$name.Modelfile" >/dev/null
  echo "created $name"
}
q=$root/models/gguf/Qwen3-0.6B-BF16.gguf
s=$root/models/gguf-smollm2/SmolLM2-360M-Instruct-F16.gguf
s4=$root/models/gguf-smollm2/SmolLM2-360M-Instruct-Q4_K_M.gguf
modelfile parity-qwen3-bf16-libtpl "$q" qwen3
modelfile parity-smollm2-f16-libtpl "$s" smollm2
modelfile parity-smollm2-f16-libtpl-cpu "$s" smollm2 "num_gpu 0"
modelfile parity-smollm2-f16-libtpl-ctx8k "$s" smollm2 "num_ctx 8192"
modelfile parity-smollm2-q4km-libtpl "$s4" smollm2
modelfile parity-smollm2-f16-notpl "$s" none
