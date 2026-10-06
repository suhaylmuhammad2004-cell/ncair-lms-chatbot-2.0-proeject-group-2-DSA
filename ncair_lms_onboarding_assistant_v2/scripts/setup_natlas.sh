#!/usr/bin/env bash
# Download a quantized N-ATLaS build and register it in Ollama under the name "natlas".
# Usage: scripts/setup_natlas.sh [QUANT] [OLLAMA_NAME]   QUANT = Q4_K_M (default, ~4.9 GB) | Q5_K_M | Q6_K | Q8_0 | F16
# Needs: ollama (https://ollama.com), pip install -U "huggingface_hub[cli]"
# Internet is needed ONLY for this download; afterwards everything runs offline.
set -euo pipefail
QUANT="${1:-Q4_K_M}"
NAME="${2:-natlas}"   # give each quantization its own name to compare them, e.g. natlas-q4 / natlas-f16
REPO="tosinamuda/N-ATLaS-GGUF"
DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$DIR"
huggingface-cli download "$REPO" --include "*${QUANT}*" --local-dir "$DIR"
GGUF="$(ls "$DIR"/*"${QUANT}"*.gguf | head -n1)"
echo "Using $GGUF"
cat > "$DIR/Modelfile" <<MF
FROM $GGUF
PARAMETER temperature 0.1
PARAMETER repeat_penalty 1.12
PARAMETER num_ctx 4096
MF
ollama create "$NAME" -f "$DIR/Modelfile"
ollama run "$NAME" "Mo fẹ́ kí ọ pẹ̀lú ìkíni kúkúrú." --verbose 2>&1 | tail -n 12
echo "Done. Model name for the app: $NAME"
