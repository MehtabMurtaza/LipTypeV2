#!/usr/bin/env bash
set -euo pipefail

(( $# >= 2 && $# <= 3 )) || {
  echo "Usage: $0 <corpus.txt> <output_prefix> [kenlm_bin_dir]"
  echo "Example: $0 runs/lm/grid_char_corpus.txt runs/lm/grid_char_5gram /opt/kenlm/bin"
  exit 1
}

CORPUS="$1"
OUT_PREFIX="$2"
KENLM_BIN_DIR="${3:-}"

[ -f "$CORPUS" ] || {
  echo "Input corpus not found: $CORPUS"
  exit 2
}

LMPLZ="${KENLM_BIN_DIR:+$KENLM_BIN_DIR/}lmplz"
BUILD_BINARY="${KENLM_BIN_DIR:+$KENLM_BIN_DIR/}build_binary"

command -v "$LMPLZ" >/dev/null 2>&1 || {
  echo "Could not find lmplz. Pass kenlm bin dir as 3rd arg, or add it to PATH."
  exit 3
}

command -v "$BUILD_BINARY" >/dev/null 2>&1 || {
  echo "Could not find build_binary. Pass kenlm bin dir as 3rd arg, or add it to PATH."
  exit 4
}

mkdir -p "$(dirname "$OUT_PREFIX")"
ARPA="${OUT_PREFIX}.arpa"
BINARY="${OUT_PREFIX}.binary"

echo "[kenlm] building 5-gram arpa: $ARPA"
"$LMPLZ" -o 5 --text "$CORPUS" --arpa "$ARPA"

echo "[kenlm] building binary: $BINARY"
"$BUILD_BINARY" "$ARPA" "$BINARY"

echo "[kenlm] done"
echo "  arpa:   $ARPA"
echo "  binary: $BINARY"

