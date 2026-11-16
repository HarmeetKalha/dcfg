#!/usr/bin/env bash
# Usage: ops/jobs/ffpp_download.sh <dataset> <compression>
#   dataset: original|Deepfakes|Face2Face|FaceSwap|NeuralTextures  (original -> original_sequences/youtube)
#   compression: c23|c40
# Resumable: the official script skips existing files (and only renames a temp file into
# place after a complete download), so re-running fetches only missing videos. An
# interrupted run may leave a stray tmp* file in the output dir; harmless, not cleaned here.
# Server EU2 (kaldir.vc.in.tum.de; EU refuses connections, CA times out; tested 2026-11-16).
set -euo pipefail
DS="${1:?dataset}"; CMP="${2:?compression}"
OUT="$HOME/data/ffpp_raw"
SCRIPT="$OUT/faceforensics_download_v4.py"
case "$DS" in original|Deepfakes|Face2Face|FaceSwap|NeuralTextures) ;; *) echo "bad dataset $DS" >&2; exit 2;; esac
case "$CMP" in c23|c40) ;; *) echo "bad compression $CMP" >&2; exit 2;; esac
echo | python -I "$SCRIPT" "$OUT" -d "$DS" -c "$CMP" -t videos --server EU2
