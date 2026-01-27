#!/bin/bash
CAPTION_PATH="$1" # {"video_id": xx, "pred": "xx"}
GT_PATH="$2"
VIDEO_DIR="$3"

EXP_NOTE=""
NUMPROCESS=20 # TODO

DIR_PATH=$(dirname "$CAPTION_PATH")
SAVE_PATH="$DIR_PATH/DiaDemBench_results_${EXP_NOTE}.jsonl" # TODO
mkdir -p "$(dirname "$SAVE_PATH")"


cd "$(dirname "$0")"
python get_judgement.py \
  --gt_path "$GT_PATH" \
  --video_dir "$VIDEO_DIR" \
  --caption_path "$CAPTION_PATH" \
  --save_path "$SAVE_PATH" \
  --num_process "$NUMPROCESS"

python save_results.py "$SAVE_PATH"