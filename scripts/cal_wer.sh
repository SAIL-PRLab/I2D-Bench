#!/bin/bash
# 单轮 CER 计算：调用 seed-tts-eval/run_wer.py。
# 用法：cal_wer.sh <tts_model> <iter_num> <lang> <data_dir> <output_dir> [num_job]
set -euo pipefail

tts_model=$1
iter_num=$2
lang=$3
data_dir=$4
output_dir=$5
num_job=${6:-$(nvidia-smi -L | wc -l)}

project_dir=$(cd "$(dirname "$0")"; cd ../; pwd)

if [ -z "$tts_model" ] || [ -z "$iter_num" ] || [ -z "$lang" ] || [ -z "$data_dir" ] || [ -z "$output_dir" ]; then
    echo "用法：./cal_wer.sh <tts_model> <iter_num> <lang> <data_dir> <output_dir> [num_job]"
    exit 1
fi

target_text=$data_dir/target_text
ref_wav=$data_dir/ref_wav.scp
tts_output_scp=$data_dir/$tts_model/iter_${iter_num}.scp
wav_ref_text=$output_dir/wav_ref_text_iter_${iter_num}
score_file=$output_dir/score_iter_${iter_num}.wer

mkdir -p "$output_dir"

for f in "$target_text" "$ref_wav" "$tts_output_scp"; do
    [ -f "$f" ] || { echo "错误：缺少文件: $f"; exit 1; }
done

# 生成 run_wer.py 输入：wav_res|wav_ref|text_ref 三列
python3 "$project_dir/scripts/prepare_seed_file.py" \
    "$tts_output_scp" "$target_text" "$ref_wav" "$wav_ref_text"

python3 "$project_dir/seed-tts-eval/prepare_ckpt.py" > /dev/null 2>&1

# 切分到 num_job 个 shard 并行调用 run_wer.py
timestamp=$(date +%s)
thread_dir=/tmp/thread_metas_${timestamp}_${tts_model}_iter${iter_num}/
mkdir -p "$thread_dir/results"

num=$(wc -l < "$wav_ref_text")
num_per_thread=$((num / num_job + 1))
split -l "$num_per_thread" --additional-suffix=.lst -d "$wav_ref_text" "$thread_dir/thread-"

for rank in $(seq 0 $((num_job - 1))); do
    thread_file=$(printf "%s/thread-%02d.lst" "$thread_dir" "$rank")
    sub_score_file=$(printf "%s/results/thread-%02d.wer.out" "$thread_dir" "$rank")
    [ -f "$thread_file" ] || continue
    CUDA_VISIBLE_DEVICES=$rank \
        python3 "$project_dir/seed-tts-eval/run_wer.py" \
        "$thread_file" "$sub_score_file" "$lang" &
done
wait

cat "$thread_dir"/results/thread-*.wer.out > "$thread_dir/results/merge.out"
python3 "$project_dir/seed-tts-eval/average_wer.py" \
    "$thread_dir/results/merge.out" "$score_file"

rm -f "$wav_ref_text"
rm -rf "$thread_dir"

echo "完成！结果保存在: $score_file"
