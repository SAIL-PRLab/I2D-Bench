#!/bin/bash
# 单轮说话人相似度：WavLM + seed-tts-eval/thirdparty/UniSpeech/.../verification_pair_list_v2.py。
# 用法：cal_sim.sh <tts_model> <iter_num> <checkpoint_path> <data_dir> <output_dir> [num_job]
set -euo pipefail

tts_model=$1
iter_num=$2
checkpoint_path=$3
data_dir=$4
output_dir=$5
num_job=${6:-$(nvidia-smi -L | wc -l)}

project_dir=$(cd "$(dirname "$0")"; cd ../; pwd)

if [ -z "$tts_model" ] || [ -z "$iter_num" ] || [ -z "$checkpoint_path" ] || [ -z "$data_dir" ] || [ -z "$output_dir" ]; then
    echo "用法：./cal_sim.sh <tts_model> <iter_num> <checkpoint_path> <data_dir> <output_dir> [num_job]"
    exit 1
fi

target_text=$data_dir/target_text
ref_wav=$data_dir/ref_wav.scp
tts_output_scp=$data_dir/$tts_model/iter_${iter_num}.scp
wav_ref_text=$output_dir/wav_ref_text_iter_${iter_num}
score_file=$output_dir/score_iter_${iter_num}.sim

mkdir -p "$output_dir"

for f in "$target_text" "$ref_wav" "$tts_output_scp" "$checkpoint_path"; do
    [ -f "$f" ] || { echo "错误：缺少文件: $f"; exit 1; }
done

echo "使用 $num_job 个并行任务"

# 生成 verification_pair_list_v2.py 输入：wav_res|wav_ref|text_ref 三列
python3 "$project_dir/scripts/prepare_seed_file.py" \
    "$tts_output_scp" "$target_text" "$ref_wav" "$wav_ref_text"

# 收集可用 GPU id
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
if [ "${#gpu_ids[@]}" -eq 0 ]; then
    echo "错误：没有可用的 GPU"
    exit 1
fi

timestamp=$(date +%s)
thread_dir=/tmp/sim_${timestamp}_${tts_model}_iter${iter_num}
mkdir -p "$thread_dir"

total_lines=$(wc -l < "$wav_ref_text")
echo "共 $total_lines 条记录待处理"

if [ "$total_lines" -eq 0 ]; then
    echo "没有需要处理的记录"
    rm -rf "$thread_dir"
    exit 0
fi

lines_per_job=$(( (total_lines + num_job - 1) / num_job ))
split -l "$lines_per_job" --additional-suffix=.lst -d "$wav_ref_text" "$thread_dir/thread-"

echo "计算相似度分数..."
for ((i=0; i<num_job; i++)); do
    thread_file=$(printf "%s/thread-%02d.lst" "$thread_dir" "$i")
    sub_score_file=$(printf "%s/thread-%02d.sim.out" "$thread_dir" "$i")
    [ -f "$thread_file" ] || continue
    device_id=${gpu_ids[$i]}
    python3 "$project_dir/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py" \
        "$thread_file" \
        --model_name wavlm_large \
        --checkpoint "$checkpoint_path" \
        --scores "$sub_score_file" \
        --wav1_start_sr 0 --wav2_start_sr 0 \
        --wav1_end_sr -1 --wav2_end_sr -1 \
        --device "cuda:$device_id" &
done
wait

cat "$thread_dir"/thread-*.sim.out | grep -v "avg score" > "$thread_dir/merge.out"
python3 "$project_dir/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification/average.py" \
    "$thread_dir/merge.out" "$score_file"

rm -f "$wav_ref_text"
rm -rf "$thread_dir"

echo "完成！结果保存在: $score_file"
