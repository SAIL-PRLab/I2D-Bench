#!/bin/bash

stage_start=1
stage_end=6
TTS_Models="TTS"  # 支持逗号分隔的多个模型，如 "Model1,Model2,Model3"
CONFIG_FILE="configs/LibriTTS/test_clean/config.yaml"
TTS_CONFIG_FILE="configs/TTS_configs.yaml"
SCRIPT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export UTMOSV2_CHACHE="$SCRIPT_ROOT/versa/UTMOSv2"
MAX_ITERATIONS_OVERRIDE=""
MAX_ENTRIES=""
NUM_WORKERS=""

get_models_config() {
    local model=$1
    local key=$2
    python3 -c "
import yaml
with open('$TTS_CONFIG_FILE', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    value = config.get('$model', {}).get('$key', '')
    if value is None:
        value = ''
    print(value)
"
}

get_dataset_config() {
    local key=$1
    python3 -c "
import yaml
with open('$CONFIG_FILE', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    value = config.get('$key', '')
    if value is None:
        value = ''
    print(value)
"
}

# Return 0 iff json_file exists and every JSONL line contains the given key.
has_metric() {
    local file="$1" key="$2"
    [ -f "$file" ] || return 1
    python3 - "$file" "$key" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        if sys.argv[2] not in json.loads(line):
            sys.exit(1)
PYEOF
}


while [[ $# -gt 0 ]]; do
  case $1 in
    --stage)
      if [[ $2 =~ ^([0-9]+)-([0-9]+)$ ]]; then
        stage_start=${BASH_REMATCH[1]}
        stage_end=${BASH_REMATCH[2]}
      elif [[ $2 =~ ^[0-9]+$ ]]; then
        stage_start=$2
        stage_end=$2
      else
        echo "Error: Invalid stage format. Use --stage N or --stage N-M"
        exit 1
      fi
      shift 2
      ;;
    --TTS_Models)
      TTS_Models="$2"  # 支持逗号分隔的多个模型
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --max_iterations)
      if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: Invalid max_iterations format. Use a positive integer."
        exit 1
      fi
      MAX_ITERATIONS_OVERRIDE="$2"
      shift 2
      ;;
    --max_entries)
      if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: Invalid max_entries format. Use a positive integer."
        exit 1
      fi
      MAX_ENTRIES="$2"
      shift 2
      ;;
    --num_workers)
      if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: Invalid num_workers format. Use a positive integer."
        exit 1
      fi
      NUM_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

if [ ! -f "$TTS_CONFIG_FILE" ]; then
    echo "Error: TTS Config file '$TTS_CONFIG_FILE' not found!"
    exit 1
fi

if ! python3 -c "import yaml" 2>/dev/null; then
    echo "Error: PyYAML is not installed. Please run: pip install pyyaml"
    exit 1
fi

IFS=',' read -ra MODEL_ARRAY <<< "$TTS_Models"

echo "======================================"
echo "Will process ${#MODEL_ARRAY[@]} model(s): ${MODEL_ARRAY[*]}"
echo "======================================"

TARGET_TEXT=$(get_dataset_config "target_text")
REF_WAV=$(get_dataset_config "ref_wav")
REF_TEXT=$(get_dataset_config "ref_text")
OUTPUT_BASE=$(get_dataset_config "output_base")
RESULTS_BASE=$(get_dataset_config "results_base")
MAX_ITERATIONS=$(get_dataset_config "max_iterations")
if [ -n "$MAX_ITERATIONS_OVERRIDE" ]; then
  MAX_ITERATIONS="$MAX_ITERATIONS_OVERRIDE"
fi
METRICS=$(get_dataset_config "metrics")
METRICS_CONFIG=$(get_dataset_config "metrics_config")
LANGUAGE=$(get_dataset_config "language")
WAVLM_CHECKPOINT=$(get_dataset_config "wavlm_checkpoint")
RUN_EVAL_EMOTION=$(get_dataset_config "run_eval_emotion")

for TTS_Model in "${MODEL_ARRAY[@]}"; do
    TTS_Model=$(echo "$TTS_Model" | xargs)
    
    VENV_PATH=$(get_models_config "$TTS_Model" "venv_path")
    SCRIPT_PATH=$(get_models_config "$TTS_Model" "script_path")
    MODEL_PATH=$(get_models_config "$TTS_Model" "model_path")

    if [ -z "$VENV_PATH" ] || [ -z "$SCRIPT_PATH" ]; then
        echo "Error: TTS Model '$TTS_Model' not found or incomplete in config file!"
        echo "Skipping model '$TTS_Model'..."
        continue
    fi

    echo "======================================"
    echo "Processing Model: $TTS_Model"
    echo "  Virtual Env: $VENV_PATH"
    echo "  Script: $SCRIPT_PATH"
    echo "  Stage: $stage_start to $stage_end"
    echo "======================================"

    # Resolve NUM_WORKERS once per model so it can be reused across stages 1, 3, 4.
    if [ -z "$NUM_WORKERS" ]; then
        NUM_WORKERS="$(python -c 'import torch; print(max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1)')"
    fi
    echo "  Num workers: $NUM_WORKERS"

    if [[ $stage_start -le 1 && $stage_end -ge 1 ]]; then
        echo "Stage 1: Running TTS synthesis..."
        if [ -f "$VENV_PATH" ]; then
            . "$VENV_PATH"
        else
            echo "Warning: Virtual environment not found at $VENV_PATH, continuing without activation..."
        fi


        tts_args=(
            --ref_wav "$REF_WAV"
            --ref_text "$REF_TEXT"
            --target_text "$TARGET_TEXT"
            --output "$OUTPUT_BASE/$TTS_Model"
            --max_iterations "$MAX_ITERATIONS"
            --num_workers "$NUM_WORKERS"
            --model_path "$MODEL_PATH"
        )
        if [ -n "$MAX_ENTRIES" ]; then
            tts_args+=(--max_entries "$MAX_ENTRIES")
        fi
        python "$SCRIPT_PATH" "${tts_args[@]}"
        if [ -f "$VENV_PATH" ]; then
            deactivate
        fi
    fi

    if [[ $stage_start -le 2 && $stage_end -ge 2 ]]; then
        echo "Stage 2: Generating SCP files."
        for i in $(seq 1 $MAX_ITERATIONS)
        do
            python utils/gen_iter_scp.py \
                --folder "$OUTPUT_BASE/$TTS_Model" \
                --iter $i 
        done
    fi

    if [[ $stage_start -le 3 && $stage_end -ge 3 ]]; then
        echo "Stage 3: Calculating all metrics."
        mkdir -p "$RESULTS_BASE/$TTS_Model"

        SPLIT_SIZE=$NUM_WORKERS
        echo "  Num workers: ${NUM_WORKERS} (SPLIT_SIZE=${SPLIT_SIZE}, --max-parallel=${SPLIT_SIZE})"

        for i in $(seq 1 $MAX_ITERATIONS)
        do
        cd versa
        ./launch_local.sh \
            "../$OUTPUT_BASE/$TTS_Model/iter_$i.scp" \
            "../$REF_WAV" \
            "../$RESULTS_BASE/$TTS_Model/iter_$i" \
            "${SPLIT_SIZE}" \
            --max-parallel="${SPLIT_SIZE}" \
            --text="../$TARGET_TEXT" \
            --config="../$METRICS_CONFIG"

        cd ..
        bash scripts/merge_results.sh \
            "$RESULTS_BASE/$TTS_Model/iter_$i/result" \
            "$RESULTS_BASE/$TTS_Model/iter_$i"
        done
    fi 

    if [[ $stage_start -le 4 && $stage_end -ge 4 ]]; then
        echo "Stage 4: Computing CER and Speaker Similarity if applicable."
        # 检查language
        if [[ "$LANGUAGE" == "zh" ]]; then
            echo "Detected Chinese test set, calculating CER and Speaker Similarity..."
            mkdir -p "$RESULTS_BASE/$TTS_Model"
            
            for i in $(seq 1 $MAX_ITERATIONS)
            do
                json_file="$RESULTS_BASE/$TTS_Model/iter_$i.gpu.txt"
                result_files=()
                
                skip_cer=false
                has_metric "$json_file" cer && skip_cer=true

                # CER
                if [ "$skip_cer" = "true" ]; then
                    echo "  [Iter $i] CER already exists, skipping calculation."
                else
                    echo "  [Iter $i] Calculating CER..."
                    bash scripts/cal_wer.sh "$TTS_Model" "$i" "$LANGUAGE" "$OUTPUT_BASE" "$RESULTS_BASE/$TTS_Model" "$NUM_WORKERS"

                    cer_result_file="$RESULTS_BASE/$TTS_Model/score_iter_${i}.wer"
                    if [ -f "$cer_result_file" ]; then
                        result_files+=("$cer_result_file")
                    fi
                fi
                
                skip_sim=false
                has_metric "$json_file" spk_similarity && skip_sim=true

                # Speaker Similarity
                if [ -n "$WAVLM_CHECKPOINT" ] && [ -f "$WAVLM_CHECKPOINT" ]; then
                    if [ "$skip_sim" = "true" ]; then
                        echo "  [Iter $i] Speaker Similarity already exists, skipping calculation."
                    else
                        echo "  [Iter $i] Calculating Speaker Similarity..."
                        bash scripts/cal_sim.sh "$TTS_Model" "$i" "$WAVLM_CHECKPOINT" "$OUTPUT_BASE" "$RESULTS_BASE/$TTS_Model" "$NUM_WORKERS"

                        sim_result_file="$RESULTS_BASE/$TTS_Model/score_iter_${i}.sim"
                        if [ -f "$sim_result_file" ]; then
                            result_files+=("$sim_result_file")
                        fi
                    fi
                else
                    echo "  [Iter $i] Skipping Speaker Similarity (checkpoint not configured or not found)"
                fi
                
                # 合并 CER/SIM 结果
                if [ ${#result_files[@]} -gt 0 ] && [ -f "$json_file" ]; then
                    echo "  [Iter $i] Merging results..."
                    python utils/merge_results_to_json.py \
                        "$json_file" \
                        "${result_files[@]}" \
                        "$json_file.tmp"
                    
                    if [ -f "$json_file.tmp" ]; then
                        mv "$json_file.tmp" "$json_file"
                    fi
                    rm -f "$json_file.tmp0"
                fi
                
                rm -f "$cer_result_file" "$sim_result_file"
            done
        else
            echo "Skipping CER and Speaker Similarity calculation (not a Chinese test set)"
        fi
    fi

    if [[ $stage_start -le 5 && $stage_end -ge 5 ]]; then
        echo "Stage 5: Evaluating Emotion (CV3-Eval) if applicable."
        if [ "$RUN_EVAL_EMOTION" = "True" ]; then
            if [ -f "$VENV_PATH" ]; then
                . "$VENV_PATH"
            else
                echo "Warning: Virtual environment not found at $VENV_PATH, continuing without activation..."
            fi

            for i in $(seq 1 $MAX_ITERATIONS)
            do
            echo "  [Iter $i] Calculating Emotion Score..."
            scp_file="$OUTPUT_BASE/$TTS_Model/iter_$i.scp"
            output_dir="$RESULTS_BASE/$TTS_Model/iter_$i"
            mkdir -p "$output_dir"
            
            if [ -f "$scp_file" ]; then
                    python CV3-Eval/scripts/eval_emotation.py \
                        "$scp_file" \
                        "$output_dir/emo_result.txt" > "$output_dir/emo_score.txt"
            else
                    echo "Warning: $scp_file not found."
            fi
            done

            if [ -f "$VENV_PATH" ]; then
                deactivate
            fi
        else
            echo "Skipping Emotion Evaluation (run_eval_emotion not true)"
        fi
    fi

    if [[ $stage_start -le 6 && $stage_end -ge 6 ]]; then
        echo "Stage 6: Calculating overall score summary."
        python utils/calculate_score.py \
            --input_dir "$RESULTS_BASE/$TTS_Model" \
            --metrics $METRICS
    fi

    echo "======================================"
    echo "Finished processing model: $TTS_Model"
    echo "======================================"

done

echo ""
echo "======================================"
echo "All models processed successfully!"
echo "======================================"
