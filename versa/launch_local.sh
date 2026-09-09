#!/bin/bash
#
# Local Launcher for VERSA Processing
# -------------------------------------------
# This script splits input audio files and launches parallel processes locally
# using either GPU, CPU, or both resources based on user selection.
#
# Usage: ./launch_local.sh <pred_wavscp> <gt_wavscp|None> <score_dir> <split_size> [--config=FILE] [--max-parallel=N] [--text=FILE]
#   <pred_wavscp>: Path to prediction wav.scp file
#   <gt_wavscp|None>: Path to ground truth wav.scp file, or "None" for reference-free
#   <score_dir>: Directory to store results
#   <split_size>: Number of chunks to split the data into
#   --config=FILE: Metrics config file (default: metrics.yaml at repo root)
#   --max-parallel=N: Max number of parallel jobs (default: 4)
#   --text=FILE: Path to text file to be processed (optional)
#
# Example: ./launch_local.sh data/pred.scp data/gt.scp results/experiment1 10 --config=metrics.yaml --max-parallel=8
# Example: ./launch_local.sh data/pred.scp None results/experiment1 16 --config=metrics.yaml --text=data/transcripts.txt

set -e  # Exit immediately if a command exits with non-zero status

# Define color codes for output messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Resolve project root (parent of this script). Used as the base for turning
# project-relative wav paths inside wav.scp files into absolute paths, so the
# downstream scorer works regardless of the working directory it is launched
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Function to display usage
show_usage() {
    echo -e "${BLUE}Usage: $0 <pred_wavscp> <gt_wavscp|None> <score_dir> <split_size> [--config=FILE] [--max-parallel=N] [--text=FILE]${NC}"
    echo -e "  <pred_wavscp>: Path to prediction wav script file"
    echo -e "  <gt_wavscp|None>: Path to ground truth wav script file (use \"None\" if not available)"
    echo -e "  <score_dir>: Directory to store results"
    echo -e "  <split_size>: Number of chunks to split the data into"
    echo -e "  --config=FILE: Metrics config file (default: metrics.yaml at repo root)"
    echo -e "  --max-parallel=N: Maximum number of parallel processes (default: 4)"
    echo -e "  --text=FILE: Path to text file to be processed (optional)"
}

# Check for minimum required arguments
if [ $# -lt 4 ]; then
    echo -e "${RED}Error: Insufficient arguments${NC}"
    show_usage
    exit 1
fi

# Parse command line arguments
PRED_WAVSCP=$1
GT_WAVSCP=$2
SCORE_DIR=$3
SPLIT_SIZE=$4
IO_TYPE=${IO_TYPE:-soundfile}

# Default settings
MAX_PARALLEL=4  # Default to 4
TEXT_FILE=""  # Optional text file
CONFIG_FILE="/mnt/data/share-oss/shenshengfan/Project/Benchmark/metrics.yaml"  # Default config path

# Parse optional arguments
shift 4
while [[ $# -gt 0 ]]; do
    case $1 in
        --config=*)
            CONFIG_FILE="${1#*=}"
            if [ ! -f "${CONFIG_FILE}" ]; then
                echo -e "${RED}Error: Config file '${CONFIG_FILE}' not found${NC}"
                exit 1
            fi
            echo -e "${YELLOW}Config file set to: ${CONFIG_FILE}${NC}"
            shift
            ;;
        --max-parallel=*)
            MAX_PARALLEL="${1#*=}"
            if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || [ "${MAX_PARALLEL}" -eq 0 ]; then
                echo -e "${RED}Error: max-parallel must be a positive integer${NC}"
                exit 1
            fi
            echo -e "${YELLOW}Maximum parallel processes set to: ${MAX_PARALLEL}${NC}"
            shift
            ;;
        --text=*)
            TEXT_FILE="${1#*=}"
            if [ ! -f "${TEXT_FILE}" ]; then
                echo -e "${RED}Error: Text file '${TEXT_FILE}' not found${NC}"
                exit 1
            fi
            echo -e "${YELLOW}Text file set to: ${TEXT_FILE}${NC}"
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ! -f "${PRED_WAVSCP}" ]; then
    echo -e "${RED}Error: Prediction wav script file '${PRED_WAVSCP}' not found${NC}"
    exit 1
fi

if [ "${GT_WAVSCP}" != "None" ] && [ ! -f "${GT_WAVSCP}" ]; then
    echo -e "${RED}Error: Ground truth wav script file '${GT_WAVSCP}' not found${NC}"
    exit 1
fi

if ! [[ "${SPLIT_SIZE}" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Split size must be a positive integer${NC}"
    exit 1
fi

# Check for GPU availability and set up GPU rank management
GPU_COUNT=0
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU processing may not work properly.${NC}"
    echo -e "${YELLOW}Continuing anyway... defaulting to 1 GPU.${NC}"
    GPU_COUNT=1
else
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    if [ "${GPU_COUNT}" -lt 1 ]; then
        echo -e "${YELLOW}Warning: No GPUs reported by nvidia-smi. Defaulting to 1.${NC}"
        GPU_COUNT=1
    else
        echo -e "${GREEN}Found ${GPU_COUNT} GPU(s)${NC}"
        # Display GPU information
        echo -e "${BLUE}Available GPUs:${NC}"
        nvidia-smi -L | while IFS= read -r line; do
            echo -e "  ${line}"
        done
    fi
fi

# Overall parallel limit (across all GPUs)
OVERALL_MAX_PARALLEL=${MAX_PARALLEL}

# Print configuration summary
echo -e "${BLUE}=== Configuration Summary ===${NC}"
echo -e "Prediction WAV script: ${PRED_WAVSCP}"
echo -e "Ground truth WAV script: ${GT_WAVSCP}"
echo -e "Output directory: ${SCORE_DIR}"
echo -e "Split size: ${SPLIT_SIZE}"
echo -e "Maximum parallel processes: ${MAX_PARALLEL}"
echo -e "Config file: ${CONFIG_FILE}"
if [ -n "${TEXT_FILE}" ]; then
    echo -e "Text file: ${TEXT_FILE}"
else
    echo -e "Text file: Not provided"
fi
echo -e "GPU processing: Enabled (${GPU_COUNT} GPUs available)"
echo ""

# Create directory structure
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p "${SCORE_DIR}"
mkdir -p "${SCORE_DIR}/pred"
mkdir -p "${SCORE_DIR}/gt"
mkdir -p "${SCORE_DIR}/text"
mkdir -p "${SCORE_DIR}/result"
mkdir -p "${SCORE_DIR}/logs"

# Filter inputs to keep only entries present in all provided files
FILTERED_PRED_WAVSCP="${SCORE_DIR}/pred_filtered.wavscp"
FILTERED_GT_WAVSCP="${SCORE_DIR}/gt_filtered.wavscp"
FILTERED_TEXT_FILE="${SCORE_DIR}/text_filtered.txt"

python - <<PY
import sys
from pathlib import Path

pred_path = Path("${PRED_WAVSCP}")
gt_arg = "${GT_WAVSCP}"
text_arg = "${TEXT_FILE}"
project_root = Path("${PROJECT_ROOT}")

gt_path = None if gt_arg == "None" else Path(gt_arg) if gt_arg else None
text_path = None if not text_arg else Path(text_arg)

out_pred = Path("${FILTERED_PRED_WAVSCP}")
out_gt = Path("${FILTERED_GT_WAVSCP}") if gt_path else None
out_text = Path("${FILTERED_TEXT_FILE}") if text_path else None

def resolve_path(p: str) -> str:
    """Resolve a wav.scp path token to an absolute path.
    Absolute paths are kept; project-relative paths are resolved against the
    project root computed by launch_local.sh."""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((project_root / pp).resolve())

def load_scp(path: Path):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(None, 1)
            if not parts:
                continue
            key = parts[0]
            if len(parts) >= 2:
                data[key] = f"{key} {resolve_path(parts[1])}"
            else:
                data[key] = line
    return data

def load(path: Path):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(None, 1)
            if not parts:
                continue
            key = parts[0]
            data[key] = line
    return data

pred = load_scp(pred_path)
gt = load_scp(gt_path) if gt_path else {}
txt = load(text_path) if text_path else {}

common = set(pred)
if gt_path:
    common &= set(gt)
if text_path:
    common &= set(txt)

if not common:
    sys.stderr.write("Error: No overlapping entries across provided files.\n")
    sys.exit(1)

for path in (out_pred.parent, out_gt.parent if out_gt else None, out_text.parent if out_text else None):
    if path:
        path.mkdir(parents=True, exist_ok=True)

keys = sorted(common)

with out_pred.open("w", encoding="utf-8") as f:
    for key in keys:
        f.write(pred[key] + "\n")

if gt_path:
    with out_gt.open("w", encoding="utf-8") as f:
        for key in keys:
            f.write(gt[key] + "\n")

if text_path:
    with out_text.open("w", encoding="utf-8") as f:
        for key in keys:
            f.write(txt[key] + "\n")

print(f"Keeping {len(common)} aligned entries.")
print(f"Prediction total: {len(pred)}")
if gt_path:
    print(f"Ground truth total: {len(gt)}")
if text_path:
    print(f"Text total: {len(txt)}")
PY

# Use filtered files for the rest of the pipeline
PRED_WAVSCP="${FILTERED_PRED_WAVSCP}"
if [ "${GT_WAVSCP}" != "None" ]; then
    GT_WAVSCP="${FILTERED_GT_WAVSCP}"
fi
if [ -n "${TEXT_FILE}" ]; then
    TEXT_FILE="${FILTERED_TEXT_FILE}"
fi

# Split prediction files
total_lines=$(wc -l < "${PRED_WAVSCP}")
source_wavscp=$(basename "${PRED_WAVSCP}")
lines_per_piece=$(( (total_lines + SPLIT_SIZE - 1) / SPLIT_SIZE ))  # Ceiling division

echo -e "${GREEN}Splitting ${total_lines} lines into ${SPLIT_SIZE} pieces (${lines_per_piece} lines per piece)...${NC}"
split -l "${lines_per_piece}" -d -a 3 "${PRED_WAVSCP}" "${SCORE_DIR}/pred/${source_wavscp}_"
pred_list=("${SCORE_DIR}/pred/${source_wavscp}_"*)

# Split ground truth files if provided
if [ "${GT_WAVSCP}" = "None" ]; then
    echo -e "${YELLOW}No ground truth audio provided, evaluation will be reference-free${NC}"
    gt_list=()
else
    target_wavscp=$(basename "${GT_WAVSCP}")
    split -l "${lines_per_piece}" -d -a 3 "${GT_WAVSCP}" "${SCORE_DIR}/gt/${target_wavscp}_"
    gt_list=("${SCORE_DIR}/gt/${target_wavscp}_"*)

    if [ ${#pred_list[@]} -ne ${#gt_list[@]} ]; then
        echo -e "${RED}Error: The number of split ground truth (${#gt_list[@]}) and predictions (${#pred_list[@]}) does not match.${NC}"
        exit 1
    fi
fi

# Split text file if provided
text_list=()
if [ -n "${TEXT_FILE}" ]; then
    echo -e "${GREEN}Splitting text file...${NC}"
    text_basename=$(basename "${TEXT_FILE}")
    split -l "${lines_per_piece}" -d -a 3 "${TEXT_FILE}" "${SCORE_DIR}/text/${text_basename}_"
    text_list=("${SCORE_DIR}/text/${text_basename}_"*)

    if [ ${#pred_list[@]} -ne ${#text_list[@]} ]; then
        echo -e "${RED}Error: The number of split text files (${#text_list[@]}) and predictions (${#pred_list[@]}) does not match.${NC}"
        exit 1
    fi
fi

# Function to run a single processing job (GPU only)
run_job() {
    local sub_pred_wavscp=$1
    local sub_gt_wavscp=$2
    local sub_text_file=$3
    local output_file=$4
    local config_file=$5
    local job_prefix=$6
    local chunk_info=$7
    local gpu_rank=${8:-""}  # GPU rank
    
    local log_file="${SCORE_DIR}/logs/gpu_${job_prefix}_$(date +%s).log"
    
    if [ -n "${gpu_rank}" ]; then
        echo -e "${BLUE}Starting GPU job for ${chunk_info}: ${job_prefix} (GPU ${gpu_rank})${NC}"
    else
        echo -e "${BLUE}Starting GPU job for ${chunk_info}: ${job_prefix}${NC}"
    fi
    
    # Pin each parallel scorer process to its assigned GPU via
    # CUDA_VISIBLE_DEVICES, so concurrent jobs do not all default to
    # device 0 and OOM each other. Without this, every parallel
    # process computes gpu_rank = args.rank % torch.cuda.device_count()
    # and calls torch.cuda.set_device(0), even when multiple GPUs
    # are physically available.
    local cuda_visible=""
    if [ -n "${gpu_rank}" ]; then
        cuda_visible="${gpu_rank}"
        echo -e "${BLUE}Starting GPU job for ${chunk_info}: ${job_prefix} (GPU ${gpu_rank})${NC}"
        CUDA_VISIBLE_DEVICES="${cuda_visible}" ./egs/run_gpu.sh \
            "${sub_pred_wavscp}" \
            "${sub_gt_wavscp}" \
            "${output_file}" \
            "${config_file}" \
            "${IO_TYPE}" \
            "${sub_text_file}" \
            "${gpu_rank}" > "${log_file}" 2>&1
    else
        echo -e "${BLUE}Starting GPU job for ${chunk_info}: ${job_prefix}${NC}"
        ./egs/run_gpu.sh \
            "${sub_pred_wavscp}" \
            "${sub_gt_wavscp}" \
            "${output_file}" \
            "${config_file}" \
            "${IO_TYPE}" \
            "${sub_text_file}" > "${log_file}" 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Completed GPU job for ${chunk_info}: ${job_prefix} (GPU ${gpu_rank})${NC}"
    else
        echo -e "${RED}Failed GPU job for ${chunk_info}: ${job_prefix} (GPU ${gpu_rank}) (exit code: ${exit_code})${NC}"
        echo -e "${RED}Check log file: ${log_file}${NC}"
    fi
    
    return $exit_code
}

# Job tracking
declare -a gpu_job_pids=()
declare -a gpu_job_info=()

RUNNING_JOBS_FILE="${SCORE_DIR}/running_jobs.txt"
COMPLETED_JOBS_FILE="${SCORE_DIR}/completed_jobs.txt"
FAILED_JOBS_FILE="${SCORE_DIR}/failed_jobs.txt"

# Clear tracking files
> "${RUNNING_JOBS_FILE}"
> "${COMPLETED_JOBS_FILE}"
> "${FAILED_JOBS_FILE}"

echo -e "${GREEN}Starting parallel processing...${NC}"

# Function to enforce overall parallel limit
wait_for_slot() {
    local idx
    while [ ${#gpu_job_pids[@]} -ge $OVERALL_MAX_PARALLEL ]; do
        # Check for completed GPU jobs
        for idx in "${!gpu_job_pids[@]}"; do
            if ! kill -0 "${gpu_job_pids[$idx]}" 2>/dev/null; then
                # Job has finished
                wait "${gpu_job_pids[$idx]}"
                local exit_code=$?
                
                if [ $exit_code -eq 0 ]; then
                    echo "${gpu_job_info[$idx]}" >> "${COMPLETED_JOBS_FILE}"
                else
                    echo "${gpu_job_info[$idx]}" >> "${FAILED_JOBS_FILE}"
                fi
                
                # Remove from tracking arrays
                unset gpu_job_pids[$idx]
                unset gpu_job_info[$idx]
                
                # Rebuild arrays to remove gaps
                gpu_job_pids=("${gpu_job_pids[@]}")
                gpu_job_info=("${gpu_job_info[@]}")
                break
            fi
        done
        sleep 1
    done
}

# Submit jobs
for ((i=0; i<${#pred_list[@]}; i++)); do
    sub_pred_wavscp=${pred_list[${i}]}
    job_prefix=$(basename "${sub_pred_wavscp}")
    chunk_info="$((i+1))/${#pred_list[@]}"

    if [ "${GT_WAVSCP}" = "None" ]; then
        sub_gt_wavscp="None"
    else
        sub_gt_wavscp=${gt_list[${i}]}
    fi
    
    # Set text file for this chunk
    if [ -n "${TEXT_FILE}" ]; then
        sub_text_file=${text_list[${i}]}
    else
        sub_text_file=""
    fi
    
    echo -e "${BLUE}Processing chunk ${chunk_info}: ${sub_pred_wavscp}${NC}"
    if [ -n "${sub_text_file}" ]; then
        echo -e "${BLUE}  Text file: ${sub_text_file}${NC}"
    fi

    # Determine expected and existing result lines to decide skipping
    output_file="${SCORE_DIR}/result/$(basename "${sub_pred_wavscp}").result.gpu.txt"
    expected_lines=$(wc -l < "${sub_pred_wavscp}")
    if [ -f "${output_file}" ]; then
        # Prefer counting non-empty lines, fallback to total
        existing_lines=$(grep -c '^[^[:space:]]' "${output_file}" 2>/dev/null || wc -l < "${output_file}")
        if [ "${existing_lines}" -eq "${expected_lines}" ]; then
            echo -e "${YELLOW}Skipping chunk ${chunk_info}: existing results match (${existing_lines}/${expected_lines})${NC}"
            continue
        else
            echo -e "${YELLOW}Existing result found but line count mismatch (${existing_lines}/${expected_lines}), re-evaluating...${NC}"
        fi
    fi

    # Enforce overall parallel limit
    wait_for_slot

    # Round-robin assign GPU rank evenly
    gpu_rank=$(( i % GPU_COUNT ))

    run_job \
        "${sub_pred_wavscp}" \
        "${sub_gt_wavscp}" \
        "${sub_text_file}" \
        "${output_file}" \
        "${CONFIG_FILE}" \
        "${job_prefix}" \
        "${chunk_info}" \
        "${gpu_rank}" &

    gpu_pid=$!
    gpu_job_pids+=($gpu_pid)
    gpu_job_info+=("GPU:$gpu_pid GPU_RANK:${gpu_rank} CHUNK:${chunk_info} FILE:${job_prefix}")
    echo "GPU:$gpu_pid GPU_RANK:${gpu_rank} CHUNK:${chunk_info} FILE:${job_prefix}" >> "${RUNNING_JOBS_FILE}"
    echo -e "  Started GPU job: PID ${gpu_pid} on GPU ${gpu_rank}"
done

# Wait for all remaining jobs to complete
echo -e "${YELLOW}Waiting for all jobs to complete...${NC}"

# Wait for GPU jobs
for pid in "${gpu_job_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
        exit_code=$?
        
        # Find job info for this PID
        for info in "${gpu_job_info[@]}"; do
            if [[ "$info" == *":$pid "* ]]; then
                if [ $exit_code -eq 0 ]; then
                    echo "$info" >> "${COMPLETED_JOBS_FILE}"
                else
                    echo "$info" >> "${FAILED_JOBS_FILE}"
                fi
                break
            fi
        done
    fi
done

# No CPU jobs to wait for

# Generate summary
echo -e "${GREEN}=== Processing Summary ===${NC}"
if [ -f "${COMPLETED_JOBS_FILE}" ]; then
    completed_count=$(wc -l < "${COMPLETED_JOBS_FILE}" 2>/dev/null || echo "0")
    echo -e "${GREEN}Completed jobs: ${completed_count}${NC}"
    
    # Show GPU usage summary
    if [ -s "${COMPLETED_JOBS_FILE}" ]; then
        echo -e "${GREEN}GPU usage summary:${NC}"
        for ((rank=0; rank<GPU_COUNT; rank++)); do
            gpu_job_count=$(grep -c "GPU_RANK:${rank}" "${COMPLETED_JOBS_FILE}" 2>/dev/null || echo "0")
            echo -e "  GPU ${rank}: ${gpu_job_count} jobs completed"
        done
    fi
fi

if [ -f "${FAILED_JOBS_FILE}" ] && [ -s "${FAILED_JOBS_FILE}" ]; then
    failed_count=$(wc -l < "${FAILED_JOBS_FILE}")
    echo -e "${RED}Failed jobs: ${failed_count}${NC}"
    echo -e "${RED}Failed job details:${NC}"
    cat "${FAILED_JOBS_FILE}"
fi

# Create merge script
MERGE_SCRIPT="${SCORE_DIR}/merge_results.sh"
cat > "${MERGE_SCRIPT}" << 'EOF'
#!/bin/bash
# Merge results script
RESULT_DIR=$1
OUTPUT_FILE=$2

if [ $# -ne 2 ]; then
    echo "Usage: $0 <result_dir> <output_file>"
    exit 1
fi

echo "Merging results from ${RESULT_DIR} to ${OUTPUT_FILE}"

# Merge GPU results if they exist
if ls "${RESULT_DIR}"/*.result.gpu.txt 1> /dev/null 2>&1; then
    echo "Merging GPU results..."
    cat "${RESULT_DIR}"/*.result.gpu.txt > "${OUTPUT_FILE%.txt}.gpu.txt"
fi

# CPU results removed in GPU-only mode

echo "Results merged successfully!"
EOF

chmod +x "${MERGE_SCRIPT}"

echo -e "${YELLOW}To merge all results, run:${NC}"
echo -e "${MERGE_SCRIPT} ${SCORE_DIR}/result ${SCORE_DIR}/final_results.txt"

echo -e "${GREEN}All processing completed!${NC}"
echo -e "Logs are available in: ${SCORE_DIR}/logs/"
echo -e "Results are available in: ${SCORE_DIR}/result/"
