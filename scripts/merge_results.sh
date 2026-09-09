#!/bin/bash
# 把 per-GPU / per-CPU 的分片结果合并成一个文件。
# 用法：merge_results.sh <result_dir> <output_file>
set -euo pipefail

RESULT_DIR=$1
OUTPUT_FILE=$2

if [ $# -ne 2 ]; then
    echo "用法：$0 <result_dir> <output_file>"
    exit 1
fi

echo "Merging results from ${RESULT_DIR} to ${OUTPUT_FILE}"

for suffix in gpu cpu; do
    pattern="${RESULT_DIR}"/*.result.${suffix}.txt
    if ls $pattern 1>/dev/null 2>&1; then
        echo "Merging ${suffix} results..."
        cat $pattern > "${OUTPUT_FILE%.txt}.${suffix}.txt"
    fi
done

echo "Results merged successfully!"
