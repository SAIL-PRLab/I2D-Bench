#!/usr/bin/env python3
"""
通用结果合并工具：将WER/CER和Speaker Similarity结果合并到GPU JSON文件中
支持自动检测结果类型
"""
import sys
import json
import os

def parse_wer_result(result_file):
    """解析WER/CER结果文件"""
    stats = {}
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return stats, 'wer'

        # 跳过header行和数据行中的统计行(如 "WER: xx%")
        for line in lines:
            line = line.strip()
            # 跳过header行和统计行
            if not line or line.startswith("utt") or line.startswith("WER:"):
                continue

            parts = line.split('\t')
            if len(parts) >= 4:
                wav_path = parts[0]
                # 尝试解析WER值（可能是小数或百分比格式）
                try:
                    wer_value = float(parts[1])
                    if wer_value > 1:  # 如果是百分比格式(如 0.252 表示 0.252%)
                        wer_value = wer_value / 100
                except ValueError:
                    continue

                text_ref = parts[2] if len(parts) > 2 else ""
                text_res = parts[3] if len(parts) > 3 else ""
                ins = int(float(parts[4])) if len(parts) > 4 and parts[4] else 0
                dele = int(float(parts[5])) if len(parts) > 5 and parts[5] else 0
                sub = int(float(parts[6])) if len(parts) > 6 and parts[6] else 0

                # 从wav_path提取utt_id
                # 路径格式: .../Qwen3-TTS/{utt_id}/iter_N.wav
                # 需要提取 {utt_id} 部分，即文件名所在的目录名
                basename = os.path.basename(wav_path)
                dirname = os.path.dirname(wav_path)
                # 如果文件名是 iter_N.wav，则utt_id是目录名
                if basename.startswith('iter_'):
                    utt_id = os.path.basename(dirname)
                else:
                    utt_id = os.path.splitext(basename)[0]

                stats[utt_id] = {
                    'wer': wer_value,
                    'text_ref': text_ref,
                    'text_res': text_res,
                    'wer_insert': ins,
                    'wer_delete': dele,
                    'wer_replace': sub
                }
    except Exception as e:
        print(f"Error parsing WER file {result_file}: {e}", file=sys.stderr)

    return stats, 'wer'

def parse_sim_result(result_file):
    """解析Speaker Similarity结果文件"""
    stats = {}
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return stats, 'sim'

        # 跳过header行、统计行(如 "ASV: xx" 和 "ASV-var: xx")
        for line in lines:
            line = line.strip()
            # 跳过header行和统计行
            if not line or line.startswith("utt") or line.startswith("ASV"):
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                # 格式: "path1|path2\tscore"
                # path1 格式: .../{utt_id}/iter_N.wav_0_-1
                pair_path = parts[0]
                try:
                    similarity = float(parts[1])
                except ValueError:
                    continue

                # 从pair_path提取utt_id
                # 路径格式: .../Qwen3-TTS/{utt_id}/iter_N.wav_0_-1|...
                # 或者: .../{utt_id}/iter_N.wav\t...
                first_path = pair_path.split('|')[0]
                # 去掉 _0_-1 后缀
                first_path = first_path.replace('_0_-1', '')
                basename = os.path.basename(first_path)
                dirname = os.path.dirname(first_path)

                # 如果文件名是 iter_N.wav，则utt_id是目录名
                if basename.startswith('iter_'):
                    utt_id = os.path.basename(dirname)
                else:
                    utt_id = os.path.splitext(basename)[0]

                stats[utt_id] = {
                    'spk_similarity': similarity
                }
    except Exception as e:
        print(f"Error parsing similarity file {result_file}: {e}", file=sys.stderr)

    return stats, 'sim'

def detect_result_type(result_file):
    """自动检测结果文件类型"""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            
        # 根据header判断类型
        if 'res_wer' in first_line or 'text_ref' in first_line:
            return 'wer'
        elif 'similarity' in first_line or len(first_line.split('\t')) == 3:
            return 'sim'
    except:
        pass
    
    # 根据文件扩展名判断
    if result_file.endswith('.wer'):
        return 'wer'
    elif result_file.endswith('.sim'):
        return 'sim'
    
    return 'unknown'

def merge_to_json(json_file, result_file, output_file):
    """
    将结果合并到JSON文件中
    自动检测结果类型（WER或Similarity）
    """
    # 检测结果类型
    result_type = detect_result_type(result_file)
    
    if result_type == 'unknown':
        print(f"Error: Cannot determine result file type: {result_file}", file=sys.stderr)
        return False
    
    # 解析结果文件
    if result_type == 'wer':
        stats, _ = parse_wer_result(result_file)
    else:  # sim
        stats, _ = parse_sim_result(result_file)
    
    if not stats:
        print(f"Warning: No data found in {result_file}", file=sys.stderr)
        return False
    
    # 处理JSON文件
    updated_count = 0
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    key = data.get('key')
                    
                    # 如果有对应的数据，就合并进去
                    if key in stats:
                        info = stats[key]
                        if result_type == 'wer':
                            data['hyp_text'] = info['text_res']
                            data['cer'] = round(info['wer'], 6)
                            data['cer_insert'] = info['wer_insert']
                            data['cer_delete'] = info['wer_delete']
                            data['cer_replace'] = info['wer_replace']
                        else:  # sim
                            data['spk_similarity'] = round(info['spk_similarity'], 6)
                        updated_count += 1
                    
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON line: {e}", file=sys.stderr)
                    fout.write(line)
        
        return True
    except Exception as e:
        print(f"Error merging results to JSON: {e}", file=sys.stderr)
        return False

def main():
    if len(sys.argv) < 4:
        print("Usage: python merge_results_to_json.py <input_json> <result_file1> [result_file2 ...] <output_json>")
        print("  result_file can be .wer (CER/WER) or .sim (Speaker Similarity)")
        print("  Multiple result files will be merged sequentially")
        sys.exit(1)
    
    json_file = sys.argv[1]
    result_files = sys.argv[2:-1]  # 所有中间参数都是结果文件
    output_file = sys.argv[-1]
    
    # 检查输入文件
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}", file=sys.stderr)
        sys.exit(1)
    
    # 过滤存在的结果文件
    valid_result_files = []
    for result_file in result_files:
        if os.path.exists(result_file):
            valid_result_files.append(result_file)
    
    if not valid_result_files:
        print(f"Error: No valid result files found", file=sys.stderr)
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 依次合并所有结果文件
    current_input = json_file
    for idx, result_file in enumerate(valid_result_files):
        # 最后一个文件直接输出到目标文件，中间文件使用临时文件
        if idx == len(valid_result_files) - 1:
            current_output = output_file
        else:
            current_output = f"{json_file}.tmp{idx}"
        
        if not merge_to_json(current_input, result_file, current_output):
            sys.exit(1)
        
        # 如果使用了临时文件，下一次迭代使用临时文件作为输入
        if idx < len(valid_result_files) - 1:
            current_input = current_output

if __name__ == '__main__':
    main()
