#!/usr/bin/env python3
"""把 iter_<N>.scp + target_text + ref_wav.scp 合并为
seed-tts-eval/run_wer.py（及 speaker_verification）所需的 wav_res|wav_ref|text_ref 三列文件。
"""

import sys
import os


def load_scp(path):
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <iter_scp> <target_text> <ref_wav_scp> <output_txt>")
        sys.exit(1)

    iter_scp, target_text_path, ref_wav_path, output_path = sys.argv[1:5]
    for f in (iter_scp, target_text_path, ref_wav_path):
        if not os.path.exists(f):
            print(f"Error: {f} not found")
            sys.exit(1)

    text_dict = load_scp(target_text_path)
    ref_dict = load_scp(ref_wav_path)

    written = 0
    with open(iter_scp, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, wav_res_path = parts
            if key not in text_dict or key not in ref_dict:
                print(f"Warning: key '{key}' missing in target_text/ref_wav.scp, skipping", file=sys.stderr)
                continue
            f_out.write(f"{wav_res_path}|{ref_dict[key]}|{text_dict[key]}\n")
            written += 1

    print(f"Wrote {written} entries to: {output_path}")


if __name__ == "__main__":
    main()
