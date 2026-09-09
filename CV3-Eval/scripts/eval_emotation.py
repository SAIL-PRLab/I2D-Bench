import os
import sys
import librosa
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from sklearn.metrics import classification_report

wav_scp = sys.argv[1]
output_file = sys.argv[2]

labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unk']

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_here, "..", ".."))
_local_model_dir = os.path.join(_project_root, "checkpoints", "emotion2vec_plus_large")
_local_weights = os.path.join(_local_model_dir, "model.pt")
if os.path.isfile(_local_weights):
    _model = _local_model_dir
    print(f"Using local emotion2vec_plus_large at {_model}", file=sys.stderr)
else:
    _model = "iic/emotion2vec_plus_large"
    print(f"WARN: {_local_weights} not found; falling back to hub id {_model}. "
          f"Run scripts/
    prepare_dataset.sh CV3-Eval to download it locally.", file=sys.stderr)

inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model=_model,
    # Disable remote code invocation entirely: with allow_remote=False
    # modelscope skips parsing the upstream repo's requirements.txt,
    # which is the proximate cause of the ParserSyntaxError on load.
    allow_remote=False)

hyp_high_list = []
ref_high_list = []

hyp_low_list = []
ref_low_list = []

with open(wav_scp, 'r') as f, open(output_file, 'w') as wf:
    for line in f:
        sc = '\t' if '\t' in line else ' '
        parts = line.strip().split(sc, 1)
        if len(parts) != 2:
            print(f"Skipping line: {line.strip()}")
            continue
        wid, path = parts
        
        try:
            y, sr = librosa.load(path, sr = 16000)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        rec_result = inference_pipeline(y, granularity="utterance", extract_embedding=False)
        scores = rec_result[0]['scores']
        hyp_emo = labels[scores.index(max(scores))]
        
        wid_parts = wid.split('_')
        if len(wid_parts) < 2:
            print(f"Skipping {wid}: invalid format")
            continue
            
        ref_emo, level = wid_parts[1], wid_parts[2]

        if level == 'high':
            hyp_high_list.append(hyp_emo)
            ref_high_list.append(ref_emo)
        else:
            hyp_low_list.append(hyp_emo)
            ref_low_list.append(ref_emo)

        wf.write(f"{wid}\t{ref_emo}\t{hyp_emo}\n")

def _safe_classification_report(name: str, y_true: list, y_pred: list) -> None:
    """Print a classification report; tolerate empty input arrays.
    sklearn>=1.4 raises ValueError on empty y_true/y_pred (the previous
    `zero_division=0` workaround no longer suppresses it), so guard
    against the case where one of the per-level buckets has no samples
    in this iter (e.g. --max_entries picked only 'high' rows)."""
    print(f"====Score for {name}====")
    if not y_true or not y_pred:
        print(f"  (skipped: no {name} samples in this iter; "
              f"y_true={len(y_true)}, y_pred={len(y_pred)})")
        return
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

_safe_classification_report("high", ref_high_list, hyp_high_list)
_safe_classification_report("low", ref_low_list, hyp_low_list)
_safe_classification_report(
    "all",
    ref_high_list + ref_low_list,
    hyp_high_list + hyp_low_list,
)
