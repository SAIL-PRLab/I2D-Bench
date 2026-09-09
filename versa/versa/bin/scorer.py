#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Scorer Interface for Speech Evaluation."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import logging
logging.basicConfig(level=logging.INFO)
import torch
import yaml

from versa.scorer_shared import (
    audio_loader_setup,
    corpus_scoring,
    list_scoring,
    load_corpus_modules,
    load_score_modules,
    load_summary,
)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Speech Evaluation Interface")
    parser.add_argument(
        "--pred",
        type=str,
        help="Wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "--score_config", type=str, default=None, help="Configuration of Score Config"
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Wav.scp for ground truth waveforms.",
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Path of ground truth transcription."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path of directory to write the results.",
    )
    parser.add_argument(
        "--cache_folder", type=str, default=None, help="Path of cache saving"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=False, help="whether to use GPU if it can"
    )
    parser.add_argument(
        "--io",
        type=str,
        default="kaldi",
        choices=["kaldi", "soundfile", "dir"],
        help="io interface to use",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    parser.add_argument(
        "--rank",
        default=0,
        type=int,
        help="the overall rank in the batch processing, used to specify GPU rank",
    )
    parser.add_argument(
        "--no_match",
        action="store_true",
        help="Do not match the groundtruth and generated files.",
    )
    # Parallelization via sharding: run multiple processes with different shard ids
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards to divide the input list into.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Shard index for this process (0-based).",
    )
    parser.add_argument(
        "--corpus_on_shards",
        action="store_true",
        help="If set, compute corpus-level metrics on each shard separately. By default, corpus-level metrics are skipped when sharding is enabled.",
    )
    return parser


def main():
    args = get_parser().parse_args()

    # In case of using `local` backend, all GPU will be visible to all process.
    if args.use_gpu:
        gpu_rank = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_rank)
        logging.info(f"using device: cuda:{gpu_rank}")

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    gen_files = audio_loader_setup(args.pred, args.io)

    # find reference file
    args.gt = None if args.gt == "None" else args.gt
    if args.gt is not None and not args.no_match:
        gt_files = audio_loader_setup(args.gt, args.io)
    else:
        gt_files = None

    # fine ground truth transcription
    if args.text is not None:
        text_info = {}
        with open(args.text) as f:
            for line in f.readlines():
                key, value = line.strip().split(maxsplit=1)
                text_info[key] = value
    else:
        text_info = None

    # Get and divide list
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if gt_files is not None and len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )

    # Shard the filelists for parallel processing if requested
    if args.num_shards is not None and args.num_shards > 1:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError(
                f"Invalid shard_id={args.shard_id}, must be in [0, {args.num_shards-1}]"
            )
        all_keys = sorted(list(gen_files.keys()))
        sel_keys = [k for i, k in enumerate(all_keys) if i % args.num_shards == args.shard_id]
        gen_files = {k: gen_files[k] for k in sel_keys}
        if gt_files is not None:
            gt_files = {k: gt_files[k] for k in sel_keys if k in gt_files}
        if text_info is not None:
            text_info = {k: text_info[k] for k in sel_keys if k in text_info}
        logging.info(
            "Sharding enabled: shard_id=%d, num_shards=%d, selected items=%d",
            args.shard_id,
            args.num_shards,
            len(gen_files),
        )

    logging.info("The number of utterances = %d" % len(gen_files))

    with open(args.score_config, "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gt_text=(True if text_info is not None else False),
        use_gpu=args.use_gpu,
    )

    if len(score_modules) > 0:
        score_info = list_scoring(
            gen_files,
            score_modules,
            gt_files,
            text_info,
            output_file=args.output_file,
            io=args.io,
        )
        logging.info("Summary: {}".format(load_summary(score_info)))
    else:
        logging.info("No utterance-level scoring function is provided.")

    corpus_score_modules = load_corpus_modules(
        score_config,
        use_gpu=args.use_gpu,
        cache_folder=args.cache_folder,
        io=args.io,
    )
    assert (
        len(corpus_score_modules) > 0 or len(score_modules) > 0
    ), "no scoring function is provided"
    if len(corpus_score_modules) > 0:
        # In sharded mode, corpus-level metrics over the full corpus are generally not valid.
        if (args.num_shards and args.num_shards > 1) and not args.corpus_on_shards:
            logging.info(
                "Sharded mode detected; skipping corpus-level metrics. "
                "Run a non-sharded pass or set --corpus_on_shards to compute per-shard corpus metrics."
            )
            return

        # If corpus_on_shards is set, compute corpus metrics on current shard's prediction list
        pred_for_corpus = args.pred
        if args.num_shards and args.num_shards > 1:
            # When sharded, use the same pred argument; corpus scorer should support directory/scp inputs
            logging.info(
                "Computing corpus-level metrics on shard %d/%d.",
                args.shard_id,
                args.num_shards,
            )

        corpus_score_info = corpus_scoring(
            pred_for_corpus,
            corpus_score_modules,
            args.gt,
            text_info,
            output_file=(args.output_file + ".corpus") if args.output_file else None,
        )
        logging.info("Corpus Summary: {}".format(corpus_score_info))
    else:
        logging.info("No corpus-level scoring function is provided.")
        return


if __name__ == "__main__":
    main()

