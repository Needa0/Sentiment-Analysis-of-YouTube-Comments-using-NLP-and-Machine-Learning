"""Command-line interface for collection, preprocessing and language classification.

Examples
--------
Process the bundled sample dataset, print language statistics and export the
cleaned dataset + charts::

    python -m youtube_sentiment.cli process --csv data/sample_comments.csv \\
        --output-dir outputs --export outputs/cleaned.csv

Collect live comments first (requires YOUTUBE_API_KEY), then process::

    python -m youtube_sentiment.cli collect --video-id dQw4w9WgXcQ --csv comments.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .data_collection import YouTubeCommentCollector
from .pipeline import PreprocessingPipeline


def _cmd_collect(args: argparse.Namespace) -> int:
    collector = YouTubeCommentCollector(api_key=args.api_key)
    df = collector.collect_to_csv(
        output_path=args.csv,
        video_id=args.video_id,
        channel_id=args.channel_id,
        max_comments=args.max_comments,
    )
    print(f"Collected {len(df)} comments -> {args.csv}")
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    pipeline = PreprocessingPipeline(use_stemming=args.stemming)
    result = pipeline.run(csv_path=args.csv, text_column=args.text_column)

    print(f"Processed {result.total_comments} comments from {args.csv}\n")
    print("Language statistics:")
    for row in result.language_stats:
        print(f"  {row['language']:<12} {row['count']:>5}  ({row['percentage']}%)")

    if args.export:
        pipeline.export_cleaned(result, args.export)
        print(f"\nCleaned dataset exported to {args.export}")

    if args.output_dir:
        paths = pipeline.generate_visualizations(result, args.output_dir, theme=args.theme)
        report_path = os.path.join(args.output_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "total_comments": result.total_comments,
                    "language_distribution": result.language_distribution,
                    "language_stats": result.language_stats,
                    "visualizations": paths,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Charts + report written to {args.output_dir}")

    print("\nLogs:")
    for line in result.logs:
        print(f"  {line}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="youtube_sentiment",
        description=(
            "Multilingual YouTube comment collection, preprocessing and "
            "language classification."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    collect = sub.add_parser("collect", help="Collect comments via YouTube Data API v3")
    collect.add_argument("--video-id", default=None)
    collect.add_argument("--channel-id", default=None)
    collect.add_argument("--csv", required=True, help="Output CSV path")
    collect.add_argument("--api-key", default=None)
    collect.add_argument("--max-comments", type=int, default=200)
    collect.set_defaults(func=_cmd_collect)

    process = sub.add_parser(
        "process", help="Clean, preprocess and language-classify a CSV"
    )
    process.add_argument("--csv", required=True)
    process.add_argument("--text-column", default="comment")
    process.add_argument("--stemming", action="store_true", help="Also apply stemming")
    process.add_argument("--export", default=None, help="Export cleaned dataset to CSV")
    process.add_argument("--output-dir", default=None, help="Write charts + report")
    process.add_argument("--theme", default="light", choices=["light", "dark"])
    process.set_defaults(func=_cmd_process)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["process", *(argv or sys.argv[1:])])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
