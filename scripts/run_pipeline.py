"""Run pipeline steps: python scripts/run_pipeline.py [step ...]  (default: all)."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from impact.pipeline import STEPS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="*", help=f"steps to run, in order: {', '.join(STEPS)}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    unknown = set(args.steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {', '.join(sorted(unknown))}")
    for name in args.steps or STEPS:
        logging.info("== %s ==", name)
        STEPS[name]()


if __name__ == "__main__":
    main()
