#!/usr/bin/env python3

# decisionTree.py — predicts how full a course section will be at a given enrollment date, using the model trained by decisionTreeModel.py.

# Note: If you haven't trained model yet run python decisionTreeModel.py
# This script does NOT parse the raw reports or retrain anything, itsaves the loaded models and outputs prediction

# Usage:
#    python decisionTree.py <course> <section> <enrollment_date YYYY-MM-DD>

# Flags:
#    --model-path <path>      bundle produced by decisionTreeModel.py (default ./decisionTree_model.joblib)
#    --data-dir <path>        override the data/ folder already into the bundle
#   --skip-instructor-api     skip the live SFU course-outlines API call for this prediction

# Example:
#    python decisionTree.py CMPT225 D100 2026-07-15


import sys
import argparse

import joblib

from decisionTreeModel import predict_fullness, MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Predict course section fullness at a given enrollment date.")
    parser.add_argument("course", help="e.g. CMPT225 or 'CMPT 225'")
    parser.add_argument("section", help="e.g. D100")
    parser.add_argument("enrollment_date", help="YYYY-MM-DD")
    parser.add_argument("--model-path", default=MODEL_PATH,
                         help=f"Trained bundle from decisionTreeModel.py. Default {MODEL_PATH}")
    parser.add_argument("--data-dir", default=None,
                         help="Override the data/ folder already into the bundle (default: use the bundle's).")
    parser.add_argument("--skip-instructor-api", action="store_true",
                         help="Skip the live SFU course-outlines API call for this prediction.")
    args = parser.parse_args()
    try:
        bundle = joblib.load(args.model_path)
    except FileNotFoundError:
        print(f"ERROR: No trained model found at {args.model_path}.")
        print("Train one first with: python decisionTreeModel.py")
        sys.exit(1)
    print(f"Running Prediction for {args.course} {args.section} on {args.enrollment_date} ---")
    print(f"Using model trained at {bundle.get('trained_at', 'unknown time')} ({args.model_path})")
    data_dir = args.data_dir if args.data_dir is not None else bundle["data_dir"]
    use_instructor_api = bundle["use_instructor_api"] and not args.skip_instructor_api
    predict_fullness(
        bundle["model"],
        bundle["offerings_df"],
        bundle["fill_df"],
        args.course,
        args.section,
        args.enrollment_date,
        data_dir=data_dir,
        lookback_terms=bundle["lookback_terms"],
        use_instructor_api=use_instructor_api,
    )
if __name__ == "__main__":
    main()