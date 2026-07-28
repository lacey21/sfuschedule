#!/usr/bin/env python3
# regressionModel.py - loads all data (reusing decisionTreeModel.py's parsing/feature
# code) and trains a regression model as a point of comparison against the decision
# tree, saving it to a bundle for later use by regression.py.
#
# This does NOT re-implement any of the report parsing / feature engineering in
# decisionTreeModel.py - it imports it directly so both models are trained and
# evaluated on identical features. That's what makes compareModels.py a fair fight.

# Usage:
#    python regressionModel.py [flags]

# Flags:
#    --data-dir <path>            data/ folder to read from (default ../data)
#    --model-path <path>          where to save the trained bundle (default ./regression_model.joblib)
#    --model-type <name>          linear | ridge | forest (default linear)
#    --test-sizes <list>          comma-separated list of sizes to compare, model picks best
#    --lookback-terms <int>       how many prior terms to average the same-week historic fill rate over (default 4)
#    --skip-instructor-api        skip the SFU course-outlines API lookup (faster, works offline)
#    --max-instructor-lookups <int>
#                                  cap on unique sections queried against the
#                                  SFU API while building the training set (default 300)

# Example:
#    python regressionModel.py --model-type ridge --test-sizes 0.1,0.2,0.3

# Required: pip install pandas openpyxl scikit-learn joblib requests
import sys
import os
import argparse
from datetime import datetime

import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Reuse everything already built for the decision tree: data loading, feature
# engineering (course ratings, instructor ratings, historic fill rate, etc.),
# and the prediction pipeline. This guarantees an apples-to-apples comparison.
from decisionTreeModel import (
    DATA_DIR,
    FEATURE_NAMES,
    load_all_offerings,
    load_course_fill_by_week,
    build_training_set,
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression_model.joblib")

MODEL_BUILDERS = {
    # StandardScaler matters here (unlike for the tree) because linear/ridge
    # regression is sensitive to feature scale - week_of_reg (~1-15) and
    # max_enrol (~10-300) would otherwise dominate the coefficients.
    "linear": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ]),
    "ridge": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0, random_state=42)),
    ]),
    # Tree-based, so no scaling needed. Included as a stronger non-linear
    # regression baseline, in case plain linear regression underfits.
    "forest": lambda: Pipeline([
        ("reg", RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, random_state=42,
        )),
    ]),
}


def build_and_train_regression(X, y, model_type="linear"):
    # Builds and fits the requested regression model on the provided feature
    # matrix X and target vector y. Mirrors build_and_train_decision_tree().
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown --model-type '{model_type}'. Choose from: {list(MODEL_BUILDERS)}")
    model = MODEL_BUILDERS[model_type]()
    model.fit(X, y)
    return model


def evaluate_split(X, y, test_size, model_type):
    # Splits dataset into training and testing sets, same random_state as
    # decisionTreeModel.py so results are directly comparable split-for-split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = build_and_train_regression(X_train, y_train, model_type)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, mae, r2, model


def compare_test_splits(X, y, test_sizes, model_type):
    # Compares different test sizes to default to best, mirrors decisionTreeModel.py.
    results = []
    for ts in test_sizes:
        mse, mae, r2, model = evaluate_split(X, y, ts, model_type)
        results.append((ts, mse, mae, r2, model))
    results.sort(key=lambda t: -t[3])  # best R2 first
    best_ts, best_mse, best_mae, best_r2, best_model = results[0]
    print(f"Split Comparison ({model_type}): Best test_size = {best_ts} "
          f"(R2={best_r2:.4f}, RMSE={best_mse ** 0.5:.4f}, MAE={best_mae:.4f})")
    print(f"{'':>19}{'test_size':>10} {'n_train':>10} {'n_test':>10} {'mse':>10} {'rmse':>10} {'mae':>10} {'r2':>10}")
    for i, (ts, mse, mae, r2, _) in enumerate(results):
        n_test = int(round(len(X) * ts))
        n_train = len(X) - n_test
        marker = " <- best" if i == 0 else ""
        print(f"{'':>19}{ts:>10.2f} {n_train:>10} {n_test:>10} {mse:>10.4f} {mse ** 0.5:>10.4f} {mae:>10.4f} {r2:>10.4f}{marker}")
    reg = best_model.named_steps.get("reg")
    if hasattr(reg, "coef_"):
        print(f"  (standardized) coefficients:")
        for name, coef in sorted(zip(FEATURE_NAMES, reg.coef_), key=lambda t: -abs(t[1])):
            print(f"    {name:<16} {coef:+.4f}")
    elif hasattr(reg, "feature_importances_"):
        for name, imp in sorted(zip(FEATURE_NAMES, reg.feature_importances_), key=lambda t: -t[1]):
            print(f"    feature importance: {name:<16} {imp:.3f}")
    return best_ts, best_mse, best_r2


def main():
    parser = argparse.ArgumentParser(description="Train a regression model on the same features as the decision "
                                                   "tree and save it for reuse / comparison.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH,
                         help=f"Where to save the trained bundle. Default {MODEL_PATH}")
    parser.add_argument("--model-type", default="linear", choices=list(MODEL_BUILDERS),
                         help="Regression model to train. Default 'linear'.")
    parser.add_argument("--test-sizes", default="0.2",
                         help="Comma-separated train/test split ratios to compare, e.g. '0.1,0.2,0.3'. "
                              "The best-scoring one is reported; the saved model is always refit on "
                              "all data regardless. Default '0.2'.")
    parser.add_argument("--lookback-terms", type=int, default=4,
                         help="How many previous terms in data/courseFillByWeek/ to average the "
                              "same-week fill rate over. Default 4.")
    parser.add_argument("--skip-instructor-api", action="store_true",
                         help="Skip the SFU course-outlines API lookup (faster, works offline); "
                              "falls back to the course-level rating everywhere.")
    parser.add_argument("--max-instructor-lookups", type=int, default=300,
                         help="Cap on unique sections queried against the SFU course-outlines API "
                              "while building the training set, to limit runtime. Default 300.")
    args = parser.parse_args()
    try:
        test_sizes = [float(t) for t in args.test_sizes.split(",") if t.strip()]
    except ValueError:
        print(f"Error: --test-sizes must be a comma-separated list of numbers, got '{args.test_sizes}'")
        sys.exit(1)

    print(f"Using data directory: {args.data_dir}")
    offerings_df = load_all_offerings(args.data_dir)
    if offerings_df.empty:
        print(f"Error: No course offering files found under {args.data_dir}/courseOfferings/. "
              f"Add at least one <termCode>.xlsx report there and try again.")
        sys.exit(1)
    print(f"Loaded {len(offerings_df)} historical (date, section) snapshots "
          f"across terms: {sorted(offerings_df['term_code'].unique().tolist())}")
    fill_df = load_course_fill_by_week(args.data_dir)
    if fill_df.empty:
        print(f"Warning: No files found under {args.data_dir}/courseFillByWeek/ — "
              f"historic_fill_rate will fall back to an overall average.")
    else:
        print(f"Loaded {len(fill_df)} courseFillByWeek snapshots "
              f"across terms: {sorted(fill_df['term_code'].unique().tolist())}")

    X, y, _ = build_training_set(offerings_df, fill_df, args.data_dir, lookback_terms=args.lookback_terms,
                                  use_instructor_api=not args.skip_instructor_api,
                                  max_instructor_lookups=args.max_instructor_lookups)
    print(f"Built {len(X)} training rows with features {FEATURE_NAMES}")

    best_ts, best_mse, best_r2 = compare_test_splits(X, y, test_sizes, args.model_type)
    print(f"Training final {args.model_type} model on split {best_ts:.0%}/{1 - best_ts:.0%} "
          f"and saving to {args.model_path}...")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=best_ts, random_state=42)
    model = build_and_train_regression(X_train, y_train, args.model_type)

    bundle = {
        "model": model,
        "model_type": args.model_type,
        "offerings_df": offerings_df,
        "fill_df": fill_df,
        "feature_names": FEATURE_NAMES,
        "lookback_terms": args.lookback_terms,
        "use_instructor_api": not args.skip_instructor_api,
        "data_dir": args.data_dir,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.model_path)) or ".", exist_ok=True)
    joblib.dump(bundle, args.model_path, compress=3)
    print(f"Saved trained model bundle to {args.model_path}")
    print(f"Run predictions with: python regression.py <course> <section> <enrollment_date>")


if __name__ == "__main__":
    main()
