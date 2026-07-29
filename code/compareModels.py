#!/usr/bin/env python3

# compareModels.py — trains the decision tree and one or more regression models on
# an IDENTICAL train/test split (same features, same random_state) and reports:
#   1. Regression metrics: MSE, RMSE, MAE, R^2
#   2. Classification metrics: accuracy, precision, recall, F1, confusion matrix
#      (Note: percent_filled is a continuous target, so "accuracy"/"F1" only make
#      sense after binning predictions into fullness categories — see --bins below.
#      This script does that binning consistently for every model so the
#      classification comparison is apples-to-apples.)
#
# This does NOT retrain/overwrite your saved decisionTree_model.joblib or
# regression_model.joblib bundles — it's a standalone evaluation run.

# Usage:
#    python compareModels.py [flags]

# Flags:
#    --data-dir <path>            data/ folder to read from (default ../data)
#    --regression-types <list>    comma-separated: linear,ridge,forest (default: linear,forest)
#    --test-size <float>          held-out fraction, same split used for every model (default 0.2)
#    --lookback-terms <int>       same meaning as in decisionTreeModel.py (default 4)
#    --skip-instructor-api        skip the live SFU course-outlines API lookup
#    --max-instructor-lookups <int>  default 300
#    --bins <list>                comma-separated upper edges of the fullness bins used
#                                  for the classification comparison, e.g. "0.7,0.95"
#                                  splits into Under-filled (<0.70), On-track (0.70-0.95),
#                                  Full (>=0.95). Default "0.7,0.95".

# Example:
#    python compareModels.py --regression-types linear,ridge,forest --test-size 0.25

import sys
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from decisionTreeModel import (
    DATA_DIR, FEATURE_NAMES,
    load_all_offerings, load_course_fill_by_week, build_training_set,
    build_and_train_decision_tree,
)
from regressionModel import build_and_train_regression, MODEL_BUILDERS


def bin_fullness(values, edges, labels):
    # Turns continuous percent_filled values into ordered category labels so
    # classification metrics (accuracy/F1/etc.) are meaningful. edges are the
    # UPPER bound of every bin except the last, e.g. edges=[0.7, 0.95] with
    # 3 labels means: <0.70, [0.70, 0.95), >=0.95.
    values = np.clip(values, 0.0, 1.0)
    idx = np.digitize(values, edges, right=False)
    return np.array([labels[i] for i in idx])


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": mse ** 0.5, "MAE": mae, "R2": r2}


def classification_metrics(y_true_labels, y_pred_labels, labels):
    return {
        "Accuracy": accuracy_score(y_true_labels, y_pred_labels),
        "Precision (macro)": precision_score(y_true_labels, y_pred_labels, labels=labels,
                                              average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true_labels, y_pred_labels, labels=labels,
                                        average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true_labels, y_pred_labels, labels=labels,
                                average="macro", zero_division=0),
        "F1 (weighted)": f1_score(y_true_labels, y_pred_labels, labels=labels,
                                   average="weighted", zero_division=0),
    }


def print_regression_table(results):
    print("\n=== Regression metrics (lower MSE/RMSE/MAE is better, higher R2 is better) ===")
    header = f"{'Model':<16}{'MSE':>10}{'RMSE':>10}{'MAE':>10}{'R2':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16}{m['MSE']:>10.4f}{m['RMSE']:>10.4f}{m['MAE']:>10.4f}{m['R2']:>10.4f}")


def print_classification_table(results):
    print("\n=== Classification metrics (fullness binned into categories) ===")
    header = f"{'Model':<16}{'Accuracy':>10}{'Prec(macro)':>13}{'Rec(macro)':>12}{'F1(macro)':>11}{'F1(wtd)':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16}{m['Accuracy']:>10.4f}{m['Precision (macro)']:>13.4f}"
              f"{m['Recall (macro)']:>12.4f}{m['F1 (macro)']:>11.4f}{m['F1 (weighted)']:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare the decision tree against one or more regression "
                                                   "models on an identical train/test split.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--regression-types", default="linear,forest",
                         help=f"Comma-separated regression models to include: {list(MODEL_BUILDERS)}. "
                              f"Default 'linear,forest'.")
    parser.add_argument("--test-size", type=float, default=0.2,
                         help="Held-out fraction, same split used for every model. Default 0.2.")
    parser.add_argument("--lookback-terms", type=int, default=4)
    parser.add_argument("--skip-instructor-api", action="store_true")
    parser.add_argument("--max-instructor-lookups", type=int, default=300)
    parser.add_argument("--bins", default="0.7,0.95",
                         help="Comma-separated upper edges for the fullness bins used in the "
                              "classification comparison. Default '0.7,0.95' -> "
                              "Under-filled (<0.70) / On-track (0.70-0.95) / Full (>=0.95).")
    args = parser.parse_args()

    try:
        edges = [float(e) for e in args.bins.split(",") if e.strip()]
    except ValueError:
        print(f"Error: --bins must be a comma-separated list of numbers, got '{args.bins}'")
        sys.exit(1)
    if edges != sorted(edges) or any(e <= 0 or e >= 1 for e in edges):
        print(f"Error: --bins edges must be strictly increasing values between 0 and 1, got '{args.bins}'")
        sys.exit(1)
    labels = [f"<{edges[0]:.0%}"] + \
             [f"{edges[i]:.0%}-{edges[i+1]:.0%}" for i in range(len(edges) - 1)] + \
             [f">={edges[-1]:.0%}"]

    reg_types = [t.strip() for t in args.regression_types.split(",") if t.strip()]
    for t in reg_types:
        if t not in MODEL_BUILDERS:
            print(f"Error: unknown --regression-types entry '{t}'. Choose from: {list(MODEL_BUILDERS)}")
            sys.exit(1)

    print(f"Using data directory: {args.data_dir}")
    offerings_df = load_all_offerings(args.data_dir)
    if offerings_df.empty:
        print(f"Error: No course offering files found under {args.data_dir}/courseOfferings/.")
        sys.exit(1)
    fill_df = load_course_fill_by_week(args.data_dir)

    X, y, _ = build_training_set(offerings_df, fill_df, args.data_dir, lookback_terms=args.lookback_terms,
                                  use_instructor_api=not args.skip_instructor_api,
                                  max_instructor_lookups=args.max_instructor_lookups)
    print(f"Built {len(X)} rows with features {FEATURE_NAMES}")

    # One split, reused for every model, so the comparison is apples-to-apples.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    print(f"Train/test split: {len(X_train)} train / {len(X_test)} test (test_size={args.test_size})\n")
    print(f"Fullness bins used for classification metrics: "
          + ", ".join(f"{lab}" for lab in labels))

    y_test_labels = bin_fullness(y_test, edges, labels)

    models = {"decision_tree": build_and_train_decision_tree(X_train, y_train)}
    for t in reg_types:
        models[t] = build_and_train_regression(X_train, y_train, t)

    reg_results, clf_results, preds_by_model = {}, {}, {}
    for name, model in models.items():
        preds = model.predict(X_test)
        preds_by_model[name] = preds
        reg_results[name] = regression_metrics(y_test, preds)
        pred_labels = bin_fullness(preds, edges, labels)
        clf_results[name] = classification_metrics(y_test_labels, pred_labels, labels)

    print_regression_table(reg_results)
    print_classification_table(clf_results)

    print("\n=== Per-model confusion matrix & classification report ===")
    for name, preds in preds_by_model.items():
        pred_labels = bin_fullness(preds, edges, labels)
        print(f"\n--- {name} ---")
        cm = confusion_matrix(y_test_labels, pred_labels, labels=labels)
        print(f"Confusion matrix (rows=actual, cols=predicted), labels={labels}:")
        print(cm)
        print(classification_report(y_test_labels, pred_labels, labels=labels, zero_division=0))

    best_reg = max(reg_results.items(), key=lambda kv: kv[1]["R2"])
    best_clf = max(clf_results.items(), key=lambda kv: kv[1]["F1 (macro)"])
    print(f"Best by R2 (regression fit):        {best_reg[0]} (R2={best_reg[1]['R2']:.4f})")
    print(f"Best by F1 macro (classification):  {best_clf[0]} (F1={best_clf[1]['F1 (macro)']:.4f})")


if __name__ == "__main__":
    main()
