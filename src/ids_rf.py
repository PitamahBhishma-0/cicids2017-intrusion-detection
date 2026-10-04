"""
Random forest baseline for CIC-IDS2017 intrusion detection.

Uses exactly the same splits and preprocessing as the 1D CNN (ids_cnn.py), so
the two models can be compared under both evaluation protocols. Default
hyperparameters, no tuning; trained on the full training split (the CNN holds
out 20% of it for early stopping).

Usage:
    python src/ids_rf.py --split random
    python src/ids_rf.py --split day
"""

import argparse
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

import common

N_ESTIMATORS = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=common.SPLITS, default="random",
                        help="evaluation protocol (default: random)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: results/<split>/rf)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.out_dir or common.ROOT / "results" / args.split / "rf"
    out.mkdir(parents=True, exist_ok=True)

    results = {"model": "random_forest", "split": args.split, "seed": args.seed,
               "n_estimators": N_ESTIMATORS}

    X_train, X_test, y_train, y_test = common.build_splits(args.split, args.seed, results)
    X_train, X_test = common.impute(X_train, X_test)
    common.record_split(results, X_train, X_test, y_train, y_test)
    common.plot_class_distribution(args.split, results, y_train, y_test, out)

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1,
                                random_state=args.seed)
    rf.fit(X_train, y_train)

    proba = rf.predict_proba(X_test)[:, 1]
    cm = common.evaluate(y_test, proba, results)
    common.plot_confusion_matrix(cm, args.split, out)

    top = sorted(zip(X_train.columns, rf.feature_importances_), key=lambda t: -t[1])[:10]
    results["top_features"] = [[name, float(score)] for name, score in top]
    print("Top features:", [name for name, _ in top[:5]])

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Artifacts written to:", out)


if __name__ == "__main__":
    main()
