#!/usr/bin/env python3
"""Run the predeclared Monte Carlo convergence study on fixed users."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actionshap.baselines import monte_carlo_attribution
from actionshap.candidates import fixed_evaluation_sets
from actionshap.models.profile import fit_item_embeddings
from actionshap.recommendation import UserGame, target_margin_utility
from actionshap.recommendation_data import load_movielens_1m, truncate_histories
from actionshap.convergence import convergence_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ratings", required=True)
    p.add_argument("--output", default="results/raw/convergence_seed42.json")
    p.add_argument("--users", type=int, default=100)
    p.add_argument("--evaluation-size", type=int, default=200)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    data = load_movielens_1m(args.ratings)
    histories = truncate_histories(data, args.n_max)
    model = fit_item_embeddings(histories, data.n_items, epochs=args.epochs, seed=args.seed)
    evaluations, _ = fixed_evaluation_sets(histories, data.test, data.n_items, args.evaluation_size, args.seed)
    rows = []
    for u in sorted(data.test)[:args.users]:
        game = UserGame(histories[u], evaluations[u], data.test[u], np.arange(evaluations[u].size))
        utility = lambda coalition, g=game: target_margin_utility(model, g, coalition)
        for row in convergence_table(utility, game.players.size, budgets=(25, 50, 100, 250), seeds=(0, 1, 2, 3, 4), reference=250):
            row["user"] = int(u)
            rows.append(row)
    # Aggregate rows by permutation count; the notebook plots these values.
    import pandas as pd
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("permutations", as_index=False).agg({
        "mean_rank_correlation_to_reference": "mean",
        "std_rank_correlation_to_reference": "mean",
        "mean_efficiency_error": "mean",
    })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": vars(args), "rows": grouped.to_dict("records")}, indent=2))
    print(grouped.to_string(index=False))

if __name__ == "__main__":
    main()
