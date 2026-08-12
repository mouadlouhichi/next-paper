"""Phase-A reviewer asset aggregation for completed revision experiments."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def aggregate_selector_runs(run_dirs: list[str | Path], output_dir: str | Path) -> pd.DataFrame:
    """Combine held-out selector summaries without rerunning simulations."""
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for run in run_dirs:
        run=Path(run)
        table=pd.read_csv(run/'heldout_selector_summary.csv')
        table.insert(0,'revision_run',run.name)
        rows.append(table)
    result=pd.concat(rows,ignore_index=True)
    result.to_csv(out/'reviewer_table_selector_holdout.csv',index=False)
    return result
