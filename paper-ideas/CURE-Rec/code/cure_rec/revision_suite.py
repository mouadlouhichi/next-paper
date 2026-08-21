"""Reviewer revision protocols for Phases B--D.

The functions here operate on completed game/evaluation tables and deliberately
keep selection and evaluation data separate. They are reusable by notebooks and
CLI workflows; no result is called causal outside CURE-Sim or an audited log.
"""
from __future__ import annotations

from itertools import permutations
from math import comb, factorial
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from cure_rec.config import INTERVENTION_NAMES, Settings
from cure_rec.game import ALL_MASKS, GameResult
from cure_rec.planner import _constraints_for_mask
from cure_rec.revision_advanced import select_objective, sampled_shapley


def objective_ablation(game: GameResult, settings: Settings, penalties=(0.0, .25, .5, 1., 2., 5., 10.)) -> pd.DataFrame:
    rows=[]
    for objective in ("maximin", "mean"):
        for mode in ("hard", "penalty"):
            grid=penalties if mode == "penalty" else (0.0,)
            for penalty in grid:
                t=time.perf_counter(); result=select_objective(game, settings, objective=objective, constraint_mode=mode, penalty=float(penalty))
                feasible, margins=_constraints_for_mask(game, int(result["mask"]), settings)
                rows.append({"objective":objective,"constraint_mode":mode,"penalty":float(penalty),"mask":int(result["mask"]),"interventions":";".join(INTERVENTION_NAMES[i] for i in range(len(INTERVENTION_NAMES)) if result["mask"] & (1<<i)),"feasible":bool(feasible),"runtime_seconds":time.perf_counter()-t,**margins})
    return pd.DataFrame(rows)


def sampled_shapley_fidelity(values: dict[int,float], budgets=(32,128,512,2048), seed=42) -> pd.DataFrame:
    exact = {name: 0.0 for name in INTERVENTION_NAMES}
    n = len(INTERVENTION_NAMES)
    for order in permutations(range(n)):
        mask = 0
        for i in order:
            nxt = mask | (1 << i)
            exact[INTERVENTION_NAMES[i]] += values[nxt] - values[mask]
            mask = nxt
    exact = {name: value / factorial(n) for name, value in exact.items()}
    rows=[]
    for budget in budgets:
        t=time.perf_counter(); estimate=sampled_shapley(values, budget, seed=seed)
        a=np.array([estimate[n] for n in INTERVENTION_NAMES]); b=np.array([exact[n] for n in INTERVENTION_NAMES])
        rank_a=pd.Series(a).rank().to_numpy(); rank_b=pd.Series(b).rank().to_numpy()
        rows.append({"budget":budget,"mae":float(np.mean(np.abs(a-b))),"max_error":float(np.max(np.abs(a-b))),"sign_agreement":float(np.mean(np.sign(a)==np.sign(b))),"rank_correlation":float(np.corrcoef(rank_a,rank_b)[0,1]),"runtime_seconds":time.perf_counter()-t})
    return pd.DataFrame(rows)


def crn_paired_difference(base_sampler, intervention_sampler, seeds, independent=False) -> pd.DataFrame:
    """Estimate variance of a fixed coalition difference under CRN and IID shocks.

    Samplers accept a seed and return one scalar utility. In CRN mode both policies
    receive the same seed; IID mode derives an independent seed for the intervention.
    """
    rows=[]
    for seed in seeds:
        base=float(base_sampler(int(seed))); other_seed=int(seed)+1000003 if independent else int(seed)
        treated=float(intervention_sampler(other_seed)); rows.append({"seed":int(seed),"difference":treated-base,"mode":"independent" if independent else "crn"})
    return pd.DataFrame(rows)


def summarize_crn(crn: pd.DataFrame, independent: pd.DataFrame) -> dict:
    a=float(crn.difference.var(ddof=1)); b=float(independent.difference.var(ddof=1))
    return {"crn_variance":a,"independent_variance":b,"variance_ratio":a/b if b else float("nan"),"crn_n":len(crn),"independent_n":len(independent)}


def paired_user_statistics(frame: pd.DataFrame, model_col="model", user_col="user_id", metric_cols=("hit", "ndcg"), bootstrap=2000, seed=42) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Paired bootstrap and exact sign/permutation summaries for per-user metrics."""
    rng=np.random.default_rng(seed); models=list(frame[model_col].unique()); ref=models[0]; rows=[]; tests=[]
    for model in models[1:]:
        merged=frame[frame[model_col].isin([ref,model])].pivot(index=user_col,columns=model_col,values=list(metric_cols))
        for metric in metric_cols:
            x=merged[(metric,model)].to_numpy(float); y=merged[(metric,ref)].to_numpy(float); d=x-y; n=len(d)
            means=[]
            for _ in range(bootstrap): means.append(float(rng.choice(d,n,replace=True).mean()))
            lo,hi=np.quantile(means,[.025,.975]); signs=np.sign(d); nonzero=signs[signs!=0]; k=int((nonzero>0).sum()); m=len(nonzero)
            # Two-sided exact sign test: double the smaller of the lower and upper
            # binomial tails. Using only the lower tail saturates at 1.0 whenever the
            # effect is positive (k near m), which previously misreported significance.
            if m:
                lower_tail=sum(comb(m,j) for j in range(0,k+1))/(2**m)
                upper_tail=sum(comb(m,j) for j in range(k,m+1))/(2**m)
                p=min(1.,2*min(lower_tail,upper_tail))
            else:
                p=1.
            rows.append({"model":model,"reference":ref,"metric":metric,"n_users":n,"difference_mean":float(d.mean()),"bootstrap_low":float(lo),"bootstrap_high":float(hi),"effect_dz":float(d.mean()/d.std(ddof=1)) if d.std(ddof=1)>0 else float("nan")})
            tests.append({"model":model,"metric":metric,"raw_p":p})
    tests_df=pd.DataFrame(tests); tests_df["holm_p"] = _holm(tests_df.raw_p.to_numpy()) if len(tests_df) else []
    return pd.DataFrame(rows), tests_df


def _holm(p):
    p=np.asarray(p,float); order=np.argsort(p); out=np.empty_like(p); running=0.
    for rank,idx in enumerate(order): running=max(running, min(1., (len(p)-rank)*p[idx])); out[idx]=running
    return out


def write_revision_assets(output_dir: str|Path, tables: dict[str,pd.DataFrame], manifest: dict) -> Path:
    out=Path(output_dir); (out/"tables").mkdir(parents=True,exist_ok=True)
    for name,table in tables.items(): table.to_csv(out/"tables"/f"{name}.csv",index=False)
    (out/"revision_manifest.json").write_text(json.dumps({**manifest,"tables":sorted(tables)},indent=2,default=str),encoding="utf-8")
    return out
