"""Exploratory CPU feasibility analysis; all resampling units are image blocks.

CLI: python analyze_results.py --root ROOT --freeze FREEZE.json --stage test
The freeze may contain analysis.primary_comparisons, each with name, family,
arm_a, arm_b and optional primary_metric(s). Arm selectors contain mode/policy
and optionally eta/ranking_weight. The difference is always A minus B.
Task seeds are averaged within image x rho x load; conditions are then averaged
equally within each image. Shared task streams are not independent replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

UNIT_KEYS = ["image_seed", "rho", "task_seed", "load"]
ARM_KEYS = ["mode", "policy", "eta", "ranking_weight"]
RATE_METRICS = ["sla_violation_rate", "prediction_mape", "decision_wrong_choice_rate",
                "probe_fraction", "missing_feedback_fraction", "panel_mape", "panel_final_mape",
                "probe_actual_cost_over_budget"]
SECOND_METRICS = ["p99_completion", "mean_wait", "probe_positive_service_cost_s",
                  "probe_signed_service_delta_s", "probe_predicted_budget_s",
                  "probe_budget_used_s", "total_actual_service_s"]
COUNT_METRICS = ["probe_count", "supervised_updates", "pair_updates", "arrived",
                 "completed", "privileged_label_count", "probe_eligible_count", "critical_eligible_count"]
RATIO_METRICS = ["probe_actual_cost_budget_ratio", "probe_reserved_budget_ratio"]
METRICS = RATE_METRICS + SECOND_METRICS + ["decision_score_regret_mean"] + COUNT_METRICS + RATIO_METRICS


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def canonical(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return "null"
    try:
        return format(float(value), ".12g")
    except (TypeError, ValueError):
        return str(value)


def prepare(raw, config):
    df = raw.copy()
    required = ["stage", "image_seed", "rho", "task_seed", "load", "mode", "policy"]
    absent = [c for c in required if c not in df]
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    for col in ["eta", "ranking_weight"]:
        if col not in df:
            df[col] = np.nan
    for col in ["image_seed", "rho", "task_seed", "load"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite unit key: {col}")
    for col in ["stage", "mode", "policy"]:
        if df[col].isna().any():
            raise ValueError(f"Missing unit/arm key: {col}")
        df[col] = df[col].astype(str)
    df["arm_id"] = df[ARM_KEYS].apply(
        lambda row: "|".join(f"{c}={canonical(row[c])}" for c in ARM_KEYS), axis=1)
    dup = df.duplicated(["stage"] + UNIT_KEYS + ["arm_id"], keep=False)
    if dup.any():
        raise ValueError("Duplicate unit/arm rows; do not pool attempts or duplicate runs: "
                         + df.loc[dup, ["stage"] + UNIT_KEYS + ["arm_id"]].head(8).to_json(orient="records"))
    # Budget ratios are descriptive diagnostics of predicted reservations,
    # not evidence of a guaranteed bound on the realized extra service time.
    budget_col = "probe_predicted_budget_s"
    if budget_col in df:
        budget = pd.to_numeric(df[budget_col], errors="raise")
        if (budget.dropna() < 0).any():
            raise ValueError("Negative predicted probe budget")
        for source, output in [("probe_positive_service_cost_s", "probe_actual_cost_budget_ratio"),
                               ("probe_budget_used_s", "probe_reserved_budget_ratio")]:
            if source in df:
                cost = pd.to_numeric(df[source], errors="raise")
                if (cost.dropna() < 0).any():
                    raise ValueError(f"Negative nonnegative probe cost: {source}")
                df[output] = cost / budget.replace(0, np.nan)
                df.loc[(budget == 0) & (cost == 0), output] = 0.0
                if source == "probe_positive_service_cost_s":
                    df["probe_actual_cost_over_budget"] = np.where(
                        cost.notna() & budget.notna(), (cost > budget + 1e-12).astype(float), np.nan)
    metrics = [m for m in METRICS if m in df]
    metric_units = {}
    supplied_units = config.get("units", {})
    rate_default = config.get("rate_columns_are", "fraction")
    if rate_default not in {"fraction", "percent"}:
        raise ValueError("rate_columns_are must be fraction or percent")
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="raise")
        if metric in RATE_METRICS:
            source_unit = supplied_units.get(metric, rate_default)
            if source_unit not in {"fraction", "percent"}:
                raise ValueError(f"Rate source unit must be fraction or percent: {metric}")
            finite = df[metric].dropna()
            if source_unit == "fraction":
                # MAPE is unbounded above, unlike probability/frequency rates.
                if (finite < 0).any() or (metric not in {"prediction_mape", "panel_mape", "panel_final_mape"}
                                         and (finite > 1 + 1e-9).any()):
                    raise ValueError(f"Invalid fraction metric: {metric}")
                df[metric] = 100.0 * df[metric]
            metric_units[metric] = "percent; difference in percentage points"
        elif metric in SECOND_METRICS:
            metric_units[metric] = "seconds"
        elif metric in COUNT_METRICS:
            metric_units[metric] = "count per run (averaged across streams)"
        elif metric in RATIO_METRICS:
            metric_units[metric] = "dimensionless per-run ratio (averaged across streams)"
        else:
            metric_units[metric] = "dimensionless normalized decision-score difference"
        if np.isinf(df[metric]).any():
            raise ValueError(f"Infinite metric: {metric}")
    return df, metrics, metric_units


def bootstrap_ci(values, repetitions, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, n, size=(repetitions, n))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))


def signflip_p(values, seed, repetitions=100000):
    """Two-sided paired sign-flip test; exact enumeration through 20 blocks."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return np.nan, "unavailable (<2 image blocks)"
    threshold = max(0.0, abs(values.mean()) - 1e-12)
    if threshold == 0.0:
        return 1.0, "exact" if n <= 20 else f"Monte Carlo {repetitions}"
    exceed = 0
    if n <= 20:
        total = 1 << n
        shifts = np.arange(n, dtype=np.uint64)
        for start in range(0, total, 4096):
            ids = np.arange(start, min(start + 4096, total), dtype=np.uint64)
            signs = 2.0 * ((ids[:, None] >> shifts) & 1) - 1.0
            exceed += int((np.abs(signs @ values / n) >= threshold).sum())
        return exceed / total, f"exact {total} sign assignments"
    rng = np.random.default_rng(seed)
    repetitions = max(100000, repetitions)
    for start in range(0, repetitions, 4096):
        count = min(4096, repetitions - start)
        signs = 2.0 * rng.integers(0, 2, size=(count, n)) - 1.0
        exceed += int((np.abs(signs @ values / n) >= threshold).sum())
    return (exceed + 1) / (repetitions + 1), f"Monte Carlo {repetitions} (+1 correction)"


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    previous = 0.0
    for position, index in enumerate(order):
        previous = max(previous, (len(values) - position) * values[index])
        adjusted[index] = min(1.0, previous)
    return adjusted


def stable_seed(base_seed, *parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def design_for_stage(stage_df, config):
    design = config.get("design", {}).get(str(stage_df["stage"].iloc[0]), {})
    levels = {key: sorted(design.get(key + "s", stage_df[key].unique().tolist()))
              for key in ["image_seed", "task_seed", "rho", "load"]}
    # Accept explicit conventional names as well.
    for key, name in [("rho", "rho_levels"), ("load", "load_levels"),
                      ("image_seed", "image_seeds"), ("task_seed", "task_seeds")]:
        if name in design:
            levels[key] = sorted(design[name])
        actual = set(stage_df[key].unique())
        if not actual.issubset(set(levels[key])):
            raise ValueError(f"Unexpected {key} outside frozen design: {actual-set(levels[key])}")
    return levels


def summaries(df, metrics, units, config, repetitions, seed):
    conditions, images, coverage = [], [], []
    for stage, stage_df in df.groupby("stage", sort=True):
        levels = design_for_stage(stage_df, config)
        n_tasks, n_conditions = len(levels["task_seed"]), len(levels["rho"]) * len(levels["load"])
        for arm, arm_df in stage_df.groupby("arm_id", sort=True):
            image_means = {metric: {} for metric in metrics}
            for image in levels["image_seed"]:
                sub = arm_df[arm_df.image_seed == image]
                complete_rows = len(sub) == n_tasks * n_conditions
                coverage.append(dict(stage=stage, arm_id=arm, image_seed=image,
                                     observed_units=len(sub), expected_units=n_tasks*n_conditions,
                                     complete_design=complete_rows))
                for metric in metrics:
                    good = sub.dropna(subset=[metric])
                    cm = good.groupby(["rho", "load"])[metric].agg(["mean", "count"])
                    if complete_rows and len(cm) == n_conditions and (cm["count"] == n_tasks).all():
                        image_means[metric][image] = float(cm["mean"].mean())
                        images.append(dict(stage=stage, arm_id=arm, image_seed=image,
                                           metric=metric, value=image_means[metric][image], unit=units[metric],
                                           conditions=n_conditions, streams=n_tasks*n_conditions))
            for metric in metrics:
                values = list(image_means[metric].values())
                lo, hi = bootstrap_ci(values, repetitions, stable_seed(seed, stage, arm, metric, "overall"))
                conditions.append(dict(stage=stage, arm_id=arm, scope="overall", rho=np.nan, load=np.nan,
                                       metric=metric, mean=np.mean(values) if values else np.nan,
                                       ci_low=lo, ci_high=hi, images=len(values), unit=units[metric]))
                for (rho, load), condition_df in arm_df.groupby(["rho", "load"], sort=True):
                    agg = condition_df.groupby("image_seed")[metric].agg(["mean", "count"])
                    values = agg.loc[agg["count"] == n_tasks, "mean"].to_numpy()
                    lo, hi = bootstrap_ci(values, repetitions, stable_seed(seed, stage, arm, metric, rho, load))
                    conditions.append(dict(stage=stage, arm_id=arm, scope="condition", rho=rho, load=load,
                                           metric=metric, mean=np.mean(values) if len(values) else np.nan,
                                           ci_low=lo, ci_high=hi, images=len(values), unit=units[metric]))
    return pd.DataFrame(conditions), pd.DataFrame(images), pd.DataFrame(coverage)


def select_arm(stage_df, selector):
    if isinstance(selector, str):
        selected = stage_df[stage_df.arm_id == selector]
    else:
        selected = stage_df
        for key, value in selector.items():
            if key not in ARM_KEYS:
                raise ValueError(f"Unknown arm selector key {key}")
            selected = selected[selected[key].map(canonical) == canonical(value)]
    arms = selected.arm_id.unique()
    if len(arms) != 1:
        raise ValueError(f"Arm selector must match exactly one configuration: {selector}; matched {arms.tolist()}")
    return str(arms[0]), selected


def paired_results(df, metrics, units, config, repetitions, seed):
    rows, exclusions = [], []
    stage = config.get("paired_stage", "test")
    if stage != "test":
        raise ValueError("Frozen primary comparisons are restricted to paired_stage='test'")
    stage_df = df[df.stage == stage]
    comparisons = config.get("primary_comparisons", [])
    if stage_df.empty or not comparisons:
        return pd.DataFrame(), pd.DataFrame()
    levels = design_for_stage(stage_df, config)
    n_tasks, n_conditions = len(levels["task_seed"]), len(levels["rho"])*len(levels["load"])
    for number, comparison in enumerate(comparisons):
        name = comparison.get("name", f"comparison_{number+1}")
        family = comparison.get("family", "feasibility_sla")
        arm_a, a = select_arm(stage_df, comparison["arm_a"])
        arm_b, b = select_arm(stage_df, comparison["arm_b"])
        merged = a[UNIT_KEYS+metrics].merge(b[UNIT_KEYS+metrics], on=UNIT_KEYS,
                                          suffixes=("_a", "_b"), validate="one_to_one")
        primary = comparison.get("primary_metrics", [comparison.get(
            "primary_metric", config.get("primary_metric", "sla_violation_rate"))])
        if any(m not in metrics for m in primary):
            raise ValueError(f"Missing primary metric for {name}: {primary}")
        for metric in metrics:
            valid = merged.dropna(subset=[metric+"_a", metric+"_b"]).copy()
            valid["difference"] = valid[metric+"_a"] - valid[metric+"_b"]
            cm = valid.groupby(["image_seed", "rho", "load"])[[metric+"_a", metric+"_b", "difference"]].mean()
            counts = valid.groupby(["image_seed", "rho", "load"]).size()
            cm = cm.loc[counts == n_tasks]
            complete_ids = cm.groupby(level="image_seed").size()
            complete_ids = complete_ids[complete_ids == n_conditions].index
            block = cm.loc[cm.index.get_level_values("image_seed").isin(complete_ids)].groupby(level="image_seed").mean()
            omitted = sorted(set(levels["image_seed"]) - set(block.index))
            if omitted:
                exclusions.append(dict(comparison=name, metric=metric, excluded_images=json.dumps(omitted),
                                       reason="missing paired task streams/metrics/conditions"))
            groups = [("overall", np.nan, np.nan, block)]
            for (rho, load), cb in cm.groupby(level=["rho", "load"], sort=True):
                groups.append(("condition", rho, load, cb.droplevel(["rho", "load"])))
            for scope, rho, load, values in groups:
                difference = values["difference"].to_numpy()
                ci_lo, ci_hi = bootstrap_ci(difference, repetitions,
                                           stable_seed(seed, name, metric, scope, rho, load))
                is_primary = scope == "overall" and metric in primary
                p, method = signflip_p(difference, stable_seed(seed, name, metric, "signflip")) if is_primary else (np.nan, "descriptive CI only")
                rows.append(dict(stage=stage, comparison=name, family=family, scope=scope,
                                 rho=rho, load=load, arm_a=arm_a, arm_b=arm_b, metric=metric,
                                 mean_a=values[metric+"_a"].mean(), mean_b=values[metric+"_b"].mean(),
                                 difference_a_minus_b=np.mean(difference) if len(difference) else np.nan,
                                 ci_low=ci_lo, ci_high=ci_hi, n_images=len(difference),
                                 unit=units[metric], primary=is_primary, exploratory=True,
                                 p_raw=p, p_holm=np.nan, holm_reject_05=False, signflip_method=method))
    result = pd.DataFrame(rows)
    if not result.empty:
        for family, indices in result[result.primary].groupby("family").groups.items():
            indices = list(indices)
            ps = result.loc[indices, "p_raw"].to_numpy()
            if np.isfinite(ps).all():
                adjusted = holm(ps)
                result.loc[indices, "p_holm"] = adjusted
                result.loc[indices, "holm_reject_05"] = adjusted <= .05
            else:
                raise ValueError(f"Incomplete primary family {family}; cannot silently shorten Holm family")
    return result, pd.DataFrame(exclusions)


def analyze(raw, freeze=None, repetitions=5000, seed=20261010):
    freeze = freeze or {}
    config = dict(freeze.get("analysis", {}))
    if "primary_comparisons" not in config and "primary_comparisons" in freeze:
        config["primary_comparisons"] = freeze["primary_comparisons"]
    df, metrics, units = prepare(raw, config)
    condition, images, coverage = summaries(df, metrics, units, config, repetitions, seed)
    paired, excluded = paired_results(df, metrics, units, config, repetitions, seed)
    arms = df[ ["stage", "arm_id"]+ARM_KEYS ].drop_duplicates().sort_values(["stage", "arm_id"])
    tables = dict(condition_summaries=condition, image_block_summaries=images,
                  coverage=coverage, paired_comparisons=paired, paired_exclusions=excluded,
                  arm_catalog=arms)
    metadata = dict(exploratory=True, primary_comparisons_stage="test", raw_rows=len(raw),
                    metric_units=units, bootstrap_repetitions=repetitions, analysis_seed=seed,
                    ci="marginal 95% percentile bootstrap over whole image blocks; comparisons bootstrap paired differences; arm summaries bootstrap image values",
                    aggregation="task seeds equal-weight within image/rho/load; rho/load conditions equal-weight within image; images equal-weight",
                    conditioning="Conditional on measured base profile and common task streams; streams shared across images/rho are not independent replications",
                    difference="A minus B; rate/MAPE differences in percentage points; time differences in seconds",
                    multiplicity="Holm within each specified overall primary family only; condition/other metric intervals descriptive and marginal",
                    incomplete_policy="No incomplete overall image block is averaged; exclusions and coverage retained",
                    derived_budget_metrics="Per-run actual-positive-cost/predicted-budget and reservation/predicted-budget ratios before aggregation; zero/zero defined as zero; positive/zero unavailable. Over-budget is a per-run indicator, not a guaranteed service-time bound.",
                    comparisons=config.get("primary_comparisons", []))
    return tables, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--stage", choices=["validation", "test", "all"], default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--no-write", action="store_true", help="Print metadata/primary rows without writing artifacts")
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error("Use at least 1000 bootstrap replicates")
    source = args.input or args.root / "raw" / "per_run_results.csv"
    freeze = json.loads(args.freeze.read_text(encoding="utf-8-sig")) if args.freeze else {}
    raw = pd.read_csv(source)
    if args.stage != "all":
        raw = raw[raw.stage == args.stage].copy()
    if raw.empty:
        raise ValueError("No rows in requested stage")
    tables, metadata = analyze(raw, freeze, args.bootstrap, args.seed)
    metadata.update(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    freeze=str(args.freeze.resolve()) if args.freeze else None,
                    freeze_sha256=hashlib.sha256(args.freeze.read_bytes()).hexdigest() if args.freeze else None,
                    analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    paired = tables["paired_comparisons"]
    primary = paired[paired.primary] if not paired.empty else paired
    if not args.no_write:
        out = args.output_dir or args.root / "analysis" / args.stage
        out.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig")
        (out / "analysis_metadata.json").write_text(json.dumps(json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(json_safe({"metadata": metadata, "primary_results": primary.to_dict(orient="records"),
                                "incomplete_blocks": int((~tables["coverage"].complete_design).sum())}),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
