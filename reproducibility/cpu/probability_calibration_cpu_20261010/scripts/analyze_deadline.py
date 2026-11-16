"""Deadline probability feasibility analysis; all resampling units are images.

CLI: python analyze_deadline.py --root ROOT --freeze FREEZE.json --stage test
The freeze may contain analysis.primary_comparisons, each with name, family,
arm_a, arm_b and optional primary_metric(s). Arm selectors contain mode/policy
and optionally eta/lambda_weight (weight is an accepted alias).
The difference is always A minus B. Task seeds are averaged within
image x rho x load x horizon; conditions are then averaged
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

UNIT_KEYS = ["image_seed", "rho", "task_seed", "load", "horizon"]
CONDITION_KEYS = ["rho", "load", "horizon"]
ARM_KEYS = ["mode", "policy", "eta", "lambda_weight", "margin"]
ALIASES = {"weight": "lambda_weight", "point_MAPE": "point_mape",
           "brier": "prob_brier", "brier_score": "prob_brier", "probs_brier": "prob_brier",
           "logloss": "prob_logloss", "log_loss": "prob_logloss", "probs_logloss": "prob_logloss",
           "safe_false_positive_rate": "safe_false_positive",
           "safe_false_negative_rate": "safe_false_negative"}
RATE_METRICS = ["sla_violation_rate", "point_mape", "decision_model_wrong_rate",
                "safe_false_positive", "safe_false_negative",
                "expired_at_dispatch_fraction", "boundary_sensitive_fraction", "panel_point_mape",
                "arrival_prefix24_sla", "arrival_prefix120_sla", "late20_sla",
                "probe_fraction", "missing_feedback_fraction", "panel_mape", "panel_final_mape",
                "probe_actual_cost_over_budget"]
SECOND_METRICS = ["p99_completion", "mean_wait", "probe_positive_service_cost_s",
                  "probe_signed_service_delta_s", "probe_predicted_budget_s",
                  "probe_budget_used_s", "total_actual_service_s"]
COUNT_METRICS = ["probe_count", "supervised_updates", "pair_updates", "arrived",
                 "completed", "privileged_label_count", "probe_eligible_count", "critical_eligible_count",
                 "arrival_prefix24_n", "arrival_prefix120_n", "calibration_clip_count",
                 "probability_updates", "calibration_updates"]
RATIO_METRICS = ["probe_actual_cost_budget_ratio", "probe_reserved_budget_ratio"]
SCORE_METRICS = ["prob_brier", "prob_logloss", "prob_brier_all", "panel_prob_brier",
                 "panel_prob_logloss", "panel_prob_squared_error", "decision_score_regret_mean"]
METRICS = RATE_METRICS + SECOND_METRICS + SCORE_METRICS + COUNT_METRICS + RATIO_METRICS


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
    for source, target in ALIASES.items():
        if source in df:
            if target in df:
                same = df[source].eq(df[target]) | (df[source].isna() & df[target].isna())
                if not same.all():
                    raise ValueError(f"Conflicting alias columns: {source}, {target}")
            else:
                df[target] = df[source]
    required = ["stage"] + UNIT_KEYS + ["mode"]
    absent = [c for c in required if c not in df]
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    for col in ["eta", "lambda_weight", "margin"]:
        if col not in df:
            df[col] = np.nan
    if "policy" not in df:
        df["policy"] = "no_probe"
    for col in UNIT_KEYS:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Non-finite unit key: {col}")
    for col in ["stage", "mode", "policy"]:
        if df[col].isna().any():
            raise ValueError(f"Missing unit/arm key: {col}")
        df[col] = df[col].astype(str)
    if (df.horizon <= 0).any():
        raise ValueError("horizon must be positive")
    if "arm_id" in df:
        df["source_arm_id"] = df["arm_id"]
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
            source_unit = "fraction" if metric == "probe_actual_cost_over_budget" else supplied_units.get(metric, rate_default)
            if source_unit not in {"fraction", "percent"}:
                raise ValueError(f"Rate source unit must be fraction or percent: {metric}")
            finite = df[metric].dropna()
            if source_unit == "fraction":
                # MAPE is unbounded above, unlike probability/frequency rates.
                if (finite < 0).any() or (metric not in {"point_mape", "panel_mape", "panel_final_mape", "panel_point_mape"}
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
        elif metric in set(SCORE_METRICS) - {"decision_score_regret_mean"}:
            metric_units[metric] = "dimensionless probability score (not percent)"
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
              for key in ["image_seed", "task_seed"] + CONDITION_KEYS}
    # Accept explicit conventional names as well.
    for key, name in [("rho", "rho_levels"), ("load", "load_levels"),
                      ("horizon", "horizons"), ("horizon", "horizon_levels"),
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
        n_tasks = len(levels["task_seed"])
        n_conditions = math.prod(len(levels[key]) for key in CONDITION_KEYS)
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
                    cm = good.groupby(CONDITION_KEYS)[metric].agg(["mean", "count"])
                    if complete_rows and len(cm) == n_conditions and (cm["count"] == n_tasks).all():
                        image_means[metric][image] = float(cm["mean"].mean())
                        images.append(dict(stage=stage, arm_id=arm, image_seed=image,
                                           metric=metric, value=image_means[metric][image], unit=units[metric],
                                           conditions=n_conditions, streams=n_tasks*n_conditions))
            for metric in metrics:
                values = list(image_means[metric].values())
                lo, hi = bootstrap_ci(values, repetitions, stable_seed(seed, stage, arm, metric, "overall"))
                conditions.append(dict(stage=stage, arm_id=arm, scope="overall", rho=np.nan, load=np.nan, horizon=np.nan,
                                       metric=metric, mean=np.mean(values) if values else np.nan,
                                       ci_low=lo, ci_high=hi, images=len(values), unit=units[metric]))
                for horizon, horizon_df in arm_df.groupby("horizon", sort=True):
                    cells = horizon_df.groupby(["image_seed", "rho", "load"])[metric].agg(["mean", "count"])
                    cells = cells[cells["count"] == n_tasks]
                    needed = len(levels["rho"]) * len(levels["load"])
                    complete = cells.groupby(level="image_seed").size()
                    complete = complete[complete == needed].index
                    cells = cells[cells.index.get_level_values("image_seed").isin(complete)]
                    hv = cells["mean"].groupby(level="image_seed").mean().to_numpy()
                    lo, hi = bootstrap_ci(hv, repetitions, stable_seed(seed, stage, arm, metric, "horizon", horizon))
                    conditions.append(dict(stage=stage, arm_id=arm, scope="horizon", rho=np.nan, load=np.nan, horizon=horizon,
                                           metric=metric, mean=np.mean(hv) if len(hv) else np.nan,
                                           ci_low=lo, ci_high=hi, images=len(hv), unit=units[metric]))
                for (rho, load, horizon), condition_df in arm_df.groupby(CONDITION_KEYS, sort=True):
                    agg = condition_df.groupby("image_seed")[metric].agg(["mean", "count"])
                    values = agg.loc[agg["count"] == n_tasks, "mean"].to_numpy()
                    lo, hi = bootstrap_ci(values, repetitions, stable_seed(seed, stage, arm, metric, rho, load, horizon))
                    conditions.append(dict(stage=stage, arm_id=arm, scope="condition", rho=rho, load=load, horizon=horizon,
                                           metric=metric, mean=np.mean(values) if len(values) else np.nan,
                                           ci_low=lo, ci_high=hi, images=len(values), unit=units[metric]))
    return pd.DataFrame(conditions), pd.DataFrame(images), pd.DataFrame(coverage)


def select_arm(stage_df, selector):
    if isinstance(selector, str):
        mask = stage_df.arm_id == selector
        if "source_arm_id" in stage_df:
            mask |= stage_df.source_arm_id == selector
        selected = stage_df[mask]
    else:
        selected = stage_df
        for key, value in selector.items():
            key = ALIASES.get(key, key)
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
    comparisons = [(c, True) for c in config.get("primary_comparisons", [])]
    comparisons += [(c, False) for c in config.get("diagnostic_comparisons", [])]
    if stage_df.empty or not comparisons:
        return pd.DataFrame(), pd.DataFrame()
    levels = design_for_stage(stage_df, config)
    n_tasks = len(levels["task_seed"])
    n_conditions = math.prod(len(levels[key]) for key in CONDITION_KEYS)
    for number, (comparison, frozen_primary) in enumerate(comparisons):
        name = comparison.get("name", f"comparison_{number+1}")
        family = comparison.get("family", "feasibility_sla")
        arm_a, a = select_arm(stage_df, comparison["arm_a"])
        arm_b, b = select_arm(stage_df, comparison["arm_b"])
        merged = a[UNIT_KEYS+metrics].merge(b[UNIT_KEYS+metrics], on=UNIT_KEYS,
                                          suffixes=("_a", "_b"), validate="one_to_one")
        primary = comparison.get("primary_metrics", [comparison.get(
            "primary_metric", config.get("primary_metric", "sla_violation_rate"))])
        primary = [ALIASES.get(metric, metric) for metric in primary]
        if any(m not in metrics for m in primary):
            raise ValueError(f"Missing primary metric for {name}: {primary}")
        for metric in metrics:
            valid = merged.dropna(subset=[metric+"_a", metric+"_b"]).copy()
            valid["difference"] = valid[metric+"_a"] - valid[metric+"_b"]
            cm = valid.groupby(["image_seed"] + CONDITION_KEYS)[[metric+"_a", metric+"_b", "difference"]].mean()
            counts = valid.groupby(["image_seed"] + CONDITION_KEYS).size()
            cm = cm.loc[counts == n_tasks]
            complete_ids = cm.groupby(level="image_seed").size()
            complete_ids = complete_ids[complete_ids == n_conditions].index
            block = cm.loc[cm.index.get_level_values("image_seed").isin(complete_ids)].groupby(level="image_seed").mean()
            omitted = sorted(set(levels["image_seed"]) - set(block.index))
            if omitted:
                exclusions.append(dict(comparison=name, metric=metric, excluded_images=json.dumps(omitted),
                                       reason="missing paired task streams/metrics/conditions"))
            groups = [("overall", np.nan, np.nan, np.nan, block)]
            for horizon, hc in cm.groupby(level="horizon", sort=True):
                needed = len(levels["rho"]) * len(levels["load"])
                complete = hc.groupby(level="image_seed").size()
                complete = complete[complete == needed].index
                hv = hc[hc.index.get_level_values("image_seed").isin(complete)].groupby(level="image_seed").mean()
                groups.append(("horizon", np.nan, np.nan, horizon, hv))
            for (rho, load, horizon), cb in cm.groupby(level=CONDITION_KEYS, sort=True):
                groups.append(("condition", rho, load, horizon, cb.droplevel(CONDITION_KEYS)))
            for scope, rho, load, horizon, values in groups:
                difference = values["difference"].to_numpy()
                ci_lo, ci_hi = bootstrap_ci(difference, repetitions,
                                           stable_seed(seed, name, metric, scope, rho, load, horizon))
                is_primary = frozen_primary and scope == "overall" and metric in primary
                p, method = signflip_p(difference, stable_seed(seed, name, metric, "signflip")) if is_primary else (np.nan, "descriptive CI only")
                rows.append(dict(stage=stage, comparison=name, family=family, scope=scope,
                                 rho=rho, load=load, horizon=horizon, arm_a=arm_a, arm_b=arm_b, metric=metric,
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


def evaluate_probe_gate(paired, coverage, config, test_present=True):
    """Evaluate only declared thresholds; never select a winner or launch work."""
    gate = config.get("probe_gate")
    if gate is None:
        return {"status": "not_declared", "passed": None, "rules": []}
    if not test_present:
        return {"status": "await_complete_test", "passed": None, "rules": []}
    if coverage.empty or not coverage[coverage.stage == "test"].complete_design.all():
        return {"status": "incomplete_design", "passed": False, "rules": []}
    definitions = gate.get("rules", [])
    if not definitions:
        raise ValueError("A declared probe_gate must have explicit nonempty rules")
    operators = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
                 ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
                 "==": lambda a, b: a == b}
    fields = {"difference_a_minus_b", "ci_low", "ci_high", "p_raw", "p_holm", "n_images"}
    evaluated = []
    for rule in definitions:
        metric = ALIASES.get(rule["metric"], rule["metric"])
        matches = paired[(paired.comparison == rule["comparison"]) & (paired.metric == metric)
                         & (paired.scope == rule.get("scope", "overall"))] if not paired.empty else paired
        for key in CONDITION_KEYS:
            if key in rule and not matches.empty:
                matches = matches[matches[key] == rule[key]]
        field, op = rule["field"], rule["op"]
        if field not in fields or op not in operators:
            raise ValueError(f"Unknown gate field/operator: {field}, {op}")
        value = float(rule["value"])
        if not math.isfinite(value):
            raise ValueError("Gate threshold must be finite")
        if len(matches) != 1 or not math.isfinite(float(matches.iloc[0][field])):
            evaluated.append(dict(rule=rule, observed=None, passed=False,
                                  reason="missing/ambiguous/nonfinite frozen statistic"))
        else:
            observed = float(matches.iloc[0][field])
            evaluated.append(dict(rule=rule, observed=observed,
                                  passed=bool(operators[op](observed, value)), reason=None))
    combiner = gate.get("combiner", "all")
    if combiner not in {"all", "any"}:
        raise ValueError("Gate combiner must be all or any")
    result = all(x["passed"] for x in evaluated) if combiner == "all" else any(x["passed"] for x in evaluated)
    return dict(status="evaluated", passed=result, combiner=combiner, rules=evaluated,
                action="diagnostic decision only; no experiment automatically launched")


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
                    aggregation="task seeds equal-weight within image/rho/load/horizon; rho/load/horizon conditions equal-weight within image; images equal-weight",
                    conditioning="Conditional on measured base profile and common task streams; streams shared across images/rho are not independent replications",
                    difference="A minus B; rate/MAPE differences in percentage points; time differences in seconds",
                    multiplicity="Holm within each specified overall primary family only; condition/other metric intervals descriptive and marginal",
                    incomplete_policy="No incomplete overall image block is averaged; exclusions and coverage retained",
                    derived_budget_metrics="Per-run actual-positive-cost/predicted-budget and reservation/predicted-budget ratios before aggregation; zero/zero defined as zero; positive/zero unavailable. Over-budget is a per-run indicator, not a guaranteed service-time bound.",
                    metric_definitions=config.get("metric_definitions", {}),
                    comparisons=config.get("primary_comparisons", []),
                    diagnostic_comparisons=config.get("diagnostic_comparisons", []))
    metadata["probe_gate"] = evaluate_probe_gate(paired, coverage, config, bool((df.stage == "test").any()))
    return tables, metadata


def synthetic_checks():
    """Synthetic-only QA. These data do not inform real method selection."""
    records = []
    for image in range(16):
        for rho in (0., .5, 1.):
            for load in (.75, 1.7):
                for horizon, effect in [(24., -.01), (120., -.02), (600., -.03)]:
                    for task in (52001, 52002):
                        for mode in ("window", "boundary", "oracle"):
                            records.append(dict(stage="test", image_seed=51001+image, rho=rho,
                                task_seed=task, load=load, horizon=horizon, mode=mode, eta=.05, weight=1.,
                                arm_id=mode, sla_violation_rate=.10+(effect if mode=="boundary" else 0.),
                                brier_score=.20-(.01 if mode=="boundary" else 0.),
                                log_loss=.70-(.02 if mode=="boundary" else 0.),
                                point_MAPE=.10, safe_false_positive=.02, safe_false_negative=.03,
                                decision_model_wrong_rate=.04, p99_completion=2.-(.2 if mode=="boundary" else 0.),
                                mean_wait=.1, arrived=horizon*10, completed=horizon*10))
    raw = pd.DataFrame(records)
    a = dict(mode="boundary", eta=.05, weight=1.)
    b = dict(mode="window", eta=.05, lambda_weight=1.)
    config = dict(design={"test": dict(image_seeds=list(range(51001,51017)),
                  task_seeds=[52001,52002], rho_levels=[0.,.5,1.], load_levels=[.75,1.7], horizons=[24.,120.,600.])},
                  primary_comparisons=[dict(name="synthetic_sla", family="synthetic", arm_a=a, arm_b=b),
                                       dict(name="synthetic_brier", family="synthetic", arm_a=a, arm_b=b, primary_metric="brier_score")],
                  diagnostic_comparisons=[dict(name="synthetic_oracle", arm_a="oracle", arm_b="window")],
                  probe_gate=dict(combiner="all", rules=[dict(comparison="synthetic_sla", metric="sla_violation_rate",
                      field="ci_high", op="<", value=0.), dict(comparison="synthetic_sla", metric="sla_violation_rate",
                      field="p_holm", op="<", value=.05)]))
    tables, meta = analyze(raw, {"analysis": config}, 1000, 20261010)
    paired = tables["paired_comparisons"]
    main = paired[(paired.comparison=="synthetic_sla") & paired.primary].iloc[0]
    assert main.n_images == 16 and abs(main.difference_a_minus_b + 2.) < 1e-12
    assert abs(main.ci_low+2.) < 1e-12 and abs(main.ci_high+2.) < 1e-12
    assert abs(main.p_raw-2/65536) < 1e-15 and abs(main.p_holm-4/65536) < 1e-15
    score = paired[(paired.comparison=="synthetic_brier") & paired.primary].iloc[0]
    assert abs(score.difference_a_minus_b+.01) < 1e-12
    logloss = paired[(paired.comparison=="synthetic_sla") & (paired.scope=="overall") & (paired.metric=="prob_logloss")].iloc[0]
    assert abs(logloss.difference_a_minus_b+.02) < 1e-12
    seconds = paired[(paired.comparison=="synthetic_sla") & (paired.scope=="overall") & (paired.metric=="p99_completion")].iloc[0]
    assert abs(seconds.difference_a_minus_b+.2) < 1e-12 and not seconds.primary
    conditions = paired[(paired.comparison=="synthetic_sla") & (paired.scope=="condition") & (paired.metric=="sla_violation_rate")]
    assert len(conditions)==18 and set(conditions.horizon)=={24.,120.,600.}
    horizons = paired[(paired.comparison=="synthetic_sla") & (paired.scope=="horizon") & (paired.metric=="sla_violation_rate")]
    assert np.allclose(horizons.sort_values("horizon").difference_a_minus_b,[-1.,-2.,-3.])
    assert not paired[paired.comparison=="synthetic_oracle"].primary.any()
    assert paired[paired.comparison=="synthetic_oracle"].p_holm.isna().all()
    assert meta["probe_gate"]["passed"] and len(paired[paired.primary])==2
    assert tables["coverage"].complete_design.all()
    margin_raw = raw[raw["mode"]=="window"].assign(mode="fixed_margin_0.15",margin=.15,arm_id="fixed_margin_0.15")
    margin_df, _, _ = prepare(margin_raw,{})
    _, margin_selected = select_arm(margin_df,dict(mode="fixed_margin_0.15",eta=.05,weight=1.,margin=.15))
    assert len(margin_selected)==576
    damaged = raw[~((raw.image_seed==51001)&(raw.horizon==600))]
    missing, missing_meta = analyze(damaged, {"analysis":config},1000,20261010)
    assert missing["paired_comparisons"].query("primary").n_images.eq(15).all()
    assert missing_meta["probe_gate"]["status"]=="incomplete_design"
    for bad, expected in [(pd.concat([raw,raw.iloc[:1]],ignore_index=True),"Duplicate"),
                          (raw.assign(lambda_weight=2.),"Conflicting")]:
        try:
            prepare(bad,{})
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"Expected failure: {expected}")
    assert np.allclose(holm([.01,.03,.5]),[.03,.06,.5])
    return dict(status="PASS", synthetic_only=True, rows=len(raw), image_blocks=16,
                streams_per_arm=576, conditions_per_image=18, bootstrap_QA=1000,
                checks=["horizon retained in duplicate/pair keys", "task/rho/load/horizon equal weights",
                        "18 separate conditions and 3 horizon summaries", "percentage-points vs seconds vs unscaled Brier/logloss",
                        "16-image exact sign-flip and family Holm", "oracle descriptive, excluded from Holm",
                        "frozen-only gate; incomplete matrix cannot release gate", "whole-image missingness",
                        "weight/lambda_weight aliases; conflicts rejected", "fixed-margin arm selector",
                        "duplicate attempts rejected"],
                example_sla_difference_pp=float(main.difference_a_minus_b), example_brier_difference=float(score.difference_a_minus_b),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


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
    parser.add_argument("--self-test", action="store_true", help="Synthetic-only QA; no raw experiment required")
    args = parser.parse_args()
    if args.bootstrap < 1000:
        parser.error("Use at least 1000 bootstrap replicates")
    if args.self_test:
        checks = synthetic_checks()
        if not args.no_write:
            out = args.output_dir or args.root / "qa"
            out.mkdir(parents=True, exist_ok=True)
            (out / "analysis_synthetic_checks.json").write_text(json.dumps(json_safe(checks),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps(json_safe(checks),ensure_ascii=False,indent=2,allow_nan=False))
        return
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
