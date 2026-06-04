#!/usr/bin/env python3
"""Analyze the final greedy policy from a multi-stage MARL experiment results file.

Usage:
    python analyze_final_policy.py <path/to/ms_marl_experiment_results.json>

Outputs (saved in the same folder as the input file):
    final_policy.json        -- the final greedy policy
    final_policy_analysis.txt -- summary statistics
"""

import json
import sys
from pathlib import Path


def get_last_key(d):
    return sorted(d.keys(), key=int)[-1]


def compute_mitigation_shares(policy):
    """Sum lever_effort_fraction across energy/methane/agriculture per agent."""
    levers = ["energy", "methane", "agriculture"]
    country_totals = {
        c: sum(sum(v["lever_effort_fraction"][l]) for l in levers)
        for c, v in policy.items()
    }
    grand_total = sum(country_totals.values())
    shares = {c: (total / grand_total if grand_total > 0 else 0.0)
              for c, total in country_totals.items()}
    return country_totals, shares, grand_total


MITIGATION_LEVEL = {0: 0.0, 1: 0.5, 2: 1.0}


def compute_agreement_stats(policy):
    """
    Per-agent agreement stats, excluding self-accepted proposals.

    Negotiation data is stored per-agent in their own view:
      promises_idx[step][j] = discrete action index (0/1/2) that country_k promises to country_j
      acceptances[step][j]  = 1 if country_k accepted the deal with country_j

    Action indices map to mitigation levels: 0 → 0.0, 1 → 0.5, 2 → 1.0
    Self-accepted: j == k (agent negotiating with itself), excluded.

    Returns per-agent:
      count_accepted              -- number of deals country_k accepted (with others)
      mean_accepted_mitigation    -- mean mitigation level committed to in accepted deals
    """
    countries = list(policy.keys())
    stats = {}

    for i, country in enumerate(countries):
        neg = policy[country]["negotiation"]
        promises_idx = neg["promises_idx"]
        acceptances = neg["acceptances"]

        accepted_count = 0
        accepted_levels = []

        for step_promises, step_acceptances in zip(promises_idx, acceptances):
            if step_promises is None or step_acceptances is None:
                continue
            for j, (promise, accept) in enumerate(zip(step_promises, step_acceptances)):
                if j == i:  # exclude self-deal
                    continue
                if accept == 1:
                    accepted_count += 1
                    accepted_levels.append(MITIGATION_LEVEL[promise])

        mean_level = (sum(accepted_levels) / len(accepted_levels)
                      if accepted_levels else float("nan"))
        stats[country] = {
            "count_accepted": accepted_count,
            "mean_accepted_mitigation": mean_level,
        }

    return stats


def compute_training_stats(training_time_stats):
    """
    training_time_stats maps env_step -> cumulative wall-clock seconds.

    Returns:
      mean_env_step_ms   -- average milliseconds per env-step
      mean_iteration_sec -- average seconds per training iteration
      total_time_h       -- total training time in hours
    """
    keys = sorted(training_time_stats.keys(), key=int)
    times = [training_time_stats[k] for k in keys]
    steps = [int(k) for k in keys]

    iter_durations = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    step_diffs = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
    steps_per_iter = step_diffs[0]  # constant (20800)

    mean_iter_sec = sum(iter_durations) / len(iter_durations)
    mean_step_ms = (mean_iter_sec * 1000.0) / steps_per_iter
    total_time_h = times[-1] / 3600.0

    return {
        "mean_env_step_ms": mean_step_ms,
        "mean_iteration_sec": mean_iter_sec,
        "total_training_time_h": total_time_h,
    }


def fmt_nan(val, fmt=".4f"):
    if val != val:  # NaN check
        return "     N/A"
    return f"{val:{fmt}}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_final_policy.py <ms_marl_experiment_results.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    output_dir = input_path.parent

    with open(input_path) as f:
        data = json.load(f)

    # ── Final checkpoint ──────────────────────────────────────────────────────
    last_key = get_last_key(data["greedy_policy"])
    final_policy = data["greedy_policy"][last_key]
    countries = list(final_policy.keys())

    # ── Save final_policy.json ────────────────────────────────────────────────
    policy_path = output_dir / "final_policy.json"
    with open(policy_path, "w") as f:
        json.dump(final_policy, f, indent=2)

    # ── Reward / cost at final checkpoint ────────────────────────────────────
    climate_rewards = data["climate_reward"][last_key]
    nc_costs = data["noncompliance_cost"][last_key]
    collective_climate = sum(climate_rewards.values())
    collective_nc = sum(nc_costs.values())

    # ── Peak temperature ──────────────────────────────────────────────────────
    temp_traj = data["temperature_trajectory"][last_key]
    peak_temp = max(temp_traj)

    # ── Mitigation shares ─────────────────────────────────────────────────────
    country_totals, shares, grand_total = compute_mitigation_shares(final_policy)

    # ── Agreement stats ───────────────────────────────────────────────────────
    agreement_stats = compute_agreement_stats(final_policy)

    # ── Training performance ──────────────────────────────────────────────────
    perf = compute_training_stats(data["training_time_stats"])

    # ── Build report ──────────────────────────────────────────────────────────
    W = 62
    sep = "=" * W
    thin = "-" * W

    lines = [
        sep,
        "FINAL GREEDY POLICY ANALYSIS",
        f"Source : {input_path.name}",
        f"Checkpoint : {last_key} env-steps",
        sep,
        "",
        "── Greedy Climate Reward ────────────────────────────────────",
        f"  {'Agent':<14} {'Individual':>12}",
        f"  {thin[:30]}",
    ]
    for c in countries:
        lines.append(f"  {c:<14} {climate_rewards[c]:>12.4f}")
    lines += [
        f"  {'Collective':<14} {collective_climate:>12.4f}",
        "",
        "── Greedy Noncompliance Cost ────────────────────────────────",
        f"  {'Agent':<14} {'Individual':>12}",
        f"  {thin[:30]}",
    ]
    for c in countries:
        lines.append(f"  {c:<14} {nc_costs[c]:>12.4f}")
    lines += [
        f"  {'Collective':<14} {collective_nc:>12.4f}",
        "",
        "── Mitigation Share (Individual) ────────────────────────────",
        f"  {'Agent':<14} {'Total Effort':>14} {'Share':>8}",
        f"  {thin[:38]}",
    ]
    for c in countries:
        lines.append(f"  {c:<14} {country_totals[c]:>14.2f} {shares[c]:>8.2%}")
    lines += [
        f"  {'TOTAL':<14} {grand_total:>14.2f} {'100.00%':>8}",
        "",
        "── Peak Temperature ─────────────────────────────────────────",
        f"  Peak temperature under policy period : {peak_temp:.4f} °C above pre-industrial",
        "",
        "── Agreement Stats (excluding self-accepted proposals) ──────",
        f"  {'Agent':<14} {'Count Accepted':>14} {'Mean Mit. Level':>16}",
        f"  {thin[:46]}",
    ]
    for c in countries:
        s = agreement_stats[c]
        lines.append(
            f"  {c:<14} {s['count_accepted']:>14d}"
            f" {fmt_nan(s['mean_accepted_mitigation']):>16}"
        )
    lines += [
        "",
        "── Performance Statistics ───────────────────────────────────",
        f"  Mean env-step time   : {perf['mean_env_step_ms']:>10.4f} ms",
        f"  Mean iteration time  : {perf['mean_iteration_sec']:>10.4f} sec",
        f"  Total training time  : {perf['total_training_time_h']:>10.4f} h",
        "",
        sep,
    ]

    report = "\n".join(lines)
    print(report)

    txt_path = output_dir / "final_policy_analysis.txt"
    with open(txt_path, "w") as f:
        f.write(report + "\n")

    print(f"\nSaved: {policy_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
