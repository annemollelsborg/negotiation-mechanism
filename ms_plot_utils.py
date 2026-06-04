import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from typing import Optional


def _steps_fmt(x, pos=None):
    x = int(round(x))
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M".replace(".0M", "M")
    if x >= 1000:
        return f"{x/1000:.0f}K"
    return str(x)


def plot_multiple_reward(
    series: list[tuple[str, pd.DataFrame]],
    dual_xaxis: bool = False,
    x_max: Optional[int] = None,
    cmap: str = "ocean",
    y_min: float = -20,
    y_max: float = -4,
    savefig: Optional[str] = None,
    figsize: tuple[float, float] = (10, 6),
    ylabel: str = "Training reward",
):
    """
    Overlay training reward curves for multiple runs.

    series : list of (name, df) tuples
        Each entry is a (label, DataFrame) pair. DataFrame index = environment
        steps (int or str), columns = agent names.
    dual_xaxis : bool
        If True, the first series is plotted on the bottom x-axis and all
        remaining series on the top x-axis.
    x_max : int or None
        If set, all series are clipped to this step count.
    """
    if not series:
        raise ValueError("series must contain at least one (name, df) pair")
    if dual_xaxis and len(series) < 2:
        dual_xaxis = False

    line_styles = ["-", "--", "-.", ":"]

    def _prep(df):
        out = df.copy()
        out.index = out.index.astype(int)
        out = out.sort_index()
        if x_max is not None:
            out = out[out.index <= x_max]
        return out

    prepped = [(name, _prep(df)) for name, df in series]

    seen: set[str] = set()
    all_agents: list[str] = []
    for _, df in prepped:
        for col in df.columns:
            if col not in seen:
                all_agents.append(col)
                seen.add(col)

    cmap_obj = plt.get_cmap(cmap)
    colors = {a: cmap_obj(x) for a, x in zip(all_agents, np.linspace(0.15, 0.85, len(all_agents)))}

    fig, ax_bot = plt.subplots(figsize=figsize)
    ax_top = ax_bot.twiny() if dual_xaxis else None

    handles_per_series: list[tuple[str, list[tuple]]] = []

    for s_idx, (name, df) in enumerate(prepped):
        ls = line_styles[s_idx % len(line_styles)]
        ax = ax_bot if (not dual_xaxis or s_idx == 0) else ax_top
        agents_in_df = [a for a in all_agents if a in df.columns]
        handles = []
        for agent in agents_in_df:
            (h,) = ax.plot(df.index, df[agent], color=colors[agent], lw=2.0,
                           ls=ls, alpha=0.85)
            handles.append((h, agent))
        handles_per_series.append((name, handles))

    bot_xmax = x_max if x_max is not None else int(prepped[0][1].index.max())
    ax_bot.set_xlim(0, bot_xmax)
    ax_bot.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax_bot.xaxis.set_major_formatter(FuncFormatter(_steps_fmt))
    ax_bot.tick_params(axis="both", labelsize=12)

    if dual_xaxis:
        top_raw_max = max(int(df.index.max()) for _, df in prepped[1:])
        top_xmax = x_max if x_max is not None else top_raw_max
        ax_top.set_xlim(0, top_xmax)
        ax_top.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax_top.xaxis.set_major_formatter(FuncFormatter(_steps_fmt))
        ax_top.tick_params(axis="x", labelsize=12)

        ax_bot.set_xlabel(f"Environment steps — {prepped[0][0]}", fontsize=14)
        top_names = " / ".join(n for n, _ in prepped[1:])
        ax_top.set_xlabel(f"Environment steps — {top_names}", fontsize=14)
    else:
        ax_bot.set_xlabel("Environment steps", fontsize=14)

    ax_bot.set_ylabel(ylabel, fontsize=14)
    ax_bot.set_ylim(y_min, y_max)
    ax_bot.grid(True, linestyle=":", alpha=0.5)

    n = len(handles_per_series)
    if n == 1:
        positions = [0.99]
        locs = ["lower right"]
    else:
        positions = [0.59 + (0.40 / (n - 1)) * i for i in range(n)]
        locs = ["lower left"] * (n - 1) + ["lower right"]

    for i, ((name, handles), x_pos, loc) in enumerate(zip(handles_per_series, positions, locs)):
        leg = ax_bot.legend(
            [h for h, _ in handles],
            [a for _, a in handles],
            title=name,
            frameon=True,
            loc=loc,
            bbox_to_anchor=(x_pos, 0.02),
            fontsize=12,
            title_fontsize=12,
        )
        if i < n - 1:
            ax_bot.add_artist(leg)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return fig, (ax_bot, ax_top) if dual_xaxis else ax_bot


def plot_reward_components(
    results: dict,
    component_keys: list[tuple[str, str]],
    smooth_window: int = 10,
    cmap: str = "cividis",
    savefig: Optional[str] = None,
    figsize: Optional[tuple] = None,
):
    """
    Plot per-agent greedy reward components over training steps.

    results : dict
        The loaded experiment results dict.
    component_keys : list of (key, label) tuples
        Each entry names a key in `results` and the y-axis label to use.
    smooth_window : int
        Rolling-mean window size.
    """
    n = len(component_keys)
    if figsize is None:
        figsize = (12, 4 * n)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, dpi=150)
    if n == 1:
        axes = [axes]

    cmap_obj = plt.get_cmap(cmap)
    agents = sorted(next(iter(results[component_keys[0][0]].values())).keys())
    colors = [cmap_obj(x) for x in np.linspace(0.15, 0.85, len(agents))]

    for ax, (key, title) in zip(axes, component_keys):
        df = pd.DataFrame.from_dict(results[key], orient="index")
        df.index = df.index.astype(int)
        df = df.sort_index()

        for agent, color in zip(agents, colors):
            smoothed = df[agent].rolling(smooth_window, min_periods=1).mean()
            ax.plot(df.index, smoothed, label=agent, color=color, linewidth=2.0, alpha=0.85)
            ax.fill_between(df.index, smoothed, alpha=0.08, color=color)

        ax.set_ylabel(title, fontsize=14)
        ax.grid(linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=12)
        ax.legend(frameon=True, fontsize=11, loc="lower right")

    axes[-1].xaxis.set_major_formatter(FuncFormatter(_steps_fmt))
    axes[-1].set_xlabel("Environment Steps", fontsize=14)
    fig.suptitle("Greedy reward components", fontsize=16, y=1.01)
    fig.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    return fig, axes


def plot_lever_consistency_dual(
    lever_data: dict,
    agent_mask,
    action_mask,
    cmap: str = "ocean",
    savefig: Optional[str] = None,
    figsize: Optional[tuple] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    max_legend_labels: Optional[int] = 12,
    primary_label: str = "SS",
    secondary_label: str = "MS",
    dual_xaxis: bool = True,
    x_max: Optional[int] = None,
):
    """
    Plot per-agent lever effort for two runs across lever subplots.

    lever_data maps lever_name -> DataFrame or (first_df, second_df) tuple.
    First series solid on bottom x-axis; second dashed on top (when dual_xaxis=True).
    """
    if not isinstance(lever_data, dict) or not lever_data:
        raise ValueError("lever_data must be a non-empty dict")

    canonical_order = ["energy", "methane", "agriculture", "adaptation"]
    lever_order = [name for name in canonical_order if name in lever_data]

    action_mask = list(action_mask)
    if len(action_mask) < len(lever_order):
        action_mask += [0] * (len(lever_order) - len(action_mask))
    selected_levers = [name for name, flag in zip(lever_order, action_mask) if flag]
    if not selected_levers:
        raise ValueError("action_mask selects no levers to plot")

    def _unpack(val):
        if isinstance(val, pd.DataFrame):
            return (val, None)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return val[0], val[1]
        raise TypeError("Each lever value must be a DataFrame or a (first_df, second_df) tuple.")

    df1, df2 = _unpack(lever_data[selected_levers[0]])
    ref_df = df1 if isinstance(df1, pd.DataFrame) else df2
    agent_cols = list(ref_df.columns)

    agent_mask = list(agent_mask)
    if len(agent_mask) < len(agent_cols):
        agent_mask += [0] * (len(agent_cols) - len(agent_mask))
    selected_agents = [col for col, flag in zip(agent_cols, agent_mask) if flag]

    def _prep(df):
        out = df.copy()
        out.index = pd.Index(out.index).astype(int)
        out = out.sort_index()
        if x_max is not None:
            out = out[out.index <= x_max]
        cols = [c for c in selected_agents if c in out.columns]
        return out[cols]

    cmap_obj = plt.get_cmap(cmap)
    colors = {a: cmap_obj(x) for a, x in zip(selected_agents, np.linspace(0.15, 0.85, len(selected_agents)))}

    n_rows = len(selected_levers)
    if figsize is None:
        figsize = (12.0, max(5, 3.5 * n_rows))

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize)
    if n_rows == 1:
        axes = [axes]

    axes_top = [ax.twiny() for ax in axes] if dual_xaxis else [None] * n_rows

    first_xmax, second_xmax = 0, 0
    panel_ymins, panel_ymaxs = [], []

    def _truncate(pairs, maxn):
        if maxn is None or len(pairs) <= maxn:
            return pairs
        shown = pairs[:maxn]
        dummy = plt.Line2D([], [], color='none')
        shown.append((dummy, f"+{len(pairs)-maxn} more"))
        return shown

    for ax_bot, ax_top, lever in zip(axes, axes_top, selected_levers):
        df_first, df_second = _unpack(lever_data[lever])
        is_last = lever == selected_levers[-1]

        first_pairs, second_pairs = [], []
        yvals = []

        if isinstance(df_first, pd.DataFrame):
            data = _prep(df_first)
            first_xmax = max(first_xmax, int(data.index.max()))
            for agent in selected_agents:
                (h,) = ax_bot.plot(data.index, data[agent], color=colors[agent], lw=2.0, alpha=0.85)
                first_pairs.append((h, agent))
                yvals.extend([data[agent].min(), data[agent].max()])

        if isinstance(df_second, pd.DataFrame):
            data = _prep(df_second)
            second_xmax = max(second_xmax, int(data.index.max()))
            ax_sec = ax_top if dual_xaxis else ax_bot
            for agent in selected_agents:
                (h,) = ax_sec.plot(data.index, data[agent], color=colors[agent], lw=2.0, ls="--", alpha=0.9)
                second_pairs.append((h, agent))
                yvals.extend([data[agent].min(), data[agent].max()])

        if yvals:
            panel_ymins.append(min(yvals)); panel_ymaxs.append(max(yvals))

        ax_bot.set_ylabel(f"{lever.title()} (fraction)", fontsize=14)
        ax_bot.grid(True, linestyle=":", alpha=0.5)
        ax_bot.tick_params(axis="y", labelsize=14)

        if is_last:
            pairs1 = _truncate(first_pairs, max_legend_labels)
            pairs2 = _truncate(second_pairs, max_legend_labels)
            leg1 = ax_bot.legend([h for h, _ in pairs1], [lab for _, lab in pairs1],
                                  title=primary_label, frameon=True,
                                  loc="lower left", bbox_to_anchor=(0.67, 0.05),
                                  fontsize=14, title_fontsize=14)
            ax_bot.add_artist(leg1)
            ax_bot.legend([h for h, _ in pairs2], [lab for _, lab in pairs2],
                          title=secondary_label, frameon=True,
                          loc="lower right", bbox_to_anchor=(1.00, 0.05),
                          fontsize=14, title_fontsize=14)

    bot_lim = x_max if x_max is not None else first_xmax
    top_lim = x_max if x_max is not None else second_xmax

    for i, (ax_bot, ax_top) in enumerate(zip(axes, axes_top)):
        ax_bot.set_xlim(0, bot_lim)
        ax_bot.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax_bot.xaxis.set_major_formatter(FuncFormatter(_steps_fmt))
        ax_bot.margins(x=0)
        ax_bot.tick_params(axis="x", labelsize=14)

        if dual_xaxis:
            ax_top.set_xlim(0, top_lim)
            ax_top.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax_top.xaxis.set_major_formatter(FuncFormatter(_steps_fmt))
            ax_top.tick_params(axis="x", labelsize=14)

        if i != n_rows - 1:
            ax_bot.tick_params(axis="x", labelbottom=False)
        if dual_xaxis and i != 0:
            ax_top.tick_params(axis="x", labeltop=False)

    if dual_xaxis:
        axes[-1].set_xlabel(f"Environment steps — {primary_label}", fontsize=14)
        axes_top[0].set_xlabel(f"Environment steps — {secondary_label}", fontsize=14)
    else:
        axes[-1].set_xlabel("Environment steps", fontsize=14)

    if y_min is not None and y_max is not None:
        for ax in axes:
            ax.set_ylim(y_min, y_max)
    else:
        for ax, ymin, ymax in zip(axes, panel_ymins, panel_ymaxs):
            ax.set_ylim(y_min if y_min is not None else ymin,
                        y_max if y_max is not None else ymax)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    return fig, (axes, axes_top)