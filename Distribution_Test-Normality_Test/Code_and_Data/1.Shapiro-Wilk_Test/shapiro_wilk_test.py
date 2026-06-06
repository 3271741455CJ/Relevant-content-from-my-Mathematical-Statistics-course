# -*- coding: utf-8 -*-
"""
Shapiro-Wilk W test and normal Q-Q plot.

Read CSV data in the prescribed format, draw a Q-Q plot, compute the W statistic
step by step, and compare W_0 with the alpha quantile W_alpha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, shapiro

MISSING_TOKENS = frozenset(
    {"", "-", "\u2014", "\u2013", "\u2015", "N/A", "NA", "nan"}
)


def _resolve_path(work_dir: Path, user_path: str) -> Path:
    p = Path(user_path.strip().strip('"').strip("'"))
    if p.is_absolute():
        return p
    return (work_dir / p).resolve()


def _parse_numeric(value) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in MISSING_TOKENS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    """Try common encodings (UTF-8, GBK) for Windows-exported CSV files."""
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin1")
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(
                csv_path,
                header=None,
                encoding=enc,
                engine="python",
            )
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        getattr(last_error, "encoding", "unknown"),
        getattr(last_error, "object", b""),
        getattr(last_error, "start", 0),
        getattr(last_error, "end", 1),
        f"Cannot decode {csv_path} with {encodings}",
    )


def load_wilk_data(csv_path: Path) -> pd.DataFrame:
    """
    Columns (no header):
        0: i
        1: x_(i)
        2: x_(n-i+1)
        3: x_(n-i+1) - x_(i)
        4: a_i
        5: optional check column a_i * diff
    """
    df = _read_csv_with_fallback(csv_path)
    if df.shape[1] < 5:
        raise ValueError(f"Need at least 5 columns, got {df.shape[1]}.")
    df.columns = ["i", "x_i", "x_ni", "diff", "a_i"] + [
        f"col_{j}" for j in range(5, df.shape[1])
    ]
    return df


def infer_sample_size(df: pd.DataFrame) -> int:
    m = len(df)
    last_x_ni = _parse_numeric(df.iloc[-1]["x_ni"])
    if last_x_ni is None:
        return 2 * m - 1
    return 2 * m


def reconstruct_order_statistics(df: pd.DataFrame, n: int) -> np.ndarray:
    order_stats = np.empty(n, dtype=np.float64)
    m = len(df)

    for row_idx in range(m):
        i = int(df.iloc[row_idx]["i"])
        x_i = _parse_numeric(df.iloc[row_idx]["x_i"])
        x_ni = _parse_numeric(df.iloc[row_idx]["x_ni"])

        if x_i is None:
            raise ValueError(f"Row {row_idx + 1}: cannot parse x_(i).")

        order_stats[i - 1] = x_i
        if x_ni is not None:
            order_stats[n - i] = x_ni

    if np.any(np.isnan(order_stats)):
        missing = np.where(np.isnan(order_stats))[0] + 1
        raise ValueError(f"Incomplete order statistics at indices: {missing.tolist()}")

    if not np.all(order_stats[:-1] <= order_stats[1:]):
        print("Warning: reconstructed order statistics are not strictly increasing.")

    return order_stats


def expected_normal_order_stats(n: int) -> np.ndarray:
    """Blom plotting positions for standard normal order-statistic expectations."""
    i = np.arange(1, n + 1, dtype=np.float64)
    probs = (i - 0.375) / (n + 0.25)
    return norm.ppf(probs)


def plot_qq(order_stats: np.ndarray, save_path: Path) -> None:
    n = len(order_stats)
    theoretical = expected_normal_order_stats(n)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(theoretical, order_stats, color="#2563eb", edgecolors="white", s=55, zorder=3)

    slope, intercept = np.polyfit(theoretical, order_stats, 1)
    x_line = np.array([theoretical.min(), theoretical.max()])
    ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.2, label="\u53c2\u8003\u76f4\u7ebf")

    ax.set_xlabel("\u6807\u51c6\u6b63\u6001\u5206\u5e03\u6b21\u5e8f\u7edf\u8ba1\u91cf\u7684\u671f\u671b\u503c")
    ax.set_ylabel("\u6837\u672c\u6b21\u5e8f\u7edf\u8ba1\u91cf\uff08\u6837\u672c\u5206\u4f4d\u6570\uff09")
    ax.set_title("\u6b63\u6001\u6982\u7387\u56fe\uff08Q-Q \u56fe\uff09")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def royston_shapiro_pvalue(w: float, n: int) -> float:
    """p-value from W and n using Royston (1995), same approximation as scipy.stats.shapiro."""
    if not (0.0 < w < 1.0):
        return 0.0 if w <= 0.0 else 1.0

    if n <= 11:
        mu = -0.0006714 * n**3 + 0.025054 * n**2 - 0.39978 * n + 0.5440
        sigma = np.exp(
            -0.0020322 * n**3 + 0.062767 * n**2 - 0.77857 * n + 1.3822
        )
    elif n <= 2000:
        ln_n = np.log(n)
        mu = (
            0.0038915 * ln_n**3
            - 0.083751 * ln_n**2
            - 0.31082 * ln_n
            - 1.5861
        )
        sigma = np.exp(0.0030302 * ln_n**2 - 0.082676 * ln_n - 0.4803)
    else:
        mu, sigma = 0.0, 1.0

    z = (np.log(1.0 - w) - mu) / sigma
    return float(1.0 - norm.cdf(z))


def shapiro_w_alpha(n: int, alpha: float) -> float:
    """Alpha quantile W_alpha with P(W <= W_alpha) = alpha under H0."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if royston_shapiro_pvalue(mid, n) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_w_statistic(
    df: pd.DataFrame, order_stats: np.ndarray, n: int
) -> tuple[float, list[str], dict]:
    lines: list[str] = []
    m = len(df)

    lines.append("=" * 60)
    lines.append("W \u68c0\u9a8c\u7edf\u8ba1\u91cf W_0 \u7684\u9010\u6b65\u8ba1\u7b97")
    lines.append("=" * 60)
    lines.append(f"\u6837\u672c\u5bb9\u91cf n = {n}")
    lines.append(f"\u5faa\u73af\u6c42\u548c\u9879\u6570 m = {m}")
    lines.append("")
    lines.append("W_0 = [sum a_i*(x_(n-i+1) - x_(i))]^2 / sum(x_j - x_bar)^2")
    lines.append("")

    weighted_sum = 0.0
    header = (
        f"{'i':>3} {'x_(i)':>10} {'x_(n-i+1)':>12} "
        f"{'diff':>10} {'a_i':>10} {'a_i*diff':>12}"
    )
    lines.append(header)
    lines.append("-" * 62)

    for row_idx in range(m):
        i = int(df.iloc[row_idx]["i"])
        x_i = _parse_numeric(df.iloc[row_idx]["x_i"])
        x_ni = _parse_numeric(df.iloc[row_idx]["x_ni"])
        diff_file = _parse_numeric(df.iloc[row_idx]["diff"])
        a_i = _parse_numeric(df.iloc[row_idx]["a_i"])

        if x_i is None or a_i is None:
            raise ValueError(f"Row {row_idx + 1}: missing x_(i) or a_i.")

        if x_ni is None:
            diff = 0.0
        else:
            diff = x_ni - x_i
            if diff_file is not None and not np.isclose(diff, diff_file, rtol=0, atol=1e-3):
                lines.append(
                    f"Note row {i}: file diff {diff_file:.4f} vs computed {diff:.4f}; "
                    "using computed value."
                )

        term = a_i * diff
        weighted_sum += term
        x_ni_display = f"{x_ni:10.4f}" if x_ni is not None else f"{'N/A':>10}"
        lines.append(
            f"{i:3d} {x_i:10.4f} {x_ni_display:>12} {diff:10.4f} {a_i:10.4f} {term:12.4f}"
        )

    lines.append("-" * 62)
    lines.append(f"sum a_i*(x_(n-i+1) - x_(i)) = {weighted_sum:.6f}")
    numerator = weighted_sum**2
    lines.append(f"numerator [sum]^2 = {numerator:.6f}")
    lines.append("")

    x_bar = float(np.mean(order_stats))
    deviations = order_stats - x_bar
    squared_deviations = deviations**2
    denominator = float(np.sum(squared_deviations))

    lines.append(f"x_bar = {x_bar:.6f}")
    lines.append("")
    lines.append(f"{'j':>3} {'x_(j)':>10} {'x_(j)-x_bar':>12} {'[x_(j)-x_bar]^2':>14}")
    lines.append("-" * 44)
    for j in range(n):
        lines.append(
            f"{j + 1:3d} {order_stats[j]:10.4f} {deviations[j]:12.4f} {squared_deviations[j]:14.4f}"
        )
    lines.append("-" * 44)
    lines.append(f"denominator sum(x_j - x_bar)^2 = {denominator:.6f}")
    lines.append("")

    w0 = numerator / denominator
    lines.append(f"W_0 = {numerator:.6f} / {denominator:.6f} = {w0:.6f}")

    details = {
        "weighted_sum": weighted_sum,
        "numerator": numerator,
        "denominator": denominator,
        "x_bar": x_bar,
        "w0": w0,
    }
    return w0, lines, details


def compare_with_critical_value(
    w0: float, n: int, alpha: float
) -> tuple[float, list[str], bool, float]:
    w_alpha = shapiro_w_alpha(n, alpha)
    p_value = royston_shapiro_pvalue(w0, n)

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("W \u68c0\u9a8c\u7ed3\u8bba")
    lines.append("=" * 60)
    lines.append(f"alpha = {alpha}")
    lines.append(f"W_0 = {w0:.6f}")
    lines.append(f"W_alpha = {w_alpha:.6f}")
    lines.append("(Royston approximation, consistent with scipy.stats.shapiro)")
    lines.append(f"p-value ~ {p_value:.6f}")
    lines.append("")

    if w0 < w_alpha:
        lines.append(f"Compare: W_0 ({w0:.6f}) < W_alpha ({w_alpha:.6f})")
        lines.append(
            f"Result: reject H0 at alpha = {alpha} (data not consistent with normality)."
        )
        reject = True
    else:
        lines.append(f"Compare: W_0 ({w0:.6f}) >= W_alpha ({w_alpha:.6f})")
        lines.append(
            f"Result: do not reject H0 at alpha = {alpha} (no strong evidence against normality)."
        )
        reject = False

    lines.append("")
    lines.append("--- scipy.stats.shapiro cross-check ---")
    return w_alpha, lines, reject, p_value


def main() -> None:
    print("=" * 60)
    print("  Shapiro-Wilk W test and Q-Q plot")
    print("=" * 60)
    print()

    work_dir_input = input("\u8bf7\u8f93\u5165\u5de5\u4f5c\u76ee\u5f55\u8def\u5f84: ").strip()
    if not work_dir_input:
        work_dir = Path.cwd()
    else:
        work_dir = Path(work_dir_input).expanduser().resolve()
        if not work_dir.is_dir():
            print(f"Error: work directory not found: {work_dir}")
            sys.exit(1)

    alpha_input = input("\u8bf7\u8f93\u5165\u663e\u8457\u6027\u6c34\u5e73 alpha (\u4f8b\u5982 0.05): ").strip()
    try:
        alpha = float(alpha_input)
    except ValueError:
        print("Error: alpha must be numeric.")
        sys.exit(1)

    data_input = input("\u8bf7\u8f93\u5165\u539f\u59cb\u6570\u636e CSV \u6587\u4ef6\u8def\u5f84: ").strip()
    result_input = input("\u8bf7\u8f93\u5165\u7ed3\u679c\u4fdd\u5b58\u76ee\u5f55\u8def\u5f84: ").strip()

    data_path = _resolve_path(work_dir, data_input)
    result_dir = _resolve_path(work_dir, result_input)

    if not data_path.is_file():
        print(f"Error: data file not found: {data_path}")
        sys.exit(1)

    result_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Work dir: {work_dir}")
    print(f"Data file: {data_path}")
    print(f"Result dir: {result_dir}")
    print(f"alpha: {alpha}")
    print()

    df = load_wilk_data(data_path)
    n = infer_sample_size(df)
    order_stats = reconstruct_order_statistics(df, n)

    print(f"n = {n}")
    print("Order statistics:")
    print(", ".join(f"{v:.4f}" for v in order_stats))
    print()

    qq_path = result_dir / "qq_plot.png"
    plot_qq(order_stats, qq_path)
    print(f"Q-Q plot saved: {qq_path}")

    w0, calc_lines, _ = compute_w_statistic(df, order_stats, n)
    for line in calc_lines:
        print(line)

    _, conclusion_lines, _, _ = compare_with_critical_value(w0, n, alpha)
    for line in conclusion_lines:
        print(line)

    w_scipy, p_scipy = shapiro(order_stats)
    print(f"scipy W = {w_scipy:.6f}, scipy p = {p_scipy:.6f}")

    report_path = result_dir / "w_test_report.txt"
    report_content = [
        "Shapiro-Wilk W test report",
        f"Data file: {data_path}",
        f"n = {n}",
        f"alpha = {alpha}",
        "",
        "Order statistics:",
        ", ".join(f"{v:.6f}" for v in order_stats),
        "",
        *calc_lines,
        *conclusion_lines,
        f"scipy.stats.shapiro: W = {w_scipy:.6f}, p = {p_scipy:.6f}",
        f"Q-Q plot: {qq_path}",
    ]
    report_path.write_text("\n".join(report_content), encoding="utf-8")

    print()
    print(f"Report saved: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
