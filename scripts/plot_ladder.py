"""Plot English and Arabic recall@5 across the retrieval ladder, from a committed results file.

    pip install matplotlib
    python scripts/plot_ladder.py --corpus unpc --out docs/figures/ladder-unpc.png

Absolute scores, not the difference: the gap is the shaded band between the two lines, so a gap that
closes because the ceiling was reached looks different from one that closes because Arabic improved. On
XQuAD both lines meet at 1.000; on the UN corpus they do not meet at all.

Matplotlib is not a dependency of the pipeline — it is needed only to regenerate the figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The ladder, in the order the write-up discusses it.
LADDER = [
    ("bm25-raw", "BM25\nraw"),
    ("bm25-norm", "BM25\nnormalized"),
    ("bm25-light", "BM25\nstemmed"),
    ("e5-base", "dense\ne5-base"),
    ("bge-m3", "dense\nbge-m3"),
    ("hybrid_bm25-light__e5-base", "hybrid\n+ e5-base"),
    ("hybrid_bm25-light__bge-m3", "hybrid\n+ bge-m3"),
    ("rerank_hybrid_bm25-light__bge-m3", "reranked\nhybrid"),
]
EN_COLOUR, AR_COLOUR, GAP_COLOUR = "#1f4e79", "#c0504d", "#c0504d"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="unpc")
    parser.add_argument("--metric", default="recall@5")
    parser.add_argument("--results", default=None, help="default: data/results/<corpus>/retrieval.json")
    parser.add_argument("--out", default=None, help="default: docs/figures/ladder-<corpus>.png")
    args = parser.parse_args()

    results = Path(args.results or f"data/results/{args.corpus}/retrieval.json")
    out = Path(args.out or f"docs/figures/ladder-{args.corpus}.png")
    data = json.loads(results.read_text(encoding="utf-8"))
    table, diffs = data["table"], data["en_minus_ar"]

    configs = [(key, label) for key, label in LADDER if f"{key}/en-en" in table]
    x = list(range(len(configs)))
    en = [table[f"{key}/en-en"][args.metric] for key, _ in configs]
    ar = [table[f"{key}/ar-ar"][args.metric] for key, _ in configs]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    ax.fill_between(x, ar, en, color=GAP_COLOUR, alpha=0.12, linewidth=0, label="English − Arabic gap")
    ax.plot(x, en, "-o", color=EN_COLOUR, linewidth=2.2, markersize=6, label="English")
    ax.plot(x, ar, "-o", color=AR_COLOUR, linewidth=2.2, markersize=6, label="Arabic")

    for i, (key, _) in enumerate(configs):
        gap = diffs[key][args.metric]["mean_diff"] if key in diffs else en[i] - ar[i]
        ax.annotate(f"{gap:+.3f}", (i, (en[i] + ar[i]) / 2), ha="center", va="center",
                    fontsize=8.5, color="#7f2f2c",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.85))
    for i in (0, len(configs) - 1):
        ax.annotate(f"{en[i]:.3f}", (i, en[i]), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=EN_COLOUR, fontweight="bold")
        ax.annotate(f"{ar[i]:.3f}", (i, ar[i]), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color=AR_COLOUR, fontweight="bold")

    n = data.get("n_questions")
    ax.set_title(f"{args.metric} across the retrieval ladder — {data.get('corpus', args.corpus)}"
                 f"{f', n = {n}' if n else ''}", fontsize=12, pad=14)
    ax.set_ylabel(args.metric)
    ax.set_xticks(x, [label for _, label in configs], fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    ax.annotate("ceiling (1.000)", (len(configs) - 1, 1.0), textcoords="offset points", xytext=(0, 5),
                ha="right", fontsize=8, color="#666666")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"{out}  ({len(configs)} configurations, {args.metric}, from {results})")


if __name__ == "__main__":
    main()
