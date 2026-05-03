"""
Bivariate Analysis using AI (Groq LLM)
======================================
Reads a CSV dataset, computes statistical summaries for every possible
column pair, sends them to an LLM to uncover hidden relationships,
and writes the insights to both JSON and CSV output files.
"""

import os
import sys
import json
import csv
import time
import itertools
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit("Error: Set GROQ_API_KEY in your .env file first.")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
    timeout=120.0,
)

MODEL = "llama-3.3-70b-versatile"
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "synthetic_data_30012026_v4.csv")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "bivariate_insights.json")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "bivariate_insights.csv")
OUTPUT_CROSSTABS = os.path.join(os.path.dirname(__file__), "bivariate_crosstabs.csv")

# Max bins for decision-tree optimal binning of numeric features
MAX_BINS = 6
# Minimum fraction of total rows a category must have to be kept (else → "Other")
CATEGORY_THRESHOLD = 0.20

# Maximum sample rows sent to the LLM per pair (keeps token usage reasonable)
SAMPLE_ROWS = 20
# How many column pairs to send per LLM call (batching to reduce API calls)
PAIRS_PER_BATCH = 15
# Columns to skip (identifiers / dates that aren't useful for bivariate stats)
SKIP_COLUMNS = {"service_id", "EDD"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def classify_column(series: pd.Series) -> str:
    """Return 'numeric' or 'categorical'."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def pair_summary(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Build a compact statistical summary for one column pair."""
    type_a = classify_column(df[col_a])
    type_b = classify_column(df[col_b])

    summary = {
        "column_a": col_a,
        "column_b": col_b,
        "type_a": type_a,
        "type_b": type_b,
        "rows": len(df),
    }

    sample = df[[col_a, col_b]].dropna().head(SAMPLE_ROWS)

    if type_a == "numeric" and type_b == "numeric":
        corr = df[[col_a, col_b]].corr().iloc[0, 1]
        summary["pearson_correlation"] = round(corr, 4) if not np.isnan(corr) else None
        summary["col_a_mean"] = round(df[col_a].mean(), 4)
        summary["col_b_mean"] = round(df[col_b].mean(), 4)
        summary["col_a_std"] = round(df[col_a].std(), 4)
        summary["col_b_std"] = round(df[col_b].std(), 4)
    elif type_a == "categorical" and type_b == "categorical":
        ct = pd.crosstab(df[col_a], df[col_b])
        summary["crosstab_shape"] = f"{ct.shape[0]}x{ct.shape[1]}"
        summary["col_a_unique"] = int(df[col_a].nunique())
        summary["col_b_unique"] = int(df[col_b].nunique())
    else:
        # mixed: one numeric, one categorical
        num_col = col_a if type_a == "numeric" else col_b
        cat_col = col_b if type_a == "numeric" else col_a
        group_means = df.groupby(cat_col)[num_col].mean()
        summary["group_means"] = {str(k): round(v, 4) for k, v in group_means.items()}
        summary["numeric_std"] = round(df[num_col].std(), 4)

    # Round floats in sample to keep prompt compact
    raw_sample = sample.values.tolist()[:5]
    compact = []
    for row in raw_sample:
        compact.append([round(v, 4) if isinstance(v, float) else v for v in row])
    summary["sample_values"] = compact
    return summary


def build_prompt(batch: list[dict]) -> str:
    """Create the AI prompt for a batch of column-pair summaries."""
    pairs_json = json.dumps(batch, indent=2, default=str)

    prompt = f"""You are a senior data scientist performing bivariate analysis.

Below is a list of column-pair statistical summaries from a telecom customer dataset.
Each entry contains: column names, data types, correlation / crosstab / group-means,
and a small sample of values.

COLUMN-PAIR SUMMARIES:
{pairs_json}

YOUR TASK:
For EACH pair, determine whether a meaningful hidden relationship exists.
Consider: correlation strength, non-linear patterns, Simpson's paradox,
confounding, interaction effects, and business implications.

RULES:
1. Only report pairs where you find a noteworthy insight (skip boring/obvious ones).
2. For each insightful pair, output EXACTLY this JSON structure:
   {{
     "feature1": "<column name>",
     "feature2": "<column name>",
     "insight": "<1-2 sentence explanation of the hidden relationship>"
   }}
3. Return a JSON array of these objects. If no pair is interesting, return [].
4. Output ONLY valid JSON—no markdown, no commentary.
"""
    return prompt


def build_ranking_prompt(insights: list[dict]) -> str:
    """Create a prompt that asks the LLM to pick the top 5 most important pairs."""
    summary = json.dumps(
        [{"feature1": i["feature1"], "feature2": i["feature2"], "insight": i["insight"]} for i in insights],
        indent=2,
    )

    return f"""You are a senior data scientist reviewing bivariate analysis results for a telecom customer dataset.

Below are {len(insights)} feature-pair insights discovered from analyzing every possible column pair.
Your job is to select the TOP 5 MOST IMPORTANT pairs — the ones that would matter most
for understanding customer behaviour, predicting churn, or driving business decisions.

ALL DISCOVERED INSIGHTS:
{summary}

YOUR TASK:
Pick exactly 5 pairs that are the most impactful and actionable.
For each, write a detailed insight (3-5 sentences) that explains:
  - What the relationship is
  - Why this pair is important for the business
  - What action or investigation it suggests

RULES:
1. Return EXACTLY 5 objects.
2. For each, output this JSON structure:
   {{
     "feature1": "<column name>",
     "feature2": "<column name>",
     "insight": "<3-5 sentence detailed explanation of why this pair is important>"
   }}
3. Return a JSON array of these 5 objects.
4. Output ONLY valid JSON — no markdown, no commentary.
"""


def call_llm(prompt: str) -> str:
    """Send prompt to Groq and return raw content."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a data analysis expert. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return resp.choices[0].message.content.strip()


def parse_llm_json(raw: str) -> list[dict]:
    """Robustly extract a JSON array from LLM output."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    print(f"  WARNING: Could not parse LLM response as JSON")
    return []


def bucket_numeric(series: pd.Series, other_series: pd.Series = None, max_bins: int = MAX_BINS) -> pd.Series:
    """Dynamically bucket a numeric column using decision-tree-based optimal binning.

    If `other_series` is provided (the paired column), uses a DecisionTree to find
    optimal split points that maximise separation w.r.t. the other variable.
    Falls back to KMeans-style binning when tree binning isn't feasible.
    """
    clean = series.dropna()
    if len(clean.unique()) <= max_bins:
        # Already few enough unique values — treat as-is
        return series.astype(str)

    # Try supervised tree-based binning when a paired column is available
    if other_series is not None:
        try:
            mask = series.notna() & other_series.notna()
            X = series[mask].values.reshape(-1, 1)
            # For numeric targets: bin the target into quantile classes for the tree
            y_raw = other_series[mask]
            if pd.api.types.is_numeric_dtype(y_raw):
                y = pd.qcut(y_raw, q=min(5, y_raw.nunique()), labels=False, duplicates="drop")
            else:
                y = y_raw.astype("category").cat.codes

            tree = DecisionTreeClassifier(
                max_leaf_nodes=max_bins,
                min_samples_leaf=max(50, int(len(X) * 0.02)),
            )
            tree.fit(X, y)

            # Extract thresholds from the tree
            thresholds = sorted(set(tree.tree_.threshold[tree.tree_.threshold != -2]))
            if len(thresholds) >= 2:
                edges = [float("-inf")] + thresholds + [float("inf")]
                binned = pd.cut(series, bins=edges, duplicates="drop")
                # Rename with cleaner labels
                labels = []
                for iv in binned.cat.categories:
                    lo = f"{iv.left:.1f}" if iv.left != float("-inf") else "min"
                    hi = f"{iv.right:.1f}" if iv.right != float("inf") else "max"
                    labels.append(f"{lo} – {hi}")
                binned = binned.cat.rename_categories(dict(zip(binned.cat.categories, labels)))
                return binned.astype(str)
        except Exception:
            pass  # Fall through to KMeans fallback

    # Fallback: KMeans-style binning via sklearn
    try:
        from sklearn.preprocessing import KBinsDiscretizer
        kbd = KBinsDiscretizer(n_bins=max_bins, encode="ordinal", strategy="kmeans", subsample=None)
        vals = series.dropna().values.reshape(-1, 1)
        kbd.fit(vals)
        edges = [float("-inf")] + list(kbd.bin_edges_[0][1:-1]) + [float("inf")]
        binned = pd.cut(series, bins=edges, duplicates="drop")
        labels = []
        for iv in binned.cat.categories:
            lo = f"{iv.left:.1f}" if iv.left != float("-inf") else "min"
            hi = f"{iv.right:.1f}" if iv.right != float("inf") else "max"
            labels.append(f"{lo} – {hi}")
        binned = binned.cat.rename_categories(dict(zip(binned.cat.categories, labels)))
        return binned.astype(str)
    except Exception:
        # Last resort: simple quantile
        try:
            binned = pd.qcut(series, q=max_bins, duplicates="drop")
        except ValueError:
            binned = pd.cut(series, bins=max_bins, duplicates="drop")
        return binned.astype(str)


def filter_categories(series: pd.Series, threshold: float = CATEGORY_THRESHOLD) -> pd.Series:
    """Keep only categories with >= threshold fraction of rows; rest become 'Other'."""
    counts = series.value_counts(normalize=True)
    keep = set(counts[counts >= threshold].index)
    if len(keep) == 0:
        # If no category meets the threshold, keep the top ones that together cover >50%
        cumsum = counts.cumsum()
        keep = set(cumsum[cumsum <= 0.8].index)
        if len(keep) == 0:
            keep = {counts.index[0]}  # at least keep the largest
    return series.apply(lambda x: x if x in keep else "Other")


def compute_crosstab(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Compute a crosstab for a pair of features, using optimal binning for numerics
    and threshold filtering for categoricals."""
    type_a = classify_column(df[col_a])
    type_b = classify_column(df[col_b])

    if type_a == "numeric":
        series_a = bucket_numeric(df[col_a], other_series=df[col_b])
    else:
        series_a = filter_categories(df[col_a].astype(str))

    if type_b == "numeric":
        series_b = bucket_numeric(df[col_b], other_series=df[col_a])
    else:
        series_b = filter_categories(df[col_b].astype(str))

    ct = pd.crosstab(series_a, series_b)

    # Convert to nested dict: {row_label: {col_label: count}}
    return ct.to_dict(orient="index")


def normalize_insight(raw: dict) -> dict:
    """Ensure every insight has exactly the 3 keys: feature1, feature2, insight."""
    return {
        "feature1": raw.get("feature1") or raw.get("column_a", ""),
        "feature2": raw.get("feature2") or raw.get("column_b", ""),
        "insight": raw.get("insight", ""),
    }


def save_results(insights: list[dict], df: pd.DataFrame):
    """Write insights + crosstabs to JSON, CSV, and a separate crosstabs CSV."""
    clean = [normalize_insight(i) for i in insights]

    # Compute crosstab for each insight pair
    print("\n  Computing crosstabs for insight pairs...")
    for item in clean:
        f1, f2 = item["feature1"], item["feature2"]
        if f1 in df.columns and f2 in df.columns:
            item["crosstab"] = compute_crosstab(df, f1, f2)
        else:
            item["crosstab"] = {}

    # JSON output (includes full crosstab dict)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved -> {OUTPUT_JSON}")

    if clean:
        # Main CSV output (crosstab as compact JSON string in a column)
        fieldnames = ["feature1", "feature2", "insight", "crosstab"]
        csv_rows = []
        for item in clean:
            csv_rows.append({
                "feature1": item["feature1"],
                "feature2": item["feature2"],
                "insight": item["insight"],
                "crosstab": json.dumps(item.get("crosstab", {}), default=str),
            })
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  CSV  saved -> {OUTPUT_CSV}")

        # Separate readable crosstabs CSV — one section per pair
        with open(OUTPUT_CROSSTABS, "w", encoding="utf-8", newline="") as f:
            for item in clean:
                f1, f2 = item["feature1"], item["feature2"]
                ct = item.get("crosstab", {})
                if not ct:
                    continue
                f.write(f"\n--- {f1} vs {f2} ---\n")
                f.write(f"Insight: {item['insight']}\n\n")
                col_labels = sorted({c for row in ct.values() for c in row})
                f.write(f"{f1} \\ {f2}," + ",".join(str(c) for c in col_labels) + "\n")
                for row_label in sorted(ct.keys()):
                    vals = ",".join(str(ct[row_label].get(c, 0)) for c in col_labels)
                    f.write(f"{row_label},{vals}\n")
        print(f"  Crosstabs saved -> {OUTPUT_CROSSTABS}")
    else:
        print("  No insights to write.")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BIVARIATE ANALYSIS — AI-Powered Hidden Relationships")
    print("=" * 60)

    # 1. Load data
    print(f"\n  Loading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # 2. Select columns
    columns = [c for c in df.columns if c not in SKIP_COLUMNS]
    print(f"  Analysable columns ({len(columns)}): {columns[:8]}{'...' if len(columns) > 8 else ''}")

    # 3. Generate all pairs
    all_pairs = list(itertools.combinations(columns, 2))
    print(f"  Total column pairs: {len(all_pairs)}")

    # 4. Compute summaries
    print("\n  Computing pair-wise statistics...")
    summaries = []
    for col_a, col_b in all_pairs:
        summaries.append(pair_summary(df, col_a, col_b))

    # 5. Batch-send to LLM
    all_insights = []
    batches = [summaries[i : i + PAIRS_PER_BATCH] for i in range(0, len(summaries), PAIRS_PER_BATCH)]
    print(f"  Sending {len(batches)} batches to LLM ({PAIRS_PER_BATCH} pairs/batch)...\n")

    for idx, batch in enumerate(batches, 1):
        pair_names = [f"({s['column_a']}, {s['column_b']})" for s in batch]
        print(f"  Batch {idx}/{len(batches)}: {pair_names[0]} ... {pair_names[-1]}")

        prompt = build_prompt(batch)
        try:
            raw = call_llm(prompt)
            insights = parse_llm_json(raw)
            all_insights.extend(insights)
            print(f"    -> {len(insights)} insights found")
        except Exception as e:
            print(f"    -> ERROR: {e}")

        # Rate-limit courtesy pause between batches
        if idx < len(batches):
            time.sleep(2)

    # 6. Rank and select top 5
    print(f"\n  Total insights discovered: {len(all_insights)}")
    if len(all_insights) > 5:
        print("  Asking LLM to pick the top 5 most important pairs...")
        normalized = [normalize_insight(i) for i in all_insights]
        ranking_prompt = build_ranking_prompt(normalized)
        try:
            raw = call_llm(ranking_prompt)
            top5 = parse_llm_json(raw)
            if len(top5) >= 1:
                all_insights = top5[:5]
                print(f"  -> Selected {len(all_insights)} top pairs")
            else:
                print("  -> Ranking failed, keeping first 5 insights")
                all_insights = all_insights[:5]
        except Exception as e:
            print(f"  -> Ranking error: {e}, keeping first 5 insights")
            all_insights = all_insights[:5]

    # 7. Save results
    save_results(all_insights, df)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
