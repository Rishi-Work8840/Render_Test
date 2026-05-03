"""
Bivariate Analysis — Top 5 Insights (Standalone)
=================================================
End-to-end pipeline that:
  1. Loads a CSV dataset
  2. Computes statistical summaries for every possible column pair
  3. Sends summaries to an LLM in batches to discover relationships
  4. Asks the LLM to rank and select the TOP 5 most important pairs
  5. Computes crosstabs (with optimal binning) for those 5 pairs
  6. Saves results to JSON, CSV, and a readable crosstabs file
"""

import os
import sys
import json
import csv
import re
import time
import itertools
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
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

# Use smaller/faster model for discovery (Pass 1) to save tokens
MODEL_DISCOVERY = "llama-3.1-8b-instant"   # fast, cheap — just finding candidates
MODEL_RANKING   = "llama-3.3-70b-versatile" # smart — picks & explains top 5

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "synthetic_data_30012026_v4.csv")
OUT_DIR = os.path.dirname(__file__)

OUTPUT_JSON = os.path.join(OUT_DIR, "bivariate_insights.json")
OUTPUT_CSV = os.path.join(OUT_DIR, "bivariate_insights.csv")
OUTPUT_CROSSTABS = os.path.join(OUT_DIR, "bivariate_crosstabs.csv")

MAX_BINS = 6                  # max buckets for numeric columns in crosstabs
CATEGORY_THRESHOLD = 0.20     # categories below 20% are merged into "Other"
SAMPLE_ROWS = 3               # sample rows sent to LLM per pair (reduced from 5)
PAIRS_PER_BATCH = 40          # column pairs per LLM call (increased from 15)
SKIP_COLUMNS = {"service_id", "EDD"}

# Pre-filtering thresholds — skip boring pairs before calling LLM
MIN_ABS_CORRELATION = 0.05    # numeric-numeric: skip if |r| < this
CHI2_P_THRESHOLD = 0.05       # categorical-categorical: skip if p > this
MIN_GROUP_MEAN_CV = 0.02      # mixed: skip if coefficient of variation of group means < this


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: STATISTICAL SUMMARIES
# ═══════════════════════════════════════════════════════════════════════════════

def classify_column(series: pd.Series) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


def pair_summary(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Build a compact statistical summary for one column pair."""
    type_a = classify_column(df[col_a])
    type_b = classify_column(df[col_b])

    info = {
        "column_a": col_a,
        "column_b": col_b,
        "type_a": type_a,
        "type_b": type_b,
        "rows": len(df),
    }

    if type_a == "numeric" and type_b == "numeric":
        corr = df[[col_a, col_b]].corr().iloc[0, 1]
        info["pearson_correlation"] = round(corr, 4) if not np.isnan(corr) else None
        info["col_a_mean"] = round(df[col_a].mean(), 4)
        info["col_b_mean"] = round(df[col_b].mean(), 4)
        info["col_a_std"] = round(df[col_a].std(), 4)
        info["col_b_std"] = round(df[col_b].std(), 4)
    elif type_a == "categorical" and type_b == "categorical":
        ct = pd.crosstab(df[col_a], df[col_b])
        info["crosstab_shape"] = f"{ct.shape[0]}x{ct.shape[1]}"
        info["col_a_unique"] = int(df[col_a].nunique())
        info["col_b_unique"] = int(df[col_b].nunique())
    else:
        num_col = col_a if type_a == "numeric" else col_b
        cat_col = col_b if type_a == "numeric" else col_a
        group_means = df.groupby(cat_col)[num_col].mean()
        info["group_means"] = {str(k): round(v, 4) for k, v in group_means.items()}
        info["numeric_std"] = round(df[num_col].std(), 4)

    sample = df[[col_a, col_b]].dropna().head(SAMPLE_ROWS).values.tolist()
    info["sample"] = [
        [round(v, 4) if isinstance(v, float) else v for v in row]
        for row in sample
    ]
    return info


def is_interesting_pair(df: pd.DataFrame, col_a: str, col_b: str) -> bool:
    """Pre-filter: return True only if this pair shows enough statistical signal
    to be worth sending to the LLM. This eliminates ~50-60% of boring pairs."""
    type_a = classify_column(df[col_a])
    type_b = classify_column(df[col_b])

    try:
        if type_a == "numeric" and type_b == "numeric":
            corr = df[[col_a, col_b]].corr().iloc[0, 1]
            return not np.isnan(corr) and abs(corr) >= MIN_ABS_CORRELATION

        elif type_a == "categorical" and type_b == "categorical":
            ct = pd.crosstab(df[col_a], df[col_b])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                return False
            _, p, _, _ = chi2_contingency(ct)
            return p < CHI2_P_THRESHOLD

        else:
            # Mixed: check if group means actually vary
            num_col = col_a if type_a == "numeric" else col_b
            cat_col = col_b if type_a == "numeric" else col_a
            group_means = df.groupby(cat_col)[num_col].mean()
            if len(group_means) < 2:
                return False
            cv = group_means.std() / abs(group_means.mean()) if group_means.mean() != 0 else 0
            return cv >= MIN_GROUP_MEAN_CV
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: LLM — DISCOVER INSIGHTS (BATCHED)
# ═══════════════════════════════════════════════════════════════════════════════

def build_discovery_prompt(batch: list[dict]) -> str:
    pairs_json = json.dumps(batch, default=str)  # compact, no indent
    return f"""You are a data scientist. Below are {len(batch)} column-pair stats from a telecom dataset.

{pairs_json}

For each pair with a noteworthy relationship, return:
{{"feature1":"<col>","feature2":"<col>","insight":"<1 sentence>"}}
Return a JSON array. Skip boring pairs. If none interesting, return [].
Output ONLY valid JSON.
"""


def build_ranking_prompt(insights: list[dict]) -> str:
    summary = json.dumps(
        [{"feature1": i["feature1"], "feature2": i["feature2"], "insight": i["insight"]}
         for i in insights],
        indent=2,
    )
    return f"""You are a senior data scientist reviewing bivariate analysis results for a telecom customer dataset.

Below are {len(insights)} feature-pair insights discovered by analysing every possible column pair.
Your job is to select the TOP 5 MOST IMPORTANT pairs — the ones that would matter most
for understanding customer behaviour, predicting churn, or driving business decisions.

ALL DISCOVERED INSIGHTS:
{summary}

YOUR TASK:
Pick exactly 5 pairs that are the most impactful and actionable.
For each, write a rich, detailed insight (3-5 sentences) explaining:
  - What the relationship is and how the data shows it
  - Why this pair is important for the business
  - What action or investigation it suggests

RULES:
1. Return EXACTLY 5 objects.
2. Use this JSON structure for each:
   {{
     "feature1": "<column name — must match exactly>",
     "feature2": "<column name — must match exactly>",
     "insight": "<3-5 sentence detailed explanation>"
   }}
3. Return a JSON array of these 5 objects.
4. Output ONLY valid JSON — no markdown, no commentary.
"""


def call_llm(prompt: str, model: str = MODEL_DISCOVERY, max_tokens: int = 2048) -> str:
    """Call the LLM with automatic retry on rate-limit (429) errors."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                # Extract wait time from error message
                wait = 60  # default
                match = re.search(r"try again in (\d+)m", err)
                if match:
                    wait = int(match.group(1)) * 60 + 10
                match2 = re.search(r"try again in (\d+\.?\d*)s", err)
                if match2 and not match:
                    wait = int(float(match2.group(1))) + 5
                if wait > 300:
                    wait = 300  # cap at 5 min
                print(f"    Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Rate limit: max retries exceeded")


def parse_llm_json(raw: str) -> list[dict]:
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
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    print("  WARNING: could not parse LLM response as JSON")
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: OPTIMAL BINNING & CROSSTAB
# ═══════════════════════════════════════════════════════════════════════════════

def bucket_numeric(series: pd.Series, other: pd.Series = None) -> pd.Series:
    """Bucket a numeric column using DecisionTree optimal binning (KMeans fallback)."""
    if series.dropna().nunique() <= MAX_BINS:
        return series.astype(str)

    # Supervised tree-based binning
    if other is not None:
        try:
            mask = series.notna() & other.notna()
            X = series[mask].values.reshape(-1, 1)
            y_raw = other[mask]
            if pd.api.types.is_numeric_dtype(y_raw):
                y = pd.qcut(y_raw, q=min(5, y_raw.nunique()), labels=False, duplicates="drop")
            else:
                y = y_raw.astype("category").cat.codes

            tree = DecisionTreeClassifier(
                max_leaf_nodes=MAX_BINS,
                min_samples_leaf=max(50, int(len(X) * 0.02)),
            )
            tree.fit(X, y)
            thresholds = sorted(set(tree.tree_.threshold[tree.tree_.threshold != -2]))
            if len(thresholds) >= 2:
                edges = [float("-inf")] + thresholds + [float("inf")]
                return _label_bins(pd.cut(series, bins=edges, duplicates="drop"))
        except Exception:
            pass

    # Fallback: KMeans binning
    try:
        kbd = KBinsDiscretizer(n_bins=MAX_BINS, encode="ordinal", strategy="kmeans", subsample=None)
        kbd.fit(series.dropna().values.reshape(-1, 1))
        edges = [float("-inf")] + list(kbd.bin_edges_[0][1:-1]) + [float("inf")]
        return _label_bins(pd.cut(series, bins=edges, duplicates="drop"))
    except Exception:
        try:
            return _label_bins(pd.qcut(series, q=MAX_BINS, duplicates="drop"))
        except ValueError:
            return _label_bins(pd.cut(series, bins=MAX_BINS, duplicates="drop"))


def _label_bins(binned: pd.Series) -> pd.Series:
    """Give interval bins human-readable labels like 'min – 100.5'."""
    labels = {}
    for iv in binned.cat.categories:
        lo = f"{iv.left:.1f}" if iv.left != float("-inf") else "min"
        hi = f"{iv.right:.1f}" if iv.right != float("inf") else "max"
        labels[iv] = f"{lo} – {hi}"
    return binned.cat.rename_categories(labels).astype(str)


def filter_categories(series: pd.Series) -> pd.Series:
    """Keep categories >= CATEGORY_THRESHOLD; merge the rest into 'Other'."""
    counts = series.value_counts(normalize=True)
    keep = set(counts[counts >= CATEGORY_THRESHOLD].index)
    if not keep:
        cumsum = counts.cumsum()
        keep = set(cumsum[cumsum <= 0.8].index) or {counts.index[0]}
    return series.apply(lambda x: x if x in keep else "Other")


def compute_crosstab(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Compute a crosstab with optimal binning for numerics and threshold filtering
    for categoricals."""
    type_a = classify_column(df[col_a])
    type_b = classify_column(df[col_b])

    sa = bucket_numeric(df[col_a], df[col_b]) if type_a == "numeric" else filter_categories(df[col_a].astype(str))
    sb = bucket_numeric(df[col_b], df[col_a]) if type_b == "numeric" else filter_categories(df[col_b].astype(str))

    return pd.crosstab(sa, sb).to_dict(orient="index")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(top5: list[dict], df: pd.DataFrame):
    """Compute crosstabs for the top 5 pairs and write all output files."""
    print("\n  Computing crosstabs for top 5 pairs...")
    for item in top5:
        f1, f2 = item["feature1"], item["feature2"]
        if f1 in df.columns and f2 in df.columns:
            item["crosstab"] = compute_crosstab(df, f1, f2)
        else:
            item["crosstab"] = {}
            print(f"  WARNING: columns '{f1}' or '{f2}' not found in dataset")

    # JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(top5, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved -> {OUTPUT_JSON}")

    # CSV
    fieldnames = ["feature1", "feature2", "insight", "crosstab"]
    rows = [{
        "feature1": t["feature1"],
        "feature2": t["feature2"],
        "insight": t["insight"],
        "crosstab": json.dumps(t.get("crosstab", {}), default=str),
    } for t in top5]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV  saved -> {OUTPUT_CSV}")

    # Readable crosstabs
    with open(OUTPUT_CROSSTABS, "w", encoding="utf-8", newline="") as f:
        for item in top5:
            ct = item.get("crosstab", {})
            if not ct:
                continue
            f1, f2 = item["feature1"], item["feature2"]
            f.write(f"\n--- {f1} vs {f2} ---\n")
            f.write(f"Insight: {item['insight']}\n\n")
            col_labels = sorted({c for row in ct.values() for c in row})
            f.write(f"{f1} \\ {f2}," + ",".join(str(c) for c in col_labels) + "\n")
            for row_label in sorted(ct.keys()):
                vals = ",".join(str(ct[row_label].get(c, 0)) for c in col_labels)
                f.write(f"{row_label},{vals}\n")
    print(f"  Crosstabs saved -> {OUTPUT_CROSSTABS}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  BIVARIATE ANALYSIS — Top 5 Hidden Relationships")
    print("=" * 60)

    # ── 1. Load data ──
    print(f"\n  Loading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── 2. Select columns ──
    columns = [c for c in df.columns if c not in SKIP_COLUMNS]
    all_pairs = list(itertools.combinations(columns, 2))
    print(f"  Analysable columns: {len(columns)}  |  Total pairs: {len(all_pairs)}")

    # ── 3. Pre-filter: skip boring pairs using statistical tests ──
    print("\n  Pre-filtering pairs (|r|≥0.05, chi² p<0.05, CV≥0.02)...")
    interesting_pairs = [(a, b) for a, b in all_pairs if is_interesting_pair(df, a, b)]
    print(f"  Kept {len(interesting_pairs)}/{len(all_pairs)} interesting pairs "
          f"(filtered out {len(all_pairs) - len(interesting_pairs)} boring ones)")

    # ── 4. Compute summaries only for interesting pairs ──
    print("  Computing statistics for interesting pairs...")
    summaries = [pair_summary(df, a, b) for a, b in interesting_pairs]

    # ── 5. LLM Pass 1 — discover insights (8B model, cheap) ──
    all_insights = []
    batches = [summaries[i:i + PAIRS_PER_BATCH] for i in range(0, len(summaries), PAIRS_PER_BATCH)]
    print(f"  Sending {len(batches)} batches to LLM ({PAIRS_PER_BATCH} pairs/batch)")
    print(f"  Discovery model: {MODEL_DISCOVERY}\n")

    for idx, batch in enumerate(batches, 1):
        names = [f"({s['column_a']}, {s['column_b']})" for s in batch]
        print(f"  Batch {idx}/{len(batches)}: {names[0]} ... {names[-1]}")
        try:
            raw = call_llm(build_discovery_prompt(batch), model=MODEL_DISCOVERY, max_tokens=2048)
            found = parse_llm_json(raw)
            all_insights.extend(found)
            print(f"    -> {len(found)} insights")
        except Exception as e:
            print(f"    -> ERROR: {e}")
        if idx < len(batches):
            time.sleep(2)

    print(f"\n  Total insights discovered: {len(all_insights)}")

    if len(all_insights) == 0:
        print("  No insights found. Exiting.")
        return

    # ── 6. LLM Pass 2 — rank and select top 5 (70B model, smart) ──
    print(f"  Asking {MODEL_RANKING} to select the top 5 most important pairs...\n")
    cleaned = [{"feature1": i.get("feature1", i.get("column_a", "")),
                "feature2": i.get("feature2", i.get("column_b", "")),
                "insight": i.get("insight", "")} for i in all_insights]
    try:
        raw = call_llm(build_ranking_prompt(cleaned), model=MODEL_RANKING, max_tokens=4096)
        top5 = parse_llm_json(raw)[:5]
    except Exception as e:
        print(f"  Ranking error: {e}  — falling back to first 5")
        top5 = cleaned[:5]

    # Normalize
    top5 = [{"feature1": t.get("feature1", ""),
             "feature2": t.get("feature2", ""),
             "insight": t.get("insight", "")} for t in top5]

    for i, t in enumerate(top5, 1):
        print(f"  {i}. {t['feature1']}  vs  {t['feature2']}")

    # ── 7. Compute crosstabs & save ──
    save_results(top5, df)

    print("\n" + "=" * 60)
    print("  DONE — Top 5 insights written to output files")
    print("=" * 60)


if __name__ == "__main__":
    main()
