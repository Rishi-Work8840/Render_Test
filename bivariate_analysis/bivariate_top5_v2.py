"""
Bivariate Analysis — Top 5 Insights (v2)
=========================================
Optimised pipeline:
  1. Sends ONLY column names + types to LLM → picks 5 best pairs (1 API call)
  2. Python computes detailed stats for those 5 pairs locally
  3. Sends compact stats to LLM → writes detailed insights (1 API call)
  4. Computes crosstabs with dynamic binning & dynamic category thresholds
  5. Saves JSON, CSV, crosstabs

Total API calls: 2  |  Token usage: ~5-10K
"""

import os, sys, json, csv, re, time
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ───────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit("Error: Set GROQ_API_KEY in your .env file first.")

api_key_2 = os.environ.get("GROQ_API_KEY_2")

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, timeout=120.0)
client_2 = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key_2, timeout=120.0) if api_key_2 else None
_active_client = client  # tracks which client is currently in use

MODEL = "llama-3.3-70b-versatile"
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "synthetic_data_30012026_v4.csv")
OUT_DIR = os.path.dirname(__file__)
OUTPUT_JSON = os.path.join(OUT_DIR, "bivariate_insights.json")
OUTPUT_CSV = os.path.join(OUT_DIR, "bivariate_insights.csv")
OUTPUT_CROSSTABS = os.path.join(OUT_DIR, "bivariate_crosstabs.csv")
SKIP_COLUMNS = {"service_id", "EDD"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_column(series: pd.Series) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


def call_llm(prompt: str, max_tokens: int = 2048) -> str:
    global _active_client
    for attempt in range(4):
        try:
            resp = _active_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                # Try switching to the other API key first
                if _active_client is client and client_2:
                    print("    Rate limited on key 1 — switching to key 2")
                    _active_client = client_2
                    continue
                elif _active_client is client_2:
                    print("    Rate limited on key 2 — switching back to key 1")
                    _active_client = client
                # If both keys exhausted (or no key 2), wait and retry
                wait = 60
                m = re.search(r"try again in (\d+)m", err)
                if m:
                    wait = int(m.group(1)) * 60 + 10
                m2 = re.search(r"try again in (\d+\.?\d*)s", err)
                if m2 and not m:
                    wait = int(float(m2.group(1))) + 5
                wait = min(wait, 300)
                print(f"    Both keys limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Rate limit: max retries exceeded on both keys")


def parse_json(raw: str) -> list[dict]:
    text = raw.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        s, e = text.find("["), text.rfind("]")
        if s != -1 and e != -1:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                pass
    print("  WARNING: could not parse LLM JSON")
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LLM picks top 5 pairs from column names only (1 API call)
# ═══════════════════════════════════════════════════════════════════════════════

def build_column_list(df: pd.DataFrame) -> str:
    """One-line-per-column summary: name, type, example values."""
    lines = []
    for col in df.columns:
        if col in SKIP_COLUMNS:
            continue
        ctype = classify_column(df[col])
        if ctype == "numeric":
            lines.append(f"{col} (numeric, range {df[col].min():.1f}–{df[col].max():.1f})")
        else:
            cats = df[col].dropna().unique()[:5].tolist()
            lines.append(f"{col} (categorical, {df[col].nunique()} unique, e.g. {cats})")
    return "\n".join(lines)


def ask_llm_for_pairs(col_summary: str) -> list[dict]:
    prompt = f"""Telecom dataset columns:
{col_summary}

Pick the 5 column pairs most likely to reveal hidden business-relevant relationships.
Return JSON: [{{"feature1":"<col>","feature2":"<col>"}},...] — exactly 5 objects, no extras."""

    raw = call_llm(prompt, max_tokens=512)
    pairs = parse_json(raw)
    return pairs[:5]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Compute detailed stats for the 5 pairs (local, no API)
# ═══════════════════════════════════════════════════════════════════════════════

def pair_stats(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    ta, tb = classify_column(df[col_a]), classify_column(df[col_b])
    info = {"feature1": col_a, "feature2": col_b, "types": f"{ta}+{tb}"}

    if ta == "numeric" and tb == "numeric":
        r = df[[col_a, col_b]].corr().iloc[0, 1]
        info["corr"] = round(r, 4) if not np.isnan(r) else None
    elif ta == "categorical" and tb == "categorical":
        ct = pd.crosstab(df[col_a], df[col_b])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            _, p, _, _ = chi2_contingency(ct)
            info["chi2_p"] = round(p, 6)
        info["shape"] = f"{ct.shape[0]}x{ct.shape[1]}"
    else:
        num_col = col_a if ta == "numeric" else col_b
        cat_col = col_b if ta == "numeric" else col_a
        gm = df.groupby(cat_col)[num_col].mean()
        info["group_means"] = {str(k): round(v, 2) for k, v in gm.items()}

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: LLM writes detailed insights for those 5 pairs (1 API call)
# ═══════════════════════════════════════════════════════════════════════════════

def ask_llm_for_insights(pairs: list[dict]) -> list[dict]:
    pairs_list = "\n".join(f"{i+1}. {p['feature1']} vs {p['feature2']}" for i, p in enumerate(pairs))
    prompt = f"""Telecom dataset — these 5 column pairs were selected for bivariate analysis:
{pairs_list}

For each pair, write a concise insight (max 50 words) covering: what relationship exists, why it matters, and what action it suggests.
Return JSON: [{{"feature1":"<col>","feature2":"<col>","insight":"<text>"}},...] — exactly 5."""

    raw = call_llm(prompt, max_tokens=2048)
    return parse_json(raw)[:5]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Dynamic binning & dynamic category thresholds
# ═══════════════════════════════════════════════════════════════════════════════

def choose_bin_count(series: pd.Series, rank: int = 1) -> int:
    """Dynamically decide how many bins to use based on the data distribution
    and the pair's importance rank (1 = most important → more bins).
    Uses Sturges' rule capped between 3 and a rank-based upper limit."""
    n = series.dropna().nunique()
    if n <= 3:
        return n
    # Rank 1 → max 6, Rank 2 → 5, Rank 3 → 4, Rank 4-5 → 3
    max_cap = max(3, 7 - rank)
    sturges = int(np.ceil(np.log2(len(series.dropna())) + 1))
    return max(3, min(sturges, max_cap))


def bucket_numeric(series: pd.Series, other: pd.Series = None, rank: int = 1) -> tuple[pd.Series, dict]:
    n_bins = choose_bin_count(series, rank)
    if series.dropna().nunique() <= n_bins:
        return series.astype(str), {}

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
                max_leaf_nodes=n_bins,
                min_samples_leaf=max(50, int(len(X) * 0.02)),
            )
            tree.fit(X, y)
            thresholds = sorted(set(tree.tree_.threshold[tree.tree_.threshold != -2]))
            if len(thresholds) >= 2:
                edges = [float("-inf")] + thresholds + [float("inf")]
                return _label_bins(pd.cut(series, bins=edges, duplicates="drop"), series)
        except Exception:
            pass

    # Fallback: KMeans
    try:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans", subsample=None)
        kbd.fit(series.dropna().values.reshape(-1, 1))
        edges = [float("-inf")] + list(kbd.bin_edges_[0][1:-1]) + [float("inf")]
        return _label_bins(pd.cut(series, bins=edges, duplicates="drop"), series)
    except Exception:
        try:
            return _label_bins(pd.qcut(series, q=n_bins, duplicates="drop"), series)
        except ValueError:
            return _label_bins(pd.cut(series, bins=n_bins, duplicates="drop"), series)


def _label_bins(binned: pd.Series, original: pd.Series = None) -> tuple[pd.Series, dict]:
    labels = {}
    legend = {}
    cats = list(binned.cat.categories)
    real_min = f"{original.min():.1f}" if original is not None else "?"
    real_max = f"{original.max():.1f}" if original is not None else "?"
    last_idx = len(cats)
    for idx, iv in enumerate(cats, 1):
        lo = real_min if iv.left == float("-inf") else f"{iv.left:.1f}"
        hi = real_max if iv.right == float("inf") else f"{iv.right:.1f}"
        key = lo
        labels[iv] = key
        legend[key] = f"{lo} - {hi}"
    return binned.cat.rename_categories(labels).astype(str), legend


def filter_categories(series: pd.Series) -> pd.Series:
    """Dynamic category filtering:
    - If 4 or fewer categories → show all, no 'Other'.
    - If >4 categories → keep the top ones that cumulatively cover 80%,
      merge the rest into 'Other'. Threshold is determined automatically."""
    n_cats = series.nunique()
    if n_cats <= 4:
        return series  # show all categories as-is

    # Dynamic threshold: keep categories covering 80% of the data
    counts = series.value_counts(normalize=True).sort_values(ascending=False)
    cumsum = counts.cumsum()
    # Keep all categories until we reach 80% coverage
    keep = set(cumsum[cumsum <= 0.80].index)
    # Always include the category that pushes us past 80% (so we actually reach 80%)
    remaining = set(counts.index) - keep
    if remaining:
        keep.add(counts[counts.index.isin(remaining)].index[0])
    # Safety: always keep at least 2 categories
    if len(keep) < 2:
        keep = set(counts.index[:2])

    return series.apply(lambda x: x if x in keep else "Other")


def _sort_bucket_labels(labels):
    """Sort labels numerically by leading number; range labels (with '-') sort last among ties."""
    def key(lbl):
        m = re.match(r"^-?[\d.]+", lbl)
        if m:
            # Range labels (e.g. "2014.1-6503.0") sort after plain "2014.1"
            is_range = 1 if "-" in lbl[len(m.group()):] else 0
            return (0, float(m.group()), is_range)
        return (1, 0, lbl)
    return sorted(labels, key=key)


def compute_crosstab(df: pd.DataFrame, col_a: str, col_b: str, rank: int = 1) -> tuple[dict, dict, dict]:
    """Returns (crosstab_dict, legend_a, legend_b)."""
    ta, tb = classify_column(df[col_a]), classify_column(df[col_b])
    if ta == "numeric":
        sa, leg_a = bucket_numeric(df[col_a], df[col_b], rank)
    else:
        sa = filter_categories(df[col_a].astype(str))
        leg_a = {}
    if tb == "numeric":
        sb, leg_b = bucket_numeric(df[col_b], df[col_a], rank)
    else:
        sb = filter_categories(df[col_b].astype(str))
        leg_b = {}
    return pd.crosstab(sa, sb).to_dict(orient="index"), leg_a, leg_b


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Save results
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(top5: list[dict], df: pd.DataFrame):
    # ── File 1: Insights only (no crosstab) ──
    insights_only = [{"feature1": t["feature1"], "feature2": t["feature2"],
                      "insight": t["insight"]} for t in top5]

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(insights_only, f, indent=2, ensure_ascii=False)
    print(f"  Insights JSON -> {OUTPUT_JSON}")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature1", "feature2", "insight"])
        w.writeheader()
        w.writerows(insights_only)
    print(f"  Insights CSV  -> {OUTPUT_CSV}")

    # ── File 2: Crosstabs only ──
    print("\n  Computing crosstabs...")
    crosstabs_data = []
    for rank, item in enumerate(top5, 1):
        f1, f2 = item["feature1"], item["feature2"]
        if f1 in df.columns and f2 in df.columns:
            ct, leg_a, leg_b = compute_crosstab(df, f1, f2, rank)
            crosstabs_data.append({"feature1": f1, "feature2": f2,
                                   "crosstab": ct, "legend_a": leg_a, "legend_b": leg_b})
            print(f"    Pair {rank}: max {max(3, 7 - rank)} bins")
        else:
            print(f"  WARNING: '{f1}' or '{f2}' not in dataset")

    with open(OUTPUT_CROSSTABS, "w", encoding="utf-8", newline="") as f:
        for entry in crosstabs_data:
            ct = entry["crosstab"]
            if not ct:
                continue
            f1, f2 = entry["feature1"], entry["feature2"]
            f.write(f"\n--- {f1} vs {f2} ---\n\n")
            col_labels = _sort_bucket_labels({c for row in ct.values() for c in row})
            f.write(f"{f1} \\ {f2}," + ",".join(str(c) for c in col_labels) + "\n")
            for row_label in _sort_bucket_labels(ct.keys()):
                vals = ",".join(str(ct[row_label].get(c, 0)) for c in col_labels)
                f.write(f"{row_label},{vals}\n")
    print(f"  Crosstabs -> {OUTPUT_CROSSTABS}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  BIVARIATE ANALYSIS — Top 5 (v2)")
    print("=" * 60)

    # 1. Load data
    print(f"\n  Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # 2. Build column summary & ask LLM to pick 5 pairs (API call #1)
    col_summary = build_column_list(df)
    print(f"\n  Columns sent to LLM ({len(col_summary.splitlines())} cols, ~{len(col_summary)} chars)")
    print("  Asking LLM to pick 5 most promising pairs...\n")
    pairs = ask_llm_for_pairs(col_summary)

    if len(pairs) < 5:
        print(f"  WARNING: LLM returned only {len(pairs)} pairs")
    for i, p in enumerate(pairs, 1):
        print(f"  {i}. {p.get('feature1', '?')}  vs  {p.get('feature2', '?')}")

    # 3. Validate pair columns exist
    print("\n  Validating selected pairs...")
    valid_pairs = []
    for p in pairs:
        f1, f2 = p.get("feature1", ""), p.get("feature2", "")
        if f1 in df.columns and f2 in df.columns:
            valid_pairs.append(p)
        else:
            print(f"  WARNING: '{f1}' or '{f2}' not found — skipping")

    if not valid_pairs:
        print("  No valid pairs. Exiting.")
        return

    # 4. Ask LLM to write detailed insights (API call #2) — no stats sent
    print("  Asking LLM for detailed insights...\n")
    top5 = ask_llm_for_insights(valid_pairs)

    for i, t in enumerate(top5, 1):
        print(f"  {i}. {t.get('feature1', '?')}  vs  {t.get('feature2', '?')}")

    # 5. Crosstabs & save
    save_results(top5, df)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
