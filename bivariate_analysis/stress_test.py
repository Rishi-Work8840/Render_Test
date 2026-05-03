"""
Stress Test — Run bivariate analysis 10 times and collect results into Excel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from bivariate_top5_v2 import (
    build_column_list, ask_llm_for_pairs, ask_llm_for_insights, CSV_PATH
)

RUNS = 10
OUTPUT_XLSX = os.path.join(os.path.dirname(__file__), "stress_test_results.xlsx")


def main():
    print("=" * 60)
    print(f"  STRESS TEST — {RUNS} runs")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    col_summary = build_column_list(df)
    print(f"  Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols\n")

    rows = []

    for run in range(1, RUNS + 1):
        print(f"  ── Run {run}/{RUNS} ──")

        # API call 1: pick pairs
        pairs = ask_llm_for_pairs(col_summary)

        # Validate
        valid = [p for p in pairs
                 if p.get("feature1", "") in df.columns and p.get("feature2", "") in df.columns]

        if not valid:
            print("    No valid pairs returned — skipping run")
            continue

        # API call 2: get insights
        insights = ask_llm_for_insights(valid)

        for item in insights:
            rows.append({
                "Run Number": run,
                "Col 1": item.get("feature1", ""),
                "Col 2": item.get("feature2", ""),
                "Insight": item.get("insight", ""),
            })

        for i, item in enumerate(insights, 1):
            print(f"    {i}. {item.get('feature1', '?')} vs {item.get('feature2', '?')}")
        print()

    # Save to Excel
    result_df = pd.DataFrame(rows)
    result_df.to_excel(OUTPUT_XLSX, index=False, sheet_name="Stress Test")

    # Auto-fit column widths
    try:
        from openpyxl import load_workbook
        wb = load_workbook(OUTPUT_XLSX)
        ws = wb.active
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 80)
        wb.save(OUTPUT_XLSX)
    except Exception:
        pass

    print(f"  Saved {len(rows)} rows to {OUTPUT_XLSX}")
    print("=" * 60)


if __name__ == "__main__":
    main()
