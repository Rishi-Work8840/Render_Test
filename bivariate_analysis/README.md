# Bivariate Analysis – AI-Powered Hidden Relationships

## What This Does
This script takes the `synthetic_data_30012026_v4.csv` dataset, compares **every possible pair** of columns (bivariate analysis), and uses an AI model (Groq / Llama 3.3 70B) to find **hidden relationships** between them.

## How It Works
1. **Loads** the CSV and identifies all numeric / categorical columns  
2. **Generates** every possible column pair (C(n, 2) combinations)  
3. **Computes** statistical summaries per pair:
   - Numeric × Numeric → Pearson correlation, means, std devs  
   - Categorical × Categorical → Crosstab shape, unique counts  
   - Mixed → Group means by category  
4. **Batches** these summaries and sends them to the LLM  
5. **Outputs** insights as both `bivariate_insights.json` and `bivariate_insights.csv`

## Output Format
Each insight contains:
| Field | Description |
|---|---|
| `column_a` | First column name |
| `column_b` | Second column name |
| `relationship_type` | linear / non-linear / categorical_association / mixed / none |
| `strength` | strong / moderate / weak |
| `direction` | positive / negative / N/A |
| `insight` | 1-2 sentence explanation |
| `business_implication` | Actionable takeaway |

## Run
```bash
pip install pandas numpy openai python-dotenv
python bivariate_analysis.py
```

Make sure `GROQ_API_KEY` is set in the `.env` file in the parent directory.
