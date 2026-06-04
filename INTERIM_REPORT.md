# Interim Submission — Credit Risk Probability Model

Date: 2026-05-31

Summary
- Task 1 (Business Understanding): Completed — `README.md` contains the **Credit Scoring Business Understanding** section addressing Basel II, proxy variable risks, and model trade-offs.
- Task 2 (EDA): Notebook scaffold created at `notebooks/eda.ipynb` and executed against the uploaded CSV `data/raw/data .csv`. EDA outputs were produced and the Top 3–5 insights were populated.

What I did
- Initialized project skeleton and essential files: `README.md`, `.gitignore`, `requirements.txt`, `src/__init__.py`, `src/data_processing.py` (stub).
- Created a structured EDA notebook skeleton that performs overview, missingness checks, distributions, correlations, outlier detection, and computes RFM aggregates when the dataset is available.
 - Executed EDA on the uploaded dataset and generated `data/processed/eda_summary.json` and `notebooks/eda_insights.md` containing computed statistics and concise insights.

How you can reproduce/run the EDA locally
1. Ensure the dataset `data/raw/data.csv` is present and contains the Xente transactions CSV. If your file is somewhere else, update the path in `notebooks/eda.ipynb`.

2. Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Start Jupyter and run `notebooks/eda.ipynb` or run the notebook programmatically.

Next steps to finish interim submission
- Review the generated EDA insights in `notebooks/eda.ipynb` and `notebooks/eda_insights.md`.
- Merge `task-1` and `task-2` branches into `main` and push to GitHub. I can create the PR and include these artifacts.

Notes
- The uploaded dataset filename is `data/raw/data .csv` (contains a space). The EDA was run on that file; consider renaming to `data.csv` to match notebook defaults.

If you'd like, I can: upload a small synthetic sample for development, or fetch the dataset from the provided Kaggle link (requires credentials) — tell me which you prefer.
