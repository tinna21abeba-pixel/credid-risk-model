# Credit Risk Probability Model for Alternative Data

This repository contains an end-to-end implementation for building, deploying, and automating a credit risk model using transaction data from an eCommerce partner.

## Project Structure

Standard project layout (partial):

- `data/` - raw and processed data (ignored in repo)
- `notebooks/` - exploratory notebooks (EDA)
- `src/` - production code (data processing, training, API)
- `tests/` - unit tests

## Credit Scoring Business Understanding

**How does the Basel II Accord's emphasis on risk measurement influence the need for an interpretable and well-documented model?**

Basel II requires banks to demonstrate sound risk measurement and governance. This drives the need for models that are interpretable, well-documented, and auditable so that business users and regulators can understand drivers of default, validate assumptions, and reproduce decisions. Documentation of data lineage, feature engineering, model assumptions, and monitoring plans is essential for compliance and ongoing model risk management.

**Why is a proxy variable necessary, and what business risks does proxy-based prediction introduce?**

When direct default labels are unavailable, a proxy (e.g., disengaged customers identified via RFM clustering) enables supervised learning. Proxy-based predictions introduce label risk: the proxy may not perfectly align with true default behavior, leading to model error and potential adverse selection (mispricing of risk). Business risks include biased decisions, regulatory pushback if the proxy is poorly justified, and deterioration in performance when real-world behavior diverges from proxy assumptions.

**Key trade-offs between an interpretable model (Logistic Regression + WoE) and a high-performance model (Gradient Boosting):**

- Interpretability: Logistic Regression with WoE is easier to explain, debug, and provision for regulatory review; Gradient Boosting is less transparent.
- Performance: Gradient Boosting often yields higher predictive performance on complex signals, especially non-linear interactions.
- Operational risk: Simpler models are easier to monitor and maintain; complex models require stronger versioning, monitoring, and explainability tooling.
- Use case fit: For decisions with high regulatory scrutiny, prefer interpretable models or augment complex models with explainability and strong documentation.

## Next steps

- Task 2: Run EDA in `notebooks/eda.ipynb` and record top insights.
