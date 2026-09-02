# Telco Customer Churn Prediction & Retention Prioritization

### An Explainable Machine Learning Framework for Identifying, Ranking, and Segmenting Customers at Risk of Churn

## 📌 Project Overview

Customer churn is one of the most important problems in subscription-based businesses. Predicting which customers are likely to leave is useful, but a practical churn system must go further:

> **Who is likely to churn, why are they at risk, and which customers should the business prioritize for retention?**

This project develops an end-to-end machine learning framework using the IBM Telco Customer Churn dataset to answer these questions.

The project combines:

- Exploratory Data Analysis (EDA)
- Data quality and consistency auditing
- Statistical association testing
- Feature engineering
- Logistic Regression
- Random Forest
- XGBoost
- Hyperparameter optimization
- Out-of-fold threshold optimization
- SHAP explainability
- Customer-level risk scoring
- Risk-tier segmentation
- Retention targeting analysis
- Business-oriented recommendations

The final system is designed not merely to classify churn, but to convert churn probabilities into an **actionable customer prioritization framework**.

---

# 🎯 Why This Project?

A standard churn classification project often stops after reporting accuracy or ROC-AUC.

That approach has several limitations.

### 1. Accuracy alone is insufficient

The dataset is imbalanced, with only **26.54%** of customers having churned.

A model can therefore achieve seemingly reasonable accuracy while failing to identify a substantial number of actual churners.

For a retention use case, missing a customer who is likely to churn can be more important than correctly identifying a customer who will stay.

---

### 2. A probability threshold is a business decision

Most binary classifiers use a default probability threshold of `0.50`.

However, the optimal threshold depends on the business objective.

If the goal is retention outreach, the company may prefer to identify more potential churners even if that means contacting some customers who would not have churned.

Therefore, this project explicitly optimizes the classification threshold using **out-of-fold predictions** rather than blindly using `0.50`.

---

### 3. Predictions need to be explainable

A churn model that simply says:

> "Customer A has a 92% probability of churn."

is less useful than one that can also explain:

> "The prediction is strongly influenced by very short tenure, contract type, internet service, and payment method."

SHAP is therefore used to provide both:

- Global model explanations
- Individual customer explanations

---

### 4. Retention teams need prioritization, not just predictions

A business cannot necessarily intervene with every customer.

The model therefore converts predicted probabilities into actionable risk tiers:

- Low Risk
- Medium Risk
- High Risk

This creates a framework for allocating retention resources according to predicted risk.

---

# 📊 Dataset

The project uses the **IBM Telco Customer Churn dataset**.

The dataset contains **7,043 customer records** and **21 original columns** covering customer demographics, tenure, services, contracts, billing, and churn outcome.

### Main variables

| Category | Variables |
|---|---|
| Customer profile | gender, SeniorCitizen, Partner, Dependents |
| Tenure | tenure |
| Phone services | PhoneService, MultipleLines |
| Internet services | InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport |
| Entertainment | StreamingTV, StreamingMovies |
| Contract | Contract |
| Billing | PaperlessBilling, PaymentMethod |
| Financial | MonthlyCharges, TotalCharges |
| Target | Churn |

`customerID` is used only for customer-level identification and is excluded from model training.

---

# 🧹 1. Data Quality Audit

Before modeling, the dataset was systematically audited for:

- Duplicate rows
- Duplicate customer IDs
- Missing values
- Data types
- Invalid categorical values
- Logical inconsistencies
- Numeric consistency

### TotalCharges issue

`TotalCharges` was initially stored as an object/string field.

After conversion to numeric, **11 blank values** were identified.

These records all had:

- `tenure = 0`
- `Churn = No`

Since customers with zero tenure had no accumulated tenure-based charges, these values were treated as `0` rather than removing the observations.

This preserved all **7,043 customers** in the analysis.

### Logical consistency checks

The following service combinations were also checked:

- PhoneService vs MultipleLines
- InternetService vs OnlineSecurity
- InternetService vs OnlineBackup
- InternetService vs DeviceProtection
- InternetService vs TechSupport
- InternetService vs StreamingTV
- InternetService vs StreamingMovies
- tenure vs TotalCharges

No widespread logical inconsistencies were identified.

---

# 📈 2. Exploratory Data Analysis

The analysis first established the overall churn baseline.

### Overall churn

| Churn | Customers | Share |
|---|---:|---:|
| No | 5,174 | 73.46% |
| Yes | 1,869 | **26.54%** |

Therefore, the baseline churn rate is:

**26.54%**

This baseline is used throughout the project when interpreting customer segments and model performance.

---

## Tenure Analysis

Customers were grouped into lifecycle cohorts:

- 0 months
- 1–6 months
- 7–12 months
- 13–24 months
- 25–48 months
- 49–72 months

The observed churn rates were:

| Tenure Cohort | Churn Rate |
|---|---:|
| 0 months | 0.00% |
| 1–6 months | **53.33%** |
| 7–12 months | 35.89% |
| 13–24 months | 28.71% |
| 25–48 months | 20.39% |
| 49–72 months | **9.51%** |

The strongest pattern is the elevated churn rate among early-tenure customers.

The 1–6 month cohort had a **53.33% observed churn rate**, approximately twice the overall baseline.

---

# 📑 3. Contract Analysis

Contract type showed one of the strongest relationships with churn.

| Contract | Customers | Churn Rate |
|---|---:|---:|
| Month-to-month | 3,875 | **42.71%** |
| One year | 1,473 | 11.27% |
| Two year | 1,695 | **2.83%** |

Month-to-month customers therefore represent an important retention-prioritization group.

Contract type was also examined jointly with tenure to understand whether the relationship persisted across customer lifecycle stages.

---

# 💳 4. Payment Method Analysis

Observed churn rates by payment method:

| Payment Method | Churn Rate |
|---|---:|
| Electronic check | **45.29%** |
| Mailed check | 19.11% |
| Bank transfer (automatic) | 16.71% |
| Credit card (automatic) | **15.24%** |

The combination of:

**Electronic check + Month-to-month contract**

had an observed churn rate of:

**53.73%**

These relationships are observational and are not interpreted as causal effects.

---

# 🌐 5. Internet Service Analysis

Observed churn rates:

| Internet Service | Churn Rate |
|---|---:|
| Fiber optic | **41.89%** |
| DSL | 18.96% |
| No internet | **7.40%** |

The combination of:

**Fiber optic + Month-to-month**

showed an observed churn rate of:

**54.61%**

---

# 💰 6. Monthly Charges Analysis

Churned customers had higher average monthly charges:

| Customer Status | Mean Monthly Charges |
|---|---:|
| Did not churn | 61.27 |
| Churned | **74.44** |

Customers were also divided into four charge bands.

The relationship was not strictly monotonic:

| Charge Band | Churn Rate |
|---|---:|
| Low | 11.24% |
| Medium-Low | 24.58% |
| Medium-High | **37.51%** |
| High | 32.88% |

Therefore, the analysis does **not** assume that increasing monthly charges always causes increasing churn.

---

# 🧪 7. Statistical Analysis

Exploratory patterns were followed by statistical testing.

For categorical variables, **Chi-square tests of independence** were performed.

Because statistical significance alone does not indicate the strength of an association, **Cramér's V** was also calculated.

### Strongest categorical associations

| Feature | Cramér's V |
|---|---:|
| Contract | **0.41** |
| OnlineSecurity | **0.35** |
| TechSupport | **0.34** |
| InternetService | **0.32** |
| PaymentMethod | **0.30** |
| OnlineBackup | 0.29 |
| DeviceProtection | 0.28 |

Contract had the strongest association with churn among the categorical variables analyzed.

The Contract × Churn chi-square statistic was:

**χ² = 1184.60, df = 2**

with an extremely small p-value.

These tests establish statistical association, not causality.

---

# 🔍 8. Multidimensional Churn Segmentation

Individual variables do not fully describe customer risk.

The project therefore examined combinations of:

- Contract
- InternetService
- OnlineSecurity
- TechSupport

The highest-risk segment with at least 50 customers was:

> **Month-to-month + Fiber optic + No Online Security + No Tech Support**

This segment contained:

- **1,524 customers**
- **925 churners**
- **60.70% observed churn rate**

---

## Top 3 High-Risk Segments

The three largest high-risk segments identified through this analysis represented:

- **34.29% of customers**
- **69.07% of observed churn**

This indicates that churn was substantially concentrated within a relatively small set of customer profiles.

---

# 🤖 9. Machine Learning

Three classification approaches were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

The dataset was divided using a stratified:

- 80% training set
- 20% test set

Resulting in:

- Training: **5,634 customers**
- Test: **1,409 customers**

The churn proportion was preserved across the split.

---

# ⚙️ 10. Feature Preprocessing

Numerical features:

- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

were standardized using `StandardScaler`.

Categorical features were transformed using:

`OneHotEncoder(handle_unknown="ignore", drop="first")`

All preprocessing was placed inside a Scikit-learn `Pipeline` / `ColumnTransformer`.

This ensures preprocessing is fitted only on the training data and applied consistently to unseen data.

---

# 🧠 11. Model Comparison

### Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 80.70% |
| Precision | 66.04% |
| Recall | 56.15% |
| F1 | 60.69% |
| ROC-AUC | 0.8422 |

### Random Forest

| Metric | Score |
|---|---:|
| Accuracy | ~78% |
| Precision | ~57% |
| Recall | ~65% |
| F1 | ~61% |
| ROC-AUC | ~0.84 |

### XGBoost

The initial XGBoost model provided higher recall and F1 than the Logistic Regression baseline at the default classification threshold.

Because churn is an imbalanced classification problem, **Average Precision** was used as the hyperparameter optimization objective.

---

# 🚀 12. XGBoost Hyperparameter Optimization

Randomized cross-validated hyperparameter search was performed using:

- 5-fold Stratified Cross-Validation
- 20 randomized parameter combinations
- Average Precision as the scoring metric

The best cross-validation Average Precision was:

**0.6679**

The selected model used:

- `n_estimators = 400`
- `max_depth = 2`
- `learning_rate = 0.05`
- `subsample = 1.0`
- `colsample_bytree = 0.8`
- `min_child_weight = 1`
- `gamma = 0.3`

Class imbalance was addressed using `scale_pos_weight`.

---

# 🎯 13. Out-of-Fold Threshold Optimization

A default threshold of `0.50` is not necessarily appropriate for a retention use case.

Instead, out-of-fold predictions were generated on the training set and used to select the probability threshold that maximized F1.

The selected threshold was:

**0.5416**

This threshold was then locked before evaluating the held-out test set.

This prevents using the test set to tune the final decision threshold.

---

# 📊 14. Final Model Performance

At the locked threshold of **0.5416**, the tuned XGBoost model achieved:

| Metric | Test Performance |
|---|---:|
| ROC-AUC | **0.8442** |
| Precision | **52.01%** |
| Recall | **76.20%** |
| F1 | **61.82%** |

The model identified:

**285 of 374 observed churners**

on the held-out test set.

This corresponds to:

**76.20% churn capture**

---

# 🎯 15. Retention Prioritization

The model was converted from a classification model into a customer prioritization system.

At the selected threshold:

- Test customers: **1,409**
- Customers flagged: **548**
- Customer targeting rate: **38.89%**
- Churners captured: **285**
- Churn capture rate: **76.20%**
- Precision: **52.01%**

Therefore:

> The framework targets approximately 39% of customers while capturing approximately 76% of observed churners.

---

# 📈 16. Retention Targeting Lift

To evaluate whether the targeted list concentrates churn effectively, targeting lift was calculated as:

**Churn capture rate / customer targeting rate**

The resulting lift was:

## **1.96×**

This means the model concentrates observed churn substantially more effectively than randomly targeting the same share of customers.

---

# 🏷️ 17. Customer Risk Tiers

Predicted churn probabilities were converted into three operational risk tiers.

| Risk Tier | Customer Share | Actual Churn Rate |
|---|---:|---:|
| Low Risk | 51.53% | **7.02%** |
| Medium Risk | 24.70% | **31.90%** |
| High Risk | 23.78% | **63.28%** |

The High Risk segment contains:

**23.78% of customers**

but has an observed churn rate of:

**63.28%**

High + Medium Risk customers represent:

**48.48% of customers**

while containing:

**86.36% of observed churners.**

This enables differentiated retention strategies rather than treating every customer identically.

---

# 🔬 18. SHAP Explainability

SHAP was used to understand how the XGBoost model generates its predictions.

The strongest features by mean absolute SHAP value were:

| Feature | Mean Absolute SHAP |
|---|---:|
| tenure | **0.524** |
| Contract – Two year | **0.469** |
| InternetService – Fiber optic | **0.379** |
| Contract – One year | **0.214** |
| TotalCharges | **0.193** |
| PaymentMethod – Electronic check | **0.187** |
| InternetService – No | **0.183** |
| MonthlyCharges | **0.160** |

This provides a model-level explanation of which variables contribute most strongly to prediction variation.

---

# 👤 19. Customer-Level Explainability

SHAP was also used to explain individual customer predictions.

The highest-risk test customer received:

**96.60% predicted churn probability**

and subsequently had:

**Actual churn = Yes**

The customer's profile included:

- Tenure = 1 month
- Month-to-month contract
- Fiber optic internet
- No Online Security
- No Tech Support
- Electronic check payment
- Monthly charges = 95.10

The local SHAP explanation showed that very low tenure was the strongest positive contributor to the model's churn prediction, followed by factors including fiber optic service and payment method.

The SHAP reconstruction also matched the model probability exactly:

**SHAP reconstructed probability = 96.60%**

**XGBoost predicted probability = 96.60%**

This validates the consistency of the local explanation.

---

# 💼 20. Business Recommendations

### 1. Strengthen early-lifecycle retention

Customers in the 1–6 month cohort have a **53.33% observed churn rate**.

Potential actions include:

- stronger onboarding
- early engagement programs
- proactive support
- first-month experience monitoring

---

### 2. Prioritize month-to-month customers

Month-to-month customers show a **42.71% observed churn rate** compared with:

- 11.27% for one-year contracts
- 2.83% for two-year contracts

Where commercially appropriate, retention teams could prioritize contract-renewal and longer-term-plan conversion opportunities.

---

### 3. Prioritize concentrated high-risk segments

The top three high-risk segments represent:

**34.29% of customers**

but account for:

**69.07% of observed churn**

This supports targeted rather than customer-wide retention campaigns.

---

### 4. Investigate fiber + month-to-month customers

The fiber optic + month-to-month combination has a:

**54.61% observed churn rate**

The highest-risk multidimensional segment identified in the analysis has a:

**60.70% observed churn rate**

These customers should be investigated for product experience, pricing, support, and service-quality issues before designing interventions.

---

### 5. Use risk-based intervention intensity

A potential operational framework:

| Risk Tier | Suggested Treatment |
|---|---|
| High Risk | Intensive retention intervention |
| Medium Risk | Proactive engagement / targeted offers |
| Low Risk | Standard engagement |

The exact intervention strategy should ultimately be validated through controlled experiments.

---

# ⚠️ 21. Limitations

### Observational data

The relationships identified in the dataset are associations and should not be interpreted as causal effects.

For example, a higher churn rate among fiber customers does not prove that fiber service causes churn.

---

### Historical dataset

Model performance can change as:

- customer behavior changes
- pricing changes
- products change
- market conditions change

---

### No intervention outcomes

The dataset contains churn outcomes but does not tell us whether a retention intervention would have prevented churn.

Therefore, the model estimates **risk**, not treatment effectiveness.

---

### Retention economics not available

The dataset does not provide:

- customer lifetime value
- retention campaign cost
- offer cost
- probability of accepting an offer
- incremental retention probability

Therefore, this project does not claim a dollar ROI.

A production system should incorporate these variables when deciding which customers are economically worth targeting.

---

### External validation

Before production use, the model should be evaluated on:

- temporally separated data
- newly collected customer data
- or an independent external dataset

---

# 🔮 22. Potential Production Extension

A production retention system could extend the current framework from:

> **Who is likely to churn?**

to:

> **Who should we intervene with to maximize expected retention ROI?**

A future decision framework could combine:

```text
Churn Probability
        ↓
Customer Lifetime Value
        ↓
Retention Offer Cost
        ↓
Expected Intervention Effectiveness
        ↓
Expected Retention ROI
        ↓
Prioritized Retention Queue