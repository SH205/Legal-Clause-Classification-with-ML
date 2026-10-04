# Legal Clause Classification with Machine Learning

## Quick Overview

**Goal of the Project:** A machine learning project that classifies legal contract clauses into 100 categories using NLP and a Random Forest classifier.


**Live Demo:** [Streamlit App](https://legal-clause-classification-with-ml-tdnchyebdfk4y9jcx9xmct.streamlit.app)

**Sample Model:** [Sample Model Notebook](https://github.com/SH205/Legal-Clause-Classification-with-ML/blob/main/notebooks/Sample_model.ipynb)

The model was reduced from **300 trees to 8 trees** to significantly reduce the model size and enable deployment through Streamlit.

### Model Size vs. Performance

| Version                       |   Accuracy |   Macro F1 | Weighted F1 |   Model Size |
| ----------------------------- | ---------: | ---------: | ----------: | -----------: |
| 300 Trees                     |     86.15% |     78.74% |      85.37% |         7 GB |
| 100 Trees                     |     85.60% |     79.66% |      84.75% |       308 MB |
| **8 Trees — Streamlit Model** | **80.31%** | **72.80%** |  **79.38%** | **25.15 MB** |

---
### Brief Description

The project compares several machine-learning approaches for legal text classification, including TF-IDF models, a neural network, BERT, RoBERTa, and T5.

After comparing the models, **TF-IDF + Random Forest** produced the strongest validation performance and was used to create an interactive Streamlit application.

### Skills Used

Natural Language Processing • Text Classification • Feature Engineering • TF-IDF • Imbalanced Classification • Machine Learning • Random Forest • Model Evaluation • Error Analysis • Classification Metrics • Model Explainability • Model Optimization • Model Deployment • Streamlit


### Technologies
Python • pandas • NumPy • scikit-learn • XGBoost • PyTorch • Hugging Face Transformers • TF-IDF • Streamlit • Matplotlib • Seaborn

### Results

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- |
| **TF-IDF + Random Forest** | **86.15%** | **78.74%** | **85.37%** |
| TF-IDF + Logistic Regression | 83.28% | 77.10% | 83.79% |
| TF-IDF + XGBoost | 81.20% | 72.79% | 80.39% |
| PyTorch Neural Network | 79.03% | 71.25% | 79.52% |
| T5 | 76.10% | 62.43% | 74.46% |
| RoBERTa | 70.45% | 46.07% | 63.68% |
| BERT | 64.25% | 36.09% | 54.83% |

The dataset contains **60,000 training examples, 10,000 validation examples, and 10,000 test examples across 100 legal clause categories**.

Because the classes are highly imbalanced, Macro F1 is particularly important. The largest class contains 3,167 examples while the smallest contains 23.

---

## Project Workflow

```
LEDGAR Legal Clauses
        ↓
Data Exploration
        ↓
TF-IDF Feature Extraction
        ↓
Multiple ML Approaches
        ↓
Model Evaluation
        ↓
Error Analysis
        ↓
Random Forest Explainability
        ↓
Saved Production Model
        ↓
Streamlit Application
```

The main TF-IDF representation uses unigrams and bigrams and produces **210,328 features**.

---

## Model Evaluation

Random Forest achieved:

- **86.15% Accuracy**
- **78.74% Macro F1**
- **85.37% Weighted F1**

The model was evaluated on a separate validation set of 10,000 examples.

The error analysis showed that many mistakes occur between semantically similar legal categories, such as:

- Applicable Laws → Governing Laws
- Defined Terms → Definitions
- Integration → Entire Agreements
- Tax Withholdings → Withholdings
- Indemnity → Indemnifications
- Modifications → Amendments

This shows that the remaining errors are concentrated in legally related categories rather than being completely random.

---

## Explainability

The project examines which TF-IDF features contribute to individual Random Forest predictions.

For example, predictions involving **Governing Laws** were associated with features such as:

- `laws`
- `state`
- `agreement`
- `construction`
- `enforcement`
- `validity`

The application exposes the prediction, confidence, alternative classifications, and important contributing features.

---

## Interactive Application

The final model is packaged into a Streamlit application.

Users can:

1. Enter a legal clause.
2. Receive a predicted clause category.
3. View prediction confidence.
4. See the top three predicted categories.
5. Inspect the features contributing to the prediction.

The model and TF-IDF vectorizer are saved so the application can make predictions without retraining.

---

## Testing on New Clauses

| Text that were tested | **Expected category** | Prediction | Confidence | **Top 3 Predictions** |
| --- | --- | --- | --- | --- |
| This contract shall be interpreted under the laws of the Commonwealth of Virginia, without regard to its conflict-of-law principles. | Governing Laws | Governing Laws | 71.5% |  Governing Laws : 71.5%, Applicable Laws: 6.5%, Compliance With Laws: 5% |
| Neither party may disclose proprietary information received during the performance of this agreement to any person outside its organization unless disclosure is required by applicable law. | Confidentiality | Confidentiality | 44.5% |Confidentiality: 44.5%,  Disclosures: 6.5%, Publicity: 6.0% |
| If any provision of this agreement is determined to be invalid or unenforceable, the remaining provisions shall continue in full force and effect. | Severability | Severability | 94.5% |Severability: 94.5%, Governing Laws: 1.0%, Interpretations: 1.0% |
| All formal notices under this agreement must be delivered electronically to the addresses designated by the parties and shall become effective upon confirmed receipt. | Notices | Notices | 31% | Notices: 31.0%, Effective Dates: 22.5%, Effectiveness: 17.0% |
| A party's failure to exercise a contractual right immediately shall not prevent that party from exercising the same right at a later date. | Waivers / No Waiver | Waivers | 14% | Waivers: 14.0%, No Waivers: 13.0%, Payments: 6.5% |

**Results**

All 5/5 clauses (100%) were assigned to their intended category. However, prediction confidence varied considerably across the examples, ranging from 14.0% to 94.5%.

This was a small qualitative test on previously unseen clauses and should not be interpreted as a formal accuracy measurement.

---

## Limitations

- The dataset contains substantial class imbalance.
- Some legally similar categories are difficult to distinguish.
- The transformer experiments used smaller subsets of the dataset and limited training configurations.
- The five manually created clauses are only a qualitative application test, not a statistical evaluation.
- The system is intended as a classification demonstration and should not be treated as legal advice.
  
--- 
## Example

<img width="1379" height="569" alt="Screenshot 2026-09-29 at 9 25 55 PM" src="https://github.com/user-attachments/assets/5d98929f-9335-4522-af78-1bd5d9d5dae0" />



<img width="1375" height="449" alt="Screenshot 2026-09-29 at 9 24 06 PM" src="https://github.com/user-attachments/assets/4712d4c0-a131-493e-8b27-335a090df510" />

