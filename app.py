import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

HF_REPO = "SH205/legal-clause-random-forest"

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Legal Clause Classification",
    page_icon="⚖️",
    layout="wide"
)


# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_models():
    st.write("1. Starting model loading...")

    st.write("2. Downloading Random Forest...")

    rf_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="legal_clause_random_forest.joblib"
    )

    st.write("3. Random Forest downloaded.")

    st.write("4. Loading Random Forest into memory...")

    rf_model = joblib.load(rf_path)

    st.write("5. ✅ Random Forest loaded.")

    st.write("6. Downloading TF-IDF...")

    tfidf_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="legal_clause_tfidf.joblib"
    )

    st.write("7. TF-IDF downloaded.")

    st.write("8. Loading TF-IDF...")

    tfidf = joblib.load(tfidf_path)

    st.write("9. ✅ TF-IDF loaded.")

    return rf_model, tfidf


rf_model, tfidf = load_models()

st.write("10. ✅ Both models loaded.")
st.write("11. ✅ Starting Streamlit interface...")

# -----------------------------
# Label names
# -----------------------------

label_names = [
    "Adjustments",
    "Agreements",
    "Amendments",
    "Anti-Corruption Laws",
    "Applicable Laws",
    "Approvals",
    "Arbitration",
    "Assignments",
    "Assigns",
    "Authority",
    "Authorizations",
    "Base Salary",
    "Benefits",
    "Binding Effects",
    "Books",
    "Brokers",
    "Capitalization",
    "Change In Control",
    "Closings",
    "Compliance With Laws",
    "Confidentiality",
    "Consent To Jurisdiction",
    "Consents",
    "Construction",
    "Cooperation",
    "Costs",
    "Counterparts",
    "Death",
    "Defined Terms",
    "Definitions",
    "Disability",
    "Disclosures",
    "Duties",
    "Effective Dates",
    "Effectiveness",
    "Employment",
    "Enforceability",
    "Enforcements",
    "Entire Agreements",
    "Erisa",
    "Existence",
    "Expenses",
    "Fees",
    "Financial Statements",
    "Forfeitures",
    "Further Assurances",
    "General",
    "Governing Laws",
    "Headings",
    "Indemnifications",
    "Indemnity",
    "Insurances",
    "Integration",
    "Intellectual Property",
    "Interests",
    "Interpretations",
    "Jurisdictions",
    "Liens",
    "Litigations",
    "Miscellaneous",
    "Modifications",
    "No Conflicts",
    "No Defaults",
    "No Waivers",
    "Non-Disparagement",
    "Notices",
    "Organizations",
    "Participations",
    "Payments",
    "Positions",
    "Powers",
    "Publicity",
    "Qualifications",
    "Records",
    "Releases",
    "Remedies",
    "Representations",
    "Sales",
    "Sanctions",
    "Severability",
    "Solvency",
    "Specific Performance",
    "Submission To Jurisdiction",
    "Subsidiaries",
    "Successors",
    "Survival",
    "Tax Withholdings",
    "Taxes",
    "Terminations",
    "Terms",
    "Titles",
    "Transactions With Affiliates",
    "Use Of Proceeds",
    "Vacations",
    "Venues",
    "Vesting",
    "Waiver Of Jury Trials",
    "Waivers",
    "Warranties",
    "Withholdings"
]


# -----------------------------
# Prediction function
# -----------------------------

def predict_clause(text):

    text_tfidf = tfidf.transform([text])

    probabilities = rf_model.predict_proba(
        text_tfidf
    )[0]

    top_indices = probabilities.argsort()[-3:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "Clause": label_names[index],
            "Probability": probabilities[index]
        })

    return pd.DataFrame(results)


# -----------------------------
# Explainability
# -----------------------------

def explain_prediction(text):

    text_tfidf = tfidf.transform([text])

    probabilities = rf_model.predict_proba(
        text_tfidf
    )[0]

    prediction = probabilities.argmax()

    predicted_label = label_names[prediction]

    feature_names = tfidf.get_feature_names_out()

    feature_values = text_tfidf.toarray()[0]

    importances = rf_model.feature_importances_

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "TF-IDF": feature_values,
        "Importance": importances
    })

    explanation["Contribution"] = (
        explanation["TF-IDF"] *
        explanation["Importance"]
    )

    explanation = explanation[
        explanation["Contribution"] > 0
    ].sort_values(
        "Contribution",
        ascending=False
    ).head(10)

    return predicted_label, explanation


# -----------------------------
# User interface
# -----------------------------

st.title("⚖️ Legal Clause Classification")

st.write(
    "Classify legal clauses into one of 100 legal document categories "
    "using a trained Random Forest + TF-IDF model."
)

st.divider()

text = st.text_area(
    "Enter a legal clause",
    height=220,
    placeholder=(
        "Example: The substantive laws of the State of Texas "
        "will govern the validity, construction and enforcement "
        "of this Agreement."
    )
)

if st.button("🔍 Classify Legal Clause", type="primary"):

    if not text.strip():

        st.warning("Please enter a legal clause.")

    else:
        st.write("✅ About to run prediction...")
        results = predict_clause(text)
        st.write("✅ Prediction completed.")

        predicted_label = results.iloc[0]["Clause"]
        confidence = results.iloc[0]["Probability"]

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Clause",
                predicted_label
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )

        st.subheader("Top 3 Predictions")

        display_results = results.copy()

        display_results["Probability"] = (
            display_results["Probability"]
            .map(lambda x: f"{x:.1%}")
        )

        st.table(display_results)

        st.subheader("Why did the model make this prediction?")

        predicted_label, explanation = explain_prediction(text)

        st.dataframe(
            explanation,
            use_container_width=True,
            hide_index=True
        )

st.divider()

st.caption(
    "Model: TF-IDF + Random Forest | "
    "Dataset: LEDGAR | "
    "100 legal clause categories"
)
