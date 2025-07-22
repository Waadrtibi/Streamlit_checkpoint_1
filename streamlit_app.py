import streamlit as st
import pandas as pd
import joblib

# 🔄 Fonction pour charger le modèle
@st.cache_resource
def load_model():
    model = joblib.load("C:/Users/Waad RTIBI/Streamlit_checkpoint_1/expresso_churn_model.pkl")
    return model

# 📘 Fonction pour collecter les entrées utilisateur
def user_input_features():
    st.sidebar.header("📋 Données client")
    region = st.sidebar.selectbox("Region", ["Dakar", "Diourbel", "Fatick", "Kaolack", "Kaffrine"])
    tenure = st.sidebar.slider("Durée d'abonnement (mois)", 0, 60, 12)
    age = st.sidebar.slider("Âge", 18, 80, 30)
    has_fiber = st.sidebar.selectbox("Fibre", ["Yes", "No"])
    monthly_spend = st.sidebar.slider("Dépenses mensuelles", 0, 100000, 25000)
    support_calls = st.sidebar.slider("Appels au support", 0, 50, 5)

    data = {
        'REGION': region,
        'TENURE': tenure,
        'AGE': age,
        'FIBER': 1 if has_fiber == "Yes" else 0,
        'MONTANT': monthly_spend,
        'FREQUENCE_RECH': support_calls,
    }
    return pd.DataFrame([data])

# 🚀 Titre de l’application
st.title("📉 Prédiction de Churn Client Expresso")

# 🧠 Chargement du modèle
model = load_model()

# 🧾 Entrée utilisateur
input_df = user_input_features()

# 🧪 Prédiction
if st.button("🔍 Prédire le churn"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Le client risque de partir. (Probabilité : {proba:.2%})")
    else:
        st.success(f"✅ Le client est fidèle. (Probabilité de churn : {proba:.2%})")
