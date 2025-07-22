import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger le modèle et les colonnes
model = joblib.load(r"C:\Users\Waad RTIBI\Streamlit_checkpoint_1\expresso_churn_model.pkl")
model_columns = joblib.load(r"C:\Users\Waad RTIBI\Streamlit_checkpoint_1\model_columns.pkl")

st.title("📱 Prédiction de Churn - Expresso")

# Interface utilisateur
ARPU = st.number_input("ARPU", min_value=0.0)
FREQ_RECH = st.number_input("FREQ_RECH", min_value=0)
FREQ_CALL = st.number_input("FREQ_CALL", min_value=0)
MONTANT = st.number_input("MONTANT", min_value=0)
REVENUE = st.number_input("REVENUE", min_value=0)
MRG = st.number_input("MRG", min_value=0)
REGION = st.selectbox("REGION", ['DAKAR', 'THIES', 'KAOLACK', 'FATICK', 'SAINT-LOUIS', 'ZIGUINCHOR', 'KAFFRINE'])

if st.button("Prédire"):
    # Créer un DataFrame avec les valeurs utilisateur
    input_dict = {
        "ARPU": ARPU,
        "FREQ_RECH": FREQ_RECH,
        "FREQ_CALL": FREQ_CALL,
        "MONTANT": MONTANT,
        "REVENUE": REVENUE,
        "MRG": MRG,
        "REGION": REGION
    }
    input_df = pd.DataFrame([input_dict])

    # Encodage
    input_encoded = pd.get_dummies(input_df)

    # Ajouter les colonnes manquantes
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Réordonner les colonnes
    input_encoded = input_encoded[model_columns]

    # Afficher les classes du modèle pour debug
    st.write("📊 Classes du modèle :", model.classes_)

    # Prédiction
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]

    # Gérer cas 1 seule classe
    if len(model.classes_) == 2:
        idx_1 = np.where(model.classes_ == 1)[0][0]
        prediction_proba = proba[idx_1]
    else:
        prediction_proba = 0.0  # ou afficher un message

    st.success(f"🔍 Résultat : {'Churn' if prediction == 1 else 'Non-Churn'} (Probabilité: {prediction_proba:.2%})")
