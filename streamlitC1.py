import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 📥 1. Charger les données
file_path = r"C:\Users\Waad RTIBI\Streamlit_checkpoint_1\Expresso_churn_cleaned.csv"
df = pd.read_csv(file_path)

# ✅ Vérifier que la colonne cible s'appelle bien 'churn'
if 'CHURN' in df.columns:
    df.rename(columns={'CHURN': 'churn'}, inplace=True)

# 🧼 2. S’assurer que la colonne 'churn' est bien binaire
df['churn'] = df['churn'].astype(int)

# 🧠 3. Préparation des features et de la cible
X = df.drop(columns=['churn'])
y = df['churn']

# 🎲 4. Division du dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🌲 5. Modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📊 6. Évaluation
y_pred = model.predict(X_test)
print("\n🎯 Résultats de l’évaluation :")
print(classification_report(y_test, y_pred))
print("🔎 Précision globale :", accuracy_score(y_test, y_pred))

# 💾 7. Sauvegarde du modèle
model_path = r"C:\Users\Waad RTIBI\Streamlit_checkpoint_1\expresso_churn_model.pkl"
joblib.dump(model, model_path)

print(f"\n✅ Modèle sauvegardé avec succès dans : {model_path}")
