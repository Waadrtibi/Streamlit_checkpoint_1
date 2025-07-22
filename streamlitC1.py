import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Charger les données
data_path = r"C:\Users\Waad RTIBI\Streamlit_checkpoint_1\Expresso_churn_cleaned.csv"
df = pd.read_csv(data_path)

# Séparer les features et la cible
X = df.drop(columns=["CHURN"])
y = df["CHURN"]

# Encodage des variables catégorielles
X_encoded = pd.get_dummies(X)

# Entraînement du modèle
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarder le modèle et les colonnes
joblib.dump(model, r"C:\Users\Waad RTIBI\Streamlit_checkpoint_1\expresso_churn_model.pkl")
joblib.dump(X_encoded.columns.tolist(), r"C:\Users\Waad RTIBI\Streamlit_checkpoint_1\model_columns.pkl")
