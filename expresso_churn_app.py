# === 📦 Importer les bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import warnings
warnings.filterwarnings('ignore')

# === 📁 Charger les données
file_path = "C:/Users/Waad RTIBI/Streamlit_checkpoint_1/Expresso_churn_dataset.csv"
df = pd.read_csv(file_path)

print("✅ Aperçu du dataset :")
print(df.head())

# === ℹ️ Informations générales
print("\n🔍 Info :")
print(df.info())

# === 📊 Générer un rapport de profilage
print("\n📄 Génération du rapport HTML...")
profile = ProfileReport(df, title="Expresso Churn Profiling Report", explorative=True)
profile.to_file("expresso_churn_report.html")

# === ❌ Gérer les valeurs manquantes
print("\n🧹 Traitement des valeurs manquantes :")
missing = df.isnull().sum()
print(missing[missing > 0])

# Exemple : remplacer ou supprimer selon les cas
if 'arpu_change' in df.columns:
    df['arpu_change'].fillna(df['arpu_change'].median(), inplace=True)

if 'seniority' in df.columns:
    df.dropna(subset=['seniority'], inplace=True)

if 'freq_top_pack_change' in df.columns:
    df.dropna(subset=['freq_top_pack_change'], inplace=True)

# === 📌 Supprimer les doublons
nb_duplicates = df.duplicated().sum()
print(f"\n🗑️ Doublons supprimés : {nb_duplicates}")
df.drop_duplicates(inplace=True)

# === ⚠️ Détection et suppression des outliers (méthode IQR)
print("\n📦 Suppression des outliers numériques :")
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if not outliers.empty:
        print(f" - {col}: {len(outliers)} valeurs supprimées")
        df = df[(df[col] >= lower) & (df[col] <= upper)]

# === 🔤 Encodage des variables catégorielles
print("\n🔠 Encodage des variables catégorielles :")
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("Colonnes catégorielles :", cat_cols)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# === 💾 Enregistrer le fichier nettoyé
output_path = "C:/Users/Waad RTIBI/Streamlit_checkpoint_1/Expresso_churn_cleaned.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Données nettoyées sauvegardées ici : {output_path}")
