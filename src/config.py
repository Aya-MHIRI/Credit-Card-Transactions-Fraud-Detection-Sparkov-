"""Configuration centrale du projet (jeu n°1 : Credit Card Transactions Fraud Detection, Sparkov)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

# Colonnes du jeu Sparkov
TARGET = "is_fraud"
TIME_COL = "trans_date_trans_time"
AMT = "amt"
CARD = "cc_num"

# Séparation chronologique 60 / 20 / 20
SPLIT = (0.6, 0.2, 0.2)
RANDOM_STATE = 42

# Budget d'alertes par défaut : part des transactions que l'équipe peut examiner.
# 1 % ≈ 1,7 x le taux de fraude de Sparkov (0,58 %) ; ~34 alertes/jour pour ~3 400 transactions/jour.
BUDGET_RATE = 0.01
# Coût d'analyse d'une alerte (même unité que le montant : dollars). À adapter.
REVIEW_COST = 5.0

KAGGLE_HANDLE = "kartik2112/fraud-detection"

# clé -> (nom affiché, type)
MODELS = {
    "xgboost": ("XGBoost", "supervisé"),
    "logreg": ("Régression logistique pondérée", "supervisé"),
    "iforest": ("Isolation Forest", "non supervisé"),
    "lof": ("Local Outlier Factor", "non supervisé"),
    "autoencoder": ("Autoencodeur (MLP)", "non supervisé"),
}
DEPLOYED = "xgboost"  # modèle exposé par l'API et expliqué par SHAP
