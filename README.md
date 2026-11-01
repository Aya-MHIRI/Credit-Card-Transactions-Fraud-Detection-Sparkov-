# Challenge 2 — Application web de détection de fraude explicable

Data Mining · II3-IA / II3-IF · ENSI · Jeu de données n°1 : **Credit Card Transactions Fraud Detection (Sparkov)**.

> ⚠️ Cette archive contient des **artefacts de démonstration** (données simulées) pour que l'application s'ouvre tout de suite.
> Un bandeau jaune « données de démonstration » s'affiche tant que tu n'as pas relancé `train.py` sur le vrai jeu.
> **Ne rends jamais les chiffres de la démo.**

---

## 1. Ce que tu dois faire (dans l'ordre)

### Étape 0 — Installer (une seule fois)
Python 3.10 ou plus. Dans un terminal ouvert **dans le dossier du projet** :

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Étape 1 — Récupérer le vrai jeu de données
Télécharge sur Kaggle **kartik2112/fraud-detection** (« Credit Card Transactions Fraud Detection ») et place les deux fichiers
`fraudTrain.csv` et `fraudTest.csv` dans le dossier `data/`.
(Alternative : si tu as un compte Kaggle configuré, `python train.py` tente le téléchargement automatique via `kagglehub`.)

### Étape 2 — Entraîner (génère tout dans `artifacts/`)
```bash
python train.py --max-cards 200 --fast   # essai rapide (quelques minutes) pour vérifier que tout marche
python train.py                          # entraînement complet (≈ 10 à 20 min, ~1,85 M de transactions)
```
Options utiles : `--review-cost 5` (coût d'analyse d'une alerte en $), `--budget 0.01` (budget d'alertes = 1 % des transactions).
Sans le vrai jeu, `python train.py --demo` génère des données simulées pour tester.

À la fin, ouvre **`artifacts/rapport_resultats.md`** : tous les tableaux (séparation, stratégies de déséquilibre, comparaison des modèles,
budgets, variables influentes) sont déjà prêts à copier dans ton compte rendu.

### Étape 3 — Lancer l'application web
```bash
streamlit run app.py
```
Ouvre <http://localhost:8501>. Six pages : Données · Préparation · Modèles · Seuil et coût · Explicabilité · API.

### Étape 4 — Lancer l'API (dans un 2e terminal, venv activé)
```bash
uvicorn api:app --reload
```
Documentation interactive : <http://localhost:8000/docs>. Exemple :
```bash
curl -X POST "http://localhost:8000/predict?top_reasons=5" -H "Content-Type: application/json" -d '{
  "cc_num": 2703186189652095, "trans_date_trans_time": "2020-07-02T02:14:00", "amt": 842.5,
  "category": "shopping_net", "gender": "F", "dob": "1988-03-09", "city_pop": 3495,
  "lat": 36.0788, "long": -81.1781, "merch_lat": 36.011293, "merch_long": -82.048315 }'
```

### (Facultatif) Tests automatiques
```bash
pip install pytest
python -m pytest tests -q
```
Ils vérifient les variables sans fuite, la cohérence API ↔ entraînement et que chaque page de l'application s'affiche.

---

## 2. Correspondance avec le sujet

| Module du sujet | Où |
|---|---|
| 1 · Données (taux de fraude, montants par classe, évolution, par heure/catégorie/genre/État, import ou téléchargement) | `app.py` page 1, `src/data.py` |
| 2 · Variables temporelles + comportementales, séparation 60/20/20, déséquilibre justifié | `src/features.py`, `src/imbalance.py`, `train.py`, page 2 |
| 3 · 1+ supervisé, 2+ non supervisés (Isolation Forest + LOF + autoencodeur), tableau comparatif | `src/models.py`, page 3 |
| 4 · PR-AUC, ROC-AUC, précision@k / rappel@k, F2, MCC, courbes PR superposées, seuil à coût minimal, curseurs live | `src/evaluation.py`, pages 3-4 |
| 5 · SHAP beeswarm, waterfall, codes motifs | `src/explain.py`, page 5 |
| 6 · API FastAPI | `api.py`, `src/serving.py`, page 6 |

```
app.py            application Streamlit            train.py     pipeline complet → artifacts/
api.py            API FastAPI                      src/         config, data, features, models, évaluation, SHAP, serving
data/             mettre fraudTrain.csv + fraudTest.csv ici       tests/       tests automatiques
artifacts/        modèles, scores, métriques, graphiques (générés par train.py)
```

---

## 3. Choix de conception à savoir défendre à l'oral

- **Séparation chronologique** (60 % / 20 % / 20 % après tri par date) : le test est toujours postérieur à l'entraînement, comme en production.
- **Pas de fuite** : les variables de vélocité (`n_tx_1h`, `n_tx_24h`, `amt_sum_24h`, `amt_ratio_30d`, `hours_since_prev`) n'utilisent que les transactions
  *strictement antérieures* de la même carte. Le même code sert à l'entraînement et à l'API (test `tests/test_api.py` : le score en ligne = le score hors ligne).
  Les colonnes `first`, `last`, `street`, `trans_num`, `merchant`, `job`, `zip` ne sont pas utilisées (identifiants / haute cardinalité).
- **Déséquilibre** : 5 stratégies (aucune, pondération, sous-échantillonnage, sur-échantillonnage, SMOTE) appliquées **au seul entraînement**, comparées sur la **validation** ;
  la meilleure PR-AUC est retenue. Conséquence à connaître : le rééchantillonnage décale les probabilités, donc le seuil retenu est souvent bas (ex. 0,01) ; il est choisi
  sur la validation précisément pour compenser ce décalage.
- **Métriques** : PR-AUC en priorité (classe très rare) ; ROC-AUC en complément ; précision@k / rappel@k pour un budget d'alertes ; F2 (le rappel compte deux fois plus) et MCC au seuil retenu.
- **Budget k** : par défaut 1 % des transactions ≈ 34 alertes/jour sur Sparkov (hypothèse : une petite équipe d'analystes). C'est un paramètre métier : modifie `BUDGET_RATE` dans `src/config.py` ou utilise `--budget`.
- **Coût métier** = montants des fraudes manquées + (coût d'analyse × nombre d'alertes). Seuil choisi sur la validation en minimisant ce coût, puis **confirmé sur le test** (le tableau montre aussi le coût du seuil optimal *a posteriori* pour mesurer l'écart).
  Le coût d'analyse par défaut (5 $) est une hypothèse : justifie la valeur que tu gardes.
- **Non supervisé** : Isolation Forest, LOF, autoencodeur (MLP) sont ajustés sans jamais voir l'étiquette. Leurs scores sont convertis en rang percentile (échelle de la validation) pour partager un curseur 0-1 avec les autres modèles.
  Un seuil non supervisé peut très mal se transférer de la validation au test (coût > « sans modèle ») : c'est un résultat à commenter, pas un bug.
- **Explicabilité** : SHAP (TreeExplainer) sur le modèle XGBoost déployé ; « codes motifs » = les 3 à 5 variables qui poussent le plus vers l'alerte.

### Limites à mentionner dans le rapport
Données **synthétiques** (simulateur Sparkov) : les fraudes suivent des règles de simulation, donc les scores sont plus optimistes que sur des données réelles.
Les probabilités ne sont pas calibrées après rééchantillonnage. Les exemples ex æquo (même seconde, même carte) ne se voient pas entre eux.
SMOTE crée des valeurs fractionnaires sur les variables binaires (catégories) : à discuter.

---

## 4. Dépannage

| Problème | Solution |
|---|---|
| `Aucun fichier CSV dans data/` | Place `fraudTrain.csv` et `fraudTest.csv` dans `data/` (étape 1). |
| Téléchargement Kaggle refusé | Configure `~/.kaggle/kaggle.json` ou télécharge à la main. |
| Mémoire insuffisante / trop lent | `python train.py --max-cards 300 --fast` (garde 300 cartes avec tout leur historique). |
| Erreur SHAP / XGBoost au chargement | `pip install -U shap xgboost` (les deux doivent être récents). |
| Port déjà utilisé | `streamlit run app.py --server.port 8502` · `uvicorn api:app --port 8001`. |
| « Aucun modèle entraîné » dans l'app | Lance `python train.py` (ou `--demo`) puis recharge la page. |
