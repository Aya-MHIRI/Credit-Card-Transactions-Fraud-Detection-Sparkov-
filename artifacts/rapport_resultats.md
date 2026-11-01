# Résultats d'entraînement

Généré le 2026-11-01T17:54:43 — 1,852,394 transactions, 9,651 fraudes (0.521%), 999 cartes.

## Séparation chronologique 60/20/20
| jeu | n | frauds | fraud_rate | start | end | fraud_amount |
|---|---|---|---|---|---|---|
| train | 1111436 | 6401 | 0.005759 | 2019-01-01 00:00:18 | 2020-04-07 07:00:14 | 3.398e+06 |
| val | 370479 | 1901 | 0.005131 | 2020-04-07 07:00:36 | 2020-08-25 01:31:21 | 9.893e+05 |
| test | 370479 | 1349 | 0.003641 | 2020-08-25 01:31:59 | 2020-12-31 23:59:34 | 7.339e+05 |

## Stratégies contre le déséquilibre (appliquées à l'entraînement, comparées en validation)
Retenue : **Sur-échantillonnage aléatoire (1 fraude : 10 normales)** (meilleure PR-AUC en validation).

| Stratégie | taille entraînement | PR-AUC (val) | ROC-AUC (val) | Précision@k (val) | Rappel@k (val) | Coût min (val) |
|---|---|---|---|---|---|---|
| Aucune (référence) | 1111436 | 0.9747 | 0.9996 | 0.5055 | 0.9853 | 2.074e+04 |
| Pondération des classes (scale_pos_weight) | 1111436 | 0.9787 | 0.9995 | 0.5055 | 0.9853 | 1.909e+04 |
| Sous-échantillonnage aléatoire (10 normales : 1 fraude) | 70411 | 0.9708 | 0.9995 | 0.5039 | 0.9821 | 1.995e+04 |
| Sur-échantillonnage aléatoire (1 fraude : 10 normales) | 1215538 | 0.9803 | 0.9996 | 0.5066 | 0.9874 | 1.89e+04 |
| SMOTE (1 fraude synthétique : 10 normales) | 1215538 | 0.9797 | 0.9996 | 0.5058 | 0.9858 | 1.901e+04 |

## Comparaison de tous les modèles
Budget k = 1.00% des transactions (3705 alertes sur le test, ≈ 28.7 alertes/jour). Coût d'analyse = 5.0 $ par alerte.
Le seuil de chaque modèle est choisi sur la **validation** (coût minimal) puis appliqué tel quel au **test**.

| Modèle | Type | PR-AUC (val) | PR-AUC (test) | ROC-AUC (test) | Précision@k (test) | Rappel@k (test) | Seuil (choisi sur val) | F2 (test) | MCC (test) | Alertes (test) | Coût val ($) | Coût test ($) | Coût test optimal ($) | Économie vs sans modèle (test) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | supervisé | 0.9803 | 0.9772 | 0.9994 | 0.3606 | 0.9904 | 0.07655 | 0.8696 | 0.7655 | 2205 | 1.89e+04 | 1.591e+04 | 1.434e+04 | 0.9783 |
| Régression logistique pondérée | supervisé | 0.3611 | 0.2957 | 0.9869 | 0.2869 | 0.788 | 0.72 | 0.5252 | 0.4172 | 5999 | 5.649e+04 | 5.149e+04 | 4.962e+04 | 0.9298 |
| Isolation Forest | non supervisé | 0.2543 | 0.1749 | 0.9418 | 0.1843 | 0.5063 | 0.9706 | 0.2259 | 0.2003 | 16033 | 1.365e+05 | 1.38e+05 | 1.264e+05 | 0.812 |
| Local Outlier Factor | non supervisé | 0.08418 | 0.03782 | 0.8766 | 0.05992 | 0.1646 | 0.8003 | 0.05745 | 0.0816 | 92513 | 5.419e+05 | 6.059e+05 | 5.538e+05 | 0.1744 |
| Autoencodeur (MLP) | non supervisé | 0.1734 | 0.1418 | 0.9196 | 0.153 | 0.4203 | 0.9436 | 0.1529 | 0.1536 | 26873 | 2.89e+05 | 2.629e+05 | 2.599e+05 | 0.6418 |

## Modèle déployé : XGBoost
- Seuil (coût minimal sur la validation) : **0.077**
- Test : précision 0.600, rappel 0.980, F2 0.870, MCC 0.766
- Coût test au seuil retenu : 15,911 $ (sans modèle : 733,912 $ ; seuil optimal a posteriori : 14,342 $)

### Précision / rappel selon le budget d'alertes (test)
| Budget (% transactions) | k (alertes) | Alertes / jour | Précision@k | Rappel@k |
|---|---|---|---|---|
| 0.1 | 371 | 2.876 | 1 | 0.275 |
| 0.25 | 927 | 7.186 | 1 | 0.6872 |
| 0.5 | 1853 | 14.36 | 0.7113 | 0.977 |
| 1 | 3705 | 28.72 | 0.3606 | 0.9904 |
| 2 | 7410 | 57.44 | 0.1811 | 0.9948 |
| 5 | 18524 | 143.6 | 0.07255 | 0.9963 |

## Variables les plus influentes (|SHAP| moyen)
| variable | |SHAP| moyen |
|---|---|
| amt | 3.181 |
| amt_sum_24h | 1.58 |
| log_amt | 0.8361 |
| n_tx_24h | 0.7565 |
| is_night | 0.5925 |
| hour_sin | 0.5806 |
| hour_cos | 0.4993 |
| hours_since_prev | 0.4877 |
