"""
Fraud Detection Pipeline avec ops0
==================================

Ce pipeline démontre les capacités d'ops0 pour créer un système de détection
de fraude en temps réel, scalable et production-ready avec zéro configuration.

Fonctionnalités démontrées:
- Décorateurs @ops0.step et @ops0.pipeline
- Gestion automatique des dépendances
- Storage transparent entre étapes
- Gestion des modèles ML
- Monitoring et alerting
- Feature engineering avancé
"""

import ops0
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Pour l'exemple, on utilise des imports conditionnels
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Note: scikit-learn requis pour l'exécution complète")


@ops0.step
def load_transactions(date: str, source: str = "s3://fraud-data/transactions") -> pd.DataFrame:
    """
    Charge les transactions pour une date donnée.

    ops0 détecte automatiquement l'usage de pandas et alloue la mémoire nécessaire.
    """
    # En production, ceci chargerait depuis S3
    # Pour l'exemple, on génère des données synthétiques
    np.random.seed(42)
    n_transactions = 10000
    n_fraudulent = int(n_transactions * 0.02)  # 2% de fraudes

    # Génération de transactions normales
    normal_transactions = pd.DataFrame({
        'transaction_id': range(n_transactions - n_fraudulent),
        'user_id': np.random.randint(1000, 5000, n_transactions - n_fraudulent),
        'merchant_id': np.random.randint(100, 500, n_transactions - n_fraudulent),
        'amount': np.random.lognormal(3.5, 1.5, n_transactions - n_fraudulent),
        'timestamp': pd.date_range(
            start=pd.to_datetime(date),
            periods=n_transactions - n_fraudulent,
            freq='1min'
        ),
        'merchant_category': np.random.choice(
            ['grocery', 'gas', 'restaurant', 'online', 'retail'],
            n_transactions - n_fraudulent
        ),
        'payment_method': np.random.choice(
            ['credit_card', 'debit_card', 'mobile_payment'],
            n_transactions - n_fraudulent,
            p=[0.5, 0.3, 0.2]
        ),
        'is_fraud': 0
    })

    # Génération de transactions frauduleuses avec des patterns anormaux
    fraud_transactions = pd.DataFrame({
        'transaction_id': range(n_transactions - n_fraudulent, n_transactions),
        'user_id': np.random.randint(1000, 5000, n_fraudulent),
        'merchant_id': np.random.randint(100, 500, n_fraudulent),
        'amount': np.concatenate([
            np.random.uniform(1000, 5000, n_fraudulent // 2),  # Montants élevés
            np.random.uniform(0.01, 1, n_fraudulent // 2)  # Micro-transactions
        ]),
        'timestamp': pd.date_range(
            start=pd.to_datetime(date),
            periods=n_fraudulent,
            freq='30s'  # Fréquence anormalement élevée
        ),
        'merchant_category': np.random.choice(
            ['jewelry', 'electronics', 'casino', 'crypto'],  # Catégories à risque
            n_fraudulent
        ),
        'payment_method': 'credit_card',  # Principalement cartes de crédit
        'is_fraud': 1
    })

    # Mélanger les transactions
    transactions = pd.concat([normal_transactions, fraud_transactions]).sample(frac=1).reset_index(drop=True)

    print(f"✅ Chargé {len(transactions)} transactions du {date}")
    print(f"   - Transactions normales: {(transactions.is_fraud == 0).sum()}")
    print(f"   - Transactions frauduleuses: {(transactions.is_fraud == 1).sum()}")

    return transactions


@ops0.step
def extract_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Extraction de features avancées pour la détection de fraude.

    ops0 détecte l'utilisation intensive de pandas et alloue automatiquement
    plus de mémoire pour cette étape.
    """
    df = transactions.copy()

    # Features temporelles
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

    # Features de montant
    df['amount_log'] = np.log1p(df['amount'])

    # Calcul des statistiques utilisateur (rolling window)
    user_stats = df.groupby('user_id').agg({
        'amount': ['mean', 'std', 'count'],
        'transaction_id': 'count'
    }).reset_index()
    user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount',
                          'user_amount_count', 'user_transaction_count']

    df = df.merge(user_stats, on='user_id', how='left')

    # Z-score du montant par rapport à l'historique utilisateur
    df['amount_zscore'] = (df['amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-5)

    # Features de vélocité (transactions rapides)
    df = df.sort_values(['user_id', 'timestamp'])
    df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
    df['velocity_flag'] = (df['time_since_last_transaction'] < 60).astype(int)  # < 1 minute

    # Features de merchant
    merchant_risk = {
        'jewelry': 3, 'electronics': 3, 'casino': 4, 'crypto': 4,
        'grocery': 1, 'gas': 1, 'restaurant': 2, 'online': 2, 'retail': 1
    }
    df['merchant_risk_score'] = df['merchant_category'].map(merchant_risk)

    # Features de pattern
    df['high_amount_flag'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    df['micro_transaction_flag'] = (df['amount'] < 1).astype(int)
    df['risky_hour_flag'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)

    # Score de risque composite
    df['risk_score'] = (
            df['velocity_flag'] * 2 +
            df['high_amount_flag'] * 3 +
            df['micro_transaction_flag'] * 1 +
            df['risky_hour_flag'] * 1 +
            df['merchant_risk_score'] +
            np.clip(df['amount_zscore'], 0, 5)
    )

    print(f"✅ Extraction de {len(df.columns)} features complétée")
    print(f"   - Features temporelles: hour, day_of_week, is_weekend, is_night")
    print(f"   - Features statistiques: z-scores, moyennes utilisateur")
    print(f"   - Features de risque: velocity, montants anormaux, merchants")

    # Sauvegarder les features pour analyse ultérieure
    ops0.save("extracted_features", df[['transaction_id', 'risk_score', 'amount_zscore']])

    return df


@ops0.step
def prepare_model_input(featured_transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare les données pour le modèle ML.

    Sépare features et target, gère les valeurs manquantes.
    """
    # Colonnes à utiliser pour la prédiction
    feature_columns = [
        'amount', 'amount_log', 'amount_zscore',
        'hour', 'day_of_week', 'is_weekend', 'is_night',
        'user_avg_amount', 'user_std_amount', 'user_transaction_count',
        'time_since_last_transaction', 'velocity_flag',
        'merchant_risk_score', 'high_amount_flag', 'micro_transaction_flag',
        'risky_hour_flag', 'risk_score'
    ]

    # Gérer les valeurs manquantes
    df = featured_transactions.copy()
    df['time_since_last_transaction'].fillna(3600, inplace=True)  # 1 heure par défaut
    df[feature_columns] = df[feature_columns].fillna(0)

    # Encoder les variables catégorielles
    payment_dummies = pd.get_dummies(df['payment_method'], prefix='payment')
    X = pd.concat([df[feature_columns], payment_dummies], axis=1)

    # Target
    y = df['is_fraud']

    print(f"✅ Données préparées: {X.shape[0]} transactions, {X.shape[1]} features")

    # Sauvegarder les colonnes pour le scoring
    ops0.save("model_features", list(X.columns))

    return X, y


@ops0.step(memory=2048)  # Override pour plus de mémoire car entraînement ML
def train_fraud_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Entraîne un modèle de détection de fraude.

    ops0 détecte l'usage de scikit-learn et alloue automatiquement
    les ressources nécessaires.
    """
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèle 1: Isolation Forest pour la détection d'anomalies
    iso_forest = IsolationForest(
        contamination=0.02,  # On s'attend à 2% de fraudes
        random_state=42,
        n_estimators=100
    )
    iso_forest.fit(X_train_scaled[y_train == 0])  # Entraîner sur transactions normales

    # Modèle 2: Random Forest pour classification supervisée
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Important pour données déséquilibrées
    )
    rf_classifier.fit(X_train_scaled, y_train)

    # Évaluation
    from sklearn.metrics import classification_report, roc_auc_score

    # Prédictions
    iso_predictions = iso_forest.predict(X_test_scaled)
    iso_predictions = (iso_predictions == -1).astype(int)  # -1 = anomalie = fraude

    rf_predictions = rf_classifier.predict(X_test_scaled)
    rf_probabilities = rf_classifier.predict_proba(X_test_scaled)[:, 1]

    # Métriques
    iso_accuracy = (iso_predictions == y_test).mean()
    rf_accuracy = (rf_predictions == y_test).mean()
    rf_auc = roc_auc_score(y_test, rf_probabilities)

    # Rapport détaillé pour Random Forest
    report = classification_report(y_test, rf_predictions, output_dict=True)

    print(f"✅ Modèles entraînés avec succès")
    print(f"   - Isolation Forest Accuracy: {iso_accuracy:.2%}")
    print(f"   - Random Forest Accuracy: {rf_accuracy:.2%}")
    print(f"   - Random Forest AUC: {rf_auc:.3f}")
    print(f"   - Precision (fraude): {report['1']['precision']:.2%}")
    print(f"   - Recall (fraude): {report['1']['recall']:.2%}")

    # Sauvegarder les modèles et le scaler
    ops0.save_model(scaler, "fraud_scaler")
    ops0.save_model(iso_forest, "fraud_isolation_forest")
    ops0.save_model(rf_classifier, "fraud_random_forest", {
        "accuracy": rf_accuracy,
        "auc": rf_auc,
        "precision": report['1']['precision'],
        "recall": report['1']['recall']
    })

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    print("\n📊 Top 10 features importantes:")
    for _, row in feature_importance.iterrows():
        print(f"   - {row['feature']}: {row['importance']:.3f}")

    ops0.save("feature_importance", feature_importance)

    return {
        "iso_accuracy": iso_accuracy,
        "rf_accuracy": rf_accuracy,
        "rf_auc": rf_auc,
        "precision": report['1']['precision'],
        "recall": report['1']['recall'],
        "f1_score": report['1']['f1-score']
    }


@ops0.step
def score_transactions(transactions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Score les nouvelles transactions avec les modèles entraînés.
    """
    # Charger les modèles
    scaler = ops0.load_model("fraud_scaler")
    iso_forest = ops0.load_model("fraud_isolation_forest")
    rf_classifier = ops0.load_model("fraud_random_forest")

    # Préparer les features (même format que l'entraînement)
    model_features = ops0.load("model_features")

    # S'assurer que toutes les colonnes sont présentes
    for col in model_features:
        if col not in features.columns:
            features[col] = 0

    X = features[model_features]
    X_scaled = scaler.transform(X)

    # Prédictions
    iso_scores = iso_forest.decision_function(X_scaled)
    iso_predictions = (iso_scores < 0).astype(int)

    rf_probabilities = rf_classifier.predict_proba(X_scaled)[:, 1]
    rf_predictions = (rf_probabilities > 0.5).astype(int)

    # Combiner les résultats
    results = transactions.copy()
    results['isolation_score'] = -iso_scores  # Plus c'est haut, plus c'est suspect
    results['isolation_prediction'] = iso_predictions
    results['rf_probability'] = rf_probabilities
    results['rf_prediction'] = rf_predictions

    # Score final combiné (moyenne pondérée)
    results['fraud_score'] = (
            0.3 * results['isolation_score'].clip(0, 1) +  # Normaliser à [0,1]
            0.7 * results['rf_probability']
    )

    # Décision finale
    results['is_fraudulent'] = (results['fraud_score'] > 0.7).astype(int)

    print(f"✅ Scoring complété pour {len(results)} transactions")
    print(f"   - Transactions suspectes (score > 0.7): {results['is_fraudulent'].sum()}")
    print(f"   - Score moyen: {results['fraud_score'].mean():.3f}")

    return results


@ops0.step
def generate_alerts(scored_transactions: pd.DataFrame) -> Dict[str, any]:
    """
    Génère des alertes pour les transactions à haut risque.

    Différents niveaux d'alerte selon le score de fraude.
    """
    high_risk = scored_transactions[scored_transactions['fraud_score'] > 0.9]
    medium_risk = scored_transactions[
        (scored_transactions['fraud_score'] > 0.7) &
        (scored_transactions['fraud_score'] <= 0.9)
        ]

    alerts = []

    # Alertes critiques
    for _, trans in high_risk.iterrows():
        alert = {
            'level': 'CRITICAL',
            'transaction_id': trans['transaction_id'],
            'user_id': trans['user_id'],
            'amount': trans['amount'],
            'fraud_score': trans['fraud_score'],
            'timestamp': trans['timestamp'],
            'reason': _get_fraud_reason(trans)
        }
        alerts.append(alert)

    # Alertes moyennes
    for _, trans in medium_risk.iterrows():
        alert = {
            'level': 'WARNING',
            'transaction_id': trans['transaction_id'],
            'user_id': trans['user_id'],
            'amount': trans['amount'],
            'fraud_score': trans['fraud_score'],
            'timestamp': trans['timestamp'],
            'reason': _get_fraud_reason(trans)
        }
        alerts.append(alert)

    # Statistiques
    stats = {
        'total_transactions': len(scored_transactions),
        'critical_alerts': len(high_risk),
        'warning_alerts': len(medium_risk),
        'total_flagged': len(high_risk) + len(medium_risk),
        'flagged_amount': high_risk['amount'].sum() + medium_risk['amount'].sum(),
        'alerts': alerts
    }

    print(f"🚨 Alertes générées:")
    print(f"   - Alertes CRITIQUES: {stats['critical_alerts']}")
    print(f"   - Alertes WARNING: {stats['warning_alerts']}")
    print(f"   - Montant total à risque: ${stats['flagged_amount']:,.2f}")

    # Sauvegarder les alertes
    if alerts:
        ops0.save(f"fraud_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}", alerts)

        # En production, ceci enverrait vraiment des notifications
        if stats['critical_alerts'] > 0:
            ops0.notify.slack(
                f"🚨 {stats['critical_alerts']} transactions critiques détectées! "
                f"Montant total: ${stats['flagged_amount']:,.2f}"
            )

    return stats


@ops0.step
def save_results(scored_transactions: pd.DataFrame, alerts_stats: Dict[str, any]) -> Dict[str, str]:
    """
    Sauvegarde les résultats pour audit et analyse.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Sauvegarder les transactions scorées
    ops0.save(f"scored_transactions_{timestamp}", scored_transactions)

    # Sauvegarder un rapport de synthèse
    report = {
        'timestamp': timestamp,
        'total_transactions': alerts_stats['total_transactions'],
        'fraudulent_transactions': alerts_stats['total_flagged'],
        'fraud_rate': alerts_stats['total_flagged'] / alerts_stats['total_transactions'],
        'total_amount': scored_transactions['amount'].sum(),
        'fraudulent_amount': alerts_stats['flagged_amount'],
        'critical_alerts': alerts_stats['critical_alerts'],
        'warning_alerts': alerts_stats['warning_alerts'],
        'model_performance': {
            'avg_fraud_score': scored_transactions['fraud_score'].mean(),
            'max_fraud_score': scored_transactions['fraud_score'].max(),
            'isolation_detections': scored_transactions['isolation_prediction'].sum(),
            'rf_detections': scored_transactions['rf_prediction'].sum()
        }
    }

    ops0.save(f"fraud_report_{timestamp}", report)

    print(f"✅ Résultats sauvegardés")
    print(f"   - Transactions scorées: scored_transactions_{timestamp}")
    print(f"   - Rapport de synthèse: fraud_report_{timestamp}")
    print(f"   - Taux de fraude détecté: {report['fraud_rate']:.2%}")

    return {
        'status': 'completed',
        'timestamp': timestamp,
        'transactions_file': f"scored_transactions_{timestamp}",
        'report_file': f"fraud_report_{timestamp}"
    }


@ops0.pipeline
def fraud_detection_pipeline(date: str = None) -> Dict[str, any]:
    """
    Pipeline complet de détection de fraude.

    Ce pipeline:
    1. Charge les transactions du jour
    2. Extrait des features avancées
    3. Prépare les données pour le ML
    4. Entraîne/met à jour les modèles
    5. Score toutes les transactions
    6. Génère des alertes pour les cas suspects
    7. Sauvegarde les résultats

    ops0 gère automatiquement:
    - L'orchestration des étapes
    - La sérialisation des données entre étapes
    - L'allocation des ressources
    - La parallélisation quand possible
    - Le monitoring et les logs
    """
    # Date par défaut: aujourd'hui
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    # Pipeline d'exécution
    transactions = load_transactions(date)
    featured_transactions = extract_features(transactions)
    X, y = prepare_model_input(featured_transactions)

    # Entraînement (peut être conditionnel en production)
    model_metrics = train_fraud_model(X, y)

    # Scoring
    scored = score_transactions(transactions, featured_transactions)

    # Alertes
    alerts_stats = generate_alerts(scored)

    # Sauvegarde
    results = save_results(scored, alerts_stats)

    # Résumé final
    summary = {
        'date': date,
        'model_metrics': model_metrics,
        'detection_stats': alerts_stats,
        'output_files': results,
        'pipeline_status': 'success'
    }

    return summary


# Fonction helper privée
def _get_fraud_reason(transaction: pd.Series) -> str:
    """Génère une explication pour pourquoi une transaction est suspecte."""
    reasons = []

    if transaction.get('velocity_flag', 0) == 1:
        reasons.append("Transaction trop rapide")

    if transaction.get('high_amount_flag', 0) == 1:
        reasons.append(f"Montant anormalement élevé (${transaction['amount']:.2f})")

    if transaction.get('risky_hour_flag', 0) == 1:
        reasons.append(f"Heure suspecte ({transaction['hour']}h)")

    if transaction.get('merchant_risk_score', 0) >= 3:
        reasons.append(f"Merchant à risque ({transaction['merchant_category']})")

    if transaction.get('amount_zscore', 0) > 3:
        reasons.append("Montant très différent du comportement habituel")

    return " | ".join(reasons) if reasons else "Pattern anormal détecté"


@ops0.pipeline(schedule="0 * * * *")  # Toutes les heures
def fraud_monitoring_pipeline() -> Dict[str, any]:
    """
    Pipeline de monitoring continu (version allégée pour production).

    Exécuté toutes les heures pour scorer les nouvelles transactions
    sans réentraîner le modèle.
    """
    # Charger seulement les transactions de la dernière heure
    current_time = datetime.now()
    one_hour_ago = (current_time - timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')

    # Pipeline simplifié
    transactions = load_transactions(one_hour_ago)
    featured_transactions = extract_features(transactions)
    scored = score_transactions(transactions, featured_transactions)
    alerts_stats = generate_alerts(scored)

    print(f"✅ Monitoring complété: {alerts_stats['total_flagged']} transactions suspectes")

    return {
        'timestamp': current_time.isoformat(),
        'transactions_processed': len(transactions),
        'frauds_detected': alerts_stats['total_flagged'],
        'status': 'success'
    }


if __name__ == "__main__":
    # Exécution locale pour test
    print("🚀 Démarrage du pipeline de détection de fraude ops0...\n")

    # Test avec la date d'aujourd'hui
    results = fraud_detection_pipeline()

    print("\n✅ Pipeline completed successfully!")
    print(f"\n📊 Summary:")
    print(f"   - Model accuracy: {results['model_metrics']['rf_accuracy']:.2%}")
    print(f"   - Transactions analyzed: {results['detection_stats']['total_transactions']}")
    print(f"   - Frauds detected: {results['detection_stats']['total_flagged']}")
    print(f"   - Output files: {results['output_files']['report_file']}")

    # To deploy to production:
    # ops0.deploy()