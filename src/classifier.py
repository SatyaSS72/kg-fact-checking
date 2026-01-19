import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from src.utils import get_logger

logger = get_logger("classifier")


def train_global_classifier(data, seed):
    logger.info("      🔬 Preparing global features")

    X_all, y_all = [], []

    logger.info("          🧮 Aggregating predicate-specific training data")
    for p, (Xp, yp) in data.items():
        X_all.extend(Xp)
        y_all.extend(yp)

    if len(X_all) == 0:
        logger.error("          ❌ No training data available for global classifier")
        raise ValueError("Global classifier received empty dataset")

    try:
        logger.info("          🧪 Validating and preparing feature matrix")
        scaler = StandardScaler()
        X_all = scaler.fit_transform(np.array(X_all))
        y_all = np.array(y_all)

        logger.info("          ⚙️  Initializing LightGBM classifier")
        clf = lgb.LGBMClassifier(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True
        )

        logger.info("          🛰️ Training global model")
        clf.fit(X_all, y_all)

    except Exception as e:
        logger.exception("          💥 Failed to train global classifier")
        raise e

    logger.info("          ✔ Global classifier training completed")
    return scaler, clf
