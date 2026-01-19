import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.utils import get_logger

logger = get_logger("models")


def train_models(data, seed):
    logger.info("          🧱 Structuring predicate-specific matrices")

    models, scalers = {}, {}

    for p, (Xp, yp) in data.items():
        if len(set(yp)) < 2:
            logger.debug(f"              ⏭️ Skipping predicate {p} (single class)")
            continue

        try:
            logger.info(f"              ▶ Training models for predicate: {p}")

            logger.info("                 📐 Scaling feature matrix")
            scaler = StandardScaler()
            Xp_scaled = scaler.fit_transform(np.array(Xp))
            yp = np.array(yp)

            logger.info("                 ⚙️  Initializing classifiers")
            tree = lgb.LGBMClassifier(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                num_leaves=63,
                min_child_samples=30,
                scale_pos_weight=3,
                random_state=seed,
                deterministic=True,
                force_col_wise=True,
                n_jobs=1,
                verbosity=-1
            )

            lr = LogisticRegression(
                C=0.5,
                max_iter=1000,
                random_state=seed
            )

            logger.info("                 🪄 Fitting models")
            tree.fit(Xp_scaled, yp)
            lr.fit(Xp_scaled, yp)

            models[p] = (tree, lr)
            scalers[p] = scaler

            logger.info("                 ✔ Predicate model trained")

        except Exception:
            logger.exception(f"                 💥 Failed to train model for predicate {p}")

    logger.info(f"            🎯 Trained models for {len(models)} predicates")
    return models, scalers
