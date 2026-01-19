from src.utils import get_logger

logger = get_logger("prediction")


def predict_fact(
    f,
    p,
    models,
    scalers,
    g_scaler,
    g_classifier,
    pred_freq,
    total_pos
):
    try:
        # Predicate-specific model
        if p in models:
            logger.debug(f"      🎯 Using predicate-specific model for {p}")

            tree, lr = models[p]
            f_scaled = scalers[p].transform(f)

            pt = tree.predict_proba(f_scaled)[0, 1]
            pl = lr.predict_proba(f_scaled)[0, 1]

            score_p = 0.65 * pt + 0.35 * pl
        else:
            logger.debug(f"      ⏭️ Predicate unseen during training: {p}")
            score_p = 0.5

        # Global classifier
        logger.debug("          🌍 Applying global classifier")
        score_g = g_classifier.predict_proba(
            g_scaler.transform(f)
        )[0, 1]

        # Score fusion
        logger.debug("          🪢 Fusing predicate-specific and global scores")
        score = 0.7 * score_p + 0.3 * score_g

        # Predicate prior
        prior = pred_freq.get(p, 1) / max(total_pos, 1)
        logger.debug("          📊 Applying predicate frequency prior")

        final_score = 0.9 * score + 0.1 * prior
        return float(final_score)

    except Exception:
        logger.exception("          💥 Prediction failed, returning default score")
        return 0.5
