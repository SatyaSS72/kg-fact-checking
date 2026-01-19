import numpy as np
import logging

logging.getLogger("pykeen").setLevel(logging.CRITICAL)

from pykeen.triples import TriplesFactory
from pykeen.training import SLCWATrainingLoop
from src.utils import get_logger

logger = get_logger("embeddings")


def train_embedding(model_cls, triples, dim, epochs, seed):
    logger.info(f"      🎛️ Training {model_cls.__name__} embeddings")

    try:
        logger.info("          📦 Creating triples factory")
        tf = TriplesFactory.from_labeled_triples(
            np.array(triples, dtype=str),
            create_inverse_triples=False
        )

        logger.info("          🧬 Initializing embedding model")
        model = model_cls(
            triples_factory=tf,
            embedding_dim=dim,
            random_seed=seed
        )

        logger.info("          🌀 Starting training loop")
        loop = SLCWATrainingLoop(model=model, triples_factory=tf)
        loop.train(
            triples_factory=tf,
            num_epochs=epochs,
            batch_size=8192
        )

    except Exception as e:
        logger.exception(f"          💥 Embedding training failed for {model_cls.__name__}")
        raise e

    logger.info(f"          🎯 {model_cls.__name__} embedding training completed")
    return model, tf
