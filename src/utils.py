import random
import numpy as np
import torch
import logging
import warnings

SEED = 42

def set_determinism():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def get_logger(name: str):
    return logging.getLogger(name)

def setup_logger(name="", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        #handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def silence_pykeens():
    logging.getLogger("pykeen").disabled = True
    logging.getLogger("pykeen").propagate = False

def filter_warnings():
    warnings.filterwarnings(
        "ignore",
        message="NTSerializer always uses UTF-8 encoding.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*MPS.*",
        category=UserWarning,
    )
