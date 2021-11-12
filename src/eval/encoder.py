import numpy as np
import torch

from src.eval.utils import EvalDataLoader
from src.models.utils import check_convert_tensor


def eval_encoder(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    results = []
    for batch in EvalDataLoader(X, batch_size):
        batch = check_convert_tensor(batch)
        results.append(model(batch, just_encoder=True).detach().cpu().numpy())
    return np.vstack(results)
