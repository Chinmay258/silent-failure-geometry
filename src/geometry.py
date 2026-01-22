import numpy as np

def dispersion(E):
    mu = E.mean(axis=0)
    return np.mean(np.linalg.norm(E - mu, axis=1))

def effective_rank(E):
    _, s, _ = np.linalg.svd(E, full_matrices=False)
    p = s / s.sum()
    return np.exp(-np.sum(p * np.log(p + 1e-10)))
