import numpy as np


def federated_aggregation(local_updates, beta=1.5):
    sizes = np.array([u['n_samples'] for u in local_updates], dtype=float)
    alphas = sizes / sizes.sum()
    weights = np.stack([u['weights'] for u in local_updates], axis=0)
    w_global = np.sum(alphas[:, None] * weights, axis=0)
    deltas = weights - w_global
    omega = float(np.sum(alphas * np.sum(deltas ** 2, axis=1)))
    rho = float(np.exp(-beta * omega))
    w_tilde = rho * w_global
    c_comm = float(np.sum([len(u['weights']) for u in local_updates]))
    f_fed = float((1.0 / max(c_comm, 1e-9)) * np.sum(alphas * np.array([u['lambda_k'] for u in local_updates])))
    return {
        'w_global': w_global,
        'w_tilde_global': w_tilde,
        'omega': omega,
        'rho': rho,
        'c_comm': c_comm,
        'f_fed': f_fed,
        'alphas': alphas,
    }
