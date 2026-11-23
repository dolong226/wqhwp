"""Ideal statevector QAOA and the angle optimizer.

The phase separator implements a (possibly rounded) Hamiltonian H, while the objective is always the ORIGINAL cost:
angles maximize E_psi[f_norm] with f = -E_w standardized.
"""
import numpy as np
import scipy.optimize as so


def mix(psi, beta, n):
    """exp(-i beta sum_q X_q) applied qubit by qubit."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(n):
        t = psi.reshape(2 ** q, 2, -1)
        a, b = t[:, 0, :], t[:, 1, :]
        psi = np.stack((c * a + s * b, s * a + c * b), axis=1).reshape(-1)
    return psi


def make_state(H, n, p):
    """x = (gamma_1..gamma_p, beta_1..beta_p) -> QAOA state with phase separator exp(-i gamma H)."""
    plus = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)

    def state(x):
        psi = plus
        for l in range(p):
            psi = mix(psi * np.exp(-1j * x[l] * H), x[p + l], n)
        return psi
    return state


def interp(x, p):
    """INTERP (Zhou et al., PRX 10, 021067, App. B): depth-p optimum -> depth-(p+1) initial point."""
    def up(v):
        v = np.concatenate([[0.0], v, [0.0]])
        return np.array([(i / p) * v[i] + ((p - i) / p) * v[i + 1] for i in range(p + 1)])
    return np.concatenate([up(x[:p]), up(x[p:])])


def optimize(H, fnorm, n, p, rs, init=None):
    """L-BFGS-B from init (if given) plus random starts: 10 at p = 1, 5 at p >= 2. Returns (angles, probabilities)."""
    state = make_state(H, n, p)

    def obj(x):
        return -float(np.dot(np.abs(state(x)) ** 2, fnorm))
    starts = [init] if init is not None else []
    for _ in range(10 if p == 1 else 5):
        starts.append(np.concatenate([rs.uniform(-1.5, 1.5, p), rs.uniform(-np.pi / 2, np.pi / 2, p)]))
    best = min((so.minimize(obj, x0, method="L-BFGS-B") for x0 in starts), key=lambda r: r.fun)
    return best.x, np.abs(state(best.x)) ** 2
