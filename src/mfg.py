"""Closed-form Nash / mean-field equilibria of optimal execution with permanent impact.

Model: q(0)=Q, q(T)=0, cost C[q] = int(kappa q'^2 + phi q^2) dt - alpha int q mu dt.
Symmetric N-player open-loop Nash  ->  q'' + a q' - k^2 q = 0,  a = alpha(1-1/N)/(2 kappa).
"""
import numpy as np
from scipy.integrate import solve_bvp, simpson

T, Q, KAPPA, PHI = 1.0, 1.0, 0.1, 1.0
K = np.sqrt(PHI / KAPPA)


def a_of(alpha, N):
    """Effective drift coefficient; N=np.inf gives the mean-field limit."""
    frac = 1.0 if np.isinf(N) else 1.0 - 1.0 / N
    return alpha * frac / (2 * KAPPA)


def q_closed(t, a):
    w = np.sqrt(a**2 / 4 + K**2)
    return Q * np.exp(-a * t / 2) * np.sinh(w * (T - t)) / np.sinh(w * T)


def dq_closed(t, a):
    w = np.sqrt(a**2 / 4 + K**2)
    e = np.exp(-a * t / 2) / np.sinh(w * T)
    return Q * e * (-a / 2 * np.sinh(w * (T - t)) - w * np.cosh(w * (T - t)))


def dq_da(t, a):
    """Analytic sensitivity of the trajectory to a (replaces SymPy)."""
    w = np.sqrt(a**2 / 4 + K**2)
    dw = a / (4 * w)
    s = T - t
    # (T-t) coth(w (T-t)) -> 1/w as t -> T; handle the endpoint safely
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(s > 0, s / np.tanh(w * s), 1.0 / w)
    return q_closed(t, a) * (-t / 2 + dw * (term - T / np.tanh(w * T)))


def q_bvp(a, n=201):
    t = np.linspace(0, T, n)
    def f(t, y):
        return np.vstack([y[1], -a * y[1] + K**2 * y[0]])
    def bc(ya, yb):
        return np.array([ya[0] - Q, yb[0]])
    y0 = np.vstack([Q * (1 - t), -Q * np.ones_like(t)])
    sol = solve_bvp(f, bc, t, y0, tol=1e-10, max_nodes=1000000)
    assert sol.success, sol.message
    return sol


def I_cost(a, n=200001):
    t = np.linspace(0, T, n)
    return simpson(KAPPA * dq_closed(t, a)**2 + PHI * q_closed(t, a)**2, x=t)


def PoA(alpha, N):
    shift = alpha * Q**2 / 2
    return (I_cost(a_of(alpha, N)) + shift) / (I_cost(0.0) + shift)


def I_closed(a):
    """Closed-form equilibrium cost I[q_N] = kappa * omega * Q^2 * coth(omega T) (Thm 3)."""
    w = np.sqrt(a**2 / 4 + K**2)
    return KAPPA * w * Q**2 / np.tanh(w * T)


def PoA_closed(alpha, N):
    """Closed-form price of anarchy (Thm 5), valid for alpha >= 0."""
    return (I_closed(a_of(alpha, N)) + alpha * Q**2 / 2) / (I_closed(0.0) + alpha * Q**2 / 2)
