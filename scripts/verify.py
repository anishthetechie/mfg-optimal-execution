"""Run every numerical check from the report and print a summary."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from mfg import T, Q, K, a_of, q_closed, q_bvp, dq_da, PoA_closed, I_closed, KAPPA

t = np.linspace(0, T, 4001)
q_ac = Q * np.sinh(K * (T - t)) / np.sinh(K * T)

err = max(np.max(np.abs(q_bvp(a_of(al, N)).sol(t)[0] - q_closed(t, a_of(al, N)))) / Q
          for al in [-1.5, -0.5, 0.5, 1.0, 2.0, 5.0] for N in [1, 2, 10, 100, np.inf])
print(f"[Thm 3] closed form vs solve_bvp, max rel. error: {err:.1e}")
print(f"[Thm 1] max |q_1 - q_AC|: {max(np.max(np.abs(q_closed(t, a_of(al, 1)) - q_ac)) for al in [-1.5, 0.5, 5.0]):.1e}")

print("[Thm 4] % of inventory sold by T/2 (mean-field):")
for al in [0.0, 1.0, 5.0, -1.0]:
    print(f"    alpha={al:+.1f}: {100 * (1 - q_closed(np.array([T / 2]), a_of(al, np.inf))[0] / Q):.1f}%")

print("[Thm 5] price of anarchy, N = inf:")
for al in [1.0, 2.0, 5.0]:
    print(f"    alpha={al}: {PoA_closed(al, np.inf):.3f}")

alpha = 1.0
a_inf = a_of(alpha, np.inf)
Ns = np.unique(np.logspace(1, 4, 13).astype(int))
errs = np.array([np.max(np.abs(q_closed(t, a_of(alpha, N)) - q_closed(t, a_inf))) for N in Ns])
slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
C_pred = alpha / (2 * KAPPA) * np.max(np.abs(dq_da(t, a_inf)))
C_num = 1e6 * np.max(np.abs(q_closed(t, a_of(alpha, 1e6)) - q_closed(t, a_inf)))
print(f"[Thm 6] convergence slope: {slope:.3f}")
print(f"[Thm 6] leading constant: predicted {C_pred:.7f}, measured (N=1e6) {C_num:.7f}")
