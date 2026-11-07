"""Regenerate figures/mfg_results.png."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mfg import T, KAPPA, a_of, q_closed, dq_da, PoA_closed

t = np.linspace(0, T, 1001)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for al, c in [(-1.0, "#888"), (0.0, "k"), (1.0, "#1f77b4"), (2.0, "#2a9d8f"), (5.0, "#e76f51")]:
    ax[0].plot(t, q_closed(t, a_of(al, np.inf)), color=c,
               label="Almgren–Chriss (α=0)" if al == 0 else f"MFG α={al}")
ax[0].set(xlabel="t", ylabel="inventory q(t)", title="Equilibrium liquidation paths")
ax[0].legend(frameon=False)

alpha, a_inf = 1.0, a_of(1.0, np.inf)
Ns = np.unique(np.logspace(1, 4, 13).astype(int))
errs = np.array([np.max(np.abs(q_closed(t, a_of(alpha, N)) - q_closed(t, a_inf))) for N in Ns])
s = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
C = alpha / (2 * KAPPA) * np.max(np.abs(dq_da(t, a_inf)))
ax[1].loglog(Ns, errs, "o", label=f"measured (slope {s:.3f})")
ax[1].loglog(Ns, C / Ns, "--", label=f"predicted C/N, C={C:.5f}")
ax[1].set(xlabel="N", ylabel="‖q_N − q_MF‖∞", title="Convergence to mean-field limit (α=1)")
ax[1].legend(frameon=False)

als = np.linspace(0.05, 5, 40)
Nl = np.unique(np.logspace(0, 3, 30).astype(int))
P = np.array([[PoA_closed(al, N) for al in als] for N in Nl])
im = ax[2].pcolormesh(als, Nl, P, shading="auto", cmap="viridis")
ax[2].set_yscale("log")
ax[2].set(xlabel="permanent impact α", ylabel="N", title="Price of anarchy PoA(α, N)")
fig.colorbar(im, ax=ax[2])
plt.tight_layout()
plt.savefig(ROOT / "figures" / "mfg_results.png", dpi=150)
print("saved figures/mfg_results.png")
