"""Each test checks one theorem from report/report.pdf."""
import numpy as np
import pytest
from mfg import T, Q, K, a_of, q_closed, q_bvp, dq_da, I_cost, I_closed, PoA_closed, KAPPA

t = np.linspace(0, T, 2001)
tin = t[1:-1]
q_ac = Q * np.sinh(K * (T - t)) / np.sinh(K * T)


@pytest.mark.parametrize("alpha", [-1.5, 0.5, 5.0])
def test_thm1_single_trader_is_almgren_chriss(alpha):
    assert np.max(np.abs(q_closed(t, a_of(alpha, 1)) - q_ac)) < 1e-15


@pytest.mark.parametrize("alpha", [-1.5, -0.5, 1.0, 5.0])
@pytest.mark.parametrize("N", [2, 10, np.inf])
def test_thm3_closed_form_matches_bvp(alpha, N):
    a = a_of(alpha, N)
    assert np.max(np.abs(q_bvp(a).sol(t)[0] - q_closed(t, a))) < 1e-10


@pytest.mark.parametrize("alpha", [-1.0, 0.5, 2.0])
def test_thm3_closed_form_cost(alpha):
    a = a_of(alpha, np.inf)
    assert abs(I_cost(a) - I_closed(a)) < 1e-10


@pytest.mark.parametrize("alpha", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("N", [2, 10, np.inf])
def test_thm4_front_loading(alpha, N):
    assert np.all(q_closed(tin, a_of(alpha, N)) < q_ac[1:-1])


@pytest.mark.parametrize("alpha", np.linspace(-5, 10, 31))
def test_thm5_excess_cost_nonnegative(alpha):
    assert I_closed(a_of(alpha, np.inf)) >= I_closed(0.0) - 1e-14


def test_thm5_poa_at_least_one():
    for al in np.linspace(0.0, 10, 21):
        for N in [1, 2, 50, np.inf]:
            assert PoA_closed(al, N) >= 1 - 1e-14


def test_thm6_convergence_rate_and_constant():
    alpha, a_inf = 1.0, a_of(1.0, np.inf)
    q_inf = q_closed(t, a_inf)
    Ns = np.array([100, 1000, 10000])
    errs = np.array([np.max(np.abs(q_closed(t, a_of(alpha, N)) - q_inf)) for N in Ns])
    slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    assert abs(slope + 1) < 0.01
    C_pred = alpha / (2 * KAPPA) * np.max(np.abs(dq_da(t, a_inf)))
    C_num = 1e6 * np.max(np.abs(q_closed(t, a_of(alpha, 1e6)) - q_inf))
    assert abs(C_num / C_pred - 1) < 1e-5
