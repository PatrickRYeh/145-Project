"""Personalized online assortment model for the IEOR 145 group simulation.

Setting
-------
* 3 products with prices r = (10, 15, 20) and starting inventory c = (6, 6, 6).
* T = 30 customers arrive one at a time. Each customer's type (A, B, C) is revealed on
  arrival, we pick an assortment from the products still in stock, and the customer
  buys at most one product (or nothing).
* Choice behaviour of each type is learned from large_choice_training_data.csv.

Conventions
-----------
* Products are 0-based in code (0, 1, 2) and 1-based in the data / output ({1}, {2}, {3}).
* Outcome index j: 0 = no purchase, j = i + 1 means product i was bought.
* A choice model is an array P[k, s, j]: probability that a type-k customer shown
  SUBSETS[s] picks outcome j. The DP and all policies work with any such array, so the
  same code runs on the MNL fit and on the model-free (empirical) estimate.
* Inventory states are arrays indexed by x = (x1, x2, x3), 0 <= x_i <= c_i.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import chi2

PRICES = np.array([10.0, 15.0, 20.0])
CAPACITY = (6, 6, 6)
HORIZON = 30
TYPES = ("A", "B", "C")
N_PROD = 3
K = len(TYPES)

# All 8 subsets of {0, 1, 2}; index 0 is the empty assortment.
SUBSETS = [()] + [s for n in (1, 2, 3) for s in itertools.combinations(range(N_PROD), n)]
N_SUB = len(SUBSETS)
MASK = np.array([[i in s for i in range(N_PROD)] for s in SUBSETS])  # (8, 3)
# OUT_OK[s, j]: outcome j is possible when SUBSETS[s] is shown.
OUT_OK = np.hstack([np.ones((N_SUB, 1), bool), MASK])


def label(s):
    """Subset in data notation, e.g. (0, 2) -> '{1,3}'."""
    return "{" + ",".join(str(i + 1) for i in s) + "}"


LABELS = [label(s) for s in SUBSETS]


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def load_counts(path="large_choice_training_data.csv"):
    """Return (df, counts) with counts[k, s, j] = # of type-k customers shown s who chose j."""
    df = pd.read_csv(path)
    idx = {lab: n for n, lab in enumerate(LABELS)}
    unknown = set(df.assortment) - set(idx)
    if unknown:
        raise ValueError(f"unrecognised assortments: {unknown}")
    counts = np.zeros((K, N_SUB, N_PROD + 1), dtype=int)
    for (t, a, r), n in df.groupby(["customer_type", "assortment", "result"]).size().items():
        counts[TYPES.index(t), idx[a], r] += n
    if (counts[:, ~OUT_OK] > 0).any():
        raise ValueError("data contains purchases of products that were not offered")
    return df, counts


def empirical_model(counts):
    """Model-free estimate: observed choice frequencies for every (type, assortment)."""
    P = np.zeros((K, N_SUB, N_PROD + 1))
    n = counts.sum(axis=2, keepdims=True)
    P[:] = np.divide(counts, n, out=np.zeros_like(P), where=n > 0)
    P[:, 0] = [1, 0, 0, 0]  # empty assortment: nobody buys
    return P


# ----------------------------------------------------------------------------------------
# MNL estimation (maximum likelihood with v0 = 1, as in Lecture 14 / MNL_main.ipynb)
# ----------------------------------------------------------------------------------------
def mnl_probs(v):
    """(8, 4) choice probabilities of an MNL with weights v (v0 = 1) for every subset."""
    w = np.hstack([np.ones((N_SUB, 1)), MASK * v])
    return w / w.sum(axis=1, keepdims=True)


def mnl_model(V):
    """Stack per-type MNL weights V (K, 3) into a choice-model array P[k, s, j]."""
    return np.stack([mnl_probs(v) for v in V])


def _mnl_negll(beta, cnt):
    P = mnl_probs(np.exp(beta))
    ll = np.sum(cnt[OUT_OK] * np.log(P[OUT_OK]))
    # gradient wrt beta_i: sum_s [n(s, i) - N_s * phi(i, s)]
    grad = cnt[:, 1:].sum(0) - (cnt.sum(1, keepdims=True) * P[:, 1:]).sum(0)
    return -ll, -grad


def _mnl_fisher(beta, cnt):
    """Negative Hessian of the log-likelihood (it does not depend on the choices made)."""
    P = mnl_probs(np.exp(beta))
    N = cnt.sum(1)
    info = np.zeros((N_PROD, N_PROD))
    for s in range(N_SUB):
        q = P[s, 1:]
        info += N[s] * (np.diag(q) - np.outer(q, q))
    return info


def fit_mnl(cnt):
    """MLE of one MNL from counts cnt[s, j] (log-likelihood is concave; Newton's method).

    Returns dict with weights v, log-weights beta, their standard errors (inverse Fisher
    information) and the maximised log-likelihood."""
    beta = np.zeros(N_PROD)
    for _ in range(100):
        negll, neg_grad = _mnl_negll(beta, cnt)
        step = np.linalg.solve(_mnl_fisher(beta, cnt), -neg_grad)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    negll, neg_grad = _mnl_negll(beta, cnt)
    se = np.sqrt(np.diag(np.linalg.inv(_mnl_fisher(beta, cnt))))
    return {"v": np.exp(beta), "beta": beta, "se_beta": se, "loglik": -negll,
            "converged": bool(np.abs(neg_grad).max() < 1e-4 * cnt.sum())}


def saturated_loglik(cnt):
    """Log-likelihood of the unrestricted model (one free distribution per assortment)."""
    N = cnt.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(cnt > 0, cnt * np.log(cnt / N), 0.0)
    return terms.sum()


def lr_test(ll_restricted, ll_full, df):
    stat = 2 * (ll_full - ll_restricted)
    return {"LR": stat, "df": df, "p_value": chi2.sf(stat, df)}


def n_free_params_saturated(cnt):
    """Each observed assortment s contributes |s| free probabilities."""
    return int(sum(len(SUBSETS[s]) for s in range(N_SUB) if cnt[s].sum() > 0))


# ----------------------------------------------------------------------------------------
# Single-customer assortment problems
# ----------------------------------------------------------------------------------------
def feasible_subsets(x):
    """Boolean (8,) mask of subsets that only use in-stock products at inventory x."""
    return np.all(~MASK | (np.asarray(x) >= 1), axis=1)


def mnl_opt_assortment_lp(v, prices):
    """Class LP for unconstrained MNL revenue maximisation (Lecture 13).

    max sum_i r_i v_i y_i  s.t.  0 <= y_i <= theta,  theta + sum_i v_i y_i = 1.
    Returns (assortment as tuple of 0-based products, optimal value)."""
    n = len(v)
    c = -np.concatenate([np.asarray(prices) * v, [0.0]])
    A_ub = np.hstack([np.eye(n), -np.ones((n, 1))])
    A_eq = np.concatenate([v, [1.0]])[None, :]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (n + 1), method="highs")
    y, theta = res.x[:n], res.x[n]
    return tuple(int(i) for i in np.flatnonzero(y > theta / 2)), -res.fun


# ----------------------------------------------------------------------------------------
# Dynamic program (exact optimum for i.i.d. type arrivals with known probabilities)
# ----------------------------------------------------------------------------------------
def _grid(cap):
    return tuple(c + 1 for c in cap)


def _feas_grid(cap):
    """FEAS[s, x...] = SUBSETS[s] is feasible at inventory x."""
    xs = np.indices(_grid(cap))  # (3, ...)
    return np.stack([np.all([xs[i] >= 1 for i in s], axis=0) if s else np.ones(_grid(cap), bool)
                     for s in SUBSETS])


def _minus_one(V, i):
    """V evaluated at x - e_i (entries with x_i = 0 are left as V[x]; they are never used)."""
    out = V.copy()
    hi = [slice(None)] * V.ndim
    lo = [slice(None)] * V.ndim
    hi[i], lo[i] = slice(1, None), slice(None, -1)
    out[tuple(hi)] = V[tuple(lo)]
    return out


def _outcome_values(Vnext, prices):
    """W[j, x] = revenue + continuation value if outcome j happens at inventory x."""
    return np.stack([Vnext] + [prices[i] + _minus_one(Vnext, i) for i in range(N_PROD)])


def q_values(P, Vnext, prices, feas):
    """Q[k, s, x] = expected (immediate + future) revenue of showing s to type k at x."""
    Q = np.einsum("ksj,j...->ks...", P, _outcome_values(Vnext, prices))
    return np.where(feas[None], Q, -np.inf)


def solve_dp(P, p_type, prices=PRICES, cap=CAPACITY, T=HORIZON):
    """Backward induction for i.i.d. arrivals with type probabilities p_type.

    V_t(x) = sum_k p_k max_{S feasible at x} [ sum_j phi_k(j, S) (r_j + V_{t+1}(x - e_j)) ]
    Returns V with shape (T+1, *grid) and policy with shape (T, *grid, K) (subset indices).
    p_type may also be a (T, K) array of per-period type distributions."""
    p_type = np.broadcast_to(np.asarray(p_type, float), (T, K))
    feas = _feas_grid(cap)
    V = np.zeros((T + 1,) + _grid(cap))
    pol = np.zeros((T,) + _grid(cap) + (K,), dtype=np.int8)
    for t in range(T - 1, -1, -1):
        Q = q_values(P, V[t + 1], prices, feas)
        pol[t] = np.moveaxis(Q.argmax(axis=1), 0, -1)
        V[t] = np.tensordot(p_type[t], Q.max(axis=1), axes=1)
    return V, pol


def opportunity_costs(Vnext):
    """Delta_i(x) = V(x) - V(x - e_i): the value of keeping one more unit of product i."""
    return np.stack([Vnext - _minus_one(Vnext, i) for i in range(N_PROD)])


# ----------------------------------------------------------------------------------------
# Heuristic policies (Markov: decision depends only on t, x and the arriving type)
# All return policy tables of shape (T, *grid, K) so they can be evaluated exactly.
# ----------------------------------------------------------------------------------------
def _static_table(score, cap, T):
    """score: (K, 8, *grid) -> best feasible subset, identical for every period."""
    feas = _feas_grid(cap)
    score = np.where(feas[None], score, -np.inf)
    best = np.moveaxis(score.argmax(axis=1), 0, -1).astype(np.int8)
    return np.broadcast_to(best, (T,) + best.shape).copy()


def adjusted_price_table(P, adj_prices, cap=CAPACITY, T=HORIZON):
    """Show the assortment maximising sum_i phi_k(i, S) * adj_i(x).

    adj_prices: (3,) constant or (3, *grid) inventory-dependent adjusted prices."""
    adj = np.asarray(adj_prices, float)
    if adj.ndim == 1:
        adj = adj.reshape((N_PROD,) + (1,) * len(cap)) * np.ones(_grid(cap))
    score = np.einsum("ksi,i...->ks...", P[:, :, 1:], adj)
    return _static_table(score, cap, T)


def policy_show_all(cap=CAPACITY, T=HORIZON):
    """Naive baseline: show every product that is still in stock."""
    xs = np.indices(_grid(cap))
    in_stock = tuple((xs[i] >= 1) for i in range(N_PROD))
    idx = {s: n for n, s in enumerate(SUBSETS)}
    tab = np.zeros(_grid(cap), dtype=np.int8)
    for x in itertools.product(*[range(c + 1) for c in cap]):
        tab[x] = idx[tuple(i for i in range(N_PROD) if in_stock[i][x])]
    return np.broadcast_to(tab[..., None], (T,) + tab.shape + (K,)).copy()


def policy_greedy(P, prices=PRICES, cap=CAPACITY, T=HORIZON):
    """Myopic greedy (Lecture 14B): revenue-optimal assortment of available products."""
    return adjusted_price_table(P, prices, cap, T)


def ib_discount(u):
    """Inventory-balancing discount Psi(u) = (e / (e - 1)) (1 - e^{-u}), u = x_i / c_i.

    From Golrezaei, Nazerzadeh & Rusmevichientong (2014), the paper cited in Lecture 14B."""
    return np.e / (np.e - 1) * (1 - np.exp(-u))


def policy_inventory_balancing(P, prices=PRICES, cap=CAPACITY, T=HORIZON):
    """Greedy with prices discounted by the fraction of inventory remaining."""
    xs = np.indices(_grid(cap)).astype(float)
    adj = np.stack([prices[i] * ib_discount(xs[i] / cap[i]) for i in range(N_PROD)])
    return adjusted_price_table(P, adj, cap, T)


def fluid_lp(P, n_type, prices=PRICES, cap=CAPACITY):
    """Choice-based fluid LP.

    max  sum_k n_k sum_S h[k,S] R_k(S)
    s.t. sum_k n_k sum_S h[k,S] phi_k(i,S) <= c_i   for each product i
         sum_S h[k,S] = 1, h >= 0                    for each type k
    n_type = expected number of arrivals of each type. Returns (value, h, bid_prices)."""
    n_type = np.asarray(n_type, float)
    R = P[:, :, 1:] @ prices  # (K, 8)
    c = -(n_type[:, None] * R).ravel()
    A_ub = np.stack([(n_type[:, None] * P[:, :, 1 + i]).ravel() for i in range(N_PROD)])
    A_eq = np.kron(np.eye(K), np.ones(N_SUB))
    res = linprog(c, A_ub=A_ub, b_ub=np.asarray(cap, float), A_eq=A_eq, b_eq=np.ones(K),
                  bounds=[(0, None)] * (K * N_SUB), method="highs")
    bid = -res.ineqlin.marginals  # duals of the capacity constraints (>= 0)
    return -res.fun, res.x.reshape(K, N_SUB), bid


def policy_bid_price(P, bid, prices=PRICES, cap=CAPACITY, T=HORIZON):
    """Static bid-price control: maximise sum_i phi_k(i, S) (r_i - pi_i) using fluid-LP duals."""
    return adjusted_price_table(P, prices - bid, cap, T)


def policy_type_blind(P, p_type, prices=PRICES, cap=CAPACITY, T=HORIZON):
    """DP that ignores the revealed type: one assortment for the type mixture."""
    P_mix = np.tensordot(np.asarray(p_type, float), P, axes=1)[None]
    _, pol = solve_dp(np.repeat(P_mix, K, axis=0), np.ones(K) / K, prices, cap, T)
    return pol


# ----------------------------------------------------------------------------------------
# Exact evaluation of Markov policies
# ----------------------------------------------------------------------------------------
def evaluate_policy(pol, P_true, arrivals, prices=PRICES, cap=CAPACITY, T=HORIZON):
    """Exact expected revenue of policy table pol under true choice model P_true.

    arrivals: (K,) i.i.d. type probabilities, or (T, K) per-period distributions
    (one-hot rows = a known, fixed arrival order). Returns E with E[0][cap] = value."""
    arrivals = np.broadcast_to(np.asarray(arrivals, float), (T, K))
    feas = _feas_grid(cap)
    E = np.zeros((T + 1,) + _grid(cap))
    kk = np.arange(K).reshape((K,) + (1,) * len(cap))
    for t in range(T - 1, -1, -1):
        s = np.moveaxis(pol[t], -1, 0)  # (K, *grid)
        if not np.take_along_axis(np.broadcast_to(feas, (K,) + feas.shape),
                                  s[:, None].astype(int), axis=1).all():
            raise ValueError(f"policy offers an out-of-stock product at t={t}")
        Pt = P_true[kk, s]  # (K, *grid, 4)
        W = _outcome_values(E[t + 1], prices)  # (4, *grid)
        val = np.einsum("k...j,j...->k...", Pt, W)
        E[t] = np.tensordot(arrivals[t], val, axes=1)
    return E


def start_value(E, cap=CAPACITY):
    return E[0][tuple(cap)]


# ----------------------------------------------------------------------------------------
# Adaptive DP: re-estimate type probabilities from the arrivals seen so far
# ----------------------------------------------------------------------------------------
class AdaptiveDP:
    """Certainty-equivalent re-solving with a Dirichlet prior on the type mix.

    Prior Dirichlet(m * p0). After seeing counts n (including the current customer), the
    posterior-mean mix is (m p0 + n) / (m + sum n) and the decision at time t is the DP
    decision against continuation values V_{t+1} solved under that mix."""

    def __init__(self, P, p0, m, prices=PRICES, cap=CAPACITY, T=HORIZON):
        self.P, self.p0, self.m = P, np.asarray(p0, float), m
        self.prices, self.cap, self.T = prices, cap, T
        self.feas = _feas_grid(cap)
        self._cache = {}

    def mix(self, n):
        return (self.m * self.p0 + np.asarray(n)) / (self.m + sum(n))

    def table(self, t, n):
        """Decision table (*grid, K) at time t given type counts n seen through time t."""
        key = (t, tuple(n))
        if key not in self._cache:
            remaining = self.T - t - 1
            V, _ = solve_dp(self.P, self.mix(n), self.prices, self.cap, remaining)
            Q = q_values(self.P, V[0], self.prices, self.feas)
            self._cache[key] = np.moveaxis(Q.argmax(axis=1), 0, -1).astype(np.int8)
        return self._cache[key]

    def table_for_order(self, order):
        """Policy table (T, *grid, K) for a known arrival order (for exact evaluation)."""
        n = np.zeros(K, int)
        out = np.zeros((self.T,) + _grid(self.cap) + (K,), np.int8)
        for t, k in enumerate(order):
            n[k] += 1
            out[t] = self.table(t, n)
        return out


# ----------------------------------------------------------------------------------------
# Monte Carlo simulation (common random numbers across policies)
# ----------------------------------------------------------------------------------------
def draw_arrivals(rng, n_runs, scenario, T=HORIZON):
    """scenario: ('iid', p) or ('fixed_counts', counts) or ('order', sequence)."""
    kind, arg = scenario
    if kind == "iid":
        return rng.choice(K, size=(n_runs, T), p=np.asarray(arg, float) / np.sum(arg))
    if kind == "fixed_counts":
        base = np.repeat(np.arange(K), arg)
        assert len(base) == T
        return rng.permuted(np.tile(base, (n_runs, 1)), axis=1)
    if kind == "order":
        return np.tile(np.asarray(arg), (n_runs, 1))
    raise ValueError(kind)


def simulate(policy, P_true, types, U, prices=PRICES, cap=CAPACITY):
    """Vectorised simulation. policy is a (T, *grid, K) table or an AdaptiveDP.

    types: (n_runs, T) arriving types; U: (n_runs, T) uniforms driving the choices.
    Returns revenue per run (n_runs,) and units sold per product (n_runs, 3)."""
    n_runs, T = types.shape
    x = np.tile(np.asarray(cap), (n_runs, 1))
    seen = np.zeros((n_runs, K), int)
    revenue = np.zeros(n_runs)
    sold = np.zeros((n_runs, N_PROD), int)
    rows = np.arange(n_runs)
    for t in range(T):
        k = types[:, t]
        seen[rows, k] += 1
        if isinstance(policy, AdaptiveDP):
            s = np.empty(n_runs, int)
            keys, inv = np.unique(seen, axis=0, return_inverse=True)
            for g, n in enumerate(keys):
                idx = np.flatnonzero(inv.ravel() == g)
                tab = policy.table(t, n)
                s[idx] = tab[x[idx, 0], x[idx, 1], x[idx, 2], k[idx]]
        else:
            s = policy[t][x[:, 0], x[:, 1], x[:, 2], k]
        probs = P_true[k, s]  # (n_runs, 4)
        j = (U[:, t:t + 1] > np.cumsum(probs, axis=1)[:, :-1]).sum(axis=1)
        bought = j > 0
        i = j[bought] - 1
        assert (x[rows[bought], i] > 0).all()
        x[rows[bought], i] -= 1
        sold[rows[bought], i] += 1
        revenue[bought] += prices[i]
    return revenue, sold


# ----------------------------------------------------------------------------------------
# Policy summaries and export
# ----------------------------------------------------------------------------------------
def state_occupancy(pol, P_true, arrivals, cap=CAPACITY, T=HORIZON):
    """occ[t][x] = probability that customer t (0-based) arrives to inventory x."""
    arrivals = np.broadcast_to(np.asarray(arrivals, float), (T, K))
    occ = np.zeros((T + 1,) + _grid(cap))
    occ[0][tuple(cap)] = 1.0
    kk = np.arange(K).reshape((K,) + (1,) * len(cap))
    for t in range(T):
        Pt = P_true[kk, np.moveaxis(pol[t], -1, 0)]  # (K, *grid, 4)
        mass = np.tensordot(arrivals[t], Pt * occ[t][None, ..., None], axes=1)  # (*grid, 4)
        nxt = mass[..., 0].copy()
        for i in range(N_PROD):
            hi = [slice(None)] * len(cap)
            lo = [slice(None)] * len(cap)
            hi[i], lo[i] = slice(1, None), slice(None, -1)
            nxt[tuple(lo)] += mass[..., i + 1][tuple(hi)]
        occ[t + 1] = nxt
    return occ


def policy_table_frame(pol, V, cap=CAPACITY, T=HORIZON):
    """Lookup table: for customer number t (1-based), inventory x and type, what to show."""
    rows = []
    for t in range(T):
        for x in itertools.product(*[range(c + 1) for c in cap]):
            for k in range(K):
                rows.append((t + 1, *x, TYPES[k], LABELS[pol[t][x + (k,)]], V[t][x]))
    return pd.DataFrame(rows, columns=["customer_number", "inv_product1", "inv_product2",
                                       "inv_product3", "customer_type", "show",
                                       "expected_revenue_from_here"])
