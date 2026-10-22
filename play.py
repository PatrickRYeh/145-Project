"""Live helper for the 30-customer simulation: tells you which assortment to show.

Usage
-----
    python play.py                                   # start of the game
    python play.py --customer 12 --inventory 3 4 2   # resume at customer 12 with stock (3, 4, 2)
    python play.py --model empirical                 # plan with raw choice frequencies

At each step, type the arriving customer's type (A/B/C), show the printed assortment,
then type what they bought (0 = nothing, 1/2/3 = product number). 'u' undoes the last
customer, 'q' quits.
"""
import argparse
from pathlib import Path

import numpy as np

import rm_model as rm

DATA = Path(__file__).resolve().parent / "large_choice_training_data.csv"


def build_policy(model, mix):
    _, counts = rm.load_counts(DATA)
    if model == "mnl":
        P = rm.mnl_model(np.array([rm.fit_mnl(counts[k])["v"] for k in range(rm.K)]))
    else:
        P = rm.empirical_model(counts)
    p = counts.sum((1, 2)) / counts.sum() if mix is None else np.asarray(mix) / np.sum(mix)
    V, pol = rm.solve_dp(P, p)
    return P, p, V, pol


def ask(prompt, valid):
    while True:
        ans = input(prompt).strip().upper()
        if ans in valid:
            return ans
        print(f"  please enter one of: {', '.join(sorted(valid))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["mnl", "empirical"], default="mnl")
    ap.add_argument("--customer", type=int, default=1, help="1-based number of the next customer")
    ap.add_argument("--inventory", type=int, nargs=3, default=list(rm.CAPACITY))
    ap.add_argument("--mix", type=float, nargs=3, default=None,
                    help="type probabilities A B C (default: frequencies in the data)")
    args = ap.parse_args()

    P, p, V, pol = build_policy(args.model, args.mix)
    print(f"Model: per-type {args.model.upper()}, type mix A/B/C = {np.round(p, 3)}, "
          f"prices {rm.PRICES.astype(int)}, capacity {rm.CAPACITY}")

    t, x, revenue = args.customer - 1, list(args.inventory), 0.0
    history = []
    while t < rm.HORIZON:
        if sum(x) == 0:
            print("\nAll inventory sold - nothing left to offer.")
            break
        print(f"\nCustomer {t + 1}/{rm.HORIZON} | inventory {tuple(x)} | revenue so far {revenue:.0f}"
              f" | expected from here {V[t][tuple(x)]:.1f}")
        typ = ask("  Type (A/B/C, u = undo, q = quit): ", {"A", "B", "C", "U", "Q"})
        if typ == "Q":
            break
        if typ == "U":
            if history:
                t, x, revenue = history.pop()
                print("  undone")
            else:
                print("  nothing to undo")
            continue
        k = rm.TYPES.index(typ)
        s = pol[t][tuple(x) + (k,)]
        probs = P[k, s]
        print(f"  SHOW {rm.LABELS[s]}   (predicted: no purchase {probs[0]:.2f}"
              + "".join(f", buy {i + 1} {probs[i + 1]:.2f}" for i in rm.SUBSETS[s]) + ")")
        out = ask("  Outcome (0 = no purchase, or product number bought): ",
                  {"0"} | {str(i + 1) for i in rm.SUBSETS[s]})
        history.append((t, list(x), revenue))
        j = int(out)
        if j > 0:
            x[j - 1] -= 1
            revenue += rm.PRICES[j - 1]
        t += 1
    print(f"\nFinal: revenue {revenue:.0f}, remaining inventory {tuple(x)}")


if __name__ == "__main__":
    main()
