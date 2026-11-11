#!/usr/bin/env python3
"""
sefa_baseline.py - matched-protocol comparison with SEFA (Norcliffe et al., "Stochastic Encodings for Active
Feature Acquisition", ICML 2025; code: https://github.com/a-norcliffe/SEFA) for the IJIES 20267001 review (R2.1).

WHAT THIS DOES (and does not do)
  * Same data, same 15% stratified validation split (random_state=0), same Group A start, same test partition and the
    same cost scenarios and metrics as afo4_experiments.py (it imports Scen and full_metrics from it).
  * SEFA is trained on the same training portion (inputs standardised with training mean/std) and, for every test flow,
    acquires features one by one starting from Group A, with SEFA's own acquisition rule, for k = 0 ... max_k steps.
  * SEFA has NO cost model and NO stopping rule. It is therefore evaluated as a fixed-budget method: accuracy after k
    acquisitions, with the cost of the features it actually chose computed afterwards under the same cost scenarios.
    This is an ADAPTATION of the published method (it is not cost-aware), and the paper must say so.
  * SEFA hyper-parameters are those of its example notebook for continuous inputs; they are NOT tuned. All features are
    treated as continuous (categorical codes are numeric after the loader's preprocessing).

USAGE (Colab, A100):
    git clone https://github.com/a-norcliffe/SEFA          # inside the AgenticFO folder, next to afo2.py
    pip install torchmetrics                                 # SEFA's metrics (torch is already installed on Colab)
    python sefa_baseline.py --dataset nsl  --seeds 0 1 2 3 4 --out r_nsl_sefa
    python sefa_baseline.py --dataset unsw --seeds 0 1 2 3 4 --out r_unsw_sefa
Self-test of the non-torch logic (no GPU, no data needed):  python sefa_baseline.py --selftest
"""
import argparse, os, sys, json, time, tempfile
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import afo2 as B
import afo4_experiments as X

SCENS = ("base", "low_gap", "high_gap", "uniform", "shared")


def run_trajectories(predict_fn, acquire_fn, Xs, A_mask, max_k, chunk):
    """Pure-numpy interface. predict_fn(x, mask) -> (n,2) probabilities; acquire_fn(x, mask) -> new mask (one more
    feature per row). Returns P[k] (n,2) and Mk[k] (n,N) uint8 for k = 0..max_k."""
    n, N = Xs.shape
    P = np.zeros((max_k + 1, n, 2), np.float32); Mk = np.zeros((max_k + 1, n, N), np.uint8)
    for s in range(0, n, chunk):
        x = Xs[s:s + chunk]; m = np.tile(A_mask, (len(x), 1)).astype(np.float32)
        for k in range(max_k + 1):
            P[k, s:s + chunk] = predict_fn(x, m); Mk[k, s:s + chunk] = m.astype(np.uint8)
            if k < max_k: m = acquire_fn(x, m)
    return P, Mk


def score_rows(M, scen, y, cats, P, Mk, seed, split, rows, name="SEFA"):
    for k in range(P.shape[0]):
        mask = Mk[k].astype(np.float32); st = np.full(len(y), k)
        for sn, sc in scen.items():
            r = X.full_metrics(M, y, cats, P[k], mask, st, sc)
            rows.append(dict(method=f"{name} k={k}", split=split, seed=seed, scenario=sn, **r))


def selftest():
    """Check shapes, mask monotonicity, cost accounting and metric plumbing with a stand-in model (no torch, no data)."""
    import types
    rs = np.random.RandomState(0); N = 12; n = 500
    M = types.SimpleNamespace(N_FEAT=N, GROUPS={"A": [0, 1, 2], "B": list(range(3, 8)), "C": list(range(8, 12))})
    M.metrics = lambda y, pred, cats: dict(acc=float((y == pred).mean()), fpr=float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)))
    scen = {s: X.Scen(M, s) for s in SCENS}
    Xs = rs.randn(n, N).astype(np.float32); y = (Xs[:, 5] + Xs[:, 9] > 0).astype(int)
    A_mask = np.isin(np.arange(N), M.GROUPS["A"]).astype(np.float32)

    def predict_fn(x, m):                         # logistic on observed features only
        z = (x * m) @ np.r_[np.zeros(5), 1., np.zeros(3), 1., np.zeros(2)]; p = 1 / (1 + np.exp(-3 * z)); return np.stack([1 - p, p], 1)

    def acquire_fn(x, m):                         # picks the first unobserved feature of a random order per row
        m = m.copy(); order = np.array([5, 9, 3, 4, 6, 7, 8, 10, 11])
        for i in range(len(x)):
            for f in order:
                if m[i, f] == 0: m[i, f] = 1; break
        return m

    P, Mk = run_trajectories(predict_fn, acquire_fn, Xs, A_mask, 9, 128)
    assert P.shape == (10, n, 2) and Mk.shape == (10, n, N)
    assert all((Mk[k + 1] >= Mk[k]).all() for k in range(9)), "masks must only grow"
    assert (Mk[0][:, :3] == 1).all() and (Mk[0][:, 3:] == 0).all(), "start must be Group A only"
    assert (Mk.sum(2)[1:] - Mk.sum(2)[:-1] == 1).all(), "exactly one feature per step"
    rows = []; score_rows(M, scen, y, None, P, Mk, 0, "test", rows)
    df = pd.DataFrame(rows); b = df[(df.scenario == "base")].sort_values("method")
    assert len(df) == 10 * len(SCENS)
    c = df[df.scenario == "base"].set_index("method").cost_pct
    assert abs(c["SEFA k=9"] - 100.0) < 1e-6, "all features acquired -> 100% of the cost"
    assert abs(c["SEFA k=2"] - 100 * scen["base"].c[[0, 1, 2, 5, 9]].sum() / scen["base"].full) < 1e-6, "cost of A + features 5, 9"
    assert abs(df[(df.scenario == "shared") & (df.method == "SEFA k=9")].cost_pct.iloc[0] - 100.0) < 1e-6
    a = df[df.scenario == "base"].set_index("method").acc
    assert a["SEFA k=2"] > a["SEFA k=0"], "informative features should help"
    s = df[df.scenario == "shared"].set_index("method").cost_pct; assert s["SEFA k=9"] > 0
    print("selftest OK:", {k: round(float(v), 1) for k, v in c.items() if k in ("SEFA k=0", "SEFA k=2", "SEFA k=9")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nsl", "unsw"]); ap.add_argument("--data", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--sefa_dir", default="SEFA")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--max_k", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=1024, help="rows per acquisition batch (lower it if the GPU runs out of memory)")
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    assert a.dataset, "--dataset is required"
    import torch
    from torch.utils.data import TensorDataset
    sys.path.insert(0, a.sefa_dir)
    from models.sefa import SEFA
    from experiments.metrics_dict import metrics_dict
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = B.get_mod(a.dataset); N = M.N_FEAT; A = np.array(M.GROUPS["A"]); A_mask = np.isin(np.arange(N), A).astype(np.float32)
    data = a.data or ("data" if a.dataset == "nsl" else "data_unsw"); out = a.out or f"results_sefa_{a.dataset}"; os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f"{out}/config.json", "w"), indent=1)
    Xtr, ytr, Xte, yte, cte = M.load_data(data)
    tri, vai = train_test_split(np.arange(len(Xtr)), test_size=0.15, stratify=ytr, random_state=0)
    Xfit, yfit, Xva, yva = Xtr[tri], ytr[tri], Xtr[vai], ytr[vai]
    mu = Xfit.mean(0); sd = Xfit.std(0); sd[sd < 1e-8] = 1.0                 # training statistics only
    Z = lambda Xa: ((Xa - mu) / sd).astype(np.float32)
    scen = {n: X.Scen(M, n) for n in SCENS}
    cfg = dict(num_con_features=N, num_cat_features=0, most_categories=0, out_dim=2, latent_dim=8, num_hidden_predictor=2,
               hidden_dim_predictor=200, num_hidden_encoder=2, hidden_dim_encoder=50, num_samples_train=100, num_samples_predict=200,
               num_samples_acquire=200, beta=5e-4, epochs=a.epochs, lr=1e-3, batchsize=128, patience=5)
    rows = []; t0 = time.time()
    for seed in a.seeds:
        print(f"== SEFA seed {seed} ({time.time()-t0:.0f}s)", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        tr = TensorDataset(torch.tensor(Z(Xfit)), torch.tensor(yfit).long(), torch.ones(len(Xfit), N))
        va = TensorDataset(torch.tensor(Z(Xva)), torch.tensor(yva).long(), torch.ones(len(Xva), N))
        model = SEFA(cfg).to(dev)
        with tempfile.TemporaryDirectory() as tmp: model.fit(train_data=tr, val_data=va, save_path=tmp, metric_f=metrics_dict["auroc"])
        model.eval()

        def predict_fn(x, m):
            with torch.no_grad():
                return model.predict(torch.as_tensor(x, device=dev), torch.as_tensor(m, device=dev)).cpu().numpy()

        def acquire_fn(x, m):
            xt = torch.as_tensor(x, device=dev); mt = torch.as_tensor(m, device=dev); md = torch.ones_like(mt)
            return model.acquire(x=xt, mask_acq=mt, mask_data=md).cpu().numpy()
        for split, Xs, ys, cs in (("val", Xva, yva, None), ("test", Xte, yte, cte)):
            P, Mk = run_trajectories(predict_fn, acquire_fn, Z(Xs), A_mask, a.max_k, a.chunk)
            score_rows(M, scen, ys, cs, P, Mk, seed, split, rows)
            pd.DataFrame(rows).to_csv(f"{out}/all_rows.csv", index=False)
            print(f"  seed {seed} {split}: k=0 acc {np.mean(P[0].argmax(1) == ys):.4f}  k={a.max_k} acc {np.mean(P[-1].argmax(1) == ys):.4f}", flush=True)
    df = pd.DataFrame(rows); t = df[df.split == "test"]
    t.groupby(["scenario", "method"])[["acc", "cost_pct", "fpr"]].agg(["mean", "std"]).to_csv(f"{out}/summary_test.csv")
    print(t[t.scenario == "base"].groupby("method")[["acc", "cost_pct"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
