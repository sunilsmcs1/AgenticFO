#!/usr/bin/env python3
"""
afo2.py - AFO v2: a masked classifier plus a cost-aware acquisition policy (Dueling Double DQN),
with validation-based model selection and Pareto (accuracy vs. cost) evaluation.

Why v2: in v1 the same sparse +-1 reward had to teach both *what to acquire* and *how to classify*.
Here the classifier is trained by supervised learning on randomly masked inputs, and the policy only
decides which feature to acquire next or to stop. The step reward is the reduction in the classifier's
cross-entropy on the true label minus lambda * cost, so the return telescopes to
        -CE(final features) + CE(initial features) - lambda * (acquisition cost).

Usage (Colab):   python afo2.py --dataset nsl  --seeds 0 1 2 --lams 0.01 0.03 0.1 0.3 1.0
                 python afo2.py --dataset unsw --seeds 0 1 2 --lams 0.01 0.03 0.1 0.3 1.0
Needs afo_eval.py (NSL-KDD) or afo_eval_unsw.py (UNSW-NB15) in the same folder, plus the data folders.
KDDTest+ / UNSW testing set are used ONLY for the final report; lambda is chosen on a validation split
carved out of the training set.
"""
import argparse, os, time, json, importlib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------- hyperparameters (report them in the paper) ----------------
CLF_HIDDEN = 256; CLF_EPOCHS = 40; CLF_BATCH = 512; CLF_LR = 1e-3
POL_HIDDEN = 256; POL_LR = 5e-4; POL_BATCH = 256; BUFFER = 300_000; WARMUP = 10_000
TARGET_SYNC = 1_000; EPS_START = 1.0; EPS_END = 0.05; EPS_FRAC = 0.5; N_ENVS = 64; GAMMA = 1.0
TOL_VAL = 0.01          # lambda rule: largest lambda whose validation accuracy is within 1 pp of the full-feature classifier

def get_mod(ds): return importlib.import_module("afo_eval" if ds == "nsl" else "afo_eval_unsw")

# ---------------- environment (torch-free; the classifier is passed in as a numpy callable) ----------------
class Env:
    def __init__(s, X, y, n, rng, clf_logp, A_idx, cost, lam):
        s.X, s.y, s.n, s.rng, s.clf_logp, s.A, s.cost, s.lam = X, y, n, rng, clf_logp, np.asarray(A_idx), cost, lam
        s.N = X.shape[1]; s.idx = rng.randint(0, len(X), n); s.mask = np.zeros((n, s.N), np.float32); s.mask[:, s.A] = 1
        s.steps = np.zeros(n, int); s.ell = np.zeros(n); s.p1 = np.zeros(n); s._ell(np.arange(n))
    def _ell(s, rows):
        lp = s.clf_logp(s.X[s.idx[rows]], s.mask[rows]); yy = s.y[s.idx[rows]]
        s.ell[rows] = -lp[np.arange(len(rows)), yy]; s.p1[rows] = np.exp(lp[:, 1])
    def state(s): return np.concatenate([s.X[s.idx] * s.mask, s.mask, s.p1[:, None]], 1).astype(np.float32)
    def valid(s):
        v = np.ones((s.n, s.N + 1), bool); v[:, :s.N] = s.mask == 0; return v
    def step(s, a):
        r = np.zeros(s.n, np.float32); stop = a == s.N; done = stop.copy(); rows = np.where(~stop)[0]
        correct = (s.p1 > 0.5) == (s.y[s.idx] == 1)
        if len(rows):
            old = s.ell[rows].copy(); s.mask[rows, a[rows]] = 1; s._ell(rows)
            r[rows] = old - s.ell[rows] - s.lam * s.cost[a[rows]]; s.steps[rows] += 1
        done |= s.steps >= s.N + 1
        return r, done, dict(feats=s.mask.sum(1).copy(), correct=correct)
    def reset(s, done):
        rows = np.where(done)[0]
        if len(rows):
            s.idx[rows] = s.rng.randint(0, len(s.X), len(rows)); s.mask[rows] = 0; s.mask[np.ix_(rows, s.A)] = 1
            s.steps[rows] = 0; s._ell(rows)

class Replay:
    def __init__(s, cap, sd, na):
        s.s = np.zeros((cap, sd), np.float32); s.s2 = np.zeros_like(s.s); s.a = np.zeros(cap, np.int64)
        s.r = np.zeros(cap, np.float32); s.d = np.zeros(cap, np.float32); s.v2 = np.ones((cap, na), bool); s.cap, s.n, s.i = cap, 0, 0
    def add(s, S, a, r, S2, d, v2):
        k = len(a); j = (s.i + np.arange(k)) % s.cap
        s.s[j], s.a[j], s.r[j], s.s2[j], s.d[j], s.v2[j] = S, a, r, S2, d, v2
        s.i = (s.i + k) % s.cap; s.n = min(s.n + k, s.cap)
    def sample(s, b, rng):
        j = rng.randint(0, s.n, b); return s.s[j], s.a[j], s.r[j], s.s2[j], s.d[j], s.v2[j]

# ---------------- batched policy rollout (used for validation, test and the Pareto points) ----------------
def run_policy(policy_q, clf_probs, X, A_idx, bs=8192):
    """policy_q(state)->Q (k,N+1); clf_probs(X,mask)->(k,2). Returns predictions, final masks, #acquisitions."""
    n, N = X.shape; pred = np.zeros(n, int); mask = np.zeros((n, N), np.float32); mask[:, A_idx] = 1
    steps = np.zeros(n, int); active = np.ones(n, bool)
    for _ in range(N + 1):
        idx = np.where(active)[0]
        if len(idx) == 0: break
        for b in range(0, len(idx), bs):
            ii = idx[b:b + bs]; m = mask[ii]; p = clf_probs(X[ii], m)
            q = np.array(policy_q(np.concatenate([X[ii] * m, m, p[:, 1:2]], 1).astype(np.float32)), dtype=float)
            q[:, :N][m > 0] = -np.inf; a = q.argmax(1); st = a == N
            pred[ii[st]] = p[st].argmax(1); active[ii[st]] = False
            ai = ii[~st]; mask[ai, a[~st]] = 1; steps[ai] += 1
    return pred, mask, steps

def latency_loop(policy_q, clf_probs, X, A_idx, n=300):
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False); t = []
    for k in idx:
        t0 = time.perf_counter(); run_policy(policy_q, clf_probs, X[k:k + 1], A_idx); t.append(time.perf_counter() - t0)
    return 1000 * np.mean(t), 1 / np.mean(t)

def cost_pct(mask, cost): return 100 * (mask * cost).sum(1).mean() / cost.sum()

def static_family(M, Xfit, yfit, Xte, yte, cte, seed, extras_order, A_idx, ks):
    rows = []
    for k in ks:
        cols = sorted(set(A_idx) | set(extras_order[:k])); cols = list(cols)
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xfit[:, cols], yfit)
        p = clf.predict(Xte[:, cols]); r = M.metrics(yte, p, cte)
        t = []
        for i in range(0, min(300, len(Xte))):
            t0 = time.perf_counter(); clf.predict(Xte[i:i + 1, cols]); t.append(time.perf_counter() - t0)
        r.update(avg_feat=len(cols), cost_pct=100 * M.COST_VEC[cols].sum() / M.COST_VEC.sum(), latency_ms=1000 * np.mean(t))
        rows.append(dict(method="Static GBM (Group A + top-k MI)", param=f"k={k}", seed=seed, split="test", **r))
    return rows

def choose_lambda(df, full_val_acc):
    v = df[(df.method == "AFO-v2") & (df.split == "val")].groupby("param").agg(acc=("acc", "mean"), cost=("cost_pct", "mean"))
    v.index = v.index.astype(float); ok = v[v.acc >= full_val_acc - TOL_VAL]
    return float(ok.index.max()) if len(ok) else float(v.acc.idxmax())

# ---------------- masked gradient-boosting classifier (handles missing features natively) ----------------
def train_masked_gbm(Xfit, yfit, A_idx, seed, reps=6, max_iter=200):
    """HistGradientBoosting trained on `reps` randomly masked copies of the training data. Masked entries are NaN,
    which the model treats as missing values; Group A is always observed (the policy starts with it observed)."""
    rng = np.random.RandomState(seed); n, N = Xfit.shape; Xs, ys = [], []
    for _ in range(reps):
        p = rng.rand(n, 1); m = rng.rand(n, N) < p; m[:, A_idx] = True
        Xs.append(np.where(m, Xfit, np.nan).astype(np.float32)); ys.append(yfit)
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=63, early_stopping=False, random_state=seed)
    return clf.fit(np.concatenate(Xs), np.concatenate(ys))

def gbm_logp(clf):
    def f(X, m):
        Xin = np.where(np.asarray(m) > 0, X, np.nan).astype(np.float32)
        return np.log(np.clip(clf.predict_proba(Xin), 1e-6, 1.0))
    return f

# ---------------- torch parts ----------------
def main():
    import torch, torch.nn as nn
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["nsl", "unsw"], required=True)
    ap.add_argument("--data", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--steps", type=int, default=12000, help="vector steps of the policy (x64 transitions)")
    ap.add_argument("--clf_epochs", type=int, default=CLF_EPOCHS)
    ap.add_argument("--clf", choices=["gbm", "mlp"], default="gbm", help="masked classifier used inside AFO (default: gradient boosting with NaN for masked features)")
    ap.add_argument("--clf_reps", type=int, default=6, help="masked copies of the training data for the GBM classifier")
    ks_default = [0, 1, 2, 3, 5, 8, 12, 20, 40]; ap.add_argument("--ks", type=int, nargs="+", default=ks_default)
    a = ap.parse_args(); M = get_mod(a.dataset); dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = a.data or ("data" if a.dataset == "nsl" else "data_unsw"); out = a.out or f"results_v2_{a.dataset}"; os.makedirs(out, exist_ok=True)
    N = M.N_FEAT; A = np.array(M.GROUPS["A"]); COST = M.COST_VEC
    Xtr, ytr, Xte, yte, cte = M.load_data(data)
    tri, vai = train_test_split(np.arange(len(Xtr)), test_size=0.15, stratify=ytr, random_state=0)
    Xfit, yfit, Xva, yva = Xtr[tri], ytr[tri], Xtr[vai], ytr[vai]
    sub = np.random.RandomState(0).choice(len(Xfit), min(20000, len(Xfit)), replace=False)
    mi = mutual_info_classif(Xfit[sub], yfit[sub], random_state=0)
    extras = [i for i in np.argsort(-mi) if i not in set(A)]; ks = [k for k in a.ks if k <= len(extras)]
    if len(extras) not in ks: ks.append(len(extras))

    class Clf(nn.Module):
        def __init__(s):
            super().__init__(); s.f = nn.Sequential(nn.Linear(2 * N, CLF_HIDDEN), nn.ReLU(), nn.Linear(CLF_HIDDEN, CLF_HIDDEN), nn.ReLU(), nn.Linear(CLF_HIDDEN, 2))
        def forward(s, x, m): return s.f(torch.cat([x * m, m], 1))
    class Dueling(nn.Module):
        def __init__(s, din, na):
            super().__init__(); s.b = nn.Sequential(nn.Linear(din, POL_HIDDEN), nn.ReLU(), nn.Linear(POL_HIDDEN, POL_HIDDEN), nn.ReLU())
            s.v = nn.Linear(POL_HIDDEN, 1); s.adv = nn.Linear(POL_HIDDEN, na)
        def forward(s, x):
            z = s.b(x); ad = s.adv(z); return s.v(z) + ad - ad.mean(1, keepdim=True)

    A_t = torch.as_tensor(A, device=dev, dtype=torch.long)
    def rand_mask(B, g):
        p = torch.rand(B, 1, generator=g, device=dev); m = (torch.rand(B, N, generator=g, device=dev) < p).float(); m[:, A_t] = 1; return m
    def train_clf(seed):
        torch.manual_seed(seed); g = torch.Generator(device=dev); g.manual_seed(seed)
        clf = Clf().to(dev); opt = torch.optim.Adam(clf.parameters(), lr=CLF_LR)
        Xt, yt = torch.as_tensor(Xfit, device=dev), torch.as_tensor(yfit, device=dev)
        Xv, yv = torch.as_tensor(Xva, device=dev), torch.as_tensor(yva, device=dev)
        gv = torch.Generator(device=dev); gv.manual_seed(123); mv = rand_mask(len(Xv), gv); best, bs = 1e9, None
        for ep in range(a.clf_epochs):
            perm = torch.randperm(len(Xt), generator=g, device=dev)
            for b in range(0, len(Xt), CLF_BATCH):
                ii = perm[b:b + CLF_BATCH]; loss = nn.functional.cross_entropy(clf(Xt[ii], rand_mask(len(ii), g)), yt[ii]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): vl = nn.functional.cross_entropy(clf(Xv, mv), yv).item()
            if vl < best: best, bs = vl, {k: v.clone() for k, v in clf.state_dict().items()}
        clf.load_state_dict(bs); clf.eval(); return clf
    def mk_logp(clf):
        def f(X, m):
            with torch.no_grad(): return torch.log_softmax(clf(torch.as_tensor(X, device=dev, dtype=torch.float32), torch.as_tensor(m, device=dev)), 1).cpu().numpy()
        return f
    def train_policy(seed, lam, clf_logp):
        torch.manual_seed(seed); rng = np.random.RandomState(seed); din = 2 * N + 1; na = N + 1
        net, tgt = Dueling(din, na).to(dev), Dueling(din, na).to(dev); tgt.load_state_dict(net.state_dict()); opt = torch.optim.Adam(net.parameters(), lr=POL_LR)
        env = Env(Xfit, yfit, N_ENVS, rng, clf_logp, A, COST, lam); buf = Replay(BUFFER, din, na); upd = 0; log = []
        for t in range(a.steps):
            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * t / (EPS_FRAC * a.steps)); S = env.state(); v = env.valid()
            with torch.no_grad(): q = net(torch.as_tensor(S, device=dev)).cpu().numpy()
            q[~v] = -np.inf; act = q.argmax(1); rnd = rng.rand(N_ENVS) < eps
            for k in np.where(rnd)[0]: act[k] = rng.choice(np.where(v[k])[0])
            r, done, info = env.step(act); S2 = env.state(); buf.add(S, act, r, S2, done.astype(np.float32), env.valid())
            if t % 200 == 0: log.append((t, float(r.mean()), float(info["feats"].mean())))
            env.reset(done)
            if buf.n >= WARMUP:
                bS, bA, bR, bS2, bD, bV2 = buf.sample(POL_BATCH, rng); bS, bS2 = torch.as_tensor(bS, device=dev), torch.as_tensor(bS2, device=dev)
                bA, bR, bD, bV2 = [torch.as_tensor(x, device=dev) for x in (bA, bR, bD, bV2)]
                with torch.no_grad():
                    an = net(bS2).masked_fill(~bV2, -1e9).argmax(1, keepdim=True); qn = tgt(bS2).gather(1, an)[:, 0]; y = bR + GAMMA * (1 - bD) * qn
                loss = nn.functional.smooth_l1_loss(net(bS).gather(1, bA[:, None])[:, 0], y); opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step(); upd += 1
                if upd % TARGET_SYNC == 0: tgt.load_state_dict(net.state_dict())
            if t % 2000 == 0: print(f"    policy lam={lam} seed={seed} step {t}/{a.steps} eps {eps:.2f} feats {info['feats'].mean():.1f}", flush=True)
        net.eval(); net.cpu(); return net, log
    def mk_q(net):
        def f(S):
            with torch.no_grad(): return net(torch.as_tensor(S, dtype=torch.float32)).numpy()
        return f

    rows = []; full_val = []; t0 = time.time()
    for seed in a.seeds:
        print(f"== seed {seed}: classifier ({time.time()-t0:.0f}s)", flush=True)
        if a.clf == "gbm": logp = gbm_logp(train_masked_gbm(Xfit, yfit, A, seed, a.clf_reps))
        else: logp = mk_logp(train_clf(seed))
        probs = lambda X, m: np.exp(logp(X, m)); ctag = "GBM" if a.clf == "gbm" else "MLP"
        allm = lambda X: np.ones_like(X, np.float32)
        for name, mk in ((f"Masked {ctag}, all features", allm), (f"Masked {ctag}, Group A only", lambda X: np.tile(np.isin(np.arange(N), A).astype(np.float32), (len(X), 1)))):
            m = mk(Xte); p = probs(Xte, m).argmax(1); r = M.metrics(yte, p, cte); r.update(avg_feat=float(m.sum(1).mean()), cost_pct=cost_pct(m, COST))
            rows.append(dict(method=name, param="-", seed=seed, split="test", **r))
        full_val.append(float((probs(Xva, allm(Xva)).argmax(1) == yva).mean()))
        rows += static_family(M, Xfit, yfit, Xte, yte, cte, seed, extras, A, ks)
        for lam in a.lams:
            net, log = train_policy(seed, lam, logp); q = mk_q(net)
            for split, X, y, c in (("val", Xva, yva, None), ("test", Xte, yte, cte)):
                pred, mask, steps = run_policy(q, probs, X, A); r = M.metrics(y, pred, c)
                r.update(avg_feat=float(mask.sum(1).mean()), cost_pct=cost_pct(mask, COST), avg_steps=float(steps.mean() + 1))
                for g in M.GROUPS: r["acq_" + g] = float(mask[:, M.GROUPS[g]].sum(1).mean())
                if split == "test": r["latency_ms"], r["flows_per_s"] = latency_loop(q, probs, Xte, A)
                rows.append(dict(method="AFO-v2", param=str(lam), seed=seed, split=split, **r))
            pd.DataFrame(log, columns=["step", "mean_step_reward", "feats"]).to_csv(f"{out}/policy_log_lam{lam}_seed{seed}.csv", index=False)
            print(f"  lam {lam} seed {seed}: test acc {rows[-1]['acc']:.4f} feats {rows[-1]['avg_feat']:.2f} cost% {rows[-1]['cost_pct']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(f"{out}/pareto_points.csv", index=False)
    df = pd.DataFrame(rows); full_val_acc = float(np.mean(full_val)); lam_star = choose_lambda(df, full_val_acc)
    json.dump(dict(selected_lambda=lam_star, full_feature_val_acc=full_val_acc, tol=TOL_VAL), open(f"{out}/selected_lambda.json", "w"))
    t = df[df.split == "test"]; cols = ["acc", "prec", "rec", "f1", "fpr", "avg_feat", "cost_pct", "latency_ms"]
    g = t.groupby(["method", "param"])[cols]; summ = g.mean().round(4).astype(str) + " +- " + g.std().fillna(0).round(4).astype(str)
    summ.to_csv(f"{out}/summary_test.csv"); print(summ.to_string()); print("selected lambda (validation rule):", lam_star)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        for k, (yc, yl) in enumerate((("acc", "Test accuracy"), ("fpr", "Test false-positive rate"))):
            for meth, st in (("Static GBM (Group A + top-k MI)", "s--"), ("AFO-v2", "o-")):
                d = t[t.method == meth].groupby("param").agg(x=("cost_pct", "mean"), y=(yc, "mean"), e=(yc, "std")).sort_values("x")
                ax[k].errorbar(d.x, d.y, yerr=d.e.fillna(0), fmt=st, label=meth, capsize=2, ms=4)
            fm = t[t.method.str.endswith("all features")]; ax[k].scatter([100], [fm[yc].mean()], marker="*", s=90, c="k", label="Masked classifier, all features")
            ax[k].set_xscale("log"); ax[k].set_xlabel("Weighted acquisition cost (% of full)"); ax[k].set_ylabel(yl); ax[k].grid(alpha=.3)
        ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{out}/pareto.png", dpi=200)
    except Exception as e: print("plot skipped:", e)

if __name__ == "__main__":
    main()
