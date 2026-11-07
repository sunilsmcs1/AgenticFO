#!/usr/bin/env python3
"""
afo4_experiments.py - extra experiments requested by the IJIES 20267001 reviewers.
It reuses the helpers of afo2.py (same data split, masked gradient-boosting classifier, Env, rollout, metrics)
and adds:

  1. Comparators that use the SAME masked classifier but no learned sequencing (isolates the value of the policy):
       - fixed cost-efficiency order, fixed k       ("Fixed order k=..")
       - fixed order + confidence stopping           ("Confidence stop tau=..")   [per-flow adaptive, non-RL]
       - greedy policy (gamma = 0)                    ("AFO greedy (gamma=0)")      [no lookahead]
  2. Ablations of AFO: terminal-accuracy reward instead of loss-reduction reward; classifier trained without
     random masking.
  3. Cost-model sensitivity: re-train/evaluate under other cost ratios and a shared-set-up cost model.
  4. Reliability of the masked classifier on the masks the policy actually visits: ECE, Brier, log-loss,
     PR-AUC; decision threshold chosen on validation vs fixed 0.5; per-category recall.
  5. Robustness: feature unavailability (dropout) and forced early stopping.
  6. Robust timing (warm-up, repeats, median/IQR, ms per decision) and a break-even extraction time.
  7. Hardware / library report.

Usage (Colab; needs afo2.py, afo_eval.py / afo_eval_unsw.py, data folders in the same directory):
    python afo4_experiments.py --dataset nsl  --seeds 0 1 2 3 4 --lam 0.1
    python afo4_experiments.py --dataset unsw --seeds 0 1 2 3 4 --lam 0.1
Use --exps to run a subset, e.g. --exps main greedy conf   (default: all).  --steps 12000 matches the paper.
Everything is written to results4_<dataset>/ . Report all numbers from ONE run; do not mix with older runs.
KDDTest+ / UNSW testing set are used only for the final report. Thresholds, tau and the lambda rule use the
validation split only.
"""
import argparse, os, time, json, importlib, platform, subprocess
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, brier_score_loss
import afo2 as B

ALL_EXPS = ["main", "greedy", "terminal", "nomask", "low_gap", "high_gap", "uniform", "shared", "conf", "fixed", "stress", "timing"]

# ------------------------------------------------------------------ cost scenarios
class Scen:
    """Per-feature cost c_i plus an optional one-off set-up cost per group (B, C) paid when the first feature of the
    group is acquired (shared collection work). Group A is observed at the start and carries no set-up cost."""
    def __init__(s, M, name):
        N = M.N_FEAT; s.name = name; s.N = N
        s.gid = np.full(N, -1); [s.gid.__setitem__(M.GROUPS[g], k) for k, g in enumerate(("B", "C"))]
        per = {"base": dict(A=.01, B=.10, C=.50), "low_gap": dict(A=.01, B=.05, C=.10), "high_gap": dict(A=.01, B=.10, C=1.0),
               "uniform": dict(A=.10, B=.10, C=.10), "shared": dict(A=.01, B=.05, C=.05)}[name]
        s.setup = np.array([.15, .45]) if name == "shared" else np.zeros(2)
        s.c = np.zeros(N)
        for g in M.GROUPS: s.c[M.GROUPS[g]] = per[g]
        s.Gm = np.zeros((N, 2), np.float32)
        for i in range(N):
            if s.gid[i] >= 0: s.Gm[i, s.gid[i]] = 1
        s.full = s.c.sum() + s.setup.sum()
    def marginal(s, mask_before, a):
        g = s.gid[a]; has = (mask_before @ s.Gm) > 0
        extra = np.where(g >= 0, s.setup[np.maximum(g, 0)] * (~has[np.arange(len(a)), np.maximum(g, 0)]), 0.0)
        return s.c[a] + extra
    def total(s, mask):                       # per-flow cost of a final mask
        return mask @ s.c + ((mask @ s.Gm) > 0) @ s.setup
    def pct(s, mask): return 100 * s.total(mask).mean() / s.full

class EnvX(B.Env):
    def __init__(s, X, y, n, rng, clf_logp, A_idx, scen, lam, reward="loss"):
        s.scen, s.reward = scen, reward; super().__init__(X, y, n, rng, clf_logp, A_idx, scen.c, lam)
    def step(s, a):
        r = np.zeros(s.n, np.float32); stop = a == s.N; done = stop.copy(); rows = np.where(~stop)[0]
        correct = (s.p1 > 0.5) == (s.y[s.idx] == 1)
        if len(rows):
            mc = s.scen.marginal(s.mask[rows], a[rows]); old = s.ell[rows].copy(); s.mask[rows, a[rows]] = 1; s._ell(rows)
            r[rows] = (old - s.ell[rows] - s.lam * mc) if s.reward == "loss" else (-s.lam * mc); s.steps[rows] += 1
        if s.reward == "terminal": r[stop] = np.where(correct[stop], 1.0, -1.0)
        done |= s.steps >= s.N + 1
        return r, done, dict(feats=s.mask.sum(1).copy(), correct=correct)

# ------------------------------------------------------------------ policy training (torch imported lazily)
def make_trainer(a, N, A, Xfit, yfit):
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    class Dueling(nn.Module):
        def __init__(s, din, na):
            super().__init__(); s.b = nn.Sequential(nn.Linear(din, B.POL_HIDDEN), nn.ReLU(), nn.Linear(B.POL_HIDDEN, B.POL_HIDDEN), nn.ReLU())
            s.v = nn.Linear(B.POL_HIDDEN, 1); s.adv = nn.Linear(B.POL_HIDDEN, na)
        def forward(s, x): z = s.b(x); ad = s.adv(z); return s.v(z) + ad - ad.mean(1, keepdim=True)
    def train(seed, lam, clf_logp, scen, gamma=1.0, reward="loss", tag=""):
        torch.manual_seed(seed); rng = np.random.RandomState(seed); din, na = 2 * N + 1, N + 1
        net, tgt = Dueling(din, na).to(dev), Dueling(din, na).to(dev); tgt.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=B.POL_LR)
        env = EnvX(Xfit, yfit, B.N_ENVS, rng, clf_logp, A, scen, lam, reward); buf = B.Replay(B.BUFFER, din, na); upd = 0
        for t in range(a.steps):
            eps = max(B.EPS_END, B.EPS_START - (B.EPS_START - B.EPS_END) * t / (B.EPS_FRAC * a.steps)); S = env.state(); v = env.valid()
            with torch.no_grad(): q = net(torch.as_tensor(S, device=dev)).cpu().numpy()
            q[~v] = -np.inf; act = q.argmax(1); rnd = rng.rand(B.N_ENVS) < eps
            for k in np.where(rnd)[0]: act[k] = rng.choice(np.where(v[k])[0])
            r, done, info = env.step(act); S2 = env.state(); buf.add(S, act, r, S2, done.astype(np.float32), env.valid()); env.reset(done)
            if buf.n >= B.WARMUP:
                bS, bA, bR, bS2, bD, bV2 = buf.sample(B.POL_BATCH, rng); bS, bS2 = torch.as_tensor(bS, device=dev), torch.as_tensor(bS2, device=dev)
                bA, bR, bD, bV2 = [torch.as_tensor(x, device=dev) for x in (bA, bR, bD, bV2)]
                with torch.no_grad():
                    an = net(bS2).masked_fill(~bV2, -1e9).argmax(1, keepdim=True); qn = tgt(bS2).gather(1, an)[:, 0]
                    y = bR + gamma * (1 - bD) * qn            # terminal indicator (1-d) and next-state action mask (bV2) as in Eq. (6)
                loss = nn.functional.smooth_l1_loss(net(bS).gather(1, bA[:, None])[:, 0], y); opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step(); upd += 1
                if upd % B.TARGET_SYNC == 0: tgt.load_state_dict(net.state_dict())
            if t % 3000 == 0: print(f"    [{tag}] seed={seed} step {t}/{a.steps} eps {eps:.2f}", flush=True)
        net.eval(); net.cpu()
        def q(S):
            with torch.no_grad(): return net(torch.as_tensor(S, dtype=torch.float32)).numpy()
        return q
    return train

# ------------------------------------------------------------------ rollouts (return final probabilities too)
def rollout(policy_q, probs, X, A, bs=8192, cap=None, unavail=None):
    """Greedy policy rollout. cap = forced early stop after `cap` acquisitions; unavail = bool (n,N) matrix of features
    that cannot actually be observed when acquired (value stays missing for the classifier)."""
    n, N = X.shape; mask = np.zeros((n, N), np.float32); mask[:, A] = 1; steps = np.zeros(n, int); active = np.ones(n, bool); pfin = np.zeros((n, 2))
    U = np.zeros((n, N), bool) if unavail is None else unavail
    for _ in range(N + 2):
        idx = np.where(active)[0]
        if len(idx) == 0: break
        for b in range(0, len(idx), bs):
            ii = idx[b:b + bs]; m = mask[ii]; me = m * (~U[ii]); p = probs(X[ii], me)
            q = np.array(policy_q(np.concatenate([X[ii] * me, m, p[:, 1:2]], 1).astype(np.float32)), dtype=float)
            q[:, :N][m > 0] = -np.inf; act = q.argmax(1); st = act == N
            if cap is not None: st |= steps[ii] >= cap
            pfin[ii[st]] = p[st]; active[ii[st]] = False; keep = ~st; mask[ii[keep], act[keep]] = 1; steps[ii[keep]] += 1
    return pfin, mask, steps

def rollout_conf(order, tau, probs, X, A):
    """Per-flow confidence stopping over a fixed acquisition order (non-RL adaptive baseline)."""
    n, N = X.shape; mask = np.zeros((n, N), np.float32); mask[:, A] = 1; steps = np.zeros(n, int); active = np.ones(n, bool); pfin = np.zeros((n, 2))
    for _ in range(len(order) + 1):
        idx = np.where(active)[0]
        if len(idx) == 0: break
        p = probs(X[idx], mask[idx]); st = (p.max(1) >= tau) | (steps[idx] >= len(order))
        pfin[idx[st]] = p[st]; active[idx[st]] = False; go = idx[~st]
        if len(go): mask[go, np.array(order)[steps[go]]] = 1; steps[go] += 1
    return pfin, mask, steps

def fixed_k(order, k, probs, X, A):
    n, N = X.shape; mask = np.zeros((n, N), np.float32); mask[:, A] = 1; mask[:, list(order[:k])] = 1
    return probs(X, mask), mask, np.full(n, k)

# ------------------------------------------------------------------ metrics
def ece(y, p, bins=10):
    e = 0.0; q = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        s = q == b
        if s.any(): e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)

def full_metrics(M, y, cats, P, mask, steps, scen, thr=0.5, extra=None):
    p1 = P[:, 1]; pred = (p1 > thr).astype(int); r = M.metrics(y, pred, cats)
    r.update(avg_feat=float(mask.sum(1).mean()), cost_pct=float(scen.pct(mask)), decisions=float(steps.mean() + 1),
             pr_auc=float(average_precision_score(y, p1)), ece=ece(y, p1), brier=float(brier_score_loss(y, p1)),
             nll=float(-np.mean(np.log(np.clip(np.where(y == 1, p1, 1 - p1), 1e-6, 1)))), thr=thr)
    if extra: r.update(extra)
    return r

def best_f1_threshold(y, p1):
    grid = np.linspace(0.05, 0.95, 91); f = [precision_recall_fscore_support(y, (p1 > t).astype(int), average="binary", zero_division=0)[2] for t in grid]
    return float(grid[int(np.argmax(f))])

def time_policy(q, probs, X, A, n=300, reps=5, warm=20):
    rs = np.random.RandomState(0); idx = rs.choice(len(X), min(n, len(X)), replace=False); lat = []; dec = []
    for k in idx[:warm]: B.run_policy(q, probs, X[k:k + 1], A)
    for _ in range(reps):
        for k in idx:
            t0 = time.perf_counter(); _, m, st = B.run_policy(q, probs, X[k:k + 1], A); lat.append(1000 * (time.perf_counter() - t0)); dec.append(st[0] + 1)
    lat = np.array(lat); return dict(lat_mean=lat.mean(), lat_median=np.median(lat), lat_q25=np.percentile(lat, 25), lat_q75=np.percentile(lat, 75),
                                     decisions=float(np.mean(dec)), ms_per_decision=float(lat.mean() / np.mean(dec)), throughput=1000 / lat.mean(), n_timings=len(lat))

def time_static(clf, cols, X, n=300, reps=5, warm=20):
    rs = np.random.RandomState(0); idx = rs.choice(len(X), min(n, len(X)), replace=False); lat = []
    for k in idx[:warm]: clf.predict(X[k:k + 1, cols])
    for _ in range(reps):
        for k in idx: t0 = time.perf_counter(); clf.predict(X[k:k + 1, cols]); lat.append(1000 * (time.perf_counter() - t0))
    return float(np.mean(lat))

def hardware():
    def sh(c):
        try: return subprocess.check_output(c, shell=True, text=True).strip()
        except Exception: return "n/a"
    info = dict(cpu=sh("lscpu | grep 'Model name' | head -1"), logical_cores=os.cpu_count(), ram=sh("free -h | head -2 | tail -1"), os=platform.platform(), python=platform.python_version())
    for lib in ("numpy", "sklearn", "torch", "pandas", "scipy"):
        try: info[lib] = importlib.import_module(lib).__version__
        except Exception: pass
    try:
        import torch; info["gpu_available"] = bool(torch.cuda.is_available()); info["torch_threads"] = torch.get_num_threads()
    except Exception: pass
    return info

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["nsl", "unsw"], required=True)
    ap.add_argument("--data", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--exps", nargs="+", default=ALL_EXPS, choices=ALL_EXPS)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999])
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8, 12, 20, 40]); ap.add_argument("--clf_reps", type=int, default=6)
    a = ap.parse_args(); M = B.get_mod(a.dataset); N = M.N_FEAT; A = np.array(M.GROUPS["A"])
    data = a.data or ("data" if a.dataset == "nsl" else "data_unsw"); out = a.out or f"results4_{a.dataset}"; os.makedirs(out, exist_ok=True)
    json.dump(hardware(), open(f"{out}/hardware.json", "w"), indent=1); json.dump(vars(a), open(f"{out}/config.json", "w"), indent=1)
    Xtr, ytr, Xte, yte, cte = M.load_data(data)
    tri, vai = train_test_split(np.arange(len(Xtr)), test_size=0.15, stratify=ytr, random_state=0)
    Xfit, yfit, Xva, yva = Xtr[tri], ytr[tri], Xtr[vai], ytr[vai]
    sub = np.random.RandomState(0).choice(len(Xfit), min(20000, len(Xfit)), replace=False)
    mi = mutual_info_classif(Xfit[sub], yfit[sub], random_state=0)          # training portion only (after validation removal)
    scen = {n: Scen(M, n) for n in ("base", "low_gap", "high_gap", "uniform", "shared")}
    trainer = make_trainer(a, N, A, Xfit, yfit) if any(e in a.exps for e in ("main", "greedy", "terminal", "nomask", "low_gap", "high_gap", "uniform", "shared", "stress", "timing")) else None
    rows, tim, stress = [], [], []; t0 = time.time()

    def order_for(sc):                                        # cost-efficiency order: MI per unit marginal cost (cheap and informative first)
        ex = [i for i in range(N) if i not in set(A)]; return sorted(ex, key=lambda i: -mi[i] / max(sc.c[i] + (sc.setup[sc.gid[i]] / 8 if sc.gid[i] >= 0 else 0), 1e-9))

    def add(method, split, seed, sc_name, P, m, st, y, c, thr=0.5, **kw):
        r = full_metrics(M, y, c, P, m, st, scen[sc_name], thr, kw.get("extra")); rows.append(dict(method=method, split=split, seed=seed, scenario=sc_name, **r))

    for seed in a.seeds:
        print(f"== seed {seed} ({time.time()-t0:.0f}s)", flush=True)
        clf = B.train_masked_gbm(Xfit, yfit, A, seed, a.clf_reps); logp = B.gbm_logp(clf); probs = lambda X, m: np.exp(logp(X, m))
        allm = lambda X: np.ones((len(X), N), np.float32); Am = lambda X: np.tile(np.isin(np.arange(N), A).astype(np.float32), (len(X), 1))
        # reference points under every cost scenario
        for sn, sc in scen.items():
            for nm, mk in (("Masked GBM, all features", allm), ("Masked GBM, Group A only", Am)):
                for split, X, y, c in (("val", Xva, yva, None), ("test", Xte, yte, cte)):
                    m = mk(X); add(nm, split, seed, sn, probs(X, m), m, np.zeros(len(X), int), y, c)
        # --- fixed order, fixed k, confidence stopping (same masked classifier, no learned policy)
        for sn in ("base", "low_gap", "high_gap", "uniform", "shared"):
            od = order_for(scen[sn])
            if "fixed" in a.exps or "conf" in a.exps:
                if "fixed" in a.exps:
                    for k in [k for k in a.ks if k <= len(od)]:
                        for split, X, y, c in (("val", Xva, yva, None), ("test", Xte, yte, cte)):
                            P, m, st = fixed_k(od, k, probs, X, A); add(f"Fixed order k={k}", split, seed, sn, P, m, st, y, c)
                if "conf" in a.exps:
                    for tau in a.taus:
                        for split, X, y, c in (("val", Xva, yva, None), ("test", Xte, yte, cte)):
                            P, m, st = rollout_conf(od, tau, probs, X, A); add(f"Confidence stop tau={tau}", split, seed, sn, P, m, st, y, c)
        # --- learned policies
        jobs = []
        if "main" in a.exps: jobs.append(("AFO (loss reward)", "base", dict(gamma=1.0, reward="loss"), clf, logp))
        if "greedy" in a.exps: jobs.append(("AFO greedy (gamma=0)", "base", dict(gamma=0.0, reward="loss"), clf, logp))
        if "terminal" in a.exps: jobs.append(("AFO terminal-accuracy reward", "base", dict(gamma=1.0, reward="terminal"), clf, logp))
        if "nomask" in a.exps:
            c0 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, early_stopping=False, random_state=seed).fit(Xfit, yfit)
            lp0 = B.gbm_logp(c0); jobs.append(("AFO, classifier without random masking", "base", dict(gamma=1.0, reward="loss"), c0, lp0))
        for sn in ("low_gap", "high_gap", "uniform", "shared"):
            if sn in a.exps: jobs.append((f"AFO (loss reward)", sn, dict(gamma=1.0, reward="loss"), clf, logp))
        pols = {}
        for name, sn, kw, cl, lp in jobs:
            q = trainer(seed, a.lam, lp, scen[sn], tag=f"{name}|{sn}", **kw); pr = lambda X, m, lp=lp: np.exp(lp(X, m))
            if name in ("AFO (loss reward)", "AFO greedy (gamma=0)") and sn == "base": pols[name] = (q, pr)
            Pv, mv, sv = rollout(q, pr, Xva, A); Pt, mt, stt = rollout(q, pr, Xte, A)
            thr_v = best_f1_threshold(yva, Pv[:, 1])
            add(name, "val", seed, sn, Pv, mv, sv, yva, None); add(name, "test", seed, sn, Pt, mt, stt, yte, cte)
            add(name + " [val-F1 threshold]", "test", seed, sn, Pt, mt, stt, yte, cte, thr=thr_v)
            pd.DataFrame(rows).to_csv(f"{out}/all_rows.csv", index=False)
            np.savez_compressed(f"{out}/test_trace_{''.join(ch if ch.isalnum() else '_' for ch in name)}_{sn}_seed{seed}.npz", pred=(Pt[:, 1] > 0.5).astype(np.int8), p1=Pt[:, 1].astype(np.float32), mask=mt.astype(np.int8), y=yte, cat=cte.astype(str))
            print(f"  {name}|{sn} seed {seed}: acc {rows[-2]['acc']:.4f} cost {rows[-2]['cost_pct']:.2f}% decisions {rows[-2]['decisions']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        # --- robustness and timing: every trained base policy (gamma=1 and greedy)
        if pols and "stress" in a.exps:
            for pname, (pq, ppr) in pols.items():
                lab = "AFO" if pname == "AFO (loss reward)" else "AFO greedy"; rs = np.random.RandomState(100 + seed)
                for dr in (0.0, 0.1, 0.3):
                    U = rs.rand(*Xte.shape) < dr; U[:, A] = False
                    P, m, st = rollout(pq, ppr, Xte, A, unavail=U); r = full_metrics(M, yte, cte, P, m, st, scen["base"]); stress.append(dict(test="feature unavailability", level=dr, method=lab, seed=seed, **r))
                    od = order_for(scen["base"]); k = int(round(np.mean(st)))                       # static comparator with the same average k and same missing features
                    mk = np.zeros_like(m); mk[:, A] = 1; mk[:, od[:k]] = 1; P2 = probs(Xte, mk * (~U)); r2 = full_metrics(M, yte, cte, P2, mk, np.full(len(Xte), k), scen["base"]); stress.append(dict(test="feature unavailability", level=dr, method=f"Fixed order k={k} (vs {lab})", seed=seed, **r2))
                for cap in (0, 1, 2, 3, 5):
                    P, m, st = rollout(pq, ppr, Xte, A, cap=cap); r = full_metrics(M, yte, cte, P, m, st, scen["base"]); stress.append(dict(test="forced early stop (max acquisitions)", level=cap, method=lab, seed=seed, **r))
            pd.DataFrame(stress).to_csv(f"{out}/stress.csv", index=False)
        if pols and "timing" in a.exps:
            for pname, (pq, ppr) in pols.items():
                d = time_policy(pq, ppr, Xte, A); d.update(seed=seed, method=pname); tim.append(d)
            for k in (0, len(order_for(scen["base"]))):
                od = order_for(scen["base"]); cols = sorted(set(A) | set(od[:k])); cs = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xfit[:, cols], yfit)
                tim.append(dict(seed=seed, method=f"Static GBM k={k}", lat_mean=time_static(cs, cols, Xte), decisions=1.0))
            pd.DataFrame(tim).to_csv(f"{out}/timing.csv", index=False)

    # ---------------------------------------------------------------- summaries
    df = pd.DataFrame(rows); df.to_csv(f"{out}/all_rows.csv", index=False); t = df[df.split == "test"]
    cols = ["acc", "prec", "rec", "f1", "fpr", "pr_auc", "ece", "brier", "avg_feat", "cost_pct", "decisions"]
    g = t.groupby(["scenario", "method"])[cols]; s = g.mean().round(4).astype(str) + " +- " + g.std().fillna(0).round(4).astype(str); s["n_seeds"] = t.groupby(["scenario", "method"]).seed.nunique()
    s.to_csv(f"{out}/summary_test.csv"); print(s.loc["base"].to_string())
    # lambda-free validation rule for the confidence tau: smallest-cost tau whose VALIDATION accuracy is within 1 pp of the all-feature masked classifier
    v = df[(df.split == "val") & (df.scenario == "base")]; fv = v[v.method == "Masked GBM, all features"].acc.mean()
    vc = v[v.method.str.startswith("Confidence")].groupby("method").agg(acc=("acc", "mean"), cost=("cost_pct", "mean"))
    if len(vc):
        ok = vc[vc.acc >= fv - B.TOL_VAL]; pick = ok.cost.idxmin() if len(ok) else vc.acc.idxmax()
        json.dump(dict(confidence_rule_pick=pick, full_val_acc=float(fv)), open(f"{out}/confidence_pick.json", "w"))
    # paired AFO vs comparators across seeds (same seed, same data)
    from scipy import stats
    af = t[(t.method == "AFO (loss reward)") & (t.scenario == "base")].set_index("seed"); pt = []
    for mname in t[t.scenario == "base"].method.unique():
        if mname == "AFO (loss reward)" or "[val-F1" in mname: continue
        o = t[(t.method == mname) & (t.scenario == "base")].set_index("seed"); cm = af.index.intersection(o.index)
        if len(cm) < 3 or not mname.startswith(("AFO", "Masked", "Fixed", "Confidence")): continue
        d = (af.loc[cm, "acc"] - o.loc[cm, "acc"]).values; ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)) if d.std() > 0 else (d.mean(), d.mean())
        pt.append(dict(comparator=mname, comp_cost_pct=o.loc[cm, "cost_pct"].mean(), afo_cost_pct=af.loc[cm, "cost_pct"].mean(), mean_acc_diff=d.mean(), ci_low=ci[0], ci_high=ci[1], p_paired_t=stats.ttest_rel(af.loc[cm, "acc"], o.loc[cm, "acc"]).pvalue, n=len(cm)))
    pd.DataFrame(pt).round(4).to_csv(f"{out}/paired_vs_comparators.csv", index=False)
    # break-even: extraction time per flow (ms, for the FULL feature set, cost assumed proportional to the simulated cost) above which
    # AFO's total time per flow (inference + extraction) is lower than that of a static classifier on all features
    if tim:
        T = pd.DataFrame(tim); be = []
        for pname in sorted({m for m in T.method if m.startswith("AFO")}):
            la = T[T.method == pname].lat_mean.mean(); ca = t[(t.method == pname) & (t.scenario == "base")].cost_pct.mean()
            for k in sorted({m for m in T.method if m.startswith("Static")}):
                ls = T[T.method == k].lat_mean.mean(); full = not k.endswith("k=0")
                be.append(dict(policy=pname, comparator=k, afo_latency_ms=la, comp_latency_ms=ls, afo_cost_pct=ca,
                               break_even_full_extraction_ms=((la - ls) / (1 - ca / 100)) if full else np.nan,
                               note="policy cheaper in total time once extraction of all features costs more than this" if full else "Group-A-only static is cheaper than the policy at any extraction cost"))
        pd.DataFrame(be).to_csv(f"{out}/break_even.csv", index=False); print(pd.DataFrame(be).to_string())
    print("done", f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
