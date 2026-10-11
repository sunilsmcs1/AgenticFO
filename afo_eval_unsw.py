#!/usr/bin/env python3
"""
afo_eval_unsw.py - UNSW-NB15 version of the evaluation harness for the AFO paper revision.

It computes REAL numbers from YOUR trained agent and baselines on the official
NSL-KDD KDDTrain+ / KDDTest+ files. Nothing in here produces placeholder results.

Quick start
-----------
1. Put KDDTrain+.txt and KDDTest+.txt in ./data/   (the 43-column NSL-KDD files).
2. Edit the three "USER SECTION" items below:
     a) GROUPS            - the exact feature indices of Groups A/B/C in your Table 1
     b) load_q_network()  - return a function state(82,) -> Q-values(43,) for a seed
     c) (optional) make_static_model() - your own static DL baseline
3. python afo_eval.py --seeds 0 1 2 3 4 5 6 7 8 9 --out results

Assumptions that MUST match your training code (edit if they do not)
- State = [masked feature values (41) | acquisition mask (41)]  -> 82 inputs (as in the paper).
- Actions 0..40 = acquire feature i; 41 = classify Normal; 42 = classify Attack  -> 43 actions.
- Categorical columns are label-encoded (not one-hot), so the state stays 82-dimensional.
- Already-acquired features are masked out of the action set (MASK_ACQUIRED).
- An episode is cut off after MAX_STEPS actions; the forced label is the better of the two
  classify-action Q-values (FORCED_RULE). State this choice in the paper.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ----------------------------- USER SECTION -----------------------------
FEATS = ["dur","proto","service","state","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl","sload","dload","sloss","dloss",
         "sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin","tcprtt","synack","ackdat","smean","dmean","trans_depth",
         "response_body_len","ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm",
         "is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm","ct_srv_dst","is_sm_ips_ports"]
N_FEAT = len(FEATS)      # 42 features (id, attack_cat and label are not features)
# Cost groups by assumed extraction effort (EDIT if you justify a different grouping in Table 1):
#  A = header / immediate fields,  B = per-flow statistics needing counting over the flow,
#  C = cross-flow connection-history features and application-layer (HTTP/FTP) features.
_GA = ["proto","service","state","sttl","dttl","swin","dwin","stcpb","dtcpb","is_sm_ips_ports"]
_GB = ["dur","spkts","dpkts","sbytes","dbytes","rate","sload","dload","sloss","dloss","sinpkt","dinpkt","sjit","djit",
       "tcprtt","synack","ackdat","smean","dmean"]
_GC = ["trans_depth","response_body_len","ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
       "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm","ct_srv_dst"]
GROUPS = {"A": [FEATS.index(n) for n in _GA], "B": [FEATS.index(n) for n in _GB], "C": [FEATS.index(n) for n in _GC]}
assert sorted(sum(GROUPS.values(), [])) == list(range(N_FEAT)), "every feature must be in exactly one group"
COSTS  = {"A": 0.01, "B": 0.10, "C": 0.50}       # Table 1 (assumed relative costs)
MAX_STEPS = 20
MASK_ACQUIRED = True
FORCED_RULE = "best_classify_q"                  # label chosen when the 20-action cap is hit

def load_q_network(seed):
    """Loads the agent saved by train_afo.py (checkpoints_unsw/afo_seed{seed}.pt)."""
    import torch
    from train_afo_unsw import DuelingDQN
    net = DuelingDQN(); net.load_state_dict(torch.load(f"checkpoints_unsw/afo_seed{seed}.pt", map_location="cpu")); net.eval()
    def q(s):
        with torch.no_grad():
            return net(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))[0].numpy()
    return q

def make_static_model(seed):
    """Static DL baseline. Replace by YOUR baseline architecture so the comparison is consistent."""
    return MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=100, early_stopping=True, random_state=seed)
# ------------------------------------------------------------------------

CLASSES = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
GROUP_OF = {i: g for g, idx in GROUPS.items() for i in idx}
COST_VEC = np.array([COSTS[GROUP_OF[i]] for i in range(N_FEAT)])

def load_data(d="data_unsw"):
    tr = pd.read_csv(os.path.join(d, "UNSW_NB15_training-set.csv")); te = pd.read_csv(os.path.join(d, "UNSW_NB15_testing-set.csv"))
    for df in (tr, te): df.columns = [c.strip() for c in df.columns]
    for c in ["proto", "service", "state"]:
        a, b = tr[c].astype(str), te[c].astype(str)
        m = {v: i for i, v in enumerate(sorted(set(a) | set(b)))}     # categories only; no label information is used
        tr[c] = a.map(m); te[c] = b.map(m)
    Xtr = tr[FEATS].astype(float).values; Xte = te[FEATS].astype(float).values
    big = np.where(Xtr.max(0) > 1000)[0]                             # heavy-tailed columns: log1p before scaling
    Xtr[:, big] = np.log1p(np.clip(Xtr[:, big], 0, None)); Xte[:, big] = np.log1p(np.clip(Xte[:, big], 0, None))
    sc = MinMaxScaler().fit(Xtr)                                     # scaler fitted on the TRAINING set only
    Xtr = sc.transform(Xtr); Xte = np.clip(sc.transform(Xte), 0, 1)
    ytr = tr["label"].astype(int).values; yte = te["label"].astype(int).values
    cte = te["attack_cat"].astype(str).str.strip().replace({"Backdoors": "Backdoor", "nan": "Normal", "-": "Normal"}).values
    return Xtr.astype(np.float32), ytr, Xte.astype(np.float32), yte, cte

def metrics(y, p, cats=None):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out = dict(acc=accuracy_score(y, p), prec=pr, rec=rc, f1=f1, fpr=fp / max(fp + tn, 1))
    if cats is not None:
        for c in CLASSES:
            m = cats == c
            out["recall_" + c] = float((p[m] == (0 if c == "Normal" else 1)).mean()) if m.any() else float("nan")
    return out

def run_agent(q, X, y, cats):
    n = len(X); preds = np.zeros(n, int); acq = np.zeros((n, N_FEAT), bool); steps = np.zeros(n, int)
    capped = np.zeros(n, bool); t_per = np.zeros(n)
    for k in range(n):
        t0 = time.perf_counter()
        mask = np.zeros(N_FEAT, np.float32); mask[list(GROUPS["A"])] = 1       # Group A observed at t=0
        for s in range(MAX_STEPS):
            state = np.concatenate([X[k] * mask, mask])
            qv = np.array(q(state), dtype=float)
            if MASK_ACQUIRED: qv[:N_FEAT][mask > 0] = -np.inf
            a = int(np.argmax(qv)); steps[k] += 1
            if a >= N_FEAT: preds[k] = a - N_FEAT; break
            mask[a] = 1
        else:
            capped[k] = True; preds[k] = int(np.argmax(qv[N_FEAT:]))            # forced label
        t_per[k] = time.perf_counter() - t0
        acq[k] = mask > 0
    ga = np.zeros(N_FEAT, bool); ga[GROUPS["A"]] = True
    res = metrics(y, preds, cats)
    res.update(avg_feat_incl_A=acq.sum(1).mean(), avg_feat_excl_A=(acq & ~ga).sum(1).mean(),
               avg_steps=steps.mean(), cap_frac=capped.mean(),
               wtd_cost_pct=100 * (acq * COST_VEC).sum(1).mean() / COST_VEC.sum(),
               latency_ms=1000 * t_per.mean(), flows_per_s=1 / t_per.mean())
    for g in GROUPS: res["acq_" + g] = acq[:, GROUPS[g]].sum(1).mean()
    return res

def timed_static(model, X, cols, n=500):
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False); t = []
    for k in idx:
        t0 = time.perf_counter(); model.predict(X[k:k + 1][:, cols]); t.append(time.perf_counter() - t0)
    return 1000 * np.mean(t), 1 / np.mean(t)

def run_static(name, model, cols, Xtr, ytr, Xte, yte, cats):
    model.fit(Xtr[:, cols], ytr)
    if hasattr(model, 'set_params') and 'n_jobs' in model.get_params(): model.set_params(n_jobs=1)   # single-thread prediction = fair latency
    p = model.predict(Xte[:, cols]); r = metrics(yte, p, cats)
    lat, thr = timed_static(model, Xte, cols)
    r.update(avg_feat_incl_A=len(cols), wtd_cost_pct=100 * COST_VEC[cols].sum() / COST_VEC.sum(), latency_ms=lat, flows_per_s=thr)
    return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data", default="data_unsw"); ap.add_argument("--out", default="results_unsw"); ap.add_argument("--ks", type=int, nargs="+", default=[8, 11])
    ap.add_argument("--skip_agent", action="store_true"); ap.add_argument("--topk_seeds", type=int, default=2,
        help="static top-k subset baselines are run only for the first N seeds (they are slow)"); ap.add_argument("--only_groupA", action="store_true", help="only run the static Group-A-only baseline (fast)"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Xtr, ytr, Xte, yte, cte = load_data(a.data); rows = []; all_cols = list(range(N_FEAT))
    if a.only_groupA:                                   # static baseline that uses ONLY the cheap Group A features
        for s_ in a.seeds:
            rows.append(dict(model="Static Group A only", seed=s_, **run_static("gA", make_static_model(s_), list(GROUPS["A"]), Xtr, ytr, Xte, yte, cte)))
        df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "groupA_raw.csv"), index=False)
        print(df.drop(columns="model").describe().loc[["mean", "std"]].T.to_string()); return
    # feature rankings are computed ONCE on a 20k-row training subsample (fast); no test data is used
    sub = np.random.RandomState(0).choice(len(Xtr), min(20000, len(Xtr)), replace=False)
    print("ranking features (MI, RFE) once ...", flush=True)
    mi = mutual_info_classif(Xtr[sub], ytr[sub], random_state=0)
    rfe = RFE(RandomForestClassifier(50, max_depth=12, random_state=0, n_jobs=-1), n_features_to_select=8, step=3).fit(Xtr[sub], ytr[sub])
    mi_order, rfe_order = np.argsort(-mi), np.argsort(rfe.ranking_)
    def save(): pd.DataFrame(rows).to_csv(os.path.join(a.out, "raw_per_seed.csv"), index=False)
    for n_, s in enumerate(a.seeds):
        print("seed", s, flush=True)
        if not a.skip_agent:                           # the agent is evaluated FIRST so the key result is saved early
            rows.append(dict(model="AFO", seed=s, **run_agent(load_q_network(s), Xte, yte, cte))); save(); print("  AFO done", flush=True)
        rows.append(dict(model="Static DL", seed=s, **run_static("dl", make_static_model(s), all_cols, Xtr, ytr, Xte, yte, cte))); save(); print("  Static DL done", flush=True)
        rows.append(dict(model="Random Forest (all 41)", seed=s, **run_static("rf", RandomForestClassifier(100, random_state=s, n_jobs=-1), all_cols, Xtr, ytr, Xte, yte, cte))); save(); print("  RF done", flush=True)
        rows.append(dict(model="Static Group A only", seed=s, **run_static("gA", make_static_model(s), list(GROUPS["A"]), Xtr, ytr, Xte, yte, cte))); save(); print("  Group A only done", flush=True)
        if n_ < a.topk_seeds:
            for k in a.ks:                             # k matched to AFO's average feature count
                for nm, order in (("MI", mi_order), ("RFE", rfe_order)):
                    cols = list(order[:k]); rows.append(dict(model=f"Static top-{k} ({nm})", seed=s, **run_static(nm, make_static_model(s), cols, Xtr, ytr, Xte, yte, cte)))
            save(); print("  static top-k done", flush=True)
    df = pd.DataFrame(rows); num = df.select_dtypes("number").columns.drop("seed")
    g = df.groupby("model")[list(num)]; mean, std = g.mean(), g.std().fillna(0)   # std is 0 when only one seed is run
    summ = mean.round(4).astype(str) + " ± " + std.round(4).astype(str)
    summ.to_csv(os.path.join(a.out, "summary_mean_std.csv")); print(summ.T.to_string())

if __name__ == "__main__":
    main()
