"""Can anything recover cross-topology performance, and what survives transfer?

Remedies (all evaluated under P4, both directions, 5 seeds, Random Forest):
  R0 baseline            global standardisation fitted on the training topology
  R1 per-recording z     each recording standardised by its own statistics
  R2 per-domain z        each topology standardised by its own statistics (transductive;
                         needs only unlabelled data from the target network)
  R3 rank-normalised     features replaced by within-recording ranks
Diagnostics:
  - ROC-AUC under P4 (binary): is there transferable ranking information that a
    better decision threshold could exploit, even when accuracy is at chance?
  - oracle-threshold balanced accuracy: the best attainable on the target domain
  - per-class recall under P4 (five-class): which leak types survive transfer
"""
import numpy as np, pandas as pd, warnings, os
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score, confusion_matrix

SEEDS = [0, 1, 2, 3, 4]
MODS = ['hydro', 'accel', 'pressure']
OUT_R, OUT_C = 'remedy_results.csv', 'perclass_results.csv'

def zscore(a, axis=0):
    mu, sd = a.mean(axis=axis, keepdims=True), a.std(axis=axis, keepdims=True)
    return (a - mu) / (sd + 1e-9)

def prep(X, rec, topo, mode, tr, te):
    if mode == 'R0':
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        return (X - mu) / sd
    if mode == 'R1':
        Z = np.empty_like(X)
        for r in np.unique(rec):
            i = rec == r
            Z[i] = zscore(X[i])
        return Z
    if mode == 'R2':
        Z = np.empty_like(X)
        for t in np.unique(topo):
            i = topo == t
            Z[i] = zscore(X[i])
        return Z
    if mode == 'R3':
        Z = np.empty_like(X)
        for r in np.unique(rec):
            i = rec == r
            sub = X[i]
            ranks = sub.argsort(0).argsort(0) / max(1, sub.shape[0] - 1)
            Z[i] = ranks
        return Z

rows, crows = [], []
if os.path.exists(OUT_R):
    rows = pd.read_csv(OUT_R).to_dict('records')
    crows = pd.read_csv(OUT_C).to_dict('records') if os.path.exists(OUT_C) else []
_done = {(r['modality'], r['direction'], r['mode']) for r in rows}
for mod in MODS:
    df = pd.read_csv(f'feats/{mod}_features.csv')
    FE = [c for c in df.columns if c.startswith(('band','spec','rms','log_rms','kurt','skew','zcr','peak','crest'))]
    X0 = df[FE].to_numpy(float)
    rec, topo = df['recording_id'].values, df['topo'].values
    yb, ym = df['is_leak'].values, df['leak'].values
    for direction, (a, b) in [('LO->BR', ('LO', 'BR')), ('BR->LO', ('BR', 'LO'))]:
        tr, te = np.where(topo == a)[0], np.where(topo == b)[0]
        for mode in ['R0', 'R1', 'R2', 'R3']:
            if (mod, direction, mode) in _done:
                continue
            X = prep(X0, rec, topo, mode, tr, te)
            for seed in SEEDS:
                # ---- binary ----
                clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X[tr], yb[tr])
                pr = clf.predict_proba(X[te])[:, 1]
                pred = clf.predict(X[te])
                auc = roc_auc_score(yb[te], pr)
                # oracle threshold on the target domain (upper bound, not attainable in practice)
                ths = np.quantile(pr, np.linspace(0.01, 0.99, 99))
                oracle = max(balanced_accuracy_score(yb[te], (pr >= t).astype(int)) for t in ths)
                rows.append(dict(modality=mod, direction=direction, mode=mode, seed=seed,
                                 task='binary', bal_acc=balanced_accuracy_score(yb[te], pred),
                                 roc_auc=auc, oracle_bal_acc=oracle))
                # ---- multiclass ----
                clf2 = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X[tr], ym[tr])
                p2 = clf2.predict(X[te])
                rows.append(dict(modality=mod, direction=direction, mode=mode, seed=seed,
                                 task='multiclass', bal_acc=balanced_accuracy_score(ym[te], p2),
                                 roc_auc=np.nan, oracle_bal_acc=np.nan))
                if mode == 'R0':
                    labs = ['CC', 'GL', 'LC', 'NL', 'OL']
                    rc = recall_score(ym[te], p2, labels=labs, average=None, zero_division=0)
                    for l, v in zip(labs, rc):
                        crows.append(dict(modality=mod, direction=direction, seed=seed,
                                          cls=l, recall=v))
            pd.DataFrame(rows).to_csv(OUT_R, index=False)
            pd.DataFrame(crows).to_csv(OUT_C, index=False)
            print(f"  saved {mod} {direction} {mode}", flush=True)
    pd.DataFrame(rows).to_csv(OUT_R, index=False)
    pd.DataFrame(crows).to_csv(OUT_C, index=False)
    print("done", mod, flush=True)

pd.DataFrame(rows).to_csv(OUT_R, index=False)
pd.DataFrame(crows).to_csv(OUT_C, index=False)
print("saved")
