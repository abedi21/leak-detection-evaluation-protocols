"""Evaluation-protocol study across all three sensor modalities.

For each modality (hydrophone, accelerometer, dynamic pressure):
  task     : binary (leak vs no-leak), multiclass (5 leak states)
  featset  : all (45 features), shape_only (amplitude-invariant subset)
  model    : RF (full grid), SVM (binary/all only, for runtime)
  protocol : P1 random-window, P2 grouped-record, P3 leave-condition-out, P4 cross-topology
  seeds    : 0, 1, 2
Majority-class baseline recorded per split. Appends to protocol_all.csv (resumable).
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

OUT = 'protocol_all.csv'
SEEDS = [0, 1, 2]
SVM_MAX_TRAIN = 2000
MODALITIES = ['hydro', 'accel', 'pressure']
PROTOCOLS = ['P1 random-window', 'P2 grouped-record', 'P3 leave-condition-out', 'P4 cross-topology']

def make_model(name, seed):
    if name == 'RF':
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    return make_pipeline(StandardScaler(), SVC(C=10, gamma='scale', random_state=seed, cache_size=800))

def splits(protocol, X, y, seed, groups, cond, topo):
    if protocol == 'P1 random-window':
        yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    elif protocol == 'P2 grouped-record':
        yield from GroupKFold(n_splits=5).split(X, y, groups)
    elif protocol == 'P3 leave-condition-out':
        for c in np.unique(cond):
            yield np.where(cond != c)[0], np.where(cond == c)[0]
    else:
        for t in np.unique(topo):
            yield np.where(topo != t)[0], np.where(topo == t)[0]

rows, done = [], set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    rows = prev.to_dict('records')
    done = set(map(tuple, prev[['modality','model','task','featset','protocol','seed']].values))

for mod in MODALITIES:
    df = pd.read_csv(f'feats/{mod}_features.csv')
    ALL = [c for c in df.columns if c.startswith(('band','spec','rms','log_rms','kurt','skew','zcr','peak','crest'))]
    SHAPE = [c for c in df.columns if c.startswith(('bandrel','spec_')) or c in ('zcr','crest','kurtosis','skew')]
    groups = df['recording_id'].values
    cond = df['condition'].astype(str).values
    topo = df['topo'].values
    TASKS = {'binary': df['is_leak'].values, 'multiclass': df['leak'].values}
    for task, y in TASKS.items():
        for fsname, feats in [('all', ALL), ('shape_only', SHAPE)]:
            X = df[feats].values
            for mname in ['RF', 'SVM']:
                if mname == 'SVM' and not (task == 'binary' and fsname == 'all'):
                    continue
                for seed in SEEDS:
                    for protocol in PROTOCOLS:
                        key = (mod, mname, task, fsname, protocol, seed)
                        if key in done:
                            continue
                        for k, (tr, te) in enumerate(splits(protocol, X, y, seed, groups, cond, topo)):
                            tr_use = tr
                            if mname == 'SVM' and len(tr) > SVM_MAX_TRAIN:
                                tr_use = np.random.RandomState(seed).choice(tr, SVM_MAX_TRAIN, replace=False)
                            m = make_model(mname, seed).fit(X[tr_use], y[tr_use])
                            yp = m.predict(X[te])
                            vals, cnts = np.unique(y[tr], return_counts=True)
                            base = np.repeat(vals[cnts.argmax()], len(te))
                            avg = 'binary' if task == 'binary' else 'macro'
                            rows.append(dict(modality=mod, model=mname, task=task, featset=fsname,
                                             protocol=protocol, seed=seed, fold=k,
                                             n_test=len(te),
                                             accuracy=accuracy_score(y[te], yp),
                                             balanced_acc=balanced_accuracy_score(y[te], yp),
                                             f1=f1_score(y[te], yp, average=avg, zero_division=0),
                                             base_accuracy=accuracy_score(y[te], base),
                                             base_balanced=balanced_accuracy_score(y[te], base)))
                        pd.DataFrame(rows).to_csv(OUT, index=False)
                        print(f"done {key}", flush=True)
print("ALL DONE")
