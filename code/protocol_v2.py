"""Protocol study v2: stricter grouping, directional transfer, 5 seeds.

Protocols
  P1  random-window        windows shuffled (the common, optimistic protocol)
  P2  grouped-record       no FILE appears in both train and test
  P2b grouped-experiment   no EXPERIMENTAL RUN appears in both: the same run recorded by
                           sensor 1 and sensor 2 is two files, so P2 still leaks through
                           the sibling sensor. Group = topo|leak|flow|noise.
  P3  leave-condition-out  test on an unseen background (flow/noise) condition
  P4a LO -> BR             train looped, test branched
  P4b BR -> LO             train branched, test looped
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

OUT = 'protocol_v2.csv'
SEEDS = [0, 1, 2, 3, 4]
MODALITIES = ['hydro', 'accel', 'pressure']
PROTOCOLS = ['P1 random-window', 'P2 grouped-record', 'P2b grouped-experiment',
             'P3 leave-condition-out', 'P4a LO->BR', 'P4b BR->LO']

def splits(protocol, X, y, seed, rec, expt, cond, topo):
    if protocol == 'P1 random-window':
        yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    elif protocol == 'P2 grouped-record':
        yield from GroupKFold(n_splits=5).split(X, y, rec)
    elif protocol == 'P2b grouped-experiment':
        yield from GroupKFold(n_splits=5).split(X, y, expt)
    elif protocol == 'P3 leave-condition-out':
        for c in np.unique(cond):
            yield np.where(cond != c)[0], np.where(cond == c)[0]
    elif protocol == 'P4a LO->BR':
        yield np.where(topo == 'LO')[0], np.where(topo == 'BR')[0]
    else:
        yield np.where(topo == 'BR')[0], np.where(topo == 'LO')[0]

rows, done = [], set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    rows = prev.to_dict('records')
    done = set(map(tuple, prev[['modality', 'task', 'protocol', 'seed']].values))

for mod in MODALITIES:
    df = pd.read_csv(f'feats/{mod}_features.csv')
    FEATS = [c for c in df.columns
             if c.startswith(('band', 'spec', 'rms', 'log_rms', 'kurt', 'skew', 'zcr', 'peak', 'crest'))]
    X = df[FEATS].values
    rec = df['recording_id'].values
    # experimental run = everything except which sensor recorded it
    # note: accel/pressure have no background-noise condition; the literal "NA" in that
    # column is read back as missing, so fill it before building the group key
    expt = (df[['topo', 'leak', 'flow', 'noise']].fillna('NA').astype(str)
              .agg('|'.join, axis=1).values)
    cond = df['condition'].astype(str).values
    topo = df['topo'].values
    for task, y in [('binary', df['is_leak'].values), ('multiclass', df['leak'].values)]:
        for seed in SEEDS:
            for protocol in PROTOCOLS:
                key = (mod, task, protocol, seed)
                if key in done:
                    continue
                for k, (tr, te) in enumerate(splits(protocol, X, y, seed, rec, expt, cond, topo)):
                    m = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(X[tr], y[tr])
                    yp = m.predict(X[te])
                    vals, cnts = np.unique(y[tr], return_counts=True)
                    base = np.repeat(vals[cnts.argmax()], len(te))
                    avg = 'binary' if task == 'binary' else 'macro'
                    rows.append(dict(modality=mod, task=task, protocol=protocol, seed=seed, fold=k,
                                     n_train=len(tr), n_test=len(te),
                                     accuracy=accuracy_score(y[te], yp),
                                     balanced_acc=balanced_accuracy_score(y[te], yp),
                                     f1=f1_score(y[te], yp, average=avg, zero_division=0),
                                     base_accuracy=accuracy_score(y[te], base),
                                     base_balanced=balanced_accuracy_score(y[te], base)))
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(f"done {key}", flush=True)
print("ALL DONE")
