"""Protocol study on learned spectrogram representations (hydrophone).

Same windows, same groups and same protocols as the hand-crafted-feature study;
the only change is the input representation (32x32 log-mel spectrogram, flattened)
and the model (multilayer perceptron trained end-to-end on that input).
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

OUT = 'spec_results.csv'
SEEDS = [0, 1, 2, 3, 4]
PROTOCOLS = ['P1 random-window', 'P2 grouped-record', 'P2b grouped-experiment',
             'P3 leave-condition-out', 'P4a LO->BR', 'P4b BR->LO']

X = np.load('spec_X.npy').reshape(9072, -1)
m = pd.read_csv('spec_meta.csv')
rec = m['recording_id'].values
expt = m[['topo', 'leak', 'flow', 'noise']].fillna('NA').astype(str).agg('|'.join, axis=1).values
cond = m['condition'].astype(str).values
topo = m['topo'].values

def splits(p, y, seed):
    if p == 'P1 random-window':
        yield from StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    elif p == 'P2 grouped-record':
        yield from GroupKFold(n_splits=5).split(X, y, rec)
    elif p == 'P2b grouped-experiment':
        yield from GroupKFold(n_splits=5).split(X, y, expt)
    elif p == 'P3 leave-condition-out':
        for c in np.unique(cond):
            yield np.where(cond != c)[0], np.where(cond == c)[0]
    elif p == 'P4a LO->BR':
        yield np.where(topo == 'LO')[0], np.where(topo == 'BR')[0]
    else:
        yield np.where(topo == 'BR')[0], np.where(topo == 'LO')[0]

rows, done = [], set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT); rows = prev.to_dict('records')
    done = set(map(tuple, prev[['task', 'protocol', 'seed']].values))

for task, y in [('binary', m['is_leak'].values), ('multiclass', m['leak'].values)]:
    for seed in SEEDS:
        for p in PROTOCOLS:
            if (task, p, seed) in done:
                continue
            for k, (tr, te) in enumerate(splits(p, y, seed)):
                clf = make_pipeline(StandardScaler(),
                                    MLPClassifier(hidden_layer_sizes=(128, 32), max_iter=60,
                                                  early_stopping=True, n_iter_no_change=5,
                                                  random_state=seed)).fit(X[tr], y[tr])
                yp = clf.predict(X[te])
                vals, cnts = np.unique(y[tr], return_counts=True)
                base = np.repeat(vals[cnts.argmax()], len(te))
                avg = 'binary' if task == 'binary' else 'macro'
                rows.append(dict(model='MLP-spectrogram', task=task, protocol=p, seed=seed, fold=k,
                                 accuracy=accuracy_score(y[te], yp),
                                 balanced_acc=balanced_accuracy_score(y[te], yp),
                                 f1=f1_score(y[te], yp, average=avg, zero_division=0),
                                 base_accuracy=accuracy_score(y[te], base)))
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"done {(task, p, seed)}", flush=True)
print("ALL DONE")
