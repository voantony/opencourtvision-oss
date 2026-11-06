"""Highlight ranker vs a user's favourites (tools/labeler favourites pass): AP / precision@N of the personal score, and a
small weight grid validated by K contiguous folds (choose on K-1 folds, score AP on the held-out one).

  python benchmarks/tune_highlight_favorites.py LABELS.json --sv points.json [--folds 5]

Report only: weights are not changed by this script. Great lost points stay forced into the reel by personal_reel().
"""
import argparse
import itertools
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opencourtvision.labels.points import load_point_labels  # noqa: E402
from opencourtvision.story.config import UserConfig  # noqa: E402
from opencourtvision.story.highlight import DEFAULT_WEIGHTS  # noqa: E402

ap_ = argparse.ArgumentParser()
ap_.add_argument("labels")
ap_.add_argument("--sv", required=True)
ap_.add_argument("--folds", type=int, default=5)
A = ap_.parse_args()
L = load_point_labels(A.labels)
sv = json.load(open(A.sv))["points"]
grid=dict(kappa_score=[0.5,0.25,0.0], team_bias=[1.2,1.0], loss_weight=[0.5,1.0], rally=[0.75,1.0], winner_long=[0.70,0.90], react=[0.30,0.0])
keys=list(grid); cfgs=[dict(zip(keys,v)) for v in itertools.product(*grid.values())]
def rows(c):
    w=replace(DEFAULT_WEIGHTS, **{k:v for k,v in c.items() if k!='loss_weight'})
    R=[r for r in L.highlight_rows(sv, weights=w, user=UserConfig(c['loss_weight'])) if r['is_point']]
    return R
def ap(R, sub):
    rs=sorted([r for r in R if r['idx'] in sub], key=lambda r:(-(r['personal'] if r['personal'] is not None else -1), r['idx']))
    nf=sum(r['favorite'] for r in rs)
    if nf==0: return None
    h=a=0
    for k,r in enumerate(rs,1):
        if r['favorite']: h+=1; a+=h/k
    return a/nf
def p_at(R, n=None):
    n = n or sum(r['favorite'] for r in R)
    rs=sorted(R,key=lambda r:(-(r['personal'] if r['personal'] is not None else -1), r['idx']))[:n]
    return sum(r['favorite'] for r in rs)/n
cache={i:rows(c) for i,c in enumerate(cfgs)}
allidx=[r['idx'] for r in cache[0]]
default=0
# 5 contiguous folds (by time ~ games)
K=A.folds; folds=[set(allidx[i*len(allidx)//K:(i+1)*len(allidx)//K]) for i in range(K)]
res=[]
for f in folds:
    train=set(allidx)-f
    best=max(cache, key=lambda i:(ap(cache[i],train), -sum(cfgs[i][k]!=cfgs[0][k] for k in keys)))
    res.append((ap(cache[default],f), ap(cache[best],f), cfgs[best]))
for r in res: print(r)
d=[x[0] for x in res if x[0] is not None]; t=[x[1] for x in res if x[1] is not None]
print('heldout AP default',round(sum(d)/len(d),3),'tuned',round(sum(t)/len(t),3))
full=sorted(cache, key=lambda i:-ap(cache[i],set(allidx)))
for i in full[:6]: print(cfgs[i], round(ap(cache[i],set(allidx)),3), p_at(cache[i]))
print('default', round(ap(cache[0],set(allidx)),3), p_at(cache[0]))
# single-knob changes
for k in keys:
    for v in grid[k][1:]:
        c=dict(cfgs[0]); c[k]=v; i=cfgs.index(c)
        print('single',k,v, round(ap(cache[i],set(allidx)),3), p_at(cache[i]), [round(ap(cache[i],f) or 0,3) for f in folds])
print('default folds',[round(ap(cache[0],f) or 0,3) for f in folds])
