"""On-disk match session (schema vamos.session.v1). Private labels and personal models never leave this folder unless the
user opts in to share (see share.py). Nothing here is committed to git."""
import json
import os
import time
import uuid

SCHEMA = "vamos.session.v1"


def new_session(work, video, mode="match", final=None, teams=None, sport="tennis"):
    os.makedirs(work, exist_ok=True)
    for sub in ("fixes", "taps", "models", "exports", "review"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    doc = dict(schema=SCHEMA, id=str(uuid.uuid4()), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               video=os.path.abspath(video), mode=mode, sport=sport, final=final,
               teams=list(teams or ["Team A", "Team B"]), user_team="A",
               share_opt_in=False, entitlements=[],   # free core; no payments in v1
               quality=None, report=None, points=[], review=[],
               personal_model=dict(path=None, last_trained=None, n_examples=0, eval=None),
               paths=dict(work=os.path.abspath(work), fixes=os.path.join(work, "fixes"),
                          taps=os.path.join(work, "taps"), models=os.path.join(work, "models"),
                          exports=os.path.join(work, "exports"), review=os.path.join(work, "review")))
    save(doc); return doc


def path_of(doc):
    return os.path.join(doc["paths"]["work"], "session.json")


def save(doc):
    p = path_of(doc)
    tmp = p + ".tmp"
    json.dump(doc, open(tmp, "w"), indent=2, default=str)
    os.replace(tmp, p)
    return p


def load(work_or_session):
    p = work_or_session if str(work_or_session).endswith(".json") else os.path.join(work_or_session, "session.json")
    doc = json.load(open(p))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"unsupported session schema {doc.get('schema')!r}")
    return doc
