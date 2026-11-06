"""Opt-in shared pool (GDPR): explicit consent + delete-my-data. Default OFF. No faces / biometrics / names ever leave the device.

v1 writes a local consent record and a redacted export under session/exports/share/ when opted in. The actual upload to a
shared training pool is a pro/cloud placeholder (tiers.PRO['cloud_storage']) — free tier keeps everything on device.
"""
import json
import os
import shutil
import time

from . import review
from .tiers import ProFeature, require


def status(session):
    return dict(opt_in=bool(session.get("share_opt_in")),
                n_fixes=len(review.list_fixes(session)),
                delete_available=True,
                upload="placeholder (pro / cloud); on-device copy only in v1")


def set_opt_in(session, on):
    """Explicit consent. on=True records consent timestamp; on=False clears the share export copy."""
    session["share_opt_in"] = bool(on)
    session["share_consent_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if on else None
    if on:
        try:
            require("cloud_storage", session.get("entitlements") or [])
            # real upload would go here for entitled users
            dest = _export(session)
            return dict(opt_in=True, exported=dest, upload="queued (pro)")
        except ProFeature:
            dest = _export(session)
            return dict(opt_in=True, exported=dest, upload="on-device only (pro placeholder; no cloud in v1)")
    # opted out: remove any local share export
    d = os.path.join(session["paths"]["exports"], "share")
    if os.path.isdir(d): shutil.rmtree(d)
    return dict(opt_in=False, exported=None)


def delete_my_data(session):
    """GDPR delete: wipe fixes, taps, personal model, share export. Session shell (video path) remains so the user can
    keep using the free app; they can re-import anytime."""
    for key in ("fixes", "taps", "models"):
        d = session["paths"][key]
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if os.path.isdir(p): shutil.rmtree(p)
            else: os.remove(p)
    d = os.path.join(session["paths"]["exports"], "share")
    if os.path.isdir(d): shutil.rmtree(d)
    session["share_opt_in"] = False
    session["share_consent_at"] = None
    session["personal_model"] = dict(path=None, last_trained=None, n_examples=0, eval=None)
    session["n_fixes"] = 0
    session["points"] = [{**p, "winner_src": None if p.get("winner_src") in ("watch", "user") and p.get("kind") else p.get("winner_src")}
                         for p in session.get("points") or []]
    return dict(deleted=True, kept=["session.json", "video reference", "pipeline caches (audio features)"])


def _export(session):
    """Redacted share payload: kind, value, struggle tags, relative t. No faces, names, or raw video."""
    d = os.path.join(session["paths"]["exports"], "share")
    os.makedirs(d, exist_ok=True)
    payload = dict(schema="vamos.share.v1", consent_at=session.get("share_consent_at"),
                   examples=[dict(kind=e["kind"], value=e["value"], t=e.get("t"),
                                  struggle_tags=e.get("struggle_tags") or [],
                                  privacy=e.get("privacy"))
                             for e in review.list_fixes(session)])
    path = os.path.join(d, "examples.json")
    json.dump(payload, open(path, "w"), indent=2)
    return path
