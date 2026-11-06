"""Ball-follow 9:16 reel from a moments config (JSON). See examples/reel_moments.example.json.

Per moment: detect + track the ball -> crop path -> sendcmd-driven crop on the source -> 1080x1920 with
0.5x slow-mo ramps at contact times and captions. Captions come only from a verified caption trigger
(`"caption_trigger"` per moment, rules in opencourtvision.captions) evaluated on the config's "events" file; a moment
without a passing trigger gets no caption, and a bare "caption" string is ignored. Then concat (video copy) + audio polish (hit-transient
emphasis, loudness) -> final mp4. Also writes ball_stats.json and ball_debug_sheet.png in the work dir.
"""
import json
import os
import pickle
import subprocess
import sys

import cv2
import numpy as np

from ..captions import moment_caption
from ..captions import ok as verified_ok
from ..events import load as load_events
from ..ffmpeg import default_font, ffmpeg_bin, fontfile_opt, is_macos
from ..render.cuts import point_end
from . import ball, camera


def _esc(s):
    return str(s).replace("'", "\u2019")  # a straight quote would end the quoted drawtext value


CAPTION_ON_S, CAPTION_MIN_S, CAPTION_MAX_S = 2.0, 1.0, 2.5     # spec §6: on screen 1.0-2.5 s, at or after the event
OUTCOME_TYPES = frozenset({"bounce", "net", "fence", "ball_dead"})   # events that make a point's outcome visible


def outcome_visible(events, after, within=3.0):
    """Time of the first outcome event (decisive bounce, net, fence) after `after`, within `within` s, else None."""
    ts = [e.t for e in (events or []) if e.type in OUTCOME_TYPES and after < e.t <= after + within]
    return min(ts) if ts else None


def resolve_moments(cfg, rallies, events=None):
    def hits_in(a, b):
        return [t for r in rallies for t in r["times"] if a <= t <= b]

    def snap(t):  # cut on a hit: start 0.12 s before the first hit at/after t
        h = hits_in(t, t + 3)
        return (h[0] - 0.12) if h else t

    out = []
    plays = [i for i, m in enumerate(cfg["moments"]) if not m.get("end_card")]
    for i, m in enumerate(cfg["moments"]):
        a = snap(m["start"]) if m.get("snap") and rallies else m["start"]
        inside = hits_in(a, m["end"]) if rallies and not m.get("end_card") else []
        # reel beats end shortly after the outcome is visible; the reel's last play clip gets the full point post-roll
        kind = "point" if plays and i == plays[-1] else "reel_beat"
        b = max(m["end"], point_end(inside[-1], outcome_visible(events, inside[-1]), clip_type=kind)) if inside else m["end"]
        cap, cap_t, why = moment_caption(m.get("caption_trigger"), events, a, b, text=m.get("caption"),
                                         aspect=cfg.get("aspect", "9:16"), sport=cfg.get("sport", "tennis"))
        out.append(dict(name=m["name"], a=a, b=b, ramps=m.get("ramps", []), caption=cap or "", caption_t=cap_t, caption_why=why,
                        punch=bool(m.get("punch_in", m.get("punch", False))), title=m.get("title", False),
                        end_card=m.get("end_card", False), end_kind=kind if inside else None, mode=m.get("mode", "auto"),
                        hits=inside))
    return out


def out_time(rel, parts):
    """Source time relative to the moment start -> output time, given the (p0, p1, speed) parts of the speed ramp."""
    o = 0.0
    for p0, p1, sp in parts:
        if rel <= p1: return o + max(0.0, rel - p0) / sp
        o += (p1 - p0) / sp
    return o


def text_filters(m, parts, title="", end=None, fo="", esc=None):
    """drawtext/drawbox filters for one moment: title card, caption (only if the moment carries a verified caption) and
    end card. The caption appears at the triggering event (never before) for CAPTION_ON_S, clamped to 1.0-2.5 s."""
    esc = esc or _esc; end = end or {}
    dur = out_time(parts[-1][1], parts) if parts else 0.0

    def dt_(s, size, y, en, col="white"):
        return (f"drawtext={fo}text='{esc(s)}':fontsize={size}:fontcolor={col}:borderw=5:bordercolor=black@0.7:"
                f"box=1:boxcolor=black@0.45:boxborderw=22:x=(w-tw)/2:y={y}:enable='{en}'")

    txt = []
    if m.get("title") and title: txt.append(dt_(title, 92, 330, "lt(t,1.6)"))
    cap = m.get("caption")
    if cap and m.get("caption_t") is not None:
        t0 = out_time(max(0.0, m["caption_t"] - m["a"]), parts)
        if m.get("title") and title: t0 = max(t0, 1.7)
        t1 = min(t0 + CAPTION_ON_S, t0 + CAPTION_MAX_S, dur)
        if t1 - t0 >= CAPTION_MIN_S: txt.append(dt_(cap, 70, 330, f"between(t,{t0:.2f},{t1:.2f})"))
    if m.get("end_card"):
        txt = ["drawbox=x=0:y=0:w=iw:h=ih:color=black@0.45:t=fill:enable='gte(t,0.25)'"]
        for text, size, col, yy in ((end.get("top", ""), 96, "white", 780), (end.get("middle", "def."), 64, "0xF2C14E", 905),
                                    (end.get("bottom", ""), 96, "white", 1000), (end.get("score", ""), 72, "white@0.9", 1150)):
            if text:
                txt.append(f"drawtext={fo}text='{esc(text)}':fontsize={size}:fontcolor={col}:x=(w-tw)/2:y={yy}:enable='gte(t,0.25)'")
    return txt


def verified_contacts(events, a, b):
    """Times of verified, non-reference contacts in [a, b]: the only ones that may raise the pan cap or steer the plan."""
    return sorted(e.t for e in events if e.type == "contact" and a <= e.t <= b and verified_ok(e))


def load_player_boxes(dirpath, name, a, n, motion_boxes, fps=60):
    """v4.6.1: per-frame player boxes from an external detector (Apple Vision export, players_<name>.pkl = dict(t0, fps,
    boxes=[[(x0, x1, area, y0, y1, side), ...] per frame] in analysis px, side 0 near / 1 far)). Frames are matched by
    time; frames the export does not cover keep the motion-blob boxes. -> (boxes, dict(source, coverage))."""
    if not dirpath: return motion_boxes, dict(source="motion", coverage=0.0)
    p = os.path.join(dirpath, f"players_{name}.pkl")
    if not os.path.exists(p): return motion_boxes, dict(source="motion", coverage=0.0, missing=p)
    with open(p, "rb") as f: d = pickle.load(f)
    vb, t0, vf = d["boxes"], float(d["t0"]), float(d.get("fps", fps))
    out, hit = [], 0
    for i in range(n):
        j = int(round((a + i / fps - t0) * vf))
        if 0 <= j < len(vb) and vb[j]:
            out.append([tuple(b) for b in vb[j]]); hit += 1
        else:
            out.append(motion_boxes[i] if i < len(motion_boxes) else [])
    return out, dict(source=d.get("source", "external"), coverage=round(hit / max(1, n), 3), file=os.path.basename(p))


def build(cfg, work, out=None, tracknet_dir=None):
    src = os.path.abspath(cfg["source"])
    out = os.path.abspath(out or cfg["output"])
    os.makedirs(work, exist_ok=True)
    rallies = json.load(open(cfg["rallies"])) if cfg.get("rallies") else []
    events = load_events(cfg["events"])[0] if cfg.get("events") else []
    font = cfg.get("font") or default_font()
    FF = ffmpeg_bin()
    SW, SH = cfg.get("source_size", [3840, 2160])
    AW, AH = 1280, 720
    S = SW / AW
    cam = dict(camera.CAMERA, **cfg.get("camera", {}))
    FPS = cfg.get("fps", 60)
    SLOW = cfg.get("slow", 0.5)
    PRE, POST = 0.35, 0.45
    hw = cfg.get("hwaccel", ball.default_hwaccel())
    vcodec = cfg.get("vcodec", "h264_videotoolbox" if is_macos() else "libx264")
    tn = ball.TrackNet(tracknet_dir) if tracknet_dir else None
    title, end = cfg.get("title", ""), cfg.get("end_card", {})

    stats = {"frames": 0, "ball_frames": 0, "ball_in_crop": 0,
             "fallback_rule": f"player framing when the ball is lost > {ball.BALL_LOST_FRAMES} frames or confidence < {ball.MIN_BALL_CONF}"}
    dbg, segs, jerk_samples, jerk_samples_r, n_wide, h_all = [], [], [], [], 0, []
    wide_flags = []
    clip_checks = {}
    for k, m in enumerate(resolve_moments(cfg, rallies, events)):
        name, a, b, ramps, punch = m["name"], m["a"], m["b"], m["ramps"], m["punch"]
        dc = os.path.join(cfg["det_cache"], f"det_{name}_{a:.2f}_{b:.2f}.pkl") if cfg.get("det_cache") else None
        if dc and os.path.exists(dc):                                 # detection cache: camera re-checks without a decode
            with open(dc, "rb") as f: hsv_pos, tn_pos, act, fd, boxes = pickle.load(f)
        else:
            cands, act, fd, small, boxes = ball.detect(ball.decode(src, a, b, AW, AH, FPS, hw), a, FPS, [a + (b - a) * 0.5],
                                                       keep_small=tn is not None, with_boxes=True)
            hsv_pos = ball.track(cands, AW, AH)
            if len(hsv_pos) == 0:
                raise RuntimeError(f"moment {name}: no frames decoded")
            tn_pos = tn.run(small) if tn else np.full((len(hsv_pos), 2), np.nan)
            small = None
            if dc:
                os.makedirs(cfg["det_cache"], exist_ok=True)
                with open(dc, "wb") as f: pickle.dump((hsv_pos, tn_pos, act, fd, boxes), f)
        n = len(hsv_pos)
        boxes, box_src = load_player_boxes(cfg.get("player_boxes"), name, a, n, boxes, FPS)
        pos = np.where(np.isnan(tn_pos), hsv_pos, tn_pos)  # fuse: TrackNet first, HSV fills gaps
        hd, td, fdz = ~np.isnan(hsv_pos[:, 0]), ~np.isnan(tn_pos[:, 0]), ~np.isnan(pos[:, 0])
        both = hd & td
        agree = float(np.mean(np.hypot(*(hsv_pos[both] - tn_pos[both]).T) < 25)) if both.any() else None
        stats.setdefault("det", {})[name] = dict(hsv=round(float(hd.mean()), 3), tracknet=round(float(td.mean()), 3), fused=round(float(fdz.mean()), 3), agree_within_25px=agree)
        for key, arr in (("hsv", hd), ("tracknet", td), ("fused", fdz)):
            stats["n_" + key] = stats.get("n_" + key, 0) + int(arr.sum())
        conf = ball.frame_confidence(hsv_pos, tn_pos)
        info = {}
        _, bx = ball.path(pos, act, SW / 2, S, boxes=boxes, conf=conf, info=info)   # we use info["target"] + bx
        fbm = info["fallback"]
        if m.get("end_card"): fbm = np.zeros_like(fbm)               # v4.5: an end card is a steady default crop, no player-framing widen
        # zoom: target (default 75 % height, wide on player framing / spread, tight on verified big shots) -> rate cap
        sp = camera.spread(boxes, S)
        big = [e.t for e in events if e.type == "contact" and a <= e.t <= b and verified_ok(e) and e.label is not None
               and (str(e.label.outcome) == "win" or str(e.label.shot_type) == "smash")]
        top, bot = camera.vertical_box(boxes, S, n)
        need = camera.need_height(top, bot, FPS, SH, cam) if not np.isnan(top).all() else None
        # offline pan plan: verified contacts -> eased contact-to-contact target, read ahead; widen before panning fast
        ct = verified_contacts(events, a, b)
        tgt = info["target"]; okb = ~np.isnan(bx); fr = np.arange(n)
        bxi = np.interp(fr, fr[okb], bx[okb]) if okb.any() else tgt
        ctx = [float(bxi[min(n - 1, max(0, int(round((t - a) * FPS))))]) for t in ct]
        cw_def = int(camera.crop_size(cam["h_default"], SH)[0])
        pp = camera.plan_pan(tgt, FPS, a, ct, ctx, cw_def, SH, cam)
        zr = camera.zoom_rates(n, FPS, a, big, punch, cam)            # 25 %/s only on a punch_in moment's verified big shot
        punch_frames = zr > cam["zoom_rate"]                         # ... and only there is the pan cap lifted
        gp = None
        if ct and any(boxes):
            # v4.3: one-shot gate planner per clip (contact gates, LP under the pan caps, widen early) replaces limit_1d
            sd = {e.t: (str(e.player).split("_")[0] if e.player else None) for e in events if e.type == "contact"}
            srv = m.get("serve", ct[0] - a >= cam.get("preserve_min_s", 1.0))    # v4.6: clip opens on a serve (override per moment)
            kinds = ["serve" if (i == 0 and srv) else None for i in range(len(ct))]
            if not srv: kinds[0] = "rally"
            # the verified side picks the server (the deeper player of that side); later hitters stay nearest-to-ball,
            # since a side filter on merged motion blobs can force a box wider than the crop (P79 4147.5, MD set)
            sides = [(sd.get(t) if sd.get(t) in ("near", "far") else None) if (i == 0 and srv) else None for i, t in enumerate(ct)]
            def _clipped(plan_):                                     # v4.7: hitters + receivers clipped as rendered
                x_ = np.clip(np.round(plan_["cx"] - plan_["cw"] / 2), 0, SW - plan_["cw"]).astype(int)
                y_ = camera.vertical_follow(top, bot, plan_["ch"], FPS, SH, cam, punch_frames) if not np.isnan(top).all() else np.zeros(n, int)
                q_ = camera.role_clip_qa(plan_["roles"], x_, y_, plan_["cw"], plan_["ch"], S, SW, FPS, a, cam, False)
                return q_["contacts_hitter_clipped"] + q_["receptions_receiver_clipped"]
            gp = camera.plan_clip_capped(boxes, pos, bx, ct, a, FPS, SW, SH, cam, S, roles_qa=_clipped, fallback=fbm,
                                         need_v=need, big_shots=big, punch_in=punch, sides=sides, kinds=kinds,
                                         bounces=[e.t for e in events if e.type == "bounce" and a <= e.t <= b])
            hfrac, cw, ch, cx = gp["h"], gp["cw"], gp["ch"], gp["cx"]
            pp["fast"] = gp["fast"]
        else:                                                        # no verified contact: 303246d follower
            need = pp["need_h"] if need is None else np.maximum(need, pp["need_h"])
            hfrac = camera.zoom_caps(camera.zoom_target(n, FPS, a, fbm, sp, big, cam, SH, need, punch), FPS, cam, zr)
            cw, ch = camera.crop_size(hfrac, SH)
            cx = np.clip(camera.follow_pan(np.clip(pp["planned"], cw / 2, SW - cw / 2), cw, cw_def, FPS, pp["fast"], punch_frames, cam),
                         cw / 2, SW - cw / 2)                  # float centre path
        x = np.clip(np.round(cx - cw / 2), 0, SW - cw).astype(int)
        if gp is not None and gp.get("planner") == "locked_wide":
            y = np.zeros(n, int)                                       # the whole source frame
        else:
            y = camera.vertical_follow(top, bot, ch, FPS, SH, cam, punch_frames)
        mode = m.get("mode", "auto")
        wide, wide_why = (mode == "wide"), "forced by config"
        if mode == "auto": wide, wide_why = camera.wide_trigger(sp, m.get("hits", []), FPS, SH, cam)
        if wide and n_wide >= cam["max_wide"]: wide, wide_why = False, f"wide limit reached ({cam['max_wide']} per reel)"
        reject = None
        if gp is not None and not wide and not m.get("end_card"):    # framing at risk: auto-wide, else flag for rejection
            rq0 = camera.role_clip_qa(gp["roles"], x, y, cw, ch, S, SW, FPS, a, cam, False)
            act, why = camera.infeasible_decision(gp, n_wide, cam, roles_qa=rq0)
            if act == "wide": wide, wide_why = True, why
            elif act == "reject": reject = why; stats.setdefault("reject_flagged", {})[name] = why
        n_wide += int(wide)
        det = ~np.isnan(bx)
        inn = det & (bx >= x + 20) & (bx <= x + cw - 20)
        stats["frames"] += n
        stats["ball_frames"] += int(det.sum())
        stats["ball_in_crop"] += int(inn.sum())
        cxs = x + cw / 2                                       # crop centre actually rendered (whole pixels): debug only
        jk, jk_r = ball.jerk_p95(cx, SW), ball.jerk_p95(cxs, SW)   # metric = float path before rounding
        if gp is not None and gp.get("cut_frames"):                # v4.9: per continuous shot (a hard cut is not a pan)
            jk, jk_r = camera.jerk_p95_cuts(cx, SW, gp["cut_frames"]), camera.jerk_p95_cuts(cxs, SW, gp["cut_frames"])
        if jk: jerk_samples.append(np.abs(np.diff(cx, 3)))
        if jk_r: jerk_samples_r.append(np.abs(np.diff(cxs.astype(float), 3)))
        clipped = (y > top + 1) | (y + ch < bot - 1) if not np.isnan(top).all() else np.zeros(n, bool)
        stats["players_clipped_frames"] = stats.get("players_clipped_frames", 0) + int(clipped.sum())
        ps = camera.partner_stats(boxes, x, cw, S, wide, SW)        # MD v4 stats: far partner / any player out, sliced
        for kk in ("far_out", "any_out", "sliced"): stats[kk + "_frames"] = stats.get(kk + "_frames", 0) + int(ps[kk].sum())
        stats["wide_frames"] = stats.get("wide_frames", 0) + (n if wide else 0)
        h_shown = np.ones(n) if wide else hfrac                 # wide mode shows the whole frame
        h_all.append(h_shown); wide_flags.append(bool(wide))
        hq = camera.crop_height_qa(h_shown, cam, cfg.get("format"), cfg.get("sport", "tennis"))
        stats[name] = dict(frames=n, ball_det=round(float(det.mean()), 2), in_crop=round(inn.sum() / max(1, det.sum()), 3),
                           player_fallback=round(float(fbm.mean()), 3), fallback_runs=int(np.sum(np.diff(np.r_[0, fbm.astype(int)]) == 1)),
                           jerk_p95=jk, jerk_p95_rounded=jk_r, jerk_qa=camera.jerk_qa(jk["px_per_frame3"] if jk else None, cam),
                           crop_h_frac=[round(float(hfrac.min()), 3), round(float(hfrac.max()), 3)], wide=bool(wide), wide_why=wide_why,
                           mean_crop_height=hq["mean_crop_height"], pct_frames_ge95=hq["pct_frames_ge95"],
                           pan_fast_frames=int(pp["fast"].sum()), exit_predicted_frames=pp["exit_frames"],
                           vertical="player box" if not np.isnan(top).all() else "center_y fallback (no player boxes)",
                           players_clipped_frames=int(clipped.sum()) if not np.isnan(top).all() else None,
                           planner=(gp.get("planner", "gate_plan") if gp is not None else "follow_pan"), reject=reject, player_boxes=box_src,
                           plan_s=(gp.get("plan_s") if gp is not None else None), plan_fallback=(gp.get("fallback") if gp is not None else None))
        if gp is not None:
            rq = camera.role_clip_qa(gp["roles"], x, y, cw, ch, S, SW, FPS, a, cam, wide)
            stats[name].update(roles_qa=rq, caps=dict(v_over_cap=round(gp["caps"][0], 3), a_over_cap=round(gp["caps"][1], 3),
                                                      quick_reversals=len(gp["caps"][2])), infeasible=gp["infeasible"])
            for key in ("contacts_hitter_clipped", "contacts", "receptions_receiver_clipped", "receptions"):
                stats[key] = stats.get(key, 0) + rq[key]
            shots_ = gp.get("shots")
            if shots_ is None: shots_ = camera.shot_context(gp["roles"], bxi, n, FPS, a, [e.t for e in events if e.type == "bounce" and a <= e.t <= b], cam, pos, S)
            lq = camera.landing_qa(shots_, x, cw, S, SW, FPS, a, wide or gp.get("planner") == "locked_wide")
            stats[name].update(landing=lq, long_shot_mode=gp.get("long_shot_mode"), cuts=gp.get("cuts", []),
                               long_shots=gp.get("long_shots"))
            pq = camera.preserve_qa(gp, x, y, cw, ch, S, FPS, a, wide, cam)    # v4.6 pre-serve framing + source edge
            stats[name].update(preserve=pq, source_edge=gp["source_edge"])
            if gp.get("serve_i") is not None:                          # ball in crop before / from the serve
                k_ = gp["serve_i"]; stats[name].update(in_crop_pre_serve=round(inn[:k_].sum() / max(1, det[:k_].sum()), 3),
                                                       in_crop_from_serve=round(inn[k_:].sum() / max(1, det[k_:].sum()), 3))
            clip_checks[name] = dict(preserve=pq, source_edge=gp["source_edge"], wide=bool(wide))
        stats["fallback_frames"] = stats.get("fallback_frames", 0) + int(fbm.sum())
        plan = camera.crop_render(x, y, cw, ch, SW, SH)
        with open(os.path.join(work, f"cmd_{name}.txt"), "w") as f:
            f.write("\n".join(camera.crop_render_cmds(plan, FPS)) + "\n")
        for dt, (i, img) in fd.items():
            im = img.copy()
            X, Wd = x[i] / S, cw[i] / S
            cv2.rectangle(im, (int(X), int(y[i] / S)), (int(X + Wd), int(min(AH - 1, (y[i] + ch[i]) / S))), (0, 255, 255), 3)
            if det[i]:
                cv2.circle(im, (int(pos[i][0]), int(pos[i][1])), 12, (0, 0, 255), 3)
            cv2.putText(im, f'{name} t={dt:.1f} ball={"yes" if det[i] else "lost"}{" PLAYERS" if fbm[i] else ""} h={hfrac[i]:.2f}{" WIDE" if wide else ""}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
            dbg.append(cv2.resize(im, (640, 360)))
        if cfg.get("plan_only"):                                      # camera checks only: stats are computed above
            print(name, stats[name], flush=True)
            continue
        # render moment: crop path -> 1080x1920, ramps (slow-mo at contact), caption
        rel = [h - a for h in ramps]
        parts, cur = [], 0.0
        for h in rel:
            parts.append((cur, h - PRE, 1.0))
            parts.append((h - PRE, h + POST, SLOW))
            cur = h + POST
        parts.append((cur, b - a, 1.0))
        parts = [p for p in parts if p[1] - p[0] > 0.02]
        head = (camera.wide_filter(FPS, len(parts)) if wide else
                camera.crop_render_head(plan, f"cmd_{name}.txt", SH, FPS, len(parts)))
        fc = [head + "".join(f"[vs{i}]" for i in range(len(parts))),
              "[0:a]aresample=48000,asplit=" + str(len(parts)) + "".join(f"[as{i}]" for i in range(len(parts)))]
        cat = ""
        for i, (p0, p1, sp) in enumerate(parts):
            if sp == 1.0:
                fc.append(f"[vs{i}]trim={p0:.4f}:{p1:.4f},setpts=PTS-STARTPTS[v{i}]")
                fc.append(f"[as{i}]atrim={p0:.4f}:{p1:.4f},asetpts=PTS-STARTPTS[a{i}]")
            else:
                fc.append(f"[vs{i}]trim={p0:.4f}:{p1:.4f},setpts=(PTS-STARTPTS)/{sp},minterpolate=fps={FPS}:mi_mode=blend[v{i}]")
                fc.append(f"[as{i}]atrim={p0:.4f}:{p1:.4f},asetpts=PTS-STARTPTS,atempo={sp}[a{i}]")
            cat += f"[v{i}][a{i}]"
        fc.append(f"{cat}concat=n={len(parts)}:v=1:a=1[cv][ca]")

        txt = text_filters(m, parts, title, end, fontfile_opt(font))
        if m.get("caption_why"): stats.setdefault("captions_dropped", {})[name] = m["caption_why"]
        fc.append("[cv]" + (",".join(txt) if txt else "null") + "[ov]")
        o = f"seg_{k:02d}_{name}.mov"
        hwa = ["-hwaccel", hw] if hw else []
        subprocess.run(["nice", "-n", "10", FF, "-nostdin", "-loglevel", "error", "-y", *hwa, "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", src,
                        "-filter_complex", ";".join(fc), "-map", "[ov]", "-map", "[ca]", "-r", str(FPS), "-c:v", vcodec, "-b:v", "13M", "-maxrate", "16M",
                        "-bufsize", "26M", "-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-ar", "48000", o], check=True, cwd=work)
        segs.append(o)
        print(name, stats[name], flush=True)

    if segs:
        with open(os.path.join(work, "concat.txt"), "w") as f:
            f.writelines(f"file '{s}'\n" for s in segs)
    af = ("asplit[a][b];[b]highpass=f=1500,lowpass=f=6000,volume=1.8[h];"
          "[a][h]amix=inputs=2:normalize=0,loudnorm=I=-12.5:TP=-1.5:LRA=11,alimiter=limit=0.84:level=false,aresample=48000")
    if segs: subprocess.run(["nice", "-n", "10", FF, "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt", "-filter_complex", f"[0:a]{af}[ao]",
                    "-map", "0:v", "-map", "[ao]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out], check=True, cwd=work)
    if dbg:
        rows = [np.hstack(dbg[i:i + 3] + [np.zeros_like(dbg[0])] * (3 - len(dbg[i:i + 3]))) for i in range(0, len(dbg), 3)]
        cv2.imwrite(os.path.join(work, "ball_debug_sheet.png"), np.vstack(rows))
    stats["pct_ball_frames_in_crop"] = round(100 * stats["ball_in_crop"] / max(1, stats["ball_frames"]), 1)
    stats["pct_frames_with_ball_detected"] = round(100 * stats["ball_frames"] / max(1, stats["frames"]), 1)
    stats["pct_frames_player_fallback"] = round(100 * stats.get("fallback_frames", 0) / max(1, stats["frames"]), 1)
    if jerk_samples:      # pooled over moments; third differences never span a cut
        p = float(np.percentile(np.concatenate(jerk_samples), 95))
        stats["jerk_p95"] = dict(px_per_frame3=float(f"{p:.6g}"), norm=float(f"{p / SW:.6g}"), frame_w=SW,
                                 note="95th pct |d3 float crop centre x| per frame^3 (before rounding); pass <= jerk_ratio x jerk_baseline")
        stats["jerk_qa"] = camera.jerk_qa(p, cam)
    if jerk_samples_r:
        p = float(np.percentile(np.concatenate(jerk_samples_r), 95))
        stats["jerk_p95_rounded"] = dict(px_per_frame3=float(f"{p:.6g}"), norm=float(f"{p / SW:.6g}"),
                                         note=f"debug: rendered whole-pixel path; any moving crop scores ~{camera.QUANT_FLOOR_PX:g}")
    stats["pct_frames_players_clipped"] = round(100 * stats.get("players_clipped_frames", 0) / max(1, stats["frames"]), 1)
    if h_all:             # soft check (warning, not a failure); doubles gate is on the non-wide frames
        fmt, sport = cfg.get("format"), cfg.get("sport", "tennis")
        hq = camera.crop_height_qa(np.concatenate(h_all), cam)
        nw = [h for h, w in zip(h_all, wide_flags) if not w]
        hq_nw = camera.crop_height_qa(np.concatenate(nw) if nw else np.array([]), cam)
        gate = camera.crop_height_qa(np.concatenate(nw) if (fmt == "doubles" and sport != "padel" and nw) else np.concatenate(h_all), cam, fmt, sport)
        stats["mean_crop_height"], stats["pct_frames_ge95"] = hq["mean_crop_height"], hq["pct_frames_ge95"]
        stats["mean_crop_height_excl_wide"], stats["pct_frames_ge95_excl_wide"] = hq_nw["mean_crop_height"], hq_nw["pct_frames_ge95"]
        stats["crop_height_check"] = dict(target=camera.crop_height_target(cam, fmt, sport), level="warning",
                                          ok=not gate["warnings"], warnings=gate["warnings"])
        for w in gate["warnings"]: print("warning: camera", w, file=sys.stderr, flush=True)
    if "contacts" in stats:   # hard gates: no hitter / receiver clipped at their contact
        stats["roles_check"] = {"pass": stats["contacts_hitter_clipped"] == 0 and stats["receptions_receiver_clipped"] == 0}
    for kk, lab in (("far_out", "far_player_out_of_frame"), ("any_out", "any_player_out_of_frame"), ("sliced", "player_sliced_by_crop_edge")):
        stats["pct_frames_" + lab] = round(100 * stats.get(kk + "_frames", 0) / max(1, stats["frames"]), 1)
    stats.update(camera.framing_checks(clip_checks, stats.get("pct_frames_far_player_out_of_frame"), cam))
    stats["pct_frames_wide"] = round(100 * stats.get("wide_frames", 0) / max(1, stats["frames"]), 1)
    stats["wide_moments"] = n_wide
    for key in ("hsv", "tracknet", "fused"):
        stats["pct_detect_" + key] = round(100 * stats.get("n_" + key, 0) / max(1, stats["frames"]), 1)
    json.dump(stats, open(os.path.join(work, "ball_stats.json"), "w"), indent=1)
    print(json.dumps(stats))
    return stats
