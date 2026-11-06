"""Story/shot finder: point_quality per point, momentum curve, story beats, difficult shots.

Excitement is now the spec's point_quality (highlight_algorithm.md v3.2, story/highlight.py), replacing the old additive
z-score weights: PQ = M_eff x (largest 3 of G_rally = 0.75*v_rally^2 + 0.05*v_duration, G_out, G_react), with the pressure
multiplier from scoring.formats.pressure_multiplier() when a per-point score is given. Audio-feature based (rallies +
loudness) unless the score carries the ending; an editing aid only, not stats/coaching. Work in progress.

MD O5: an audio-only reaction raises a point's rank and may extend its clip, but it is a standalone beat ("loudest reaction")
only for points whose reaction pose has confirmed (`pose_confirmed`). Fault-like / serve-only segments score 0 (non-play).
"""
import numpy as np

from .config import MatchSettings, UserConfig
from .highlight import DEFAULT_WEIGHTS, PointInput, personal_reel, score_points


def build(segs, lb, start, changeover_after_idx=(), score=None, pose_confirmed=(), settings=None, user=None, weights=DEFAULT_WEIGHTS):
    """segs: list of dict(A,B,rallies=[{t0,t1,hits,times,amax}],hits,fault_like,serve_only)
    lb: broadband loudness dB per 10 ms. changeover_after_idx: segment indices followed by a changeover.
    score: optional per-point list of dicts (break_point, set_point, match_point, deuce, tb_game_point, user_set_point,
    user_match_point; winner 'host'/'guest' and ending 'winner'|'ace'|'service_winner'|... when known) for pressure and outcome.
    pose_confirmed: point numbers (1-based) whose post-point reaction is pose-confirmed (O5).
    settings: MatchSettings (user_team); user: UserConfig (loss_weight) -> a personal reel when the score carries winners."""
    P = []
    for i, s in enumerate(segs):
        times = [t for r in s['rallies'] for t in r['times']]; d = np.diff(times) if len(times) > 1 else np.array([])
        l = s['rallies'][-1]['t1']; f = s['rallies'][0]['t0']
        post = lb[int((l + .3) * 100):int((l + 4) * 100)]; pre = lb[max(0, int(f * 100) - 300):int(f * 100)]
        base = float(np.median(pre)) if len(pre) else float(np.median(lb))
        react_peak = float(np.percentile(post, 95) - base) if len(post) else 0
        react_dur = float((post > base + 8).sum() / 100) if len(post) else 0
        fast = int((d < 0.8).sum()); tempo_change = float(np.abs(np.diff(d)).max()) if len(d) > 2 else 0
        amax = max(r['amax'] for r in s['rallies'])
        best = max(s['rallies'], key=lambda r: r['hits']); bd = np.diff(best['times']) if best['hits'] > 1 else np.array([9.0])
        clean = (best['t1'] - best['t0']) <= 30 and float(np.median(bd)) <= 2.0 and len(s['rallies']) == 1 and (s['B'] - s['A']) <= 40
        P.append(dict(idx=i, A=s['A'], B=s['B'], first_hit=f, last_hit=l, hits=s['hits'], rally_hits=best['hits'], clean=clean, fast=fast, tempo=tempo_change,
                      amax=amax, react=react_peak, react_dur=react_dur, fault_like=bool(s.get('fault_like')), serve_only=bool(s.get('serve_only')),
                      rally_span=float(best['t1'] - best['t0'])))
    n = len(P)
    if n == 0: return {}
    st = settings or MatchSettings(pose_confirmed_reactions=frozenset(pose_confirmed))
    if pose_confirmed and not st.pose_confirmed_reactions:
        st = MatchSettings(st.user_team, st.user_team_confirmed, st.sport, st.deuce_rule, frozenset(pose_confirmed))
    play = [p for p in P if not (p['fault_like'] or p['serve_only'])]
    pscore = list(score or [])
    keys = ('match_point', 'set_point', 'break_point', 'deuce', 'thirty_all', 'tb_game_point', 'pickleball_game_point', 'user_set_point', 'user_match_point')
    inputs = [PointInput(point=p['idx'] + 1, server=None, winner=(pscore[p['idx']].get('winner') if p['idx'] < len(pscore) else None),
                         rally_shots=p['rally_hits'], rally_span_s=p['rally_span'], fast_exchanges=p['fast'], react_db=p['react'],
                         react_dur=p['react_dur'], t_last=p['last_hit'],
                         ending=(pscore[p['idx']].get('ending') if p['idx'] < len(pscore) else None),
                         pressure_flags={k: bool(pscore[p['idx']].get(k)) for k in keys} if p['idx'] < len(pscore) else {})
              for p in play]
    scored = {s_['point'] - 1: s_ for s_ in score_points(inputs, st, weights)}
    for j, p in enumerate(P):
        hs = scored.get(j)
        ctx = []
        if j == n - 1: ctx.append('final point')        # final-point override: always the 'close' beat (§8), not a score bonus
        if j == 0: ctx.append('opening point')
        if j - 1 in changeover_after_idx: ctx.append('first point after changeover')
        if pscore and j < len(pscore):
            for k in ('match_point', 'set_point', 'break_point', 'deuce'):
                if pscore[j].get(k): ctx.append(k.replace('_', ' '))
        p['excitement'] = round(float(hs['pq']), 4) if hs else 0.0
        p['context'] = ctx
        p['pq_terms'] = {k: round(float(hs[k]), 4) for k in ('M', 'g_rally', 'g_out', 'g_react')} if hs else {}
        p['reaction_present'] = bool(hs and hs['reaction_present'])
        p['reaction_beat_ok'] = bool(hs and hs['reaction_beat_ok'])
    ex = np.array([p['excitement'] for p in P]); w = max(3, n // 15)
    mom = np.convolve(ex, np.ones(w) / w, mode='same')
    # difficult shots (audio signatures): reflex volley (<0.6 s after previous hit), get after apparent winner
    # (loud hit followed by another hit and the rally continues), lob/overhead (gap >= 2.2 s mid-rally then loud hit)
    D = []
    for p, s in zip(P, segs):
        for r in s['rallies']:
            t = r['times']
            for k in range(1, len(t)):
                gap = t[k] - t[k-1]; sc, why = 0, []
                if 0.36 <= gap < 0.6 and 1 < k < len(t) - 1: sc += 1 + (0.6 - gap) * 3; why.append(f'reflex volley ({gap:.2f}s)')
                if k >= 2 and k < len(t) - 1 and gap >= 2.2: sc += 1.2; why.append('lob/overhead pattern')
                if k == len(t) - 1 and len(t) >= 6 and gap < 0.9: sc += 0.6; why.append('fast finish')
                if sc: D.append(dict(point=p['idx'] + 1, t=round(t[k], 2), score=round(sc, 2), why=why))
    # get after apparent winner: loudest hit of rally (top amplitude unknown per hit -> approximate with long gap + continuation)
    D.sort(key=lambda d: -d['score'])
    def tag(p, kind=''):
        # internal ranking tag from audio-only signals: never shown on screen (rule R3). Captions come only from a
        # verified trigger in opencourtvision.captions; `caption_trigger` names the one a reel moment for this beat should ask for.
        if 'final point' in p['context']: return 'final point', 'match_point'
        if kind == 'loudest reaction' and p['react'] >= 15 and p['reaction_beat_ok']: return 'loud reaction', None
        if p['clean'] and p['rally_hits'] >= 12: return f"{p['rally_hits']}-hit rally (audio)", 'rally_of_match'
        if p['hits'] >= 15: return 'long rally (audio)', 'rally_of_match'
        if p['fast'] >= 6: return 'fast exchanges (audio)', 'net_exchange'
        if p['react'] >= 15 and p['reaction_beat_ok']: return 'loud reaction', None
        return '', None
    rank = sorted(P, key=lambda p: -p['excitement'])
    beats = []
    def add(kind, p, why):
        if p is None: return
        beats.append(dict(beat=kind, point=p['idx'] + 1, start=p['A'], end=p['B'], first_hit=p['first_hit'], last_hit=p['last_hit'],
                          excitement=p['excitement'], tag=tag(p, kind)[0], caption_trigger=tag(p, kind)[1], why=why))
    real = [p for p in P if not (p['fault_like'] or p['serve_only']) and (p['B'] - p['A']) <= 40]
    add('opener', real[0] if real else None, 'first point of the match')
    med = float(np.median([p['excitement'] for p in real])) if real else 0
    rise = [(mom[p['idx']] - mom[max(0, p['idx'] - w)], p) for p in real if p['excitement'] > med and 0 < p['idx'] < n - 1]
    add('turning point', max(rise, key=lambda x: x[0])[1] if rise else None, 'biggest rise in rolling excitement (momentum run)')
    add('longest rally', max(real, key=lambda p: p['hits']), 'most hits')
    conf = [p for p in real if p['reaction_beat_ok']]      # O5: standalone reaction beat only once pose confirms it
    add('loudest reaction', max(conf, key=lambda p: p['react']) if conf else None, "post-point reaction (pose-confirmed)")
    add('peak', next((p for p in rank if p in real), None), 'highest excitement score')
    add('close', P[-1], 'final point')
    hook = max(beats, key=lambda b: b['excitement'])
    personal = None
    if pscore and any(x.get('winner') for x in pscore):
        pr = personal_reel(list(scored.values()), (user or UserConfig()).loss_weight, weights=weights)
        personal = dict(loss_weight=pr['loss_weight'], selected=pr['selected'], order=pr['order'], climax=pr['climax'], great=pr['great'])
    return dict(match_start=start, n_points=n, hook=hook, beats=beats, personal=personal,
                reaction_extends=[p['idx'] + 1 for p in P if p['reaction_present']],
                top_points=[dict(point=p['idx'] + 1, start=p['A'], excitement=p['excitement'], hits=p['hits'], rally_hits=p['rally_hits'], clean=p['clean'], react=round(p['react'], 1),
                                 fast=p['fast'], context=p['context'], tag=tag(p)[0], pq_terms=p['pq_terms'],
                                 reaction_extends_clip=p['reaction_present']) for p in rank[:15]],
                difficult_shots=D[:15], momentum=[round(float(m), 2) for m in mom],
                note='PQ per highlight_algorithm.md v3.2 (provisional weights); audio-only reactions extend/rank but are no standalone beat until pose confirms (O5); distance-covered/positional difficulty not computed; tags are internal; on-screen captions only via verified triggers (opencourtvision.captions)')
