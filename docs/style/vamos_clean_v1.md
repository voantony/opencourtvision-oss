# Vamos Highlight Style v1 — "Vamos Clean"

**Owner:** Media Director (Antony Vo)  
**For:** Engineer render preset (replaces placeholder `"Vamos Clean"`)  
**Status:** Decision — encode as-is. Not a menu of options.  
**Date:** 19 Oct 2026  
**Supersedes for reels:** aesthetic deltas vs `mvp_edit_style.md` v1.1 where this file is more specific. Hard rules R-WHOLE / R-9x16 / Antony constraints below always win.

Machine-readable twin: `preset.json`. Grade asset: `vamos_grade.cube`. Mock frames: `frames/`. Research notes: `references.md`.

---

## 0. Hard constraints (Antony — non-negotiable)

1. **Restraint and quality over quantity.** Drop beats under the match p60 clip_score. Shorter is better than padded.
2. **Slow-mo 0–2 per reel**, only on frame-verified exceptional shots. **Zero is fine.** No gimmicks (no zoom-blur, flash, shake, glitch, whip-pan as effect).
3. **Caption integrity (R-WHOLE):** a caption may only appear on a clip that shows the **whole event** it names. Every label verified against the frames. No hype labels on ordinary shots. No `MATCH POINT` on a teaser.
4. **Whole points only:** every reel clip runs from a few seconds before the serve to **1.5–2.0 s** after the point is decided (floor 2.0 s after last contact; aim 2.5 s). Never mid-rally fragments or teasers.
5. **9:16 never letterboxed or wide.** True `crop_9x16_tracked` only.
6. **Camera style intent only in this doc:** dynamic zoom subtle and slow, smooth pans, no pumping. Framing code is Engineer's; Media Director only pass/fails against the fixed camera checks.
7. **No face / biometric ID.** Players by court side or role. No real-person likeness in any generated asset.

---

## 1. Positioning

**Night-session broadcast, celebration moments.** Closer to ATP/Tennis TV official packages and Grand Slam score bugs than to fan montage chaos. Take from Wivorn-class creators: **emotional beat selection, music that serves the hits, clean titles** — not rapid punch-in spam or emoji rain.

Tone: premium, bilingual-ready (`¡Vamos!`), quiet confidence. One accent colour (Volt). Celebration colours (Sol, Fiesta) only on verified celebration moments, never as chrome.

---

## 2. Type and fonts (open licence only)

| Role | Family | Weight | Licence | Notes |
|---|---|---|---|---|
| Stakes captions (`MATCH POINT`, `BREAK POINT`, …) | **Barlow Condensed** | Black Italic 900 | OFL (Google Fonts) | ALL CAPS, tracking **+40** (≈ +2%) |
| Names on bug / cards | **Barlow Condensed** | SemiBold 600 | OFL | Title case or ALL CAPS surnames; truncate at surname |
| Headlines / intro line 3 | **Barlow Condensed** | Bold 700 | OFL | |
| Cheer / wordmark moments only | **Archivo Black** | Regular 400 | OFL | `¡VAMOS!` only — never body |
| All numbers (score, games, timers) | **JetBrains Mono** | Medium 500 / Bold 700 | OFL | Tabular figures; Medium for finished sets (Mist), Bold for live |
| UI chips / secondary | **Inter** | 600–700 | OFL | Not burned into match video except debug |

### Sizes (px at delivery resolution)

| Element | 1080×1920 (9:16) | 1920×1080 (16:9) | 3840×2160 (4K master) |
|---|---|---|---|
| Stakes caption | 56 px, tracking +40 | 48 px, +40 | 96 px, +40 |
| Bug names | 28 px | 22 px | 44 px |
| Bug live point | 26 px Bold | 20 px Bold | 40 px Bold |
| Bug finished-set games | 22 px Medium Mist | 18 px | 36 px |
| Intro line 1–2 (Mist) | 28 px | 24 px | 48 px |
| Intro line 3 (teams) | 40 px Bold | 36 px | 72 px |
| Result winners | 48 px Bold | 56 px | 112 px |
| Result score | 52 px Mono Bold | 64 px | 128 px |
| Game chip | 22 px | 18 px | 36 px |

**Anti-alias:** subpixel off for Mono digits (snap to whole pixels). Captions: 1 px Night stroke at 40% under Chalk for clay/grass readability when no chip behind.

---

## 3. Palette

| Token | Hex | Edit use |
|---|---|---|
| Night | `#0B0F14` | Card / bug fill at **90%** opacity (intro band **80%**) |
| Graphite | `#161C24` | Secondary rows, chips |
| Border | `#2A3340` | 1 px bug / card stroke |
| Chalk | `#F2F4F7` | Primary text |
| Mist | `#8E99AB` | Secondary labels, finished-set games, losers on result |
| Volt | `#C8F43C` | **Only accent:** serve dot, live point cell, winner underline, caption underline |
| Sol | `#FFB627` | Celebration confetti / wordmark dots only — not in match chrome |
| Fiesta | `#FF4D6D` | Recording / errors only — never in highlight reels |
| Court Blue | `#2F7BFF` | Not used in match edits |

Volt never as body text on Chalk. Night text on Volt for the live point cell.

---

## 4. Grade (broadcast-clean, not a "look")

Gentle normalisation so phone / club cameras sit next to broadcast packages. **Not** a stylised LUT look; if the source is already clean, the delta is barely visible.

### Numeric adjustments (order)

1. Exposure **+0.08 stop** (×1.057 linear-ish in sRGB domain)  
2. Shadow lift **+0.014** with falloff below luma 0.35  
3. Contrast **1.08** around pivot **0.45**  
4. Soft highlight roll-off above **0.85**  
5. Saturation **+6%**  
6. Mild midtone warmth: **+0.008 R / −0.004 B** (Gaussian mid mask σ≈0.18)

**File:** `vamos_grade.cube` — 33³ IRIDAS `.cube`, DOMAIN 0–1.

### ffmpeg equivalent (when LUT not available)

```bash
# Approx match to vamos_grade.cube (sRGB)
eq=contrast=1.08:brightness=0.02:saturation=1.06,\
colorbalance=rs=0.02:gs=0:bs=-0.01:rm=0.01:bm=-0.008,\
curves=master='0/0.02 0.45/0.45 0.85/0.82 1/0.94'
```

Preferred path: `lut3d=vamos_grade.cube:interp=tetrahedral`. Apply **after** crop/reframe, **before** overlays.

**Never:** vignette, film grain, teal-orange crush, heavy halation, fake anamorphic flares.

---

## 5. Score bug

### Layout (compact, server-first, fixed rows)

```
┌─────────────────────────────────────────────┐
│ ● VO / CARDONA          6   2      [ 30 ]   │  Night 90%, Volt ● = serving team
│   MARI / MOUGA          1   1        15     │  User's team always on top row
└─────────────────────────────────────────────┘
```

- Live point cell: Volt fill + Night text on the row being scored into after the point; during the point both cells use Graphite/Chalk except the serving team's serve dot stays Volt.
- Finished-set games: Mist. Current-set games: Chalk.
- Width: **≤ 70%** of frame in 9:16; **≤ 26%** in 16:9.
- Height: **~5.4%** of frame height (two rows) in 9:16; **~6.5%** in 16:9.

### Position

| Aspect | Position |
|---|---|
| 9:16 1080×1920 | Horizontally centred; top of bug at **y = 320** (16.7%), below universal top safe (288). Never in top 288 or bottom 672. |
| 16:9 1920×1080 | Top-left inside 5% title-safe: **x ≥ 96, y ≥ 54**. |
| 4K 3840×2160 | Scale 2× from 16:9. |

### When shown

- **Full-match cut / 4K highlights / top-3:** on throughout (hide only if score unverified — then games-only or fully hide per `mvp_edit_style` §5.2).
- **Story reel:** only on beats where the score changed or a stakes caption is shown.
- Score updates at `t_dec + 0.5 s` (never mid-rally). Point flip **120 ms**. Game cell Volt pulse **400 ms**.

### Motion in / out

| Event | Duration | Easing |
|---|---|---|
| First appear on a beat | 180 ms fade | ease-out cubic |
| Hide for ball-in-bug conflict (±0.5 s of contact in bug box) | 150 ms fade | linear |
| Clear on cut | **must be clear** — 150 ms fade ending at the cut; never straddles a cut |

---

## 6. Captions (stakes and shot)

### Allowed stakes (verified score state only)

`BREAK POINT`, `SET POINT`, `MATCH POINT`, `MATCH POINT SAVED` (post-resolution only), `PUNTO DE ORO` / `STAR POINT` (verified deciding point only), `GAME POINT · …` (pickleball rules per mvp §4.3). Optional `· 4–5` only when game score verified.

### Shot labels (verified frames only)

Examples: `RUNNING GET`, `NET CORD`, `ACE`. Factual. Never hype (`INSANE`, `CRAZY`, emoji).

### Timing and motion

| | Value |
|---|---|
| On-screen duration | **0.8–1.2 s** at start of point pre-roll (stakes); shot labels span the verified shot ±0.2 s |
| Cap | ≤ 3 stakes captions per reel; ≤ 1 caption per 8 s average |
| In | 160 ms fade + 2 px upward ease-out |
| Out | 150 ms fade; **must end before the next cut** (SwingVision lesson) |
| Underline | Volt, 3 px, full text width, appears with caption |

Zone 9:16: **y 60–72%** of frame, x centred but bias **left of geometric centre by ~8%** when the right action rail risks collision (compose caption centre near x=450 on 1080). Zone 16:9: lower third **y 72–85%**.

---

## 7. Transitions

| Transition | Allowed? | When | Duration |
|---|---|---|---|
| **Hard cut** | **Default — always** | Point → point, beat → beat | 0 (video); audio 200 ms crossfade |
| Dip to black | Rare | Only between discrete whole points when music is on **and** the score jumps by a game or more | **80 ms** down + **80 ms** up |
| Dissolve | Loop point only | Outro last 0.3 s ↔ hook first frame match | ≤ 300 ms |
| Whip / flash / zoom-blur / glitch / wipe | **Never** | — | — |

---

## 8. Slow-mo and speed ramps

| Rule | Value |
|---|---|
| Budget | **0–2** per reel; zero preferred |
| Eligibility | Frame-verified exceptional shot only (winner on the stretch, diving get, overhead smash winner, etc.). Engineer camera T1 punch-in candidates ∩ Media Director label verify |
| Max factor | **0.45×** (≈ 45% speed) |
| Ramp in / out | Cosine ease, **280 ms** each side; never a hard speed step |
| Hold | ≤ **700 ms** at the slow plateau |
| Audio | Court audio stays real-time **or** pitched with `atempo` to match; prefer keep hits at real-time ducking under a soft whoosh **only if** licensed SFX pack provides it — default: **no added whoosh**, just atempo |
| Interpolation | Prefer optical-flow (`minterpolate=mi_mode=mci`) when fps ≥ 50 source; else frame-blend. Never ship juddery 0.45× |
| Forbidden | Slow-mo on ordinary groundstrokes, on the serve toss alone, or to hit a music beat |

---

## 9. Cut sync (beats and hits)

1. **Primary cut point:** action — typically the contact that ends the previous point's post-roll / starts the next pre-roll boundary.  
2. **Music beat snap:** allowed **only if** a beat falls within **+0…+80 ms** after the action cut. The outgoing clip may only get **longer**, never shorter.  
3. **Never** trim, speed, or retime a whole point to land on a beat.  
4. End cards and result cards are never moved for music.  
5. Hit emphasis: original transient kept; boost ≤ **+2 dB** at ~3 kHz presence; no synthetic hit samples.

---

## 10. Music and SFX

| Rule | Value |
|---|---|
| Default | **Court audio only** |
| Music allowed | Opt-in on story / per-player / bloopers / recaps. Never on full-match, 4K highlights, top-3, best-point-today |
| Recommended share path | Leave music off in-file; user picks platform sound at share (licence-safe) |
| If baking music | Royalty-free catalogue with stored licence ID in export metadata. Sources: **Epidemic Sound** (commercial), **Artlist**, **Uppbeat** (check plan), **Pixabay Music** (verify current licence). No user-ripped / "trending" audio baked in |
| Music bed level | **−6 LU** under target loudness before ducking |
| Duck under contacts / voice | **−11 dB** (30 ms attack, 250 ms release) |
| Loudness targets | **−14 LUFS** integrated (9:16 social); **−16 LUFS** (16:9). True peak ≤ **−1 dBTP** |
| SFX | No added crowd, no risers, no whooshes by default. Court ambience stays |

---

## 11. Intro, outro, branding

| Card | Spec |
|---|---|
| **Intro** | ≤ **1.5 s**, over live first-point pre-roll (never black). Night **80%** lower band at y ≈ 70–88% (9:16) or lower-third (16:9). L1 Mist: `TENNIS · DOUBLES`. L2 Mist: `VENUE · DD MON YYYY` (omit venue if unknown). L3 Chalk Bold: `USER TEAM  vs  OPPOSITION` (user first). |
| **Result / outro** | **2.0–3.0 s** after final point outcome shown (R-WHOLE). Winner first, Volt underline, `def.`, losers Mist, score JetBrains Mono from winner's side. Source: user-entered final score agreeing with rules engine. |
| **Logo C (`¡V!` Bang)** | Outro only. Horizontal mark ≤ **12%** frame width in 16:9; ≤ **28%** in 9:16 (platform bottom UI forces higher placement — sit just above bottom safe). Chalk / mark at **60%** opacity (9:16 may use 90% if contrast fails). Never over live play. Never on junior exports. Removable on paid tiers. |
| **Watermark during play** | **None** by default. |
| **Reel loop** | Outro last **0.3 s** matches hook first frame in zoom and court side. |

---

## 12. Safe zones (pixels)

### 9:16 at 1080×1920 — Vamos universal (strictest of TikTok / Reels / Shorts)

| Edge | px | % |
|---|---|---|
| Top keep-clear | **288** | 15.0% |
| Bottom keep-clear | **672** | 35.0% |
| Left keep-clear | **120** | 11.1% |
| Right keep-clear | **300** | 27.8% |

Usable box: **x 120–780, y 288–1248** (660×960). Centre of safe box is **left of frame centre** (compose key action ~x 450, not 540).

Per-platform reference (do not use alone for baked overlays):

| Platform | Top | Bottom | Left | Right |
|---|---|---|---|---|
| TikTok | 240 | 660 | 120 | 300 |
| IG / FB Reels | 269 | 672 | 65 | 65 |
| YT Shorts | 288 | 672 | 48 | 192 |

### 16:9 at 1920×1080

| Edge | px |
|---|---|
| All sides title-safe | **96** (5%) |
| Caption keep clear of bottom | bottom **54** (5%) absolute minimum; prefer action above y=850 |

### 4K 3840×2160

Scale 16:9 figures ×2.

**Bug / caption vs ball:** if ball track enters bug box within ±0.5 s of contact, fade bug to 0% for that window (150 ms fades).

---

## 13. Clip timing (reel)

| | Tennis | Notes |
|---|---|---|
| Pre-roll | serve contact **− 3.0 to − 4.0 s** | Antony prefers ready + full motion; floor −3.0 for R-9x16 `PRE_MIN` |
| Post-roll | ≥ `t_last + 2.0`, aim `max(t_last+2.5, t_dec+1.5)` | |
| Between points | Hard cut; no dead footage | |

---

## 14. Effects budget (per story reel)

| Effect | Max |
|---|---|
| Slow-mo | 0–2 |
| Verified punch-ins (camera) | ≤ 3 |
| Stakes captions | ≤ 3; ≤ 1 / 8 s |
| Dip-to-black | ≤ 2 |
| Music | opt-in |

---

## 15. What this preset is not

- Not SwingVision clone (no mid-point teasers, no letterbox 9:16, no unverified hype labels).  
- Not Wivorn tribute montage (we keep whole points; we do not retime for music).  
- Not a coaching stats dump (stats live behind the "Match stats" chip in-app, not burned into the social reel).

---

## 16. Engineer encode checklist

- [ ] Load `preset.json` as `style.preset = "vamos_clean_v1"`  
- [ ] Apply `vamos_grade.cube` after reframe, before overlays  
- [ ] Fonts: ship OFL files (already in `demo_2026-10-17/fonts/` + Archivo Black / Inter from Google Fonts)  
- [ ] R-WHOLE / R-9x16 validation via `caption_rules.py` before export  
- [ ] Camera path untouched by this preset (style only)  
- [ ] Pass/fail still against fixed camera checks (Media Director)

