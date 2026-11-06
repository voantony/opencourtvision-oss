#!/usr/bin/env python3
"""Vamos CC BY bootstrap footage: polite resumable download + pre-label pass (worker; launched by vamos_cc_footage.sh).

Per video: re-verify the licence from yt-dlp metadata (must be "Creative Commons Attribution"), download (source
resolution up to 2160p, VP9 preferred over AV1 for Apple decode), log to footage_credits_manifest.csv, then pre-label:
  1. contact sheet (1 frame/min, seeks only) + automatic checks: fixed camera, vertical, burned-in overlay, music hint
  2. warm-up / non-play trim from audio hits (detection.audio + detection.match_start.warmup_cutoff) -> trim.json sidecar
     (no re-encode)
  3. court line detection -> homography (sport court model) + line-support / reprojection quality
  4. per rally window: HSV+motion ball tracking (reel.ball.detect/track), moving-player blobs + IoU tracks,
     bounces (image-y reversal away from audio hits), contacts (audio hits)
  5. prelabels.json (+ ball.csv, players.csv) and qa.json; qa_summary.csv over all videos
All pre-labels are unverified machine guesses for the annotation pass (verified=false).
Outputs under BASE: <sport>/<id>.mp4, meta/, prelabels/<sport>/<id>/, footage_credits_manifest.csv, progress.tsv, run.log.
"""
import argparse
import csv
import datetime
import json
import os
import queue
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np

# sport, video_id, singles/doubles, channel, channel_url, title  (verified CC BY list, red flags removed;
# order = pickleball, padel, tennis). Excluded red flags: v9Nsu4I4VeY (SEO farm), EdB_ma9eiJ0 + UtHK_7cSEyg (content-farm
# re-uploads), 63TwSJNN0Os + Fm1P3rZvJb8 (broadcasts), g0n86XlTuDI (possible minor), HePfyAvyU7M + lJRlP5O6_84 +
# eoJ6l5BaY3k (school/minors), 13BV_YnKKsE + gn6IRQU8alc (broadcasts).
SOURCES = """
pickleball	vvZ55Lb0QXk	D	pickleball4you	https://www.youtube.com/channel/UC83z_qzETSgcemFzZdtmFvw	2026.09.05 MXD4.5+ (Open age) Min Zeng + Marco Wong vs Naomi Tran + Eurene Mak (Gold Medal)
pickleball	22IghwcHzXY	D	pickleball4you	https://www.youtube.com/channel/UC83z_qzETSgcemFzZdtmFvw	2022.07.17 MXD4.0 Christine Fan +Ming Wang vs Cathy Yu +Fanshi Lu (Gold Medal Game)
pickleball	ezAxDf6oCAU	D	pickleball4you	https://www.youtube.com/channel/UC83z_qzETSgcemFzZdtmFvw	2022.07.15 WD3.5 Phyllis Money +Corinne Susini vs Suzanne Gordon +Christina Hall (Gold Medal Game)
pickleball	5uAUWp6H5EY	D	pickleball4you	https://www.youtube.com/channel/UC83z_qzETSgcemFzZdtmFvw	2022.07.31 MXD4.5(age 60-69) Eileen Dowell+ Miroslaw Wojcik vs Marilyn Cross+Dave Anderson (Round 1)
pickleball	Aft2A3dka10	D	pickleball4you	https://www.youtube.com/channel/UC83z_qzETSgcemFzZdtmFvw	2026.09.06 MD4.5+(Open) Liam Villeneuve + Steven Liang vs Sean Dolphin + Tyler Tseng (R_2 backdraw)
pickleball	6sPwWQ_EVFk	D	Craig vs Pickleball	https://www.youtube.com/channel/UCoRUrqRSyLdZrph7rDF8lHw	2.75 DUPR Rotating Partner Tournament 🏆 Rd5 | Craig/Toan vs Hoang/Viet | Saigon, Vietnam, 14 May 25.
pickleball	7SzJeMV8K2s	D	Craig vs Pickleball	https://www.youtube.com/channel/UCoRUrqRSyLdZrph7rDF8lHw	So close again! Round Robin at Liber Pickleball. Round 7.
pickleball	YeXC1Rsd5Ik	D	Craig vs Pickleball	https://www.youtube.com/channel/UCoRUrqRSyLdZrph7rDF8lHw	7.0 DUPR Fixed Round Robin 🏆 Rd3 Jamie/Josh vs Bill/Huy | Saigon, Vietnam. 19/2/25.
pickleball	v5BHuxV1imM	D	Craig vs Pickleball	https://www.youtube.com/channel/UCoRUrqRSyLdZrph7rDF8lHw	2.75 DUPR Tournament 🏆 Rd2 Craig/jJun vs Huy/Hungbui | Dig it out. Saigon, Vietnam. 5/3/25
pickleball	63HheyDLY3M	D	BT Pickleball	https://www.youtube.com/channel/UCSUhWFY9_gvq3C4eBLWjppg	Pickleball Doubles 01/05/2024
pickleball	o_NeuHjTEVE	D	BT Pickleball	https://www.youtube.com/channel/UCSUhWFY9_gvq3C4eBLWjppg	Pickleball Doubles Mixed 3.5 to 4.5 in Melbourne, FL 06/04/2023 Game 1
pickleball	QhLECLlXU3c	D	BT Pickleball	https://www.youtube.com/channel/UCSUhWFY9_gvq3C4eBLWjppg	Pickleball Doubles Challenge Court in Melbourne, FL 06/09/2023 Game 1
pickleball	tZdmHZH_dxc	D	BT Pickleball	https://www.youtube.com/channel/UCSUhWFY9_gvq3C4eBLWjppg	The ball is melting! 4.0+ Mixed Doubles in the Florida heat - Melbourne, FL July 06, 2024
pickleball	ULvt8yhmjlc	D	Sigal Pickleball	https://www.youtube.com/channel/UCCCCsr1yhR9G_6MyJcuisxA	Very good game with two evenly matched sides! | Life Time RH ToC | 3.5-4.0 Men's Doubles | Game 4
pickleball	lebVxqwVSB0	D	Sigal Pickleball	https://www.youtube.com/channel/UCCCCsr1yhR9G_6MyJcuisxA	Life Time Novi Spring Classic Tournament | 3.5-4.0 Men's Doubles | Game 4
pickleball	OsnXfpuy1Lc	D	All Things Pickleball TV	https://www.youtube.com/channel/UCTdwavsR79UNRmJrIZMWH6w	3.5 Mixed Doubles SEMI FINAL The Final 4RTH Double Eliminations Pickleball Tournament Orem UT.
pickleball	Eot9G8H_bDs	D	All Things Pickleball TV	https://www.youtube.com/channel/UCTdwavsR79UNRmJrIZMWH6w	3.5 Men's Doubles - Elimination Bracketology - 2nd EB Team  - Club Pickleball USA Orem, UT
pickleball	JDuVy5k37tQ	D	New England Pickleball LIVE	https://www.youtube.com/channel/UCGUcn_i-b7IY9u8vU1xSfbQ	ELEVENO Pickleball Winter Classic Pickleball Tournament-Mens Doubles 4.0+ Semifinal
pickleball	LwqH1LsxYk4	D	New England Pickleball LIVE	https://www.youtube.com/channel/UCGUcn_i-b7IY9u8vU1xSfbQ	ELEVENO Pickleball Winter Classic Pickleball Tournament-Womens Doubles 4.0+ Final
pickleball	08iySuAXyLM	D	Herman Radtke	https://www.youtube.com/channel/UCuvoFeQMOEwrrtFpXXg1YmA	2021 NP Palm Desert Amateur Open 4.0 Men’s Doubles -  Match 1
pickleball	lat4bLVaY_o	D	Ishan R	https://www.youtube.com/channel/UCFR0jUjwRW0A2V-Jr0ajBdQ	Kitchen Battles- 3.75 Pickleball Doubles
pickleball	ql3Cc8K4ewE	D	Vernonchan.com	https://www.youtube.com/channel/UCyZVHLo8WdO5AQNN9hXoWWw	Thirstday Pickleball | TOUGH! Men vs ladies! Dave/ST vs Eunice/Gyna
pickleball	BiSfbA4CVMM	D	Chasing Pickleballs	https://www.youtube.com/channel/UCQn04yJl81xKf7qvJ_V4DGg	4.2 DUPR Open Play In California
pickleball	KVOTfkkS5e0	D	Chasing Pickleballs	https://www.youtube.com/channel/UCQn04yJl81xKf7qvJ_V4DGg	4.0 Game FINALS | April's FIRST Tournament | She Made It Here After One Year | August 2025
pickleball	peqPWLjRIqI	D/S	Outside the Cubicle	https://www.youtube.com/channel/UCdQCzNUV0DROZ7QcELMWOlw	4.0 Pickleball Doubles Tournament! (As a Beginner) | A2M Sports
pickleball	bY4IHIl0uNM	D	BK Pickleball	https://www.youtube.com/channel/UCKNprKuubbYgY8CKf-fUTEg	Extreme Competition!!!! 5.0 Mixed Doubles Ft. Cliff Pickleball
pickleball	qY2Zyt_JHCw	D	Eddie and Webby Pickleball	https://www.youtube.com/channel/UCXyYMsVqBlgUmZ5kcadS3Rg	Pickleplex Summer Slam 2019 - Eddie and Mark - Part 1 of 2
pickleball	93S-PKQ_Np8	D	Rapid River Pickleball League	https://www.youtube.com/channel/UCKB584-z9CB9LnIHKM7iF5w	BANDITS vs. TITANS | RRPL 2025
pickleball	AbmplFOGGkA	D	Duf	https://www.youtube.com/channel/UCM897K4cz5Jec8OawciE5Lg	Selkirk Boomstik Lunch time Doubles Play at the US Open National Pickleball Center
pickleball	VvJxw4rs2BY	D	AnigPie Cams	https://www.youtube.com/channel/UCvTB8PxrLawEKLLTA9u3teA	AnigPie Cams is live! Women's Open Play Pickleball
pickleball	r_yC3tiun0M	D	Sandy Austin	https://www.youtube.com/channel/UCZPhq0QqV1DWsN1D8E6jfCA	7 March 2020 Pickleball Tournament
pickleball	FKYS7Jckr1Y	D	Arthur Warick	https://www.youtube.com/channel/UC9qie0kirjabytmxhkSrnvQ	January 13, 2018  Parkside Pickleball Tournament
padel	rTP7SVtYNg4	D	Amateur Padel	https://www.youtube.com/channel/UCnY3YfSon2KM4y9awSs-Yrg	Bogdan/Malek VS Tudor/Costi - Padel FZN Arena
padel	2p6PcKP0720	D	Amateur Padel	https://www.youtube.com/channel/UCnY3YfSon2KM4y9awSs-Yrg	Bogdan/Costi VS Tudor/Ludovico - Padel FZN Arena - Level 2.5
padel	kGTdsXX2QMw	D	Padel Match Point	https://www.youtube.com/channel/UC-HVV4JL9YIsW4BgL9w9svw	Partita Padel amatoriale competitiva al Merate Rattix Arena - Full match - Partido Padél
padel	GKWNVYeEAlE	D	Padel Match Point	https://www.youtube.com/channel/UC-HVV4JL9YIsW4BgL9w9svw	Intermediate Level 3.5+ | Padel Match | Padel Match Point - @ Mariano Sports Arena
padel	632k3fKraeA	D	Padel Match Point	https://www.youtube.com/channel/UC-HVV4JL9YIsW4BgL9w9svw	Padel match level 3 - Partido de Padél - Partita padel competitiva - Full match
padel	228Grxn5Upo	D	Padel Match Point	https://www.youtube.com/channel/UC-HVV4JL9YIsW4BgL9w9svw	Partita Padel amatoriale - Padel match al Elle Padel - full match - 30 marzo 2024
padel	WRBD7jvdKqE	D	Jamie	https://www.youtube.com/channel/UCFo5vZYlk4qUzKTt8mjyH_g	Jamie's vs Kai / Will.... full match
padel	CR9H0SJarXg	D	REPARACIONES BARATAS LOW COST	https://www.youtube.com/channel/UCYlqxk-B6ryP0b1lFOH9F5Q	PÁDEL en Vivo# Partido AMATEUR 3ra Categoría# Cornellá Barcelona España
padel	GbakseAGK_Y	D	Padel indoor alcorcon	https://www.youtube.com/channel/UCn54AqGw8iigSr2w6XmULRw	Final Torneo 6 de mayo 2016
padel	V9z9kfoHfMM	D	Web de Padel	https://www.youtube.com/channel/UCDwZeZZCfXeL3zHOrxvR2nQ	Partido completo Good Training Zaragoza
padel	MKY3lTbJKf8	D	AblaViveelPadel	https://www.youtube.com/channel/UCNGQrnUowmTUm1ovjykA9cA	III Liga Pádel 2016 🎾 Sierra❄Nevada - Aficionado
padel	ZUSfVwuTEZ0	D	Jose Luis Mármol Rodríguez	https://www.youtube.com/channel/UCDC65imLzaCFzJR5bEefA1A	4 Padel Nuñez Guille Vs Mariano JL 141214
padel	dgyDyPJxZXU	D	Alfonso Montilla	https://www.youtube.com/channel/UCP-w4Pxi1jNeIMwQ2GOL59Q	Antonio Miranda y rodolfo ramos vs alfonso montilla y javier perez
padel	31l7LebSSPQ	D	pablo pastor	https://www.youtube.com/channel/UCzz9Py2ljyqvpTW7Be0QHWQ	padel 17/02/2012
padel	6nMll0Wo9G0	D	Padel a Full	https://www.youtube.com/channel/UCaTGwfmWPQmcBKPqd91lo8w	APA +35 LIBRES - SemiFinal - Poroto Perez y Canello VS Mónico y Daud- Torneo Nacional de Veteranos
padel	alQJ7sB8ztI	S	Padel Tommy 	https://www.youtube.com/channel/UCYGXSZxoRk493_EO0a9tySQ	Cross Padel - Tommy vs BO 1st match - Full match
padel	nptZ7bnXJmo	D	Through The Glass	https://www.youtube.com/channel/UCcKqYBCrPlJMBGhJv6IOPYA	Padel in Barcelona 2.0 - 2.5 | Match Highlights
padel	axrS5di8Y2s	D	Federacion de Padel CV FPCV	https://www.youtube.com/channel/UCdphvjpcJUXgY4_-_br9ktA	OPEN AUTONÓMICO 24* CLUB DE TENIS HELIOS | FINALES
padel	oelFNgIJyuI	D	Swish Live	https://www.youtube.com/channel/UC6HUcX9LdyVdIuzRLeWtugg	Padel Livestream with Swish Live
tennis	t0JCK2q9aZU	D	BHBK	https://www.youtube.com/channel/UCXpKBrbgEhiySXsfCggqqpg	Social Tennis Mens Doubles | Lee / Ryan v EJ / Daniel
tennis	eIxpMgMIiDo	D	BHBK	https://www.youtube.com/channel/UCXpKBrbgEhiySXsfCggqqpg	Social Tennis Mens Doubles | Teddy / Mieler v Ryan / Garry
tennis	godyFYrW650	D	BHBK	https://www.youtube.com/channel/UCXpKBrbgEhiySXsfCggqqpg	Social Tennis | Belconnen Tennis Club
tennis	XD7pOGqj5AI	D	Sammy Chang	https://www.youtube.com/channel/UC7s09373BsBm8MZWc0UhnQw	Summer 2026 SGV 3.5-4.5 Tri-Level Match #4
tennis	JReOUveKesY	D	Sammy Chang	https://www.youtube.com/channel/UC7s09373BsBm8MZWc0UhnQw	Fall 2026 SGV Mens 3.5 SoCal Doubles 18 & Over Match #2
tennis	5lNqTW9qFqs	D	Andy & A Kids	https://www.youtube.com/channel/UCpZT9pY-N9Qeg6bRHgbJZXg	USTA 3.5/4.0 Doubles - Andy & Steven vs. James & Max
tennis	bFuMpsuFK7Y	D	knockitclose	https://www.youtube.com/channel/UCIpjeWV-BxIV_sQhT443KPg	12 Mins Of Incredible Doubles Tennis (3.0 Men's USTA)
tennis	lTyVR5LnRr8	D	Matthew Chio	https://www.youtube.com/channel/UCdgcJZkmjlqscdkkqshWZ-g	DAVID vs GOLIATH High Level Doubles!!! - 11 UTR / 9 UTR vs. 10 UTR / 9 UTR | USTA 5.0 Ep.1
tennis	lUTREL5vsY8	D	Ishan R	https://www.youtube.com/channel/UCFR0jUjwRW0A2V-Jr0ajBdQ	Weekend Tennis 2: Ishan R and Steven H vs Zeb J and Alex S
tennis	dpTsS7_SeHE	D	Ishan R	https://www.youtube.com/channel/UCFR0jUjwRW0A2V-Jr0ajBdQ	Weekend Tennis 1: Zeb J and Ishan R vs Steven H and Alex S
tennis	kA-Gj9n8eEA	D	Phil Moore	https://www.youtube.com/channel/UCOqcli2-4qT1yEdqeU7vVXQ	February 04 Tennis Racquet Club of Columbus 4.0
tennis	cVotKJZfPvs	D	Dave Kroondyk	https://www.youtube.com/channel/UCV_yR1MDjHIR6nG2vaDtFMA	2013 USTA Nationals - Winsanity
tennis	SiuwHk-h33k	D	iHitWithSteve.com	https://www.youtube.com/channel/UCiiWYtqFEmJlOXdtS9IJkSg	Brian & Daniel vs Kevin & Steve Lam Part 1
tennis	6RUgVstuip4	D	MPTennis	https://www.youtube.com/channel/UCFxH4IPA7ItHuiIsChplWLQ	SNHU VS QUEENS COLLEGE NY 3 Doubles Match 2022
tennis	pW4225TgwH8	S	Ліловий Ананас	https://www.youtube.com/channel/UClydDeSJ48u873EY7EH9v3Q	Tennis Amateur Match / NTRP 3.5 vs 4.5
tennis	6S1xH8HAr_0	S	I Love Tennis VN	https://www.youtube.com/channel/UCwoUT4TTlOO6bQMHe4vwFSQ	Nghia vs Tom Set 2 NTRP (4.0-4.5) VTF (750-800) Tennis Ladder match Saigon- Vietnam Court level view
tennis	uyj3iAuMBCU	S	The Tennis Grind	https://www.youtube.com/channel/UCCcm0yytgYjJG3TmgQiYw2A	NTRP 4.0 Tennis | Singles Practice Match #2
tennis	mqlqrA7Vk7Q	S	Kevin Liang	https://www.youtube.com/channel/UC3QqLFm3tBge_WLf2xKBXcQ	USTA 4.5 Gets Humbled by a Top 5.0
tennis	3J9pryPB2II	S	danill tennis	https://www.youtube.com/channel/UCno8DbEKKDX-vSYEci-Ia_Q	Tennis League Spring 2 May 7, 2025 Full Highlights : W 5-7, 5-2 Ret.
tennis	66kb9uHbDrM	S	Hawaii Tennis Pro	https://www.youtube.com/channel/UCHIzbfBaC5wjdF61Jqrfdvg	NTRP 5.0:  N vs G Jan 1, 2012
tennis	LMPH9am7MPw	S	JanusTennisClub	https://www.youtube.com/channel/UCKXfhJdbzXCmVd03XSAyWKQ	Diretta Tennis JTC vs Trodica 1°parte
"""

MIN_FREE_GB = 30
SPORT_ORDER = {"pickleball": 0, "padel": 1, "tennis": 2}
BOT = ("Sign in to confirm", "HTTP Error 429", "Too Many Requests", "HTTP Error 403")
MAN_COLS = ["sport", "video_id", "url", "title", "channel", "channel_url", "license_verified", "license_verified_at",
            "download_datetime_madrid", "local_path", "duration_s", "resolution", "fps", "vcodec", "filesize_mb",
            "upload_date", "format_note", "attribution_line", "status", "reason"]
STOP = False        # stop everything (SIGTERM)
DL_STOP = False     # stop downloading only (disk guard); already-downloaded videos still get processed
LOCK = threading.RLock()


def now():
    return datetime.datetime.now(ZoneInfo("Europe/Madrid")).strftime("%Y-%m-%d %H:%M:%S %Z")


def log(*a):
    print(now(), "[cc]", *a, flush=True)


def on_term(sig, frm):
    global STOP
    STOP = True
    log("stop requested; finishing at the next checkpoint")


def sources():
    out = []
    for ln in SOURCES.strip().splitlines():
        sp, vid, sd, ch, chu, title = ln.split("\t")
        out.append(dict(sport=sp, id=vid, sd=sd, channel=ch, channel_url=chu, title=title))
    return sorted(out, key=lambda r: SPORT_ORDER[r["sport"]])


class State:
    def __init__(self, base):
        self.base = base
        self.man_path = os.path.join(base, "footage_credits_manifest.csv")
        self.prog_path = os.path.join(base, "progress.tsv")
        self.man = {}
        if os.path.exists(self.man_path):
            with open(self.man_path) as f:
                self.man = {r["video_id"]: r for r in csv.DictReader(f)}
        self.prog = {}
        if os.path.exists(self.prog_path):
            with open(self.prog_path) as f:
                self.prog = {r["video_id"]: r for r in csv.DictReader(f, delimiter="\t")}

    def save(self):
        with LOCK: self._save()

    def _save(self):
        rows = sorted(self.man.values(), key=lambda r: (SPORT_ORDER.get(r["sport"], 9), r["video_id"]))
        tmp = self.man_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, MAN_COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        os.replace(tmp, self.man_path)
        cols = ["sport", "video_id", "download", "prelabel", "duration_s", "resolution", "rallies", "court", "ball_rate", "note", "updated"]
        rows = sorted(self.prog.values(), key=lambda r: (SPORT_ORDER.get(r["sport"], 9), r["video_id"]))
        tmp = self.prog_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        os.replace(tmp, self.prog_path)

    def p(self, r, **kw):
        with LOCK: self._p(r, **kw)

    def _p(self, r, **kw):
        d = self.prog.setdefault(r["id"], dict(sport=r["sport"], video_id=r["id"], download="todo", prelabel="todo"))
        d.update(kw, updated=now()); self._save()


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def disk_ok(base, need_gb=0.0):
    f = free_gb(base)
    if f - need_gb < MIN_FREE_GB:
        log(f"disk guard: {f:.0f} GB free (need {need_gb:.1f} GB + {MIN_FREE_GB} GB reserve); stopping cleanly")
        return False
    return True


def other_heavy_jobs():
    """Other users' heavy ffmpeg/python jobs (> 80% CPU), excluding anything under our BASE and our own tree."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,pcpu=,args="], capture_output=True, text=True).stdout
    except Exception:
        return []
    me, hits = os.getpid(), []
    for ln in out.splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 3: continue
        pid, cpu, args = int(parts[0]), float(parts[1]), parts[2]
        if pid == me or "VamosFootage" in args or "cc_footage" in args: continue
        exe = os.path.basename(args.split()[0]).lower()
        if (exe.startswith("ffmpeg") or exe.startswith("python")) and cpu > 80:
            hits.append(f"{pid} {cpu:.0f}% {args[:80]}")
    return hits


def wait_if_busy():
    if os.environ.get("VAMOS_NO_WAIT"): return
    told = False
    while not STOP:
        h = other_heavy_jobs()
        if not h: return
        if not told: log("waiting: another heavy job is running:", "; ".join(h[:3])); told = True
        time.sleep(300)


def sleep_interruptible(s):
    end = time.time() + s
    while time.time() < end and not STOP:
        time.sleep(min(10, end - time.time()))


# ----------------------------------------------------------------------------------------------- download
def yt(base):
    return [os.path.join(base, "venv", "bin", "yt-dlp"), "--no-playlist", "--sleep-requests", "2", "--retries", "5",
            "--fragment-retries", "10", "--retry-sleep", "exp=5:120", "--extractor-retries", "2"]


# No AV1: Apple M2 has no AV1 hardware decode (VideoToolbox fails on it). VP9/H.264 up to 2160p, higher fps preferred.
FMT = ["-f", "bv*[vcodec!*=av01]+ba/b[vcodec!*=av01]/bv*+ba/b", "-S", "res:2160,fps,acodec:m4a", "--merge-output-format", "mp4"]


def download(base, st, r, between_fn):
    global DL_STOP
    vid = r["id"]; url = f"https://www.youtube.com/watch?v={vid}"
    path = os.path.join(base, r["sport"], f"{vid}.mp4")
    m = st.man.setdefault(vid, {})
    m.update(sport=r["sport"], video_id=vid, url=url, title=r["title"], channel=r["channel"], channel_url=r["channel_url"],
             attribution_line=f'"{r["title"]}" by {r["channel"]} ({url}), licensed CC BY 3.0 (YouTube CC BY); modified: trimmed/analysed')
    if m.get("status") == "downloaded" and os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True); os.makedirs(os.path.join(base, "meta"), exist_ok=True)
    backoff = 600
    for attempt in range(7):
        if STOP: return None
        p = subprocess.run(yt(base) + FMT + ["-J", "--skip-download", url], capture_output=True, text=True, timeout=600)
        if p.returncode == 0 and p.stdout.strip():
            info = json.loads(p.stdout); break
        err = p.stderr
        if any(b in err for b in BOT):
            log(vid, f"YouTube rate-limit/bot-check; backing off {backoff // 60} min (processing pending videos meanwhile)")
            st.p(r, download="backoff")
            t0 = time.time(); between_fn(); sleep_interruptible(max(0, backoff - (time.time() - t0)) + random.randint(0, 120))
            backoff = min(backoff * 2, 3600); continue
        reason = (err.strip().splitlines() or ["?"])[-1][:200]
        m.update(status="skipped", reason="metadata error: " + reason); st.p(r, download="skipped", note=reason[:60]); return None
    else:
        m.update(status="pending", reason="rate-limited; re-run to retry"); st.p(r, download="pending"); return None
    lic = info.get("license") or ""
    keep = ("id", "title", "channel", "channel_url", "uploader", "license", "duration", "upload_date", "width", "height", "fps",
            "vcodec", "format_id", "description", "tags", "was_live", "chapters")
    json.dump({k: info.get(k) for k in keep} | {"license_checked_at": now()}, open(os.path.join(base, "meta", f"{vid}.json"), "w"), indent=1)
    m.update(license_verified=lic, license_verified_at=now(), upload_date=info.get("upload_date", ""),
             channel=info.get("channel") or r["channel"], channel_url=info.get("channel_url") or r["channel_url"], title=info.get("title") or r["title"])
    if not lic.startswith("Creative Commons Attribution"):
        log(vid, "licence did not re-verify as CC BY:", repr(lic))
        m.update(status="skipped", reason=f'licence not CC BY at download time ("{lic}")'); st.p(r, download="skipped", note="licence"); return None
    est = (info.get("filesize") or info.get("filesize_approx") or 0) / 1e9
    if not est:
        est = sum((f.get("filesize") or f.get("filesize_approx") or 0) for f in info.get("requested_formats") or []) / 1e9
    if not disk_ok(base, est * 1.1):
        DL_STOP = True; m.update(status="pending", reason="disk guard"); st.p(r, download="pending", note="disk"); return None
    ij = os.path.join(base, "meta", f"{vid}.full.info.json"); json.dump(info, open(ij, "w"))
    log(vid, f"downloading {info.get('width')}x{info.get('height')}@{info.get('fps')} {info.get('vcodec')} ~{est:.2f} GB")
    st.p(r, download="downloading")
    p = subprocess.run(yt(base) + ["--load-info-json", ij, "--download-archive", os.path.join(base, "meta", "download_archive.txt"),
                                   "-o", os.path.join(base, r["sport"], "%(id)s.%(ext)s"), "--merge-output-format", "mp4"],
                       capture_output=True, text=True, timeout=4 * 3600)
    if p.returncode != 0 or not os.path.exists(path):
        reason = (p.stderr.strip().splitlines() or ["?"])[-1][:200]
        if any(b in p.stderr for b in BOT):
            m.update(status="pending", reason="download rate-limited; re-run"); st.p(r, download="pending"); sleep_interruptible(1800); return None
        m.update(status="skipped", reason="download error: " + reason); st.p(r, download="error", note=reason[:60]); return None
    os.remove(ij)
    m.update(status="downloaded", reason="", download_datetime_madrid=now(), local_path=path, duration_s=info.get("duration"),
             resolution=f"{info.get('width')}x{info.get('height')}", fps=info.get("fps"), vcodec=info.get("vcodec"),
             filesize_mb=round(os.path.getsize(path) / 1e6, 1), format_note=f"yt-dlp format {info.get('format_id')}; source resolution kept up to 2160p")
    st.p(r, download="done", duration_s=info.get("duration"), resolution=m["resolution"])
    log(vid, "downloaded", m["filesize_mb"], "MB")
    return path


# ----------------------------------------------------------------------------------------------- pre-label
COURTS = {  # width, length (m), x lines, y lines (court plane, y=0 near baseline), extra segments ((x0,y0),(x1,y1))
    "tennis": dict(W=10.97, L=23.77, xs=[0, 1.37, 9.60, 10.97], ys=[0, 5.485, 18.285, 23.77],
                   seg=[((5.485, 5.485), (5.485, 18.285))]),
    "padel": dict(W=10.0, L=20.0, xs=[0, 10.0], ys=[0, 3.05, 16.95, 20.0], seg=[((5, 3.05), (5, 16.95))]),
    "pickleball": dict(W=6.10, L=13.41, xs=[0, 6.10], ys=[0, 4.57, 8.84, 13.41], seg=[((3.05, 0), (3.05, 4.57)), ((3.05, 8.84), (3.05, 13.41))]),
}


def ff(): return os.environ.get("OCV_FFMPEG", "ffmpeg")
def ffp(): return os.environ.get("OCV_FFPROBE", "ffprobe")


def probe(path):
    j = json.loads(subprocess.run([ffp(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,r_frame_rate:format=duration",
                                   "-of", "json", path], capture_output=True, text=True).stdout)
    s = j["streams"][0]; a, b = s["r_frame_rate"].split("/")
    PROBE_CODEC[path] = s.get("codec_name", "")
    return int(s["width"]), int(s["height"]), float(a) / float(b), float(j["format"]["duration"])


PROBE_CODEC = {}
HW_CODECS = {"h264", "hevc", "vp9"}      # VideoToolbox on Apple Silicon; AV1 etc. -> software (dav1d)


def grab(path, t, w=640):
    import cv2  # noqa
    p = subprocess.run([ff(), "-nostdin", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-vf", f"scale={w}:-2",
                        "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    if not p.stdout: return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def contact_sheet(path, dur, out_jpg):
    import cv2
    ts = list(np.arange(30, max(31, dur - 5), 60.0)) or [dur / 2]
    if len(ts) > 120: ts = list(np.linspace(30, dur - 5, 120))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(N_WORKERS) as ex:
        got = list(ex.map(lambda t: (t, grab(path, t, 480)), ts))
    frames = [(t, f) for t, f in got if f is not None]
    if not frames: return [], {}
    h, w = frames[0][1].shape[:2]; cols = 8; rows = (len(frames) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, (t, f) in enumerate(frames):
        f = cv2.resize(f, (w, h)); y, x = (i // cols) * h, (i % cols) * w; sheet[y:y + h, x:x + w] = f
        cv2.putText(sheet, f"{int(t // 60)}:{int(t % 60):02d}", (x + 6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(out_jpg, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # automatic checks
    g = np.stack([cv2.cvtColor(cv2.resize(f, (w, h)), cv2.COLOR_BGR2GRAY).astype(np.float32) for _, f in frames])
    med = np.median(g, 0)
    def ncc(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))
    sims = [ncc(x, med) for x in g]
    std = g.std(0); edges = cv2.Canny(med.astype(np.uint8), 60, 160) > 0
    band = np.zeros_like(edges); bh = h // 6; band[:bh] = True; band[-bh:] = True
    static_edges = edges & (std < 4) & band
    chk = dict(n_samples=len(frames), sim_to_median_p10=round(float(np.percentile(sims, 10)), 3),
               frac_samples_off_view=round(float(np.mean(np.array(sims) < 0.6)), 3),
               fixed_camera=bool(np.percentile(sims, 10) > 0.6 and np.mean(np.array(sims) < 0.6) < 0.15),
               overlay_suspect=bool(static_edges.mean() > 0.004), overlay_edge_density=round(float(static_edges.mean()), 4))
    return frames, chk


def music_hint(sr, x):
    import scipy.signal as ss
    if len(x) < sr * 30: return dict(music_suspect=None)
    f, t, S = ss.spectrogram(x[: sr * 1800], fs=sr, nperseg=1024, noverlap=0)
    S = S[(f > 200) & (f < 4000)] + 1e-12
    flat = np.exp(np.log(S).mean(0)) / S.mean(0)
    loud = 10 * np.log10(S.mean(0))
    active = loud > np.percentile(loud, 20) + 6
    tonal = float(np.mean(flat[active] < 0.05)) if active.any() else 0.0
    return dict(music_suspect=bool(tonal > 0.5), tonal_frac=round(tonal, 3))


def detect_court(frames, sport):
    """frames: list of BGR (any size, same). -> dict(detected, corners_px (near-L, near-R, far-R, far-L, in frame px of the
    given frames), H_img2court, line_support, reproj_err_px) or detected False."""
    import cv2
    model = COURTS[sport]
    img = np.median(np.stack(frames), 0).astype(np.uint8)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    mask = ((th > 20) & (hsv[..., 1] < 130) & (hsv[..., 2] > 110)).astype(np.uint8) * 255   # clay lines are tinted (S~90)
    mask[: int(h * 0.05)] = 0
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, 60, minLineLength=max(30, w // 25), maxLineGap=12)
    if segs is None or len(segs) < 4: return dict(detected=False, reason="too few line segments")
    L = []
    for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(float):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        a, b = y1 - y2, x2 - x1; n = np.hypot(a, b); a, b = a / n, b / n; c = -(a * x1 + b * y1)
        L.append((ang, a, b, c, np.hypot(x2 - x1, y2 - y1), (x1 + x2) / 2, (y1 + y2) / 2))
    def cluster(lines, key):
        lines = sorted(lines, key=key); out = []
        for l in lines:
            if out and abs(key(l) - key(out[-1][-1])) < 10 and abs(((l[0] - out[-1][-1][0] + 90) % 180) - 90) < 4: out[-1].append(l)
            else: out.append([l])
        res = []
        for g in out:
            wts = np.array([x[4] for x in g]); i = int(np.argmax(wts)); res.append((g[i], float(wts.sum())))
        return [l for l, _ in sorted(res, key=lambda z: -z[1])]
    horiz = [l for l in L if min(l[0], 180 - l[0]) < 15]
    slant = [l for l in L if 15 <= min(l[0], 180 - l[0]) <= 89]
    H_ = cluster(horiz, key=lambda l: l[6])[:6]
    S_ = cluster(slant, key=lambda l: l[5] + (h - l[6]) * (np.cos(np.radians(l[0]))) / max(1e-3, abs(np.sin(np.radians(l[0])))))[:6]
    if len(H_) < 2 or len(S_) < 2: return dict(detected=False, reason=f"lines: {len(H_)} horizontal, {len(S_)} slanted")
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    W_, Lm = model["W"], model["L"]
    mlines = [((x, 0), (x, Lm)) for x in model["xs"]] + [((0, y), (W_, y)) for y in model["ys"]] + model["seg"]
    samp = []
    for (x0, y0), (x1, y1) in mlines:
        for t in np.linspace(0, 1, 40): samp.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    samp = np.array(samp, np.float32)
    full = np.array([(0, 0), (W_, 0), (W_, Lm), (0, Lm)], np.float32)

    def inter(l1, l2):
        a1, b1, c1 = l1[1:4]; a2, b2, c2 = l2[1:4]; d = a1 * b2 - a2 * b1
        if abs(d) < 1e-6: return None
        return ((b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d)
    best = None
    ypairs = [(model["ys"][i], model["ys"][j]) for i in range(len(model["ys"])) for j in range(i + 1, len(model["ys"]))]
    xpairs = [(model["xs"][i], model["xs"][j]) for i in range(len(model["xs"])) for j in range(i + 1, len(model["xs"]))]
    if sport != "tennis": xpairs = [(0, W_)]
    for i in range(len(H_)):
        for j in range(len(H_)):
            if i == j or H_[i][6] <= H_[j][6]: continue   # i = lower in image = nearer
            for k in range(len(S_)):
                for m in range(len(S_)):
                    if k == m: continue
                    pts = [inter(H_[i], S_[k]), inter(H_[i], S_[m]), inter(H_[j], S_[m]), inter(H_[j], S_[k])]
                    if any(p is None for p in pts): continue
                    pts = np.array(pts, np.float32)
                    if pts[0, 0] >= pts[1, 0] or pts[3, 0] >= pts[2, 0]: continue
                    if np.abs(pts).max() > 4 * max(w, h): continue
                    for (ya, yb) in ypairs:
                        for (xa, xb) in xpairs:
                            dst = np.array([(xa, ya), (xb, ya), (xb, yb), (xa, yb)], np.float32)
                            try: Hc2i = cv2.getPerspectiveTransform(dst, pts)
                            except cv2.error: continue
                            cc = cv2.perspectiveTransform(full[None], Hc2i)[0]
                            if cv2.contourArea(cc) < 0.05 * w * h or not cv2.isContourConvex(cc.astype(np.float32)): continue
                            if (cc[1, 0] - cc[0, 0]) < 0.9 * (cc[2, 0] - cc[3, 0]) or max(cc[2, 1], cc[3, 1]) > min(cc[0, 1], cc[1, 1]) - 0.05 * h: continue
                            pr = cv2.perspectiveTransform(samp[None], Hc2i)[0]
                            ok = (pr[:, 0] >= 0) & (pr[:, 0] < w) & (pr[:, 1] >= 0) & (pr[:, 1] < h)
                            if ok.mean() < 0.6: continue
                            d = np.full(len(pr), 20.0); d[ok] = dist[pr[ok, 1].astype(int), pr[ok, 0].astype(int)]
                            good = (d < 3).reshape(-1, 40); P = pr.reshape(-1, 40, 2)
                            step = np.hypot(*np.diff(P, axis=1).transpose(2, 0, 1))
                            expl = float((step * (good[:, 1:] & good[:, :-1])).sum())   # image px of model lines on detected lines
                            sup = float(np.mean(d[ok] < 3)); err = float(np.median(np.minimum(d[ok], 20)))
                            if best is None or expl > best[3]: best = (sup, err, Hc2i, expl)
    if best is None: return dict(detected=False, reason="no valid court hypothesis")
    sup, err, Hc2i, expl = best
    corners = cv2.perspectiveTransform(np.array([[(0, 0), (W_, 0), (W_, Lm), (0, Lm)]], np.float32), Hc2i)[0]
    Hi2c = np.linalg.inv(Hc2i); Hi2c /= Hi2c[2, 2]
    vis = img.copy()
    pr = cv2.perspectiveTransform(samp[None], Hc2i)[0]
    for (px, py) in pr:
        if 0 <= px < w and 0 <= py < h: cv2.circle(vis, (int(px), int(py)), 2, (0, 0, 255), -1)
    COURT_VIS[0] = vis
    return dict(detected=bool(sup >= 0.65), weak=bool(0.5 <= sup < 0.65), line_support=round(sup, 3), reproj_err_px_median=round(err, 2),
                corners_px=corners.round(1).tolist(), H_img2court=Hi2c.tolist(), court_m=[W_, Lm],
                note="corners: near-left, near-right, far-right, far-left; line_support = share of in-image model-line samples within 3 px of a detected line; hypothesis chosen by explained line length")


COURT_VIS = [None]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def track_players(boxes_per_frame, fps):
    tracks, live, nid = [], [], 0
    for f, bb in enumerate(boxes_per_frame):
        dets = [(x0, y0, x1, y1) for (x0, x1, _a, y0, y1) in bb]
        used = set(); nl = []
        for tr in live:
            last = tr["boxes"][-1][1:]
            cand = [(iou(last, d), n) for n, d in enumerate(dets) if n not in used]
            sc, n = max(cand, default=(0, -1))
            if sc < 0.2 and n >= 0:
                cx, cy = (last[0] + last[2]) / 2, (last[1] + last[3]) / 2
                dd = [(np.hypot((d[0] + d[2]) / 2 - cx, (d[1] + d[3]) / 2 - cy), n) for n, d in enumerate(dets) if n not in used]
                dmin, n2 = min(dd, default=(1e9, -1))
                sc, n = (0.2, n2) if dmin < 40 else (0, -1)
            if sc >= 0.2:
                used.add(n); tr["boxes"].append((f, *dets[n])); tr["miss"] = 0; nl.append(tr)
            elif tr["miss"] < int(fps): tr["miss"] += 1; nl.append(tr)
        for n, d in enumerate(dets):
            if n not in used:
                tr = dict(id=nid, boxes=[(f, *d)], miss=0); nid += 1; tracks.append(tr); nl.append(tr)
        live = nl
    return [t for t in tracks if len(t["boxes"]) >= max(3, fps // 2)]


def bounces_from(pos, ts, hits, fps):
    out = []
    y = pos[:, 1]
    for k in range(2, len(y) - 2):
        if np.isnan(y[k - 2:k + 3]).any(): continue
        if y[k] >= y[k - 1] and y[k] >= y[k + 1] and y[k] - y[k - 2] > 1.5 and y[k] - y[k + 2] > 1.5:
            t = ts[k]
            if len(hits) and np.min(np.abs(np.asarray(hits) - t)) <= 0.08: continue
            if out and t - out[-1]["t"] < 0.4: continue
            out.append(dict(t=round(float(t), 3), x=float(pos[k, 0]), y=float(pos[k, 1])))
    return out


# ----------------------------------------------------------------------------------------------- rally workers
def perf_cores():
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            v = int(subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip())
            if v > 0: return v
        except Exception:
            pass
    return max(1, (os.cpu_count() or 2) // 2)


CHUNK_S = float(os.environ.get("VAMOS_CHUNK_S", "15"))
N_WORKERS = int(os.environ.get("VAMOS_WORKERS", "0")) or perf_cores()
POOL = None
TN_CFG = dict(use=False, weights=None, device=None, why="not initialised")
_W = {}


def _init_worker(tn_cfg):
    import cv2
    cv2.setNumThreads(1)
    _W["tn"] = None; _W["engine"] = False
    try:
        from opencourtvision.ball.tracker import BallEngine  # noqa: F401
        _W["engine"] = True
    except Exception:
        pass
    if tn_cfg.get("use") and _W["engine"]:
        try:
            from opencourtvision.ball.tracknet3 import TrackNetV3
            _W["tn"] = TrackNetV3(weights=tn_cfg["weights"], device=tn_cfg.get("device"), threads=1)
        except Exception as e:
            print(now(), "[worker] TrackNetV3 unavailable:", repr(e), flush=True)


def decode_frames(path, a, b, aw, ah, fps, hw_ok=True):
    """BGR frames of path[a:b] at aw x ah, fps. VideoToolbox hardware decode on macOS for H.264/HEVC/VP9, software otherwise."""
    hw = ["-hwaccel", "videotoolbox"] if (sys.platform == "darwin" and hw_ok) else []
    cmd = [ff(), "-nostdin", "-loglevel", "error", *hw, "-threads", "2", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", path,
           "-vf", f"fps={fps},scale={aw}:{ah}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = aw * ah * 3; got = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n: break
            got += 1
            yield np.frombuffer(buf, np.uint8).reshape(ah, aw, 3)
    finally:
        p.stdout.close(); p.wait()
    if got == 0 and hw:                 # hardware decode refused this stream -> software
        yield from decode_frames(path, a, b, aw, ah, fps, hw_ok=False)


def player_blobs(g_small, prev2_small, ah):
    """Moving player-sized blobs at half resolution -> [(x0, x1, area, y0, y1)] in analysis px (same tuple layout as reel.ball)."""
    import cv2
    if prev2_small is None: return []
    mm = (cv2.absdiff(g_small, prev2_small) > 18).astype(np.uint8)
    big = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(big, 8)
    hs = g_small.shape[0]
    return [(2.0 * st[k][0], 2.0 * (st[k][0] + st[k][2]), int(st[k][4]) * 4, 2.0 * st[k][1], 2.0 * (st[k][1] + st[k][3]))
            for k in range(1, n) if 62 <= st[k][4] <= 10000 and st[k][1] > hs * 0.12]


def rally_job(j):
    import cv2
    t0 = time.perf_counter()
    boxes = []
    g1 = g2 = None

    def frames():
        nonlocal g1, g2
        for f in decode_frames(j["path"], j["a"], j["b"], j["aw"], j["ah"], j["fps"], j.get("hw", True)):
            g = cv2.cvtColor(cv2.resize(f, (j["aw"] // 2, j["ah"] // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            boxes.append(player_blobs(g, g2, j["ah"])); g2, g1 = g1, g
            yield f
    if _W.get("engine"):
        from opencourtvision.ball.tracker import BallEngine
        gate = None
        if j["Hc2i"] is not None:
            try:
                from opencourtvision.ball.geometry import CourtGate
                gate = CourtGate(np.array(j["Hc2i"]), width=j["court_m"], length=j["court_l"], image_size=(j["aw"], j["ah"]))
            except Exception:
                gate = None
        eng = BallEngine(gate=gate, tracknet=_W.get("tn"), fps=j["fps"], t0=j["a"], hits=j["hits"],
                         reacquire="tracknet" if _W.get("tn") is not None else "hsv_full")
        res = eng.run(frames())
        pos, raw = res["pos"], res["raw"]
        bnc = [dict(t=float(b["t"]), x=float(b["x"]), y=float(b["y"])) for b in res["bounces"] if b.get("x") is not None]
        tracker = "engine-v2 BallEngine + TrackNetV3 (" + str(getattr(_W.get("tn"), "device", "")) + ")" if _W.get("tn") is not None else "engine-v2 BallEngine (HSV re-acquire)"
    else:
        from opencourtvision.reel.ball import detect, track
        cands, _act, _dbg, _small, bx = detect(frames(), j["a"], j["fps"], with_boxes=True)
        pos = track(cands, j["aw"], j["ah"]) if cands else np.zeros((0, 2)); raw = pos
        ts = j["a"] + np.arange(len(pos)) / j["fps"]
        bnc = bounces_from(pos, ts, j["hits"], j["fps"]); tracker = "reel.ball HSV"
    nfr = len(boxes)
    tr = track_players(boxes, j["fps"])
    rate = float(np.mean(~np.isnan(pos[:, 0]))) if len(pos) else 0.0
    raw_rate = float(np.mean(~np.isnan(raw[:, 0]))) if len(raw) else 0.0
    return dict(n=j["n"], c=j.get("c", 0), a=j["a"], pos=np.round(pos, 1).tolist(), rate=rate, raw_rate=raw_rate, bounces=bnc,
                tracks=[dict(id=t["id"], boxes=t["boxes"]) for t in tr], frames=nfr,
                players_seen=float(np.mean([len(b) for b in boxes])) if boxes else 0.0, secs=time.perf_counter() - t0, tracker=tracker)


def get_pool():
    global POOL
    if POOL is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        POOL = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp.get_context("spawn"), initializer=_init_worker, initargs=(TN_CFG,))
    return POOL


def run_rallies(jobs):
    """Run rally jobs on the pool, in order of submission; None if a stop was requested."""
    pool = get_pool(); futs = []; it = iter(jobs); k = 0
    pending = set()
    while True:
        while len(pending) < 2 * N_WORKERS:
            j = next(it, None)
            if j is None: break
            if STOP: break
            if k % 20 == 0: wait_if_busy()
            f = pool.submit(rally_job, j); futs.append(f); pending.add(f); k += 1
        if not pending: break
        import concurrent.futures as cf
        _d, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
        if STOP and not pending: break
    if STOP and k < len(jobs): return None
    return [f.result() for f in futs]


def init_tracknet(base):
    """Decide once whether TrackNetV3 runs (torch + MPS/CPU + verified weights)."""
    try:
        # Off by default: on the M2 Pro, TrackNetV3 (badminton weights, not fine-tuned) re-acquire is GPU-bound at ~75 frames/s total
        # across all workers (vs ~270 frames/s per worker for HSV re-acquire) and did not raise the live-ball rate on pickleball.
        # Opt in with VAMOS_TN=1 once fine-tuned weights exist.
        if os.environ.get("VAMOS_NO_TN") or not os.environ.get("VAMOS_TN"): raise RuntimeError("off by default (set VAMOS_TN=1)")
        import torch

        from opencourtvision.ball.tracknet3 import fetch_weights, pick_device
        w = fetch_weights()
        TN_CFG.update(use=True, weights=w, device=pick_device(torch), why="ok")
    except Exception as e:
        TN_CFG.update(use=False, why=repr(e)[:200])
    log("ball tracker:", f"TrackNetV3 on {TN_CFG['device']}" if TN_CFG["use"] else f"engine-v2 HSV/repo HSV (TrackNetV3 off: {TN_CFG['why']})",
        f"| {N_WORKERS} rally workers (performance cores)")


def prelabel(base, r, path, args):
    import cv2

    from opencourtvision.detection.audio import detect_hits, extract_wav, group_rallies, loudness, read_wav
    from opencourtvision.detection.match_start import warmup_cutoff
    vid, sport = r["id"], r["sport"]
    out = os.path.join(base, "prelabels", sport, vid); os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "qa.json")): return json.load(open(os.path.join(out, "qa.json")))
    t_start = time.time(); notes = []
    W, H, fps_src, dur = probe(path)
    qa = dict(video_id=vid, sport=sport, singles_doubles=r["sd"], channel=r["channel"], resolution=f"{W}x{H}", fps=round(fps_src, 2),
              duration_s=round(dur, 1), vertical=H > W)
    # 1 contact sheet + checks
    frames, chk = contact_sheet(path, dur, os.path.join(out, "contact_sheet.jpg")); qa.update(chk)
    # 2 audio -> hits/rallies -> warm-up trim
    wav = os.path.join(out, "audio16k.wav")
    extract_wav(path, wav); sr, x = read_wav(wav)
    qa.update(music_hint(sr, x))
    le, bg, lb = loudness(sr, x); t_hits, amp = detect_hits(le, bg); R = group_rallies(t_hits, amp)
    os.remove(wav)
    if not R:
        qa.update(usable=False, failure_notes="no audio hits detected (silent/music track?)"); json.dump(qa, open(os.path.join(out, "qa.json"), "w"), indent=1); return qa
    try:
        start, ev = warmup_cutoff(R)
    except Exception as e:
        start, ev = R[0]["t0"], {"method": "first hit", "error": str(e)}
    start = max(0.0, float(start) - 4.0)
    rallies = [g for g in R if g["t0"] >= start and g["hits"] >= 2]
    end = min(dur, (rallies[-1]["t1"] if rallies else R[-1]["t1"]) + 3.0)
    json.dump(dict(video=os.path.basename(path), play_start_s=round(start, 2), play_end_s=round(end, 2), lead_in_trimmed_s=round(start, 2),
                   method="audio hits + warmup_cutoff (tuned on tennis doubles); start = first match rally - 4 s pre-roll", evidence=ev,
                   ffmpeg_trim_hint=f"-ss {start:.2f} -to {end:.2f} (stream copy snaps to keyframes)"), open(os.path.join(out, "trim.json"), "w"), indent=1, default=str)
    kind = video_kind([g for g in R if g["t0"] >= start], start, dur, qa)
    qa.update(kind)
    reasons = []
    if qa["vertical"]: reasons.append("vertical video")
    if not qa.get("fixed_camera", True): reasons.append("moving camera / edited (contact-sheet frames disagree)")
    if qa.get("music_suspect"): notes.append("music suspected on audio track (audio-hit contacts unreliable)")
    if qa.get("overlay_suspect"): notes.append("static burned-in overlay suspected in top/bottom band (mask for training)")
    # 3 court (from play-time samples)
    tsel = np.linspace(start + 5, max(start + 6, end - 5), 9) if end - start > 20 else np.linspace(1, max(2, dur - 1), 9)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(N_WORKERS) as ex:
        sel = [f for f in ex.map(lambda t: grab(path, t, 1280), tsel) if f is not None]
    court = detect_court(sel, sport) if sel else dict(detected=False, reason="no frames")
    sx = W / sel[0].shape[1] if sel else 1.0
    if court.get("corners_px"):
        court["corners_px"] = (np.array(court["corners_px"]) * sx).round(1).tolist()
        Hs = np.array(court["H_img2court"]) @ np.diag([1 / sx, 1 / sx, 1]); court["H_img2court"] = (Hs / Hs[2, 2]).tolist()
    if COURT_VIS[0] is not None: cv2.imwrite(os.path.join(out, "court_check.jpg"), COURT_VIS[0]); COURT_VIS[0] = None
    qa.update(court_detected=court.get("detected", False), court_line_support=court.get("line_support"), court_reproj_err_px=court.get("reproj_err_px_median"))
    if reasons:
        qa.update(usable=False, failure_notes="; ".join(reasons + notes), rally_count=len(rallies))
        json.dump(dict(video_id=vid, court=court, rallies=[], video_kind=kind["video_kind"], label_mode=kind["label_mode"]),
                  open(os.path.join(out, "prelabels.json"), "w"), indent=1)
        json.dump(qa, open(os.path.join(out, "qa.json"), "w"), indent=1); return qa
    # 4 ball + players per rally window: rallies run in parallel on the performance cores (one process each, VideoToolbox
    #   decode at 1280 px / <= 30 fps, engine-v2 BallEngine with TrackNetV3 re-acquire on MPS when available)
    aw = 1280; ah = int(round(aw * H / W / 2) * 2); scale = W / aw
    fps_a = 30 if fps_src > 35 else max(24, round(fps_src))
    Hc = np.array(court["H_img2court"]) if court.get("detected") else None
    Hc2i_a = None
    if Hc is not None:
        Hc2i_a = (np.diag([1 / scale, 1 / scale, 1.0]) @ np.linalg.inv(Hc)).tolist()
    def to_court(px):
        if Hc is None: return None
        p = Hc @ np.array([px[0], px[1], 1.0]); return [round(float(p[0] / p[2]), 2), round(float(p[1] / p[2]), 2)]
    # rally windows are cut into <= CHUNK_S pieces so every performance core stays busy even when audio segmentation
    # merges a whole game into one long "rally" (common in pickleball); tracks restart at piece boundaries
    jobs = []
    for n, g in enumerate(rallies):
        a0, b0 = max(0, g["t0"] - 1.0), min(dur, g["t1"] + 1.5)
        cuts = list(np.arange(a0, b0, CHUNK_S)) + [b0]
        for c, (a, b) in enumerate(zip(cuts, cuts[1:])):
            if b - a < 0.5: continue
            jobs.append(dict(n=n + 1, c=c, path=path, a=float(a), b=float(b), hits=[t for t in g["times"] if a - 0.1 <= t <= b + 0.1], aw=aw, ah=ah,
                             fps=fps_a, Hc2i=Hc2i_a, court_m=COURTS[sport]["W"], court_l=COURTS[sport]["L"], hw=PROBE_CODEC.get(path) in HW_CODECS))
    qa["decode"] = f"{PROBE_CODEC.get(path)} -> " + ("VideoToolbox" if sys.platform == "darwin" and PROBE_CODEC.get(path) in HW_CODECS else "software")
    results = run_rallies(jobs)
    if results is None: return None
    if results and sum(x["frames"] for x in results) == 0:
        raise RuntimeError(f"decoder returned no frames ({PROBE_CODEC.get(path)}); check ffmpeg decode support")
    ball_rows, player_rows, out_r, det_rates, raw_rates, cont, players_seen = [], [], [], [], [], [], []
    frames_done = sum(x["frames"] for x in results); worker_s = sum(x["secs"] for x in results)
    by_rally = {}
    for res in results: by_rally.setdefault(res["n"], []).append(res)
    for n0, g in enumerate(rallies):
        n = n0; pieces = by_rally.get(n0 + 1, [])
        bnc_all, big_ids, fr_tot, live_tot = [], [], 0, 0.0
        for res in pieces:
            a = res["a"]; pos = np.array(res["pos"], float).reshape(-1, 2); ts = a + np.arange(len(pos)) / fps_a
            fr_tot += len(pos); live_tot += res["rate"] * len(pos); raw_rates.append(res["raw_rate"])
            bnc = [dict(t=round(bb["t"], 3), x=round(bb["x"] * scale, 1), y=round(bb["y"] * scale, 1)) for bb in res["bounces"]]
            for bb in bnc: bb["court_m"] = to_court((bb["x"], bb["y"]))
            bnc_all += bnc
            tr = [dict(t, id=(n + 1) * 10000 + res["c"] * 100 + t["id"]) for t in res["tracks"]]; nfr = max(1, res["frames"])
            big = sorted(tr, key=lambda t: -len(t["boxes"]))[: (2 if r["sd"] == "S" else 4)]
            cont.append(float(np.mean([len(t["boxes"]) / nfr for t in big])) if big else 0.0); big_ids += [t["id"] for t in big]
            players_seen.append(res["players_seen"])
            for k in range(len(pos)):
                if not np.isnan(pos[k, 0]): ball_rows.append((n + 1, round(ts[k], 3), round(pos[k, 0] * scale, 1), round(pos[k, 1] * scale, 1)))
            for t in tr:
                for (f, x0, y0, x1, y1) in t["boxes"][:: max(1, fps_a // 10)]:
                    foot = ((x0 + x1) / 2 * scale, y1 * scale); cm = to_court(foot)
                    player_rows.append((n + 1, t["id"], round(a + f / fps_a, 3), round(x0 * scale), round(y0 * scale), round(x1 * scale), round(y1 * scale),
                                        *(cm if cm else ("", ""))))
        rate = live_tot / fr_tot if fr_tot else 0.0; det_rates.append(rate)
        out_r.append(dict(rally=n + 1, start_s=round(g["t0"] - 0.5, 2), end_s=round(g["t1"] + 1.0, 2), n_contacts=g["hits"],
                          contacts=[dict(t=round(t, 3), source="audio", verified=False) for t in g["times"]],
                          bounces=[dict(bb, source="ball_track", verified=False) for bb in bnc_all], ball_detect_rate=round(rate, 3),
                          player_tracks=big_ids))
    tracker = results[0]["tracker"] if results else "none"
    with open(os.path.join(out, "ball.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rally", "t_s", "x_px", "y_px"]); w.writerows(ball_rows)
    with open(os.path.join(out, "players.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rally", "track_id", "t_s", "x0", "y0", "x1", "y1", "court_x_m", "court_y_m"]); w.writerows(player_rows)
    json.dump(dict(schema="vamos-prelabel-v0", video_id=vid, sport=sport, video=os.path.relpath(path, base), width=W, height=H, fps=fps_src,
                   analysis=dict(width=aw, height=ah, fps=fps_a), trim="trim.json", court=court, rallies=out_r,
                   video_kind=kind["video_kind"], label_mode=kind["label_mode"], mode_reasons=kind["mode_reasons"],
                   files=dict(ball="ball.csv", players="players.csv", contact_sheet="contact_sheet.jpg", court_check="court_check.jpg"),
                   note="machine pre-labels, all unverified: contacts = audio onsets; bounces = image-y reversals away from contacts; "
                        "players = moving-blob IoU tracks (static players are missed); ball = engine-v2 BallEngine live-ball positions (TrackNetV3 "
                        "re-acquire, badminton weights) or reel.ball HSV fallback; coords in source px; court_m via ground homography"),
              open(os.path.join(out, "prelabels.json"), "w"), indent=1)
    lens = [g["t1"] - g["t0"] for g in rallies]
    exp_p = 2 if r["sd"] == "S" else 4
    qa.update(ball_tracker=tracker, ball_raw_detect_rate=round(float(np.mean(raw_rates)), 3) if raw_rates else 0,
              analysis_frames=frames_done, analysis_fps_per_worker=round(frames_done / worker_s, 1) if worker_s else None, workers=N_WORKERS)
    qa.update(usable=True, rally_count=len(rallies), mean_rally_len_s=round(float(np.mean(lens)), 2) if lens else 0,
              mean_contacts_per_rally=round(float(np.mean([g["hits"] for g in rallies])), 2) if rallies else 0,
              ball_detect_rate=round(float(np.mean(det_rates)), 3) if det_rates else 0,
              player_track_continuity=round(float(np.mean(cont)), 3) if cont else 0,
              mean_moving_players_per_frame=round(float(np.mean(players_seen)), 2) if players_seen else 0, expected_players=exp_p,
              play_share=round(sum(lens) / dur, 3), processing_s=round(time.time() - t_start))
    if not court.get("detected"): notes.append(f"court not detected ({court.get('reason') or 'line support ' + str(court.get('line_support'))})")
    if qa["ball_detect_rate"] < 0.3: notes.append(f"low ball detection ({tracker}; ball colour/size/blur)")
    if qa["rally_count"] < 5: notes.append("few rallies found (audio hits weak?)")
    qa["failure_notes"] = "; ".join(notes)
    json.dump(qa, open(os.path.join(out, "qa.json"), "w"), indent=1)
    return qa


FULL_MATCH_MIN_S = 20 * 60     # shorter uploads are tagged partial even when the rally pattern looks like match play


def video_kind(rallies, start, dur, qa=None):
    """full_match vs partial_or_practice for the label pass (pure; audio rally groups only, no decode). Uses the app's
    mode suggestion (opencourtvision.editloop.modes.suggest_mode): a video it would not score as a match, an upload
    shorter than FULL_MATCH_MIN_S, or an edited / moving-camera video is partial_or_practice and gets label_mode
    "rallies" (rally boundaries only, no point winners or score)."""
    from opencourtvision.editloop.modes import suggest_mode
    sug = suggest_mode(rallies, start, dur)
    reasons = list(sug["reasons"])
    if dur is not None and dur < FULL_MATCH_MIN_S: reasons.append(f"short upload ({dur / 60:.0f} min < {FULL_MATCH_MIN_S // 60} min)")
    if qa and qa.get("fixed_camera") is False: reasons.append("edited / moving camera (highlight upload?)")
    full = not reasons
    return dict(video_kind="full_match" if full else "partial_or_practice", label_mode="match" if full else "rallies",
                mode_reasons=reasons, mode_evidence=sug["evidence"])


def retag(base):
    """Add video_kind / label_mode to existing pre-labels from their saved audio contacts (no decode, cheap)."""
    n = 0
    for sp in SPORT_ORDER:
        d = os.path.join(base, "prelabels", sp)
        if not os.path.isdir(d): continue
        for vid in sorted(os.listdir(d)):
            qp, pp, tp = (os.path.join(d, vid, f) for f in ("qa.json", "prelabels.json", "trim.json"))
            if not (os.path.exists(qp) and os.path.exists(pp)): continue
            qa, pl = json.load(open(qp)), json.load(open(pp))
            start = json.load(open(tp)).get("play_start_s", 0.0) if os.path.exists(tp) else 0.0
            R, prev = [], None
            for g in pl.get("rallies", []):
                ts = [c["t"] for c in g.get("contacts", [])]
                if not ts: continue
                R.append(dict(t0=ts[0], t1=ts[-1], hits=len(ts), pause_before=(ts[0] - prev) if prev is not None else ts[0])); prev = ts[-1]
            k = video_kind(R, start, qa.get("duration_s"), qa)
            qa.update(k); pl.update({x: k[x] for x in ("video_kind", "label_mode", "mode_reasons")})
            json.dump(qa, open(qp, "w"), indent=1); json.dump(pl, open(pp, "w"), indent=1); n += 1
    write_qa_summary(base)
    return n


def write_qa_summary(base):
    rows = []
    for sp in SPORT_ORDER:
        d = os.path.join(base, "prelabels", sp)
        if not os.path.isdir(d): continue
        for vid in sorted(os.listdir(d)):
            q = os.path.join(d, vid, "qa.json")
            if os.path.exists(q): rows.append(json.load(open(q)))
    if not rows: return
    cols = ["sport", "video_id", "channel", "singles_doubles", "resolution", "fps", "duration_s", "usable", "fixed_camera", "vertical",
            "overlay_suspect", "music_suspect", "court_detected", "court_line_support", "court_reproj_err_px", "rally_count", "mean_rally_len_s",
            "mean_contacts_per_rally", "ball_detect_rate", "player_track_continuity", "mean_moving_players_per_frame", "play_share",
            "processing_s", "video_kind", "label_mode", "failure_notes"]
    with open(os.path.join(base, "prelabels", "qa_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/VamosFootage"))
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-prelabel", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prelabel-file", nargs=2, metavar=("SPORT", "VIDEO"), help="pre-label one local file (test)")
    ap.add_argument("--retag", action="store_true", help="only add video_kind / label_mode to existing pre-labels (no decode)")
    args = ap.parse_args()
    if args.retag:
        print(f"retagged {retag(args.base)} videos"); return
    base = args.base; os.makedirs(base, exist_ok=True)
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    open(os.path.join(base, "worker.pid"), "w").write(str(os.getpid()))
    if args.prelabel_file:
        init_tracknet(base)
        sp, p = args.prelabel_file; r = dict(id=os.path.splitext(os.path.basename(p))[0], sport=sp, sd="D", channel="local")
        t0 = time.time(); qa = prelabel(base, r, p, args)
        if POOL is not None: POOL.shutdown(wait=True)
        qa["wall_s"] = round(time.time() - t0, 1); qa["s_per_video_min"] = round(qa["wall_s"] / (qa["duration_s"] / 60), 2)
        print(json.dumps(qa, indent=1)); return
    src = sources(); st = State(base)
    if args.limit: src = src[: args.limit]
    for r in src: st.prog.setdefault(r["id"], dict(sport=r["sport"], video_id=r["id"], download="todo", prelabel="todo"))
    st.save()
    log(f"{len(src)} videos in plan; {free_gb(base):.0f} GB free; ffmpeg={ff()}")
    if args.dry_run:
        import cv2  # noqa: F401
        import scipy  # noqa: F401

        import opencourtvision  # noqa: F401
        from opencourtvision.reel.ball import detect  # noqa: F401
        p = subprocess.run(yt(base) + ["--version"], capture_output=True, text=True)
        log("dry run OK: imports fine; yt-dlp", p.stdout.strip(), "; first videos:", ", ".join(f"{r['sport']}/{r['id']}" for r in src[:5])); return

    for r in src:   # one-off fix-up: e6dbc0e labelled AV1 files with zero decoded frames -> redo those (bad results, not finished)
        qp = os.path.join(base, "prelabels", r["sport"], r["id"], "qa.json")
        if os.path.exists(qp):
            q0 = json.load(open(qp))
            if q0.get("usable") and q0.get("mean_moving_players_per_frame", 1) == 0 and q0.get("ball_detect_rate", 1) == 0 and "analysis_frames" not in q0:
                os.replace(qp, qp + ".bad_e6dbc0e"); st.p(r, prelabel="todo", note="redo: old run decoded 0 frames (AV1)")
                log(r["id"], "old pre-label decoded no frames (AV1 + VideoToolbox); re-processing")
    init_tracknet(base)
    q = queue.Queue()
    queued = set()

    def needs_prelabel(r):
        path = os.path.join(base, r["sport"], f"{r['id']}.mp4")
        return (not args.no_prelabel and os.path.exists(path) and st.man.get(r["id"], {}).get("status") == "downloaded"
                and st.prog[r["id"]].get("prelabel") not in ("done", "unusable", "error"))
    for r in src:                       # already downloaded, not yet labelled -> straight into the queue (resume)
        if needs_prelabel(r): q.put(r); queued.add(r["id"])

    def downloader():
        """Producer: polite sequential downloads (sleeps/backoff live here only); hands each finished file to the labeller."""
        global DL_STOP
        try:
            for r in src:
                if STOP or DL_STOP: break
                if not disk_ok(base): DL_STOP = True; break
                if st.man.get(r["id"], {}).get("status") in ("downloaded", "skipped"): continue
                got = download(base, st, r, lambda: None)
                st.save()
                if got:
                    if needs_prelabel(r) and r["id"] not in queued: q.put(r); queued.add(r["id"])
                    sleep_interruptible(random.randint(45, 100))
        except Exception as e:
            log("downloader error:", repr(e))
        finally:
            q.put(None)
    th = None
    if not args.no_download:
        th = threading.Thread(target=downloader, name="downloader", daemon=True); th.start()
    else:
        q.put(None)
    while not STOP:                     # consumer: label one video at a time (rallies fan out over the worker pool)
        try:
            r = q.get(timeout=30)
        except queue.Empty:
            continue
        if r is None: break
        path = os.path.join(base, r["sport"], f"{r['id']}.mp4")
        wait_if_busy()
        if STOP: break
        log(r["id"], "pre-label start"); st.p(r, prelabel="running"); t0 = time.time()
        try:
            qa = prelabel(base, r, path, args)
        except Exception as e:
            log(r["id"], "pre-label error:", repr(e)); st.p(r, prelabel="error", note=repr(e)[:80]); continue
        if qa is None: st.p(r, prelabel="todo"); break
        st.p(r, prelabel="done" if qa.get("usable") else "unusable", rallies=qa.get("rally_count"), court=qa.get("court_line_support"),
             ball_rate=qa.get("ball_detect_rate"), note=(qa.get("failure_notes") or "")[:80])
        write_qa_summary(base)
        el = time.time() - t0
        log(r["id"], "pre-label", "done" if qa.get("usable") else "UNUSABLE", f"in {el:.0f} s = {el / max(1e-6, qa.get('duration_s', 0) / 60):.1f} s per video-minute",
            qa.get("failure_notes", ""))
    if th is not None and STOP: th.join(timeout=60)
    if POOL is not None: POOL.shutdown(wait=True, cancel_futures=True)
    write_qa_summary(base); st.save()
    n = {k: sum(1 for m in st.man.values() if m.get("status") == k) for k in ("downloaded", "skipped", "pending")}
    log("finished pass:", n, "| re-run the script to resume anything pending" if n["pending"] or STOP or DL_STOP else "")


if __name__ == "__main__":
    main()
