"use strict";
(function () {
  var CFG = window.OCV_GAME_CONFIG;
  var I18N = window.OCV_I18N;
  var R = window.OCVRound;
  var SUB = window.OCVSubmit;
  var SHARE = window.OCVShare;
  var FORM = window.OCV_SUBMIT_CONFIG;
  var KEY = "ocv-call-the-point-v1";
  var app = document.getElementById("app");
  var lang = "en";
  var practice = [];
  var clips = [];
  var videos = [];
  var device = "";
  var boardFile = null;
  var state = load();
  var sessionCalls = 0;
  var sessionCorrect = 0;
  var sessionAnswered = 0;
  var warmupResult = null;

  function load() {
    try {
      var s = JSON.parse(localStorage.getItem(KEY) || "{}");
      return {
        nickname: s.nickname || "",
        adult: !!s.adult,
        pointsCalled: s.pointsCalled || 0,
        days: s.days || [],
        creditOptIn: !!s.creditOptIn,
        used: s.used || [],
        pending: s.pending || null,
        sent: !!s.sent,
        hotStreak: s.hotStreak || 0,
        bestStreak: s.bestStreak || 0
      };
    } catch (e) {
      return { nickname: "", adult: false, pointsCalled: 0, days: [], creditOptIn: false, used: [], pending: null, sent: false, hotStreak: 0, bestStreak: 0 };
    }
  }
  function save() {
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* private mode */ }
  }
  function t(key) { return I18N.t(lang, key); }
  function el(tag, text, attrs) {
    var node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (attrs) Object.keys(attrs).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }
  function clear() { while (app.firstChild) app.removeChild(app.firstChild); }
  function today() { return new Date().toISOString().slice(0, 10); }
  function streakLabel() {
    var n = R.streak(state.days, today());
    return n > 0 ? String(n) : "—";
  }
  function practiceIds() { return practice.map(function (p) { return p.id; }); }

  function creditLine(videoId) {
    var p = el("p", null, { class: "small credit", id: "credit" });
    p.appendChild(document.createTextNode(R.creditLineText(videos, videoId, lang) + " "));
    p.appendChild(el("a", lang === "es" ? "Cr\u00e9ditos" : "Credits", { href: "../CC_ATTRIBUTION.txt" }));
    return p;
  }

  function refreshFooter() {
    var a = document.getElementById("delete-data");
    if (a) {
      a.textContent = t("deleteData");
      a.href = SUB.deleteMailto(state.nickname, device);
    }
    var note = document.getElementById("privacy-id");
    if (note) note.textContent = t("privacyId");
  }

  function headerCount(phase, index, total) {
    var key = phase === "warmup" ? "warmupCount" : "pointCount";
    return el("p", I18N.fill(t(key), { i: index + 1, n: total }), { class: "mono" });
  }

  function sessionAccuracyText() {
    if (!sessionAnswered) return "—";
    var pct = Math.round((sessionCorrect / sessionAnswered) * 100);
    return sessionCorrect + "/" + sessionAnswered + " · " + pct + "%";
  }

  function flameRow() {
    var row = el("div", null, { class: "flame-row" });
    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("class", "flame");
    svg.setAttribute("aria-hidden", "true");
    var a = document.createElementNS("http://www.w3.org/2000/svg", "path");
    a.setAttribute("d", "M12 2c2 4 2 6 0 8 3-1 6 2 6 6a6 6 0 1 1-12 0c0-3 2-5 4-8 0 2 1 3 2 2z");
    a.setAttribute("fill", "#FFB627");
    var b = document.createElementNS("http://www.w3.org/2000/svg", "path");
    b.setAttribute("d", "M12 14c1 2 3 2 3 4a3 3 0 1 1-6 0c0-2 1-3 3-4z");
    b.setAttribute("fill", "#C8F43C");
    svg.appendChild(a); svg.appendChild(b);
    row.appendChild(svg);
    var streakWrap = el("span", null, { class: "hot-streak" });
    streakWrap.appendChild(el("span", String(state.hotStreak), { class: "mono", id: "streak" }));
    streakWrap.appendChild(document.createTextNode(" " + t("streakLabel")));
    row.appendChild(streakWrap);
    var accWrap = el("span", null, { class: "session-acc" });
    accWrap.appendChild(document.createTextNode(t("accuracyLabel") + " "));
    accWrap.appendChild(el("span", sessionAccuracyText(), { class: "mono", id: "accuracy" }));
    row.appendChild(accWrap);
    return row;
  }

  function gradeProgressBlock() {
    var p = R.gradeProgress(state.pointsCalled, CFG.grades);
    var wrap = el("div", null, { class: "grade-progress", id: "grade-progress" });
    if (!p.next) {
      wrap.appendChild(el("p", t("gradeMax"), { class: "small" }));
    } else {
      var bar = el("div", null, { class: "progress-bar" });
      var fill = el("div", null, { class: "progress-fill" });
      fill.style.width = p.pct + "%";
      bar.appendChild(fill);
      wrap.appendChild(bar);
      wrap.appendChild(el("p", I18N.fill(t("gradeNext"), { n: p.need, grade: p.next }), { class: "small", id: "grade-next" }));
    }
    return wrap;
  }

  function bestStreakLine() {
    return el("p", I18N.fill(t("bestStreak"), { n: state.bestStreak }), { class: "mono", id: "best-streak" });
  }

  function stage(file) {
    var box = el("div", null, { class: "stage", id: "stage" });
    box.appendChild(el("p", t("clipLabel"), { class: "clip-label" }));
    var img = el("img", null, { class: "ph", id: "placeholder", src: "../assets/court.svg", alt: "" });
    var video = el("video", null, { id: "clip", playsinline: "", muted: "", controls: "" });
    video.hidden = true;
    if (file) {
      video.src = file;
      video.addEventListener("loadeddata", function () {
        video.hidden = false;
        img.hidden = true;
      });
    }
    box.appendChild(img);
    box.appendChild(video);
    return box;
  }

  function callButtons(onCall, locked, chosen) {
    var wrap = el("div", null, { class: "stack", id: "calls" });
    var pair = el("div", null, { class: "pair" });
    [["A", "callA"], ["B", "callB"]].forEach(function (item) {
      var b = el("button", t(item[1]), { type: "button", class: "btn call" + (chosen === item[0] ? " on" : ""), "data-call": item[0] });
      b.disabled = !!locked;
      b.addEventListener("click", function () { onCall(item[0]); });
      pair.appendChild(b);
    });
    var fault = el("button", t("callF"), { type: "button", class: "btn quiet call" + (chosen === "F" ? " on" : ""), "data-call": "F" });
    fault.disabled = !!locked;
    fault.addEventListener("click", function () { onCall("F"); });
    wrap.appendChild(pair);
    wrap.appendChild(fault);
    wrap.appendChild(el("p", t("faultSub"), { class: "small fault-sub", id: "fault-sub" }));
    return wrap;
  }

  function showStart() {
    clear();
    document.title = t("title");
    app.appendChild(el("h1", t("title")));
    var nick = el("input", null, { id: "nick", type: "text", maxlength: "32", autocomplete: "off" });
    nick.placeholder = t("nickname");
    nick.value = state.nickname;
    nick.placeholder = t("nickname");
    nick.setAttribute("aria-label", t("nickname"));
    nick.addEventListener("input", function () { state.nickname = nick.value.trim(); save(); refreshFooter(); });
    var lab = el("label", null, { class: "check" });
    var box = el("input", null, { id: "adult", type: "checkbox" });
    box.checked = false;
    lab.appendChild(box);
    lab.appendChild(document.createTextNode(t("adult")));
    var play = el("button", t("play"), { type: "button", class: "btn volt", id: "play" });
    play.disabled = true;
    box.addEventListener("change", function () {
      play.disabled = !box.checked;
      state.adult = box.checked;
      save();
    });
    play.addEventListener("click", function () {
      if (!box.checked || !practice.length) return;
      showWarmup(0, null);
    });
    app.appendChild(nick);
    app.appendChild(lab);
    app.appendChild(play);
    app.appendChild(gradeProgressBlock());
    if (state.bestStreak > 0) app.appendChild(bestStreakLine());
    app.appendChild(el("p", t("honest"), { class: "small", id: "honest" }));
    var boardLink = el("button", t("boardTitle"), { type: "button", class: "textlink", id: "open-board" });
    boardLink.addEventListener("click", showBoard);
    app.appendChild(boardLink);
    refreshFooter();
  }

  function showWarmup(index, chosen) {
    var point = practice[index];
    clear();
    var head = el("div", null, { class: "point-head" });
    head.appendChild(flameRow());
    head.appendChild(headerCount("warmup", index, practice.length));
    app.appendChild(head);
    app.appendChild(stage(point.file));
    app.appendChild(creditLine(point.video_id));
    if (chosen) {
      var fb = R.playerFeedback("warmup", point, chosen, lang);
      var card = el("div", null, { class: "card feedback-card", id: "feedback-card" });
      card.appendChild(el("p", fb.text, { class: fb.right ? "pill-ok" : "pill-bad", id: "feedback" }));
      card.appendChild(el("p", fb.reason, { id: "reason" }));
      if (warmupResult && warmupResult.xp > 0) {
        card.appendChild(el("p", I18N.fill(t("xp"), { n: warmupResult.xp }), { class: "mono", id: "warmup-xp" }));
      }
      card.appendChild(el("p", t("warmupTag"), { class: "small" }));
      app.appendChild(card);
    } else {
      app.appendChild(el("p", t("legend"), { class: "legend", id: "legend" }));
      app.appendChild(callButtons(function (call) {
        sessionCalls += 1;
        var fb = R.playerFeedback("warmup", point, call, lang);
        sessionAnswered += 1;
        if (fb.right) sessionCorrect += 1;
        var res = R.applyStreak(state.hotStreak, fb.right, CFG.streakBonus);
        state.hotStreak = res.streak;
        if (res.streak > state.bestStreak) state.bestStreak = res.streak;
        warmupResult = { right: fb.right, bonus: res.bonus, xp: fb.right ? CFG.xpPerCall + res.bonus : 0 };
        save();
        showWarmup(index, call);
      }, false, null));
      app.appendChild(el("p", t("warmupTag"), { class: "small warmup-tag" }));
    }
    if (chosen) {
      var next = el("button", t("next"), { type: "button", class: "btn volt", id: "next" });
      next.addEventListener("click", function () {
        if (index + 1 < practice.length) showWarmup(index + 1, null);
        else beginRound();
      });
      app.appendChild(next);
    } else {
      var skip = el("button", t("skip"), { type: "button", class: "textlink", id: "skip-warmup" });
      skip.addEventListener("click", beginRound);
      app.appendChild(skip);
    }
    refreshFooter();
  }

  function beginRound() {
    var picked = R.pickRound(clips, state.used, CFG.roundSize, practiceIds());
    if (picked.length < CFG.roundSize) {
      clear();
      app.appendChild(el("p", t("honest")));
      return;
    }
    state.pending = {
      round_id: (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now()),
      ids: picked.map(function (c) { return c.id; }),
      calls: {},
      index: 0,
      pointsAtStart: state.pointsCalled,
      gradesAtStart: R.unlockedNames(state.pointsCalled, CFG.grades)
    };
    state.sent = false;
    save();
    showLive(null);
  }

  function currentClip() {
    var id = state.pending.ids[state.pending.index];
    for (var i = 0; i < clips.length; i++) if (clips[i].id === id) return R.sanitizeClip(clips[i]);
    return { id: id, file: "", engine: null };
  }

  function showLive(chosen) {
    var pending = state.pending;
    var clip = currentClip();
    clear();
    var head = el("div", null, { class: "point-head" });
    head.appendChild(flameRow());
    head.appendChild(headerCount("live", pending.index, pending.ids.length));
    app.appendChild(head);
    app.appendChild(stage(clip.file));
    app.appendChild(creditLine(clip.video_id));
    app.appendChild(el("p", t("legend"), { class: "legend", id: "legend" }));
    var feedback = el("p", "", { class: "feedback", id: "feedback" });
    var engine = el("p", "", { class: "engine", id: "engine", hidden: "" });
    app.appendChild(feedback);
    app.appendChild(engine);
    if (chosen) {
      var live = R.liveFeedback(CFG.xpPerCall);
      var face = R.playerFeedback("live", clip, chosen, lang);
      feedback.textContent = t("called") + " " + I18N.fill(t("xp"), { n: live.xp });
      engine.hidden = !face.showEngine;
      engine.textContent = face.engineText;
    }
    app.appendChild(callButtons(function (call) {
      if (pending.calls[clip.id]) return;
      pending.calls[clip.id] = call;
      state.pointsCalled += 1;
      sessionCalls += 1;
      if (state.used.indexOf(clip.id) < 0) state.used.push(clip.id);
      save();
      showLive(call);
    }, !!chosen, chosen));
    if (chosen) {
      var last = pending.index + 1 >= pending.ids.length;
      var next = el("button", last ? t("summaryGo") : t("next"), { type: "button", class: "btn volt", id: "next" });
      next.addEventListener("click", function () {
        if (last) finishRound();
        else { pending.index += 1; save(); showLive(null); }
      });
      app.appendChild(next);
    }
    refreshFooter();
  }

  function finishRound() {
    var day = today();
    if (state.days.indexOf(day) < 0) state.days.push(day);
    save();
    showSummary();
  }

  function roundCalls() {
    return R.submissionCalls(state.pending.ids, state.pending.calls, practiceIds());
  }

  function showSummary() {
    clear();
    var n = state.pending.ids.length;
    app.appendChild(el("h1", t("title")));
    app.appendChild(el("p", I18N.fill(t("summaryCount"), { n: n }), { class: "mono", id: "sum-count" }));
    app.appendChild(el("p", t("beat"), { id: "sum-beat" }));
    var grades = el("ul", null, { class: "grades", id: "grades" });
    R.gradeState(state.pointsCalled, CFG.grades, state.pending.gradesAtStart).forEach(function (g) {
      grades.appendChild(el("li", g.name + " " + g.need, { class: g.cls }));
    });
    app.appendChild(grades);
    app.appendChild(gradeProgressBlock());
    app.appendChild(bestStreakLine());
    var opt = el("label", null, { class: "check" });
    var optBox = el("input", null, { id: "credits-opt", type: "checkbox" });
    optBox.checked = false;
    opt.appendChild(optBox);
    opt.appendChild(document.createTextNode(t("credits")));
    optBox.addEventListener("change", function () { state.creditOptIn = optBox.checked; save(); });
    app.appendChild(opt);
    var more = el("button", t("more"), { type: "button", class: "btn volt", id: "more" });
    more.addEventListener("click", beginRound);
    app.appendChild(more);
    var ready = SUB.submitReady(FORM);
    var sendLabel = ready ? (state.sent ? t("resend") : t("send")) : t("sendSoon");
    var send = el("button", sendLabel, { type: "button", class: "btn outline", id: "send" });
    send.disabled = !ready;
    send.addEventListener("click", postCalls);
    app.appendChild(send);
    app.appendChild(el("p", t("sendNote"), { class: "small", id: "send-note" }));
    if (state.sent) app.appendChild(el("p", t("sent"), { id: "send-status", class: "small" }));
    var mail = el("a", t("email"), { id: "email-fallback", class: "textlink", href: mailHref() });
    app.appendChild(mail);
    var share = el("button", t("share"), { type: "button", class: "btn outline", id: "share" });
    share.addEventListener("click", showShare);
    app.appendChild(share);
    var boardBtn = el("button", t("boardTitle"), { type: "button", class: "textlink", id: "open-board" });
    boardBtn.addEventListener("click", showBoard);
    app.appendChild(boardBtn);
    refreshFooter();
  }

  function mailHref() {
    var calls = roundCalls();
    var json = SUB.callsJson({ round_id: state.pending.round_id, calls: calls });
    return SUB.mailtoUrl(state.nickname, json);
  }

  function postCalls() {
    if (!SUB.submitReady(FORM) || !state.pending) return;
    var calls = roundCalls();
    var spec = SUB.encodeBody(FORM, {
      nickname: state.nickname,
      round_id: state.pending.round_id,
      calls: calls,
      device_id: device
    });
    fetch(spec.url, {
      method: "POST",
      mode: spec.mode,
      credentials: spec.credentials,
      headers: { "Content-Type": spec.contentType },
      body: spec.body
    }).then(function () {
      state.sent = true;
      save();
      showSummary();
    }).catch(function () { /* the email link stays on the summary */ });
  }

  function currentGrade() {
    var names = R.unlockedNames(state.pointsCalled, CFG.grades);
    return names.length ? names[names.length - 1] : "";
  }

  function showShare() {
    var panel = el("div", null, { class: "share-panel", id: "share-panel" });
    var model = SHARE.cardCopy(I18N, lang, state.pointsCalled, currentGrade());
    ["fourfive", "story"].forEach(function (name) {
      var canvas = el("canvas");
      SHARE.draw(canvas, model, name);
      panel.appendChild(canvas);
      var a = el("a", name, { class: "textlink", download: "vamos-" + name + ".png" });
      a.href = canvas.toDataURL("image/png");
      panel.appendChild(a);
    });
    app.appendChild(panel);
  }

  function showBoard() {
    clear();
    app.appendChild(el("h1", t("boardTitle")));
    app.appendChild(el("p", t("boardHint"), { class: "small" }));
    var head = el("div", null, { class: "board-head" });
    head.appendChild(el("span", t("boardName")));
    head.appendChild(el("span", t("boardAccuracy")));
    app.appendChild(head);
    var list = el("ol", null, { class: "board", id: "board" });
    var rows = R.leaderboardRows(boardFile, CFG.minChecks);
    if (!rows.length) {
      var li = el("li");
      li.appendChild(el("span", t("boardName")));
      li.appendChild(el("span", "—", { class: "mono" }));
      list.appendChild(li);
    } else {
      rows.forEach(function (row) {
        var li = el("li");
        li.appendChild(el("span", row.nickname));
        li.appendChild(el("span", row.accuracy_band || "—", { class: "mono" }));
        list.appendChild(li);
      });
    }
    app.appendChild(list);
    var back = el("button", t("back"), { type: "button", class: "textlink", id: "back-start" });
    back.addEventListener("click", showStart);
    app.appendChild(back);
    refreshFooter();
  }

  function boot() {
    document.getElementById("lang-en").addEventListener("click", function () { lang = "en"; document.documentElement.lang = "en"; showStart(); });
    document.getElementById("lang-es").addEventListener("click", function () { lang = "es"; document.documentElement.lang = "es"; showStart(); });
    device = SUB.ensureDeviceId(localStorage, function () { return crypto.randomUUID(); });
    if (new URLSearchParams(location.search).get("lang") === "es") lang = "es";
    var practiceP = fetch("practice.json", { cache: "no-cache", credentials: "omit" }).then(function (r) { return r.json(); });
    var clipsP = fetch("clips.json", { cache: "no-cache", credentials: "omit" }).then(function (r) { return r.json(); });
    var boardP = fetch("leaderboard.json", { cache: "no-cache", credentials: "omit" }).then(function (r) {
      if (!r.ok) return null;
      return r.json();
    }).catch(function () { return null; });
    Promise.all([practiceP, clipsP, boardP]).then(function (all) {
      practice = (all[0] && all[0].points) || [];
      clips = (all[1] && all[1].clips) || [];
      videos = (all[1] && all[1].videos) || [];
      boardFile = all[2];
      showStart();
    }).catch(function () { showStart(); });
  }
  boot();
})();
