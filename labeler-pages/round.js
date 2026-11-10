"use strict";
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OCVRound = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var CALLS = { A: true, B: true, F: true };
  var STRIP = ["answer", "reason_en", "reason_es", "role", "check", "gold", "is_check", "answer_token"];

  function videoOf(point) {
    return (point && (point.video_id || point.video)) || "";
  }

  function present(point) {
    var engine = point && point.engine;
    if (engine !== "A" && engine !== "B" && engine !== "F") engine = null;
    return { engine: engine, verdict: false };
  }

  function pointClasses() { return "point"; }

  function liveFeedback(xp) {
    return { verdict: null, xp: xp };
  }

  function answerWord(code, lang) {
    if (code === "F") return lang === "es" ? "falta" : "Fault";
    if (code === "A" || code === "B") return code;
    return "—";
  }

  function playerFeedback(phase, point, call, lang) {
    if (phase === "warmup") {
      var name = answerWord(point && point.answer, lang);
      var right = call === (point && point.answer);
      var text = right
        ? (lang === "es" ? "Bien, era " + name + "." : "Right, it was " + name + ".")
        : (lang === "es" ? "Era " + name + "." : "It was " + name + ".");
      return {
        showEngine: false,
        text: text,
        reason: lang === "es" ? ((point && point.reason_es) || "") : ((point && point.reason_en) || "")
      };
    }
    var engine = present(point).engine;
    var called = engine === "F" ? (lang === "es" ? "Saque falta" : "Fault serve") : engine;
    var engineText = !engine
      ? (lang === "es" ? "El motor cantó —" : "The engine called —")
      : (lang === "es" ? "El motor cantó " + called : "The engine called " + called);
    return { showEngine: true, text: null, engineText: engineText };
  }

  function sanitizeClip(point) {
    var out = {
      id: point.id,
      file: point.file || "",
      video_id: videoOf(point),
      engine: present(point).engine
    };
    return out;
  }

  function sameFace(a, b) {
    var pa = present(a);
    var pb = present(b);
    return pa.engine === pb.engine && pa.verdict === pb.verdict && pointClasses(a) === pointClasses(b);
  }

  function pickRound(clips, used, size) {
    var skip = {};
    (used || []).forEach(function (id) { skip[id] = true; });
    var out = [];
    (clips || []).forEach(function (clip) {
      if (out.length >= size) return;
      if (!skip[clip.id]) out.push(sanitizeClip(clip));
    });
    return out;
  }

  function submissionCalls(ids, calls, practiceIds) {
    var banned = {};
    (practiceIds || []).forEach(function (id) { banned[id] = true; });
    var out = [];
    (ids || []).forEach(function (id) {
      if (banned[id]) return;
      if (!CALLS[calls[id]]) return;
      out.push({ point_id: id, call: calls[id] });
    });
    return out;
  }

  function gradeState(points, grades, previous) {
    var current = null;
    (grades || []).forEach(function (g) { if (points >= g.points) current = g.name; });
    return (grades || []).map(function (g) {
      var unlocked = points >= g.points;
      var cls = "locked";
      if (unlocked && g.name !== current) cls = "have";
      if (g.name === current) cls = "current";
      var seen = previous || [];
      if (unlocked && seen.indexOf(g.name) < 0 && g.name === current) cls = "fresh";
      return { name: g.name, need: g.points, cls: cls };
    });
  }

  function unlockedNames(points, grades) {
    var names = [];
    (grades || []).forEach(function (g) { if (points >= g.points) names.push(g.name); });
    return names;
  }

  function prevDay(iso) {
    var t = Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10));
    return new Date(t - 86400000).toISOString().slice(0, 10);
  }

  function streak(days, today) {
    var set = {};
    (days || []).forEach(function (d) { set[d] = true; });
    if (!set[today]) return 0;
    var n = 0;
    var d = today;
    while (set[d]) { n += 1; d = prevDay(d); }
    return n;
  }

  function leaderboardRows(file, minChecks) {
    var min = minChecks == null ? 10 : minChecks;
    var rows = (file && file.rows) || [];
    var eligible = [];
    rows.forEach(function (row) {
      if (!row || row.opt_in !== true || !row.nickname) return;
      if (typeof row.check_count !== "number" || row.check_count < min) return;
      if (typeof row.accuracy !== "number") return;
      eligible.push(row);
    });
    eligible.sort(function (a, b) {
      if (b.accuracy !== a.accuracy) return b.accuracy - a.accuracy;
      return String(a.nickname).localeCompare(String(b.nickname));
    });
    return eligible.map(function (row) {
      return { nickname: String(row.nickname), accuracy_band: row.accuracy_band || "—" };
    });
  }

  return {
    CALLS: CALLS,
    STRIP: STRIP,
    videoOf: videoOf,
    present: present,
    pointClasses: pointClasses,
    liveFeedback: liveFeedback,
    answerWord: answerWord,
    playerFeedback: playerFeedback,
    sanitizeClip: sanitizeClip,
    sameFace: sameFace,
    pickRound: pickRound,
    submissionCalls: submissionCalls,
    gradeState: gradeState,
    unlockedNames: unlockedNames,
    streak: streak,
    leaderboardRows: leaderboardRows
  };
});
