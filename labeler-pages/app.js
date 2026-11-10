// OpenCourtVision volunteer labeler (static). No answer key, no network calls except clips.json and the clips.
"use strict";
(function () {
  var KEY = "ocv-labeler-session-01-v1";
  var REPO = "https://gitlab.com/antonyvo/opencourtvision";
  var USER_RE = /^[A-Za-z0-9_.-]{1,255}$/;
  var data = null, i = 0;
  var st = load();

  function load() {
    try { var s = JSON.parse(localStorage.getItem(KEY) || "{}"); return { user: s.user || "", answers: s.answers || {}, at: s.at || 0 }; }
    catch (e) { return { user: "", answers: {}, at: 0 }; }
  }
  function save() { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) { /* private mode: keep in memory */ } }
  function $(id) { return document.getElementById(id); }

  function clipAns(id) { return st.answers[id] || (st.answers[id] = { winner: null, server: null }); }

  function show() {
    var c = data.clips[i];
    var v = $("clip");
    if (v.getAttribute("src") !== c.file) { v.src = c.file; }
    v.playbackRate = parseFloat($("speed").value);
    var a = st.answers[c.id] || {};
    $("progress").textContent = "Clip " + (i + 1) + " of " + data.clips.length + "  ·  answered " + doneCount() + " / " + data.clips.length;
    each(".win", function (b) { b.classList.toggle("on", a.winner === b.dataset.v); });
    each(".srv", function (b) { b.classList.toggle("on", (a.server || a.winner) ? (a.server || "") === b.dataset.v : false); });
    var vid = byId(c.video);
    $("credit").textContent = vid ? "Source: \u201c" + vid.title + "\u201d by " + vid.channel + ", CC BY." : "";
    $("prev").disabled = i === 0; $("next").disabled = i === data.clips.length - 1;
    $("done-count").textContent = doneCount() + " of " + data.clips.length + " clips answered.";
    st.at = i; save();
  }
  function each(sel, f) { Array.prototype.forEach.call(document.querySelectorAll(sel), f); }
  function byId(id) { for (var k = 0; k < data.videos.length; k++) if (data.videos[k].id === id) return data.videos[k]; return null; }
  function doneCount() { var n = 0; data.clips.forEach(function (c) { var a = st.answers[c.id]; if (a && a.winner) n++; }); return n; }

  function setWinner(v) { clipAns(data.clips[i].id).winner = v; save(); show(); }
  function setServer(v) { var a = clipAns(data.clips[i].id); a.server = v || null; save(); show(); }
  function go(d) { var j = i + d; if (j >= 0 && j < data.clips.length) { i = j; show(); playFromStart(); } }
  function playFromStart() { var v = $("clip"); try { v.currentTime = 0; } catch (e) {} var p = v.play(); if (p && p.catch) p.catch(function () {}); }

  function userOk() {
    var u = $("user").value.trim().replace(/^@/, "");
    var ok = USER_RE.test(u) && u.indexOf("@") < 0;
    $("user-err").hidden = ok || u === "";
    if (ok) { st.user = u; save(); }
    return ok ? u : null;
  }

  function payload(u) {
    var answers = [];
    data.clips.forEach(function (c) {
      var a = st.answers[c.id];
      if (a && a.winner) answers.push({ id: c.id, winner: a.winner, server: a.winner === "S" ? null : (a.server || null) });
    });
    return { schema: "ocv.volunteer_labels.v1", session: data.session, gitlab_username: u, answers: answers };
  }
  function need() {
    var u = userOk();
    if (!u) { alert("Enter your GitLab username first."); $("user").focus(); return null; }
    if (!doneCount()) { alert("Answer at least one clip first."); return null; }
    return payload(u);
  }

  function download() {
    var p = need(); if (!p) return;
    var blob = new Blob([JSON.stringify(p, null, 1) + "\n"], { type: "application/json" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "labels.json";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }
  function issue() {
    var p = need(); if (!p) return;
    var body = "/confidential\n\nVolunteer labels for " + p.session + " from @" + p.gitlab_username +
      " (" + p.answers.length + " of " + data.clips.length + " clips), made with the OpenCourtVision Pages labeler.\n\n```json\n" +
      JSON.stringify(p) + "\n```\n";
    var url = REPO + "/-/issues/new?issue[title]=" + encodeURIComponent("Labels: " + p.session + " from @" + p.gitlab_username) +
      "&issue[description]=" + encodeURIComponent(body);
    window.open(url, "_blank", "noopener,noreferrer");
  }

  function keys(e) {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var k = e.key;
    if (k === "a" || k === "A") setWinner("A");
    else if (k === "b" || k === "B") setWinner("B");
    else if (k === "s" || k === "S") setWinner("S");
    else if (k === "1") setServer("A");
    else if (k === "2") setServer("B");
    else if (k === "0") setServer("");
    else if (k === "r" || k === "R") playFromStart();
    else if (k === "ArrowRight") go(1);
    else if (k === "ArrowLeft") go(-1);
    else return;
    e.preventDefault();
  }

  function init(d) {
    data = d;
    i = Math.min(Math.max(st.at | 0, 0), data.clips.length - 1);
    $("user").value = st.user;
    $("user").addEventListener("input", userOk);
    each(".win", function (b) { b.addEventListener("click", function () { setWinner(b.dataset.v); }); });
    each(".srv", function (b) { b.addEventListener("click", function () { setServer(b.dataset.v); }); });
    $("prev").addEventListener("click", function () { go(-1); });
    $("next").addEventListener("click", function () { go(1); });
    $("replay").addEventListener("click", playFromStart);
    $("speed").addEventListener("change", function () { $("clip").playbackRate = parseFloat(this.value); });
    $("clip").addEventListener("loadedmetadata", function () { this.playbackRate = parseFloat($("speed").value); });
    $("download").addEventListener("click", download);
    $("issue").addEventListener("click", issue);
    $("clear").addEventListener("click", function () {
      if (!confirm("Delete your username and answers from this browser?")) return;
      try { localStorage.removeItem(KEY); } catch (e) {}
      st = { user: "", answers: {}, at: 0 }; $("user").value = ""; i = 0; show();
    });
    document.addEventListener("keydown", keys);
    var ul = $("credit-list");
    data.videos.forEach(function (v) {
      var li = document.createElement("li"), a = document.createElement("a");
      a.href = v.url; a.rel = "noopener noreferrer"; a.textContent = "\u201c" + v.title + "\u201d";
      var ch = document.createElement("a"); ch.href = v.channel_url; ch.rel = "noopener noreferrer"; ch.textContent = v.channel;
      li.appendChild(a); li.appendChild(document.createTextNode(" by ")); li.appendChild(ch);
      li.appendChild(document.createTextNode(", CC BY"));
      ul.appendChild(li);
    });
    show();
  }

  fetch("clips.json", { cache: "no-cache", credentials: "omit" })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(init)
    .catch(function (e) { $("progress").textContent = "Could not load the clip list (" + e.message + "). Reload the page."; });
})();
