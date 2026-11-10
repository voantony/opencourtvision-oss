"use strict";
const assert = require("assert");
const round = require("../labeler-pages/round.js");
const game = require("../labeler-pages/game_config.js");
const share = require("../labeler-pages/share.js");
const i18n = require("../labeler-pages/i18n.js");
const submit = require("../labeler-pages/submit.js");
const form = require("../labeler-pages/submit_config.js");

const hidden = { id: "hold-check-1", engine: "A", answer: "B", role: "check", answer_token: "HOLD_ANSWER_CHECK_7c1e" };
const plain = { id: "live-1", engine: "A" };
assert.strictEqual(round.sameFace(hidden, plain), true);
assert.strictEqual(round.pointClasses(hidden), "point");
assert.strictEqual(round.pointClasses(plain), "point");
assert.deepStrictEqual(round.present(hidden), { engine: "A", verdict: false });
assert.strictEqual(round.liveFeedback(game.xpPerCall).verdict, null);
assert.ok(!("answer" in round.sanitizeClip(hidden)));

const calls = round.submissionCalls(
  ["synth-practice-1", "live-1"],
  { "synth-practice-1": "A", "live-1": "F" },
  ["synth-practice-1"]
);
assert.deepStrictEqual(calls, [{ point_id: "live-1", call: "F" }]);

assert.deepStrictEqual(game.grades.map((g) => g.points), [1, 10, 50, 200]);
assert.strictEqual(game.xpPerCall, 10);
const fresh = round.gradeState(10, game.grades, []);
assert.strictEqual(fresh.find((g) => g.name === "Intermedio").cls, "fresh");
assert.strictEqual(fresh.find((g) => g.name === "Pro").cls, "locked");

const board = round.leaderboardRows({
  rows: [
    { nickname: "tapper", points: 5000, accuracy: 0.2, check_count: 1, opt_in: true, accuracy_band: "low" },
    { nickname: "steady", points: 12, accuracy: 0.8, check_count: 10, opt_in: true, accuracy_band: "80-89%" },
    { nickname: "quiet", points: 80, accuracy: 0.99, check_count: 30, opt_in: false, accuracy_band: "90%+" }
  ]
}, 10);
assert.deepStrictEqual(board.map((row) => row.nickname), ["steady"]);
assert.ok(!("points" in board[0]));

assert.deepStrictEqual(share.SIZES.fourfive, [1080, 1350]);
assert.deepStrictEqual(share.SIZES.story, [1080, 1920]);
assert.ok(share.SIZES.square === undefined);
const warmWrong = round.playerFeedback("warmup", {
  answer: "A", engine: "B", reason_en: "The far side's volley goes straight into the net, so the near side wins the point.", reason_es: "lejos"
}, "B", "en");
assert.strictEqual(warmWrong.showEngine, false);
assert.strictEqual(warmWrong.right, false);
assert.strictEqual(warmWrong.text, "Incorrect");
assert.ok(!("engineText" in warmWrong));
assert.ok(!/engine/i.test(warmWrong.text + warmWrong.reason));
assert.strictEqual(round.playerFeedback("warmup", { answer: "F", reason_en: "Let: the serve clipped the net and landed in, so it's replayed and no point is scored." }, "F", "en").text, "Correct");
assert.strictEqual(round.playerFeedback("warmup", { answer: "B", reason_es: "El remate del lado cercano se va a la red, así que punto para el lado lejano." }, "B", "es").text, "Correcto");
const liveOnly = round.pickRound([
  { id: "2p6PcKP0720_p4", a_side_team: "A" },
  { id: "2p6PcKP0720_p1", a_side_team: "A" },
  { id: "632k3fKraeA_p1", a_side_team: null }
], [], 10, ["2p6PcKP0720_p4"]);
assert.deepStrictEqual(liveOnly.map((c) => c.id), ["2p6PcKP0720_p1"]);
assert.strictEqual(round.toTeamCall("A", "A"), "A");
assert.strictEqual(round.toTeamCall("A", "B"), "B");
assert.strictEqual(round.toTeamCall("B", "B"), "A");
assert.strictEqual(round.toTeamCall("F", "B"), "F");
assert.strictEqual(i18n.t("en", "legend"), "A: near side · B: far side");
assert.strictEqual(i18n.t("es", "legend"), "A: lado cercano · B: lado lejano");
assert.strictEqual(i18n.t("en", "faultSub"), "Serve doesn't count (fault or let)");
assert.ok(!i18n.copy.en.reminder);
assert.ok(!String(i18n.copy.en.legend).includes("shirt"));
const liveFace = round.playerFeedback("live", { engine: "A", answer: "SECRET" }, "B", "en");
assert.strictEqual(liveFace.showEngine, true);
assert.strictEqual(liveFace.engineText, "The engine called A");
assert.ok(!String(liveFace.engineText).includes("SECRET"));
const banned = ["Ops", "session-02", "synthetic", "submitEnabled", "intake", "accuracy number", "Not the call."];
["en", "es"].forEach((lang) => {
  Object.entries(i18n.copy[lang]).forEach(([key, value]) => {
    banned.forEach((phrase) => assert.ok(!String(value).includes(phrase), lang + " " + key + " includes " + phrase));
  });
});
assert.strictEqual(i18n.t("en", "honest"), "Public Creative Commons (CC BY) clips. Vamos is a work in progress.");
assert.strictEqual(i18n.t("es", "honest"), "Clips públicos de Creative Commons (CC BY). Vamos está en desarrollo.");
assert.strictEqual(i18n.t("en", "sendSoon"), "Sending opens soon");
assert.strictEqual(i18n.t("es", "sendSoon"), "El envío abre pronto");
assert.strictEqual(i18n.t("en", "boardAccuracy"), "Accuracy");
assert.strictEqual(i18n.t("es", "boardAccuracy"), "Precisión");
assert.strictEqual(i18n.t("en", "back"), "← Back");
assert.strictEqual(i18n.t("es", "back"), "← Volver");

const card = share.cardCopy(i18n, "en", 12, "Intermedio");
assert.strictEqual(card.host, "https://opencourtvision-10a09b.gitlab.io");
assert.ok(card.line.includes("12"));
assert.ok(!card.host.includes("antonyvo.gitlab.io"));

assert.strictEqual(form.submitEnabled, false);
assert.strictEqual(submit.submitReady(form), false);
assert.strictEqual(submit.submitReady({ submitEnabled: true, formAction: "", fields: {} }), false);
const on = Object.assign({}, form, { submitEnabled: true });
const batch = [];
for (let i = 1; i <= 10; i += 1) {
  batch.push({ point_id: "se-p" + String(i).padStart(2, "0"), call: ["A", "B", "F"][(i - 1) % 3] });
}
const store = { bag: {}, getItem(k) { return this.bag[k] || null; }, setItem(k, v) { this.bag[k] = v; } };
let n = 0;
const storedId = submit.ensureDeviceId(store, () => "device-" + (++n));
assert.strictEqual(submit.ensureDeviceId(store, () => "other"), storedId);
assert.strictEqual(storedId, "device-1");
const spec = submit.encodeBody(on, { nickname: "se-test", round_id: "se-test-round", calls: batch, device_id: storedId });
assert.strictEqual(spec.mode, "no-cors");
assert.strictEqual(spec.credentials, "omit");
assert.strictEqual(spec.contentType, "application/x-www-form-urlencoded");
assert.ok(spec.json.length <= 2000);
const params = new URLSearchParams(spec.body);
assert.deepStrictEqual([...params.keys()], [
  "entry.343297185", "entry.787571459", "entry.1118633972", "entry.1163912343", "entry.806055965"
]);
assert.strictEqual(params.get("entry.806055965"), storedId);
assert.strictEqual(form.fields.deviceId, "entry.806055965");
const unset = submit.encodeBody(
  Object.assign({}, on, { fields: Object.assign({}, on.fields, { deviceId: "entry.DEVICE_ID_PLACEHOLDER" }) }),
  { nickname: "se-test", round_id: "se-test-round", calls: batch, device_id: storedId }
);
assert.ok(!new URLSearchParams(unset.body).has("entry.DEVICE_ID_PLACEHOLDER"));
assert.ok(!unset.body.includes(storedId));
const del = submit.deleteMailto("se-test", storedId);
assert.ok(del.startsWith("mailto:antonyvo@mail.grokbot.com?subject="));
assert.ok(decodeURIComponent(del).includes("device_id: " + storedId));
assert.ok(decodeURIComponent(del).includes("nickname: se-test"));
assert.ok(decodeURIComponent(del).includes("Delete my labels"));
const callsMail = submit.mailtoUrl("se-test", spec.json);
assert.ok(!decodeURIComponent(callsMail).includes(storedId));
assert.ok(!card.host.includes(storedId));
assert.ok(!card.line.includes(storedId));
assert.strictEqual(params.get("entry.343297185"), "se-test");
assert.strictEqual(params.get("entry.1118633972"), "I'm 18 or older");
assert.strictEqual(params.get("entry.1163912343"), form.clientVersion);
const parsed = JSON.parse(params.get("entry.787571459"));
assert.strictEqual(parsed.round_id, "se-test-round");
assert.strictEqual(parsed.calls.length, 10);
assert.ok(parsed.calls.every((row) => row.point_id && row.call));
assert.ok(!Object.prototype.hasOwnProperty.call(parsed, "email"));
const fs = require("fs");
const clipsDoc = JSON.parse(fs.readFileSync("labeler-pages/clips.json", "utf8"));
clipsDoc.clips.forEach((clip) => {
  const line = round.creditLineText(clipsDoc.videos, clip.video, "en");
  assert.ok(line.includes("CC BY"), clip.id);
  assert.ok(line.includes("Source:"), clip.id);
  const video = clipsDoc.videos.find((v) => v.id === clip.video);
  assert.ok(line.includes(video.title), clip.id);
});
console.log("FORM_BODY " + spec.body);
