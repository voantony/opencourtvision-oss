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
  answer: "A", engine: "B", reason_en: "The near side hit a clean winner.", reason_es: "El lado cercano."
}, "B", "en");
assert.strictEqual(warmWrong.showEngine, false);
assert.strictEqual(warmWrong.text, "It was A.");
assert.ok(!/engine/i.test(warmWrong.text));
assert.strictEqual(round.playerFeedback("warmup", { answer: "A", reason_en: "x" }, "A", "en").text, "Right, it was A.");
assert.strictEqual(round.playerFeedback("warmup", { answer: "F", reason_es: "red" }, "A", "es").text, "Era falta.");
assert.strictEqual(round.playerFeedback("warmup", { answer: "B", reason_es: "x" }, "B", "es").text, "Bien, era B.");
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
assert.strictEqual(card.host, "antonyvo.gitlab.io/opencourtvision");
assert.ok(card.line.includes("12"));
assert.ok(!card.host.startsWith("http"));

assert.strictEqual(form.submitEnabled, false);
assert.strictEqual(submit.submitReady(form), false);
assert.strictEqual(submit.submitReady({ submitEnabled: true, formAction: "", fields: {} }), false);
const on = Object.assign({}, form, { submitEnabled: true });
const batch = [];
for (let i = 1; i <= 10; i += 1) {
  batch.push({ point_id: "se-p" + String(i).padStart(2, "0"), call: ["A", "B", "F"][(i - 1) % 3] });
}
const spec = submit.encodeBody(on, { nickname: "se-test", round_id: "se-test-round", calls: batch });
assert.strictEqual(spec.mode, "no-cors");
assert.strictEqual(spec.credentials, "omit");
assert.strictEqual(spec.contentType, "application/x-www-form-urlencoded");
assert.ok(spec.json.length <= 2000);
const params = new URLSearchParams(spec.body);
assert.deepStrictEqual([...params.keys()], [
  "entry.343297185", "entry.787571459", "entry.1118633972", "entry.1163912343"
]);
assert.strictEqual(params.get("entry.343297185"), "se-test");
assert.strictEqual(params.get("entry.1118633972"), "I'm 18 or older");
assert.strictEqual(params.get("entry.1163912343"), form.clientVersion);
const parsed = JSON.parse(params.get("entry.787571459"));
assert.strictEqual(parsed.round_id, "se-test-round");
assert.strictEqual(parsed.calls.length, 10);
assert.ok(parsed.calls.every((row) => row.point_id && row.call));
assert.ok(!Object.prototype.hasOwnProperty.call(parsed, "email"));
console.log("FORM_BODY " + spec.body);
