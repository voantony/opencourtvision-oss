"use strict";
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OCVSubmit = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var FIELD_ORDER = ["nickname", "calls", "adult18", "clientVersion"];

  function submitReady(config) {
    if (!config || config.submitEnabled !== true) return false;
    if (!config.formAction) return false;
    var fields = config.fields || {};
    for (var i = 0; i < FIELD_ORDER.length; i++) {
      if (!fields[FIELD_ORDER[i]]) return false;
    }
    if (!config.adult18Value || !config.clientVersion) return false;
    return true;
  }

  function callsJson(round) {
    var calls = (round.calls || []).map(function (c) {
      return { point_id: c.point_id, call: c.call };
    });
    return JSON.stringify({ round_id: round.round_id, calls: calls });
  }

  function deviceFieldActive(entry) {
    return /^entry\.\d+$/.test(entry || "");
  }

  function ensureDeviceId(storage, randomUUID) {
    var key = "ocv-device-id";
    var cur = storage.getItem(key);
    if (cur) return cur;
    var id = randomUUID();
    storage.setItem(key, id);
    return id;
  }

  function deleteMailto(nickname, deviceId) {
    var body = "I want anything I've shared with the OpenCourtVision / Vamos maintainers deleted.\n\n"
      + "nickname: " + (nickname || "") + "\n"
      + "device_id: " + deviceId + "\n";
    return "mailto:antonyvo@mail.grokbot.com?subject=" + encodeURIComponent("Delete my labels")
      + "&body=" + encodeURIComponent(body);
  }

  function encodeBody(config, round) {
    var json = callsJson(round);
    if (json.length > 2000) throw new Error("batch over 2000 characters");
    var fields = config.fields;
    var seen = {};
    FIELD_ORDER.forEach(function (name) {
      var key = fields[name];
      if (seen[key]) throw new Error("duplicate form field");
      seen[key] = true;
    });
    var sendDevice = deviceFieldActive(fields.deviceId) && round.device_id;
    if (sendDevice && seen[fields.deviceId]) throw new Error("duplicate form field");
    var params = new URLSearchParams();
    params.set(fields.nickname, round.nickname);
    params.set(fields.calls, json);
    params.set(fields.adult18, config.adult18Value);
    params.set(fields.clientVersion, config.clientVersion);
    if (sendDevice) params.set(fields.deviceId, round.device_id);
    var body = params.toString();
    var parsed = new URLSearchParams(body);
    var keys = [];
    parsed.forEach(function (_v, k) { keys.push(k); });
    if (keys.length !== (sendDevice ? 5 : 4)) throw new Error("payload has extra fields");
    FIELD_ORDER.forEach(function (name) {
      if (keys.indexOf(fields[name]) < 0) throw new Error("missing field");
    });
    return {
      url: config.formAction,
      method: "POST",
      mode: "no-cors",
      credentials: "omit",
      contentType: "application/x-www-form-urlencoded",
      body: body,
      json: json
    };
  }

  function issueUrl(repo, nickname, json) {
    var description = "/confidential\n\nVolunteer calls from " + nickname + ".\n\n```json\n" + json + "\n```\n";
    return repo + "/-/issues/new?issue[title]=" + encodeURIComponent("Calls from " + nickname) +
      "&issue[description]=" + encodeURIComponent(description);
  }

  function mailtoUrl(nickname, json) {
    var body = "nickname: " + nickname + "\n" + json + "\n";
    return "mailto:antonyvo@mail.grokbot.com?subject=" + encodeURIComponent("Call the Point calls") +
      "&body=" + encodeURIComponent(body);
  }

  return {
    submitReady: submitReady,
    callsJson: callsJson,
    encodeBody: encodeBody,
    issueUrl: issueUrl,
    mailtoUrl: mailtoUrl,
    deviceFieldActive: deviceFieldActive,
    ensureDeviceId: ensureDeviceId,
    deleteMailto: deleteMailto,
    FIELD_ORDER: FIELD_ORDER
  };
});
