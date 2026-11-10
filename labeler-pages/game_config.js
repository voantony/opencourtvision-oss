"use strict";
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OCV_GAME_CONFIG = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  return {
    xpPerCall: 10,
    roundSize: 10,
    minChecks: 10,
    grades: [
      { id: "iniciacion", name: "Iniciación", points: 1 },
      { id: "intermedio", name: "Intermedio", points: 10 },
      { id: "avanzado", name: "Avanzado", points: 50 },
      { id: "pro", name: "Pro", points: 200 }
    ]
  };
});
