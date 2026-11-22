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
    streakBonus: [
      { streak: 3, bonus: 5 },
      { streak: 5, bonus: 10 },
      { streak: 10, bonus: 20 }
    ],
    grades: [
      { id: "iniciacion", name: "Iniciación", points: 1 },
      { id: "intermedio", name: "Intermedio", points: 10 },
      { id: "avanzado", name: "Avanzado", points: 50 },
      { id: "pro", name: "Pro", points: 200 }
    ]
  };
});
