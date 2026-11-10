"use strict";
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OCV_I18N = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var SHARE_HOST = "antonyvo.gitlab.io/opencourtvision";
  var copy = {
    en: {
      title: "Call the Point",
      nickname: "Nickname (optional)",
      adult: "I'm 18+",
      play: "Play",
      honest: "Public Creative Commons (CC BY) clips. Vamos is a work in progress.",
      warmupCount: "Warm-up {i} of {n}",
      pointCount: "Point {i} of {n}",
      callA: "A won",
      callB: "B won",
      callF: "Fault serve",
      reminder: "A is the near team in the match's first clip. After the teams switch ends, follow the shirt colour.",
      was: "It was {call}.",
      wasRight: "Right, it was {call}.",
      faultShort: "Fault",
      called: "Called!",
      xp: "+{n} XP",
      engine: "The engine called {call}",
      engineDash: "The engine called —",
      faultName: "Fault serve",
      summaryCount: "{n} points called",
      beat: "You beat the engine on — of 10",
      credits: "Add me to credits",
      more: "Play 10 more",
      send: "Send my calls",
      resend: "Resend",
      sent: "Sent",
      sendNote: "Sends your nickname and calls only. No email, no video.",
      email: "Didn't work? Email instead",
      share: "Share",
      shareLine: "I called {n} points on Vamos",
      boardTitle: "Leaderboard",
      boardHint: "Opt-in accuracy on hidden checks.",
      boardName: "Nickname",
      boardAccuracy: "Accuracy",
      skip: "Skip warm-up",
      next: "Next",
      summaryGo: "Round summary",
      sendSoon: "Sending opens soon",
      back: "← Back"
    },
    es: {
      title: "¡Canta el punto!",
      nickname: "Apodo (opcional)",
      adult: "Soy mayor de 18 años",
      play: "Jugar",
      honest: "Clips públicos de Creative Commons (CC BY). Vamos está en desarrollo.",
      warmupCount: "Calentamiento {i} de {n}",
      pointCount: "Punto {i} de {n}",
      callA: "Ganó A",
      callB: "Ganó B",
      callF: "Saque falta",
      reminder: "A es el equipo cercano en el primer clip del partido. Si cambian de lado, sigue el color de la camiseta.",
      was: "Era {call}.",
      wasRight: "Bien, era {call}.",
      faultShort: "falta",
      called: "¡Cantado!",
      xp: "+{n} XP",
      engine: "El motor cantó {call}",
      engineDash: "El motor cantó —",
      faultName: "Saque falta",
      summaryCount: "{n} puntos cantados",
      beat: "Le ganaste al motor en — de 10",
      credits: "Incluidme en los créditos",
      more: "Jugar 10 más",
      send: "Enviar mis cantos",
      resend: "Reenviar",
      sent: "Enviado",
      sendNote: "Envía solo tu apodo y tus cantos. Sin correo, sin vídeo.",
      email: "¿No funcionó? Enviar por correo",
      share: "Compartir",
      shareLine: "Canté {n} puntos en Vamos",
      boardTitle: "Clasificación",
      boardHint: "Precisión con opt-in sobre las comprobaciones ocultas.",
      boardName: "Apodo",
      boardAccuracy: "Precisión",
      skip: "Saltar calentamiento",
      next: "Siguiente",
      summaryGo: "Resumen de la ronda",
      sendSoon: "El envío abre pronto",
      back: "← Volver"
    }
  };
  function t(lang, key) { return (copy[lang] || copy.en)[key]; }
  function fill(s, map) {
    return String(s).replace(/\{(\w+)\}/g, function (_m, k) { return map[k] != null ? map[k] : ""; });
  }
  return { copy: copy, t: t, fill: fill, SHARE_HOST: SHARE_HOST };
});
