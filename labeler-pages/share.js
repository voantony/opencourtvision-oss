"use strict";
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.OCVShare = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var SIZES = {
    fourfive: [1080, 1350],
    story: [1080, 1920]
  };
  var HOST = "https://opencourtvision-10a09b.gitlab.io";

  function cardCopy(i18n, lang, n, grade) {
    var t = i18n.t(lang, "shareLine");
    var line = i18n.fill(t, { n: n });
    var model = {
      titleEn: "Call the Point",
      titleEs: "¡Canta el punto!",
      line: line,
      host: HOST,
      grade: grade || ""
    };
    var blob = JSON.stringify(model).toLowerCase();
    if (blob.indexOf("accuracy") >= 0 || blob.indexOf("prize") >= 0 || blob.indexOf("premio") >= 0) {
      throw new Error("share card must not mention accuracy or prizes");
    }
    return model;
  }

  function draw(canvas, model, sizeName) {
    var size = SIZES[sizeName];
    canvas.width = size[0];
    canvas.height = size[1];
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0B0F14";
    ctx.fillRect(0, 0, size[0], size[1]);
    var y0 = sizeName === "story" ? 280 : 160;
    ctx.strokeStyle = "#F2F4F7";
    ctx.lineWidth = 8;
    ctx.strokeRect(140, y0, 800, sizeName === "story" ? 980 : 640);
    ctx.beginPath();
    ctx.moveTo(140, y0 + (sizeName === "story" ? 490 : 320));
    ctx.lineTo(940, y0 + (sizeName === "story" ? 490 : 320));
    ctx.stroke();
    ctx.fillStyle = "#C8F43C";
    ctx.beginPath();
    ctx.arc(700, y0 + 220, 28, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#FFB627";
    ctx.beginPath();
    ctx.arc(700, y0 + 220, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#F2F4F7";
    ctx.font = "italic 900 92px 'Barlow Condensed', Impact, sans-serif";
    ctx.fillText(model.titleEn, 120, size[1] - 420);
    ctx.fillStyle = "#C8F43C";
    ctx.font = "italic 900 64px 'Barlow Condensed', Impact, sans-serif";
    ctx.fillText(model.titleEs, 120, size[1] - 330);
    ctx.fillStyle = "#F2F4F7";
    ctx.font = "600 48px Inter, sans-serif";
    wrap(ctx, model.line, 120, size[1] - 250, 840, 58);
    if (model.grade) {
      ctx.fillStyle = "#FFB627";
      ctx.font = "700 36px 'JetBrains Mono', monospace";
      ctx.fillText(model.grade, 120, size[1] - 140);
    }
    ctx.fillStyle = "#8E99AB";
    ctx.font = "700 32px 'JetBrains Mono', monospace";
    ctx.fillText(model.host, 120, size[1] - 80);
  }

  function wrap(ctx, text, x, y, max, lh) {
    var words = String(text).split(" ");
    var line = "";
    words.forEach(function (w) {
      var trial = line ? line + " " + w : w;
      if (ctx.measureText(trial).width > max) {
        ctx.fillText(line, x, y);
        line = w;
        y += lh;
      } else line = trial;
    });
    if (line) ctx.fillText(line, x, y);
  }

  return { SIZES: SIZES, HOST: HOST, cardCopy: cardCopy, draw: draw };
});
