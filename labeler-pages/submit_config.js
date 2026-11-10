/* Ops flips submitEnabled after a hand-sent round lands in the form.
   Leave it false in git. Do not add fields. */
(function (root, factory) {
  var config = factory();
  if (typeof module === "object" && module.exports) module.exports = config;
  root.OCV_SUBMIT_CONFIG = config;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  return {
    submitEnabled: false,
    formAction: "https://docs.google.com/forms/d/e/1FAIpQLScCAE6SFnNN1I5kPnSgprXXybDimnRO1fC-Wyp5UkTAY3_fZA/formResponse",
    fields: {
      nickname: "entry.343297185",
      calls: "entry.787571459",
      adult18: "entry.1118633972",
      clientVersion: "entry.1163912343"
    },
    adult18Value: "I'm 18 or older",
    clientVersion: "2026.11.10"
  };
});
