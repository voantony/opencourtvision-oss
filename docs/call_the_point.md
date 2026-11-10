# Call the Point

The labeling game is a static page under `label/` on the Pages site. It is still a draft: the three warm-up points in `labeler-pages/practice.json` are a synthetic example. Real session-02 practice points are not in this tree yet.

A round is 10 public CC BY clips. Warm-up answers stay on the device and are not part of the submission. Hidden check answers and gold answers are not shipped to the browser. An engine pick, when one exists, is shown after every call, including checks, so a check clip looks like any other clip. The call locks before that line appears.

Grades (Iniciación, Intermedio, Avanzado, Pro) follow how many points you have called. The leaderboard is opt-in accuracy on hidden checks only, after a minimum number of those checks. Calling a lot of points does not earn a rank.

## Sending a round

`labeler-pages/submit_config.js` holds the Google Form endpoint and entry ids. `submitEnabled` stays `false` until Ops has seen one round land, then Ops turns it on. The button posts only the nickname, the calls JSON (`round_id` plus `{point_id, call}`), the adult confirmation `I'm 18 or older`, and the client version. Nothing else.

The small link "Didn't work? Email instead" opens a mailto to antonyvo@mail.grokbot.com. The sender's email address is deleted on intake.

Do not POST the form from CI or from the test suite.
