# Scoring a model

A point-winner model is scored on a private held-out set. That set is not in git and it is not published. The public result is an aggregate for the model leaderboard: recognition only, no prizes, no point-level answers, no footage.

Private match data is never shared. Do not send private video, private labels, or the names of people on court.

What you submit is a `predict(points)` function. What comes back is the aggregate. The runner lives with the eval-submission tool (`tools/eval_submission.py`) when that file is in your tree. The held-out file is passed in locally. It is not read from the repository.
