# labels/ball/: ball position ground truth for `ocv-bench-ball --labels`

Made with `ocv-label-ball` (template + contact sheets, or the click UI on a machine with a display). One CSV per clip:

```
clip,frame,t,x,y,visibility,status
16,0,139.1670,512.0,233.5,1,ok
16,3,139.2670,,,0,ok
```

- `t` is on the video's clock, `x, y` in pixels of the labelling frame size (default 960×540; pass the same `--size`
  to `ocv-bench-ball`), `visibility` 1/0, `status` `ok` (checked), `pred` (unchecked pre-fill) or `skip`.
- Only `status=ok` rows are ground truth.
- Real match labels are private: `labels/ball/*.csv` is git-ignored. Keep them here locally or in your work dir.
