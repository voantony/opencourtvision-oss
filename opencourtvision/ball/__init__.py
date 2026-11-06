"""Ball engine v2 (Step 1 of the engine rewrite): TrackNetV3 re-acquire + ROI/Kalman HSV tracking + live-ball filter.

Modules: tracknet3 (vendored MIT TrackNetV3 + weight fetch), hsv (colour/motion candidates), geometry (court gate,
size gate), live (live-ball filter), bounce (bounce detection + tennis-vision cleanup), tracker (BallEngine),
proxy (label-free proxy metrics for ocv-bench-ball).
"""
