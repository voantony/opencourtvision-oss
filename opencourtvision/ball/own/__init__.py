"""Vamos ball tracker (own implementation, Apache-2.0 like the rest of the repo; no third-party code or weights).

physics  numpy-only arc segmentation, physics-densified pseudo-labels, bounce / hit events from direction changes
         near player boxes.
heatnet  a small heatmap U-Net over 3 stacked frames (TrackNet-style input/output, written from the papers'
         descriptions: Huang et al. 2019, arXiv 1907.03698; Sun et al. 2020), sized for on-device real time.
         torch is imported lazily so the physics part and the tests run without it.
"""
