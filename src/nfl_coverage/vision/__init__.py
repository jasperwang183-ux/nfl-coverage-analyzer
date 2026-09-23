"""Phase 2: computer vision pipeline — footage to field-space tracks.

Modules:
  - detect.py: YOLO person detection, foot points, field-space dedupe
  - homography.py: pixel-to-field calibration from clicked point pairs
  - teams.py: jersey-color clustering -> offense/defense/referee
  - still.py: field points -> canonical rel_frame-0 frame; line-player filter
  - visibility.py: field-visibility guard (is the analysis region in frame?)
  - track.py (planned): multi-object tracking across clip frames
"""
