"""Jersey-color team clustering and cluster-level side assignment."""

import numpy as np
import pytest

from nfl_coverage.vision.teams import cluster_teams, sides_from_clusters, torso_colors

GRASS = (60, 140, 70)  # BGR green
RED = (40, 40, 200)
WHITE = (235, 235, 235)


def _image_with_boxes(colors, size=(400, 600)):
    """Grass-colored image with a 30x50 solid rectangle per color; returns boxes."""
    img = np.full((*size, 3), GRASS, dtype=np.uint8)
    boxes = []
    for i, c in enumerate(colors):
        x1, y1 = 20 + (i % 8) * 70, 30 + (i // 8) * 120
        img[y1 : y1 + 50, x1 : x1 + 30] = c
        boxes.append((x1, y1, x1 + 30, y1 + 50))
    return img, boxes


def test_two_teams_cluster_cleanly():
    img, boxes = _image_with_boxes([RED] * 5 + [WHITE] * 5)
    labels, is_player = cluster_teams(torso_colors(img, boxes))
    assert is_player.all()
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_flat_gray_is_a_lighting_variant_of_white_not_an_outlier():
    # a flat mid-gray torso is what a shaded white jersey looks like in hard sun;
    # officials are caught by their stripes (referee_mask), not by grayness
    img, boxes = _image_with_boxes([RED] * 5 + [WHITE] * 5 + [(120, 120, 120)])
    labels, is_player = cluster_teams(torso_colors(img, boxes))
    assert is_player.all()
    assert labels[10] == labels[5]


def test_lone_extreme_outlier_does_not_hijack_the_split():
    # a bright magenta UI element among two real teams must not become a "team"
    img, boxes = _image_with_boxes([RED] * 5 + [WHITE] * 5 + [(255, 0, 255)])
    labels, is_player = cluster_teams(torso_colors(img, boxes))
    assert is_player[:10].all() and not is_player[10]
    assert labels[0] != labels[5]


def test_sides_majority_vote_overrides_stragglers():
    # cluster 0 mostly left of LOS but one member projected past it (homography
    # error) — the whole cluster must still be offense
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    field_x = np.array([40.0, 42.0, 44.0, 45.8, 46.0, 47.0, 50.0, 55.0])
    sides = sides_from_clusters(labels, field_x, los=45.2, offense="left")
    assert list(sides[:4]) == ["Offense"] * 4
    assert list(sides[4:]) == ["Defense"] * 4


def test_sides_offense_right():
    labels = np.array([0, 0, 1, 1])
    field_x = np.array([80.0, 78.0, 70.0, 68.0])
    sides = sides_from_clusters(labels, field_x, los=75.0, offense="right")
    assert list(sides) == ["Offense", "Offense", "Defense", "Defense"]


def test_geometry_overrides_color_misread_away_from_los():
    # a washed-out deep safety (12 yd past the LOS) mis-clustered with the white
    # offense must still come out Defense: geometry is certain that far out
    labels = np.array([0, 0, 0, 1, 1, 0])
    field_x = np.array([40.0, 42.0, 44.9, 48.0, 52.0, 58.0])
    sides = sides_from_clusters(labels, field_x, los=45.2, offense="left")
    assert sides[-1] == "Defense"
    assert list(sides[:3]) == ["Offense"] * 3


def test_too_few_detections_rejected():
    with pytest.raises(ValueError, match="at least 4"):
        cluster_teams(np.zeros((3, 3)))
