"""Officials by stripe-gray torso; post-snap sides by pre-snap team colors."""

import cv2
import numpy as np

from nfl_coverage.vision.teams import TeamColors, cluster_teams, referee_mask, torso_colors

WHITE, ORANGE, BLACK = (245, 245, 245), (30, 90, 245), (20, 20, 20)  # BGR
GRASS = (40, 140, 40)


def _canvas(n: int, w: int = 60, h: int = 120):
    img = np.zeros((h + 20, n * (w + 20) + 20, 3), dtype=np.uint8)
    img[:] = GRASS
    boxes = [(20 + i * (w + 20), 10, 20 + i * (w + 20) + w, 10 + h) for i in range(n)]
    return img, boxes


def _fill(img, box, color):
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = color


def _stripes(img, box, period=6):
    x1, y1, x2, y2 = box
    for x in range(x1, x2):  # officials wear vertical stripes
        img[y1:y2, x] = BLACK if (x // period) % 2 else WHITE
    return cv2.GaussianBlur(img, (3, 3), 0)  # mild broadcast blur


def test_referee_mask_flags_stripes_not_jerseys():
    img, boxes = _canvas(4)
    _fill(img, boxes[0], WHITE)
    _fill(img, boxes[1], ORANGE)
    _fill(
        img, boxes[2], (120, 120, 120)
    )  # flat gray: could be a shaded white jersey, not an official
    img = _stripes(img, boxes[3])
    assert referee_mask(img, boxes).tolist() == [False, False, False, True]


def test_shadowed_white_jersey_is_not_a_referee():
    img, boxes = _canvas(1)
    _fill(img, boxes[0], (200, 205, 215))  # dim but warm white, no dark stripes
    assert not referee_mask(img, boxes)[0]


def test_reference_assigns_post_snap_sides_by_color_only():
    # pre-snap: 5 white on the left (offense), 5 orange on the right
    img, boxes = _canvas(10)
    for b in boxes[:5]:
        _fill(img, b, WHITE)
    for b in boxes[5:]:
        _fill(img, b, ORANGE)
    colors = torso_colors(img, boxes)
    labels, is_player = cluster_teams(colors)
    sides = np.array(["Offense"] * 5 + ["Defense"] * 5)
    ref = TeamColors.from_frame(colors, labels, is_player, sides)

    # post-snap: an orange body and a white body in any position; a purple stranger
    img2, boxes2 = _canvas(3)
    _fill(img2, boxes2[0], ORANGE)
    _fill(img2, boxes2[1], WHITE)
    _fill(img2, boxes2[2], (200, 60, 120))  # matches neither team
    sides2, players2 = ref.assign(torso_colors(img2, boxes2))
    assert sides2[:2].tolist() == ["Defense", "Offense"]
    assert players2.tolist() == [True, True, False]


def test_two_tone_jersey_reads_as_its_base_color_not_a_referee():
    # royal blue base with big yellow numbers: 70/30 split like the Rams on film
    img, boxes = _canvas(1)
    x1, y1 = boxes[0][:2]
    _fill(img, boxes[0], (200, 60, 0))  # BGR royal blue
    img[y1 + 40 : y1 + 55, x1 + 15 : x1 + 45] = (0, 220, 255)  # yellow number, ~30% of torso
    desc = torso_colors(img, boxes)[0]
    assert desc[2] < 110  # dark mode is the blue base (Lab b well below neutral)
    assert desc[5] > 150  # bright mode is the yellow number
    assert not referee_mask(img, boxes)[0]


def test_shaded_white_jersey_is_not_a_referee_and_clusters_with_lit_one():
    # the sun splits a white torso into a lit white half and a dark gray half
    img, boxes = _canvas(3)
    _fill(img, boxes[0], WHITE)
    x1, y1, x2, _ = boxes[1]
    _fill(img, boxes[1], WHITE)
    img[y1 : y1 + 65, x1:x2] = (95, 95, 95)  # top half in shadow
    _fill(img, boxes[2], ORANGE)
    assert not referee_mask(img, boxes)[1]
    labels, is_player = cluster_teams(np.vstack([torso_colors(img, boxes)] * 2))
    assert is_player.all()
    assert labels[0] == labels[1] != labels[2]
