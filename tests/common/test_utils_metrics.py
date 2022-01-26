import numpy as np
import pytest

from deepocr.utils import metrics


@pytest.mark.parametrize(
    "gt, pred, raw, caseless, unidecode, unicase",
    [
        [['grass', '56', 'True', 'EUR'], ['grass', '56', 'true', '€'], .5, .75, .75, 1],
        [['éléphant', 'ça'], ['elephant', 'ca'], 0, 0, 1, 1],
    ],
)
def test_text_match(gt, pred, raw, caseless, unidecode, unicase):
    metric = metrics.TextMatch()
    with pytest.raises(AssertionError):
        metric.summary()

    with pytest.raises(AssertionError):
        metric.update(['a', 'b'], ['c'])

    metric.update(gt, pred)
    assert metric.summary() == dict(raw=raw, caseless=caseless, unidecode=unidecode, unicase=unicase)

    metric.reset()
    assert metric.raw == metric.caseless == metric.unidecode == metric.unicase == metric.total == 0


@pytest.mark.parametrize(
    "box1, box2, iou, abs_tol",
    [
        [[[0, 0, .5, .5]], [[0, 0, .5, .5]], 1, 0],  # Perfect match
        [[[0, 0, .5, .5]], [[.5, .5, 1, 1]], 0, 0],  # No match
        [[[0, 0, 1, 1]], [[.5, .5, 1, 1]], 0.25, 0],  # Partial match
        [[[.2, .2, .6, .6]], [[.4, .4, .8, .8]], 4 / 28, 1e-7],  # Partial match
        [[[0, 0, .1, .1]], [[.9, .9, 1, 1]], 0, 0],  # Boxes far from each other
        [np.zeros((0, 4)), [[0, 0, .5, .5]], 0, 0],  # Zero-sized inputs
        [[[0, 0, .5, .5]], np.zeros((0, 4)), 0, 0],  # Zero-sized inputs
    ],
)
def test_box_iou(box1, box2, iou, abs_tol):
    iou_mat = metrics.box_iou(np.asarray(box1), np.asarray(box2))
    assert iou_mat.shape == (len(box1), len(box2))
    if iou_mat.size > 0:
        assert abs(iou_mat - iou) <= abs_tol


@pytest.mark.parametrize(
    "mask1, mask2, iou, abs_tol",
    [
        [
            [[[True, True, False], [True, True, False]]],
            [[[True, True, False], [True, True, False]]],
            1,
            0
        ],  # Perfect match
        [
            [[[True, False, False], [False, False, False]]],
            [[[True, True, False], [True, True, False]]],
            0.25,
            0
        ],  # Partial match
    ],
)
def test_mask_iou(mask1, mask2, iou, abs_tol):
    iou_mat = metrics.mask_iou(np.asarray(mask1), np.asarray(mask2))
    assert iou_mat.shape == (len(mask1), len(mask2))
    if iou_mat.size > 0:
        assert abs(iou_mat - iou) <= abs_tol

    # Incompatible spatial shapes
    with pytest.raises(AssertionError):
        metrics.mask_iou(np.zeros((2, 3, 5), dtype=bool), np.ones((3, 2, 5), dtype=bool))


@pytest.mark.parametrize(
    "rbox1, rbox2, iou, abs_tol",
    [
        [[[[0, 0], [.5, 0], [.5, .5], [0, .5]]], [[[0, 0], [.5, 0], [.5, .5], [0, .5]]], 1, 0],  # Perfect match
        [[[[0, 0], [.5, 0], [.5, .5], [0, .5]]], [[[.5, .5], [1, .5], [1, 1], [.5, 1]]], 0, 1e-4],  # No match
        [[[[0, 0], [1., 0], [1., 1.], [0, 1.]]], [[[.5, .5], [1, .5], [1., 1.], [.5, 1]]], 0.25, 5e-3],  # Partial match
        [
            [[[.2, .2], [.6, .2], [.6, .6], [.2, .6]]],
            [[[.4, .4], [.8, .4], [.8, .8], [.4, .8]]], 4 / 28, 7e-3
        ],  # Partial match
        [
            [[[0, 0], [.05, 0], [.05, .05], [0, .05]]],
            [[[.5, .5], [1, .5], [1, 1], [.5, 1]]], 0, 0
        ],  # Boxes far from each other
        [np.zeros((0, 4, 2)), [[[0, 0], [.05, 0], [.05, .05], [0, .05]]], 0, 0],  # Zero-sized inputs
        [[[[0, 0], [.05, 0], [.05, .05], [0, .05]]], np.zeros((0, 4, 2)), 0, 0],  # Zero-sized inputs
    ],
)
def test_polygon_iou(rbox1, rbox2, iou, abs_tol):
    mask_shape = (256, 256)
    iou_mat = metrics.polygon_iou(np.asarray(rbox1), np.asarray(rbox2), mask_shape)
    assert iou_mat.shape == (len(rbox1), len(rbox2))
    if iou_mat.size > 0:
        assert abs(iou_mat - iou) <= abs_tol

    # Ensure broadcasting doesn't change the result
    iou_matbis = metrics.polygon_iou(np.asarray(rbox1), np.asarray(rbox2), mask_shape, use_broadcasting=False)
    assert np.all((iou_mat - iou_matbis) <= 1e-7)

    # Incorrect boxes
    with pytest.raises(AssertionError):
        metrics.polygon_iou(np.zeros((2, 5), dtype=float), np.ones((3, 4), dtype=float), mask_shape)


@pytest.mark.parametrize(
    "box, shape, mask",
    [
        [
            [[0, 0], [.5, 0], [.5, .5], [0, .5]], (2, 2),
            [[True, False], [False, False]],
        ],
    ],
)
def test_rbox_to_mask(box, shape, mask):
    masks = metrics.rbox_to_mask(np.asarray(box)[None, ...], shape)
    assert masks.shape == (1, *shape)
    assert np.all(masks[0] == np.asarray(mask, dtype=bool))


@pytest.mark.parametrize(
    "gts, preds, iou_thresh, recall, precision, mean_iou",
    [
        [[[[0, 0, .5, .5]]], [[[0, 0, .5, .5]]], 0.5, 1, 1, 1],  # Perfect match
        [[[[0, 0, 1, 1]]], [[[0, 0, .5, .5], [.6, .6, .7, .7]]], 0.2, 1, 0.5, 0.13],  # Bad match
        [[[[0, 0, 1, 1]]], [[[0, 0, .5, .5], [.6, .6, .7, .7]]], 0.5, 0, 0, 0.13],  # Bad match
        [[[[0, 0, .5, .5]], [[0, 0, .5, .5]]], [[[0, 0, .5, .5]], None], 0.5, 0.5, 1, 1],  # No preds on 2nd sample
    ],
)
def test_localization_confusion(gts, preds, iou_thresh, recall, precision, mean_iou):

    metric = metrics.LocalizationConfusion(iou_thresh)
    for _gts, _preds in zip(gts, preds):
        metric.update(np.asarray(_gts), np.zeros((0, 4)) if _preds is None else np.asarray(_preds))
    assert metric.summary() == (recall, precision, mean_iou)
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.matches == metric.tot_iou == 0


@pytest.mark.parametrize(
    "gts, preds, iou_thresh, recall, precision, mean_iou",
    [
        [
            [[[[.05, .05], [.15, .05], [.15, .15], [.05, .15]]]],
            [[[[.05, .05], [.15, .05], [.15, .15], [.05, .15]]]], 0.5, 1, 1, 1
        ],  # Perfect match
        [
            [[[[.1, .05], [.2, .05], [.2, .15], [.1, .15]]]],
            [[[[.1, .05], [.3, .05], [.3, .15], [.1, .15]], [[.6, .6], [.8, .6], [.8, .8], [.6, .8]]]],
            0.2, 1, 0.5, 0.25
        ],  # Bad match
        [
            [[[[.05, .05], [.15, .05], [.15, .15], [.05, .15]]],
             [[[.25, .25], [.35, .25], [35, .35], [.25, .35]]]],
            [[[[.05, .05], [.15, .05], [.15, .15], [.05, .15]]], None],
            0.5, 0.5, 1, 1,
        ],  # Empty
    ],
)
def test_r_localization_confusion(gts, preds, iou_thresh, recall, precision, mean_iou):

    metric = metrics.LocalizationConfusion(iou_thresh, use_polygons=True, mask_shape=(1000, 1000))
    for _gts, _preds in zip(gts, preds):
        metric.update(np.asarray(_gts), np.zeros((0, 5)) if _preds is None else np.asarray(_preds))
    assert metric.summary()[:2] == (recall, precision)
    assert abs(metric.summary()[2] - mean_iou) <= 5e-3
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.matches == metric.tot_iou == 0


@pytest.mark.parametrize(
    "gt_boxes, gt_words, pred_boxes, pred_words, iou_thresh, recall, precision, mean_iou",
    [
        [  # Perfect match
            [[[0, 0, .5, .5]]], [["elephant"]],
            [[[0, 0, .5, .5]]], [["elephant"]],
            0.5,
            {"raw": 1, "caseless": 1, "unidecode": 1, "unicase": 1},
            {"raw": 1, "caseless": 1, "unidecode": 1, "unicase": 1},
            1,
        ],
        [  # Bad match
            [[[0, 0, .5, .5]]], [["elefant"]],
            [[[0, 0, .5, .5]]], [["elephant"]],
            0.5,
            {"raw": 0, "caseless": 0, "unidecode": 0, "unicase": 0},
            {"raw": 0, "caseless": 0, "unidecode": 0, "unicase": 0},
            1,
        ],
        [  # Good match
            [[[0, 0, 1, 1]]], [["EUR"]],
            [[[0, 0, .5, .5], [.6, .6, .7, .7]]], [["€", "e"]],
            0.2,
            {"raw": 0, "caseless": 0, "unidecode": 1, "unicase": 1},
            {"raw": 0, "caseless": 0, "unidecode": .5, "unicase": .5},
            0.13,
        ],
        [  # No preds on 2nd sample
            [[[0, 0, .5, .5]], [[0, 0, .5, .5]]], [["Elephant"], ["elephant"]],
            [[[0, 0, .5, .5]], None], [["elephant"], []],
            0.5,
            {"raw": 0, "caseless": .5, "unidecode": 0, "unicase": .5},
            {"raw": 0, "caseless": 1, "unidecode": 0, "unicase": 1},
            1,
        ],
    ],
)
def test_ocr_metric(
    gt_boxes, gt_words, pred_boxes, pred_words, iou_thresh, recall, precision, mean_iou
):
    metric = metrics.OCRMetric(iou_thresh)
    for _gboxes, _gwords, _pboxes, _pwords in zip(gt_boxes, gt_words, pred_boxes, pred_words):
        metric.update(
            np.asarray(_gboxes),
            np.zeros((0, 4)) if _pboxes is None else np.asarray(_pboxes),
            _gwords,
            _pwords
        )
    _recall, _precision, _mean_iou = metric.summary()
    assert _recall == recall
    assert _precision == precision
    assert _mean_iou == mean_iou
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.tot_iou == 0
    assert metric.raw_matches == metric.caseless_matches == metric.unidecode_matches == metric.unicase_matches == 0
    # Shape check
    with pytest.raises(AssertionError):
        metric.update(
            np.asarray(_gboxes),
            np.zeros((0, 4)),
            _gwords,
            ["I", "have", "a", "bad", "feeling", "about", "this"],
        )


@pytest.mark.parametrize(
    "gt_boxes, gt_classes, pred_boxes, pred_classes, iou_thresh, recall, precision, mean_iou",
    [
        [  # Perfect match
            [[[0, 0, .5, .5]]], [[0]],
            [[[0, 0, .5, .5]]], [[0]],
            0.5, 1, 1, 1,
        ],
        [  # Bad match
            [[[0, 0, .5, .5]]], [[0]],
            [[[0, 0, .5, .5]]], [[1]],
            0.5, 0, 0, 1,
        ],
        [  # No preds on 2nd sample
            [[[0, 0, .5, .5]], [[0, 0, .5, .5]]], [[0], [1]],
            [[[0, 0, .5, .5]], None], [[0], []],
            0.5, .5, 1, 1,
        ],
    ],
)
def test_detection_metric(
    gt_boxes, gt_classes, pred_boxes, pred_classes, iou_thresh, recall, precision, mean_iou
):
    metric = metrics.DetectionMetric(iou_thresh)
    for _gboxes, _gclasses, _pboxes, _pclasses in zip(gt_boxes, gt_classes, pred_boxes, pred_classes):
        metric.update(
            np.asarray(_gboxes),
            np.zeros((0, 4)) if _pboxes is None else np.asarray(_pboxes),
            np.array(_gclasses, dtype=np.int64),
            np.array(_pclasses, dtype=np.int64),
        )
    _recall, _precision, _mean_iou = metric.summary()
    assert _recall == recall
    assert _precision == precision
    assert _mean_iou == mean_iou
    metric.reset()
    assert metric.num_gts == metric.num_preds == metric.tot_iou == 0
    assert metric.num_matches == 0
    # Shape check
    with pytest.raises(AssertionError):
        metric.update(
            np.asarray(_gboxes),
            np.zeros((0, 4)),
            np.array(_gclasses, dtype=np.int64),
            np.array([1, 2], dtype=np.int64)
        )


def test_nms():
    boxes = [
        [0.1, 0.1, 0.2, 0.2, 0.95],
        [0.15, 0.15, 0.19, 0.2, 0.90],  # to suppress
        [0.5, 0.5, 0.6, 0.55, 0.90],
        [0.55, 0.5, 0.7, 0.55, 0.85],  # to suppress
    ]
    to_keep = metrics.nms(np.asarray(boxes), thresh=0.2)
    assert to_keep == [0, 2]


def test_box_ioa():
    boxes = [
        [0.1, 0.1, 0.2, 0.2],
        [0.15, 0.15, 0.2, 0.2],
    ]
    mat = metrics.box_ioa(np.array(boxes), np.array(boxes))
    assert mat[1, 0] == mat[0, 0] == mat[1, 1] == 1.
    assert abs(mat[0, 1] - .25) <= 1e-7
