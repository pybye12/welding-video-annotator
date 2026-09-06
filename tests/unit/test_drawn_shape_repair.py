"""Turning what someone drew into a shape shapely will accept.

A trace that crosses its own edge is trivially easy to draw when
outlining spatter: you follow the blob, double back, and the outline
touches itself. Shapely refuses to intersect that polygon — it raises
``TopologyException: side location conflict`` — and because PyQt aborts
the process when an exception escapes a slot, the whole app closed.
"""

import pytest
import shapely
from shapely.geometry import Polygon

from digitalsreeni_image_annotator.annotator_window import (
    clip_drawn_shape,
    largest_polygon_part,
)


BOWTIE = [(400, 600), (500, 600), (400, 650), (500, 650)]


def test_the_bowtie_that_crashed_the_app_still_crashes_raw_shapely():
    """Pins the actual failure, so the fix is not guarding a ghost.

    If a future shapely stops raising here, the guard becomes belt and
    braces rather than load-bearing — worth knowing either way.
    """
    boundary = Polygon([(0, 0), (960, 0), (960, 960), (0, 960)])
    assert not Polygon(BOWTIE).is_valid
    with pytest.raises(shapely.errors.GEOSException):
        Polygon(BOWTIE).intersection(boundary)


def test_a_self_crossing_outline_is_repaired_into_its_largest_lobe():
    """The annotator was drawing one blob; the crossing made two.

    Taking the larger lobe keeps the shape they meant and drops the
    sliver the stray crossing produced.
    """
    result = clip_drawn_shape(BOWTIE, 960, 960)

    assert result is not None
    assert result.is_valid
    assert result.area > 0
    # make_valid splits this bowtie into two equal triangles of 1250.
    # Getting 2500 back would mean the crossing outline was kept whole.
    assert result.area == pytest.approx(1250.0)


def test_an_ordinary_outline_is_left_alone():
    square = [(10, 10), (110, 10), (110, 110), (10, 110)]
    result = clip_drawn_shape(square, 960, 960)
    assert result.area == pytest.approx(10000)


def test_a_shape_hanging_over_the_edge_is_clipped_to_the_image():
    result = clip_drawn_shape([(-50, -50), (50, -50), (50, 50), (-50, 50)], 960, 960)
    assert result.area == pytest.approx(2500)


def test_a_shape_entirely_outside_the_image_yields_nothing():
    """The caller shows "outside the image" rather than saving a ghost."""
    assert clip_drawn_shape([(-99, -99), (-50, -99), (-50, -50)], 960, 960) is None


def test_a_zero_area_drag_yields_nothing():
    """A box dragged to no width is a line, and a line is not a mask."""
    assert clip_drawn_shape([(10, 10), (10, 10), (10, 10), (10, 10)], 960, 960) is None
    assert clip_drawn_shape([(10, 10), (10, 200), (10, 200), (10, 10)], 960, 960) is None


def test_too_few_points_yields_nothing_instead_of_raising():
    assert clip_drawn_shape([(10, 10), (20, 20)], 960, 960) is None
    assert clip_drawn_shape([], 960, 960) is None


def test_largest_polygon_part_ignores_lines_and_points():
    """``make_valid`` can return a collection with stray bits in it.

    Only a polygon with area can become an annotation; a dangling edge
    left by the repair must not be mistaken for the shape.
    """
    from shapely.geometry import GeometryCollection, LineString, Point

    big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    small = Polygon([(20, 20), (22, 20), (22, 22), (20, 22)])
    collection = GeometryCollection(
        [LineString([(0, 0), (5, 5)]), Point(1, 1), small, big]
    )

    # Not `is`: a GeometryCollection rebuilds the geometries it holds,
    # so the result is an equal polygon rather than the same object.
    assert largest_polygon_part(collection).equals(big)
    assert largest_polygon_part(GeometryCollection([Point(1, 1)])) is None
    assert largest_polygon_part(Polygon()) is None
    assert largest_polygon_part(None) is None
