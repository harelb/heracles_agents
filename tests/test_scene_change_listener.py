"""The scene change report the chat agent is given."""

from types import SimpleNamespace as NS

from heracles_agents.scene_change_listener import REPORT_PREFIX, describe_scene_changes


def _p(x=0.0, y=0.0):
    return NS(x=x, y=y, z=0.0)


def _c(kind, symbol, label, old=None, new=None):
    return NS(
        kind=kind,
        symbol=symbol,
        semantic_label=label,
        old_position=old or _p(),
        new_position=new or _p(),
    )


def test_report_lists_each_change_with_class_and_position():
    msg = NS(
        source="awcd/hamilton",
        map_version=12,
        changes=[
            _c(0, "O99", "cone", new=_p(4.0, -2.0)),
            _c(1, "O16", "cone", old=_p(1.0, 2.0)),
            _c(2, "O5", "", old=_p(1.0, 1.0), new=_p(3.0, 4.0)),
        ],
    )
    assert describe_scene_changes(msg) == "\n".join(
        [
            f"{REPORT_PREFIX} Change detection updated the map "
            "(source awcd/hamilton, map_version 12):",
            "- ADDED O99 (cone) at (4.0, -2.0)",
            "- REMOVED O16 (cone), was at (1.0, 2.0)",
            "- MOVED O5 (unknown class) from (1.0, 1.0) to (3.0, 4.0)",
        ]
    )


def test_our_own_picks_and_places_are_not_reported():
    msg = NS(source="executor/hamilton", map_version=3, changes=[_c(2, "O5", "box")])
    assert describe_scene_changes(msg) == ""


def test_an_empty_batch_is_not_reported():
    assert describe_scene_changes(NS(source="awcd/x", map_version=1, changes=[])) == ""
