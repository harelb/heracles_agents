"""The edit tools against a real Neo4j: each edit is logged, so the writer can undo it.

Wipes the database it connects to: uses the dedicated test instance
(HERACLES_TEST_NEO4J_URI) and skips without it, or without heracles_ros.
"""

import os
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("heracles_ros.change_log")

from heracles.query_interface import Neo4jWrapper  # noqa: E402
from heracles_ros import change_log  # noqa: E402
from heracles_ros.map_state import (  # noqa: E402
    get_map_version,
    seed_db,
    set_rebuilding,
)
from heracles_ros.scene_change import object_center, place_exists  # noqa: E402

from heracles_agents.scene_graph_edit import (  # noqa: E402
    REBUILDING,
    REFUSED,
    gate,
    move_scene_object,
    remove_scene_object,
    remove_scene_place,
)

URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "")
AUTH = ("neo4j", "neo4j_pw")
DSG_PATH = os.path.expanduser(
    os.environ.get(
        "HERACLES_TEST_DSG",
        "~/colcon_ws/assets/adt4_output/plan_repair_graph/hydra/backend/dsg.json",
    )
)
CONF = NS(
    uri=URI,
    username=NS(get_secret_value=lambda: AUTH[0]),
    password=NS(get_secret_value=lambda: AUTH[1]),
)


@pytest.fixture
def db():
    if not URI or not os.path.exists(DSG_PATH):
        pytest.skip("No test Neo4j (HERACLES_TEST_NEO4J_URI) or test scene graph")
    seed_db(URI, AUTH, DSG_PATH)
    db = Neo4jWrapper(URI, AUTH, atomic_queries=True, print_profiles=False)
    db.connect()
    gate.allowed, gate.wrote, gate.report = True, False, None
    yield db
    gate.allowed = gate.wrote = False
    gate.report = None
    db.close()


def _a_place(db):
    return db.query(
        "MATCH (p:MeshPlace)-[:MESH_PLACE_CONNECTED]->() RETURN p.nodeSymbol AS p LIMIT 1"
    )[0]["p"]


def test_a_deleted_place_is_logged_and_can_be_undone(db):
    place = _a_place(db)
    v0 = get_map_version(db)
    result = remove_scene_place(place, dsgdb_conf=CONF)
    assert result.startswith(f"Deleted place {place}") and "undo_scene_edit" in result
    assert not place_exists(db, place) and gate.wrote
    assert get_map_version(db) == v0 + 1
    (entry,) = change_log.entries(db)
    assert (entry["source"], entry["op"], entry["data"]) == (
        "sget/chat",
        "remove_place",
        {"symbol": place},
    )
    # What the writer's undo does.
    change_log.deactivate(db, [entry["seq"]])
    change_log.rebuild(db, DSG_PATH)
    assert place_exists(db, place)


def test_object_edits_are_announced_and_logged(db):
    reports = []
    gate.report = lambda changes, version: reports.append((changes, version))
    assert remove_scene_object("o4", dsgdb_conf=CONF).startswith("Removed O4")
    assert object_center(db, "O4") is None
    ((changes, version),) = reports
    assert [(c["kind"], c["symbol"]) for c in changes] == [(1, "O4")]
    assert version == get_map_version(db)
    assert [e["op"] for e in change_log.entries(db)] == ["remove_object"]


def test_moving_keeps_the_height_unless_given(db):
    z = object_center(db, "O2")[2]
    move_scene_object("O2", 1.0, 2.0, dsgdb_conf=CONF)
    assert object_center(db, "O2") == pytest.approx((1.0, 2.0, z))


def test_unknown_targets_change_nothing(db):
    v0 = get_map_version(db)
    assert remove_scene_place("t999999", dsgdb_conf=CONF).startswith("Refused")
    assert remove_scene_object("O999", dsgdb_conf=CONF).startswith("Refused")
    assert get_map_version(db) == v0 and change_log.entries(db) == []
    assert not gate.wrote


def test_no_edits_while_the_map_is_rebuilt(db):
    set_rebuilding(db, True)
    try:
        assert remove_scene_place(_a_place(db), dsgdb_conf=CONF) == REBUILDING
    finally:
        set_rebuilding(db, False)


def test_edits_need_sget(db):
    gate.allowed = False
    assert remove_scene_object("O4", dsgdb_conf=CONF) == REFUSED
    assert object_center(db, "O4") is not None
