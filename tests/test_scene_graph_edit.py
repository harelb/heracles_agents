"""Map writes run only for an SGET request."""

import pytest

from heracles_agents.scene_graph_edit import (
    NO_HAND_CREATE,
    PROTECTED,
    REFUSED,
    add_scene_object,
    creates_map_node,
    gate,
    gated,
    is_edit_request,
    is_write_query,
    remembering,
    resend_offer,
    turn_allows_edits,
)


@pytest.fixture
def query():
    ran, bumps = [], []
    run = gated(
        lambda cypher_string, dsgdb_conf=None: ran.append(cypher_string) or "[]",
        bump=bumps.append,
        check=lambda conf: "",
    )
    gate.allowed = gate.wrote = False
    gate.last_goal = None
    yield run, ran, bumps
    gate.allowed = gate.wrote = False
    gate.last_goal = None


def test_edit_requests():
    assert is_edit_request("SGET: delete t444")
    assert is_edit_request("  sget: delete t444")
    assert not is_edit_request("delete t444, SGET: later")
    assert turn_allows_edits(["[Scene change report] ...", "SGET: delete t444"])
    assert not turn_allows_edits(["Go to t444", "[Scene change report] ..."])


def test_write_queries():
    assert is_write_query("MATCH (p:MeshPlace {nodeSymbol: 't444'}) DETACH DELETE p")
    assert is_write_query("MATCH ()-[r]-() DELETE r")
    assert is_write_query("MATCH (o:Object) SET o.class = 'cone'")
    assert is_write_query("CALL apoc.refactor.rename.label('A', 'B')")
    assert not is_write_query("MATCH (o:Object) RETURN o.nodeSymbol, o.center")
    # Keywords inside names and comments are not writes.
    assert not is_write_query("MATCH (o:Object {class: 'reset button'}) RETURN o")
    assert not is_write_query("MATCH (o) // delete later\nRETURN o")


def test_reads_always_run(query):
    run, ran, bumps = query
    run("MATCH (o:Object) RETURN o")
    assert ran == ["MATCH (o:Object) RETURN o"] and bumps == []


def test_writes_are_refused_without_sget(query):
    run, ran, bumps = query
    assert run("MATCH (p:MeshPlace {nodeSymbol: 't444'}) DETACH DELETE p") == REFUSED
    assert ran == [] and bumps == []


def test_writes_run_for_sget_and_bump_the_map(query):
    run, ran, bumps = query
    gate.allowed = True
    run("MATCH (p:MeshPlace {nodeSymbol: 't444'}) DETACH DELETE p", dsgdb_conf="conf")
    assert len(ran) == 1 and bumps == ["conf"] and gate.wrote


def test_robots_and_map_version_are_never_written(query):
    run, ran, _ = query
    gate.allowed = True
    assert run("MATCH (r:Robot) DETACH DELETE r") == PROTECTED
    assert run("MATCH (m:_MapState) SET m.version = 0") == PROTECTED
    assert ran == []


def test_after_an_edit_the_last_goal_is_offered_again(query):
    run, _, _ = query
    send = remembering(lambda goal, **kw: f"Sent goal {goal}")
    send("(visited-place t500)", constraints_json='[["forbidden-poi", "o3"]]')
    gate.allowed = True
    run("MATCH (p:MeshPlace {nodeSymbol: 't444'}) DETACH DELETE p")
    assert gate.wrote
    offer = resend_offer(gate.last_goal)
    assert "(visited-place t500)" in offer and "forbidden-poi" in offer
    assert "send this goal again" in offer


def test_no_offer_without_a_goal_sent(query):
    assert resend_offer(gate.last_goal) == ""


def test_objects_and_places_are_not_created_by_hand(query):
    run, ran, _ = query
    gate.allowed = True
    hand_made = "CREATE (o:Object {nodeSymbol: 'O21', class: 'chair'})"
    assert creates_map_node(hand_made)
    assert creates_map_node(
        "MATCH (p:MeshPlace) MERGE (p)-[:CONTAINS]->(o:Object {class: 'x'})"
    )
    assert creates_map_node("CREATE (t:MeshPlace {nodeSymbol: 't1'})")
    # Linking existing nodes, or a label only inside MATCH, is fine.
    assert not creates_map_node(
        "MATCH (o:Object {nodeSymbol: 'O3'}), (p:MeshPlace {nodeSymbol: 't4'}) "
        "CREATE (p)-[:CONTAINS]->(o)"
    )
    assert not creates_map_node("MATCH (o:Object) SET o.class = 'CREATE (x:Object)'")
    assert run(hand_made) == NO_HAND_CREATE and ran == []


def test_an_edit_that_breaks_the_map_is_reported():
    gate.allowed = True
    try:
        run = gated(
            lambda cypher_string, dsgdb_conf=None: "[]",
            bump=lambda conf: None,
            check=lambda conf: "Node O21 has no attr_type property.",
        )
        result = run("MATCH (o:Object {nodeSymbol: 'O0'}) REMOVE o.attr_type")
        assert (
            "cannot be built" in result and "O21" in result and "Fix or undo" in result
        )
    finally:
        gate.allowed = gate.wrote = False


def test_adding_an_object_needs_sget():
    gate.allowed = False
    assert add_scene_object("chair", 5.0, -8.0) == REFUSED
