"""Scene graph edits by language, only when the user asks with "SGET:".

run_cypher_query can write to Neo4j, so whether the agent edits the map was up
to the LLM. With the gate installed, a write query (DELETE, SET, CREATE, ...)
runs only in a turn that a user message starting with "SGET:" (Scene Graph
EdiT) began; otherwise it is refused and the agent is told why. Robot nodes
and the map version are never written by hand.

Objects and places are not created by hand either: the published scene graph
is built from attributes a Cypher CREATE leaves out, and one such node stops
the map being published. Objects are added with add_scene_object, which writes
them the way change detection does. After every write the map is built once,
and if the edit broke it the agent is told so it can fix it.

After a write the map version is bumped, so the heracles publisher republishes
the edited map at once instead of at its next period. A plan already running
is not changed by the edit, so after an edit the chat offers to send the last
goal again (resend_offer); the goal manager replans it on the edited map.
"""

import functools
import re
from typing import Iterable

from heracles_agents.dsg_interfaces import HeraclesDsgInterface
from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import register_tool

EDIT_PREFIX = "SGET:"

_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b"
    r"|\bCALL\s+(apoc|db)\.",
    re.IGNORECASE,
)
_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|//[^\n]*")
_PROTECTED = re.compile(r":\s*(Robot|_MapState)\b")
_CLAUSE = re.compile(
    r"\b(CREATE|MERGE|OPTIONAL\s+MATCH|MATCH|WITH|RETURN|SET|DETACH|DELETE|REMOVE"
    r"|UNWIND|CALL|FOREACH)\b",
    re.IGNORECASE,
)
_MAP_NODE = re.compile(r"\(\s*\w*\s*:\s*(Object|MeshPlace)\b", re.IGNORECASE)

REFUSED = (
    "Refused: this query would change the scene graph. Map edits run only for a "
    "user message that starts with 'SGET:'. Ask the user to resend the request "
    "as 'SGET: <what to change>'."
)
PROTECTED = "Refused: Robot nodes and the map version are not edited by hand."
NO_HAND_CREATE = (
    "Refused: objects and places are not created with Cypher, since the map "
    "needs attributes a CREATE leaves out. Add an object with add_scene_object; "
    "places cannot be added."
)


def is_edit_request(text: str) -> bool:
    return text.lstrip().upper().startswith(EDIT_PREFIX)


def turn_allows_edits(contents: Iterable[str]) -> bool:
    """Whether the user messages starting this agent turn include an SGET request."""
    return any(is_edit_request(c) for c in contents)


def is_write_query(cypher: str) -> bool:
    # Quoted names ('reset area', "Set") and comments are not keywords.
    return bool(_WRITE.search(_LITERAL.sub("''", cypher)))


def creates_map_node(cypher: str) -> bool:
    """Whether a CREATE or MERGE clause makes an Object or MeshPlace node."""
    parts = _CLAUSE.split(_LITERAL.sub("''", cypher))
    return any(
        kw.upper() in ("CREATE", "MERGE") and _MAP_NODE.search(body)
        for kw, body in zip(parts[1::2], parts[2::2])
    )


class EditGate:
    def __init__(self):
        self.allowed = False
        self.wrote = False  # a write ran in this turn
        self.last_goal = None  # (pddl_goal, constraints_json) last sent


def resend_offer(last_goal) -> str:
    """What the chat says after an edit: offer to replan the last goal."""
    if last_goal is None:
        return ""
    goal, constraints = last_goal
    with_constraints = f" with constraints {constraints}" if constraints else ""
    return (
        f"The map changed. The last goal sent was {goal}{with_constraints}. "
        "A plan already running does not change with the map: shall I send "
        "this goal again so the robot replans on the edited map?"
    )


gate = EditGate()


def _connect(dsgdb_conf):
    from heracles.query_interface import Neo4jWrapper

    auth = (
        dsgdb_conf.username.get_secret_value(),
        dsgdb_conf.password.get_secret_value(),
    )
    return Neo4jWrapper(dsgdb_conf.uri, auth, atomic_queries=True)


def _bump_map_version(dsgdb_conf) -> None:
    from heracles_ros.map_state import bump_map_version

    with _connect(dsgdb_conf) as db:
        bump_map_version(db)


def _map_problem(dsgdb_conf) -> str:
    """Why the scene graph cannot be built from the database, or ""."""
    from heracles.graph_interface import db_to_spark_dsg

    with _connect(dsgdb_conf) as db:
        try:
            db_to_spark_dsg(db)
        except Exception as exc:
            return str(exc)
    return ""


def broken_map_warning(problem: str) -> str:
    return (
        "WARNING: after this edit the scene graph cannot be built, so planners "
        f"keep the old map: {problem} Fix or undo the edit now, then check again."
    )


def gated(query_fn, bump=_bump_map_version, check=_map_problem):
    """run_cypher_query's function, refusing writes unless gate.allowed."""

    @functools.wraps(query_fn)
    def run(cypher_string, dsgdb_conf=None):
        if not is_write_query(cypher_string):
            return query_fn(cypher_string, dsgdb_conf=dsgdb_conf)
        if not gate.allowed:
            return REFUSED
        bare = _LITERAL.sub("''", cypher_string)
        if _PROTECTED.search(bare):
            return PROTECTED
        if creates_map_node(cypher_string):
            return NO_HAND_CREATE
        result = query_fn(cypher_string, dsgdb_conf=dsgdb_conf)
        bump(dsgdb_conf)
        gate.wrote = True
        problem = check(dsgdb_conf)
        return f"{result}\n{broken_map_warning(problem)}" if problem else result

    return run


def add_scene_object(
    object_class: str,
    x: float,
    y: float,
    z: float = 0.0,
    dsgdb_conf: HeraclesDsgInterface = None,
) -> str:
    """Add an object to the scene graph, as change detection would (SGET only)."""
    if not gate.allowed:
        return REFUSED
    from heracles_ros.map_state import bump_map_version
    from heracles_ros.scene_change import add_object

    with _connect(dsgdb_conf) as db:
        sym = add_object(db, "", object_class, (x, y, z))["symbol"]
        rows = db.query(
            f"MATCH (p:MeshPlace)-[:CONTAINS]->(o:Object {{nodeSymbol: '{sym}'}}) "
            "RETURN p.nodeSymbol AS p"
        )
        bump_map_version(db)
    gate.wrote = True
    place = f", in place {rows[0]['p']}" if rows else ""
    return f"Added {sym} ({object_class}) at ({x}, {y}, {z}){place}."


add_object_tool = ToolDescription(
    name="add_scene_object",
    description=(
        "Add an object to the scene graph at a map position. Only for a user "
        "message starting with 'SGET:'. The object gets the next free id and is "
        "contained by its nearest MeshPlace."
    ),
    parameters=[
        FunctionParameter("object_class", str, "Class of the object, e.g. 'chair'"),
        FunctionParameter("x", float, "x position in the map frame (m)"),
        FunctionParameter("y", float, "y position in the map frame (m)"),
        FunctionParameter("z", float, "z position (m), default 0", required=False),
    ],
    function=add_scene_object,
)


def remembering(send_fn):
    """send_pddl_goal_with_constraints's function, noting the goal it sends."""

    @functools.wraps(send_fn)
    def send(
        pddl_goal_string, constraints_json="", robot_name=None, planner_topic=None
    ):
        result = send_fn(
            pddl_goal_string,
            constraints_json=constraints_json,
            robot_name=robot_name,
            planner_topic=planner_topic,
        )
        gate.last_goal = (pddl_goal_string, constraints_json)
        return result

    return send


def install() -> None:
    """Gate map writes, add add_scene_object, note sent goals.

    Call before the LlmAgent is built.
    """
    import heracles_agents.tools.cypher_query_tool  # noqa: F401  registers it
    import heracles_agents.tools.pddl_repair_tool  # noqa: F401
    from heracles_agents.tool_registry import ToolRegistry

    tool = ToolRegistry.tools["run_cypher_query"]
    if not getattr(tool.function, "_sget_gated", False):
        tool.function = gated(tool.function)
        tool.function._sget_gated = True
    register_tool(add_object_tool)
    tool = ToolRegistry.tools["send_pddl_goal_with_constraints"]
    if not getattr(tool.function, "_sget_remembers", False):
        tool.function = remembering(tool.function)
        tool.function._sget_remembers = True
