"""Scene graph edits by language, only when the user asks with "SGET:".

run_cypher_query can write to Neo4j, so whether the agent edits the map was up
to the LLM. With the gate installed, a write query (DELETE, SET, CREATE, ...)
runs only in a turn that a user message starting with "SGET:" (Scene Graph
EdiT) began; otherwise it is refused and the agent is told why. Robot nodes
and the map version are never written by hand.

Objects and places are not created by hand either: the published scene graph
is built from attributes a Cypher CREATE leaves out, and one such node stops
the map being published. Objects are added with add_scene_object, which writes
them the way change detection does, and places with add_scene_place, which
copies a nearby place's attributes. After every write the map is built once,
and if the edit broke it the agent is told so it can fix it.

An edit changes the same map change detection does, so it is announced the
same way: the objects it removed, added or moved go out as a scene change
(source "sget/chat") through gate.report, and the planner judges them against
the running plan -- a deleted object the plan picks pauses the robots.

After a write the map version is bumped, so the heracles publisher republishes
the edited map at once instead of at its next period. A plan already running
is not changed by the edit, so after an edit the chat offers to send the last
goal again (resend_offer); the goal manager replans it on the edited map.
"""

import functools
import json
import re
from typing import Iterable

from heracles_agents.dsg_interfaces import HeraclesDsgInterface
from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import register_tool

EDIT_PREFIX = "SGET:"
SGET_SOURCE = "sget/chat"
ADDED, REMOVED, MOVED = 0, 1, 2  # heracles_ros.scene_change

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
    "needs attributes a CREATE leaves out. Add an object with add_scene_object "
    "and a place with add_scene_place."
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
        # report(changes, map_version): announce what an edit changed.
        self.report = None
        # Goals are sent only once the user confirmed them: turn counts agent
        # turns, user_turn says whether a user message (not only a scene change
        # report) began this one, and pending is (goal key, turn shown).
        self.turn = 0
        self.user_turn = False
        self.pending = None

    def start_turn(self, user_turn: bool) -> None:
        self.turn += 1
        self.user_turn = user_turn
        self.allowed = self.wrote = False

    def show(self, goal, constraints) -> None:
        """The user has been shown this goal: their next message may confirm it."""
        self.pending = (goal_key(goal, constraints), self.turn)


NOT_CONFIRMED = (
    "NOT SENT: the user has not confirmed this goal yet. Show the user this "
    "exact goal and constraints and ask them to confirm. Call the tool again "
    "with the same goal once they say yes."
)


def goal_key(goal: str, constraints) -> tuple:
    """A goal and its constraints, written the same way however they were typed."""
    try:
        facts = json.loads(constraints) if constraints else []
    except ValueError:
        facts = [str(constraints)]
    facts = sorted(json.dumps([str(x).lower() for x in f]) for f in facts)
    return " ".join((goal or "").lower().split()), tuple(facts)


def goal_symbols(goal: str) -> list:
    """The object and place ids a PDDL goal names, e.g. ["o16", "t4"]."""
    return sorted(set(re.findall(r"\b[ot]\d+\b", goal or "")))


def resend_offer(last_goal, missing=()) -> str:
    """What the chat says after an edit: offer to replan the last goal."""
    if last_goal is None:
        return ""
    goal, constraints = last_goal
    with_constraints = f" with constraints {constraints}" if constraints else ""
    if missing:
        return (
            f"The map changed. The last goal sent was {goal}{with_constraints}, "
            f"but {', '.join(missing)} is no longer in the map, so it cannot be "
            "sent again as it is. What should the robot do instead?"
        )
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


def _bump_map_version(dsgdb_conf) -> int:
    from heracles_ros.map_state import bump_map_version

    with _connect(dsgdb_conf) as db:
        return bump_map_version(db)


def _objects(dsgdb_conf) -> dict:
    """{symbol: (class, (x, y, z))} of every object in the map."""
    with _connect(dsgdb_conf) as db:
        rows = db.query(
            "MATCH (o:Object) RETURN o.nodeSymbol AS s, o.class AS c, o.center AS p"
        )
    return {
        r["s"]: (
            r["c"] or "",
            None if r["p"] is None else (r["p"].x, r["p"].y, r["p"].z),
        )
        for r in rows
    }


def missing_symbols(symbols, dsgdb_conf) -> list:
    """Which of these goal ids (o16, t4) the map no longer has."""
    if not symbols:
        return []
    with _connect(dsgdb_conf) as db:
        rows = db.query(
            "MATCH (n) WHERE (n:Object OR n:MeshPlace) AND toLower(n.nodeSymbol) IN "
            f"{[s.lower() for s in symbols]} RETURN toLower(n.nodeSymbol) AS s"
        )
    have = {r["s"] for r in rows}
    return [s for s in symbols if s.lower() not in have]


def object_changes(before: dict, after: dict, min_move_m: float = 0.05) -> list:
    """What an edit did to the objects, as scene change records."""
    changes = []
    for sym, (label, old) in before.items():
        if sym not in after:
            changes.append(
                {
                    "kind": REMOVED,
                    "symbol": sym,
                    "label": label,
                    "old": old,
                    "new": None,
                }
            )
            continue
        new = after[sym][1]
        if old and new and sum((a - b) ** 2 for a, b in zip(old, new)) > min_move_m**2:
            changes.append(
                {"kind": MOVED, "symbol": sym, "label": label, "old": old, "new": new}
            )
    for sym, (label, new) in after.items():
        if sym not in before:
            changes.append(
                {"kind": ADDED, "symbol": sym, "label": label, "old": None, "new": new}
            )
    return changes


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


def gated(query_fn, bump=_bump_map_version, check=_map_problem, objects=_objects):
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
        before = objects(dsgdb_conf) if gate.report else None
        result = query_fn(cypher_string, dsgdb_conf=dsgdb_conf)
        version = bump(dsgdb_conf)
        gate.wrote = True
        if gate.report:
            changes = object_changes(before, objects(dsgdb_conf))
            if changes:
                gate.report(changes, version)
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
        added = add_object(db, "", object_class, (x, y, z))
        sym = added["symbol"]
        rows = db.query(
            f"MATCH (p:MeshPlace)-[:CONTAINS]->(o:Object {{nodeSymbol: '{sym}'}}) "
            "RETURN p.nodeSymbol AS p"
        )
        version = bump_map_version(db)
    gate.wrote = True
    if gate.report:
        gate.report([added], version)
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


def add_scene_place(
    x: float,
    y: float,
    connect_to: str,
    symbol: str = "",
    dsgdb_conf: HeraclesDsgInterface = None,
) -> str:
    """Add a traversable place, connected both ways to the places named (SGET only)."""
    if not gate.allowed:
        return REFUSED
    from heracles_ros.map_state import bump_map_version

    neighbours = [n.strip().lower() for n in connect_to.split(",") if n.strip()]
    symbol = symbol.strip().lower()
    if symbol and not re.fullmatch(r"t\d+", symbol):
        return f"Refused: a place id is t followed by a number, not '{symbol}'."
    with _connect(dsgdb_conf) as db:
        known = {
            r["s"]
            for r in db.query(
                "MATCH (p:MeshPlace) WHERE p.nodeSymbol IN "
                f"{neighbours + [symbol]} RETURN p.nodeSymbol AS s"
            )
        }
        if symbol in known:
            return f"Refused: place {symbol} already exists."
        missing = [n for n in neighbours if n not in known]
        if not neighbours or missing:
            return f"Refused: connect_to must name existing places; missing: {missing or 'none given'}."
        if not symbol:
            top = db.query(
                "MATCH (p:MeshPlace) RETURN max(toInteger(substring(p.nodeSymbol, 1))) AS n"
            )[0]["n"]
            symbol = f"t{(top or 0) + 1}"
        # A new place looks like its nearest neighbour (attribute type, clearance)
        # except for where it is, so the scene graph can be built from it.
        db.query(
            "MATCH (src:MeshPlace) WITH src, "
            f"(src.pos_x - {x})^2 + (src.pos_y - {y})^2 AS d ORDER BY d LIMIT 1 "
            "CREATE (p:MeshPlace) SET p = properties(src), "
            f"p.nodeSymbol = '{symbol}', p.center = point({{x: {x}, y: {y}, z: 0.0}}), "
            f"p.pos_x = {x}, p.pos_y = {y}, p.pos_z = 0.0"
        )
        db.query(
            f"MATCH (p:MeshPlace {{nodeSymbol: '{symbol}'}}), (q:MeshPlace) "
            f"WHERE q.nodeSymbol IN {neighbours} "
            "CREATE (p)-[:MESH_PLACE_CONNECTED]->(q), (q)-[:MESH_PLACE_CONNECTED]->(p)"
        )
        version = bump_map_version(db)
    gate.wrote = True
    problem = _map_problem(dsgdb_conf)
    done = f"Added place {symbol} at ({x}, {y}), connected to {', '.join(neighbours)} (map_version {version})."
    return f"{done}\n{broken_map_warning(problem)}" if problem else done


add_place_tool = ToolDescription(
    name="add_scene_place",
    description=(
        "Add a traversable place (MeshPlace) at a map position, connected both "
        "ways to existing places. Only for a user message starting with 'SGET:'."
    ),
    parameters=[
        FunctionParameter("x", float, "x position in the map frame (m)"),
        FunctionParameter("y", float, "y position in the map frame (m)"),
        FunctionParameter(
            "connect_to",
            str,
            "Comma-separated places to connect it to, e.g. 't1104, t1405'",
        ),
        FunctionParameter(
            "symbol",
            str,
            "Id for the new place, e.g. 't999'; empty for the next free one",
            required=False,
        ),
    ],
    function=add_scene_place,
)


def confirming(send_fn):
    """send_pddl_goal_with_constraints's function, sending only confirmed goals.

    A goal goes out only if it was shown to the user in an earlier turn and a
    user message began this one; otherwise it is held as shown and the agent
    is told to ask. So nothing is sent from a scene change report, or in the
    turn that first proposes a goal.
    """

    @functools.wraps(send_fn)
    def send(
        pddl_goal_string, constraints_json="", robot_name=None, planner_topic=None
    ):
        key = goal_key(pddl_goal_string, constraints_json)
        confirmed = (
            gate.user_turn
            and gate.pending is not None
            and gate.pending[0] == key
            and gate.pending[1] < gate.turn
        )
        if not confirmed:
            gate.show(pddl_goal_string, constraints_json)
            return NOT_CONFIRMED
        gate.pending = None
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
    """Gate map writes, add add_scene_object, send only confirmed goals.

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
    register_tool(add_place_tool)
    tool = ToolRegistry.tools["send_pddl_goal_with_constraints"]
    if not getattr(tool.function, "_sget_confirms", False):
        tool.function = confirming(tool.function)
        tool.function._sget_confirms = True
