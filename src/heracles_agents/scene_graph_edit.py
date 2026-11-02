"""Scene graph edits by language, only when the user asks with "SGET:".

run_cypher_query can write to Neo4j, so whether the agent edits the map was up
to the LLM. With the gate installed, a write query (DELETE, SET, CREATE, ...)
runs only in a turn that a user message starting with "SGET:" (Scene Graph
EdiT) began; otherwise it is refused and the agent is told why. Robot nodes
and the map version are never written by hand.

After a write the map version is bumped, so the heracles publisher republishes
the edited map at once instead of at its next period. A plan already running
is not changed by the edit, so after an edit the chat offers to send the last
goal again (resend_offer); the goal manager replans it on the edited map.
"""

import functools
import re
from typing import Iterable

EDIT_PREFIX = "SGET:"

_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b"
    r"|\bCALL\s+(apoc|db)\.",
    re.IGNORECASE,
)
_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|//[^\n]*")
_PROTECTED = re.compile(r":\s*(Robot|_MapState)\b")

REFUSED = (
    "Refused: this query would change the scene graph. Map edits run only for a "
    "user message that starts with 'SGET:'. Ask the user to resend the request "
    "as 'SGET: <what to change>'."
)
PROTECTED = "Refused: Robot nodes and the map version are not edited by hand."


def is_edit_request(text: str) -> bool:
    return text.lstrip().upper().startswith(EDIT_PREFIX)


def turn_allows_edits(contents: Iterable[str]) -> bool:
    """Whether the user messages starting this agent turn include an SGET request."""
    return any(is_edit_request(c) for c in contents)


def is_write_query(cypher: str) -> bool:
    # Quoted names ('reset area', "Set") and comments are not keywords.
    return bool(_WRITE.search(_LITERAL.sub("''", cypher)))


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


def _bump_map_version(dsgdb_conf) -> None:
    from heracles.query_interface import Neo4jWrapper
    from heracles_ros.map_state import bump_map_version

    auth = (
        dsgdb_conf.username.get_secret_value(),
        dsgdb_conf.password.get_secret_value(),
    )
    with Neo4jWrapper(dsgdb_conf.uri, auth, atomic_queries=True) as db:
        bump_map_version(db)


def gated(query_fn, bump=_bump_map_version):
    """run_cypher_query's function, refusing writes unless gate.allowed."""

    @functools.wraps(query_fn)
    def run(cypher_string, dsgdb_conf=None):
        if not is_write_query(cypher_string):
            return query_fn(cypher_string, dsgdb_conf=dsgdb_conf)
        if not gate.allowed:
            return REFUSED
        if _PROTECTED.search(_LITERAL.sub("''", cypher_string)):
            return PROTECTED
        result = query_fn(cypher_string, dsgdb_conf=dsgdb_conf)
        bump(dsgdb_conf)
        gate.wrote = True
        return result

    return run


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
    """Gate run_cypher_query and note sent goals. Call before the LlmAgent is built."""
    import heracles_agents.tools.cypher_query_tool  # noqa: F401  registers it
    import heracles_agents.tools.pddl_repair_tool  # noqa: F401
    from heracles_agents.tool_registry import ToolRegistry

    tool = ToolRegistry.tools["run_cypher_query"]
    if not getattr(tool.function, "_sget_gated", False):
        tool.function = gated(tool.function)
        tool.function._sget_gated = True
    tool = ToolRegistry.tools["send_pddl_goal_with_constraints"]
    if not getattr(tool.function, "_sget_remembers", False):
        tool.function = remembering(tool.function)
        tool.function._sget_remembers = True
