"""Scene graph edits by language, only when the user asks with "SGET:".

run_cypher_query can write to Neo4j, so whether the agent edits the map was up
to the LLM. With the gate installed, a write query (DELETE, SET, CREATE, ...)
runs only in a turn that a user message starting with "SGET:" (Scene Graph
EdiT) began; otherwise it is refused and the agent is told why. Robot nodes
and the map version are never written by hand.

After a write the map version is bumped, so the heracles publisher republishes
the edited map at once instead of at its next period.
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
        return result

    return run


def install() -> None:
    """Gate the registered run_cypher_query. Call before the LlmAgent is built."""
    import heracles_agents.tools.cypher_query_tool  # noqa: F401  registers it
    from heracles_agents.tool_registry import ToolRegistry

    tool = ToolRegistry.tools["run_cypher_query"]
    if not getattr(tool.function, "_sget_gated", False):
        tool.function = gated(tool.function)
        tool.function._sget_gated = True
