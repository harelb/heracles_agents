"""Tell a chat agent when the map changes under the user's request.

The scene change writer announces every change it applied to Neo4j on
/scene_graph/changes_applied (heracles_ros_interfaces/SceneChangeMsg). This
listens there in a background thread and hands each batch to a callback as a
short report, written for the LLM: what changed, the class, and where. The
agent's prompt says how to judge such a report against what the user asked.

Changes our own robots made (source "executor/<robot>": a pick or place the
plan already accounts for) are not reported.
"""

import threading
from typing import Callable

REPORT_PREFIX = "[Scene change report]"

_KIND_NAMES = {0: "ADDED", 1: "REMOVED", 2: "MOVED"}


def _xy(p) -> str:
    return f"({p.x:.1f}, {p.y:.1f})"


def describe_scene_changes(msg) -> str:
    """A SceneChangeMsg as one report for the agent, or "" if there is nothing to tell."""
    if msg.source.startswith("executor/") or not msg.changes:
        return ""
    lines = [
        f"{REPORT_PREFIX} Change detection updated the map "
        f"(source {msg.source}, map_version {msg.map_version}):"
    ]
    for c in msg.changes:
        kind = _KIND_NAMES.get(c.kind, "CHANGED")
        what = f"{c.symbol} ({c.semantic_label or 'unknown class'})"
        if kind == "ADDED":
            lines.append(f"- ADDED {what} at {_xy(c.new_position)}")
        elif kind == "REMOVED":
            lines.append(f"- REMOVED {what}, was at {_xy(c.old_position)}")
        else:
            lines.append(
                f"- {kind} {what} from {_xy(c.old_position)} to {_xy(c.new_position)}"
            )
    return "\n".join(lines)


class SceneChangeListener:
    """Calls on_report(text) from a background thread for each applied change."""

    def __init__(
        self,
        on_report: Callable[[str], None],
        topic: str = "/scene_graph/changes_applied",
    ):
        self.on_report = on_report
        self.topic = topic
        self._executor = None
        self._thread = None

    def start(self) -> None:
        import rclpy
        from heracles_ros_interfaces.msg import SceneChangeMsg
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions

        if not rclpy.ok():
            # The chat UI owns Ctrl-C; rclpy must not take it over.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node("chat_agent_scene_changes")
        self._node.create_subscription(SceneChangeMsg, self.topic, self._on_msg, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _on_msg(self, msg) -> None:
        text = describe_scene_changes(msg)
        if text:
            self.on_report(text)

    def stop(self) -> None:
        import rclpy

        if self._executor is not None:
            self._executor.shutdown()
            self._node.destroy_node()
            self._executor = None
        rclpy.try_shutdown()
