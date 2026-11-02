#!/usr/bin/env python3
"""ChatDSG with plan repair tool support.

Also listens for scene changes (heracles_agents.scene_change_listener): each
one is put to the agent as a report, so it can recheck whether the change
affects what the user asked for. Reports and user messages take turns: one
arriving while the agent is busy waits, and waiting ones go in together.

The scene graph is edited only for a message starting with "SGET:"; other
turns cannot write to Neo4j (heracles_agents.scene_graph_edit).
"""

# Register the repair tool before the agent loads
# Import the app class from the original chatdsg
import importlib.util
import os
import threading

import yaml
from rich.markup import escape
from textual.widgets import RichLog

import heracles_agents.tools.pddl_repair_tool  # noqa: F401
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext
from heracles_agents.scene_change_listener import SceneChangeListener
from heracles_agents.scene_graph_edit import gate, install, turn_allows_edits

original = os.path.join(os.path.dirname(__file__), "..", "chatdsg", "chatdsg.py")
spec = importlib.util.spec_from_file_location("chatdsg_original", original)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

install()  # gate map writes on SGET, before the agent binds its tools


class RepairChatApp(mod.InputDisplayApp):
    def __init__(self, agent):
        super().__init__(agent)
        self._busy = False
        self._waiting = []
        self._listener = SceneChangeListener(
            lambda text: self.call_from_thread(self.post_scene_change, text)
        )

    def on_mount(self) -> None:
        self._listener.start()

    def on_unmount(self) -> None:
        self._listener.stop()

    def action_submit(self) -> None:
        """Called when ctrl+b is pressed."""
        box = self.query_one("#text_area", mod.MyTextArea)
        text, box.text = box.text, ""
        self._take_turn(f"[bold black on white]User:[/] {escape(text)}", text)

    def post_scene_change(self, report: str) -> None:
        self._take_turn(
            f"[bold black on yellow]Scene change:[/] {escape(report)}", report
        )

    def _take_turn(self, shown: str, content: str) -> None:
        log = self.query_one(RichLog)
        log.write(shown)
        log.write("")
        self._waiting.append(content)
        if not self._busy:
            self._run_agent()

    def _run_agent(self) -> None:
        if not self._waiting:
            self._busy = False
            return
        self._busy = True
        contents, self._waiting = self._waiting, []
        content = "\n\n".join(contents)
        gate.allowed = turn_allows_edits(contents)
        self.messages += mod.new_user_message(content)
        initial_length = len(self.messages)
        cxt = AgentContext(self.agent)
        cxt.history = self.messages
        log = self.query_one(RichLog)

        def run():
            try:
                cxt.run()
                for r in cxt.get_agent_responses()[initial_length:]:
                    self.call_from_thread(log.write, r.parsed_response)
                    self.call_from_thread(log.write, "")
            finally:
                gate.allowed = False
                self.call_from_thread(self._run_agent)  # next waiting turn, if any

        threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    with open("agent_config.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    agent = LlmAgent(**yml)
    app = RepairChatApp(agent)
    app.run()
