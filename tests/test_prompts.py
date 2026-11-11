"""heracles_agents.prompts: the installed, single source of the plan-repair chat's
planning knowledge (the repair chats in examples/ and the ADT4 web console)."""

import os
import re

import pytest

from heracles_agents import prompts
from heracles_agents.prompt import PromptSettings

ROOT = os.path.join(os.path.dirname(__file__), "..")
CHATS = {
    "fast_downward": os.path.join(
        ROOT, "examples", "chatdsg_repair", "agent_prompt.yaml"
    ),
    "grstaps": os.path.join(
        ROOT, "examples", "chatdsg_repair_grstaps", "agent_prompt.yaml"
    ),
}
_ROBOT_FIRST = re.compile(
    r"\((?:visited-object-by|visited-place-by|inspected-by|visited-region-by|"
    r"explored-region-by|object-placed-by|holding|at-place|at-object)\s+([a-z]\w*)"
)


def _rules_in(text, engine):
    """The required goal rules, as phrases of the rule text."""
    assert "the goal MUST name that robot" in text
    for form in (
        "visited-object-by",
        "visited-place-by",
        "inspected-by",
        "object-placed-by",
    ):
        assert form in text, form
    assert "use the robot-free forms" in text
    for form in ("(visited-object o)", "(visited-place t)", "(object-in-place o t)"):
        assert form in text, form
    assert "let the planner allocate" in text
    assert "Never pick a robot yourself" in text
    assert "not the nearest robot" in text and "not the selected robot" in text
    assert "Never split the targets" in text
    assert '"Visit all cones" (cones O1 ... O15) is ONE goal' in text
    assert (
        "(and (visited-object o1) (visited-object o2) ... (visited-object o15))" in text
    )
    assert "only when the user explicitly asks for it" in text  # the nearest robot
    assert "One goal per request covers the whole team" in text
    assert "A new goal replaces the old one" in text
    assert "re-send the tasks of the last goal that are still wanted" in text
    # the only ask left about robots: a name that is not in the fleet
    assert "that is not in it" in text and "ask which robot they mean" in text
    assert "without naming a robot you know, ask" not in text
    assert "then give the task to that robot with a -by predicate" not in text
    if engine == "fast_downward":
        assert "holding / at-place / at-object" in text
        assert "plus a use-robots constraint naming exactly" in text
        assert '"Visit all the chairs with both euclid and hamilton' in text
    else:
        for form in ("visited-region-by", "explored-region-by", "(visited-region r)",
                     "(explored-region r)"):  # fmt: skip
            assert form in text, form
        assert "no use-robots constraint" in text
        assert "bind each task to one of the allowed robots with the -by forms" in text
        assert "plus a use-robots constraint" not in text


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_team_goal_rules(engine):
    _rules_in(prompts.team_goal_rules(engine), engine)


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_repair_goal_prompt_has_rules_domain_and_examples(engine):
    text = prompts.repair_goal_prompt(engine)
    _rules_in(text, engine)
    assert "<PDDL Domain>" in text and "<Runtime Constraints>" in text
    assert "<Examples>" in text and "User: Instruction:" in text
    # "Put X at Y" (heracles_agents #8): the place nearest Y, never X's own place
    assert "MeshPlace closest to Y" in text
    assert "X is already at Y: send no goal" in text


def test_repair_goal_prompt_renames_tools_for_another_chat():
    text = prompts.repair_goal_prompt(
        "fast_downward", names={"send_pddl_goal_with_constraints": "propose_goal"}
    )
    assert "send_pddl_goal_with_constraints" not in text and "propose_goal" in text


def test_unknown_engine_is_an_error():
    with pytest.raises(ValueError):
        prompts.repair_goal_prompt("fd")


def test_descriptions():
    assert "MESH PLACES" in prompts.scene_graph_description()
    assert "ROOMS" in prompts.scene_graph_description("hierarchical")
    assert "point.distance" in prompts.cypher_interface_description()
    assert os.path.isfile(prompts.path("team_goal_rules.yaml"))


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_chatdsg_repair_prompt_comes_from_the_package(engine, monkeypatch):
    monkeypatch.setenv("HERACLES_EVALUATION_PATH", ROOT)
    prompt = PromptSettings(base_prompt=CHATS[engine]).base_prompt
    _rules_in(prompt.system, engine)
    assert prompt.system == prompts.repair_chat_prompt(engine)["system"]
    assert "<<" not in prompt.system  # every marker filled
    assert prompt.domain_description == prompts.domain_description(engine)
    assert prompt.interface_description == prompts.cypher_interface_description()
    assert prompt.scene_graph_description == prompts.scene_graph_description()
    assert len(prompt.in_context_examples) == len(prompts.in_context_examples(engine))
    assert prompt.labelspace_description  # the example's own labelspace
    assert prompt.novel_instruction_template == "{question}"
    # the repair chat's own parts stay
    assert "SCENE CHANGE REPORTS" in prompt.system and "SGET" in prompt.system
    assert 'returns "NOT SENT"' in prompt.system
    if engine == "fast_downward":
        assert '["use-robots", "<name>", ...]' in prompt.system
        assert "This graph has no Room nodes" in prompt.system
    else:
        assert "Use only region symbols the scene graph has" in prompt.system
        assert "Rooms (regions) are labeled" in prompt.system


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_examples_only_name_robots_the_user_named(engine):
    """No example picks a robot itself (the old "hilbert picks it up" one did). A
    follow-up ("Also have ...") re-sends robots of the examples before it."""
    earlier = ""
    for ex in prompts.in_context_examples(engine):
        user = ex["user"].lower()
        if user.strip().startswith("instruction: also"):
            user += earlier
        earlier += ex["user"].lower()
        for goal in re.findall(
            r"pddl_goal_string=(\(.*?\))(?: and|$)", ex["assistant"]
        ):
            for robot in _ROBOT_FIRST.findall(goal):
                assert robot in user, (ex["user"], goal)
