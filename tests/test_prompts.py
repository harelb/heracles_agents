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


_STAY_WORDS = ('"be at"', '"stay at"', '"wait at"', '"park at"', '"stop at"', '"block"')
_AT_EXAMPLES = (
    '- "Euclid, be at O20": (at-object euclid o20)',
    '- "Hamilton, wait at t7": (at-place hamilton t7)',
    '- "Euclid visits O8 then stays at t12": '
    "(and (visited-object-by euclid o8) (at-place euclid t12))",
)
_GO_TO_O21 = (
    '- "Hamilton, go to O21. Gauss, inspect O26": '
    "(and (visited-object-by hamilton o21) (inspected-by gauss o26))"
)


def _fast_downward_at_rules_in(text):
    """Fast Downward: "be / stay / wait at" is at-place / at-object; "go to" is not."""
    assert "use (at-place <robot> <t>) / (at-object <robot> <o>)" in text
    for word in _STAY_WORDS:
        assert word in text, word
    assert "If the user explicitly asks for the at predicate, use it." in text
    assert "at-place / at-object always need a robot" in text
    assert "ask which robot should end there" in text
    assert '"Go to", "visit", "check" or "look at"' in text
    for example in _AT_EXAMPLES:
        assert example in text, example
    assert _GO_TO_O21 in text  # "go to" stays visited-object-by


def test_fast_downward_rules_map_end_or_stay_to_at_predicates():
    _fast_downward_at_rules_in(prompts.team_goal_rules("fast_downward"))


def test_grstaps_rules_explain_the_limit_and_name_fast_downward():
    text = prompts.team_goal_rules("grstaps")
    assert "GRSTAPS-X cannot hold a robot at a final position" in text
    assert "Fast Downward can (the console's planner toggle)" in text
    assert "before constraint" in text
    assert 'Never say that "the system" or "the interface" does not support' in text
    for example in _AT_EXAMPLES:
        assert example not in text
    examples = text.split("Examples (")[1]
    assert "(at-place" not in examples and "(at-object" not in examples
    assert _GO_TO_O21 in text


def test_chatdsg_repair_has_the_at_rules():
    _fast_downward_at_rules_in(prompts.repair_chat_prompt("fast_downward")["system"])
    grstaps = prompts.repair_chat_prompt("grstaps")["system"]
    assert "Fast Downward can (the console's planner toggle)" in grstaps
    assert "(at-object euclid o20)" not in grstaps


def test_fast_downward_domain_uses_at_for_end_or_stay():
    text = prompts.domain_description("fast_downward")
    assert "(at-place ?r ?t): Robot ?r must END at place ?t" in text
    assert "(at-object ?r ?o): Robot ?r must END at object ?o" in text
    assert 'use "at-" predicates for the final destination' in text
    assert "when the user wants a robot to end or stay somewhere" in text
