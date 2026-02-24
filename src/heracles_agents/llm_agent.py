from functools import partial
from typing import Optional, Callable

from pydantic import BaseModel, field_serializer, field_validator
from pydantic_ai import Agent

from heracles_agents.prompt import PromptSettings
from heracles_agents.tool_registry import ToolRegistry


class ModelInfo(BaseModel):
    """Settings that affect fundamental model performance.

    e.g., model size, temperature, seed.
    Tool calling details are handled elsewhere
    """

    model: str
    temperature: float = 1.0
    seed: Optional[int] = None


def apply_bound_args(tool_name, bound_args):
    args_to_bind = {}
    for arg_name, fields in bound_args.items():
        arg_type = ToolRegistry.get_arg_type(tool_name, arg_name)
        if arg_type is str or arg_type is int or arg_type is float:
            arg_instance = arg_type(fields)
        else:
            arg_instance = arg_type(**fields)
        args_to_bind[arg_name] = arg_instance
    function = partial(ToolRegistry.tools[tool_name], **args_to_bind)
    return function, args_to_bind


class AgentInfo(BaseModel):
    """Configuration for "agentic" behaviors, e.g., tool calling"""

    prompt_settings: PromptSettings
    tools: dict[str, Callable]
    tool_interface: str  # Openai vs. custom vs. ???
    max_iterations: int

    @field_validator("tools", mode="before")
    @classmethod
    def lookup_tools(cls, tools):
        tool_functions = {}
        for t in tools:
            tool_name = t["name"]
            if tool_name not in ToolRegistry.tools:
                raise ValueError(
                    f"Unknown tool {tool_name}. Known tools: {list(ToolRegistry.tools.keys())}"
                )
            if "bound_args" in t:
                function, _ = apply_bound_args(tool_name, t["bound_args"])
                tool_functions[tool_name] = function
            else:
                tool_functions[tool_name] = ToolRegistry.tools[tool_name]

        return tool_functions

    @field_serializer("tools")
    def serialize_tools(self, tools: dict[str, Callable]):
        # Just return the names of the tools, as the functions themselves are not serializable
        # We lose bound_args during serialization here, but that's a limitation of not having 
        # a dedicated object for it anymore. In pydantic-ai, tools are usually statically defined anyway.
        return [{"name": name} for name in tools.keys()]


def create_agent(agent_info: AgentInfo) -> Agent:
    """Creates a Pydantic-AI Agent based on the AgentInfo configuration."""
    # Build a list of functions from the resolved tool descriptions
    agent_tools = list(agent_info.tools.values())

    model_name = "openai:gpt-4o" # default if nothing else is specified.
    
    agent = Agent(
        model_name,
        system_prompt=agent_info.prompt_settings.base_prompt.to_openai_json("")[0]["content"],
        tools=agent_tools,
    )
    return agent
