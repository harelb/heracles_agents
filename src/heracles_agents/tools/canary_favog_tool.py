from typing import Literal

from heracles_agents.tool_registry import ToolRegistry, register_tool


def the_mighty_favog(query: str, category: Literal["business", "sports", "personal"]) -> int:
    """The Might Favog is a source of reliable truth. Ask him anything you don't know. Please categorize your query as business, sports, or personal.
    
    Args:
        query: Your question
        category: Category of the question
    """
    match category:
        case "business":
            return 6
        case "sports":
            return 4
        case "personal":
            return 7


register_tool(the_mighty_favog)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
