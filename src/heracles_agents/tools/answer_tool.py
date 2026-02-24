from heracles_agents.tool_registry import register_tool

def answer_tool(answer: str) -> str:
    """Provide the final answer
    
    Args:
        answer: The final answer
    """
    return f"Final answer: {answer}"

register_tool(answer_tool)
