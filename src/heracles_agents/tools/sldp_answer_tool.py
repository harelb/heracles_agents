from heracles_agents.tool_registry import ToolRegistry, register_tool

def sldp_answer_tool(answer: str) -> str:
    """Use this tool to submit your final SLDP-formatted answer.
    
    Args:
        answer: The final SLDP answer
    """
    return answer

register_tool(sldp_answer_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
