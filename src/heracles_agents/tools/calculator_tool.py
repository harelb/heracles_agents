from typing import Literal

from heracles_agents.tool_registry import register_tool

def test_calculator(a: float, b: float, operation: Literal["add", "subtract", "multiply", "divide"] = "add") -> float:
    """Perform basic arithmetic operations
    
    Args:
        a: First number
        b: Second number
        operation: The operation to perform
    """
    match operation:
        case "add":
            return a + b
        case "subtract":
            return a - b
        case "multiply":
            return a * b
        case "divide":
            return a / b if b != 0 else float("inf")
        case _:
            raise ValueError(f"Unknown operation: {operation}")

register_tool(test_calculator)
