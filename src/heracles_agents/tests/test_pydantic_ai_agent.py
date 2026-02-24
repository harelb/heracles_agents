import pytest
from pydantic_ai.models.test import TestModel

from heracles_agents.llm_agent import create_agent, AgentInfo
from heracles_agents.prompt import PromptSettings, Prompt

def test_agent_initialization():
    """Test that the agent can be initialized with pydantic-ai."""
    prompt = Prompt(system="You are an assistant.")
    prompt_settings = PromptSettings(base_prompt=prompt)
    
    agent_info = AgentInfo(
        prompt_settings=prompt_settings,
        tools=[{"name": "answer_tool"}],
        tool_interface="openai",  # Note: preserved purely for backward compatibility of schemas
        max_iterations=5,
    )
    
    agent = create_agent(agent_info)
    assert agent is not None
    
@pytest.mark.asyncio
async def test_agent_tool_calling():
    """Test that the agent correctly calls tools using TestModel."""
    prompt = Prompt(system="You are an assistant that can query a database.")
    prompt_settings = PromptSettings(base_prompt=prompt)
    
    agent_info = AgentInfo(
        prompt_settings=prompt_settings,
        tools=[{"name": "answer_tool"}],
        tool_interface="custom",
        max_iterations=5,
    )
    
    agent = create_agent(agent_info)
    
    # We use TestModel to verify tool calls
    test_model = TestModel(call_tools=["answer_tool"])
    
    result = await agent.run("Find the red chair.", model=test_model)
    
    # Verify the tool was called
    tool_calls = [
        m for m in result.new_messages() 
        if any(type(p).__name__ == "ToolCallPart" for p in getattr(m, 'parts', []))
    ]
    
    assert len(tool_calls) > 0, "Agent did not call the tool"
