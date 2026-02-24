# ruff: noqa: F811
import logging
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from heracles_agents.llm_agent import create_agent, AgentInfo
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart

logger = logging.getLogger(__name__)

class SldpComparison(BaseModel):
    comparison_type: Literal["SLDP"]
    relation: str  # equal, subset, superset


class PddlComparison(BaseModel):
    comparison_type: Literal["PDDL"]
    relation: str  # equal, subset, superset


class LLmJudgeComparison(BaseModel):
    comparison_type: Literal["LLM_JUDGE"]


ComparisonType = Union[SldpComparison, PddlComparison, LLmJudgeComparison]


class EvalQuestion(BaseModel):
    name: str
    question: str
    solution: str
    uid: str | int
    tags: Optional[list[str]] = None
    correctness_comparator: ComparisonType = Field(discriminator="comparison_type")


class AgentResponse(BaseModel):
    # "full" response from the LLM (what format?)
    raw_response: str
    # interpretable response from the LLM (e.g., tool call, parsed final answer)
    parsed_response: Optional[str]


class AgentSequence(BaseModel):
    # What the "purpose" of this agent sequence is. Eventually should be more
    # structured/dispatchable than string?
    description: str

    responses: list[AgentResponse]


class QuestionAnalysis(BaseModel):
    # Information that is relevant about evaluating the response quality of the
    # "whole question"

    valid_answer_format: bool
    correct: bool
    input_tokens: int
    output_tokens: int
    n_tool_calls: int


class AnalyzedQuestion(BaseModel):
    # TODO: do we need to deal with partially-completed response lists?
    question: EvalQuestion
    sequences: list[AgentSequence]
    answer: Optional[str]
    analysis: Optional[QuestionAnalysis]


class AnalyzedQuestions(BaseModel):
    analyzed_questions: list[AnalyzedQuestion]


class AnalyzedExperiment(BaseModel):
    experiment_configurations: dict[str, AnalyzedQuestions]
    metadata: dict = Field(default_factory=dict)


class AgentContext:
    def __init__(self, agent_info: AgentInfo):
        self.agent_info = agent_info
        self.history = []
        self.n_tool_calls = 0
        self.initial_input_tokens = 0
        self.total_output_tokens = 0
        self.agent = create_agent(self.agent_info)
        self.user_prompt = ""

    def initialize_agent(self, prompt):
        system_prompt = ""
        user_prompt = ""
        openai_json = prompt.to_openai_json(prompt.novel_instruction) # returns list of dicts
        for msg in openai_json:
            if msg["role"] == "developer" or msg["role"] == "system":
                system_prompt += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                user_prompt += msg["content"] + "\n\n"
                
        self.user_prompt = user_prompt.strip()
        self.agent._system_prompt = system_prompt.strip()
        logger.info("Agent initialized with prompt")

    def run(self):
        try:
            result = self.agent.run_sync(self.user_prompt)
            self.history = result.all_messages()
            
            self.n_tool_calls = 0
            for msg in result.new_messages():
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if 'ToolCall' in type(part).__name__:
                            self.n_tool_calls += 1
                            
            answer = result.data
            return True, answer
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            return False, str(e)

    def get_agent_responses(self):
        responses = []
        for msg in self.history:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        responses.append(AgentResponse(raw_response=part.content, parsed_response=part.content))
                    elif isinstance(part, ToolReturnPart):
                        # Pydantic-ai ToolReturnPart does not have content formatted as string sometimes, maybe stringify it
                        responses.append(AgentResponse(raw_response=f"Tool Return: {part.content}", parsed_response=f"Tool Return: {part.content}"))
            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        responses.append(AgentResponse(raw_response=part.content, parsed_response=part.content))
                    elif isinstance(part, ToolCallPart):
                        responses.append(AgentResponse(raw_response=f"Tool Call: {part.tool_name}({part.args})", parsed_response=f"Tool Call: {part.tool_name}({part.args})"))
        return responses

