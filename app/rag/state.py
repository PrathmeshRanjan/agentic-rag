from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description="Use kb for HR/policy questions; direct for greetings/simple chat."
    )

class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="Good means evidence can answer the question; weak means not enough evidence."
    )

class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    trace: list[str]
    citations: list[dict[str, str]]
    