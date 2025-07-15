from typing import Any, Dict
import sys
import os
üst_klasör = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
)
sys.path.append(üst_klasör)

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}