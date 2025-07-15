from typing import Any, Dict
import sys
import os
üst_klasör = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
)
sys.path.append(üst_klasör)

from graph.chains.generation import generation_chain
from graph.state import GraphState



def generate(state:GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}