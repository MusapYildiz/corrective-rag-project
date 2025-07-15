from typing import Any, Dict

from langchain.schema import Document
from langchain_tavily import TavilySearch
import sys
import os
üst_klasör = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
)
sys.path.append(üst_klasör)

from graph.state import GraphState



web_search_tool = TavilySearch(max_results=1)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents")

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join(docs)
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
