from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()