from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings

load_dotenv()


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250, chunk_overlap = 0
)

docs_splits = text_splitter.split_documents(docs_list)

"""vector_store = Chroma.from_documents(
    documents=docs_splits,
    collection_name="rag-chroma",
    embedding=SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2"),
    persist_directory="./.chroma"
)"""

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
).as_retriever()


#Örnek Kullanım
"""
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
    chain_type="stuff",
    retriever=retriever
)

answer = qa_chain("What are types of adversarial attacks in LLMs?")
print(answer)
"""