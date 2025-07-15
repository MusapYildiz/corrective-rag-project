from dotenv import load_dotenv

load_dotenv()

from graph.graph2 import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "What is the weather in Istanbul today?"}))