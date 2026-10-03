from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# 1. Initialize local LLM (Llama 3.1) via LangChain
llm = Ollama(model="llama3.1")

# 2. Initialize local Embedding model (Nomic Embed) via LangChain
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Test LLM generation
print("Testing Llama 3.1 connection...")
response = llm.invoke("Give me a one-sentence tip for writing clean code.")
print(f"LLM Response: {response}")

# 4. Test embedding generation
print("\nTesting local embeddings...")
vector = embeddings.embed_query("Local RAG pipeline test")
print(f"Generated embedding vector length: {len(vector)}")
print("LangChain + Ollama setup verified successfully!")
