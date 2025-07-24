from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# 1. Initialize LLM and memory
llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 2. Retriever for external documents
vector_store = FAISS.load_local("faiss_index", embeddings=None)
retriever = vector_store.as_retriever()

# 3. Build conversational RAG chain
rag = ConversationalRetrievalChain(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

def chat_with_memory(query: str):
    resp = rag({"question": query})
    answer = resp["answer"]
    # memory automatically updated
    return answer, resp["source_documents"]

# Usage
if __name__ == "__main__":
    ans, docs = chat_with_memory("What did we discuss about allergies?")
    print("Answer:", ans)
    for doc in docs:
        print("•", doc.metadata.get("source"))
