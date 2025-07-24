from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA

# 1. Initialize models and retriever
llm = ChatOpenAI(temperature=0)
embed_model = OpenAIEmbeddings()
vector_store = FAISS.load_local("faiss_index", embed_model)

# 2. Rewrite chain
rewrite_prompt = PromptTemplate(
    "Rewrite the user question to make it clearer and more detailed:\n\n"
    "User: {query}\n\nRewritten:",
    input_variables=["query"]
)
rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)

# 3. Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

def answer_with_query_rewrite(user_query: str):
    rewritten = rewrite_chain.run(query=user_query)
    print("🔄 Rewritten query:", rewritten.strip())

    result = qa_chain({"query": rewritten})
    answer = result["result"]
    docs = result["source_documents"]
    return answer, docs

# Usage
if __name__ == "__main__":
    ans, ctx = answer_with_query_rewrite("model quality issues")
    print("Answer:", ans)
    for doc in ctx:
        print("•", doc.metadata.get("source"), doc.page_content[:100], "...")
