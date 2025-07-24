from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
import time

# Initialize models and retriever
llm = ChatOpenAI(temperature=0)
vector_store = FAISS.load_local("faiss_index", embeddings=None)

# 1. Retriever
retriever = vector_store.as_retriever()

# 2. Retrieval Evaluator (LLM as judge)
eval_prompt = PromptTemplate(
    "Score the following documents for relevance to the query on scale of 0-1:\n\n"
    "Query: {query}\n\nDocuments:\n{docs}\n\nScores:"
)
evaluator_chain = LLMChain(llm=llm, prompt=eval_prompt)

# 3. Generator chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
)

def corrective_rag(query: str, threshold: float = 0.7):
    docs = retriever.get_relevant_documents(query)
    scores = evaluator_chain.run(query=query, docs="\n---\n".join([d.page_content[:200] for d in docs]))
    avg_score = sum(map(float, scores.strip().split())) / len(docs)

    if avg_score < threshold:
        # Re-retrieve with extended query or fallback to web search
        query2 = query + " (explain in detail)"
        docs = retriever.get_relevant_documents(query2)

    return qa_chain({"query": query, "retrieved_documents": docs})

if __name__ == "__main__":
    answer = corrective_rag("What causes allergies?")
    print(answer["result"])
