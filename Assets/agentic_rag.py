from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain import OpenAIEmbeddings

# Setup
llm = ChatOpenAI(temperature=0.0)  # acts as orchestrator and generator
vector_index = FAISS.load_local("faiss", OpenAIEmbeddings())

# Prompt templates
query_analysis = PromptTemplate(
    "Analyze the query and decide retrieval tools: \nUser Query: {query}\n\n"
    "Return JSON with fields 'use_vector', 'use_web' or 'direct_generate'."
)

tool_selector = PromptTemplate(
    "Based on the query decomposition: {analysis}. Generate sub-query for vector or web search."
)

retrieval_prompt = PromptTemplate(
    "Subquery: {subquery}\n\nReturn a retrieval prompt for the retriever."
)

generation_prompt = PromptTemplate(
    "Question: {query}\n\nContext:\n{context}\n\nAnswer:"
)

# Chains
from langchain.chains import LLMChain, RetrievalQA

analyzer = LLMChain(llm=llm, prompt=query_analysis)
selector = LLMChain(llm=llm, prompt=tool_selector)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_index.as_retriever(),
    prompt=generation_prompt
)

def agentic_rag(query: str):
    analysis = analyzer.run(query=query)
    # parse JSON logic here (omitted for brevity)

    subquery = selector.run(analysis=analysis)

    docs = vector_index.similarity_search(subquery, k=3)
    context = "\n---\n".join([d.page_content for d in docs])

    answer = llm.generate([{"text": f"{query}\n\nContext:\n{context}"}]).generations[0].text
    return answer, docs

if __name__ == "__main__":
    resp, docs = agentic_rag("Explain risks of insulin for elderly kidney patients")
    print("Response:\n", resp)
    for d in docs:
        print("•", d.metadata.get("source"))
