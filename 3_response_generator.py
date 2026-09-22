from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

persistant_directory = "db/chroma.db"

#Load embedding and vector sotre
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistant_directory, embedding_function=embedding_model, collection_metadata={"hnsw:space": "cosine"})

#Search for relavant documents
query = "In what year does the Tesla begin production of thr Roadster"

retriever = db.as_retriever(search_kwargs={"k":3})

# retriever = db.as_retriever(
#     search_type="similarity_score_threashold",
#     search_kwargs={
#         "k":5,
#         "score_threashold": 0.3 #only return chunks with cosine similarity >= 0.3
#     }
# )

relavant_docs = retriever.invoke(query)

#Combining the query and the relavant document contents
combined_input = f""""Based on the following documents, please answer for this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relavant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say, I don't have enough information to answer that question based on the provided documents.
"""

model = ChatOpenAI(model="gpt-4o")

message = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

result = model.invoke(message)

print("\n--- Generated Response ---")
print("Content only:")
print(result.content)