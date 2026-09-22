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

print(f"User Query: {query}")
#Result
print("---Context---")
for i, doc in enumerate(relavant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
