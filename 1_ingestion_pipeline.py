import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#1. Load Document Method
def load_documents(docs_path="docs"):
    print(f"******Loading documents from {docs_path}...******")

    #check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists. Please make sure path is correct and contains the documents.")

    #Load all .txt files from the docs directory
    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt file found in {docs_path} directory. Please add the documents.")

    for index, doc in enumerate(documents[:2]): #shows only first two documents
        print(f"\nDocument {index+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")

    return documents


#2. Chuncking method
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("\n******Splitting documents into chunks******")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f" Source: {chunk.metadata['source']}")
            print(f" Length: {len(chunk.page_content)} characters")
            print(f" Content:")
            print(chunk.page_content)
            print("-"*30)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks)-5} more chunks")

    return chunks

#3. Creating Vector DB
def create_vector_store(chunks, persistant_directory="db/chroma.db"):
    print(f"******Creating Embeddings and Storing in ChromaDB******")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("---Creating Vector Store---")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persistant_directory, collection_metadata={"hnsw:space":"cosine"})
    print("---Finish Creating Vector Store---")
    print(f"Vectore Store created and saved to {persistant_directory}")

    return vector_store



def main():

    #1. Loading the files
    documents = load_documents(docs_path="docs")
    #2. Chunking the files
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=0)
    #3. Embedding and storing in a Vector DB
    vectore_store = create_vector_store(chunks)

if __name__ == "__main__":
    main()