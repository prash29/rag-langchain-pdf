import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

CHROMA_PATH = "chroma"
DATA_PATH = "data"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
Now, Answer the question based on the above context: {question}
"""

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


class PopulateDatabase():
    def __init__(self, reset = False, chunk_size = 1000, chunk_overlap = 80) -> None:
        self.reset = reset
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        

    def clear_database(self):
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
    
    def load_documents(self):
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        return document_loader.load()
    
    def calculate_chunk_ids(self, chunks : list[Document]):
        last_page_id = None
        curr_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            curr_page_id = f"{source}:{page}"

            # If page ID is the same as last one, increment index
            if curr_page_id == last_page_id:
                curr_chunk_index+=1
            else:
                curr_chunk_index = 0
            
            # Calc. Chunk ID
            chunk_id = f"{curr_page_id}:{curr_chunk_index}"
            last_page_id = curr_page_id
            chunk.metadata["id"] = chunk_id
        return chunks
    
    def add_to_chromadb(self, chunks: list[Document]):
        # Load the existing database
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )

        # Calculate IDs for pages
        chunk_ids = self.calculate_chunk_ids(chunks)

        # Add/Update the documents to the DB
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in the DB : {len(existing_ids)}")

        # Only add new documents to the DB
        new_chunks = []
        for chunk in chunk_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
        
        if len(new_chunks):
            print(f"Adding new documents : {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids = new_chunk_ids)
            db.persist()
        else:
            print(f"No new documents to add right now!")
    
    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = len,
            is_separator_regex=False
        )
        return text_splitter.split_documents(documents)
    
    def execute(self):
        if self.reset:
            print("Clearing the database!")
            self.clear_database()
        
        # Create/update the data store
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.add_to_chromadb(chunks)


def query_rag(query_text: str):
    # Prepare the DB
    print("Preparing the DB")
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function
    )

    # Search the DB
    print("Searching the DB")
    results = db.similarity_search_with_score(query_text, k = 3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query_text)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text} \n\nSources: {sources}"
    print(formatted_response)
    return response_text, sources
