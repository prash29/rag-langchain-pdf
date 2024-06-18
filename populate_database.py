import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from utils import *
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Resets the Chroma database")
    args = parser.parse_args()
    db_populator = PopulateDatabase(reset=args.reset)
    db_populator.execute()


if __name__ == "__main__":
    print("=========== Populating the database ============")
    main()