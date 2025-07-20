import os
import uuid
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import PyPDF2
import fitz  # PyMuPDF
from pathlib import Path
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_DIRECTORY_PATH = "../pdf_datasets/"

api_key_txt='dummy_api_key'  # Replace with your actual API key
tenant_txt='dummy_tenant_id'  # Replace with your actual tenant ID

class PDFToChromaIngester:
    def __init__(self, chroma_db_path: str = "./chroma_db", collection_name: str = "airline_travel_docs"):
        """Initialize the PDF ingester for ChromaDB"""
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name

        # Initialize ChromaDB client
        #self.client = chromadb.PersistentClient(path=chroma_db_path)

        self.client = chromadb.CloudClient(api_key=api_key_txt,
                                           tenant=tenant_txt,
                                           database='skyline_airways_vector')
        

        # Delete existing collection if it exists (fresh start)
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except:
            pass

        # Create new collection with explicit embedding function
        try:
            from sentence_transformers import SentenceTransformer

            # Custom embedding function using BAAI/bge-small-en-v1.5
            class CustomEmbeddingFunction:
                def __init__(self):
                    self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')

                def __call__(self, input: List[str]) -> List[List[float]]:
                    processed_input = [f"passage: {text}" for text in input]
                    embeddings = self.model.encode(processed_input, show_progress_bar=False, normalize_embeddings=True)
                    return embeddings.tolist()

            self.embedding_function = CustomEmbeddingFunction()

            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Created collection with BGE embedding: {collection_name}")

        except Exception as e:
            logger.warning(f"SentenceTransformer failed: {e}")

            # Fallback: Try with default ChromaDB embedding
            try:
                self.collection = self.client.create_collection(name=collection_name)
                logger.info(f"Created collection with default embedding: {collection_name}")
            except Exception as e2:
                logger.error(f"Failed to create collection: {e2}")
                raise e2

    def extract_text_pypdf2(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2 from {pdf_path}: {e}")
            return ""

    def extract_text_pymupdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF from {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 50) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                sentence_ends = ['. ', '! ', '? ', '\n\n']
                best_break = -1
                for i in range(end - overlap, end):
                    for ending in sentence_ends:
                        if text[i:i+len(ending)] == ending:
                            best_break = i + len(ending)
                if best_break != -1:
                    end = best_break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def ingest_pdf(self, pdf_path: str, metadata: Dict[str, Any] = None) -> bool:
        try:
            text = self.extract_text_pypdf2(pdf_path)
            if not text.strip():
                logger.warning(f"PyPDF2 failed for {pdf_path}, trying PyMuPDF...")
                text = self.extract_text_pymupdf(pdf_path)
            if not text.strip():
                logger.error(f"Could not extract text from {pdf_path}")
                return False
            chunks = self.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks from {os.path.basename(pdf_path)}")
            file_metadata = {
                "source": pdf_path,
                "filename": os.path.basename(pdf_path),
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            documents_list = []
            metadatas_list = []
            ids_list = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **file_metadata,
                    "chunk_index": i,
                    "chunk_id": f"{os.path.basename(pdf_path)}_chunk_{i}"
                }
                documents_list.append(chunk)
                metadatas_list.append(chunk_metadata)
                ids_list.append(str(uuid.uuid4()))
            if documents_list:
                logger.info(f"Adding {len(documents_list)} documents to ChromaDB")
                assert len(documents_list) == len(metadatas_list) == len(ids_list)
                self.collection.add(
                    documents=documents_list,
                    metadatas=metadatas_list,
                    ids=ids_list
                )
                logger.info("Successfully added documents to collection.")
            logger.info(f"Successfully ingested {pdf_path} with {len(chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error ingesting {pdf_path}: {e}")
            return False

    def ingest_directory(self, directory_path: str, metadata: Dict[str, Any] = None) -> Dict[str, bool]:
        results = {}
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return results
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            result = self.ingest_pdf(str(pdf_file), metadata)
            results[str(pdf_file)] = result
        return results

    def search_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        try:
            results = self.collection.query(
                query_texts=[f"query: {query}"],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {}

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def search_for_rag(self, query: str, n_results: int = 5, filter_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            search_params = {
                "query_texts": [f"query: {query}"],
                "n_results": n_results
            }
            if filter_metadata:
                search_params["where"] = filter_metadata
            results = self.collection.query(**search_params)
            if results and results.get('documents'):
                formatted_results = []
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                        "distance": results['distances'][0][i] if results.get('distances') else None,
                        "id": results['ids'][0][i] if results.get('ids') else None
                    }
                    formatted_results.append(result)
                return {
                    "query": query,
                    "results": formatted_results,
                    "total_results": len(formatted_results)
                }
            return {"query": query, "results": [], "total_results": 0}
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {"query": query, "results": [], "total_results": 0, "error": str(e)}

    def get_context_for_rag(self, query: str, max_context_length: int = 4000) -> str:
        search_results = self.search_for_rag(query, n_results=10)
        if not search_results.get("results"):
            return ""
        context_parts = []
        current_length = 0
        for i, result in enumerate(search_results["results"]):
            content = result["content"]
            source = result["metadata"].get("filename", "Unknown")
            chunk_id = result["metadata"].get("chunk_id", f"chunk_{i}")
            context_piece = f"[Source: {source}, Chunk: {chunk_id}]\n{content}\n"
            if current_length + len(context_piece) > max_context_length:
                break
            context_parts.append(context_piece)
            current_length += len(context_piece)
        return "\n---\n".join(context_parts)

def demonstrate_rag_workflow(ingester: PDFToChromaIngester):
    print("=== INGESTING DOCUMENTS ===")
    results = ingester.ingest_directory(
        PDF_DIRECTORY_PATH,
        metadata={"category": "academic", "project": "rag_demo"}
    )
    print("\n=== COLLECTION STATS ===")
    stats = ingester.get_collection_stats()
    print(json.dumps(stats, indent=2))
    print("\n=== RAG SEARCH DEMO ===")
    query = "What all things to check before taking a travel insurance for international flights?"
    search_results = ingester.search_for_rag(query, n_results=3)
    print(f"Query: {search_results['query']}")
    print(f"Found {search_results['total_results']} results")
    context = ingester.get_context_for_rag(query, max_context_length=2000)
    print(f"\nFormatted context for LLM ({len(context)} chars):")
    print(context[:500] + "..." if len(context) > 500 else context)
    return context

if __name__ == "__main__":
    print("Starting PDF ingestion process...")
    ingester = PDFToChromaIngester(
        chroma_db_path="./airline_chroma_db",
        collection_name="airline_travel_docs"
    )
    print("PDF ingester initialized successfully.")
    results = ingester.ingest_directory(
        PDF_DIRECTORY_PATH,
        metadata={"category": "airline_docs", "ingestion_date": "2025-07-13"}
    )
    print("\nPDF ingestion completed.")
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    print(f"Successfully processed: {successful}/{total} files")
    for file_path, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {os.path.basename(file_path)}")
    stats = ingester.get_collection_stats()
    print(f"\nCollection stats: {stats}")
    if stats.get("total_documents", 0) > 0:
        print("\nTesting search functionality Query : refund policy ")
        search_results = ingester.search_documents("refund policy", n_results=4)
        if search_results and search_results.get('documents'):
            print(f"Found {len(search_results['documents'][0])} search results")
            for i, doc in enumerate(search_results['documents'][0]):
                print(f"\nResult {i+1}:")
                print(f"Text: {doc[:150]}...")
                if search_results.get('metadatas') and search_results['metadatas'][0]:
                    metadata = search_results['metadatas'][0][i]
                    print(f"Source: {metadata.get('filename', 'Unknown')}")
        else:
            print("No search results found")
    print("\nPDFToChromaIngester completed successfully!")
    context = demonstrate_rag_workflow(ingester)
    print("\nRAG workflow demonstration completed.")
    print(f"Context length: {len(context)} characters")
