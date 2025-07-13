import os
import uuid
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import PyPDF2
import fitz  # PyMuPDF
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_DIRECTORY_PATH = "/content/"

class PDFToChromaIngester:
    def __init__(self, chroma_db_path: str = "./chroma_db", collection_name: str = "airline_travel_docs"):
        """Initialize the PDF ingester for ChromaDB"""
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Delete existing collection if it exists (fresh start)
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except:
            pass
        
        # Create new collection with explicit embedding function
        try:
            # Try with sentence transformers first
            from sentence_transformers import SentenceTransformer
            
            # Create a simple custom embedding function
            class CustomEmbeddingFunction:
                def __init__(self):
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
                
                def __call__(self, input: List[str]) -> List[List[float]]:
                    embeddings = self.model.encode(input, show_progress_bar=False)
                    return embeddings.tolist()
            
            self.embedding_function = CustomEmbeddingFunction()
            
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Created collection with SentenceTransformer: {collection_name}")
            
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
        """Extract text from PDF using PyPDF2"""
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
        """Extract text from PDF using PyMuPDF"""
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
        """Split text into chunks for better retrieval"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at a sentence or paragraph boundary
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
        """Ingest a single PDF file into ChromaDB"""
        try:
            # Extract text - try both methods
            text = self.extract_text_pypdf2(pdf_path)
            
            if not text.strip():
                logger.warning(f"PyPDF2 failed for {pdf_path}, trying PyMuPDF...")
                text = self.extract_text_pymupdf(pdf_path)
            
            if not text.strip():
                logger.error(f"Could not extract text from {pdf_path}")
                return False
            
            # Chunk the text
            chunks = self.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks from {os.path.basename(pdf_path)}")
            
            # Prepare metadata
            file_metadata = {
                "source": pdf_path,
                "filename": os.path.basename(pdf_path),
                "total_chunks": len(chunks),
                **(metadata or {})
            }

            print('file_metadata:', file_metadata)
            
            # Prepare data for ChromaDB
            documents_list = []
            metadatas_list = []
            ids_list = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **file_metadata,
                    "chunk_index": i,
                    "chunk_id": f"{os.path.basename(pdf_path)}_chunk_{i}"
                }

                print('chunk_metadata:', chunk_metadata)

                documents_list.append(chunk)
                metadatas_list.append(chunk_metadata)
                ids_list.append(str(uuid.uuid4()))

                #print('Length of documents:', len(documents))
                #print('Length of ids:', len(ids))
                #print('Length of metadatas:', len(metadatas))
                #logger.info('ids:', ids)

                #print('Insert chunk into ChromaDB:', chunk[:50], '...')  # Print first 50 chars of chunk

            if documents_list:  # Only add if we have documents
                logger.info(f"Adding {len(documents_list)} documents to ChromaDB")

                # Verify all lists have the same length (optional but good practice)
                assert len(documents_list) == len(metadatas_list) == len(ids_list), \
                f"Length mismatch: docs={len(documents_list)}, metadata={len(metadatas_list)}, ids={len(ids_list)}"

                print(f"docs={len(documents_list)}, metadata={len(metadatas_list)}, ids={len(ids_list)}")

                try:
                    self.collection.add(
                        documents=documents_list[0:2],
                        metadatas=metadatas_list[0:2],
                        ids=ids_list[0:2]
                    )
                    print("Reached after .add()")
                    logger.info("Successfully added documents to collection.")
                except Exception as e:
                    logger.error(f"Failed to add documents to collection: {e}")
                    return False

            logger.info(f"Successfully ingested {pdf_path} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting {pdf_path}: {e}")
            return False
    
    def ingest_directory(self, directory_path: str, metadata: Dict[str, Any] = None) -> Dict[str, bool]:
        """Ingest all PDF files from a directory"""
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
        """Search documents in the collection"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

# Usage example
if __name__ == "__main__":
    print("Starting PDF ingestion process...")
    
    # Initialize the ingester
    ingester = PDFToChromaIngester(
        chroma_db_path="./airline_chroma_db",
        collection_name="airline_travel_docs"
    )

    print("PDF ingester initialized successfully.")

    # Ingest all PDFs from a directory
    results = ingester.ingest_directory(
        PDF_DIRECTORY_PATH,
        metadata={"category": "airline_docs", "ingestion_date": "2025-07-13"}
    )

    print("\nPDF ingestion completed.")
    print(f"Results summary:")
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    print(f"Successfully processed: {successful}/{total} files")
    
    # Show detailed results
    for file_path, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {os.path.basename(file_path)}")
    
    # Get collection statistics
    stats = ingester.get_collection_stats()
    print(f"\nCollection stats: {stats}")
    
    # Test search
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