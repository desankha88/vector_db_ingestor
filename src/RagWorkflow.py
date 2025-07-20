import chromadb
import logging
from chromadb.config import Settings
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
max_context_length = 2000
my_collection_name="airline_travel_docs"
api_key_txt='dummy_api_key'  # Replace with your actual API key
tenant_txt='dummy_tenant_id'

def demonstrate_rag_workflow(  query_txt : str , filter_metadata: Dict[str, Any] = None):
    """Demonstrate RAG workflow with a sample query"""

    client = chromadb.HttpClient(
        ssl=True,
        host='api.trychroma.com',
        tenant=tenant_txt,
        database='skyline_airways_vector',
        headers={'x-chroma-token': api_key_txt}
        )

    collection = client.get_collection(name=my_collection_name)

    print(client.list_collections())

    print("\n=== RAG SEARCH DEMO ===")

    if query_txt is None or query_txt.strip() == "":
        logger.error("Query text is empty or None.")
        return {"query": "", "results": [], "total_results": 0, "error": "Query text is empty or None."}

    query = query_txt

    n_results=3
    
    try:
        search_params = {
            "query_texts": [query],  
            "n_results": n_results
            }

        if filter_metadata:
            search_params["where"] = filter_metadata
        results = collection.query(**search_params)
        
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
            
            search_results = {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
                }
        else:
            search_results = {"query": query, "results": [], "total_results": 0}

        logger.info("Querying with:", search_params)
        logger.info("Raw results:", results)

        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return {"query": query, "results": [], "total_results": 0, "error": str(e)}
    
    
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
    context = "\n---\n".join(context_parts)
    
    print(f"Query: {search_results['query']}")
    print(f"Found {search_results['total_results']} results")
        
    print(f"\nFormatted context for LLM ({len(context)} chars):")
    print(context[:500] + "..." if len(context) > 500 else context)
    return context



if __name__ == "__main__":
        
    # Demonstrate the RAG workflow
    context = demonstrate_rag_workflow( "How to do web check-in?  ?", )

    # Optionally, you can use this context with an LLM for further processing
    # For example, passing it to a language model for generating responses