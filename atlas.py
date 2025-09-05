#!/usr/bin/env python3

import argparse
import sys
import os
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
from file_analyzer import FileAnalyzer
from typing import List, Dict, Any, Optional
import hashlib
import json
import asyncio

# Import environment configuration and validation
from env_config import EnvironmentConfig
from env_validator import EnvironmentValidator

# Don't load .env by default - will be handled by EnvironmentConfig
# load_dotenv()

class Neo4jGraphStore:
    """Neo4j graph store for real metrics tracking"""
    def __init__(self, uri=None, username=None, password=None):
        from langchain_neo4j import Neo4jGraph
        
        # Use environment variables or defaults
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'password')
        
        try:
            self.graph = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                enhanced_schema=True,
                refresh_schema=True
                )
            print(f"✅ Connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            print(f"   Using URI: {self.uri}")
            print(f"   Using Username: {self.username}")
            raise
    
    def get_stats(self):
        """Get current node and relationship counts from Neo4j"""
        try:
            # Count nodes
            node_result = self.graph.query("MATCH (n) RETURN count(n) as node_count")
            node_count = node_result[0]['node_count'] if node_result else 0
            
            # Count relationships
            rel_result = self.graph.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = rel_result[0]['rel_count'] if rel_result else 0
            
            return {'nodes': node_count, 'relationships': rel_count}
        except Exception as e:
            print(f"⚠️ Error getting graph stats: {e}")
            return {'nodes': 0, 'relationships': 0}
    
    def add_graph_documents(self, graph_documents):
        """Add graph documents to Neo4j"""
        if graph_documents:
            self.graph.add_graph_documents(graph_documents,
                                            include_source=True,
                                            baseEntityLabel=True)
            print(f"✅ Added {len(graph_documents)} graph documents to Neo4j")
    
    def query(self, query_str, params=None):
        """Execute a Cypher query on the Neo4j database
        
        Args:
            query_str: Cypher query string
            params: Optional parameters for the query
            
        Returns:
            Query results
        """
        return self.graph.query(query_str, params or {})
    
    # Display a summary of the database. show list each label and relationship type with counts, in descending order.  
    def display_summary(self):  
        """Display a comprehensive summary of the Neo4j database"""
        try:
            # Get overall stats first
            stats = self.get_stats()
            
            # Get node labels and counts
            node_labels = self.graph.query(
                """CALL db.labels() YIELD label MATCH (n) WHERE label IN labels(n) 
                RETURN label, count(n) as node_count ORDER BY node_count DESC""")
            
            # Get relationship types and counts
            rel_types = self.graph.query(
                """CALL db.relationshipTypes() YIELD relationshipType
                MATCH ()-[r]->()
                WHERE type(r) = relationshipType
                RETURN relationshipType, count(r) as relationship_count
                ORDER BY relationship_count DESC""")
            
            # Display header
            print("\n" + "="*80)
            print("📊 KNOWLEDGE GRAPH DATABASE SUMMARY")
            print("="*80)
            
            # Overall statistics
            print(f"\n🎯 OVERVIEW:")
            print(f"   Total Nodes:        {stats['nodes']:,}")
            print(f"   Total Relationships: {stats['relationships']:,}")
            print(f"   Node Types:         {len(node_labels)}")
            print(f"   Relationship Types: {len(rel_types)}")
            
            # Node labels section
            if node_labels:
                print(f"\n📋 NODE LABELS ({len(node_labels)} types):")
                print("-" * 60)
                
                # Calculate max label length for alignment
                max_label_length = max(len(label['label']) for label in node_labels) if node_labels else 0
                max_count_length = max(len(f"{label['node_count']:,}") for label in node_labels) if node_labels else 0
                
                # Header
                print(f"   {'Label':<{max_label_length}} | {'Count':>{max_count_length}} | Percentage")
                print(f"   {'-' * max_label_length}-+-{'-' * max_count_length}-+-----------")
                
                # Node data with percentages
                total_nodes = stats['nodes']
                for label in node_labels:
                    count = label['node_count']
                    percentage = (count / total_nodes * 100) if total_nodes > 0 else 0
                    print(f"   {label['label']:<{max_label_length}} | {count:>{max_count_length},} | {percentage:>8.1f}%")
                
                print(f"   {'-' * max_label_length}-+-{'-' * max_count_length}-+-----------")
                print(f"   {'TOTAL':<{max_label_length}} | {total_nodes:>{max_count_length},} | {100.0:>8.1f}%")
            else:
                print(f"\n📋 NODE LABELS: No node labels found")
            
            # Relationship types section
            if rel_types:
                print(f"\n🔗 RELATIONSHIP TYPES ({len(rel_types)} types):")
                print("-" * 60)
                
                # Calculate max relationship type length for alignment
                max_rel_length = max(len(rel['relationshipType']) for rel in rel_types) if rel_types else 0
                max_rel_count_length = max(len(f"{rel['relationship_count']:,}") for rel in rel_types) if rel_types else 0
                
                # Header
                print(f"   {'Relationship Type':<{max_rel_length}} | {'Count':>{max_rel_count_length}} | Percentage")
                print(f"   {'-' * max_rel_length}-+-{'-' * max_rel_count_length}-+-----------")
                
                # Relationship data with percentages
                total_relationships = stats['relationships']
                for rel in rel_types:
                    count = rel['relationship_count']
                    percentage = (count / total_relationships * 100) if total_relationships > 0 else 0
                    print(f"   {rel['relationshipType']:<{max_rel_length}} | {count:>{max_rel_count_length},} | {percentage:>8.1f}%")
                
                print(f"   {'-' * max_rel_length}-+-{'-' * max_rel_count_length}-+-----------")
                print(f"   {'TOTAL':<{max_rel_length}} | {total_relationships:>{max_rel_count_length},} | {100.0:>8.1f}%")
            else:
                print(f"\n🔗 RELATIONSHIP TYPES: No relationship types found")
            
            # Footer
            print("\n" + "="*80)
            print("💾 Database ready for querying and analysis")
            print("="*80)
            
        except Exception as e:
            print(f"⚠️ Error displaying summary: {e}")
            return
    
    def setup_vector_index(self, embedding_dimension: int = 1536):
        """Setup vector index in Neo4j for document search"""
        try:
            # Create vector index for document chunks
            vector_index_query = f"""
            CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
            FOR (d:DocumentChunk) ON (d.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {embedding_dimension},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            self.graph.query(vector_index_query)
            
            # Create text index for full-text search
            text_index_query = """
            CREATE FULLTEXT INDEX document_content IF NOT EXISTS
            FOR (d:DocumentChunk) ON (d.content, d.search_terms)
            """
            self.graph.query(text_index_query)
            
            print("✅ Vector and text indexes created in Neo4j")
            
        except Exception as e:
            print(f"⚠️ Error setting up vector index: {e}")
    
    def add_document_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add document chunks with embeddings to Neo4j"""
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
            
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                
                # Prepare batch query
                query = """
                UNWIND $batch as item
                CREATE (d:DocumentChunk {
                    id: item.chunk.id,
                    content: item.chunk.content,
                    filename: item.chunk.metadata.filename,
                    source_file: item.chunk.metadata.source_file,
                    file_type: item.chunk.metadata.file_type,
                    chunk_index: item.chunk.metadata.chunk_index,
                    total_chunks: item.chunk.metadata.total_chunks,
                    chunk_size: item.chunk.metadata.chunk_size,
                    search_terms: item.chunk.search_terms,
                    embedding: item.embedding,
                    created_at: datetime(item.chunk.metadata.created_at),
                    indexed_at: datetime()
                })
                
                // Connect to source file if it exists
                WITH d, item
                OPTIONAL MATCH (f:Document {name: item.chunk.metadata.filename})
                FOREACH (file IN CASE WHEN f IS NOT NULL THEN [f] ELSE [] END |
                    CREATE (file)-[:CONTAINS_CHUNK]->(d)
                )
                """
                
                # Prepare batch data
                batch_data = []
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    # Extract search terms from content
                    search_terms = self._extract_search_terms(chunk['content'])
                    
                    batch_data.append({
                        'chunk': chunk,
                        'embedding': embedding,
                        'search_terms': ' '.join(search_terms)
                    })
                
                self.graph.query(query, {'batch': batch_data})
                
                if (i + batch_size) % 500 == 0:
                    print(f"   📄 Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
            
            print(f"✅ Added {len(chunks)} document chunks to Neo4j vector index")
            
        except Exception as e:
            print(f"❌ Error adding document chunks: {e}")
            raise
    
    def _extract_search_terms(self, content: str) -> List[str]:
        """Extract search terms from document content"""
        import re
        
        terms = set()
        
        # Extract function/class names
        function_pattern = r'(?:def|function|class|interface)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        terms.update(re.findall(function_pattern, content, re.IGNORECASE))
        
        # Extract variable names
        variable_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        terms.update(re.findall(variable_pattern, content))
        
        # Extract import statements
        import_pattern = r'(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        terms.update(re.findall(import_pattern, content))
        
        # Extract meaningful words (3+ characters)
        word_pattern = r'\b[a-zA-Z]{3,}\b'
        words = re.findall(word_pattern, content.lower())
        
        # Filter common words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'this', 'that', 'with', 'have', 'will', 'your', 'from', 'they', 'been', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'what', 'about', 'when', 'where', 'could', 'should', 'after', 'first', 'well', 'just', 'also', 'then', 'them', 'these', 'some', 'come', 'only', 'into', 'know', 'work', 'life', 'year', 'back', 'make', 'take', 'look', 'good', 'think', 'being'}
        meaningful_words = [word for word in words if word not in common_words and len(word) >= 3]
        terms.update(meaningful_words[:20])  # Limit to prevent bloat
        
        return list(terms)[:50]  # Limit total terms
    
    def search_documents(self, query: str, k: int = 10, embedding_vector: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Search documents using vector similarity and/or text search"""
        try:
            results = []
            
            if embedding_vector:
                # Vector similarity search
                vector_query = """
                CALL db.index.vector.queryNodes('document_embeddings', $k, $embedding)
                YIELD node, score
                RETURN node.id as id,
                       node.content as content,
                       node.filename as filename,
                       node.source_file as source_file,
                       node.file_type as file_type,
                       node.chunk_index as chunk_index,
                       score
                ORDER BY score DESC
                """
                
                vector_results = self.graph.query(vector_query, {
                    'embedding': embedding_vector,
                    'k': k
                })
                results.extend(vector_results)
            
            # Fallback to text search if no vector search or additional results needed
            if not embedding_vector or len(results) < k:
                text_query = """
                CALL db.index.fulltext.queryNodes('document_content', $query)
                YIELD node, score
                RETURN node.id as id,
                       node.content as content,
                       node.filename as filename,
                       node.source_file as source_file,
                       node.file_type as file_type,
                       node.chunk_index as chunk_index,
                       score
                ORDER BY score DESC
                LIMIT $limit
                """
                
                text_results = self.graph.query(text_query, {
                    'query': query,
                    'limit': k - len(results) if results else k
                })
                results.extend(text_results)
            
            return results[:k]
            
        except Exception as e:
            print(f"⚠️ Error searching documents: {e}")
            return []
    
    def get_document_index_stats(self) -> Dict[str, int]:
        """Get document index statistics"""
        try:
            # Count document chunks
            chunk_result = self.graph.query("MATCH (d:DocumentChunk) RETURN count(d) as chunk_count")
            chunk_count = chunk_result[0]['chunk_count'] if chunk_result else 0
            
            # Count unique files
            file_result = self.graph.query("MATCH (d:DocumentChunk) RETURN count(DISTINCT d.source_file) as file_count")
            file_count = file_result[0]['file_count'] if file_result else 0
            
            # Count total search terms
            terms_result = self.graph.query("""
                MATCH (d:DocumentChunk) 
                WHERE d.search_terms IS NOT NULL 
                RETURN sum(size(split(d.search_terms, ' '))) as term_count
            """)
            term_count = terms_result[0]['term_count'] if terms_result else 0
            
            return {
                'documents_indexed': file_count,
                'content_chunks': chunk_count,
                'search_terms_extracted': term_count
            }
            
        except Exception as e:
            print(f"⚠️ Error getting document index stats: {e}")
            return {'documents_indexed': 0, 'content_chunks': 0, 'search_terms_extracted': 0}

                                        

def initialize_llm(provider, model):
    """Initialize LLM based on provider"""
    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0)
    elif provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0,max_tokens=10000)
    elif provider == 'ollama':
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0)
    elif provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0, convert_system_message_to_human=True)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def chunk_document(content: str, file_metadata, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Chunk document content for vector indexing
    
    Args:
        content: Document content to chunk
        file_metadata: File metadata object
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with content and metadata
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split content into chunks
        chunks = text_splitter.split_text(content)
        
    except ImportError:
        print("⚠️ LangChain text splitter not available, using basic chunking")
        # Basic chunking fallback
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - chunk_overlap
    
    # Create chunk objects with metadata
    chunk_objects = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Skip empty chunks
            chunk_id = hashlib.md5(f"{file_metadata.file_path}_{i}".encode()).hexdigest()
            
            chunk_obj = {
                'id': chunk_id,
                'content': chunk.strip(),
                'metadata': {
                    'source_file': str(file_metadata.file_path),
                    'filename': file_metadata.filename,
                    'file_type': file_metadata.file_type,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'file_size': file_metadata.size_bytes,
                    'chunk_size': len(chunk),
                    'created_at': file_metadata.created_at.isoformat() if file_metadata.created_at else None,
                    'modified_at': file_metadata.modified_at.isoformat() if file_metadata.modified_at else None
                }
            }
            chunk_objects.append(chunk_obj)
    
    return chunk_objects


def generate_embeddings(texts: List[str], llm_provider: str, model: str) -> List[List[float]]:
    """Generate embeddings for text chunks"""
    try:
        if llm_provider == 'openai':
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=model if 'embed' in model else 'text-embedding-3-small')
        elif llm_provider == 'anthropic':
            # Anthropic doesn't have embeddings, fall back to OpenAI
            from langchain_openai import OpenAIEmbeddings
            print("⚠️ Anthropic doesn't provide embeddings, using OpenAI embeddings")
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        elif llm_provider == 'ollama':
            from langchain_ollama import OllamaEmbeddings
            # Use a common embedding model for Ollama
            embedding_model = 'nomic-embed-text' if 'nomic' not in model else model
            embeddings = OllamaEmbeddings(model=embedding_model)
        elif llm_provider == 'google':
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            # Use Google's text embedding model
            embedding_model = 'models/text-embedding-004' if 'embed' not in model else model
            embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
        else:
            raise ValueError(f"Unsupported embedding provider: {llm_provider}")
        
        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 500 == 0:
                print(f"   🔢 Generated embeddings for {min(i + batch_size, len(texts))}/{len(texts)} chunks...")
        
        return all_embeddings
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        raise


async def index_documents(files_metadata: List[Any], batch_size: int, llm_provider: str, model: str, graph_store) -> Dict[str, int]:
    """
    Index documents with vector embeddings in Neo4j
    
    Args:
        files_metadata: List of file metadata objects
        batch_size: Batch size for processing
        llm_provider: LLM provider for embeddings
        model: Model name for embeddings
        graph_store: Neo4j graph store instance
        
    Returns:
        Dictionary with indexing statistics
    """
    print(f"\n🔍 Starting Document Vector Indexing...")
    print(f"📊 Processing {len(files_metadata)} files in batches of {batch_size}")
    
    # Setup vector index in Neo4j
    try:
        graph_store.setup_vector_index()
    except Exception as e:
        print(f"⚠️ Error setting up vector index, continuing anyway: {e}")
    
    all_chunks = []
    processed_files = 0
    failed_files = 0
    
    # Process files in batches
    total_batches = (len(files_metadata) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(files_metadata))
        batch_files = files_metadata[start_idx:end_idx]
        
        print(f"\n🔄 Processing Batch {batch_num + 1}/{total_batches}")
        print(f"   Files {start_idx + 1}-{end_idx} of {len(files_metadata)}")
        
        batch_chunks = []
        
        # Chunk documents in this batch
        for file_metadata in batch_files:
            try:
                # Read file content
                with open(file_metadata.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Chunk the document
                chunks = chunk_document(content, file_metadata)
                batch_chunks.extend(chunks)
                processed_files += 1
                
            except Exception as e:
                print(f"   ⚠️ Error processing {file_metadata.filename}: {str(e)}")
                failed_files += 1
                continue
        
        # Generate embeddings for this batch
        if batch_chunks:
            try:
                print(f"   🔢 Generating embeddings for {len(batch_chunks)} chunks...")
                texts = [chunk['content'] for chunk in batch_chunks]
                embeddings = generate_embeddings(texts, llm_provider, model)
                
                # Add to Neo4j
                print(f"   💾 Adding chunks to Neo4j vector index...")
                graph_store.add_document_chunks(batch_chunks, embeddings)
                
                all_chunks.extend(batch_chunks)
                
                print(f"   ✅ Batch {batch_num + 1} complete - {len(batch_files)} files, {len(batch_chunks)} chunks")
                
            except Exception as e:
                print(f"   ❌ Error processing batch {batch_num + 1}: {str(e)}")
                failed_files += len(batch_files)
                continue
        
        progress = ((batch_num + 1) / total_batches * 100)
        print(f"   📈 Overall Progress: {progress:.1f}%")
    
    # Get final statistics
    final_stats = graph_store.get_document_index_stats()
    
    print(f"\n🎉 Document Indexing Complete!")
    print(f"📈 Final Statistics:")
    print(f"   📚 {final_stats['documents_indexed']} documents indexed")
    print(f"   📄 {final_stats['content_chunks']} content chunks created")
    print(f"   🔤 {final_stats['search_terms_extracted']} search terms extracted")
    print(f"   ✅ {processed_files} files processed successfully")
    if failed_files > 0:
        print(f"   ⚠️ {failed_files} files failed to process")
    
    return final_stats


def display_banner():
    """Display the Atlas startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     █████╗ ████████╗██╗      █████╗ ███████╗                                 ║
║    ██╔══██╗╚══██╔══╝██║     ██╔══██╗██╔════╝                                 ║
║    ███████║   ██║   ██║     ███████║███████╗                                 ║
║    ██╔══██║   ██║   ██║     ██╔══██║╚════██║                                 ║
║    ██║  ██║   ██║   ███████╗██║  ██║███████║                                 ║
║    ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝                                 ║
║                                                                              ║
║              🗺️  Agentic AI Application Landscape Discovery  🗺️                ║
║                                                                              ║
║    🔍 Analyze • 🧠 Discover • 📊 Map • 🤖 Understand                         ║
║                                                                              ║
║    Navigate the complexity of your application ecosystem with AI-powered     ║
║    analysis, knowledge graph generation, and intelligent discovery.          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def create_parser():
    parser = argparse.ArgumentParser(
        prog='atlas',
        description='Agentic AI solution for analysis and discovery of complex application landscapes'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments that all commands share
    def add_common_args(subparser):
        # Environment configuration arguments
        subparser.add_argument(
            '--env',
            help='Environment to use (default: from ATLAS_ENV or development). Can be any name - will look for .env.{env} file.'
        )
        subparser.add_argument(
            '--env-file',
            help='Path to custom environment file (overrides --env)'
        )
        subparser.add_argument(
            '--export',
            action='store_true',
            help='Export environment variables to shell before running command (useful for other tools)'
        )
        
        # LLM configuration arguments
        # Note: No defaults set here - they're applied from environment in main() function
        subparser.add_argument(
            '--llm-provider',
            choices=['openai', 'anthropic', 'ollama', 'google'],
            help='LLM provider name (default: from DEFAULT_LLM_PROVIDER environment variable)'
        )
        subparser.add_argument(
            '--model',
            help='Model name to use (default: from DEFAULT_LLM_MODEL environment variable)'
        )
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze application landscapes')
    add_common_args(analyze_parser)
    
    # Analysis-specific arguments

    # Add single filename argument
    analyze_parser.add_argument(
        '--file-name',
        help='Single file path to analyze (use instead of --folder-path)'
    )
    # Add folder path argument

    analyze_parser.add_argument(
        '--folder-path',
        required=False,
        help='Path to the folder containing files to analyze (required unless --file-name is provided)'
    )
    analyze_parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process'
    )
    analyze_parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of files to process in each batch (default: 5)'
    )
    analyze_parser.add_argument(
        '--delete-kb',
        action='store_true',
        default=False,
        help='Delete existing knowledge base before processing (default: false)'
    )
    analyze_parser.add_argument(
        '--file-type-filter',
        help='File type filter (e.g., .py, .js, .java). Default is null (wildcard)'
    )
    analyze_parser.add_argument(
        '--generate-knowledge-graph',
        action='store_true',
        default=False,
        help='Generate knowledge graph from processed files (default: false)'
    )
    analyze_parser.add_argument(
        '--index-documents',
        action='store_true',
        default=False,
        help='Index documents for search and retrieval (default: false)'
    )
    analyze_parser.add_argument(
        '--analysis-context',
        choices=['legacy', 'oracle', 'itsm'],
        help='Analysis context/persona: legacy (default), oracle (Oracle Forms/PLSQL), or itsm (IT Service Management tickets). Default from DEFAULT_ANALYSIS_CONTEXT environment variable.'
    )
    
    # Chat command  
    chat_parser = subparsers.add_parser('chat', help='Interactive chat interface')
    add_common_args(chat_parser)
    
    # Refine command
    refine_parser = subparsers.add_parser('refine', help='Refine and optimize knowledge graph')
    add_common_args(refine_parser)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate environment configuration and connectivity')
    validate_parser.add_argument(
        '--env',
        help='Environment to validate (default: from ATLAS_ENV or development). Can be any name - will look for .env.{env} file.'
    )
    validate_parser.add_argument(
        '--env-file',
        help='Path to custom environment file (overrides --env)'
    )
    validate_parser.add_argument(
        '--component',
        choices=['neo4j', 'llm', 'mcp', 'all'],
        default='all',
        help='Component to validate (default: all)'
    )
    validate_parser.add_argument(
        '--save-report',
        help='Save validation report to specified file'
    )
    
    return parser

def export_environment_variables(env_config=None, show_values=False):
    """Export environment variables from loaded config to shell environment.
    Prints export commands that can be sourced by shell.
    SECURITY: Does not display secret values by default for security.
    
    Args:
        env_config: Environment configuration object (can be None for manual fallback)
        show_values: If True, shows actual values (use with caution)
    """
    import sys
    from pathlib import Path
    
    def parse_env_file_manual(env_file_path):
        """Manually parse .env file as fallback when dotenv is not available."""
        env_vars = {}
        try:
            with open(env_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            env_vars[key.strip()] = value.strip()
        except Exception:
            pass
        return env_vars
    
    # Define which variables contain secrets/sensitive data
    SECRET_VARS = {
        'NEO4J_PASSWORD', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 
        'GEMINI_API_KEY', 'LANGSMITH_API_KEY', 'TAVILY_API_KEY'
    }
    
    # Check if output is being piped/redirected (for shell sourcing)
    is_piped = not sys.stdout.isatty()
    
    if not is_piped:
        print("# Environment variables from Atlas configuration", file=sys.stderr)
        print("# To export to your shell, run:", file=sys.stderr)
        env_arg = f" --env {getattr(env_config, 'env', 'default')}" if env_config and hasattr(env_config, 'env') and env_config.env else ""
        print(f"# source <(python atlas.py analyze{env_arg} --export --folder-path dummy)", file=sys.stderr)
        print("", file=sys.stderr)
    
    # Get environment variables - try from os.environ first, fallback to manual parsing
    exported_vars = []
    secret_vars_found = []
    
    # All variables to check (extended list)
    all_vars = [
        'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD',
        'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY', 'TAVILY_API_KEY',
        'OLLAMA_BASE_URL', 'DEFAULT_LLM_PROVIDER', 'DEFAULT_LLM_MODEL',
        'LANGSMITH_API_KEY', 'LANGSMITH_PROJECT', 'LANGSMITH_ENDPOINT', 'LANGSMITH_TRACING',
        'ATLAS_LOG_LEVEL', 'ATLAS_OUTPUT_DIR', 'LOCAL_NEO4J_MCP_SERVER_PATH'
    ]
    
    # First try to get from os.environ (if dotenv was loaded successfully)
    env_vars_found = {}
    for var in all_vars:
        value = os.getenv(var)
        if value:
            env_vars_found[var] = value
    
    # If no variables found in os.environ, try manual parsing as fallback
    if not env_vars_found:
        if not is_piped:
            print("# Falling back to manual .env file parsing", file=sys.stderr)
        
        # Try to find and parse .env file manually
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            manual_vars = parse_env_file_manual(env_file)
            for var in all_vars:
                if var in manual_vars:
                    env_vars_found[var] = manual_vars[var]
    
    # Export the found variables
    for var in all_vars:
        if var in env_vars_found:
            value = env_vars_found[var]
            if var in SECRET_VARS:
                # For secrets: export actual value only when piped (for shell sourcing)
                # Never display actual secret values to interactive console
                if is_piped:
                    print(f'export {var}="{value}"')
                else:
                    print(f'export {var}="[SECRET_MASKED]"')
                secret_vars_found.append(var)
                if not is_piped:
                    print(f"# {var}=[HIDDEN] (secret value masked for security)", file=sys.stderr)
            else:
                # Non-secret values: show normally
                print(f'export {var}="{value}"')
                exported_vars.append(var)
                if not is_piped:
                    print(f"# {var}={value}", file=sys.stderr)
    
    if not is_piped:
        if not exported_vars and not secret_vars_found:
            print("# No environment variables found to export", file=sys.stderr)
            print("# Make sure .env file exists and contains valid variables", file=sys.stderr)
        else:
            total_vars = len(exported_vars) + len(secret_vars_found)
            print(f"", file=sys.stderr)
            print(f"# Exported {total_vars} variables:", file=sys.stderr)
            if exported_vars:
                print(f"#   Public: {', '.join(exported_vars)}", file=sys.stderr)
            if secret_vars_found:
                print(f"#   Secret: {', '.join(secret_vars_found)} [values hidden]", file=sys.stderr)
    
    return exported_vars + secret_vars_found

def validate_provider_config(provider):
    """Validate that the required configuration exists for the provider"""
    if provider == 'openai':
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found in environment variables")
            sys.exit(1)
    elif provider == 'anthropic':
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("Error: ANTHROPIC_API_KEY not found in environment variables")
            sys.exit(1)
    elif provider == 'google':
        if not os.getenv('GEMINI_API_KEY'):
            print("Error: GEMINI_API_KEY not found in environment variables")
            sys.exit(1)
    elif provider == 'ollama':
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        print(f"Using Ollama at: {ollama_url}")
        # Note: Ollama doesn't require API keys, just the service to be running

async def analyze_command(args):
    # Check if required arguments are set
    if not args.llm_provider:
        print("❌ Error: No LLM provider specified. Use --llm-provider or set DEFAULT_LLM_PROVIDER in environment")
        sys.exit(1)
    if not args.model:
        print("❌ Error: No model specified. Use --model or set DEFAULT_LLM_MODEL in environment")
        sys.exit(1)
    
    print(f"Analyze command called with:")
    print(f"  LLM Provider: {args.llm_provider}")
    print(f"  Model: {args.model}")
    print(f"  Folder Path: {args.folder_path}")
    print(f"  File Name: {args.file_name if args.file_name else 'None (processing folder)'}")
    print(f"  Max Files: {args.max_files if args.max_files else 'No limit'}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Delete Knowledge Base: {args.delete_kb}")
    print(f"  File Type Filter: {args.file_type_filter if args.file_type_filter else 'All files (wildcard)'}")
    print(f"  Generate Knowledge Graph: {args.generate_knowledge_graph}")
    print(f"  Index Documents: {args.index_documents}")
    print(f"  Analysis Context: {args.analysis_context}")
    
    validate_provider_config(args.llm_provider)
    
    # Validate arguments - either folder_path or file_name must be provided
    if not args.folder_path and not args.file_name:
        print("Error: Either --folder-path or --file-name must be provided")
        sys.exit(1)
    
    # If file_name is provided, folder_path is not required
    if args.file_name and not args.folder_path:
        print("Note: Processing single file, --folder-path not required")
    
    if args.llm_provider == 'ollama':
        print(f"  Ollama Base URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    
    # Initialize appropriate analyzer based on analysis context
    if args.analysis_context == 'itsm':
        from itsm_data_analyzer import ITSMDataAnalyzer
        analyzer = ITSMDataAnalyzer()
        print(f"🎫 Using ITSM data analyzer")
    else:  # legacy or oracle (both use FileAnalyzer now)
        analyzer = FileAnalyzer()
        if args.analysis_context == 'oracle':
            print(f"🔧 Using Oracle Forms/PL-SQL analyzer (FileAnalyzer with Oracle schema)")
        else:
            print(f"🏗️ Using legacy application analyzer (FileAnalyzer with legacy schema)")
    
    # Handle single file processing vs directory processing
    if args.file_name:
        # Single file processing
        print("\n🔍 Starting single file analysis...")
        
        # Use file_name as the full file path
        file_path = Path(args.file_name)
        
        # Validate file exists
        if not file_path.exists():
            print(f"Error: File '{file_path}' does not exist")
            sys.exit(1)
        
        if not file_path.is_file():
            print(f"Error: Path '{file_path}' is not a file")
            sys.exit(1)
        
        try:
            # Process single file using analyzer's methods
            should_include, exclusion_reason = analyzer.should_include_file(file_path, args.file_type_filter)
            
            if not should_include:
                print(f"❌ File '{file_path.name}' excluded: {exclusion_reason}")
                sys.exit(1)
            
            # Extract metadata for the single file
            file_metadata = analyzer.extract_metadata(file_path)
            files_metadata = [file_metadata]
            
            # Create grouped files structure
            grouped_files = analyzer.group_by_type(files_metadata)
            
            # Generate summary for single file
            summary = analyzer.get_analysis_summary(files_metadata)
            
            print(f"✅ Single file analysis complete!")
            print(f"  File: {file_metadata.filename}")
            print(f"  Type: {file_metadata.file_type}")
            print(f"  Size: {file_metadata.size_bytes:,} bytes")
            print(f"  Lines: {file_metadata.line_count:,}")
            
        except Exception as e:
            print(f"❌ Error processing file '{file_path.name}': {str(e)}")
            sys.exit(1)
    else:
        # Directory processing (existing logic)
        if not args.folder_path:
            print("Error: --folder-path is required for directory analysis")
            sys.exit(1)
            
        print("\n🔍 Starting directory analysis...")
        
        # Validate folder path exists
        folder_path = Path(args.folder_path)
        if not folder_path.exists():
            print(f"Error: Folder path '{args.folder_path}' does not exist")
            sys.exit(1)
        
        if not folder_path.is_dir():
            print(f"Error: Path '{args.folder_path}' is not a directory")
            sys.exit(1)
        
        # Perform directory analysis
        try:
            files_metadata, grouped_files, summary = analyzer.analyze_directory(
                folder_path=args.folder_path,
                max_files=args.max_files,
                file_type_filter=args.file_type_filter
            )
        
            # Display results for directory processing
            print(f"\n📊 Analysis Complete!")
            
            # Processing stats table
            print("\n" + "="*60)
            print("PROCESSING STATISTICS")
            print("="*60)
            stats_table = [
                ["Files found", f"{summary['processing_stats']['files_found']:,}"],
                ["Files ignored", f"{summary['processing_stats']['files_ignored']:,}"],
                ["Files skipped (errors)", f"{summary['processing_stats']['files_skipped']:,}"],
                ["Files to be processed", f"{summary['processing_stats']['files_processed']:,}"],
                ["Processing errors", f"{summary['processing_stats']['errors']:,}"]
            ]
            
            for row in stats_table:
                print(f"{row[0]:<25} {row[1]:>15}")
            
            # Show ignored files breakdown in table format
            if summary['processing_stats']['files_ignored'] > 0:
                print(f"\n" + "-"*50)
                print("FILES NOT PROCESSED")
                print("-"*50)
                
                ignored_table = [
                    ["Unknown file types", f"{summary['processing_stats']['unknown_files_ignored']:,}"]
                ]
                
                if summary['ignored_files_breakdown']:
                    for file_type, count in sorted(summary['ignored_files_breakdown'].items()):
                        if file_type != 'unknown':  # Already shown above
                            ignored_table.append([f"Ignored ({file_type})", f"{count:,}"])
                
                for row in ignored_table:
                    print(f"{row[0]:<30} {row[1]:>15}")
            
            # Summary statistics table
            print(f"\n" + "="*60)
            print("SUMMARY STATISTICS (Processed Files)")
            print("="*60)
            summary_stats = [
                ["Total Files", f"{summary['total_files']:,}"],
                ["Total Size", f"{summary['total_size_bytes']:,} bytes"],
                ["Total Lines", f"{summary['total_lines']:,}"]
            ]
            
            for row in summary_stats:
                print(f"{row[0]:<25} {row[1]:>25}")
            
            # Files by type table
            print(f"\n" + "="*80)
            print("FILES BY TYPE (Will be Processed)")
            print("="*80)
            print(f"{'File Type':<20} {'Count':<10} {'Size (bytes)':<15} {'Lines':<15}")
            print("-"*80)
            
            for file_type, stats in sorted(summary['file_types'].items()):
                print(f"{file_type:<20} {stats['count']:<10,} {stats['total_size_bytes']:<15,} {stats['total_lines']:<15,}")
            
            # Show processing errors if any
            if analyzer.processing_errors:
                print(f"\n⚠️  Processing Errors:")
                for error in analyzer.processing_errors[:5]:  # Show first 5 errors
                    print(f"  {error}")
                if len(analyzer.processing_errors) > 5:
                    print(f"  ... and {len(analyzer.processing_errors) - 5} more errors")
            
            print(f"\n✅ File Discovery Complete! Found {len(files_metadata)} files to process.")
            
            # Display unknown file types summary if any were found
            unknown_files_summary = analyzer.get_unknown_files_summary()
            if unknown_files_summary:
                print(f"\n" + "="*100)
                print("UNKNOWN FILE TYPES NOT PROCESSED")
                print("="*100)
                print(f"Total unknown files: {analyzer.unknown_files_ignored:,}")
                print()
                
                # Sort by count descending, then by extension
                sorted_extensions = sorted(
                    unknown_files_summary.items(), 
                    key=lambda x: (-x[1]['count'], x[0])
                )
                
                print(f"{'Extension':<15} {'Count':<8} {'Size':<12} {'Example Files'}")
                print("-"*100)
                
                for extension, details in sorted_extensions:
                    # Format size in human-readable format
                    size_bytes = details['total_size_bytes']
                    if size_bytes >= 1024 * 1024 * 1024:  # GB
                        size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
                    elif size_bytes >= 1024 * 1024:  # MB
                        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                    elif size_bytes >= 1024:  # KB
                        size_str = f"{size_bytes / 1024:.1f} KB"
                    else:
                        size_str = f"{size_bytes} bytes"
                    
                    ext_display = extension if extension != 'no_extension' else '(no ext)'
                    
                    # Format example files (first 2)
                    example_files_display = ""
                    if details['example_files']:
                        examples = []
                        for example in details['example_files'][:2]:
                            try:
                                relative_path = Path(example).relative_to(Path.cwd())
                                examples.append(str(relative_path))
                            except ValueError:
                                examples.append(Path(example).name)
                        example_files_display = ", ".join(examples)
                        if len(details['example_files']) > 2:
                            example_files_display += f" (+{len(details['example_files']) - 2} more)"
                    
                    print(f"{ext_display:<15} {details['count']:<8,} {size_str:<12} {example_files_display}")
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            sys.exit(1)
    
    # Shared processing logic for both single file and directory processing
    if not files_metadata:
        print("⚠️ No files to process.")
        return
    
    # Processing workflows if requested
    kg_stats = None
    index_stats = None
    
    if args.generate_knowledge_graph and files_metadata:
        print(f"\n🧠 Starting Knowledge Graph Generation...")
        print(f"📊 Processing {len(files_metadata)} files in batches of {args.batch_size}")
        
        # Initialize LLM based on provider
        llm_model = initialize_llm(args.llm_provider, args.model)
        # Initialize Neo4j graph store
        try:
            graph_store = Neo4jGraphStore()
        except Exception as e:
            print(f"❌ Failed to initialize Neo4j graph store: {e}")
            print("💡 Please ensure Neo4j is running and connection details are correct")
            print("   Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD environment variables if needed")
            return
        
        # Capture initial state
        initial_stats = graph_store.get_stats()
        print(f"📈 Initial Graph State:")
        print(f"   Nodes: {initial_stats['nodes']}")
        print(f"   Relationships: {initial_stats['relationships']}")

        # Check if we need to delete existing knowledge base
        if args.delete_kb and initial_stats['nodes'] > 0:
            print(f"🗑️ Deleting existing knowledge base...")
            try:
                graph_store.graph.query("MATCH (n) DETACH DELETE n")
                print(f"   ✅ Knowledge base cleared")
            except Exception as e:
                print(f"   ❌ Error clearing knowledge base: {e}")
                return
        
        # Process files in batches
        total_batches = (len(files_metadata) + args.batch_size - 1) // args.batch_size
        processed_files = 0

        for batch_num in range(total_batches):
            start_idx = batch_num * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(files_metadata))
            batch = files_metadata[start_idx:end_idx]
            
            print(f"\n🔄 Processing Batch {batch_num + 1}/{total_batches}")
            print(f"   Files {start_idx + 1}-{end_idx} of {len(files_metadata)}")
            
            batch_start_stats = graph_store.get_stats()
            
            try:
                # Generate knowledge graph for batch
                if args.analysis_context == 'itsm':
                    # ITSM analyzer doesn't take analysis_context parameter
                    graph_documents = await analyzer.create_knowledge_graph(batch, llm_model, graph_store)
                else:
                    # FileAnalyzer takes analysis_context parameter
                    graph_documents = await analyzer.create_knowledge_graph(batch, llm_model, graph_store, args.analysis_context)
                
                batch_end_stats = graph_store.get_stats()
                nodes_added = batch_end_stats['nodes'] - batch_start_stats['nodes']
                relationships_added = batch_end_stats['relationships'] - batch_start_stats['relationships']
                
                print(f"   ✅ Batch {batch_num + 1} complete - {len(batch)} files processed")
                print(f"   📊 Batch Results: +{nodes_added} nodes, +{relationships_added} relationships")
                print(f"   📄 Generated {len(graph_documents)} graph documents")
                
                processed_files += len(batch)
                progress = (processed_files/len(files_metadata)*100)
                print(f"   📈 Overall Progress: {processed_files}/{len(files_metadata)} files ({progress:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Error processing batch {batch_num + 1}: {str(e)}")
                analyzer.update_failed_files_list(batch, str(e))
                # print(e)
                continue
        
        # Capture final state and show summary
        final_stats = graph_store.get_stats()
        total_nodes_added = final_stats['nodes'] - initial_stats['nodes']
        total_relationships_added = final_stats['relationships'] - initial_stats['relationships']
        
        print(f"\n🎉 Knowledge Graph Generation Complete!")
        print(f"📈 Final Graph State:")
        print(f"   Total Nodes: {final_stats['nodes']} (+{total_nodes_added})")
        print(f"   Total Relationships: {final_stats['relationships']} (+{total_relationships_added})")
        print(f"   Files Processed: {processed_files}/{len(files_metadata)}")

        failed_files = analyzer.get_failed_files_list()
        if failed_files:
            print(f"⚠️ Some files failed to process:")
            for file_metadata, error_reason in failed_files:
                print(f"  {file_metadata.path}: {error_reason}")
            print(f"   Total Failed Files: {len(failed_files)}")
        
        kg_stats = {
            'nodes_created': total_nodes_added,
            'relationships_created': total_relationships_added,
            'files_processed': processed_files
        }
    
    if args.index_documents and files_metadata:
        print(f"\n🔍 Starting Document Vector Indexing...")
        print(f"📊 Processing {len(files_metadata)} files for vector indexing")
        
        # Initialize LLM for embeddings (reuse existing one or create new)
        llm_model = initialize_llm(args.llm_provider, args.model)
        
        try:
            # Use the same graph store instance
            index_stats = await index_documents(files_metadata, args.batch_size, args.llm_provider, args.model, graph_store)
        except Exception as e:
            print(f"❌ Error during document indexing: {str(e)}")
            print("💡 Please ensure your embedding provider is properly configured")
            index_stats = {'documents_indexed': 0, 'content_chunks': 0, 'search_terms_extracted': 0}
        
        # Final completion summary
        if kg_stats or index_stats:
            print(f"\n🎯 All Processing Complete!")
            print(f"=" * 60)
            
            if kg_stats:
                print(f"🧠 Knowledge Graph:")
                print(f"   📊 {kg_stats['nodes_created']:,} nodes, {kg_stats['relationships_created']:,} relationships")
                print(f"   📁 {kg_stats['files_processed']:,} files processed")
                
                graph_store.display_summary()
            
            if index_stats:
                print(f"🔍 Search Index:")
                print(f"   📚 {index_stats['documents_indexed']:,} documents indexed")
                print(f"   🔤 {index_stats['search_terms_extracted']:,} search terms extracted")
                print(f"   📄 {index_stats['content_chunks']:,} content chunks created")
            
            print(f"   💾 Data ready for querying and analysis!")
        else:
            print(f"\n💡 Tip: Use --generate-knowledge-graph and --index-documents flags to enable processing workflows.")

async def validate_command(args):
    """Run environment validation checks."""
    print("\n🔍 Atlas Environment Validator")
    print("="*60)
    
    # Load environment configuration
    env_config = EnvironmentConfig(env=args.env, env_file=args.env_file)
    env_result = env_config.load_environment()
    
    if not env_result['success']:
        print(f"\n❌ Failed to load environment: {env_result['error']}")
        if 'attempted_files' in env_result:
            print(f"   Attempted files: {', '.join(env_result['attempted_files'])}")
        print("\n💡 Tip: Create a .env file or use --env-file to specify a custom environment file")
        sys.exit(1)
    
    # Create and run validator
    validator = EnvironmentValidator(env_config)
    
    if args.component == 'all':
        await validator.validate_all()
    else:
        await validator.validate_component(args.component)
    
    # Display report
    validator.display_validation_report()
    
    # Save report if requested
    if args.save_report:
        validator.save_validation_report(args.save_report)
    
    # Exit with appropriate code
    if validator.validation_results['summary']['failed'] > 0:
        sys.exit(1)


def chat_command(args):
    # Check if required arguments are set
    if not args.llm_provider:
        print("❌ Error: No LLM provider specified. Use --llm-provider or set DEFAULT_LLM_PROVIDER in environment")
        sys.exit(1)
    if not args.model:
        print("❌ Error: No model specified. Use --model or set DEFAULT_LLM_MODEL in environment")
        sys.exit(1)
    
    print(f"🚀 Launching Atlas Discovery Agent...")
    print(f"  LLM Provider: {args.llm_provider}")
    print(f"  Model: {args.model}")
    
    validate_provider_config(args.llm_provider)
    
    if args.llm_provider == 'ollama':
        print(f"  Ollama Base URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    
    # Prepare command to launch Streamlit app
    script_dir = Path(__file__).parent
    chat_app_path = script_dir / "chat_app.py"
    
    if not chat_app_path.exists():
        print(f"❌ Error: Chat application not found at {chat_app_path}")
        sys.exit(1)
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(chat_app_path),
        "--", 
        "--llm-provider", args.llm_provider,
        "--model", args.model
    ]
    
    # Add environment arguments if specified
    if hasattr(args, 'env') and args.env:
        cmd.extend(["--env", args.env])
    if hasattr(args, 'env_file') and args.env_file:
        cmd.extend(["--env-file", args.env_file])
    
    # Add port if specified (future enhancement)
    # if hasattr(args, 'port') and args.port:
    #     cmd.extend(["--server.port", str(args.port)])
    
    print(f"💬 Starting chat interface...")
    print(f"🌐 Opening in your default browser...")
    
    try:
        # Launch Streamlit app
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print(f"❌ Chat application exited with code {result.returncode}")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Chat session ended by user")
    except FileNotFoundError:
        print(f"❌ Error: Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching chat application: {str(e)}")
        sys.exit(1)


async def main():
    # Display banner
    display_banner()
    
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load environment configuration for all commands
    if args.command != 'validate':  # validate command handles its own env loading
        env_config = EnvironmentConfig(
            env=getattr(args, 'env', None),
            env_file=getattr(args, 'env_file', None)
        )
        
        # Load environment
        env_result = env_config.load_environment()
        if not env_result['success']:
            print(f"\n❌ Failed to load environment configuration: {env_result['error']}")
            print(f"   Attempted files: {', '.join(env_result['attempted_files'])}")
            print("\n💡 Tip: Create a .env file or use --env-file to specify a custom environment file")
            sys.exit(1)
        
        # Handle export flag - export variables and exit
        if hasattr(args, 'export') and args.export:
            try:
                export_environment_variables(env_config)
            except Exception as e:
                # Fallback to manual parsing if env_config fails
                print(f"# Warning: Environment config failed ({e}), using manual fallback", file=sys.stderr)
                export_environment_variables(None)
            return  # Exit after exporting
        
        # Apply default settings if not provided in command line
        defaults = env_config.get_default_llm_config()
        if hasattr(args, 'llm_provider'):
            if not args.llm_provider and defaults['provider']:
                args.llm_provider = defaults['provider']
                print(f"📌 Using default LLM provider from environment: {args.llm_provider}")
        if hasattr(args, 'model'):
            if not args.model and defaults['model']:
                args.model = defaults['model']
                print(f"📌 Using default model from environment: {args.model}")
        if hasattr(args, 'analysis_context'):
            if not args.analysis_context:
                args.analysis_context = env_config.get_default_analysis_context()
                print(f"📌 Using default analysis context from environment: {args.analysis_context}")
    
    # Route to appropriate command handler
    if args.command == 'analyze':
        await analyze_command(args)
    elif args.command == 'chat':
        chat_command(args)
    elif args.command == 'refine':
        from refine_knowledge_base import refine_command
        await refine_command(args)
    elif args.command == 'validate':
        await validate_command(args)
    else:
        parser.print_help()
        sys.exit(1)

def run():
    """Entry point for the atlas script"""
    import asyncio
    asyncio.run(main())

if __name__ == '__main__':
    run()
