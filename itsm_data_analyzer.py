#!/usr/bin/env python3

from file_analyzer import FileAnalyzer, FileMetadata
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
import json
import csv
from datetime import datetime


class ITSMDataAnalyzer(FileAnalyzer):
    """
    Specialized data analyzer for IT Service Management (ITSM) ticket systems.
    Inherits from FileAnalyzer but provides ITSM-specific analysis capabilities.
    Can process various ITSM data formats including CSV exports, JSON dumps, and structured data.
    """
    
    # ITSM-specific file type mappings (extends the base mappings)
    ITSM_FILE_TYPE_MAPPINGS = {
        # ITSM data formats
        '.csv': 'itsm_csv',         # ITSM CSV exports
        '.json': 'itsm_json',       # ITSM JSON data
        '.jsonl': 'itsm_jsonl',     # ITSM JSON Lines
        '.xml': 'itsm_xml',         # ITSM XML exports
        '.xlsx': 'itsm_excel',      # ITSM Excel exports
        '.tsv': 'itsm_tsv',         # ITSM Tab-separated values
        
        # ITSM-specific files
        '.tickets': 'itsm_tickets', # ITSM ticket data
        '.incidents': 'itsm_incidents', # ITSM incident data
        '.problems': 'itsm_problems',   # ITSM problem data
        '.changes': 'itsm_changes',     # ITSM change data
        '.assets': 'itsm_assets',       # ITSM asset data
        '.cmdb': 'itsm_cmdb',          # ITSM CMDB data
        
        # Log files from ITSM systems
        '.log': 'itsm_log',         # Override base log mapping
        '.audit': 'itsm_audit',     # ITSM audit logs
        '.history': 'itsm_history', # ITSM history data
    }
    
    def __init__(self, ignore_list=None):
        """Initialize ITSM Data Analyzer with ITSM-specific extensions"""
        super().__init__(ignore_list)
        # Extend the base file type mappings with ITSM-specific ones
        self.FILE_TYPE_MAPPINGS.update(self.ITSM_FILE_TYPE_MAPPINGS)
        
        # Add ITSM file extensions to text-based extensions
        itsm_text_extensions = {
            '.tickets', '.incidents', '.problems', '.changes', 
            '.assets', '.cmdb', '.audit', '.history'
        }
        self.TEXT_BASED_EXTENSIONS.update(itsm_text_extensions)
    
    def get_additional_instructions(self, file_type: str, context: str) -> str:
        """
        Get ITSM-specific additional instructions for LLMGraphTransformer
        
        Returns:
            String with ITSM-specific additional instructions
        """
        itsm_context = "IT Service Management ticket analysis"
        
        default_instructions = """You are a highly experienced IT Service Management professional who understands ITSM processes, ticket systems, and service operations.
You are focused on extracting ITSM entities such as incidents, problems, changes, assets, services, users, and their relationships.
You should also identify process flows, escalation patterns, and service dependencies from ITSM data.

Create a knowledge graph from ITSM tickets and documentation extracting the main entities and relationships:
- Map incident tickets to problems, changes, and root causes
- Extract service relationships, dependencies, and impact analysis
- Identify user roles, assignment groups, and escalation paths
- Extract temporal patterns in ticket creation, resolution, and closure
- Include ITSM metadata such as priority, severity, category, and status
- Identify configuration items (CIs) and service catalog relationships
- Map SLA compliance, resolution times, and performance metrics
- Extract knowledge base articles and their relationships to tickets
Important constraints:
- Do not create orphaned nodes with no parent relationships
- Do not create duplicate nodes or relationships
- Focus on ITSM process flows and service relationships
- Pay attention to temporal aspects and trend analysis
- Maintain data privacy and do not expose sensitive user information"""

        specific_instructions = ""
        
        # ITSM-specific file type instructions
        if file_type == 'itsm_csv':
            specific_instructions = "ITSM CSV-specific instructions: Focus on extracting ticket records, identifying columns for ticket ID, status, priority, category, assignment group, and resolution details. Map relationships between tickets and users, services, and CIs."
        elif file_type == 'itsm_json':
            specific_instructions = "ITSM JSON-specific instructions: Focus on extracting nested ticket data structures, API responses, and complex relationships. Identify service hierarchies, user profiles, and configuration item relationships."
        elif file_type == 'itsm_jsonl':
            specific_instructions = "ITSM JSON Lines-specific instructions: Focus on processing streaming ticket data, time-series analysis, and bulk import/export formats. Identify patterns across multiple tickets and temporal relationships."
        elif file_type == 'itsm_tickets':
            specific_instructions = "ITSM Tickets-specific instructions: Focus on extracting ticket lifecycle, status transitions, assignment history, and resolution paths. Identify patterns in ticket escalation and closure."
        elif file_type == 'itsm_incidents':
            specific_instructions = "ITSM Incidents-specific instructions: Focus on extracting incident details, impact and urgency classifications, affected services, and resolution steps. Map relationships to problems and changes."
        elif file_type == 'itsm_problems':
            specific_instructions = "ITSM Problems-specific instructions: Focus on extracting root cause analysis, related incidents, workarounds, and known error records. Identify patterns in problem resolution."
        elif file_type == 'itsm_changes':
            specific_instructions = "ITSM Changes-specific instructions: Focus on extracting change requests, approval workflows, implementation plans, and risk assessments. Map relationships to incidents and problems."
        elif file_type == 'itsm_assets':
            specific_instructions = "ITSM Assets-specific instructions: Focus on extracting configuration items, asset relationships, dependencies, and impact analysis. Identify service dependencies and asset lifecycle management."
        elif file_type == 'itsm_cmdb':
            specific_instructions = "ITSM CMDB-specific instructions: Focus on extracting configuration management data, CI relationships, service maps, and dependency chains. Identify business services and technical components."
        elif file_type == 'itsm_log':
            specific_instructions = "ITSM Log-specific instructions: Focus on extracting system events, user actions, workflow transitions, and performance metrics. Identify patterns in system usage and potential issues."
        elif file_type == 'itsm_audit':
            specific_instructions = "ITSM Audit-specific instructions: Focus on extracting compliance data, access logs, change history, and security events. Identify audit trails and compliance patterns."
        elif file_type == 'itsm_history':
            specific_instructions = "ITSM History-specific instructions: Focus on extracting historical trends, ticket volume patterns, resolution time analysis, and performance metrics over time."
        else:
            # Fall back to parent class for standard file types
            return super().get_additional_instructions(file_type, itsm_context)
        
        return f"{itsm_context}\n\n" \
               f"{specific_instructions}\n\n" \
               f"{default_instructions}\n\n"
    
    def _parse_itsm_data(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """
        Parse ITSM data from various formats into structured records
        
        Args:
            file_path: Path to the data file
            content: Raw file content
            
        Returns:
            List of structured ITSM records
        """
        file_type = self.get_file_type(file_path)
        records = []
        
        try:
            if file_type == 'itsm_csv' or file_path.suffix.lower() == '.csv':
                # Parse CSV data
                import io
                csv_reader = csv.DictReader(io.StringIO(content))
                records = list(csv_reader)
                
            elif file_type == 'itsm_json' or file_path.suffix.lower() == '.json':
                # Parse JSON data
                data = json.loads(content)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    # Handle various JSON structures
                    if 'tickets' in data:
                        records = data['tickets']
                    elif 'incidents' in data:
                        records = data['incidents']
                    elif 'result' in data:
                        records = data['result'] if isinstance(data['result'], list) else [data['result']]
                    else:
                        records = [data]
                        
            elif file_type == 'itsm_jsonl' or file_path.suffix.lower() == '.jsonl':
                # Parse JSON Lines data
                for line in content.strip().split('\n'):
                    if line.strip():
                        records.append(json.loads(line))
                        
            else:
                # For other formats, treat as plain text and create a single record
                records = [{
                    'content': content,
                    'source_file': str(file_path),
                    'file_type': file_type
                }]
                
        except Exception as e:
            print(f"⚠️ Error parsing ITSM data from {file_path}: {e}")
            # Fallback to plain text processing
            records = [{
                'content': content,
                'source_file': str(file_path),
                'file_type': file_type,
                'parse_error': str(e)
            }]
        
        return records
    
    async def create_knowledge_graph(self, 
                              files_metadata: List[FileMetadata], 
                              llm_model,
                              graph_store) -> List[GraphDocument]:
        """
        Create ITSM-specific knowledge graph using LangChain LLMGraphTransformer
        
        Args:
            files_metadata: List of FileMetadata objects
            llm_model: Language model for graph transformation
            graph_store: Graph vector store for adding documents
            
        Returns:
            List of GraphDocument objects
        """
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        
        content_type = files_metadata[0].file_type if files_metadata else 'unknown'
        context = "IT Service Management ticket analysis"

        documents = []
        for metadata in files_metadata:
            # Read file contents
            file_contents = self._read_file_contents(Path(metadata.path))
            
            # Parse ITSM data if it's a structured format
            if metadata.file_type.startswith('itsm_'):
                parsed_records = self._parse_itsm_data(Path(metadata.path), file_contents)
                
                # Create separate documents for each parsed record if we have many records
                if len(parsed_records) > 1 and len(parsed_records) < 1000:  # Reasonable limit
                    for i, record in enumerate(parsed_records):
                        doc_content = json.dumps(record, indent=2) if isinstance(record, dict) else str(record)
                        doc = Document(
                            page_content=doc_content,
                            metadata={
                                "filename": metadata.filename,
                                "file_type": metadata.file_type,
                                "content_type": self.get_file_type(Path(metadata.path)),
                                "source": str(Path(metadata.path)),
                                "source_uri": metadata.file_source_uri,
                                "record_index": i,
                                "total_records": len(parsed_records),
                                "size": metadata.size_bytes,
                                "modified_date": metadata.modified_date.isoformat(),
                            }
                        )
                        documents.append(doc)
                else:
                    # For single records or very large datasets, use original content
                    doc = Document(
                        page_content=file_contents,
                        metadata={
                            "filename": metadata.filename,
                            "file_type": metadata.file_type,
                            "content_type": self.get_file_type(Path(metadata.path)),
                            "source": str(Path(metadata.path)),
                            "source_uri": metadata.file_source_uri,
                            "total_records": len(parsed_records),
                            "size": metadata.size_bytes,
                            "line_count": metadata.line_count,
                            "modified_date": metadata.modified_date.isoformat(),
                        }
                    )
                    documents.append(doc)
            else:
                # Standard document processing for non-ITSM specific files
                doc = Document(
                    page_content=file_contents,
                    metadata={
                        "filename": metadata.filename,
                        "file_type": metadata.file_type,
                        "content_type": self.get_file_type(Path(metadata.path)),
                        "source": str(Path(metadata.path)),
                        "source_uri": metadata.file_source_uri,
                        "size": metadata.size_bytes,
                        "line_count": metadata.line_count,
                        "modified_date": metadata.modified_date.isoformat(),
                    }
                )
                documents.append(doc)

        # Display document information
        for doc in documents:
            print(f"ITSM Document: {doc.metadata['filename']}")
            print(f"File Type: {doc.metadata['file_type']}")
            print(f"Content Type: {doc.metadata['content_type']}")
            if 'total_records' in doc.metadata:
                print(f"Total Records: {doc.metadata['total_records']}")
            print(f"Content Preview: {doc.page_content[:200]}...")

        # Configure LLMGraphTransformer for ITSM analysis
        transformer = LLMGraphTransformer(
            llm=llm_model,
            node_properties=True,
            relationship_properties=True,
        )
    
        # Transform to graph documents
        graph_documents = await transformer.aconvert_to_graph_documents(documents)

        # Display ITSM-specific graph document information
        for graph_doc in graph_documents:
            print(f"ITSM Graph - Nodes: {len(graph_doc.nodes)}, Relationships: {len(graph_doc.relationships)}")
            
            # Display ITSM-specific nodes
            itsm_node_types = set()
            for node in graph_doc.nodes:
                itsm_node_types.add(node.type)
                if node.type in ['Incident', 'Problem', 'Change', 'Asset', 'Service', 'User', 'ConfigurationItem']:
                    print(f"ITSM Node: {node.id} ({node.type})")
            
            print(f"ITSM Node Types Found: {itsm_node_types}")

        # Add to graph store
        if graph_store:
            graph_store.add_graph_documents(graph_documents)
        
        return graph_documents