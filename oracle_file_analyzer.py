#!/usr/bin/env python3
"""
DEPRECATED: This module is deprecated as of [current date].
Oracle analysis functionality has been consolidated into the main FileAnalyzer class.
Please use FileAnalyzer with analysis_context='oracle' instead.

Example:
    analyzer = FileAnalyzer()
    graph_documents = await analyzer.create_knowledge_graph(
        files_metadata, llm_model, graph_store, analysis_context='oracle'
    )

This file is kept for backward compatibility and will be removed in a future version.
"""

import warnings
warnings.warn(
    "oracle_file_analyzer.py is deprecated. Use FileAnalyzer with analysis_context='oracle' instead.",
    DeprecationWarning,
    stacklevel=2
)

from file_analyzer import FileAnalyzer, FileMetadata
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument


class OracleFileAnalyzer(FileAnalyzer):
    """
    Specialized file analyzer for Oracle Forms and PL/SQL applications.
    Inherits from FileAnalyzer but provides Oracle-specific analysis capabilities.
    """
    
    # Oracle-specific file type mappings (extends the base mappings)
    ORACLE_FILE_TYPE_MAPPINGS = {
        # Oracle Forms files
        '.fmb': 'oracle_form',      # Oracle Forms binary
        '.fmx': 'oracle_form',      # Oracle Forms executable
        '.fmt': 'oracle_form',      # Oracle Forms template
        '.pll': 'oracle_library',   # Oracle Forms library
        '.plx': 'oracle_library',   # Oracle Forms library executable
        '.mmb': 'oracle_menu',      # Oracle Forms menu
        '.mmx': 'oracle_menu',      # Oracle Forms menu executable
        '.olb': 'oracle_objectlib', # Oracle Forms object library
        
        # Oracle Reports files
        '.rdf': 'oracle_report',    # Oracle Reports definition
        '.rex': 'oracle_report',    # Oracle Reports executable
        '.rep': 'oracle_report',    # Oracle Reports
        
        # PL/SQL files
        '.pks': 'plsql_package',    # PL/SQL package specification
        '.pkb': 'plsql_package',    # PL/SQL package body
        '.prc': 'plsql_procedure',  # PL/SQL procedure
        '.fnc': 'plsql_function',   # PL/SQL function
        '.trg': 'plsql_trigger',    # PL/SQL trigger
        '.typ': 'plsql_type',       # PL/SQL type
        '.tps': 'plsql_type',       # PL/SQL type specification
        '.tpb': 'plsql_type',       # PL/SQL type body
        
        # Oracle SQL files
        '.sql': 'oracle_sql',       # Override base SQL mapping
        '.ddl': 'oracle_ddl',       # Oracle DDL
        '.dml': 'oracle_dml',       # Oracle DML
        '.plsql': 'oracle_plsql',   # PL/SQL code
        
        # Oracle configuration files
        '.ora': 'oracle_config',    # Oracle configuration
        '.conf': 'oracle_config',   # Oracle configuration
        '.properties': 'oracle_config', # Oracle properties
    }
    
    def __init__(self, ignore_list=None):
        """Initialize Oracle File Analyzer with Oracle-specific extensions"""
        super().__init__(ignore_list)
        # Extend the base file type mappings with Oracle-specific ones
        self.FILE_TYPE_MAPPINGS.update(self.ORACLE_FILE_TYPE_MAPPINGS)
        
        # Add Oracle file extensions to text-based extensions for line counting
        oracle_text_extensions = {
            '.pks', '.pkb', '.prc', '.fnc', '.trg', '.typ', '.tps', '.tpb',
            '.sql', '.ddl', '.dml', '.plsql', '.ora', '.conf'
        }
        self.TEXT_BASED_EXTENSIONS.update(oracle_text_extensions)
    
    def get_additional_instructions(self, file_type: str, context: str) -> str:
        """
        Get Oracle-specific additional instructions for LLMGraphTransformer
        
        Returns:
            String with Oracle-specific additional instructions
        """
        oracle_context = "Oracle Forms and PL/SQL application analysis"
        
        default_instructions = """You are a highly experienced Oracle database and application developer who understands Oracle Forms, PL/SQL, and related Oracle technologies.
You are focused on extracting Oracle-specific entities such as forms, blocks, items, triggers, packages, procedures, functions, tables, views, and their relationships.
You should also identify business domain entities and data flow patterns from Oracle applications.

Create a knowledge graph from Oracle Forms, PL/SQL code, and documentation extracting the main entities and relationships:
- Map Oracle Forms structure (forms, blocks, items, canvases, triggers) to the knowledge graph
- Extract PL/SQL packages, procedures, functions, cursors, exceptions, and their dependencies
- Identify database objects (tables, views, sequences, indexes) and their relationships
- Extract data flow patterns and business logic from triggers and PL/SQL code
- Include Oracle-specific metadata such as form properties, item properties, and database object attributes
- Identify Oracle Reports, Oracle Workflow, and other Oracle stack components
- Map database constraints, foreign keys, and referential integrity relationships
Important constraints:
- Do not create orphaned nodes with no parent relationships
- Do not create duplicate nodes or relationships
- Focus on Oracle-specific patterns and conventions
- Pay attention to PL/SQL exception handling and transaction control"""

        specific_instructions = ""
        
        # Oracle-specific file type instructions
        if file_type == 'oracle_form':
            specific_instructions = "Oracle Forms-specific instructions: Focus on extracting form blocks, items, triggers, canvases, and their relationships. Identify data block relationships to database tables and master-detail relationships between blocks."
        elif file_type == 'oracle_library':
            specific_instructions = "Oracle Forms Library-specific instructions: Focus on extracting reusable procedures, functions, and objects. Identify dependencies between libraries and forms that use them."
        elif file_type == 'oracle_menu':
            specific_instructions = "Oracle Forms Menu-specific instructions: Focus on extracting menu structure, menu items, and their associated forms or procedures. Identify security and role-based access patterns."
        elif file_type == 'oracle_report':
            specific_instructions = "Oracle Reports-specific instructions: Focus on extracting report structure, data models, queries, and formatting. Identify parameter dependencies and data source relationships."
        elif file_type == 'plsql_package':
            specific_instructions = "PL/SQL Package-specific instructions: Focus on extracting package specifications and bodies, public and private procedures/functions, cursors, exceptions, and package variables. Identify dependencies between packages."
        elif file_type == 'plsql_procedure':
            specific_instructions = "PL/SQL Procedure-specific instructions: Focus on extracting procedure parameters, local variables, cursors, exception handling, and database object dependencies."
        elif file_type == 'plsql_function':
            specific_instructions = "PL/SQL Function-specific instructions: Focus on extracting function parameters, return types, local variables, and dependencies. Pay attention to deterministic functions and purity levels."
        elif file_type == 'plsql_trigger':
            specific_instructions = "PL/SQL Trigger-specific instructions: Focus on extracting trigger events, timing, affected tables, trigger logic, and dependencies on other database objects."
        elif file_type == 'plsql_type':
            specific_instructions = "PL/SQL Type-specific instructions: Focus on extracting object types, collection types, member methods, and type dependencies."
        elif file_type == 'oracle_sql':
            specific_instructions = "Oracle SQL-specific instructions: Focus on extracting Oracle-specific SQL syntax, hints, analytic functions, hierarchical queries, and database object references."
        elif file_type == 'oracle_ddl':
            specific_instructions = "Oracle DDL-specific instructions: Focus on extracting table definitions, indexes, constraints, sequences, synonyms, and their relationships. Pay attention to Oracle-specific features like partitioning and materialized views."
        elif file_type == 'oracle_config':
            specific_instructions = "Oracle Configuration-specific instructions: Focus on extracting configuration parameters, connection strings, resource definitions, and environment settings."
        else:
            # Fall back to parent class for standard file types
            return super().get_additional_instructions(file_type, oracle_context)
        
        return f"{oracle_context}\n\n" \
               f"{specific_instructions}\n\n" \
               f"{default_instructions}\n\n"
    
    async def create_knowledge_graph(self, 
                              files_metadata: List[FileMetadata], 
                              llm_model,
                              graph_store) -> List[GraphDocument]:
        """
        Create Oracle-specific knowledge graph using LangChain LLMGraphTransformer
        
        Args:
            files_metadata: List of FileMetadata objects
            llm_model: Language model for graph transformation
            graph_store: Graph vector store for adding documents
            
        Returns:
            List of GraphDocument objects
        """
        # Use the parent class implementation but with Oracle context
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        
        content_type = files_metadata[0].file_type if files_metadata else 'unknown'
        context = "Oracle Forms and PL/SQL application analysis"

        documents = []
        for metadata in files_metadata:
            # Read file contents
            file_contents = self._read_file_contents(Path(metadata.path))
            
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
            print(f"Oracle Document: {doc.metadata['filename']}")
            print(f"File Type: {doc.metadata['file_type']}")
            print(f"Content Type: {doc.metadata['content_type']}")
            print(f"Source: {doc.metadata['source']}")
            print(f"Content Preview: {doc.page_content[:200]}...")

        # Configure LLMGraphTransformer for Oracle analysis
        transformer = LLMGraphTransformer(
            llm=llm_model,
            node_properties=True,
            relationship_properties=True,
        )
    
        # Transform to graph documents
        graph_documents = await transformer.aconvert_to_graph_documents(documents)

        # Display Oracle-specific graph document information
        for graph_doc in graph_documents:
            print(f"Oracle Graph - Nodes: {len(graph_doc.nodes)}, Relationships: {len(graph_doc.relationships)}")
            
            # Display Oracle-specific nodes
            oracle_node_types = set()
            for node in graph_doc.nodes:
                oracle_node_types.add(node.type)
                if node.type in ['OracleForm', 'PLSQLPackage', 'OracleTable', 'PLSQLProcedure', 'PLSQLFunction']:
                    print(f"Oracle Node: {node.id} ({node.type})")
            
            print(f"Oracle Node Types Found: {oracle_node_types}")

        # Add to graph store
        if graph_store:
            graph_store.add_graph_documents(graph_documents)
        
        return graph_documents