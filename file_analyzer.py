#!/usr/bin/env python3

import os
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2
import yaml
from pathlib import Path as PathLib


@dataclass
class FileMetadata:
    """Metadata for a single file"""
    filename: str
    extension: str
    file_type: str
    modified_date: datetime
    size_bytes: int
    line_count: int
    file_source_uri: str
    path: str


class FileAnalyzer:
    """
    Analyzes files in a directory structure, extracts metadata,
    and groups files by type with optional filtering.
    """
    # Retain a list of Documents that failed to be processed that can be retried later, the list is a list of FileMetadata objects and the error reason
    failed_files_list : List[tuple[FileMetadata, str]] = []

    # Helper function to update the failed files list, takes a list of FileMetadata objects
    
    def update_failed_files_list(self, failed_batch: List[FileMetadata], reason: Optional[str] = None):
        """
        Update the list of failed files that can be retried later
        Args:
            failed_batch: List of FileMetadata objects that failed processing
            reason: Optional reason for failure, can be logged or used later
        """
        for file_metadata in failed_batch:
            self.failed_files_list.append((file_metadata, reason if reason else "Unknown error"))

    def get_failed_files_list(self) -> List[FileMetadata]:
        """
        Returns the list of failed files that can be retried later
        """
        return self.failed_files_list
    

    # Default ignore list - files/directories to skip
    DEFAULT_IGNORE_LIST = {
        # Version control
        '.git', '.svn', '.hg', '.bzr',
        # IDE and editor files
        '.idea', '.vscode', '.eclipse', '__pycache__', '.pytest_cache',
        # Build/dependency directories
        'node_modules', 'target', 'build', 'dist', '.gradle', 'bin',
        # OS files
        '.DS_Store', 'Thumbs.db', 'Desktop.ini',
        # Log files
        '*.log', '*.tmp', '*.temp',
        # Backup files
        '*.bak', '*.backup', '*.swp', '*.swo',
        # Archive files
        '*.zip', '*.tar', '*.gz', '*.rar', '*.7z',
        # Binary/executable files
        '*.exe', '*.dll', '*.so', '*.dylib', '*.class',
        # Image files (typically not analyzed for code)
        '*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.ico', '*.svg',
        # Media files
        '*.mp4', '*.avi', '*.mov', '*.mp3', '*.wav',
        # Claude-specific files
        'CLAUDE.md',
    }
    
    # Comprehensive file type mapping
    FILE_TYPE_MAPPINGS = {
        # Programming Languages
        '.py': 'python',
        '.pyx': 'python',
        '.pyi': 'python',
        '.java': 'java',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.mjs': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.scala': 'scala',
        '.sc': 'scala',
        '.go': 'golang',
        '.cpp': 'c++',
        '.cc': 'cpp',
        '.cxx': 'c++',
        '.c++': 'c++',
        '.hpp': 'cpp',
        '.hxx': 'cpp',
        '.h++': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.cs': 'csharp',
        '.csx': 'csharp',
        '.rb': 'ruby',
        '.rbw': 'ruby',
        '.rake': 'ruby',
        '.gemspec': 'ruby',
        
        # Shell Scripts
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.fish': 'shell',
        '.ksh': 'ksh',
        '.bat': 'bat',
        '.cmd': 'bat',
        '.ps1': 'powershell',
        '.psm1': 'powershell',
        '.psd1': 'powershell',
        
        # Markup and Documentation
        '.md': 'markdown',
        '.markdown': 'markdown',
        '.rst': 'markdown',
        '.txt': 'text',
        '.rtf': 'text',
        
        # Database and Query
        '.sql': 'sql',
        '.ddl': 'sql',
        '.dml': 'sql',
        '.plsql': 'sql',
        
        # Mainframe/Legacy
        '.bms': 'bms',
        '.bmc': 'bmc',
        '.csd': 'csd',
        '.cbl': 'cobol',
        '.cob': 'cobol',
        '.cobol': 'cobol',
        '.cpy': 'copybook',
        '.copy': 'copybook',
        '.jcl': 'jcl',
        '.prc': 'prc',
        '.proc': 'prc',
        
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
        '.fnc': 'plsql_function',   # PL/SQL function
        '.trg': 'plsql_trigger',    # PL/SQL trigger
        '.typ': 'plsql_type',       # PL/SQL type
        '.tps': 'plsql_type',       # PL/SQL type specification
        '.tpb': 'plsql_type',       # PL/SQL type body
        
        # Oracle configuration files
        '.ora': 'oracle_config',    # Oracle configuration
        
        # Oracle Forms APEX conversion files
        '.pld': 'plsql_library_text',  # PL/SQL library text format (converted from .pll)
        
        # Scripting
        '.pl': 'perl',
        '.pm': 'perl',
        '.perl': 'perl',
        
        # Office Documents
        '.pdf': 'pdf',
        '.doc': 'word',
        '.docx': 'word',
        '.xls': 'excel',
        '.xlsx': 'excel',
        '.ppt': 'powerpoint',
        '.pptx': 'powerpoint',
        
        # Data Formats
        '.csv': 'csv',
        '.tsv': 'csv',
        '.json': 'json',
        '.jsonl': 'json',
        '.xml': 'xml',
        '.xsd': 'xml',
        '.xsl': 'xml',
        '.xslt': 'xml',
        '.html': 'html',
        '.htm': 'html',
        '.xhtml': 'html',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        
        # Configuration
        '.ini': 'text',
        '.cfg': 'text',
        '.conf': 'text',
        '.config': 'text',
        '.properties': 'text',
        '.toml': 'text',
        '.log': 'log',
        '.zip': 'zip',
        '.tar': 'tar',
        '.gz': 'gzip',
    }
    
    # Text-based file extensions that can be line-counted
    TEXT_BASED_EXTENSIONS = {
        '.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.scala', '.go', 
        '.cpp', '.cc', '.cxx', '.c++', '.hpp', '.hxx', '.h++', '.c', '.h',
        '.cs', '.csx', '.rb', '.rbw', '.rake', '.gemspec', '.sh', '.bash',
        '.zsh', '.fish', '.ksh', '.bat', '.cmd', '.ps1', '.psm1', '.psd1',
        '.md', '.markdown', '.rst', '.txt', '.rtf', '.sql', '.ddl', '.dml',
        '.plsql', '.bms', '.bmc', '.csd', '.cbl', '.cob', '.cobol', '.cpy',
        '.copy', '.jcl', '.prc', '.proc', '.pl', '.pm', '.perl', '.csv',
        '.tsv', '.json', '.jsonl', '.xml', '.xsd', '.xsl', '.xslt', '.html',
        '.htm', '.xhtml', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.config',
        '.properties', '.toml', '.pks', '.pkb', '.fnc', '.trg', '.typ', '.tps', '.tpb', '.ora', '.pld'
    }
    
    def __init__(self, ignore_list: Optional[set[str]] = None):
        """
        Initialize FileAnalyzer
        
        Args:
            ignore_list: Optional set of patterns/names to ignore (extends default)
        """
        self.ignore_list = self.DEFAULT_IGNORE_LIST.copy()
        if ignore_list:
            self.ignore_list.update(ignore_list)
        self.reset_stats()
    
    def reset_stats(self):
        """Reset analysis statistics"""
        self.total_files_found = 0
        self.files_processed = 0
        self.files_skipped = 0
        self.files_ignored = 0
        self.unknown_files_ignored = 0
        self.ignored_files_by_type = defaultdict(int)
        self.processing_errors = []
    
    def load_graph_schema(self, analysis_context: str = "legacy") -> Dict[str, List[str]]:
        """
        Load graph schema configuration from YAML file
        
        Args:
            analysis_context: The analysis context (legacy, oracle, etc.)
            
        Returns:
            Dictionary with 'allowed_nodes' and 'allowed_relationships' lists
        """
        try:
            # Construct path to schema file
            current_dir = PathLib(__file__).parent
            schema_file = current_dir / "config" / "graph_schemas" / f"{analysis_context}.yaml"
            
            if not schema_file.exists():
                print(f"⚠️ Schema file not found: {schema_file}")
                print(f"   Falling back to default (no restrictions)")
                return {"allowed_nodes": [], "allowed_relationships": []}
            
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_config = yaml.safe_load(f)
            
            # Validate schema structure
            if not isinstance(schema_config, dict):
                print(f"⚠️ Invalid schema format in {schema_file}")
                return {"allowed_nodes": [], "allowed_relationships": []}
            
            allowed_nodes = schema_config.get('allowed_nodes', [])
            allowed_relationships = schema_config.get('allowed_relationships', [])
            
            print(f"📋 Loaded schema for '{analysis_context}' context:")
            print(f"   Allowed nodes: {len(allowed_nodes)}")
            print(f"   Allowed relationships: {len(allowed_relationships)}")
            
            return {
                "allowed_nodes": allowed_nodes,
                "allowed_relationships": allowed_relationships
            }
            
        except Exception as e:
            print(f"⚠️ Error loading schema for '{analysis_context}': {e}")
            return {"allowed_nodes": [], "allowed_relationships": []}
    
    def get_file_type(self, file_path: Path) -> str:
        """
        Determine file type based on extension and Oracle APEX conversion suffixes
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type string
        """
        filename = file_path.name.lower()
        
        # Check for Oracle APEX conversion suffixes first
        oracle_conversion_suffixes = {
            '_fmb.xml': 'oracle_form_xml',      # Converted Oracle Forms XML
            '_olb.xml': 'oracle_library_xml',   # Converted Object Library XML  
            '_mmb.xml': 'oracle_menu_xml',      # Converted Menu Module XML
        }
        
        for suffix, file_type in oracle_conversion_suffixes.items():
            if filename.endswith(suffix):
                return file_type
        
        # Fall back to extension-based detection
        extension = file_path.suffix.lower()
        return self.FILE_TYPE_MAPPINGS.get(extension, 'unknown')
    
    def count_lines(self, file_path: Path) -> int:
        """
        Count lines in a text file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of lines in the file
        """
        # Use get_file_type to handle both extensions and suffixes
        file_type = self.get_file_type(file_path)
        
        # Check if this file type should be line-counted
        oracle_conversion_text_types = {
            'oracle_form_xml', 'oracle_library_xml', 'oracle_menu_xml', 'plsql_library_text'
        }
        
        is_text_based = (
            file_path.suffix.lower() in self.TEXT_BASED_EXTENSIONS or
            file_type in oracle_conversion_text_types
        )
        
        if not is_text_based:
            return 0
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except (OSError, UnicodeDecodeError):
            try:
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    return sum(1 for _ in f)
            except OSError:
                return 0
    
    def extract_metadata(self, file_path: Path) -> FileMetadata:
        """
        Extract comprehensive metadata from a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileMetadata object with all relevant information
        """
        stat = file_path.stat()
        
        return FileMetadata(
            filename=file_path.name,
            extension=file_path.suffix.lower(),
            file_type=self.get_file_type(file_path),
            modified_date=datetime.fromtimestamp(stat.st_mtime),
            size_bytes=stat.st_size,
            line_count=self.count_lines(file_path),
            file_source_uri=file_path.absolute().as_uri(),
            path=str(file_path.absolute())
        )
    
    def is_ignored(self, file_path: Path) -> tuple[bool, str]:
        """
        Check if a file/directory should be ignored based on ignore list
        
        Args:
            file_path: Path to check
            
        Returns:
            Tuple of (should_ignore, reason)
        """
        file_name = file_path.name
        
        # Check against ignore patterns
        for pattern in self.ignore_list:
            if pattern.startswith('*'):
                # Wildcard pattern (e.g., *.log)
                if file_name.endswith(pattern[1:]):
                    return True, f"Ignoring: '{pattern}'"
            elif pattern.startswith('.') and len(pattern) > 1:
                # Extension or hidden directory (e.g., .git, .py)
                if file_name == pattern or file_name.endswith(pattern):
                    return True, f"Ignoring: '{pattern}'"
            else:
                # Exact match (e.g., node_modules)
                if file_name == pattern:
                    return True, f"Ignoring: '{pattern}'"
        
        return False, ""
    
    def should_include_file(self, file_path: Path, file_type_filter: Optional[str] = None) -> tuple[bool, str]:
        """
        Check if a file should be included based on filters
        
        Args:
            file_path: Path to the file
            file_type_filter: Optional file type filter (e.g., '.py', 'python')
            
        Returns:
            Tuple of (should_include, reason_if_excluded)
        """
        if not file_path.is_file():
            return False, "not a file"
        
        # Check ignore list
        is_ignored, ignore_reason = self.is_ignored(file_path)
        if is_ignored:
            return False, ignore_reason
        
        # Get file type
        file_type = self.get_file_type(file_path)
        
        # Skip unknown file types
        if file_type == 'unknown':
            return False, "unknown file type"
        
        # Apply file type filter if provided
        if file_type_filter:
            # Normalize filter to lower case for case-insensitive comparison, handle comma separated list of extensions
            if ',' in file_type_filter:
                # Split by comma and strip whitespace
                filters = [f.strip().lower() for f in file_type_filter.split(',')]
                if not any(file_type.lower() == f or file_path.suffix.lower() == f or file_path.suffix.lower().lstrip('.') == f for f in filters):
                    return False, f"doesn't match filter '{file_type_filter}'"
            else:
                # Single filter case
                file_type_filter = file_type_filter.strip().lower()
                if file_type_filter.startswith('.'):
                    # Handle extension filter (e.g., '.py')
                    if file_path.suffix.lower() != file_type_filter.lower():
                        return False, f"doesn't match filter '{file_type_filter}'"
                else:
                    # Handle file type filter (e.g., 'python')
                    if file_type_filter.lower() != file_type.lower():
                        # Also check against extension
                        if not (file_type_filter.lower() == file_path.suffix.lower() or 
                                file_type_filter.lower() == file_path.suffix.lower().lstrip('.')):
                            return False, f"doesn't match filter '{file_type_filter}'"
        # If all checks passed, include the file
        
        return True, ""
    
    def traverse_directory(self, 
                          folder_path: Path, 
                          max_files: Optional[int] = None,
                          file_type_filter: Optional[str] = None) -> list[FileMetadata]:
        """
        Traverse directory structure and collect file metadata
        
        Args:
            folder_path: Root directory to traverse
            max_files: Maximum number of files to process
            file_type_filter: Optional file type filter
            
        Returns:
            List of FileMetadata objects
        """
        self.reset_stats()
        files_metadata = []
        
        try:
            # Use rglob for recursive traversal
            for file_path in folder_path.rglob('*'):
                # Skip directories that are in ignore list
                if file_path.is_dir():
                    is_ignored, _ = self.is_ignored(file_path)
                    if is_ignored:
                        continue
                
                if not file_path.is_file():
                    continue
                
                self.total_files_found += 1
                
                should_include, exclusion_reason = self.should_include_file(file_path, file_type_filter)
                
                if not should_include:
                    self.files_ignored += 1
                    
                    # Track specific exclusion reasons
                    if exclusion_reason == "unknown file type":
                        self.unknown_files_ignored += 1
                        file_type = "unknown"
                    else:
                        file_type = self.get_file_type(file_path)
                    
                    self.ignored_files_by_type[file_type] += 1
                    continue
                
                # Check max files limit
                if max_files and len(files_metadata) >= max_files:
                    break
                
                try:
                    metadata = self.extract_metadata(file_path)
                    files_metadata.append(metadata)
                    self.files_processed += 1
                    
                except Exception as e:
                    self.files_skipped += 1
                    self.processing_errors.append(f"Error processing {file_path}: {str(e)}")
                    e.with_traceback(None)  # Clear traceback to avoid memory issues
                    
                    
        except Exception as e:
            self.processing_errors.append(f"Error traversing directory {folder_path}: {str(e)}")
        
        return files_metadata
    
    def group_by_type(self, files_metadata: list[FileMetadata]) -> dict[str, list[FileMetadata]]:
        """
        Group files by their type
        
        Args:
            files_metadata: List of FileMetadata objects
            
        Returns:
            Dictionary with file types as keys and lists of FileMetadata as values
        """
        grouped = defaultdict(list)
        for metadata in files_metadata:
            grouped[metadata.file_type].append(metadata)
        
        return dict(grouped)
    
    def get_analysis_summary(self, files_metadata: list[FileMetadata]) -> dict:
        """
        Generate analysis summary statistics
        
        Args:
            files_metadata: List of FileMetadata objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not files_metadata:
            return {
                'total_files': 0,
                'total_size_bytes': 0,
                'total_lines': 0,
                'file_types': {},
                'processing_stats': {
                    'files_found': self.total_files_found,
                    'files_processed': self.files_processed,
                    'files_skipped': self.files_skipped,
                    'files_ignored': self.files_ignored,
                    'unknown_files_ignored': self.unknown_files_ignored,
                    'errors': len(self.processing_errors)
                },
                'ignored_files_breakdown': dict(self.ignored_files_by_type)
            }
        
        total_size = sum(f.size_bytes for f in files_metadata)
        total_lines = sum(f.line_count for f in files_metadata)
        
        # Count by file type
        type_counts = defaultdict(int)
        type_sizes = defaultdict(int)
        type_lines = defaultdict(int)
        
        for metadata in files_metadata:
            type_counts[metadata.file_type] += 1
            type_sizes[metadata.file_type] += metadata.size_bytes
            type_lines[metadata.file_type] += metadata.line_count
        
        file_types = {}
        for file_type in type_counts:
            file_types[file_type] = {
                'count': type_counts[file_type],
                'total_size_bytes': type_sizes[file_type],
                'total_lines': type_lines[file_type]
            }
        
        return {
            'total_files': len(files_metadata),
            'total_size_bytes': total_size,
            'total_lines': total_lines,
            'file_types': file_types,
            'processing_stats': {
                'files_found': self.total_files_found,
                'files_processed': self.files_processed,
                'files_skipped': self.files_skipped,
                'files_ignored': self.files_ignored,
                'unknown_files_ignored': self.unknown_files_ignored,
                'errors': len(self.processing_errors)
            },
            'ignored_files_breakdown': dict(self.ignored_files_by_type)
        }
    
    def analyze_directory(self, 
                         folder_path: str,
                         max_files: Optional[int] = None,
                         file_type_filter: Optional[str] = None) -> tuple[list[FileMetadata], dict[str, list[FileMetadata]], dict]:
        """
        Complete directory analysis workflow
        
        Args:
            folder_path: Path to directory to analyze
            max_files: Maximum number of files to process
            file_type_filter: Optional file type filter
            
        Returns:
            Tuple of (all_files_metadata, grouped_by_type, analysis_summary)
        """
        path = Path(folder_path)
        
        # Traverse and collect metadata
        files_metadata = self.traverse_directory(path, max_files, file_type_filter)
        
        # Group by type
        grouped_files = self.group_by_type(files_metadata)
        
        # Generate summary
        summary = self.get_analysis_summary(files_metadata)
        
        return files_metadata, grouped_files, summary
    
    # Return a set of additional instructions for the LLMGraphTransformer using the file type extensions and a given context
    def get_additional_instructions(self, file_type: str, context: str, analysis_context: str = "legacy") -> str:
        """
        Get additional instructions for LLMGraphTransformer based on context
        
        Returns:
            String with additional instructions
        """
        default_instructions =  """You are a highly experienced software engineering professional who understands legacy code including mainframe environments.
including source code, scripts and all forms of documentation, that describe behaviour. You are focused on extracting technical entities such as systems, applications, programs, modules, libraries etc.
You should also identify business domain entities from the documents you encounter.

Create a knowledge graph from source code and documentation extracting the main entities and relationships:
- Map the structure and semantics of the code to the entities and relationships in the knowledge graph
- Extract software functional behaviour and dependencies from module headers and comments
- For other document types such as markdown or text files, extract the main entities and relationships from the text content
- Include the file metadata in the graph properities nodes, such as filename, file type, size, modified date, and line count, complexity
- Identify and extract business domain entities from the text content, such as customer, product, order
- Identify and extract technical entities such as systems, applications, programs, modules, libraries, and scripts
Important constraints:
- Do not create orphaned nodes with no parent relationships
- Do not create duplicate nodes or relationships
- Do not create nodes for code variables or code literals"""

        # Handle Oracle context with different default instructions
        if analysis_context == "oracle":
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
            context = oracle_context

        specific_instructions = ""
        # for each file type in the FILE_TYPE_MAPPINGS, eg python, shell scripts, sql return a different set of instructions
        if file_type == 'python':
            specific_instructions = f"Python-specific instructions: Focus on extracting classes, functions, methods, and their relationships. Identify imports and dependencies between modules."
        elif file_type == 'javascript':
            specific_instructions = f"JavaScript-specific instructions: Focus on extracting functions, classes, and their relationships. Identify imports and dependencies between modules. Pay attention to asynchronous patterns."
        elif file_type == 'typescript':
            specific_instructions = f"TypeScript-specific instructions: Focus on extracting interfaces, types, classes, and their relationships. Identify imports and dependencies between modules. Pay attention to type annotations."
        elif file_type == 'java':
            specific_instructions = f"Java-specific instructions: Focus on extracting classes, interfaces, methods, and their relationships. Identify imports and dependencies between packages. Pay attention to annotations."
        elif file_type == 'c++':
            specific_instructions = f"C++-specific instructions: Focus on extracting classes, functions, templates, and their relationships. Identify includes and dependencies between headers. Pay attention to namespaces."
        elif file_type == 'c#':
            specific_instructions = f"C# specific instructions: Focus on extracting classes, interfaces, methods, and their relationships. Identify namespaces and dependencies between assemblies. Pay attention to attributes."
        elif file_type == 'ruby':           
            specific_instructions = f"Ruby-specific instructions: Focus on extracting classes, modules, methods, and their relationships. Identify requires and dependencies between files. Pay attention to metaprogramming patterns." 
        elif file_type == 'shell':
            specific_instructions = f"Shell script-specific instructions: Focus on extracting functions, variables, and their relationships. Identify sourced files and dependencies between scripts. Pay attention to environment variables."
        elif file_type == 'sql':
            specific_instructions = f"SQL-specific instructions: Focus on extracting tables, views, stored procedures, and their relationships. Identify foreign keys and dependencies between tables. Pay attention to query patterns."
        elif file_type == 'markdown':
            specific_instructions = f"Markdown-specific instructions: Focus on extracting headings, links, images, and their relationships. Identify sections and dependencies between documents. Pay attention to formatting patterns."
        elif file_type == 'text':
            specific_instructions = f"Text-specific instructions: Focus on extracting key entities, relationships, and patterns from the text content. Identify sections and dependencies between documents. Pay attention to semantic meaning."
        elif file_type == 'pdf':
            specific_instructions = f"PDF-specific instructions: Focus on extracting text content, headings, links, and their relationships. Identify sections and dependencies between pages. Pay attention to formatting patterns."
        elif file_type == 'csv':
            specific_instructions = f"CSV-specific instructions: Focus on extracting columns, rows, and their relationships. Identify headers and dependencies between data fields. Pay attention to data types."
        elif file_type == 'xml':
            specific_instructions = f"XML-specific instructions: Focus on extracting elements, attributes, and their relationships. Identify dependencies between documents. Pay attention to schema definitions."
        elif file_type == 'yaml':
            specific_instructions = f"YAML-specific instructions: Focus on extracting keys, values, and their relationships. Identify dependencies between documents. Pay attention to data structures."
        elif file_type == 'json':
            specific_instructions = f"JSON-specific instructions: Focus on extracting keys, values, and their relationships. Identify dependencies between objects. Pay attention to data structures."
        elif file_type == 'bms':
            specific_instructions = f"BMS-specific instructions: Focus on extracting screen definitions, fields, and their relationships. Identify dependencies between maps. Pay attention to CICS conventions."
        elif file_type == 'bmc':
            specific_instructions = f"BMC-specific instructions: Focus on extracting screen definitions, fields, and their relationships. Identify dependencies between maps. Pay attention to CICS conventions."
        elif file_type == 'csd':
            specific_instructions = """
            CSD-specific instructions: Focus on extracting CICS resources and their relationships, 
            for example FILE, PROGRAM and TRANSACTION definitions, using the DEFINE statement. 
            Identify dependencies between CICS components. 
            Pay attention to CICS conventions."""
        elif file_type == 'cobol':
            specific_instructions = f"COBOL-specific instructions: Focus on extracting programs, sections, paragraphs, and their relationships. Identify data divisions and dependencies between files. Pay attention to COBOL conventions."
        elif file_type == 'copybook':
            specific_instructions = f"Copybook-specific instructions: Focus on extracting data structures, fields, and their relationships. Identify dependencies between copybooks and programs. Pay attention to COBOL conventions."
        elif file_type == 'jcl':
            specific_instructions = f"JCL-specific instructions: Focus on extracting jobs, steps, and their relationships. Identify dependencies between jobs and resources. Pay attention to JCL conventions."
        elif file_type == 'prc':
            specific_instructions = f"PRC-specific instructions: Focus on extracting procedures, parameters, and their relationships. Identify dependencies between procedures and programs. Pay attention to COBOL conventions."
        elif file_type == 'prc':
            specific_instructions = f"PRC-specific instructions: Focus on extracting procedures, parameters, and their relationships. Identify dependencies between procedures and programs. Pay attention to COBOL conventions."
        elif file_type == 'perl':
            specific_instructions = f"Perl-specific instructions: Focus on extracting packages, subroutines, and their relationships. Identify dependencies between modules. Pay attention to Perl conventions."
        elif file_type == 'zip':    
            specific_instructions = f"ZIP-specific instructions: Focus on extracting files, directories, and their relationships. Identify dependencies between compressed files. Pay attention to archive structures."
        elif file_type == 'tar':
            specific_instructions = f"TAR-specific instructions: Focus on extracting files, directories, and their relationships. Identify dependencies between archived files. Pay attention to archive structures."
        elif file_type == 'gzip':
            specific_instructions = f"GZIP-specific instructions: Focus on extracting compressed files and their relationships. Identify dependencies between compressed files. Pay attention to compression formats."
        # Oracle-specific file type instructions
        elif file_type == 'oracle_form':
            specific_instructions = "Oracle Forms-specific instructions: Focus on extracting form blocks, items, triggers, canvases, and their relationships. Identify data block relationships to database tables and master-detail relationships between blocks."
        elif file_type == 'oracle_library':
            specific_instructions = "Oracle Forms Library-specific instructions: Focus on extracting reusable procedures, functions, and objects. Identify dependencies between libraries and forms that use them."
        elif file_type == 'oracle_menu':
            specific_instructions = "Oracle Forms Menu-specific instructions: Focus on extracting menu structure, menu items, and their associated forms or procedures. Identify security and role-based access patterns."
        elif file_type == 'oracle_report':
            specific_instructions = "Oracle Reports-specific instructions: Focus on extracting report structure, data models, queries, and formatting. Identify parameter dependencies and data source relationships."
        elif file_type == 'plsql_package':
            specific_instructions = "PL/SQL Package-specific instructions: Focus on extracting package specifications and bodies, public and private procedures/functions, cursors, exceptions, and package variables. Identify dependencies between packages."
        elif file_type == 'plsql_function':
            specific_instructions = "PL/SQL Function-specific instructions: Focus on extracting function parameters, return types, local variables, and dependencies. Pay attention to deterministic functions and purity levels."
        elif file_type == 'plsql_trigger':
            specific_instructions = "PL/SQL Trigger-specific instructions: Focus on extracting trigger events, timing, affected tables, trigger logic, and dependencies on other database objects."
        elif file_type == 'plsql_type':
            specific_instructions = "PL/SQL Type-specific instructions: Focus on extracting object types, collection types, member methods, and type dependencies."
        elif file_type == 'oracle_config':
            specific_instructions = "Oracle Configuration-specific instructions: Focus on extracting configuration parameters, connection strings, resource definitions, and environment settings."
        # Oracle APEX conversion file type instructions
        elif file_type == 'oracle_form_xml':
            specific_instructions = "Oracle Forms XML (APEX Conversion)-specific instructions: Focus on extracting XML structure representing converted Forms modules. Identify form blocks, items, triggers, and canvases from XML elements. Map original Forms metadata preserved in XML format to knowledge graph entities."
        elif file_type == 'oracle_library_xml':
            specific_instructions = "Oracle Object Library XML (APEX Conversion)-specific instructions: Focus on extracting XML structure representing converted Object Libraries. Identify reusable form objects, their properties, and dependencies from XML elements. Map library object relationships preserved in XML format."
        elif file_type == 'oracle_menu_xml':
            specific_instructions = "Oracle Menu XML (APEX Conversion)-specific instructions: Focus on extracting XML structure representing converted Menu Modules. Identify menu items, their hierarchy, associated forms or procedures, and security settings from XML elements."
        elif file_type == 'plsql_library_text':
            specific_instructions = "PL/SQL Library Text (.pld)-specific instructions: Focus on extracting PL/SQL procedures, functions, cursors, and exceptions from text format library files. These are converted from binary .pll files for APEX migration. Identify library dependencies and public/private interfaces."
        else:
            specific_instructions = f"General instructions for {file_type}: Focus on extracting relevant entities, relationships, and patterns based on the file type. Identify dependencies and interactions between components. Pay attention to conventions and best practices for the specific file type."

        return f"{context}\n\n" \
                f"{specific_instructions}\n\n" \
                f"{default_instructions}\n\n"      

    
    async def create_knowledge_graph(self, 
                              files_metadata: List[FileMetadata], 
                              llm_model,
                              graph_store,
                              analysis_context: str = "legacy") -> List[GraphDocument]:
        """
        Create knowledge graph using LangChain LLMGraphTransformer
        
        Args:
            files_metadata: List of FileMetadata objects
            llm_model: Language model for graph transformation
            graph_store: Graph vector store for adding documents
            
        Returns:
            List of GraphDocument objects
        """
       
        # Create documents from file metadata and contents

        content_type = files_metadata[0].file_type if files_metadata else 'unknown'
        
        # Set context based on analysis_context
        if analysis_context == "oracle":
            context = "Oracle Forms and PL/SQL application analysis"
        else:
            context = "Legacy Application modernization"
        
        # Load graph schema configuration
        schema_config = self.load_graph_schema(analysis_context)

        documents = []
        for metadata in files_metadata:
            # Read file contents
            file_contents = self._read_file_contents(Path(metadata.path))
            
            # Create comprehensive page content with metadata and file contents

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

        # process the document batch, handle exceptions and record failures

        
        for doc in documents:
            # display the document content and metadata
            print(f"Document: {doc.metadata['filename']}")
            print(f"File Type: {doc.metadata['file_type']}")
            print(f"Content Type: {doc.metadata['content_type']}")
            print(f"Source: {doc.metadata['source']}")
            print(f"Source URI: {doc.metadata['source_uri']}")
            print(f"Size: {doc.metadata['size']} bytes")
            print(f"Line Count: {doc.metadata['line_count']}")
            print(f"Modified Date: {doc.metadata['modified_date']}")
            print(f"Content: {doc.page_content[:500]}...")  # Display first 500 chars of content

        # Configure LLMGraphTransformer with schema restrictions
        transformer_kwargs = {
            "llm": llm_model,
            "node_properties": True,
            "relationship_properties": True,
            "additional_instructions": self.get_additional_instructions(content_type, context, analysis_context)
        }
        
        # Add allowed nodes and relationships if specified in schema
        allowed_nodes=[]
        allowed_relationships=[]
        if schema_config.get("allowed_nodes"):
            transformer_kwargs["allowed_nodes"] = schema_config["allowed_nodes"]
            allowed_nodes = schema_config["allowed_nodes"]
            print(f"🎯 Restricting to {len(schema_config['allowed_nodes'])} allowed node types")
        
        if schema_config.get("allowed_relationships"):
            transformer_kwargs["allowed_relationships"] = schema_config["allowed_relationships"]
            allowed_relationships= schema_config["allowed_relationships"]
            print(f"🔗 Restricting to {len(schema_config['allowed_relationships'])} allowed relationship types")
        

        transformer = LLMGraphTransformer(**transformer_kwargs)
     
        
        # Transform to graph documents
        graph_documents = await transformer.aconvert_to_graph_documents(documents)

        # Display graph document information
        for graph_doc in graph_documents:
            print(f"Node Count: {len(graph_doc.nodes)}")
            print(f"Relationship Count: {len(graph_doc.relationships)}")
            # print(f"Source: {graph_doc.source}")

            # display node details
            print("Nodes:")
            for node in graph_doc.nodes:
                print(f"Node ID: {node.id}, Type: {node.type}")
                # print(f"Node ID: {node.id}, Type: {node.type}, Properties: {node.properties}")

            # display relationship details
            for rel in graph_doc.relationships:
                print(f"Relationship: {rel}")
        
        # Add to graph store if provided
        if graph_store:
            graph_store.add_graph_documents(graph_documents)
        
        return graph_documents
    
    def _read_file_contents(self, file_path: Path) -> str:
        """
        Read file contents with support for PDF and text files
        
        Args:
            file_path: Path to the file
            
        Returns:
            File contents as string
        """
        if file_path.suffix.lower() == '.pdf':
            return self._read_pdf_contents(file_path)
        else:
            return self._read_text_contents(file_path)
    
    def _read_pdf_contents(self, file_path: Path) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            return f"[Unable to read PDF contents: {e}]"
    
    def _read_text_contents(self, file_path: Path) -> str:
        """
        Read text file contents with encoding fallback
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File contents as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            try:
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    return f.read()
            except OSError:
                return f"[Unable to read file contents: {file_path}]"