# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Plan & Review

### Before starting work

- Always use planning mode to make a plan.
- After producing the plan, make sure you write the plan to .claude/tasks/$TASK_NAME.md, where $TASK_NAME represents the name of the activity or task.
- The plan should be a detailed implementation plan, task breakdown and the rationale for each step.
- If the task requires external knowledge or a certain package, use research to obtain the latest information (e.g. use the Task tool for research). If the external resources are Gen-AI related ensure that the most recent or up-to-date information is used. Be aware of stale or superseded content.
- Do not overplan, always aim for an MVP.
- Once you generate the plan, ask for review. Do not continue to execute until the plan is approved by the user.

### During implementation

- You must update the plan as you progress the work.
- After completing the tasks in the plan, you should update and append detailed descriptions of the changes you have made, so following tasks can easily hand over to other engineers.

## Project Overview

Atlas is an AI-powered application landscape discovery system that uses knowledge graphs and vector indexing to analyze complex landscapes including. Atlas can analyse multiple data sources, both structured and unstructured. These data sources can be application source code assets of any kind, including configuration, documentation, runbooks, service desk tickets, data schemas, batch jobs, ETL and many other asset types. It combines Graph RAG (Retrieval-Augmented Generation) with Neo4j graph database technology to provide intelligent analysis capabilities.

## Atlas Initial Concept

![Atlas Initial Concept Draft](./images/Atlas%20Initial%20concept%20draft.png)

## Roadmap and goals

Atlas is intended to be an extensible agentic ai platform. Extension and augmentation is a key concept which is to be used for every feature. Incorporate and integrate rather than build from scratch.

### Data integration and context creation

Altas provides AI-enabled insights based on the ingest of multiple data sources and formats, including both unstructured and structured. Where possible MCP-based integrations are preferred. Open standards and protocols are used at all times.

Atlas also includes key context engineering aspects such as long-term and short-term memory.

Knowledge, insight and recommendations are created from an integrated context.


## Core Commands

### Prerequisites
- Python 3.10 or higher (3.10, 3.11, 3.12, 3.13 are supported)
- Neo4j Database (local or cloud)
- API keys for chosen LLM providers

### Development Environment
```bash
# Install dependencies using uv (preferred)
uv sync

# Or with pip
pip install -r requirements.txt

# Install development dependencies
uv sync --group dev
```

### Main Application Commands
```bash
# Analyze legacy applications and generate knowledge graph
uv run python atlas.py analyze --folder-path /path/to/code --generate-knowledge-graph

# Analyze Oracle Forms/PL-SQL applications (specialized context)
uv run python atlas.py analyze --folder-path /path/to/oracle --analysis-context oracle --generate-knowledge-graph

# Analyze ITSM ticket data (specialized context)
uv run python atlas.py analyze --folder-path /path/to/itsm_data --analysis-context itsm --generate-knowledge-graph

# Launch interactive chat interface
uv run python atlas.py chat

# Refine and optimize existing knowledge graph
uv run python atlas.py refine

# Validate environment configuration and connectivity
uv run python atlas.py validate
```

### Environment Management

Atlas supports multiple environments for easy configuration switching:

```bash
# Use development environment (loads .env.dev)
uv run python atlas.py analyze --env dev --folder-path /path/to/code --generate-knowledge-graph

# Use staging environment (loads .env.staging)
uv run python atlas.py chat --env staging

# Use production environment (loads .env.prod)
uv run python atlas.py refine --env production

# Use custom environment file
uv run python atlas.py analyze --env-file .env.client1 --folder-path /path/to/code

# Validate specific environment
uv run python atlas.py validate --env production

# Validate only Neo4j connectivity
uv run python atlas.py validate --component neo4j

# Save validation report
python atlas.py validate --save-report validation_prod_report.json
```

## Analysis Contexts

Atlas supports multiple analysis contexts through specialized analyzers that provide domain-specific knowledge extraction:

### Legacy Application Analysis (Default)
- **Context**: `--analysis-context legacy` (default)
- **Analyzer**: `FileAnalyzer`
- **Purpose**: General application modernization and legacy code analysis
- **Supported Formats**: All standard programming languages, mainframe formats (COBOL, JCL, BMS), documentation
- **Focus**: Code structure, dependencies, business logic extraction

### Oracle Forms/PL-SQL Analysis
- **Context**: `--analysis-context oracle`
- **Analyzer**: `OracleFileAnalyzer`
- **Purpose**: Oracle Forms and PL/SQL application analysis
- **Supported Formats**: 
  - Oracle Forms: `.fmb`, `.fmx`, `.fmt`, `.pll`, `.plx`, `.mmb`, `.mmx`, `.olb`
  - Oracle Reports: `.rdf`, `.rex`, `.rep`
  - PL/SQL: `.pks`, `.pkb`, `.prc`, `.fnc`, `.trg`, `.typ`, `.tps`, `.tpb`
  - Oracle SQL: `.sql`, `.ddl`, `.dml`, `.plsql`
  - Configuration: `.ora`, `.conf`, `.properties`
- **Focus**: Oracle-specific entities (forms, blocks, triggers, packages, procedures), database relationships

### ITSM Data Analysis
NOTE: This feature is still in development and is a placeholder only.
- **Context**: `--analysis-context itsm`
- **Analyzer**: `ITSMDataAnalyzer`
- **Purpose**: IT Service Management ticket and process analysis
- **Supported Formats**:
  - Data exports: `.csv`, `.json`, `.jsonl`, `.xml`, `.xlsx`, `.xls`, `.tsv`
  - ITSM-specific: `.tickets`, `.incidents`, `.problems`, `.changes`, `.assets`, `.cmdb`
  - Logs: `.log`, `.audit`, `.history`
- **Focus**: ITSM entities (incidents, problems, changes, assets, services), process flows, temporal patterns

## Architecture Overview

### Core Components

- **atlas.py**: Main CLI application with three commands (analyze, chat, refine)
- **file_analyzer.py**: Core file processing engine for legacy application analysis
- **oracle_file_analyzer.py**: Specialized analyzer for Oracle Forms and PL/SQL applications
- **itsm_data_analyzer.py**: Specialized analyzer for ITSM data and ticket systems
- **chat_app.py**: Streamlit-based interactive chat interface with MCP (Model Context Protocol) integration
- **refine_knowledge_base.py**: Graph optimization module for analyzing and improving knowledge graphs

### Key Classes

- **FileAnalyzer**: Handles legacy code file processing, metadata extraction, and batch analysis
- **OracleFileAnalyzer**: Specialized processor for Oracle Forms, PL/SQL, and Oracle database artifacts
- **ITSMDataAnalyzer**: Specialized processor for ITSM data formats including tickets, incidents, and service data
- **Neo4jGraphStore**: Manages Neo4j connections, knowledge graph storage, and vector indexing
- **KnowledgeGraphRefiner**: Analyzes graph structure and generates optimization recommendations
- **MCPToolManager**: Manages Model Context Protocol tools for enhanced chat capabilities

### Technology Stack

- **LangChain Ecosystem**: Core LLM orchestration and knowledge graph generation
- **Neo4j**: Graph database with vector search capabilities
- **Streamlit**: Web-based chat interface
- **FastMCP**: Model Context Protocol integration
- **Multiple LLM Providers**: OpenAI, Anthropic, Google Gemini, Ollama

## Configuration

### Environment Setup

Atlas supports multiple environment configurations for different deployment scenarios (development, staging, production). This allows easy switching between different Neo4j databases, API keys, and default settings.

#### Basic Setup
```bash
# Copy the main template
cp .env.template .env

# Or create environment-specific files
cp .env.dev.template .env.dev
cp .env.staging.template .env.staging  
cp .env.prod.template .env.prod
```

#### Environment File Structure
Atlas looks for environment files in this order:
1. Custom file specified with `--env-file`
2. Environment-specific file (e.g., `.env.dev` for `--env dev`)
3. Default `.env` file as fallback

#### Required Environment Variables
- **LLM Provider Keys**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`
- **Neo4j Connection**: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`

#### Default LLM Settings (New Feature)
Set default provider and model to avoid specifying them on each command:
```bash
# In your .env file
DEFAULT_LLM_PROVIDER=anthropic
DEFAULT_LLM_MODEL=claude-sonnet-4-20250514
```

With these defaults set, you can run:
```bash
# Instead of: python atlas.py chat --llm-provider anthropic --model claude-sonnet-4-20250514
python atlas.py chat
```

#### Environment Validation
Use the validate command to check your environment configuration:
```bash
# Validate current environment
python atlas.py validate

# Validate specific environment
python atlas.py validate --env production

# Validate only Neo4j connectivity
python atlas.py validate --component neo4j

# Get detailed validation report
python atlas.py validate --save-report environment_check.json
```

### Neo4j Setup
Atlas requires a Neo4j database. See `NEO4J_SETUP.md` for detailed setup instructions including:
- Local Docker setup
- Neo4j Aura cloud configuration
- MCP server configuration for chat interface

## Development Patterns

### LLM Provider Integration
The system supports multiple LLM providers through a unified interface. When adding new providers:
- Follow the pattern in existing provider implementations
- Implement both chat and embedding capabilities
- Ensure proper error handling and fallback mechanisms

### Knowledge Graph Processing
- Knowledge graphs are generated using LangChain's experimental graph transformers
- Document chunking preserves metadata and enables vector search
- Neo4j stores both graph relationships and vector embeddings

### MCP Tool Integration
The chat interface uses Model Context Protocol for enhanced capabilities:
- **External Configuration**: MCP servers are configured via `mcp_servers.json` file
- **Dynamic Tool Loading**: Tools are loaded based on enabled servers in configuration
- **Environment Variable Interpolation**: Supports `${VAR}` and `${VAR:default}` syntax
- **Fallback Support**: Graceful fallback to hardcoded Neo4j if configuration fails
- **Multiple Server Types**: Support for Neo4j, filesystem, web search, GitHub, and custom servers

## Common Workflows

### Full Analysis Pipeline
1. **Analyze**: Process files and generate knowledge graph with vector indexing
2. **Chat**: Interactive exploration of the generated knowledge
3. **Refine**: Optimize graph structure and generate reports

### Context-Specific Workflows

#### Legacy Application Modernization
```bash
# Set up development environment
python atlas.py validate --env dev

# Analyze legacy codebase
python atlas.py analyze --env dev --folder-path /path/to/legacy --analysis-context legacy --generate-knowledge-graph --index-documents

# Interactive exploration
python atlas.py chat --env dev

# Optimize knowledge graph
python atlas.py refine --env dev
```

#### Oracle Forms Migration Analysis
```bash
# Validate production environment first
python atlas.py validate --env production --component neo4j

# Analyze Oracle Forms and PL/SQL
python atlas.py analyze --env production --folder-path /path/to/oracle --analysis-context oracle --generate-knowledge-graph --file-type-filter .fmb,.pks,.sql

# Focus on specific Oracle components
python atlas.py analyze --env production --file-name /path/to/form.fmb --analysis-context oracle --generate-knowledge-graph
```

#### ITSM Process Analysis
```bash
# Use staging environment for testing
python atlas.py validate --env staging

# Analyze ITSM ticket exports
python atlas.py analyze --env staging --folder-path /path/to/itsm_exports --analysis-context itsm --generate-knowledge-graph --file-type-filter .csv,.json

# Process ServiceNow/Remedy data dumps
python atlas.py analyze --env staging --file-name tickets_export.xlsx --analysis-context itsm --generate-knowledge-graph
```

### Adding New File Types
Extend `file_analyzer.py` to support additional file formats:
- Add file extension handling
- Implement content extraction logic
- Ensure metadata preservation

### Adding New MCP Servers
Add new MCP tools through configuration (no code changes required):
1. **Edit `mcp_servers.json`**: Add server configuration with executable, args, and environment requirements
2. **Set Environment Variables**: Configure required environment variables in `.env` file
3. **Enable Server**: Set `"enabled": true` in the server configuration
4. **Restart Chat**: The new tools will be available in the next chat session

### MCP Configuration Format
Example server configuration in `mcp_servers.json`:
```json
{
  "servers": {
    "my-server": {
      "enabled": true,
      "description": "My custom MCP server",
      "command": {
        "executable": "uvx",
        "args": ["my-mcp-package"]
      },
      "args": ["--transport", "stdio", "--api-key", "${MY_API_KEY}"],
      "transport": "stdio",
      "environment": {
        "required": ["MY_API_KEY"],
        "optional": ["MY_OPTIONAL_CONFIG"]
      }
    }
  }
}
```

## Testing and Quality

### Test Structure
- Tests are located in `tests/` directory
- Use pytest for test execution
- Coverage reports show code coverage metrics

### Code Quality Tools
- **Black**: Code formatting (line length: 88)
- **MyPy**: Type checking (requires type hints)
- **Flake8**: Code linting and style checks
- **Pre-commit**: Automated quality checks

## Key Design Principles

### Graph RAG Architecture
The system implements Graph RAG, which provides superior performance over traditional vector-only RAG through:
- Multi-hop reasoning across connected entities
- Contextual relevance via graph relationships
- Explainable reasoning paths
- Reduced hallucination through structured constraints

### Modular Design
Each component is designed for extensibility:
- LLM providers are interchangeable
- File analyzers can be extended for new formats
- Knowledge graph schemas evolve dynamically
- MCP tools provide pluggable capabilities

### Production Ready
The codebase includes enterprise features:
- Comprehensive error handling and logging
- Performance metrics and statistics tracking
- Professional report generation
- Scalable batch processing