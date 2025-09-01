# Atlas 🗺️

**Agentic AI Application Landscape Discovery**

Atlas is an intelligent platform for analyzing, understanding, and discovering complex application landscapes through AI-powered analysis, using knowledge graphs and hybrid document document vector indexing.

## 📑 Contents

- [Transforming Knowledge Graph Development and Retrieval using LLMs](#transforming-knowledge-graph-development-and-retrieval-using-llms)
  - [Automated Entity Extraction and Schema Generation](#automated-entity-extraction-and-schema-generation)
  - [Semantic Understanding and Contextualization](#semantic-understanding-and-contextualization)
  - [Dynamic Schema Evolution and Maintenance](#dynamic-schema-evolution-and-maintenance)
  - [Graph RAG Superiority Over Traditional RAG](#graph-rag-superiority-over-traditional-rag)
  - [Agentic AI and Interactive Knowledge Discovery](#agentic-ai-and-interactive-knowledge-discovery)
  - [The benefits of Graph RAG solutions](#the-benefits-of-graph-rag-solutions)
- [🆕 Recent Enhancements](#-recent-enhancements)
- [✨ Atlas Features](#-atlas-features)
- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [📚 Usage](#-usage)
  - [Analyze Command](#-analyze-command)
  - [Refine Command](#-refine-command)
  - [Chat Command](#-chat-command)
- [🛠️ Provider Configuration](#-provider-configuration)
  - [OpenAI](#openai)
  - [Anthropic](#anthropic)
  - [Ollama (Local)](#ollama-local)
- [📊 Neo4j Setup](#-neo4j-setup)
  - [Local Neo4j](#local-neo4j)
  - [Neo4j Aura (Cloud)](#neo4j-aura-cloud)
- [🏗️ Architecture](#-architecture)
- [🔧 Key Components](#-key-components)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

## Transforming Knowledge Graph Development and Retrieval using LLMs

LLMs have fundamentally transformed how we work with knowledge graphs, addressing longstanding challenges in schema design and query management, unlocking new capabilities that can be used to help with understanding complex domains.

### **Automated Entity Extraction and Schema Generation**

LLMs excel at parsing unstructured text to identify entities, relationships, and concepts that would traditionally require extensive manual effort. They can automatically extract domain-specific entities from documents, emails, reports, and other text sources, including source code, then map these to appropriate ontological structures. This dramatically reduces the time from raw data to actionable knowledge representation.

The models can also suggest and iteratively refine knowledge graph schemas by analyzing patterns in the extracted entities and relationships. This addresses one of the biggest historical barriers - the upfront investment in schema design that often required deep domain expertise and extensive planning.

### **Semantic Understanding and Contextualization**

Unlike traditional rule-based extraction systems, LLMs bring semantic understanding that captures nuanced relationships and context-dependent meanings. They can disambiguate entities (distinguishing between "Apple" the company versus the fruit), understand implicit relationships, and maintain consistency across large document collections.

This semantic capability extends to multi-modal data integration, where LLMs can help connect structured databases, documents, images, and other data sources into coherent knowledge representations.

### **Dynamic Schema Evolution and Maintenance**

LLMs enable knowledge graphs to evolve organically as new information emerges. Rather than requiring manual schema updates, the models can suggest new entity types, relationships, or structural modifications based on incoming data patterns. This makes knowledge graphs more adaptive to changing business requirements and emerging domains.

### **Graph RAG Superiority Over Traditional RAG**

Graph RAG systems leverage the rich relational structure of knowledge graphs to provide more sophisticated retrieval and reasoning capabilities. Key advantages include:

**Multi-hop reasoning**: Graph RAG can traverse relationships to answer complex questions requiring inference across multiple connected entities, something traditional vector-based RAG struggles with.

**Contextual relevance**: The graph structure provides semantic context that helps retrieve not just similar documents, but relationally relevant information that may not be lexically similar.

**Explainable reasoning paths**: Graph traversal provides clear audit trails showing how conclusions were reached, making the AI's reasoning more transparent and trustworthy.

**Reduced hallucination**: The structured nature of knowledge graphs provides constraints that help ground LLM responses in factual relationships rather than generating plausible-sounding but incorrect information.

### **Agentic AI and Interactive Knowledge Discovery**

LLMs enable sophisticated agentic behaviors when working with knowledge graphs, such as automated hypothesis generation, systematic exploration of knowledge gaps, and intelligent questioning strategies. These agents can autonomously identify areas where the knowledge graph needs expansion or correction, then orchestrate data collection and validation processes.

The combination has made knowledge graphs accessible to domain experts without technical backgrounds, democratizing advanced knowledge management capabilities across organizations while maintaining the precision and structure that makes knowledge graphs so valuable for complex reasoning tasks.

### The benefits of Graph RAG solutions
Multiple sources underline Graph RAG's significant advantages over traditional RAG approaches. Microsoft Research's foundational paper showed that GraphRAG provides substantial improvements in question-and-answer performance when conducting document analysis, particularly excelling where baseline RAG "struggles to connect the dots". Data.world's study revealed GraphRAG improved LLM response accuracy by 3x across 43 business questions, while RobustQA benchmarking showed Graph RAG achieving 86.31% accuracy compared to traditional RAG's 72.36%. Neo4j research found 77.6% improvement in MRR and 0.32 BLEU score gains, and LinkedIn reported 62.5% reduction in ticket resolution time through Graph RAG's multi-hop reasoning capabilities. Academic evaluations in ArXiv papers systematically demonstrate GraphRAG's distinct strengths across Question Answering and Query-based Summarization tasks, with studies showing equivalent or better quality responses using just 2-3% of the tokens compared to traditional approaches.

**Reference URLs:**

- Microsoft Research GraphRAG Paper: https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
- Neo4j GraphRAG Manifesto (Data.world study): https://neo4j.com/blog/genai/graphrag-manifesto/
- RobustQA Benchmarking Analysis: https://www.ankursnewsletter.com/p/graph-rag-vs-traditional-rag-a-comparative
- LinkedIn Implementation Results: https://www.chitika.com/rag-vs-graph-rag-which-one-is-the-real-game-changer/
- ArXiv Systematic Evaluation Paper: https://arxiv.org/abs/2502.11371
- Token Efficiency Study: https://diamantai.substack.com/p/graph-rag-explained


## 🆕 Recent Enhancements

### Triple Relationship Support in Knowledge Graphs
Atlas now supports precise relationship definitions using triple format `[Source, Relationship, Target]` in graph schemas, providing:
- **Fine-grained control** over which node types can connect
- **Schema validation** to ensure only valid relationships are created
- **Better graph quality** with reduced noise and hallucinations
- **Clearer semantics** for domain-specific knowledge graphs

### Flexible Environment Configuration
- **Any environment name** is now supported (not just dev/staging/prod)
- **Custom environments** automatically look for `.env.{name}` files
- **Client-specific** configurations like `.env.client1`, `.env.uat`, etc.
- **Git security** - all `.env.*` files are automatically ignored

### Default Configuration from Environment
- Set `DEFAULT_LLM_PROVIDER` and `DEFAULT_LLM_MODEL` to avoid specifying them on each command
- Set `DEFAULT_ANALYSIS_CONTEXT` to avoid specifying `--analysis-context` (defaults to 'legacy')
- Run commands without specifying these arguments every time
- Command-line arguments still override environment defaults when needed

## ✨ Atlas Features

### 🔍 **Repo and Folder Analysis**
- **Multi-file Processing**: Analyze entire folder hierarchies or individual files
- **Intelligent Filtering**: Support for file type filters and batch processing
- **Metadata Extraction**: Comprehensive file analysis with statistics
- **Multiple Analysis Contexts**: Specialized analyzers for different domains (legacy, Oracle, ITSM)
- **Environment Management**: Support for development, staging, and production environments

### 🧠 **Knowledge Graph Generation**
- **Neo4j Integration**: Create rich knowledge graphs from code analysis
- **Entity Relationships**: Automatic extraction of functions, classes, and dependencies
- **Triple Relationship Support**: Define precise relationships as [Source, Relationship, Target] tuples
- **Schema Validation**: Configurable schemas with allowed nodes and relationships per context
- **Real-time Statistics**: Track nodes, relationships, and processing progress

### 📄 **Document Vector Indexing**
- **Semantic Search**: Vector embeddings for intelligent document search
- **Multi-provider Support**: OpenAI, Anthropic, Ollama, and Google embeddings
- **Chunking & Indexing**: Smart document chunking with metadata preservation
- **Neo4j Vector Search**: Integrated vector and full-text search capabilities

### 🔧 **Knowledge Graph Refinement**
- **Graph Analysis**: Comprehensive analysis of graph structure and connectivity
- **LLM Optimization**: AI-powered recommendations for graph improvements
- **Professional Reports**: Generate markdown and HTML reports with visualizations
- **Iterative Improvement**: Version-controlled analysis reports

### 💬 **Interactive Chat Interface**
- **MCP Tool Integration**: Model Context Protocol for enhanced capabilities
- **Multi-provider Support**: OpenAI, Anthropic, Ollama, and Google Gemini
- **Streamlit UI**: Modern web interface for conversations
- **Environment Validation**: Built-in connectivity and configuration validation

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Neo4j Database (local or cloud)
- API keys for your chosen LLM provider

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/nickydwyer/Atlas.git
cd Atlas
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment:**
```bash
# Copy and configure main environment file
cp .env.template .env
# Edit .env with your API keys and Neo4j configuration

# Or create environment-specific configurations
cp .env.dev.template .env.dev
cp .env.staging.template .env.staging
cp .env.prod.template .env.prod
```

### Environment Configuration

Atlas supports multiple environment configurations for different deployment scenarios. This allows easy switching between different Neo4j databases, API keys, and default settings.

#### Flexible Environment Naming
Atlas supports any environment name, not just the traditional dev/staging/prod:
- Common environments map to predefined files: `dev` → `.env.dev`, `prod` → `.env.prod`
- Any custom environment name works: `client1` → `.env.client1`, `testing` → `.env.testing`
- All `.env.*` files are automatically ignored by git for security

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
```env
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
OLLAMA_BASE_URL=http://localhost:11434

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

#### Default Settings (New Feature)
Set default provider, model, and analysis context to avoid specifying them on each command:
```bash
# In your .env file
DEFAULT_LLM_PROVIDER=anthropic
DEFAULT_LLM_MODEL=claude-sonnet-4-20250514
DEFAULT_ANALYSIS_CONTEXT=oracle
```

With these defaults set, you can run:
```bash
# Instead of: python atlas.py analyze --llm-provider anthropic --model claude-sonnet-4-20250514 --analysis-context oracle --folder-path /path/to/code
python atlas.py analyze --folder-path /path/to/code --generate-knowledge-graph

# The defaults work for all commands
python atlas.py chat
python atlas.py refine
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

#### Exporting Environment Variables

Atlas provides both automated and manual ways to export environment variables for use by other programs:

##### Automated Export (Recommended)
Use the `--export` flag to securely export variables to your shell:

```bash
# Export variables from default environment
source <(python atlas.py analyze --export --folder-path dummy)

# Export variables from specific environment
source <(python atlas.py analyze --env dev --export --folder-path dummy)
source <(python atlas.py analyze --env staging --export --folder-path dummy)
source <(python atlas.py analyze --env prod --export --folder-path dummy)

# Export and run command in one line
source <(python atlas.py chat --env prod --export --folder-path dummy) && neo4j-admin console

# Create aliases for easy environment switching
alias atlas-env-dev='source <(python atlas.py analyze --env dev --export --folder-path dummy)'
alias atlas-env-staging='source <(python atlas.py analyze --env staging --export --folder-path dummy)'
alias atlas-env-prod='source <(python atlas.py analyze --env prod --export --folder-path dummy)'

# Usage with aliases
atlas-env-dev
python some_other_tool.py  # Now has access to Atlas environment variables

atlas-env-prod
neo4j-admin console  # Uses production Neo4j settings
```

**Troubleshooting Export Issues:**

If environment variables aren't being set properly:

```bash
# 1. Test the export function directly (shows masked secrets for security)
python atlas.py analyze --export --folder-path dummy

# 2. Check what gets exported when piped (actual values for shell sourcing)
python atlas.py analyze --export --folder-path dummy > /tmp/atlas_exports.sh
cat /tmp/atlas_exports.sh  # Review the export commands
source /tmp/atlas_exports.sh  # Apply them
rm /tmp/atlas_exports.sh  # Clean up

# 3. Verify variables are set
echo $NEO4J_URI
echo $OPENAI_API_KEY  # Should show the actual key if properly sourced

# 4. Debug with manual commands
python atlas.py validate  # Check if .env file loads properly
```

##### Manual Export (Advanced Users)
For advanced users who need manual control:

```bash
# Safe export with source (handles spaces and special characters)
set -a && source .env.dev && set +a
set -a && source .env.staging && set +a
set -a && source .env.prod && set +a

# Export specific variables (be careful with secrets)
export NEO4J_URI=$(grep '^NEO4J_URI=' .env.prod | cut -d'=' -f2-)
export DEFAULT_LLM_PROVIDER=$(grep '^DEFAULT_LLM_PROVIDER=' .env.prod | cut -d'=' -f2-)

# One-time environment for single command (safer for secrets)
env $(grep -v '^#' .env.prod | xargs) python external_tool.py
```

##### Security Notes
- 🔒 The `--export` flag masks secret values in console output for security
- 🔒 Secret values are still exported to shell environment for program use
- 🔒 Avoid displaying secrets in console or logs
- 🔒 Use `--export` instead of manual commands when possible

##### Viewing Current Environment
```bash
# View Atlas-related environment variables
printenv | grep -E "(OPENAI|ANTHROPIC|GEMINI|NEO4J|DEFAULT_LLM|ATLAS)"

# Temporarily override environment for single command
NEO4J_URI=bolt://localhost:7688 python atlas.py validate --component neo4j
```

## 📚 Usage

### 🔍 Analyze Command

**Basic Analysis:**
```bash
python atlas.py analyze --folder-path /path/to/your/code
```

**With Knowledge Graph Generation:**
```bash
python atlas.py analyze --folder-path /path/to/your/code \
  --generate-knowledge-graph \
  --llm-provider openai \
  --model gpt-4o-mini
```

**Using Different Environments:**
```bash
# Use development environment
python atlas.py analyze --env dev --folder-path /path/to/your/code --generate-knowledge-graph

# Use staging environment  
python atlas.py analyze --env staging --folder-path /path/to/your/code --generate-knowledge-graph

# Use production environment
python atlas.py analyze --env production --folder-path /path/to/your/code --generate-knowledge-graph

# Use custom environment names (automatically looks for .env.{name})
python atlas.py analyze --env client1 --folder-path /path/to/your/code --generate-knowledge-graph
python atlas.py analyze --env testing --folder-path /path/to/your/code --generate-knowledge-graph
python atlas.py analyze --env uat --folder-path /path/to/your/code --generate-knowledge-graph

# Use custom environment file
python atlas.py analyze --env-file .env.client1 --folder-path /path/to/your/code --generate-knowledge-graph
```

**Analysis Contexts (Specialized Analyzers):**
```bash
# Legacy application analysis (default)
python atlas.py analyze --folder-path /path/to/legacy --analysis-context legacy --generate-knowledge-graph

# Oracle Forms/PL-SQL analysis
python atlas.py analyze --folder-path /path/to/oracle --analysis-context oracle --generate-knowledge-graph

# ITSM ticket data analysis
python atlas.py analyze --folder-path /path/to/itsm_data --analysis-context itsm --generate-knowledge-graph
```

**With Document Vector Indexing:**
```bash
python atlas.py analyze --folder-path /path/to/your/code \
  --generate-knowledge-graph \
  --index-documents \
  --llm-provider openai \
  --model gpt-4o-mini
```

**Single File Analysis:**
```bash
python atlas.py analyze --file-name /path/to/specific/file.py \
  --generate-knowledge-graph \
  --llm-provider anthropic \
  --model claude-sonnet-4-20250514
```

**With Google Gemini:**
```bash
python atlas.py analyze --folder-path /path/to/your/code \
  --generate-knowledge-graph \
  --index-documents \
  --llm-provider google \
  --model gemini-2.5-pro
```

**Advanced Options:**
```bash
python atlas.py analyze --folder-path /path/to/code \
  --generate-knowledge-graph \
  --index-documents \
  --max-files 100 \
  --batch-size 10 \
  --file-type-filter .py \
  --delete-kb \
  --llm-provider openai \
  --model gpt-4o-mini
```

**Environment Export:**
```bash
# Export environment variables for other tools
python atlas.py analyze --env prod --export --folder-path dummy

# Combine with shell sourcing for immediate use
source <(python atlas.py analyze --env prod --export --folder-path dummy)
neo4j-admin console  # Now uses exported Neo4j settings
```

### 🔧 Refine Command

**Analyze and Optimize Knowledge Graph:**
```bash
# Using default environment
python atlas.py refine --llm-provider anthropic --model claude-sonnet-4-20250514

# Using specific environment
python atlas.py refine --env production --llm-provider anthropic --model claude-sonnet-4-20250514

# Using default LLM settings (if configured in .env)
python atlas.py refine --env production
```

This will:
- Analyze graph structure and connectivity
- Generate optimization recommendations
- Create detailed markdown and HTML reports
- Provide iterative improvement suggestions

### 💬 Chat Command

**Interactive Chat Interface:**
```bash
# Using default environment
python atlas.py chat --llm-provider openai --model gpt-4o

# Using specific environment
python atlas.py chat --env staging --llm-provider openai --model gpt-4o

# Using default LLM settings (if configured in .env)
python atlas.py chat --env production
```

**With MCP Tools:**
```bash
python atlas.py chat --llm-provider anthropic --model claude-sonnet-4-20250514
```

## 🛠️ Provider Configuration

### OpenAI
```bash
# Set your API key
export OPENAI_API_KEY="your-api-key"

# Recommended models
--model gpt-4o-mini           # Fast and cost-effective
--model gpt-4o               # Most capable
--model text-embedding-3-small  # For embeddings
```

### Anthropic
```bash
# Set your API key
export ANTHROPIC_API_KEY="your-api-key"

# Recommended models
--model claude-sonnet-4-20250514  # Latest and most capable
--model claude-haiku-20240307     # Fast and efficient
```

### Google Gemini
```bash
# Set your API key
export GEMINI_API_KEY="your-api-key"

# Recommended models
--model gemini-2.5-pro        # Latest Gemini 2.5 Pro
--model gemini-2.5-flash      # Fast and efficient Gemini 2.5 Flash
--model gemini-1.5-pro        # Most capable stable model
--model models/text-embedding-004  # For embeddings
```

### Ollama (Local)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama2
ollama pull mistral
ollama pull codellama
ollama pull nomic-embed-text  # For embeddings

# Use with Atlas
--model llama2
--model mistral
--model codellama
```

## 📊 Neo4j Setup

### Local Neo4j
For detailed instructions on setting up Neo4j locally (including Docker usage), see the [Neo4j Setup Guide](./NEO4J_SETUP.md).

### MCP (Model Context Protocol) Configuration
Atlas chat interface supports MCP tools for enhanced capabilities. For detailed MCP configuration instructions, see the [MCP Configuration Guide](./MCP_CONFIGURATION.md).

### Neo4j Aura (Cloud)
1. Create account at https://neo4j.com/aura/
2. Create new database
3. Copy connection details to `.env`

## 🏗️ Architecture

```
Atlas/
├── atlas.py                 # Main CLI application
├── chat_app.py             # Streamlit chat interface
├── file_analyzer.py        # Universal file analyzer (legacy, Oracle, and general)
├── itsm_data_analyzer.py   # ITSM data analyzer
├── refine_knowledge_base.py # Graph refinement and optimization
├── env_config.py           # Environment configuration manager
├── env_validator.py        # Environment validation utilities
├── config/                 # Configuration directory
│   └── graph_schemas/      # Graph schema definitions
│       ├── legacy.yaml     # Legacy application schema
│       ├── oracle.yaml     # Oracle Forms/PL-SQL schema
│       └── itsm.yaml       # ITSM data schema
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── .env.template          # Main environment template
├── .env.dev.template      # Development environment template
├── .env.staging.template  # Staging environment template
└── .env.prod.template     # Production environment template
```

## 🔧 Key Components

### Core Analyzers
- **FileAnalyzer**: Universal analyzer supporting legacy applications, Oracle Forms/PL-SQL, and general code
- **ITSMDataAnalyzer**: ITSM ticket and service management data processing

### Graph Schema Configuration
- **YAML-based Schemas**: Define allowed nodes and relationships per analysis context
- **Triple Relationships**: Support for [Source, Relationship, Target] tuples for precise control
- **Schema Validation**: Automatic validation of relationships against allowed node types
- **Extensible Design**: Easy to add new contexts and schemas

### Infrastructure Components
- **Neo4jGraphStore**: Manages knowledge graph and vector indexing
- **KnowledgeGraphRefiner**: Analyzes and optimizes graph structure
- **ReportGenerator**: Creates professional analysis reports
- **MCPToolManager**: Handles Model Context Protocol tools

### Analysis Contexts
- **Legacy Context**: General application modernization and legacy code analysis
- **Oracle Context**: Oracle Forms, PL/SQL packages, triggers, and database relationships
- **ITSM Context**: IT Service Management processes, incidents, problems, and assets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) for LLM integration
- Uses [Neo4j](https://neo4j.com/) for graph database capabilities
- Powered by [Streamlit](https://streamlit.io/) for web interface
- Supports [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), [Google Gemini](https://ai.google.dev/), and [Ollama](https://ollama.ai/)

---

**Atlas** - Navigate the complexity of your application ecosystem with AI-powered analysis 🚀