# Atlas 🗺️

**Agentic AI Application Landscape Discovery**

Atlas is an intelligent platform for analyzing, understanding, and discovering complex application landscapes through AI-powered analysis, using knowledge graphs and hybrid document document vector indexing.

## TL;DR
The following comprehensive documentation describes the Atlas concept in detail, to find out how to get started quickly with installing, running Atlas and analysing your needs head to the [quickstart guide](#-quick-start)

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

### Predefined Relationship Support in Knowledge Graphs
Atlas now supports precise relationship 'triple' format definitions (`[Source, Relationship, Target]`) in graph schemas, providing:
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
- **Precise Relationship Support**: Define relationships as [Source, Relationship, Target] tuples
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
- Python 3.10+ (3.10, 3.11, 3.12, 3.13 are supported)
- uv - Fast Python package manager (recommended)
- Neo4j Database (local or cloud)
- API keys for your chosen LLM provider

### Installation

1. **Install uv (Fast Python Package Manager):**
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Clone the repository:**
```bash
git clone https://github.com/nickydwyer/Atlas.git
cd Atlas
```

3. **Install dependencies with uv:**
```bash
# Recommended method - automatically creates venv and installs dependencies
uv sync
```

4. **Configure environment:**
```bash
cp .env.template .env
# Edit .env with your API keys and Neo4j configuration
```

### Running Atlas

For comprehensive usage instructions, including all commands, options, and advanced features, please refer to the **[User Guide](docs/user-guide.md)**.

**Quick command examples:**
```bash
# Analyze code and generate knowledge graph
uv run python atlas.py analyze --folder-path /path/to/code --generate-knowledge-graph

# Launch interactive chat interface
uv run python atlas.py chat

# Refine and optimize knowledge graph
uv run python atlas.py refine

# Validate environment configuration
uv run python atlas.py validate
```

### Documentation

- **[User Guide](docs/user-guide.md)** - Complete usage instructions and command reference
- **[Neo4j Setup Guide](docs/neo4j-setup.md)** - Neo4j installation and configuration
- **[MCP Configuration Guide](docs/mcp-configuration.md)** - Model Context Protocol server setup
- **[Azure Deployment Guide](docs/azure-windows-deployment.md)** - Azure VM deployment


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
- **Prescribed Relationships**: Support for [Source, Relationship, Target] tuples for precise control
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