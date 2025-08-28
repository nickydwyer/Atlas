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


## ✨ Atlas Features

### 🔍 **Repo and Folder Analysis**
- **Multi-file Processing**: Analyze entire folder hierarchies or individual files
- **Intelligent Filtering**: Support for file type filters and batch processing
- **Metadata Extraction**: Comprehensive file analysis with statistics

### 🧠 **Knowledge Graph Generation**
- **Neo4j Integration**: Create rich knowledge graphs from code analysis
- **Entity Relationships**: Automatic extraction of functions, classes, and dependencies
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
cp .env.template .env
# Edit .env with your API keys and Neo4j configuration
```

### Environment Configuration

Edit `.env` file with your settings:
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

### 🔧 Refine Command

**Analyze and Optimize Knowledge Graph:**
```bash
python atlas.py refine --llm-provider anthropic --model claude-sonnet-4-20250514
```

This will:
- Analyze graph structure and connectivity
- Generate optimization recommendations
- Create detailed markdown and HTML reports
- Provide iterative improvement suggestions

### 💬 Chat Command

**Interactive Chat Interface:**
```bash
python atlas.py chat --llm-provider openai --model gpt-4o
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
Please see the separate Neo4j setup guide for instructions on how to setup Neo4j locally using Docker and also how to conigure the MCP servers used by the Atlas chat. 
For detailed instructions on setting up Neo4j locally (including Docker usage and MCP server configuration), see the [Neo4j Setup Guide](./NEO4J_SETUP.md).

### Neo4j Aura (Cloud)
1. Create account at https://neo4j.com/aura/
2. Create new database
3. Copy connection details to `.env`

## 🏗️ Architecture

```
Atlas/
├── atlas.py                 # Main CLI application
├── chat_app.py             # Streamlit chat interface
├── file_analyzer.py        # Code analysis and processing
├── refine_knowledge_base.py # Graph refinement and optimization
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── .env.template          # Environment template
```

## 🔧 Key Components

- **FileAnalyzer**: Processes and analyzes code files
- **Neo4jGraphStore**: Manages knowledge graph and vector indexing
- **KnowledgeGraphRefiner**: Analyzes and optimizes graph structure
- **ReportGenerator**: Creates professional analysis reports
- **MCPToolManager**: Handles Model Context Protocol tools

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