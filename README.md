# Atlas 🗺️

**Agentic AI Application Landscape Discovery**

An intelligent system for analyzing, understanding, and discovering complex application landscapes through AI-powered analysis, knowledge graph generation, and document vector indexing.

## ✨ Features

### 🔍 **Code Analysis**
- **Multi-file Processing**: Analyze entire codebases or individual files
- **Intelligent Filtering**: Support for file type filters and batch processing
- **Metadata Extraction**: Comprehensive file analysis with statistics

### 🧠 **Knowledge Graph Generation**
- **Neo4j Integration**: Create rich knowledge graphs from code analysis
- **Entity Relationships**: Automatic extraction of functions, classes, and dependencies
- **Real-time Statistics**: Track nodes, relationships, and processing progress

### 📄 **Document Vector Indexing**
- **Semantic Search**: Vector embeddings for intelligent document search
- **Multi-provider Support**: OpenAI, Anthropic, and Ollama embeddings
- **Chunking & Indexing**: Smart document chunking with metadata preservation
- **Neo4j Vector Search**: Integrated vector and full-text search capabilities

### 🔧 **Knowledge Graph Refinement**
- **Graph Analysis**: Comprehensive analysis of graph structure and connectivity
- **LLM Optimization**: AI-powered recommendations for graph improvements
- **Professional Reports**: Generate markdown and HTML reports with visualizations
- **Iterative Improvement**: Version-controlled analysis reports

### 💬 **Interactive Chat Interface**
- **MCP Tool Integration**: Model Context Protocol for enhanced capabilities
- **Multi-provider Support**: OpenAI, Anthropic, and Ollama
- **Streamlit UI**: Modern web interface for conversations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Neo4j Database (local or cloud)
- API keys for your chosen LLM provider

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/atlas.git
cd atlas
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
```bash
# Using Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Or install locally
# Visit: https://neo4j.com/download/
```

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
- Supports [OpenAI](https://openai.com/), [Anthropic](https://anthropic.com/), and [Ollama](https://ollama.ai/)

---

**Atlas** - Navigate the complexity of your application ecosystem with AI-powered analysis 🚀