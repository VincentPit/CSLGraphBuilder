# GraphBuilder - Enterprise Knowledge Graph Builder 🚀

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/neo4j-5.21%2B-green.svg)](https://neo4j.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **🎉 TRANSFORMATION COMPLETE: GraphBuilder has been fully transformed into an enterprise-grade architecture with clean project organization, Domain-Driven Design, sophisticated CLI, and advanced configuration management!**
>
> **📁 CLEAN STRUCTURE: All legacy files have been organized into the `legacy/` directory, leaving a clean, production-ready project structure.**

**Transform unstructured content into sophisticated knowledge graphs with enterprise-grade AI-powered extraction capabilities.**

Build intelligent knowledge graphs from diverse content sources using state-of-the-art LLM models and Neo4j database. Now featuring enterprise architecture, advanced processing pipelines, and comprehensive tooling for production deployments.

**Languages**: [English](README.md) | [中文](README_ZH.md)

## 🆕 What's New in v2.0 Enterprise

### **�️ Enterprise Architecture**
- **Domain-Driven Design**: Sophisticated layered architecture with clean boundaries
- **Repository Pattern**: Advanced data access with Neo4j and in-memory implementations  
- **Use Cases**: Business logic encapsulation with comprehensive error handling
- **Dependency Injection**: Configurable service composition with factory patterns

### **🎛️ Advanced CLI Interface**
- **Rich Console Output**: Beautiful tables, progress bars, and formatted displays
- **Batch Processing**: Efficient multi-document processing with concurrency control
- **Configuration Commands**: Interactive configuration management and validation
- **Comprehensive Help**: Detailed command documentation and examples

### **⚙️ Sophisticated Configuration**
- **Environment-based**: YAML, JSON, and .env file support with validation
- **Runtime Updates**: Dynamic configuration changes without restart
- **Profile Management**: Multiple environments (dev, staging, production)
- **Security**: Encrypted credentials and API key management

### **🔄 Advanced Processing Pipeline**
- **Async Architecture**: High-performance asynchronous processing
- **Task Orchestration**: Sophisticated workflow management with dependencies
- **Progress Tracking**: Real-time status updates with detailed metrics
- **Error Recovery**: Intelligent retry mechanisms with exponential backoff

## ✨ Core Features

### **🧠 AI-Powered Extraction**
- **Multi-LLM Support**: OpenAI GPT, Azure OpenAI, and custom models
- **Entity Recognition**: Advanced NER with confidence scoring and validation
- **Relationship Mapping**: Contextual relationship extraction with type inference
- **Content Understanding**: Semantic analysis with domain-specific processing

### **🌐 Content Processing**
- **Multi-format Support**: HTML, PDF, JSON, XML, CSV, Markdown, and plain text
- **Smart Chunking**: Adaptive text segmentation with context preservation
- **Parallel Crawling**: Concurrent web scraping with rate limiting
- **Content Validation**: Sophisticated input validation and sanitization

### **📊 Graph Operations**
- **Neo4j Integration**: Direct database integration with connection pooling
- **Entity Deduplication**: Intelligent similarity matching and merging
- **Relationship Optimization**: Graph structure analysis and enhancement
- **Query Generation**: Advanced Cypher query construction and optimization

### **🛡️ Enterprise Features**
- **Audit Logging**: Comprehensive activity tracking and compliance reporting
- **Security**: API rate limiting, input validation, and data encryption
- **Monitoring**: Performance metrics, health checks, and alerting
- **Scalability**: Horizontal scaling with load balancing support

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)  
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Enterprise Architecture](#-enterprise-architecture)
- [CLI Reference](#-cli-reference)
- [API Documentation](#-api-documentation)
- [Examples](#-examples)
- [Migration Guide](#-migration-guide)
- [Contributing](#-contributing)
- [License](#-license)

## � Clean Project Structure

After the enterprise transformation and cleanup, GraphBuilder now has a clean, organized structure:

```
GraphBuilder/
├── 📁 src/graphbuilder/          # 🚀 Enterprise package (Domain-Driven Design)
│   ├── cli/                      # Command-line interface
│   ├── core/                     # Core functionality and interfaces
│   ├── domain/                   # Business domain models and services
│   ├── application/              # Use cases and orchestration
│   └── infrastructure/           # Repositories, services, configuration
├── 📁 legacy/                    # 📦 Original files (preserved for reference)
├── 📁 tests/                     # Test suites (unit, integration, e2e)
├── 📁 docs/                      # Documentation 
├── 📁 examples/                  # Usage examples
├── 📁 ImageEmbed/                # Image processing modules
├── 📄 setup.py                  # Package installation
├── 📄 pyproject.toml            # Modern Python packaging
├── 📄 requirements.txt          # Dependencies
├── 📄 migrate.py                # Migration utilities
└── 📄 README.md                 # This file
```

**✨ Clean & Organized**: Legacy files safely moved to `legacy/` directory for reference while maintaining a clean production structure.

## �🚀 Quick Start

### **Option 1: Package Installation (Recommended)**

```bash
# Install as a package
pip install -e .

# Run CLI directly
graphbuilder --help

# Or run as module
python -m graphbuilder --help
```

### **Option 2: Development Setup**

```bash
# 1. Clone the repository
git clone https://github.com/VincentPit/GraphBuilder.git
cd GraphBuilder

# 2. Set up environment
conda env create -f environment.yml
conda activate graph

# 3. Install in development mode
pip install -e .
pip install -r condaenv.rnehk3uw.requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your settings

# 4. Test the setup
python demo.py

# 5. Process your first URL
python cli.py url https://www.dfrobot.com
```

## 📦 Installation

### **✅ Prerequisites**

- **Python 3.8+** with pip and virtual environment support
- **Neo4j 5.21+** with APOC plugin enabled
- **LLM API Access**: OpenAI API key or Azure OpenAI endpoint
- **System Requirements**: Minimum 8GB RAM (16GB recommended)

### **Method 1: Production Installation**

```bash
# Install from PyPI (when published)
pip install graphbuilder

# Or install from source
pip install git+https://github.com/VincentPit/GraphBuilder.git
```

### **Method 2: Development Installation**

```bash
# Clone repository
git clone https://github.com/VincentPit/GraphBuilder.git
cd GraphBuilder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .[dev]
```

### **Method 3: Using Conda**

```bash
# Create conda environment from file
conda env create -f environment.yml
conda activate graph

# Install additional dependencies
pip install -e .[dev]
```

## ⚙️ Configuration

### **Environment Setup**

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration file
nano .env  # or your preferred editor
```

### **Essential Configuration**

```env
# Database Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# LLM Configuration
OPENAI_API_KEY=your_api_key
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-3.5-turbo

# Processing Settings
CHUNK_SIZE=1000
MAX_CONCURRENT_TASKS=5
```

### **Neo4j Database Setup**

**Option A: Docker (Recommended)**
```bash
docker run \
    --name neo4j-graphbuilder \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v neo4j_data:/data \
    -v neo4j_logs:/logs \
    -v neo4j_import:/var/lib/neo4j/import \
    -v neo4j_plugins:/plugins \
    --env NEO4J_AUTH=neo4j/your_password \
    --env NEO4J_PLUGINS='["apoc"]' \
    --env NEO4J_dbms_security_procedures_unrestricted=apoc.* \
    --env NEO4J_dbms_security_procedures_allowlist=apoc.* \
    neo4j:5.21
```

**Option B: System Installation**
```bash
# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 5' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j=1:5.21.0

# Configure APOC plugin
sudo wget -P /var/lib/neo4j/plugins/ \
    https://github.com/neo4j/apoc/releases/download/5.21.0/apoc-5.21.0-core.jar

# Edit configuration
sudo nano /etc/neo4j/neo4j.conf
```

**Neo4j Configuration (neo4j.conf)**:
```bash
# Enable APOC procedures
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

# Memory settings (adjust based on your system)
server.memory.heap.initial_size=1G
server.memory.heap.max_size=2G
server.memory.pagecache.size=1G
```

### **Advanced Configuration**

GraphBuilder supports sophisticated configuration through multiple sources:

- **Environment Variables**: `.env` file or system environment
- **YAML Configuration**: `config/settings.yaml`
- **Runtime Configuration**: Dynamic updates via API
- **Profile-based**: Development, staging, production profiles

**Sample YAML Configuration** (`config/settings.yaml`):
```yaml
database:
  uri: neo4j://localhost:7687
  username: neo4j
  password: ${NEO4J_PASSWORD}
  pool_size: 50

llm:
  provider: openai
  model_name: gpt-3.5-turbo
  temperature: 0.1
  max_tokens: 2000

processing:
  chunk_size: 1000
  overlap_size: 100
  max_concurrent_tasks: 5
  batch_size: 10
```

## 📖 Usage

### **Command Line Interface**

```bash
# Show all available commands
graphbuilder --help

# Process a single URL
graphbuilder process --url "https://example.com"

# Process a local file
graphbuilder process --file "document.pdf"

# Process with custom extraction settings
graphbuilder process --url "https://example.com" \
  --extract-entities --extract-relationships \
  --output results.json
```

### **Batch Processing**

```bash
# Process multiple documents
graphbuilder batch --input-dir ./documents --pattern "*.pdf"

# Parallel processing with concurrency control
graphbuilder batch --input-dir ./docs \
  --max-concurrent 5 \
  --output ./results

# Process with specific extraction options
graphbuilder batch --input-dir ./content \
  --extract-entities --extract-relationships \
  --deduplicate --optimize
```

### **Graph Management**

```bash
# List processed documents
graphbuilder list-documents --status completed --limit 50

# Show document details
graphbuilder list-documents --document-id doc123 --detailed

# Optimize knowledge graph
graphbuilder optimize --deduplicate --analyze-structure

# Show configuration info
graphbuilder config-info
```

### **Python API Usage**

**Basic Document Processing:**

```python
import asyncio
from graphbuilder.infrastructure.config.settings import load_config
from graphbuilder.application.use_cases.document_processing import ProcessDocumentUseCase

async def process_document_example():
    # Load configuration
    config = load_config()
    
    # Initialize services (use dependency injection in practice)
    # doc_repo, graph_repo, llm_service, extractor = setup_services(config)
    
    # Process document
    use_case = ProcessDocumentUseCase(config, doc_repo, graph_repo, llm_service, extractor)
    result = await use_case.execute(document_id="example-doc")
    
    if result.success:
        print(f"Extracted {result.data['extracted_entities']} entities")
        print(f"Extracted {result.data['extracted_relationships']} relationships")
    else:
        print(f"Processing failed: {result.error}")

# Run async function
asyncio.run(process_document_example())
```

**Batch Processing:**

```python
from graphbuilder.application.use_cases.batch_processing import BatchProcessDocumentsUseCase

async def batch_process_example():
    config = load_config()
    use_case = BatchProcessDocumentsUseCase(config, services)
    
    # Define processing options
    options = ProcessingOptions(
        extract_entities=True,
        extract_relationships=True,
        deduplicate=True,
        max_concurrent_tasks=5
    )
    
    # Process batch
    result = await use_case.execute(
        document_ids=["doc1", "doc2", "doc3"],
        options=options
    )
    
    print(f"Successfully processed: {result.data['successful_count']}")
    print(f"Failed: {result.data['failed_count']}")
```

## 🏗️ Enterprise Architecture

### **Domain-Driven Design Structure**

GraphBuilder follows clean architecture principles with sophisticated separation of concerns:

```
src/graphbuilder/
├── core/                    # Enterprise core functionality
│   ├── exceptions/         # Custom exception hierarchy
│   ├── interfaces/         # Abstract base classes and protocols
│   └── logging/           # Structured logging and audit trails
├── domain/                 # Business domain layer
│   ├── models/            # Rich domain entities and value objects
│   │   ├── graph.py      # Graph entities and relationships
│   │   ├── document.py   # Document processing models
│   │   └── processing.py # Processing workflow models
│   └── services/          # Domain service contracts
├── application/           # Application orchestration layer
│   ├── use_cases/        # Business use case implementations
│   │   ├── document_processing.py
│   │   ├── batch_processing.py
│   │   └── graph_optimization.py
│   └── services/         # Application service implementations
├── infrastructure/       # External concerns and technical details
│   ├── repositories/     # Data persistence (Neo4j, file system)
│   ├── services/         # External service integrations (LLM, web)
│   ├── config/          # Configuration management and validation
│   └── adapters/        # External API adapters and clients
└── cli/                 # Command-line interface and entry points
```

### **Key Architectural Principles**

- **Dependency Inversion**: High-level modules independent of low-level details
- **Single Responsibility**: Each module has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Interface Segregation**: Focused, cohesive interfaces
- **Async/Await**: Non-blocking I/O throughout the stack

### **Configuration Management**

**Hierarchical Configuration Loading**:
1. **Default Settings**: Built-in sensible defaults
2. **Environment Files**: `.env` files for deployment-specific settings
3. **YAML Configuration**: `config/settings.yaml` for complex configurations
4. **Environment Variables**: Runtime overrides
5. **CLI Arguments**: Command-line parameter overrides

**Configuration Validation**:
- **Pydantic Models**: Type-safe configuration with automatic validation
- **Environment Detection**: Automatic development/staging/production profile selection
- **Secrets Management**: Secure handling of API keys and passwords

## 🎛️ CLI Reference

### **Core Commands**

#### `graphbuilder process`
Process a single document with advanced extraction options.

```bash
graphbuilder process [OPTIONS] SOURCE

Options:
  --url TEXT              Process web page by URL
  --file PATH             Process local file
  --extract-entities      Extract named entities
  --extract-relationships Extract entity relationships
  --output PATH          Save results to file
  --format [json|csv]    Output format (default: json)
  --chunk-size INT       Text chunking size (default: 1000)
  --verbose              Enable verbose logging
  --help                 Show help message
```

#### `graphbuilder batch`
Process multiple documents in parallel with queue management.

```bash
graphbuilder batch [OPTIONS] INPUT_DIR

Options:
  --input-dir PATH        Input directory path
  --pattern TEXT         File pattern (e.g., "*.pdf", "*.html")
  --output PATH          Output directory
  --max-concurrent INT   Maximum concurrent tasks (default: 5)
  --extract-entities     Extract entities for all documents
  --extract-relationships Extract relationships for all documents
  --deduplicate          Remove duplicate entities
  --optimize             Optimize graph structure
  --resume               Resume interrupted batch processing
  --status-file PATH     Custom status tracking file
  --help                 Show help message
```

#### `graphbuilder list-documents`
List and query processed documents with filtering.

```bash
graphbuilder list-documents [OPTIONS]

Options:
  --status [pending|processing|completed|failed]  Filter by status
  --document-id TEXT     Show specific document details
  --detailed             Show detailed information
  --limit INT           Maximum results (default: 20)
  --offset INT          Results offset for pagination
  --format [table|json] Output format (default: table)
  --help                Show help message
```

#### `graphbuilder optimize`
Optimize knowledge graph structure and performance.

```bash
graphbuilder optimize [OPTIONS]

Options:
  --deduplicate         Remove duplicate entities and relationships
  --analyze-structure   Analyze and report graph structure metrics
  --merge-similar      Merge similar entities (similarity threshold)
  --threshold FLOAT    Similarity threshold for merging (0.0-1.0)
  --dry-run            Show what would be optimized without changes
  --backup             Create backup before optimization
  --help               Show help message
```

#### `graphbuilder config-info`
Display current configuration and system information.

```bash
graphbuilder config-info [OPTIONS]

Options:
  --show-secrets        Show masked secret values (for debugging)
  --validate           Validate current configuration
  --format [table|json] Output format (default: table)
  --help               Show help message
```

### **Global Options**

Available for all commands:

- `--config PATH`: Specify custom configuration file
- `--profile TEXT`: Use specific configuration profile
- `--log-level [DEBUG|INFO|WARNING|ERROR]`: Set logging level
- `--no-color`: Disable colored output
- `--quiet`: Suppress all output except errors

## 💡 Examples

### **Example 1: Research Paper Processing**

Extract entities and relationships from academic papers:

```bash
# Process a directory of PDF research papers
graphbuilder batch --input-dir ./research_papers --pattern "*.pdf" \
  --extract-entities --extract-relationships \
  --max-concurrent 3 \
  --output ./results

# Optimize the resulting knowledge graph
graphbuilder optimize --deduplicate --merge-similar --threshold 0.85
```

### **Example 2: Web Content Analysis**

Build a knowledge graph from web pages:

```bash
# Process web content with entity extraction
graphbuilder process --url "https://en.wikipedia.org/wiki/Artificial_intelligence" \
  --extract-entities --extract-relationships \
  --output ai_knowledge.json

# Process multiple related pages (requires URL list file)
cat urls.txt | while read url; do
  graphbuilder process --url "$url" --extract-entities --extract-relationships
done
```

### **Example 3: Enterprise Document Pipeline**

Set up a production processing pipeline:

```bash
# 1. Configure for production environment
export GRAPHBUILDER_PROFILE=production
export NEO4J_URI=neo4j+s://your-cluster.databases.neo4j.io:7687

# 2. Process large document collections
graphbuilder batch --input-dir /data/documents \
  --pattern "*" \
  --max-concurrent 10 \
  --extract-entities --extract-relationships \
  --deduplicate --optimize \
  --status-file /logs/processing_status.json

# 3. Monitor and analyze results
graphbuilder list-documents --status completed --detailed
graphbuilder optimize --analyze-structure
```

### **Example 4: Integration with Python Applications**

```python
import asyncio
from pathlib import Path
from graphbuilder.infrastructure.config.settings import load_config
from graphbuilder.application.use_cases.batch_processing import BatchProcessDocumentsUseCase
from graphbuilder.domain.models.processing import ProcessingOptions

async def process_company_docs():
    """Process company documentation and build knowledge graph."""
    
    # Load configuration
    config = load_config(profile="production")
    
    # Setup services (in practice, use dependency injection)
    services = await setup_services(config)
    
    # Define processing pipeline
    use_case = BatchProcessDocumentsUseCase(config, services)
    
    # Configure processing options
    options = ProcessingOptions(
        extract_entities=True,
        extract_relationships=True,
        deduplicate=True,
        max_concurrent_tasks=5,
        chunk_size=1200,
        similarity_threshold=0.8
    )
    
    # Process documents
    document_paths = Path("./company_docs").glob("**/*")
    document_ids = [str(path) for path in document_paths if path.is_file()]
    
    result = await use_case.execute(document_ids, options)
    
    print(f"Processing completed:")
    print(f"  Successful: {result.data['successful_count']}")
    print(f"  Failed: {result.data['failed_count']}")
    print(f"  Total entities: {result.data['total_entities']}")
    print(f"  Total relationships: {result.data['total_relationships']}")
    
    return result

# Run the processing pipeline
asyncio.run(process_company_docs())
```

## 🔄 Migration Guide

### **From Legacy Scripts to Enterprise Architecture**

If you're migrating from the original script-based structure, use the provided migration script:

```bash
# Run the automated migration script
python migrate.py

# The script will:
# 1. Backup original files to backup_YYYYMMDD_HHMMSS/
# 2. Move files to new enterprise structure
# 3. Update import statements automatically
# 4. Generate migration report
```

### **Manual Migration Steps**

If you need to migrate manually or have custom modifications:

#### 1. **Update Import Statements**

**Before (Legacy)**:
```python
from dbAccess import Neo4jConnection
from processing import process_documents
from llm import get_llm_response
```

**After (Enterprise)**:
```python
from graphbuilder.infrastructure.repositories.neo4j_repository import Neo4jDocumentRepository
from graphbuilder.application.use_cases.document_processing import ProcessDocumentUseCase
from graphbuilder.infrastructure.services.llm_service import LLMService
```

#### 2. **Update Configuration**

**Before (Legacy)**:
```python
# Hardcoded configuration
NEO4J_URI = "bolt://localhost:7687"
OPENAI_API_KEY = "your-key"
```

**After (Enterprise)**:
```python
from graphbuilder.infrastructure.config.settings import load_config

config = load_config()  # Automatically loads from environment and config files
```

#### 3. **Update Processing Logic**

**Before (Legacy)**:
```python
def process_url(url):
    content = fetch_content(url)
    entities = extract_entities(content)
    save_to_neo4j(entities)
```

**After (Enterprise)**:
```python
async def process_url(url: str):
    use_case = ProcessDocumentUseCase(config, services)
    result = await use_case.execute(document_id=url)
    return result
```

### **Configuration Migration**

Create new configuration files based on your legacy settings:

**Create `.env`**:
```env
# Convert your old hardcoded values
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-3.5-turbo
```

**Create `config/settings.yaml`** (optional):
```yaml
database:
  uri: ${NEO4J_URI}
  username: ${NEO4J_USERNAME}
  password: ${NEO4J_PASSWORD}

processing:
  chunk_size: 1000
  max_concurrent_tasks: 5
```

### **Testing Migration**

After migration, test the new system:

```bash
# Test CLI functionality
graphbuilder config-info

# Test basic processing
graphbuilder process --url "https://example.com" --extract-entities

# Test Python API
python -c "
import asyncio
from graphbuilder.infrastructure.config.settings import load_config
config = load_config()
print('Configuration loaded successfully')
print(f'Neo4j URI: {config.database.uri}')
"
```

## 🔍 Troubleshooting

### **Common Issues**

#### **Database Connection Failed**
```bash
# Check Neo4j status
docker ps | grep neo4j
# or
systemctl status neo4j

# Test connection
graphbuilder config-info --validate
```

#### **API Rate Limiting**
```bash
# Adjust rate limiting in configuration
export LLM_RATE_LIMIT=10  # requests per minute
export MAX_CONCURRENT_TASKS=2
```

#### **Memory Issues with Large Documents**
```bash
# Reduce chunk size for large documents
graphbuilder process --file large_doc.pdf --chunk-size 500

# Process in smaller batches
graphbuilder batch --input-dir ./docs --max-concurrent 2
```

#### **Import Errors After Migration**
```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import graphbuilder; print(graphbuilder.__file__)"
```

### **Performance Optimization**

#### **Neo4j Tuning**
```bash
# Increase memory allocation
server.memory.heap.max_size=4G
server.memory.pagecache.size=2G

# Enable parallelization
dbms.transaction.timeout=300s
dbms.cypher.parallel_runtime_support=all
```

#### **Processing Optimization**
```bash
# Optimize for throughput
export CHUNK_SIZE=1500
export MAX_CONCURRENT_TASKS=8
export BATCH_SIZE=20

# Optimize for accuracy
export CHUNK_SIZE=800
export MAX_CONCURRENT_TASKS=3
export LLM_TEMPERATURE=0.1
```

## 🤝 Contributing

We welcome contributions from the community! Here's how to get involved:

### **Quick Contribution Guide**

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes and add tests
4. **Commit** changes: `git commit -m 'Add amazing feature'`
5. **Push** to branch: `git push origin feature/amazing-feature`
6. **Open** a Pull Request

### **Development Setup**
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/GraphBuilder.git
cd GraphBuilder

# Install development dependencies
pip install -e .[dev]
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Code formatting and linting
black . && isort .
flake8 src/
mypy src/

# Run full test suite
tox
```

### **Contribution Guidelines**

- **Code Style**: Follow PEP 8, use Black and isort for formatting
- **Testing**: Write tests for new features, maintain >90% coverage
- **Documentation**: Update docstrings and README for new features
- **Type Hints**: Use comprehensive type annotations
- **Commits**: Use conventional commit messages

### **Areas for Contribution**

- 🐛 **Bug Reports**: Report issues with detailed reproduction steps
- ✨ **Feature Requests**: Suggest new capabilities and improvements
- 📚 **Documentation**: Improve guides, examples, and API docs
- 🧪 **Testing**: Add test cases and improve test coverage
- 🔧 **Performance**: Optimize processing pipelines and database queries
- 🌐 **Integrations**: Add support for new data sources and LLM providers
"
```

# Crawler Settings
MAX_CRAWL_LIMIT=10
MAX_WORKERS=5
ALLOWED_DOMAINS=dfrobot

# Processing Settings
CHUNK_SIZE=200
CHUNK_OVERLAP=20
```

#### 4. Verify Installation

```bash
# Test database connection
python cli.py config

# Run demo to verify all components
python demo.py

# Test with a simple URL
python cli.py url https://www.dfrobot.com --output test_results.json
```

## 📊 Configuration

GraphBuilder uses environment variables for configuration. All settings can be customized in your `.env` file:

### Database Settings
```bash
NEO4J_URI=bolt://localhost:7687      # Neo4j connection URI
NEO4J_USER=neo4j                     # Database username
NEO4J_PASSWORD=your_password         # Database password
NEO4J_DATABASE=neo4j                 # Database name
```

### LLM Settings
```bash
LLM_MODEL_NAME=azure_ai_gpt_4o       # Model identifier
# Azure OpenAI format: model,endpoint,key,version
LLM_MODEL_CONFIG_azure_ai_gpt_4o=gpt-4o,https://your-endpoint.openai.azure.com/,your-api-key,2024-02-01
```

### Crawler Settings
```bash
MAX_CRAWL_LIMIT=10                   # Maximum URLs to process
MAX_WORKERS=5                        # Concurrent worker threads
CRAWL_DELAY=1                        # Delay between requests (seconds)
ALLOWED_DOMAINS=dfrobot              # Comma-separated allowed domains
```

### Processing Settings
```bash
CHUNK_SIZE=200                       # Text chunk size (tokens)
CHUNK_OVERLAP=20                     # Overlap between chunks
MAX_CHUNKS_ALLOWED=1000              # Maximum chunks per document
```

## 🎯 Usage

GraphBuilder provides multiple interfaces for different use cases:

### 1. Command Line Interface (Recommended)

#### Process Single URL
```bash
# Basic URL processing
python cli.py url https://www.dfrobot.com

# With custom parameters
python cli.py url https://www.dfrobot.com \
    --allowed-nodes "Product,Company,Technology" \
    --allowed-relationships "MANUFACTURES,DEVELOPS" \
    --model azure_ai_gpt_4o \
    --output results.json
```

#### Crawl Multiple URLs
```bash
# Crawl starting from multiple URLs
python cli.py crawl https://www.dfrobot.com https://www.dfrobot.com/blog \
    --max-urls 20 \
    --max-workers 3

# Advanced crawling with filtering
python cli.py crawl https://www.dfrobot.com \
    --max-urls 50 \
    --allowed-nodes "Product,Company,Person,Technology" \
    --allowed-relationships "MANUFACTURES,DEVELOPS,WORKS_FOR"
```

#### Process JSON Data
```bash
# Process from JSON file
python cli.py json data/sample.json

# Example JSON format:
cat > sample.json << EOF
{
  "urls": ["https://www.dfrobot.com/product-1.html"],
  "allowed_nodes": ["Product", "Company", "Technology"],
  "allowed_relationships": ["MANUFACTURES", "DEVELOPS"],
  "model": "azure_ai_gpt_4o"
}
EOF
```

#### Monitor Status
```bash
# Check overall status
python cli.py status

# Check specific document status
python cli.py status --file-name "https://www.dfrobot.com"

# Reset crawler state
python cli.py reset-crawler

# View configuration
python cli.py config
```

### 2. Programmatic Usage

```python
from app import app
from logger_config import setup_logging

# Setup logging
setup_logging(log_level="INFO")

# Process single URL
result = app.process_single_url(
    url="https://www.dfrobot.com/product-arduino-uno.html",
    allowed_nodes=["Product", "Company", "Technology", "Feature"],
    allowed_relationships=["MANUFACTURES", "HAS_FEATURE", "COMPATIBLE_WITH"]
)

print(f"Processing result: {result}")

# Crawl multiple URLs
result = app.crawl_and_process_urls(
    start_urls=["https://www.dfrobot.com"],
    allowed_nodes=["Product", "Company"],
    allowed_relationships=["MANUFACTURES"],
    max_workers=3
)

# Process JSON data
json_data = {
    "urls": ["https://www.dfrobot.com/category-arduino.html"],
    "allowed_nodes": ["Product", "Category", "Brand"],
    "allowed_relationships": ["BELONGS_TO", "MANUFACTURED_BY"]
}
result = app.process_from_json_data(json_data)

# Always shutdown gracefully
app.shutdown()
```

### 3. Legacy Scripts (Backward Compatibility)

For users familiar with the original interface:

```bash
# Improved main scripts
python main_improved.py           # Enhanced URL processing
python main_json_improved.py      # Enhanced JSON processing

# Original scripts (still supported)
python main_url.py               # Original URL processing
python main_json.py              # Original JSON processing
python main_para.py              # Parallel processing
```

## 🏗️ Architecture

GraphBuilder follows a modern layered architecture:

```
┌─────────────────────────────────────────────────────┐
│                CLI Interface                        │
├─────────────────────────────────────────────────────┤
│              Application Layer                      │
│                   (app.py)                          │
├─────────────────────────────────────────────────────┤
│                Services Layer                       │
│  ┌─────────────┬─────────────┬─────────────────────┐ │
│  │ LLM Service │ Crawler     │ Document Processor  │ │
│  │             │ Service     │                     │ │
│  └─────────────┴─────────────┴─────────────────────┘ │
├─────────────────────────────────────────────────────┤
│              Database Layer                         │
│  ┌─────────────────────┬─────────────────────────┐   │
│  │ Connection Manager  │ Repositories            │   │
│  └─────────────────────┴─────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│              Data Models                            │
│  ┌─────────────┬─────────────┬─────────────────────┐ │
│  │ SourceNode  │ ChunkDoc    │ Entities            │ │
│  └─────────────┴─────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Core Components

- **Application Layer**: Orchestrates all operations
- **Services Layer**: Business logic for different operations  
- **Database Layer**: Data persistence and retrieval
- **Data Models**: Type-safe entity definitions
- **Configuration**: Environment-based settings
- **Utilities**: Validation, helpers, and common functions

## 📚 API Reference

### Main Application (`app.py`)

```python
class GraphBuilderApp:
    def process_single_url(url: str, allowed_nodes: List[str] = None, 
                          allowed_relationships: List[str] = None) -> Dict
    def crawl_and_process_urls(start_urls: List[str], **kwargs) -> Dict
    def process_from_json_data(json_data: Dict) -> Dict
    def get_processing_status(file_name: str = None) -> Dict
    def reset_crawler() -> Dict
```

### Services

```python
# LLM Service
llm_service.generate_graph_documents(chunk_documents, allowed_nodes, allowed_relationships)

# Crawler Service  
crawler_service.crawl_urls_parallel(start_urls, process_callback, max_workers)

# Document Service
document_processor.process_url_to_graph(url, allowed_nodes, allowed_relationships)
```

## 💡 Examples

### Example 1: E-commerce Product Analysis
```bash
python cli.py url https://www.dfrobot.com/product-arduino-uno.html \
    --allowed-nodes "Product,Brand,Category,Feature,Price,Specification" \
    --allowed-relationships "MANUFACTURED_BY,BELONGS_TO,HAS_FEATURE,PRICED_AT" \
    --output product_analysis.json
```

### Example 2: Company Knowledge Base
```bash
python cli.py crawl https://www.dfrobot.com \
    --max-urls 100 \
    --allowed-nodes "Company,Product,Technology,Person,Location,Event" \
    --allowed-relationships "DEVELOPS,MANUFACTURES,LOCATED_IN,PARTICIPATES_IN"
```

### Example 3: Technical Documentation
```python
result = app.process_single_url(
    url="https://www.dfrobot.com/wiki/arduino-tutorials",
    allowed_nodes=["Tutorial", "Component", "Code", "Concept"],
    allowed_relationships=["TEACHES", "USES", "IMPLEMENTS", "EXPLAINS"]
)
```

## 🧪 Advanced Features

### Visual Embeddings (Experimental)

GraphBuilder supports experimental visual embeddings using CLIP:

```python
# Enable image processing
from ImageEmbed.clip import process_images

# Process images from web pages
images = process_images(url="https://www.dfrobot.com/products")
```

**Note**: Visual embeddings require switching to CLIP's text model for consistency.

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check Neo4j status
   sudo systemctl status neo4j
   
   # Verify connection settings
   python cli.py config
   ```

2. **LLM API Errors**
   ```bash
   # Verify API configuration
   echo $LLM_MODEL_CONFIG_azure_ai_gpt_4o
   
   # Test with minimal example
   python demo.py
   ```

3. **Memory Issues**
   ```bash
   # Reduce chunk size and worker count
   export CHUNK_SIZE=100
   export MAX_WORKERS=2
   ```

4. **Permission Errors**
   ```bash
   # Check file permissions
   chmod +x *.py
   
   # Verify log directory
   mkdir -p logs
   ```

### Debug Mode

Enable detailed logging:
```bash
python cli.py url https://example.com --log-level DEBUG
```

Check logs:
```bash
tail -f logs/graphbuilder.log
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black . && isort .

# Type checking
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DFRobot Co.** - Original project sponsor
- **llm-graph-builder** - Inspiration and foundation
- **Neo4j Community** - Graph database platform
- **LangChain** - LLM integration framework
- **Contributors** - All who helped improve this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/VincentPit/GraphBuilder/issues)
- **Discussions**: [GitHub Discussions](https://github.com/VincentPit/GraphBuilder/discussions)
- **Documentation**: [Wiki](https://github.com/VincentPit/GraphBuilder/wiki)

---

**Made with ❤️ for the open source community**