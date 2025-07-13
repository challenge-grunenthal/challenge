# 💊 Pharmaceutical AI Assistant

A AI-powered assistant for pharmaceutical research and analysis, built with LangGraph's ReAct framework and integrated with multiple specialized tools.

## 🚀 Features

This AI assistant provides comprehensive pharmaceutical insights through three specialized tools:

### 🔍 FDA Adverse Events Tool
- **Purpose**: Retrieves drug safety data from the FDA database
- **Functionality**: Calls the FDA API to fetch adverse event reports for specific drugs
- **Use Cases**: Drug safety analysis, adverse event monitoring, regulatory compliance

### 🧠 Neo4j Knowledge Graph Tool
- **Purpose**: Queries pharmaceutical knowledge graphs using natural language
- **Functionality**: Uses LangChain's `GraphCypherQAChain` method to interact with Neo4j database
- **Use Cases**: Drug-manufacturer relationships, pharmaceutical network analysis, knowledge discovery

### 📄 PDF RAG Tool
- **Purpose**: Performs Retrieval-Augmented Generation (RAG) on company reports
- **Functionality**: Processes and searches through the 2023-2024 company report using vector embeddings
- **Use Cases**: Document analysis, report insights, company information retrieval

## 🏗️ Architecture

The system is built using **LangGraph's ReAct framework**, which enables the AI agent to:
- **Reason** about which tools to use for each query
- **Act** by calling the appropriate tools
- **Observe** the results and provide comprehensive answers

### Core Components

- **Agent** (`agent/agent.py`): Main AI agent using ReAct pattern with LangGraph
- **Streamlit App** (`app.py`): User interface for interacting with the assistant
- **Tools**: Three specialized tools for different data sources
- **Configuration**: Secure credential management for API keys and database connections

## 🛠️ Setup and Installation


### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd grunenthal
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

Before using the assistant, you need to configure the following in the Streamlit interface:

### Required Configuration
- **OpenAI API Key**: Your OpenAI API key for AI model access
- **Neo4j Database URI**: Connection string to your Neo4j database
- **Neo4j Username**: Database username
- **Neo4j Password**: Database password

### Configuration Process
1. Open the Streamlit app in your browser
2. Fill in all configuration fields in the sidebar
3. The assistant will automatically initialize once all fields are completed
4. You can reset the configuration at any time using the "Reset Agent Configuration" button

## 💡 Usage Examples: 9 Comprehensive Examples

Here are the top 9 examples used during development and testing, each demonstrating different capabilities of the pharmaceutical AI assistant:

### 🔍 FDA Adverse Events Examples

**1. "What adverse events are reported for TRAMADOL?"**
- **Why Helpful**: Tests basic FDA API connectivity and drug name recognition. Validates that the agent can handle common pain medications and return structured adverse event data.

**2. "Show me safety data for OXYCODONE including serious adverse events"**
- **Why Helpful**: Tests filtering capabilities and specific data extraction. Ensures the agent can distinguish between different severity levels and provide focused results.

**3. "Compare adverse events between ASPIRIN and IBUPROFEN"**
- **Why Helpful**: Tests comparative analysis across multiple drugs. Validates the agent's ability to handle multi-drug queries and present comparative insights.

### 🧠 Knowledge Graph Examples

**4. "Which manufacturers are connected to drugs containing REVLIMID?"**
- **Why Helpful**: Tests Neo4j connectivity and relationship queries. Validates the agent can traverse pharmaceutical networks and identify manufacturer-drug relationships.

**5. "Find all drugs manufactured by PFIZER in the knowledge graph"**
- **Why Helpful**: Tests reverse relationship queries. Ensures the agent can work backwards from manufacturers to find associated drugs, demonstrating bidirectional graph traversal.

**6. "What are the therapeutic categories for drugs containing METFORMIN?"**
- **Why Helpful**: Tests therapeutic classification queries. Validates the agent can identify drug categories and therapeutic uses within the knowledge graph.

### 📄 Document Analysis Examples

**7. "What information can you find about Grünenthal's revenue in 2023?"**
- **Why Helpful**: Tests PDF RAG functionality with specific financial data extraction. Validates the agent can search through company reports and extract precise numerical information.

**8. "Summarize Grünenthal's research and development activities from the annual report"**
- **Why Helpful**: Tests document summarization capabilities. Ensures the agent can process large text sections and provide coherent summaries of complex topics.

**9. "What are the key strategic initiatives mentioned in the company report?"**
- **Why Helpful**: Tests strategic information extraction. Validates the agent's ability to identify and extract high-level business information from corporate documents.

## 🧹 Code Quality

This project follows strict code quality standards using:

- **Black**: Code formatting and style consistency
- **Ruff**: Fast Python linter and formatter
- **isort**: Import statement sorting and organization

### Running Code Quality Tools

```bash

black .
ruff check . --fix~
ruff format .
isort .
```

## 📁 Project Structure

```
grunenthal/
├── agent/
│   ├── __init__.py
│   └── agent.py              # Main ReAct agent implementation
├── tools/
│   ├── __init__.py
│   ├── fda_tool.py           # FDA API integration
│   ├── neo4j_tool.py         # Neo4j knowledge graph queries
│   ├── pdf_rag_tool.py       # PDF RAG implementation
│   └── pdf_data/             # PDF documents directory
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 📋 Project Scope & Limitations

- **Single interaction only** (no chat history maintained)
- **Limited to 2023-2024 report** (could be extended to multiple pdfs or with web-search as suggested)
- **Basic Neo4j integration** using LangChain's pre-built GraphCypherQAChain (no time to dive in the KG schema)
- **Core FDA API functionality** with essential adverse event filtering (without subject matter experise)
- **9 example prompts** for testing (not comprehensive coverage)
- **Time-limited resources**: Neo4j access (until Tuesday morning at 11am) and OpenAI API credits (<5 dollars)