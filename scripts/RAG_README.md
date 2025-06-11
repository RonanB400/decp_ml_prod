# RAG Query System

A Retrieval-Augmented Generation (RAG) system for querying SQL databases using natural language.

## Overview

This module has been refactored from a Jupyter notebook into a clean, reusable Python class that can be easily imported and used in other projects.

## Key Improvements

### ✅ **Modularity**
- Converted from notebook cells to a proper `RAGQuerySystem` class
- Clean separation of concerns with private methods
- Easy to import and use in other projects

### ✅ **Error Handling**
- Comprehensive error handling throughout the codebase
- Graceful fallbacks for failed operations
- Detailed logging for debugging

### ✅ **Configuration**
- Configurable database path
- Adjustable result limits (`top_k`)
- Flexible LLM model selection

### ✅ **Code Quality**
- Fixed all major linting errors
- Removed unused imports
- Proper type hints
- Clean, readable code structure

### ✅ **Usability**
- No hardcoded paths
- Proper database connection testing
- Database information retrieval methods
- Example usage scripts

## Installation

```bash
# Install required dependencies
pip install langchain langchain-community langgraph mistralai
```

## Usage

### Basic Usage

```python
from scripts.rag_query import RAGQuerySystem

# Initialize the system
rag_system = RAGQuerySystem()

# Test connection
if rag_system.test_connection():
    # Ask a question
    result = rag_system.query("How many rows are in the table?")
    print(result['answer'])
```

### Advanced Usage

```python
# Custom configuration
rag_system = RAGQuerySystem(
    db_path="/path/to/your/database.sqlite",
    llm_model="mistral-large-latest",
    top_k=10
)

# Get database information
db_info = rag_system.get_database_info()
print(f"Tables: {db_info['tables']}")

# Process multiple queries
questions = [
    "What is the total count of records?",
    "Show me the first 5 entries",
    "What columns are available?"
]

for question in questions:
    result = rag_system.query(question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"SQL: {result['query']}")
    print("-" * 40)
```

## API Reference

### RAGQuerySystem Class

#### Constructor
```python
RAGQuerySystem(db_path=None, llm_model=LLM_MODEL_NAME, top_k=10)
```

**Parameters:**
- `db_path` (optional): Path to SQLite database file
- `llm_model`: LLM model name for Mistral AI
- `top_k`: Maximum number of results to return

#### Methods

##### `query(question: str) -> Dict[str, Any]`
Process a natural language question and return results.

**Returns:**
```python
{
    "question": "Original question",
    "query": "Generated SQL query", 
    "result": "Database query result",
    "answer": "Natural language answer"
}
```

##### `test_connection() -> bool`
Test database connectivity.

##### `get_database_info() -> Dict[str, Any]`
Get database schema and table information.

## Environment Setup

1. Set your Mistral AI API key:
```bash
export MISTRAL_API_KEY="your-api-key-here"
```

2. Ensure your database file exists at the expected location or provide a custom path.

## Example Script

Run the example script to see the system in action:

```bash
python scripts/example_usage.py
```

## Error Handling

The system includes comprehensive error handling:

- **Database Connection**: Validates database file existence and connectivity
- **API Errors**: Handles LLM API failures gracefully
- **Query Errors**: Provides fallback queries for malformed SQL
- **Missing Dependencies**: Clear error messages for missing packages

## Logging

The system uses Python's logging module for debugging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Migration from Notebook

If migrating from the original notebook:

1. Replace direct function calls with class methods
2. Initialize the `RAGQuerySystem` class
3. Use the `query()` method instead of manual graph execution
4. Remove IPython-specific imports and display calls

## Dependencies

- `langchain`
- `langchain-community` 
- `langgraph`
- `mistralai`
- `typing-extensions`

## Configuration File

Ensure your `config.py` contains:

```python
LLM_MODEL_NAME = "mistral-large-latest"  # or your preferred model
``` 


```python
"""
Example usage of the RAGQuerySystem module.

This script demonstrates how to use the improved RAG Query System
for natural language database queries.
"""

import os
import sys

# Add scripts directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import after path modification
from rag_query import RAGQuerySystem


def main():
    """Demonstrate usage of RAGQuerySystem."""
    print("🚀 RAG Query System Example")
    print("=" * 40)
    
    try:
        # Initialize the system with custom parameters
        print("📡 Initializing RAG Query System...")
        rag_system = RAGQuerySystem(
            db_path=None,  # Uses default path
            top_k=5        # Limit results to 5
        )
        
        # Test database connection
        print("🔌 Testing database connection...")
        if not rag_system.test_connection():
            print("❌ Failed to connect to database")
            return
        print("✅ Database connection successful!")
        
        # Get database information
        print("\n📊 Database Information:")
        db_info = rag_system.get_database_info()
        print(f"   Dialect: {db_info['dialect']}")
        print(f"   Tables: {db_info['tables']}")
        
        # Example queries
        questions = [
            "How many rows are there in the table?",
            "What are the first 3 entries in the dataset?",
            "Show me some sample data from the table"
        ]
        
        print("\n🤖 Running example queries...")
        print("-" * 40)
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Query {i}: {question}")
            
            # Process the query
            result = rag_system.query(question)
            
            # Display results
            print(f"🔍 Generated SQL: {result.get('query', 'N/A')}")
            print(f"📊 Database Result: {result.get('result', 'N/A')}")
            print(f"💬 AI Answer: {result.get('answer', 'N/A')}")
            print("-" * 40)
        
        print("\n✨ Example completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Database file not found: {e}")
        print("💡 Make sure the database file exists at the expected location")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your API key and database configuration")

```