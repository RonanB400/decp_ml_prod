import os
import getpass
from typing import Optional, Dict, Any
import logging

from config import LLM_MODEL_NAME, TEMPERATURE

from google.cloud import bigquery
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict, Annotated
from langgraph.graph import START, StateGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery configuration
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_TABLE = os.environ.get("BQ_TABLE")


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


class RAGQuerySystem:
    """A RAG system for querying BigQuery databases using natural language."""
    
    def __init__(self, 
                 gcp_project: Optional[str] = None,
                 bq_dataset: Optional[str] = None,
                 bq_table: Optional[str] = None,
                 llm_model: str = LLM_MODEL_NAME,
                 top_k: int = 10):
        """
        Initialize the RAG Query System for BigQuery.
        
        Args:
            gcp_project: GCP Project ID
            bq_dataset: BigQuery dataset name
            bq_table: BigQuery table name
            llm_model: Name of the LLM model to use
            top_k: Maximum number of results to return
        """
        self.llm_model = llm_model
        self.top_k = top_k
        self.gcp_project = gcp_project or GCP_PROJECT
        self.bq_dataset = bq_dataset or BQ_DATASET
        self.bq_table = bq_table or BQ_TABLE
        self._setup_bigquery()
        self._setup_llm()
        self._setup_prompts()
        self._setup_graph()
    
    def _setup_bigquery(self) -> None:
        """Setup BigQuery client and validate configuration."""
        if not all([self.gcp_project, self.bq_dataset, self.bq_table]):
            missing = []
            if not self.gcp_project:
                missing.append("GCP_PROJECT")
            if not self.bq_dataset:
                missing.append("BQ_DATASET")
            if not self.bq_table:
                missing.append("BQ_TABLE")
            raise ValueError(
                f"Missing required environment variables: {missing}")
        
        try:
            self.client = bigquery.Client(project=self.gcp_project)
            self.table_id = f"{self.gcp_project}.{self.bq_dataset}.{self.bq_table}"
            logger.info(f"Connected to BigQuery project: {self.gcp_project}")
            logger.info(f"Target table: {self.table_id}")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
    def _setup_llm(self) -> None:
        """Setup the language model."""
        if not os.environ.get("MISTRAL_API_KEY"):
            api_key = getpass.getpass("Enter API key for Mistral AI: ")
            os.environ["MISTRAL_API_KEY"] = api_key
        
        try:
            self.llm = init_chat_model(self.llm_model,
                                       model_provider="mistralai",
                                       temperature=TEMPERATURE)
            logger.info(f"Initialized LLM: {self.llm_model} "
                        f"with temperature: {TEMPERATURE}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _setup_prompts(self) -> None:
        """Setup prompt templates."""
        system_message = """
Given an input question, create a syntactically correct BigQuery SQL query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Use the following table information:
{table_info}

Important BigQuery SQL rules:
- Use backticks (`) for table and column names if they contain special chars
- Use standard SQL functions compatible with BigQuery
- Be mindful of BigQuery-specific syntax and functions
"""

        user_prompt = "Question: {input}"
        
        self.query_prompt_template = ChatPromptTemplate([
            ("system", system_message), 
            ("user", user_prompt)
        ])
    
    def _setup_graph(self) -> None:
        """Setup the LangGraph workflow."""
        graph_builder = StateGraph(State).add_sequence([
            self._write_query, 
            self._execute_query, 
            self._generate_answer
        ])
        graph_builder.add_edge(START, "_write_query")
        self.graph = graph_builder.compile()
    
    def _get_table_schema(self) -> str:
        """Get schema information for the BigQuery table."""
        try:
            table = self.client.get_table(self.table_id)
            schema_info = []
            schema_info.append(f"Table: {self.table_id}")
            schema_info.append("Columns:")
            
            for field in table.schema:
                schema_info.append(f"  - {field.name}: {field.field_type}")
                if field.description:
                    schema_info.append(f"    Description: {field.description}")
            
            return "\n".join(schema_info)
        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return f"Table: {self.table_id}\nSchema information unavailable"
    
    def _write_query(self, state: State) -> Dict[str, str]:
        """Generate SQL query to fetch information."""
        try:
            table_info = self._get_table_schema()
            
            prompt = self.query_prompt_template.invoke({
                "top_k": self.top_k,
                "table_info": table_info,
                "input": state["question"],
            })
            
            structured_llm = self.llm.with_structured_output(QueryOutput)
            result = structured_llm.invoke(prompt)
            
            logger.info(f"Generated query: {result['query']}")
            return {"query": result["query"]}
        
        except Exception as e:
            logger.error(f"Error generating query: {e}")
            return {"query": f"SELECT * FROM `{self.table_id}` LIMIT 1"}
    
    def _execute_query(self, state: State) -> Dict[str, str]:
        """Execute BigQuery SQL query."""
        try:
            query_job = self.client.query(state["query"])
            result = query_job.result()
            
            # Convert to dataframe and then to string representation
            df = result.to_dataframe()
            
            if df.empty:
                result_str = "No results found."
            else:
                # Limit the output to avoid overwhelming the LLM
                if len(df) > 50:
                    df_display = df.head(50)
                    result_str = (f"Showing first 50 rows out of {len(df)} "
                                  f"total results:\n{df_display.to_string()}")
                else:
                    result_str = df.to_string()
             
            logger.info("BigQuery query executed successfully")
            return {"result": result_str}
        
        except Exception as e:
            logger.error(f"Error executing BigQuery query: {e}")
            return {"result": f"Error executing query: {str(e)}"}
    
    def _generate_answer(self, state: State) -> Dict[str, str]:
        """Answer question using retrieved information as context."""
        try:
            prompt = (
                "Given the following user question, corresponding SQL query, "
                "and SQL result, answer the user question.\n\n"
                f'Question: {state["question"]}\n'
                f'SQL Query: {state["query"]}\n'
                f'SQL Result: {state["result"]}'
            )
            
            response = self.llm.invoke(prompt)
            
            logger.info("Generated answer successfully")
            return {"answer": response.content}
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {"answer": f"Error generating answer: {str(e)}"}
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question and return the result.
        
        Args:
            question: Natural language question about the database
            
        Returns:
            Dictionary containing question, generated query, result, and answer
        """
        try:
            raw_result = {}
            for step in self.graph.stream(
                {"question": question}, 
                stream_mode="updates"
            ):
                raw_result.update(step)
            
            # Extract clean values from the graph output
            write_query_data = raw_result.get('_write_query', {})
            execute_query_data = raw_result.get('_execute_query', {})
            generate_answer_data = raw_result.get('_generate_answer', {})
            
            clean_result = {
                "question": question,
                "query": write_query_data.get('query', ''),
                "result": execute_query_data.get('result', ''),
                "answer": generate_answer_data.get('answer', '')
            }
            
            return clean_result
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "question": question,
                "query": "",
                "result": "",
                "answer": f"Error processing query: {str(e)}"
            }
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the connected BigQuery table."""
        try:
            table = self.client.get_table(self.table_id)
            return {
                "project": self.gcp_project,
                "dataset": self.bq_dataset,
                "table": self.bq_table,
                "table_id": self.table_id,
                "num_rows": table.num_rows,
                "schema": self._get_table_schema()
            }
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {
                "project": self.gcp_project,
                "dataset": self.bq_dataset,
                "table": self.bq_table,
                "table_id": self.table_id,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """Test BigQuery connection."""
        try:
            # Simple query to test connection
            query = f"SELECT 1 as test FROM `{self.table_id}` LIMIT 1"
            query_job = self.client.query(query)
            query_job.result()
            return True
        except Exception as e:
            logger.error(f"BigQuery connection test failed: {e}")
            return False


def main():
    """Example usage of the RAG Query System with BigQuery."""
    try:
        # Initialize the system
        rag_system = RAGQuerySystem()
        
        # Test connection
        if not rag_system.test_connection():
            print("Failed to connect to BigQuery")
            return
        
        # Print database info
        db_info = rag_system.get_database_info()
        print(f"BigQuery Project: {db_info.get('project', 'N/A')}")
        print(f"Dataset: {db_info.get('dataset', 'N/A')}")
        print(f"Table: {db_info.get('table', 'N/A')}")
        print(f"Number of rows: {db_info.get('num_rows', 'N/A')}")
        
        # Example query
        question = "How many rows are there in the table?"
        result = rag_system.query(question)
        
        print(f"\nQuestion: {result.get('question', 'N/A')}")
        print(f"Generated SQL: {result.get('query', 'N/A')}")
        print(f"Result: {result.get('result', 'N/A')}")
        print(f"Answer: {result.get('answer', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()

