import os
import getpass
from typing import Optional, Dict, Any
import logging

from config import LLM_MODEL_NAME, TEMPERATURE

from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict, Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


class RAGQuerySystem:
    """A RAG system for querying SQL databases using natural language."""
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 llm_model: str = LLM_MODEL_NAME,
                 top_k: int = 10):
        """
        Initialize the RAG Query System.
        
        Args:
            db_path: Path to the SQLite database file
            llm_model: Name of the LLM model to use
            top_k: Maximum number of results to return
        """
        self.llm_model = llm_model
        self.top_k = top_k
        self._setup_database(db_path)
        self._setup_llm()
        self._setup_prompts()
        self._setup_graph()
    
    def _setup_database(self, db_path: Optional[str] = None) -> None:
        """Setup database connection."""
        if db_path is None:
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
            db_path = os.path.join(data_path, 'datalab.sqlite')
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        logger.info(f"Connected to database: {self.db.dialect}")
        logger.info(f"Available tables: {self.db.get_usable_table_names()}")
    
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


        # Complementary rules if needed
        '''

        🚨 ABSOLUTELY CRITICAL SQL FORMATTING RULES - NEVER IGNORE THESE 🚨:

        1. ANY table name that contains dots (.), spaces, or hyphens (-) MUST be 
        enclosed in single quotes.
        2. This is NON-NEGOTIABLE. Failure to do this will cause syntax errors.
        3. Before writing any query, scan ALL table names for special characters.

        EXAMPLES:
        ✅ CORRECT: SELECT COUNT(*) FROM 'data.gouv.fr.2022.clean';
        ❌ WRONG: SELECT COUNT(*) FROM data.gouv.fr.2022.clean;
        ✅ CORRECT: SELECT * FROM 'my-table' WHERE id = 1;
        ❌ WRONG: SELECT * FROM my-table WHERE id = 1;

        REMEMBER: If a table name has dots, spaces, or hyphens → USE SINGLE QUOTES!

        Before finalizing your query, double-check that ALL table names with special 
        characters are properly quoted with single quotes.

        '''


        """Setup prompt templates."""
        system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
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
    
    def _fix_table_quotes(self, query: str) -> str:
        """
        Post-process SQL query to ensure the specific table name with dots
        is properly quoted.
        """
        # Specifically handle the data.gouv.fr.2022.clean table
        table_name = "data.gouv.fr.2022.clean"
        quoted_table = "'data.gouv.fr.2022.clean'"
        if table_name in query and quoted_table not in query:
            query = query.replace(table_name, quoted_table)
        
        return query
    
    def _write_query(self, state: State) -> Dict[str, str]:
        """Generate SQL query to fetch information."""
        try:
            prompt = self.query_prompt_template.invoke({
                "dialect": self.db.dialect,
                "top_k": self.top_k,
                "table_info": self.db.get_table_info(),
                "input": state["question"],
            })
            
            structured_llm = self.llm.with_structured_output(QueryOutput)
            result = structured_llm.invoke(prompt)
            
            # Post-process query to ensure table names are properly quoted
            processed_query = self._fix_table_quotes(result["query"])
            
            logger.info(f"Generated query: {processed_query}")
            return {"query": processed_query}
        
        except Exception as e:
            logger.error(f"Error generating query: {e}")
            return {"query": "SELECT 1;"}  # Fallback query
    
    def _execute_query(self, state: State) -> Dict[str, str]:
        """Execute SQL query."""
        try:
            execute_query_tool = QuerySQLDatabaseTool(db=self.db)
            result = execute_query_tool.invoke(state["query"])
             
            logger.info("Query executed successfully")
            return {"result": str(result)}
        
        except Exception as e:
            logger.error(f"Error executing query: {e}")
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
        """Get information about the connected database."""
        return {
            "dialect": self.db.dialect,
            "tables": self.db.get_usable_table_names(),
            "schema": self.db.get_table_info()
        }
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            self.db.run("SELECT 1;")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


def main():
    """Example usage of the RAG Query System."""
    try:
        # Initialize the system
        rag_system = RAGQuerySystem()
        
        # Test connection
        if not rag_system.test_connection():
            print("Failed to connect to database")
            return
        
        # Print database info
        db_info = rag_system.get_database_info()
        print(f"Database dialect: {db_info['dialect']}")
        print(f"Available tables: {db_info['tables']}")
        
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

