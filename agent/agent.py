import json
import os

# Import our custom tools
import sys
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Import LangGraph components
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages
from pydantic import SecretStr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.fda_tool import get_adverse_events
from tools.neo4j_tool import Neo4jTool
from tools.pdf_rag_tool import PDFTool

# Global variables for tools and model
neo4j_tool: Optional[Neo4jTool] = None
pdf_tool: Optional[PDFTool] = None
model: Optional[ChatOpenAI] = None
tools: List = []
tools_by_name: Dict[str, Any] = {}
_initialized = False  # Flag to track if agent has been initialized


# Define the tools as LangChain tools (will be redefined with proper context)
@tool
def fda_adverse_events_tool(drug_name: str, limit: int = 10) -> str:
    """Get adverse events data for a specific drug from FDA database.

    Args:
        drug_name: The name of the drug to search for (e.g., "TRAMADOL")
        limit: Maximum number of results to return (default: 10)

    Returns:
        JSON string containing adverse events data
    """
    try:
        results = get_adverse_events(drug_name, limit)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error retrieving FDA data: {str(e)}"


@tool
def neo4j_query_tool(question: str) -> str:
    """Query the Neo4j knowledge graph with natural language questions.

    Args:
        question: Natural language question about the knowledge graph

    Returns:
        Answer to the question based on the graph data
    """
    global neo4j_tool
    try:
        if neo4j_tool is None:
            return "Error: Neo4j tool not initialized"
        result = neo4j_tool.ask_question(question)
        return str(result)
    except Exception as e:
        return f"Error querying Neo4j: {str(e)}"


@tool
def pdf_search_tool(question: str) -> str:
    """Search the PDF document for information and generate answers.

    Args:
        question: Question about the PDF content

    Returns:
        JSON string containing both the retrieved documents and the generated answer
    """
    global pdf_tool
    try:
        if pdf_tool is None:
            return "Error: PDF tool not initialized. Please check the initialization logs."
        
        # Check if vector store is initialized
        if not hasattr(pdf_tool, 'vector_store') or pdf_tool.vector_store is None:
            return "Error: PDF vector store not initialized. The PDF file may not have been loaded successfully during initialization."
        
        docs, answer = pdf_tool.search_text(question)

        # Convert Document objects to serializable format
        serializable_docs = []
        for doc in docs:
            serializable_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })

        result = {
            "retrieved_documents": serializable_docs,
            "answer": answer,
            "total_documents": len(docs),
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error searching PDF: {str(e)}"


def reset_agent():
    """Reset the agent to allow reinitialization with new configuration"""
    global neo4j_tool, pdf_tool, model, tools, tools_by_name, _initialized

    # Close existing connections
    if neo4j_tool:
        neo4j_tool.close()

    # Reset all variables
    neo4j_tool = None
    pdf_tool = None
    model = None
    tools = []
    tools_by_name = {}
    _initialized = False

    print("🔄 Agent reset completed")


def initialize_agent_with_config(
    openai_api_key: str, neo4j_uri: str, neo4j_username: str, neo4j_password: str
):
    """Initialize the agent with user-provided configuration"""
    global neo4j_tool, pdf_tool, model, tools, tools_by_name, _initialized

    # Only initialize once
    if _initialized:
        print("✅ Agent already initialized, reusing existing configuration")
        return

    # Set environment variables for tools that need them
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USERNAME"] = neo4j_username
    os.environ["NEO4J_PASSWORD"] = neo4j_password

    # Initialize tools
    neo4j_tool = Neo4jTool()
    pdf_tool = PDFTool()

    # Initialize PDF tool vector store
    try:
        # Try multiple possible paths for the PDF file
        possible_paths = [
            # Current working directory relative path
            "./tools/pdf_data/report_2023_2024.pdf",
            # Absolute path from current file
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tools",
                "pdf_data",
                "report_2023_2024.pdf",
            ),
            # Path relative to the agent directory
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "tools",
                "pdf_data",
                "report_2023_2024.pdf",
            ),
        ]
        
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Agent file location: {os.path.abspath(__file__)}")
        print(f"🔍 Trying PDF paths: {possible_paths}")
        
        pdf_path = None
        for path in possible_paths:
            exists = os.path.exists(path)
            print(f"🔍 Path {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
            if exists:
                pdf_path = path
                break
        
        if pdf_path is None:
            print(f"⚠️ PDF file not found. Tried paths: {possible_paths}")
            print(f"⚠️ Current directory contents: {os.listdir('.')}")
            raise FileNotFoundError("PDF file not found in any expected location")
        
        print(f"📄 Using PDF path: {pdf_path}")
        pdf_tool.create_vector_store(pdf_path)
        print("✅ PDF vector store initialized")
    except Exception as e:
        print(f"⚠️ PDF vector store initialization failed: {e}")
        print(f"⚠️ Exception type: {type(e).__name__}")
        import traceback
        print(f"⚠️ Full traceback: {traceback.format_exc()}")

    # Initialize Neo4j connection
    try:
        neo4j_tool.connect()
        neo4j_tool.initialize_qa_chain()
        print("✅ Neo4j connection initialized")
    except Exception as e:
        print(f"⚠️ Neo4j initialization failed: {e}")

    # Define the model
    model = ChatOpenAI(model="gpt-4", temperature=0, api_key=SecretStr(openai_api_key))

    # Create the tools list
    tools = [fda_adverse_events_tool, neo4j_query_tool, pdf_search_tool]

    # Create tools by name mapping
    tools_by_name = {tool.name: tool for tool in tools}

    _initialized = True
    print("✅ Agent initialized with user configuration")


# Define tasks
@task
def call_model(messages):
    """Call model with a sequence of messages."""
    if not model:
        raise ValueError(
            "Model not initialized. Call initialize_agent_with_config() first."
        )

    system_prompt = """You are a helpful AI assistant with access to three specialized tools:

1. FDA Adverse Events Tool: Get adverse events data for drugs from the FDA database
2. Neo4j Knowledge Graph Tool: Query a pharmaceutical knowledge graph with natural language
3. PDF Search Tool: Search and answer questions about a pharmaceutical company report

When a user asks a question, think about which tool(s) would be most helpful to answer it.
You can use multiple tools if needed to provide a comprehensive answer.

Always provide clear, helpful responses and explain what information you found."""

    # Create a prompt with system message
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), MessagesPlaceholder(variable_name="messages")]
    )

    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)

    # Get the formatted messages
    formatted_messages = prompt.format_messages(messages=messages)

    # Call the model
    response = model_with_tools.invoke(formatted_messages)
    return response


@task
def call_tool(tool_call):
    """Execute a tool call and return the result as a ToolMessage."""
    if not tools_by_name:
        raise ValueError(
            "Tools not initialized. Call initialize_agent_with_config() first."
        )

    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])


# Define entrypoint
@entrypoint()
def agent(messages):
    """Main agent entrypoint that orchestrates model calls and tool execution."""
    llm_response = call_model(messages).result()
    while True:
        if not llm_response.tool_calls:
            break

        # Execute tools
        tool_result_futures = [
            call_tool(tool_call) for tool_call in llm_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]

        # Append to message list
        messages = add_messages(messages, [llm_response, *tool_results])

        # Call model again
        llm_response = call_model(messages).result()

    return llm_response


def run_agent(
    question: str,
    openai_api_key: Optional[str] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
) -> str:
    """Run the agent with a given question and configuration."""
    try:
        # Initialize agent if configuration provided and not already initialized
        if (
            all([openai_api_key, neo4j_uri, neo4j_username, neo4j_password])
            and not _initialized
        ):
            initialize_agent_with_config(
                openai_api_key, neo4j_uri, neo4j_username, neo4j_password
            )
        elif not _initialized:
            return "Error: Agent not initialized. Please provide all configuration parameters."

        # Prepare the user message
        user_message = HumanMessage(content=question)

        # Run the agent
        result = agent.invoke([user_message])

        # Return the final response content
        return result.content
    except Exception as e:
        return f"Error running agent: {str(e)}"


def run_agent_with_streaming(
    question: str,
    openai_api_key: Optional[str] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
):
    """Run the agent with streaming and yield steps as they happen.

    Args:
        question: The user's question
        openai_api_key: OpenAI API key
        neo4j_uri: Neo4j database URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password

    Yields:
        dict: Step information with keys:
            - task_name: str
            - content: str
            - step_type: str ('model_call', 'tool_call', 'final_answer')
            - is_final: bool (True for final answer)
    """
    try:
        # Initialize agent if configuration provided and not already initialized
        if (
            all([openai_api_key, neo4j_uri, neo4j_username, neo4j_password])
            and not _initialized
        ):
            initialize_agent_with_config(
                openai_api_key, neo4j_uri, neo4j_username, neo4j_password
            )
        elif not _initialized:
            yield {
                "task_name": "error",
                "content": "Error: Agent not initialized. Please provide all configuration parameters.",
                "step_type": "error",
                "is_final": True,
            }
            return

        # Prepare the user message
        user_message = HumanMessage(content=question)

        # Stream the agent execution
        for step in agent.stream([user_message]):
            for task_name, message in step.items():
                if task_name == "agent":
                    continue  # Skip the main agent step

                # Determine step type and content
                step_type = "model_call"
                content = ""
                is_final = False

                if task_name == "call_model":
                    # Check if message has tool_calls attribute (AIMessage)
                    if (
                        hasattr(message, "tool_calls")
                        and hasattr(message.tool_calls, "__len__")
                        and len(message.tool_calls) > 0
                    ):
                        # Model is calling tools
                        tool_calls_info = []
                        for tool_call in message.tool_calls:
                            tool_calls_info.append(
                                f"Tool: {tool_call['name']}\nArguments: {tool_call['args']}"
                            )
                        content = (
                            "🤔 Thinking and deciding which tools to use...\n\n"
                            + "\n\n".join(tool_calls_info)
                        )
                        step_type = "tool_decision"
                    else:
                        # Final answer
                        content = message.content
                        step_type = "final_answer"
                        is_final = True
                elif task_name == "call_tool":
                    # Tool execution
                    if hasattr(message, "content"):
                        content = f"🔧 Executing tool...\n\nResult:\n{message.content}"
                    else:
                        content = f"🔧 Executing tool...\n\nResult:\n{str(message)}"
                    step_type = "tool_execution"

                yield {
                    "task_name": task_name,
                    "content": content,
                    "step_type": step_type,
                    "is_final": is_final,
                }

    except Exception as e:
        error_msg = f"Error running agent: {str(e)}"
        yield {
            "task_name": "error",
            "content": error_msg,
            "step_type": "error",
            "is_final": True,
        }