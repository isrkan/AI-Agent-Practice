import asyncio
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing_extensions import TypedDict

load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent
SERVERS_DIR = PROJECT_ROOT / "servers"

# Define multiple server configurations - Each server is identified by a name and connection parameters
server_configs = {
    "math": {
        "command": "python",
        # The full absolute path to our math_server.py file
        "args": [str(SERVERS_DIR / "math_server.py")],
        "transport": "stdio"  # Communication via standard input/output
    },
    "weather": {
        "command": "python",
        # The full absolute path to our weather_server.py file
        "args": [str(SERVERS_DIR / "weather_server.py")],
        "transport": "stdio"
    }
}

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)


# Define the agent state
class AgentState(TypedDict):
    """
    Represents the state of our agent throughout its execution.

    Attributes:
        messages: The conversation history between user and agent
    """
    # The add_messages function defines how messages are accumulated
    messages: Annotated[list, add_messages]


def create_agent_graph(llm, tools):
    """
    Create a custom LangGraph agent with manual state management.

    Args:
        llm: The language model to use
        tools: List of tools available to the agent

    Returns:
        Compiled StateGraph representing the agent
    """

    # Bind tools to the LLM - This provides the LLM with tool schemas so it can generate tool calls
    llm_with_tools = llm.bind_tools(tools)

    # Define the call_model node
    def call_model(state: AgentState):
        """
        The main agent decision-making node.

        This function:
        1. Takes the current conversation state
        2. Passes messages to the LLM with available tools
        3. Returns the LLM's decision (tool call or final response)

        Args:
            state: Current agent state with message history

        Returns:
            Updated state with the agent's response
        """
        messages = state["messages"]
        # Invoke the LLM with the current messages
        response = llm_with_tools.invoke(messages)
        # Return the response (LangGraph automatically adds it to messages)
        return {"messages": [response]}

    # Create the graph using AgentState
    builder = StateGraph(AgentState)

    # Add nodes to the graph
    builder.add_node("call_model", call_model)  # The decision-making node
    builder.add_node("tools", ToolNode(tools))  # The tool execution node

    # Add edges
    builder.add_edge(START, "call_model")  # Set the entry point

    # Add conditional edges from call_model
    # tools_condition checks if the LLM wants to use tools
    builder.add_conditional_edges(
        "call_model",
        tools_condition,  # Returns "tools" if tool calls exist (if the last message has tool calls), otherwise END
    )

    # After tools execute, always return to call_model for next decision
    builder.add_edge("tools", "call_model")

    # Compile the graph into a runnable agent
    graph = builder.compile()

    return graph


async def main():
    # Initialize MultiServerMCPClient - it creates MCP client with server configurations
    client = MultiServerMCPClient(server_configs)

    try:
        # Load tools from all connected servers
        # This method:
        # 1. Connects to each configured server
        # 2. Retrieves tool schemas
        # 3. Converts them to LangChain tool format
        tools = await client.get_tools()
        print(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")

        # Create custom agent graph
        agent = create_agent_graph(llm, tools)

        print("\n=== Agent Graph Structure ===")
        print("Nodes: call_model, tools")
        print("Flow: START -> call_model -> [conditional] -> tools -> call_model -> END")
        print("================================\n")

        # Test with queries requiring different tools
        result = await agent.ainvoke({
            "messages": [HumanMessage(content="What is 54 + 2 * 3? Also, what's the weather in London?")]
        })

        print("\n=== Final Response ===")
        print(result["messages"][-1].content)

        print("\n=== Full Message History ===")
        for i, msg in enumerate(result["messages"]):
            print(f"\n[{i}] {type(msg).__name__}:")
            if hasattr(msg, 'content'):
                print(f"  Content: {msg.content}")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"  Tool Calls: {msg.tool_calls}")

    finally:
        # Clean up resources if needed
        pass


if __name__ == "__main__":
    asyncio.run(main())