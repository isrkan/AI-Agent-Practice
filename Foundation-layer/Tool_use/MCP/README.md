# MCP (Model Context Protocol)

AI applications increasingly require agents that can interact with external tools and services in a standardized way. The model context protocol (MCP) addresses this challenge by providing a unified interface for AI models to communicate with various tools and data sources. However, integrating MCP into production-ready agent systems requires understanding both the protocol itself and how to orchestrate complex agent workflows. This guide demonstrates how to build AI agents using MCP in combination with LangChain and LangGraph frameworks.

## Project Structure
Our MCP agent project is organized into the following structure:

```
project/
├── main.py                 # Host application with agent logic
└── servers/
    ├── math_server.py      # MCP server for math operations
    └── weather_server.py   # MCP server for weather information
```

This structure separates concerns:
- **main.py**: Contains the agent orchestration logic, MCP client initialization, and execution flow
- **servers/**: Contains independent MCP server implementations that expose tools


## Step 1: Environment setup and dependencies
Before we can work with MCP and build our agent, we need to install the necessary packages and set up our environment.

### Install Required Packages

```bash
pip install langchain-mcp-adapters langchain-openai langgraph python-dotenv
```

- `langchain-mcp-adapters`: Provides the bridge between MCP servers and LangChain's tool interface. Installing it also automatically installs the `mcp` package, which includes the `FastMCP` server used to build MCP-compatible tools.
- `langchain-openai`: OpenAI integration for LangChain
- `langgraph`: Graph-based agent orchestration framework

### Configure API Keys
Create a `.env` file in your project root:
```bash
OPENAI_API_KEY=our-openai-api-key-here
```

## Step 2: Create MCP servers with tools
Now we will create MCP servers. Each server will run as a separate logical component (separate Python file) that exposes tools through the MCP protocol. The server acts as a tool providers, packaging Python functions into a standardized interface that any MCP client can consume.

The MCP server is responsible for:
1. Tool registration: Defining what tools are available using the `@mcp.tool()` decorator
2. Schema generation: Automatically creating tool descriptions and parameter schemas
3. Execution handling: Running the actual tool code when called
4. Response formatting: Returning results in a standardized format

### Math server (servers/math_server.py)
Create a file `servers/math_server.py` with the following content:

```python
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server for custom tools
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

We define a FastMCP server that exposes two math tools using the `@mcp.tool()` decorator. The decorator automatically handles the MCP protocol details, converting our Python functions into MCP-compatible tools with proper schemas. The docstrings become tool descriptions that help the LLM understand when to use each tool. The `mcp.run(transport="stdio")` call starts the server and handles stdio communication.
- `FastMCP("Math")` creates a server with the name "Math"
- The `@mcp.tool()` decorator registers functions as MCP tools
- Function signatures are automatically converted to JSON schemas
- Docstrings provide natural language descriptions for the LLM
- The server runs as a separate process and communicates via stdio
- Each server can have multiple tools

### Weather server (servers/weather_server.py)
Create a file `servers/weather_server.py` with the following content:

```python
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    # This is a mock implementation - in production, we would call a real weather API
    return "Hot as hell"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Similar to the math server, we create a weather server with a tool for getting weather information. The tool is decorated with `@mcp.tool()` and includes a docstring. Note that this tool is async, demonstrating that MCP supports both synchronous and asynchronous functions. We define the tool with `async def` because it's meant to simulate calling an external API, like a weather service. Asynchronous functions let the server handle multiple requests efficiently without waiting (or "blocking") while one tool finishes. This is especially useful for slow operations like network requests. In a production environment, this would call an actual weather API.


## Step 3: Initialize the MCP client and connect to servers
Now we will look at how the host application (main.py) initializes the MCP client and connects to our servers. This is the critical bridge between our agent and the tools. The client's job is to:
1. Connect to one or more MCP servers
2. Discover available tools from each server
3. Convert tool schemas into LangChain-compatible format
4. Route tool calls to the appropriate server

### Client-server communication flow
When the agent wants to use a tool:
```
Agent → Client.call_tool("add", {"a": 2, "b": 2})
  ↓
Client identifies which server has "add" (math_server)
  ↓
Client sends request to MCP math server via stdio
  ↓
Server executes the tool and returns result
  ↓
Client formats result for Agent
  ↓
Agent receives: 4
```

### Server Configuration in main.py
In `main.py`, we define the server configurations:

```python
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

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
```

This configuration tells the MCP client how to launch and communicate with each server:
- **Key name** ("math", "weather"): Identifies the server
- **command**: The executable to run ("python")
- **args**: Arguments to pass (path to the server file). The `Path` usage ensures the paths work correctly regardless of where we run the script from
- **transport**: Communication method ("stdio" for standard input/output)


### Loading tools from servers
In the `main()` function, we initialize the client and load tools:

```python
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
```

The `MultiServerMCPClient` is initialized with our server configurations. The client configuration specifies how to launch each server (command and args) and the transport mechanism (stdio). 
- Each server is configured with a unique name and runs as an independent process
- The client manages all server connections
- Tools from all servers are combined into a single list
- The client handles all protocol complexity automatically

When we call `await client.get_tools()`, it:
1. Launches each server as a subprocess
2. Establishes stdio communication channels
3. Queries each server for available tools
4. Converts tool schemas to LangChain-compatible format
5. Returns a list of tools ready to use

The `client.get_tools()` function is asynchronous — it launches each tool server as a background process and communicates with them. Using `await` ensures that Python waits for those servers to start and respond, without freezing the entire app. This allows our program to remain responsive and efficient while it fetches tools over stdio.

## Step 4: Define the agent state
Before building the agent graph, we need to define the state structure that will be passed between nodes. The state maintains the conversation context and tracks the agent's workflow.

```python
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    Represents the state of our agent throughout its execution.

    Attributes:
        messages: The conversation history between user and agent
    """
    # The add_messages function defines how messages are accumulated
    messages: Annotated[list, add_messages]
```

We define a TypedDict that represents the agent's state. The `messages` field stores the conversation history, which is passed between nodes in the graph. The `Annotated` type with `add_messages` tells LangGraph how to merge messages when updating state—it appends new messages to the existing list rather than replacing it.
- `TypedDict` provides type hints for the state structure
- `add_messages` is a LangGraph reducer that handles message accumulation
- The state is immutable - each node returns updates that are merged into a new state
- Messages include user inputs, agent responses and tool calls/results


## Step 5: Build the LangGraph agent
LangGraph uses a graph structure to define agent workflows. We will create a graph with nodes for the agent's decision-making and tool execution, connected by edges that define the execution flow. The agent operates in a cycle:
1. call_model node: LLM analyzes messages and decides whether to call tools or respond
2. tools_condition: LangGraph's prebuilt routing function that checks for tool calls
3. tools node: Executes tool calls and returns results
4. Loop back: Returns to call_model with tool results for further processing

### Agent graph creation
In `main.py`, the `create_agent_graph` function builds the agent:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)

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
```

This code constructs the agent's execution graph:
1. LLM with tools: `llm.bind_tools(tools)` provides the LLM with tool schemas, enabling it to generate structured tool calls
2. call_model node: The decision-making function that:
   - Takes the current conversation state
   - Invokes the LLM with message history
   - Returns the LLM's response (either a tool call or final answer)
3. Graph structure:
   - START → call_model: Entry point of the graph
   - call_model → tools_condition: Conditional routing based on whether tool calls are present
   - tools_condition → tools or END: Routes to tools node if tool calls exist, otherwise ends
   - tools → call_model: After tool execution, return to call_model for next decision
4. tools_condition: LangGraph's prebuilt function that automatically:
   - Checks if the last message contains tool calls
   - Routes to "tools" node if present
   - Routes to END if no tool calls (final response)

The graph structure allows for iterative tool use and reasoning


## Step 6: Run the agent
Now let's look at how to execute the agent and test it with queries. The `main()` function in `main.py` orchestrates everything:

```python
import asyncio
from langchain_core.messages import HumanMessage

async def main():
    # Initialize MultiServerMCPClient - it creates MCP client with server configurations
    client = MultiServerMCPClient(server_configs)

    try:
        # Load tools from all connected servers
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
```

The main function:
1. Initialize client: Creates `MultiServerMCPClient` with server configurations
2. Load tools: Calls `await client.get_tools()` to connect to servers and retrieve tools
3. Create agent: Builds the LangGraph agent with the LLM and tools
4. Execute query: Invokes the agent with a test query that requires multiple tools:
   - Math calculation: "54 + 2 * 3"
   - Weather query: "what's the weather in London?"
5. Display results: Shows both the final response and the complete message history, including:
   - User message
   - Agent's tool calls
   - Tool execution results
   - Agent's final response