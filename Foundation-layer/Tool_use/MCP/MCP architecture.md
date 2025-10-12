## MCP architecture
At a high level, MCP allows us to expose functionality (like APIs, local tools, or domain-specific services) in a standardized way that clients (AI agents) can discover, understand, and invoke. This decouples the agent's reasoning logic from the actual execution of tasks, enabling better modularity, scalability, and reusability.

### MCP server
An **MCP server** is a process that exposes tools, resources, and prompts through the standardized MCP interface. Think of it as a service that packages functionality (like a calculator, database access, or API calls) in a way that AI agents can discover and use. Servers can be:
- **Built-in tools**: Custom Python functions we write and expose
- **External services**: Wrappers around APIs like Google Search, databases, or file systems
- **Specialized domains**: Domain-specific toolsets (e.g., data analysis, web scraping)

Servers run independently and communicate via standard input/output (stdio) or HTTP transport mechanisms.

### MCP client
An **MCP client** is the component that connects to one or more MCP servers and makes their tools available to our application. The client:
- Discovers what tools are available from connected servers
- Translates tool schemas into formats our AI framework understands
- Handles the communication protocol between our agent and the servers
- Manages multiple server connections simultaneously

In our implementation, `MultiServerMCPClient` from LangChain adapters acts as this client layer.

### Host application
The **host** is our main application - in our case, the Python environment running the LangChain/LangGraph agent. The host:
- Initializes and manages the MCP client
- Runs the AI agent's decision-making logic
- Orchestrates the flow between the LLM, tools, and user
- Handles the overall application lifecycle

### Transport mechanisms
MCP supports different ways for clients and servers to communicate. Understanding these transport mechanisms is crucial for choosing the right approach for your use case:

#### 1. **stdio (Standard Input/Output)**
- **What it is**: Communication happens through standard input and output streams.
- **How it works**: The client launches the server as a subprocess and communicates by writing to its stdin and reading from its stdout.
- **Best for**: Local tools, development, and simple deployments where the server runs on the same machine as the client.
- **Advantages**: Simple to set up, no network configuration needed and automatic process management.
- **Limitations**: Server must be on the same machine, not suitable for remote services and limited scalability.

**Example configuration:**
```python
"math": {
    "command": "python",
    "args": ["./servers/math_server.py"],
    "transport": "stdio"  # Server runs as subprocess
}
```

#### 2. **HTTP (Hypertext Transfer Protocol)**
- **What it is**: Standard web-based communication using HTTP requests and responses.
- **How it works**: The server runs as a web service (e.g., on port 8000), and the client makes HTTP requests to it.
- **Best for**: Remote services, production deployments and microservices architecture.
- **Advantages**: Server can run anywhere (local or remote), standard web protocols, easy to scale horizontally and can be load-balanced.
- **Limitations**: Requires network configuration, ,More complex setup and need to manage server lifecycle separately.

**Example configuration:**
```python
"weather": {
    "url": "http://localhost:8000/mcp",
    "transport": "http"  # Server runs as web service
}
```

#### 3. **SSE (Server-Sent Events)**
- **What it is**: A streaming protocol where the server can push updates to the client over HTTP.
- **How it works**: Similar to HTTP but maintains an open connection for real-time updates.
- **Best for**: Tools that need to stream results or provide progress updates.
- **Advantages**: Real-time streaming capabilities, built on HTTP (firewall-friendly) and efficient for long-running operations.
- **Limitations**: More complex than basic HTTP and requires server support for streaming.

**Example configuration:**
```python
"streaming_tool": {
    "url": "http://localhost:8000/mcp",
    "transport": "sse"  # Server supports streaming
}
```

**Choosing the right transport:**
- Use **stdio** for local development and simple tools
- Use **HTTP** for production deployments and remote services
- Use **SSE** when you need real-time streaming or progress updates


### Example to how they work together
```
┌─────────────────────────────────────────────────────────┐
│                    Host Application                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │         LangGraph Agent (Decision Maker)         │   │
│  │              ↕                                   │   │
│  │         LangChain LLM (GPT-4)                    │   │
│  └──────────────────────────────────────────────────┘   │
│                       ↕                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │           MCP Client (MultiServerMCPClient)      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                        ↕
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌──────────────────┐          ┌──────────────────┐
│   MCP Server 1   │          │   MCP Server 2   │
│  (Custom Tools)  │          │ (Google Search)  │
│                  │          │                  │
│ - Calculator     │          │ - Web Search     │
│ - Weather        │          │ - News Lookup    │
└──────────────────┘          └──────────────────┘
 (stdio transport)             (stdio transport)
```

The beauty of this architecture is separation of concerns: servers focus on providing tools, the client handles protocol details, and the host manages the agent logic. This makes the system modular and easy to extend.