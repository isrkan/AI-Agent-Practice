# Tool calling

Tool use is the capability that transforms AI agents from pure language models into action-taking systems that can interact with the real world. By integrating tools, agents can perform concrete actions like searching the web, querying databases, calling APIs, executing code, or manipulating files. This bridges the gap between language understanding and practical utility, enabling agents to accomplish tasks that require more than just text generation.

Without tools, an AI agent is limited to generating text based on its training data and the current conversation context. With tools, the agent becomes a powerful orchestrator that can:
- Access real-time information beyond its training cutoff
- Perform calculations and data analysis
- Interact with external systems and APIs
- Execute multi-step workflows across different services
- Take actions in the physical or digital world

The tool registry calling is a traditional, code-based method where tools are defined as Python functions, registered in a central registry, and made available to the LLM through function calling capabilities. This approach gives us full control over tool implementation and is ideal for applications where all tools are custom-built and tightly integrated with our codebase.

### When to use tool registry
- We have a small to medium number of custom tools (typically < 20)
- All tools are implemented in the same codebase
- We need fine-grained control over tool execution and error handling
- Our tools are tightly coupled with our application logic
- We want minimal external dependencies

### Implementation guide

#### Step 1: Design the tool architecture
Before implementing tools, plan the tool architecture:
1. **Identify required tools:** List all the actions our agent needs to perform. Common categories include:
   - Information retrieval (web search, database queries, API calls)
   - Data processing (calculations, transformations, analysis)
   - External actions (sending emails, creating files, making purchases)
   - System operations (file management, process control)
2. **Define tool interfaces:** For each tool, specify:
   - Name: A clear, descriptive identifier (e.g., `web_search`, `calculate`, `send_email`)
   - Description: A natural language explanation of what the tool does (this helps the LLM decide when to use it)
   - Parameters: Input arguments with types and descriptions
   - Return type: What the tool returns
   - Error handling: How failures should be communicated
3. Plan tool organization: Decide how to structure our tools:
   - Group related tools together (e.g., all database tools in one module)
   - Consider tool dependencies and execution order
   - Think about tool composition (tools that call other tools)

#### Step 2: Create a modular tool structure
Organize our tools in a dedicated directory structure for maintainability and scalability:
```
├── tools/
│   ├── __init__.py
│   ├── search_tools.py      # Web search, document search
│   ├── data_tools.py        # Calculations, data processing
│   ├── communication_tools.py  # Email, messaging, notifications
│   ├── file_tools.py        # File operations
│   └── tool_registry.py     # Central registry for all tools
```

#### Step 3: Implement individual tools
Each tool is implemented as a Python function with proper type hints and documentation. The function signature and docstring are crucial because they are used to generate the tool schema that the LLM sees.

**Example: Search tools (tools/search_tools.py)**

```python
# tools/search_tools.py
from typing import List, Dict
import requests

def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for information using a search API.
    
    This tool performs a web search and returns the most relevant results.
    Use this when you need current information or facts not in your training data.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5, max: 10)
        
    Returns:
        A list of search results, each containing 'title', 'url', and 'snippet'
        
    Example:
        results = web_search("latest AI developments", num_results=3)
    """
    # Implementation using a search API (e.g., Google Custom Search, Bing, SerpAPI)
    try:
        # Example implementation (replace with actual API call)
        api_key = os.getenv("SEARCH_API_KEY")
        response = requests.get(
            "https://api.search-provider.com/search",
            params={"q": query, "num": num_results, "key": api_key}
        )
        response.raise_for_status()
        
        results = response.json().get("results", [])
        return [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "snippet": r.get("snippet")
            }
            for r in results[:num_results]
        ]
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]

def document_search(query: str, collection: str = "default") -> List[Dict[str, str]]:
    """
    Search through stored documents using semantic search.
    
    This tool searches your document database for relevant information.
    Use this when you need to retrieve information from your knowledge base.
    
    Args:
        query: The search query
        collection: The document collection to search in
        
    Returns:
        A list of relevant document chunks with their content and metadata
    """
    # Implementation using vector database (e.g., Pinecone, Weaviate, ChromaDB)
    try:
        # Example implementation
        from our_vector_db import search_documents
        results = search_documents(query, collection=collection, top_k=5)
        return results
    except Exception as e:
        return [{"error": f"Document search failed: {str(e)}"}]
```

**Key points for tool implementation:**
- **Comprehensive docstrings:** The docstring is used to generate the tool description for the LLM. Make it clear and detailed.
- **Type hints:** Use proper type hints for all parameters and return values. This helps with schema generation.
- **Error handling:** Always wrap tool logic in try-except blocks and return meaningful error messages.
- **Security:** Never use `eval()` or `exec()` without proper sandboxing. Validate all inputs.
- **Idempotency:** When possible, make tools idempotent (safe to call multiple times with the same inputs).

#### Step 4: Create the tool registry
The tool registry is a central module that collects all tools and provides them to the LLM in a standardized format. Thoroughly test each tool before integrating with the agent.

**Example: Tool registry (tools/tool_registry.py)**

```python
# tools/tool_registry.py
from typing import List, Callable, Dict, Any
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# Import all tool functions
from tools.search_tools import web_search, document_search
from tools.data_tools import calculate, analyze_data

class ToolRegistry:
    """
    Central registry for all agent tools.
    
    This class manages tool registration, retrieval, and provides
    tools in formats compatible with different LLM frameworks.
    """
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all available tools."""
        # Register search tools
        self.register_tool("web_search", web_search)
        self.register_tool("document_search", document_search)
        
        # Register data tools
        self.register_tool("calculate", calculate)
        self.register_tool("analyze_data", analyze_data)
        
        # Add more tools as needed
    
    def register_tool(self, name: str, func: Callable):
        """Register a single tool."""
        self._tools[name] = func
    
    def get_tool(self, name: str) -> Callable:
        """Retrieve a tool by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> Dict[str, Callable]:
        """Get all registered tools."""
        return self._tools

        def get_langchain_tools(self) -> List[StructuredTool]:
        """
        Convert registered tools to LangChain StructuredTool format.
        
        LangChain's StructuredTool.from_function() automatically extracts
        the schema from function signatures and docstrings, creating
        tools that are compatible with LangChain agents and can be
        bound to LLMs for function calling.
        
        Returns:
            A list of LangChain StructuredTool objects
        """
        langchain_tools = []
        
        for name, func in self._tools.items():
            # LangChain automatically extracts schema from function signature and docstring
            tool = StructuredTool.from_function(
                func=func,
                name=name,
                description=func.__doc__ or f"Tool: {name}"
            )
            langchain_tools.append(tool)
        
        return langchain_tools

# Create a global registry instance
tool_registry = ToolRegistry()
```

- **Centralized management:** All tools are registered in one place, making it easy to add, remove, or modify tools.
- **Framework compatibility:** The registry can convert tools to different formats (LangChain, OpenAI, etc.).
- **Schema generation:** Automatically generates tool schemas from function signatures and docstrings.
- **Easy retrieval:** Provides methods to get individual tools or all tools at once.

#### Step 5: Integrate tools with our LLM
Now connect the tool registry to our LLM. The exact implementation depends on our framework.

**Example: Using LangChain**

```python
# main.py
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools.tool_registry import tool_registry

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Get tools from registry
tools = tool_registry.get_langchain_tools()

# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant with access to various tools. Use them when needed to help the user."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Use the agent
response = agent_executor.invoke({
    "input": "Search the web for the latest AI news and calculate the average of these numbers: 10, 20, 30, 40, 50"
})

print(response["output"])
```

**Key points:**
- **Tool binding:** The LLM is provided with tool schemas so it knows what tools are available.
- **Automatic tool calling:** The LLM decides when to use tools based on the user's request.
- **Tool execution:** When the LLM requests a tool call, our code executes the tool and returns the result.
- **Iterative process:** The agent may call multiple tools in sequence to complete a task.