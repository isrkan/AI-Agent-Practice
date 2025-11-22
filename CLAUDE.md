# CLAUDE.md - AI Agent Practice Repository Guide

## Project Overview

This repository is a comprehensive educational and reference guide for building production-ready AI agents. It follows a layered architecture that progressively covers foundational components through deployment and safety considerations.

**Purpose:** Provide hands-on implementations, tutorials, and best practices for AI agent development across multiple layers of complexity.

## Repository Structure

```
AI-Agent-Practice/
├── Foundation-layer/           # Core infrastructure
│   ├── LLMs/                   # Language model integration
│   ├── Memory/                 # Short-term and long-term memory
│   └── Tool_use/               # Function calling and MCP
├── Behavior-layer/             # Agent behavior shaping
│   ├── Prompt-engineering/     # 21+ prompt techniques
│   └── Structured-output/      # Schema-based responses
├── Intelligence-layer/         # Reasoning and knowledge
│   ├── Context-engineering/    # Context management
│   ├── Planning_and_reasoning/ # Extended thinking
│   └── RAG-system/             # Retrieval-augmented generation
├── Architecture-and-orchestration-layer/  # Workflows
│   └── Agent-loop-and-control-flow/       # Design patterns
├── Application-development-layer/         # Production apps
│   └── Application-integration/           # Streamlit chatbot
└── Safety-layer/               # Guardrails and compliance
    └── Guardrails-system/      # Safety monitoring
```

## Layer Descriptions

### Foundation-layer
Core building blocks for AI agents:
- **LLMs/**: API-based (OpenAI, Gemini) and open-source (LLaMA) integrations
- **Memory/**: In-memory and Redis-based persistence for session and long-term memory
- **Tool_use/**: MCP server implementations with function calling examples

### Behavior-layer
Shapes agent responses and output formatting:
- **Prompt-engineering/**: Comprehensive techniques from zero-shot to chain-of-thought
- **Structured-output/**: Pydantic models for schema-based generation

### Intelligence-layer
Advanced reasoning and knowledge retrieval:
- **Context-engineering/**: Context management and failure mitigation
- **RAG-system/**: 19 retrieval techniques across 8 categories including GraphRAG, RAPTOR, Self-RAG

### Architecture-and-orchestration-layer
Complex workflow orchestration:
- **Agentic design patterns/**: ReAct and tool-use patterns
- **Workflow patterns/**: Routing, parallelization, orchestrator-worker patterns

### Application-development-layer
Production deployment examples with complete Streamlit chatbot implementation.

### Safety-layer
Guardrails integration with Qualifire for prompt injection detection, PII protection, and content safety.

## Technology Stack

- **LLM Frameworks**: LangChain, LangGraph
- **LLM Providers**: OpenAI, Google Generative AI (Gemini), LLaMA
- **Memory**: In-memory storage, Redis
- **Tools**: FastMCP for tool servers
- **UI**: Streamlit
- **Safety**: Qualifire guardrails

## File Naming Conventions

- **Jupyter Notebooks**: Title Case with spaces (e.g., `Chain of thought prompting.ipynb`)
- **Python Scripts**: snake_case (e.g., `math_server.py`)
- **Markdown Guides**: lowercase with hyphens (e.g., `MCP architecture.md`)
- **Directories**: Hyphenated descriptive names (e.g., `Agent-loop-and-control-flow`)

## Code Conventions

### Python Style
```python
# Standard pattern for Python files:
# 1. Imports (stdlib, third-party, local)
# 2. Configuration and constants
# 3. Helper functions with Google-style docstrings
# 4. Main logic/classes
# 5. Entry point (if __name__ == "__main__")
```

### Documentation Requirements
- Each major directory includes a README.md with implementation guides
- Functions use Google-style docstrings with type hints
- Notebooks are self-contained with clear output demonstrations

### Configuration Management
- API keys stored in `.env` files (never hardcoded)
- Environment variables for all sensitive configuration
- Modular LLM registry pattern for model selection

## Development Workflow

### Adding New Content

1. **Identify the appropriate layer** based on the content type
2. **Create a new directory** if introducing a new concept/technique
3. **Include a README.md** with:
   - Overview of the concept
   - Step-by-step implementation guide
   - Common issues and solutions
4. **Add implementation** as Jupyter notebook or Python script
5. **Follow naming conventions** for consistency

### Notebook Development

- Make notebooks standalone and runnable independently
- Include clear markdown explanations between code cells
- Show example outputs for verification
- Progress from simple to complex implementations

### Python Script Development

- Use type hints throughout
- Include comprehensive docstrings
- Implement proper error handling
- Follow the standard file structure pattern

## Key Implementation Patterns

### LLM Integration
```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize with environment variables
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

### Memory Systems
- Short-term: Session-level context in memory
- Long-term: Redis-backed persistent storage
- Dual-layer: Combined approach for complex agents

### MCP Servers
Located in `Foundation-layer/Tool_use/MCP/servers/`:
- Servers implement specific tools (math, weather)
- Main agent orchestrates tool selection and execution

### RAG Techniques
Organized by category in `Intelligence-layer/RAG-system/RAG-techniques/`:
- Context and content enrichment (semantic chunking, context windows, contextual compression)
- Query enhancement (query transformations, hypothetical document embedding)
- Ranking and reranking (fusion retrieval, reranking)
- Iterative techniques (Self-RAG, Corrective RAG, adaptive retrieval, feedback loops)
- Structured retrieval (GraphRAG, RAPTOR)
- Indexing (hierarchical indices)
- Evaluation (DeepEval framework)
- Explainability and transparency

## Best Practices for AI Assistants

### When Modifying This Repository

1. **Preserve the layered architecture** - Place content in the appropriate layer
2. **Maintain naming consistency** - Follow existing conventions strictly
3. **Include documentation** - Every new component needs a README
4. **Test implementations** - Ensure notebooks run without errors
5. **Use relative imports** - For Python modules within the same directory

### Understanding Content Organization

- **Concepts build progressively** - Each layer builds on previous ones
- **Notebooks are educational** - Focus on clarity over optimization
- **READMEs are implementation guides** - Follow their step-by-step patterns

### Common Tasks

**Adding a new prompt technique:**
1. Create notebook in `Behavior-layer/Prompt-engineering/Prompt_engineering_techniques/`
2. Follow naming pattern: `Technique name.ipynb`
3. Include examples with multiple LLM providers if possible

**Adding a new RAG technique:**
1. Identify category (context enrichment, query enhancement, ranking, iterative, structured retrieval, indexing, evaluation, or explainability)
2. Place in appropriate subdirectory under `Intelligence-layer/RAG-system/RAG-techniques/`
3. Include evaluation if applicable

**Adding a new tool/server:**
1. Create server in `Foundation-layer/Tool_use/MCP/servers/`
2. Follow existing server patterns (math_server.py, weather_server.py)
3. Update main.py to register the new server

## File Statistics

- **Total Content Files**: 84
- **Jupyter Notebooks**: 61 (72.6%)
- **Python Scripts**: 6 (7.1%)
- **Markdown Documentation**: 17 (20.2%)

## Important Files

### Core Documentation
- `Foundation-layer/LLMs/README.md` - LLM selection guide
- `Foundation-layer/Memory/README.md` - Memory architecture
- `Foundation-layer/Tool_use/MCP/README.md` - MCP implementation guide
- `Behavior-layer/Prompt-engineering/README.md` - 7-step prompt methodology
- `Intelligence-layer/RAG-system/README.md` - Production RAG guide
- `Intelligence-layer/Context-engineering/README.md` - Context management

### Production Examples
- `Application-development-layer/Application-integration/AI-Chatbot-Streamlit-App/app.py`
- `Safety-layer/Guardrails-system/Qualifire guardrails/app.py`

### MCP Implementation
- `Foundation-layer/Tool_use/MCP/main.py` - Agent orchestration
- `Foundation-layer/Tool_use/MCP/servers/*.py` - Tool servers

## Testing and Validation

- RAG evaluation using DeepEval framework
- Prompt effectiveness evaluation techniques included
- Safety testing through Qualifire guardrails

## Security Considerations

- Never hardcode API keys or secrets
- Use environment variables for all configuration
- Implement input validation for user-facing applications
- Apply guardrails for content safety
- Validate structured outputs against schemas

## Contributing Guidelines

1. Follow the existing directory structure and naming conventions
2. Include comprehensive documentation with any new content
3. Ensure code is runnable and produces expected outputs
4. Use type hints and docstrings in Python code
5. Test notebooks from a fresh kernel before committing

## Common Gotchas

- Notebooks must be run with appropriate API keys in environment
- Some RAG techniques require specific vector store setup
- MCP servers must be started before running the main agent
- Redis must be running for long-term memory examples
