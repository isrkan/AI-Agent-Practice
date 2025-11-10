# Structured Output

Structured output is the practice of ensuring AI agents return data in predictable, machine-readable formats rather than free-form text. In AI agent development, structured output transforms unpredictable natural language responses into well-defined data structures (like JSON objects, Pydantic models, or typed dictionaries) that can be reliably processed, validated, and integrated into larger systems. This is essential for building production-ready AI agents that need to interact with databases, APIs, and other software components.

When building AI agents, unstructured text responses create several challenges:
- **Unpredictable formats**: Free-form text varies in structure, making parsing unreliable
- **Validation difficulties**: Hard to verify that responses contain required information
- **Integration complexity**: Difficult to pass data between systems and components
- **Error handling**: Challenging to detect and recover from malformed responses
- **Type safety**: No guarantees about data types, leading to runtime errors

Structured output solves these problems by:
- **Enforcing schemas**: Defining exact data structures the agent must follow
- **Enabling validation**: Automatically checking that responses match expected formats
- **Simplifying integration**: Providing consistent interfaces for downstream systems
- **Improving reliability**: Reducing parsing errors and unexpected failures
- **Supporting type checking**: Catching errors at development time rather than runtime

## Implementation guide
Follow these steps to implement structured output in the AI agent system:

### Step 1: Define the data requirements
Before implementing structured output, clearly identify what information the agent needs to extract or generate. This becomes our schema definition.

**What to define:**
- **Required fields**: Information that must always be present
- **Optional fields**: Information that may or may not be available
- **Data types**: Whether each field is a string, number, boolean, list, or nested object
- **Validation rules**: Constraints like minimum/maximum values, string patterns, or allowed options
- **Field descriptions**: Clear explanations that help the LLM understand what to extract

**Example:** For a customer feedback analysis agent, we might need:
- Sentiment (required): positive/negative/neutral
- Confidence score (required): 0.0 to 1.0
- Key themes (required): list of topics mentioned
- Specific issues (optional): list of problems identified
- Suggested actions (optional): list of recommended responses

### Step 2: Choose the appropriate schema type
LangChain supports multiple schema types, each with different trade-offs:

**Pydantic models** (Recommended for most cases):
- Full validation support with custom validators
- Rich type system with nested models
- Automatic documentation generation
- Best for complex, production applications

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class FeedbackAnalysis(BaseModel):
    """Structured analysis of customer feedback"""
    sentiment: Sentiment = Field(description="Overall sentiment of the feedback")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)
    key_themes: List[str] = Field(description="Main topics or themes mentioned")
    specific_issues: Optional[List[str]] = Field(default=None, description="Specific problems identified")
    suggested_actions: Optional[List[str]] = Field(default=None, description="Recommended responses")
```

**Python dataclasses** (For simpler cases):
- Lighter weight than Pydantic
- Basic type hints without complex validation
- Good for straightforward data structures

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FeedbackAnalysis:
    sentiment: str
    confidence: float
    key_themes: List[str]
    specific_issues: Optional[List[str]] = None
    suggested_actions: Optional[List[str]] = None
```

**TypedDict** (For dictionary-based workflows):
- Type hints for dictionaries
- No runtime validation
- Useful when working with existing dict-based code

```python
from typing import TypedDict, List, Optional

class FeedbackAnalysis(TypedDict):
    sentiment: str
    confidence: float
    key_themes: List[str]
    specific_issues: Optional[List[str]]
    suggested_actions: Optional[List[str]]
```

**When to use each:**
- **Pydantic**: Production applications, complex validation, nested structures
- **Dataclasses**: Simple schemas, minimal dependencies, basic type checking
- **TypedDict**: Legacy code integration, dictionary-heavy workflows, no validation needed

### Step 3: Select the structured output strategy
LangChain provides two strategies for generating structured output, each with different capabilities:

**Provider strategy**:
- Uses the LLM's native structured output capabilities
- Best performance and reliability
- Supported by OpenAI, Anthropic, and other major providers
- Automatically falls back to best available method

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create structured output chain with provider strategy
structured_llm = llm.with_structured_output(
    FeedbackAnalysis,
    method="json_schema",  # Uses provider's native support
    include_raw=True  # Include raw response for debugging
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer feedback analyst. Extract structured insights from feedback."),
    ("human", "{feedback_text}")
])

chain = prompt | structured_llm
```

**Tool calling strategy** (For complex multi-step workflows):
- Converts schema into a function the LLM can call
- Useful for agents that need to choose between multiple output formats
- More overhead but greater flexibility

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=llm,
    tools=[],
    response_format=ToolStrategy(
        schema=FeedbackAnalysis,
        tool_message_content="Feedback analysis completed successfully"
    )
)
```

**When to use each:**
- **Provider strategy**: Production applications, best performance, native LLM support
- **Tool calling strategy**: Multi-format outputs, complex agent workflows, function calling needed

### Step 4: Implement the structured output chain
Create a complete chain that processes input and returns structured output:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define schema (from Step 2)
class FeedbackAnalysis(BaseModel):
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)
    key_themes: List[str] = Field(description="Main topics mentioned in the feedback")
    specific_issues: Optional[List[str]] = Field(default=None, description="Specific problems identified")
    suggested_actions: Optional[List[str]] = Field(default=None, description="Recommended responses")

# Initialize LLM with structured output (Step 3)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,  # Deterministic outputs for consistency
    timeout=30,  # Prevent hanging
    max_retries=3  # Automatic retry on failures
)

structured_llm = llm.with_structured_output(
    FeedbackAnalysis,
    method="json_schema",
    include_raw=True  # Keep raw response for debugging
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a customer feedback analyst. Analyze the feedback and extract:
    - Overall sentiment (positive, negative, or neutral)
    - Confidence in your analysis (0.0 to 1.0)
    - Key themes or topics mentioned
    - Specific issues if any are identified
    - Suggested actions for addressing the feedback
    
    Be thorough but concise in your analysis."""),
    ("human", "{feedback_text}")
])

# Build the chain
feedback_analysis_chain = prompt | structured_llm

# Use the chain
try:
    feedback = """The product quality is excellent and delivery was fast, but the packaging 
    was damaged and customer support took 3 days to respond to my inquiry about returns."""
    
    result = feedback_analysis_chain.invoke({"feedback_text": feedback})
    
    # Access structured output
    analysis = result["parsed"]
    logger.info(f"Analysis completed: {analysis.sentiment} (confidence: {analysis.confidence})")
    logger.info(f"Key themes: {', '.join(analysis.key_themes)}")
    
    if analysis.specific_issues:
        logger.warning(f"Issues identified: {', '.join(analysis.specific_issues)}")
    
    if analysis.suggested_actions:
        logger.info(f"Suggested actions: {', '.join(analysis.suggested_actions)}")
        
except Exception as e:
    logger.error(f"Analysis failed: {str(e)}")
    raise
```

**Key implementation principles:**
- **Set temperature to 0**: Ensures consistent, deterministic outputs
- **Include timeouts**: Prevents hanging on slow responses
- **Enable retries**: Automatically recovers from transient failures
- **Use include_raw=True**: Keeps raw response for debugging
- **Add comprehensive logging**: Tracks execution and identifies issues
- **Implement error handling**: Gracefully handles failures

### Step 5: Integrate with LangGraph for stateful workflows
For complex AI agents that maintain state across multiple interactions, integrate structured output with LangGraph:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define agent state with structured output
class AgentState(TypedDict):
    feedback_text: str
    analysis: Optional[FeedbackAnalysis]
    conversation_history: Annotated[list, operator.add]
    retry_count: int

# Define nodes that use structured output
def analyze_feedback(state: AgentState) -> AgentState:
    """Analyze feedback and return structured output"""
    try:
        result = feedback_analysis_chain.invoke({"feedback_text": state["feedback_text"]})
        analysis = result["parsed"]
        
        return {
            **state,
            "analysis": analysis,
            "conversation_history": [{"role": "assistant", "content": f"Analysis complete: {analysis.sentiment}"}],
            "retry_count": 0
        }
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return {
            **state,
            "retry_count": state.get("retry_count", 0) + 1
        }

def should_retry(state: AgentState) -> str:
    """Decide whether to retry analysis"""
    if state.get("analysis") is not None:
        return "complete"
    elif state.get("retry_count", 0) < 3:
        return "retry"
    else:
        return "failed"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze", analyze_feedback)

# Add edges
workflow.set_entry_point("analyze")
workflow.add_conditional_edges(
    "analyze",
    should_retry,
    {
        "complete": END,
        "retry": "analyze",
        "failed": END
    }
)

# Compile the graph
app = workflow.compile()

# Use the graph
initial_state = {
    "feedback_text": "Product is great but shipping was slow",
    "analysis": None,
    "conversation_history": [],
    "retry_count": 0
}

final_state = app.invoke(initial_state)
if final_state["analysis"]:
    print(f"Sentiment: {final_state['analysis'].sentiment}")
    print(f"Themes: {', '.join(final_state['analysis'].key_themes)}")
else:
    print("Analysis failed after retries")
```

**LangGraph integration benefits:**
- **State management**: Maintains structured data across multiple steps
- **Conditional logic**: Routes based on structured output content
- **Retry mechanisms**: Automatically retries failed extractions
- **Complex workflows**: Chains multiple structured output operations
- **Debugging**: Visualizes state transitions and data flow

### Step 6: Handle nested and complex structures
We often need to extract hierarchical data with multiple levels of nesting:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

# Define nested models for hierarchical data
class CustomerInfo(BaseModel):
    """Customer identification and contact details"""
    customer_id: Optional[str] = Field(default=None, description="Unique customer identifier")
    name: str = Field(description="Customer full name")
    email: Optional[str] = Field(default=None, description="Contact email")
    account_type: str = Field(description="Account tier: basic, premium, or enterprise")

class ProductMention(BaseModel):
    """Product referenced in feedback"""
    product_name: str = Field(description="Name of the product")
    product_id: Optional[str] = Field(default=None, description="Product SKU or ID")
    sentiment: str = Field(description="Sentiment about this specific product")
    issues: Optional[List[str]] = Field(default=None, description="Issues with this product")

class ActionItem(BaseModel):
    """Recommended action to take"""
    priority: str = Field(description="Priority level: high, medium, or low")
    action: str = Field(description="Specific action to take")
    department: str = Field(description="Department responsible: support, product, sales, etc.")
    deadline: Optional[date] = Field(default=None, description="Suggested completion date")

class ComprehensiveFeedbackAnalysis(BaseModel):
    """Complete structured analysis of customer feedback"""
    customer: CustomerInfo = Field(description="Customer information")
    overall_sentiment: str = Field(description="Overall feedback sentiment")
    confidence: float = Field(description="Analysis confidence score", ge=0.0, le=1.0)
    products_mentioned: List[ProductMention] = Field(description="Products referenced in feedback")
    key_themes: List[str] = Field(description="Main topics or themes")
    action_items: List[ActionItem] = Field(description="Recommended actions")
    feedback_date: date = Field(description="Date feedback was received")
    requires_immediate_attention: bool = Field(description="Whether this needs urgent response")

# Use with structured output
complex_llm = llm.with_structured_output(
    ComprehensiveFeedbackAnalysis,
    method="json_schema"
)

complex_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an advanced customer feedback analyst. Extract comprehensive structured 
    information including customer details, product mentions, sentiment analysis, and action items.
    Be thorough and identify all relevant information from the feedback."""),
    ("human", "{feedback_text}")
])

complex_chain = complex_prompt | complex_llm
```

**Best practices for nested structures:**
- **Keep nesting depth reasonable**: Avoid more than 3-4 levels of nesting
- **Use clear field descriptions**: Help the LLM understand each nested component
- **Make optional fields explicit**: Use `Optional[]` for fields that may not be present
- **Validate nested data**: Add validators to nested models, not just top-level
- **Provide examples**: Include example outputs in our prompts for complex structures