# Prompt Engineering

Prompt engineering is the practice of designing and optimizing prompts to guide AI agents effectively. In AI agent development, well-crafted prompts serve as the primary interface between our intentions and the agent's behavior, defining how the agent thinks, responds and makes decisions.

### Implementation guide
Follow these steps to develop effective prompts for our AI agent system:

#### Step 1: Define clear goals and success criteria
Before writing any prompt, start by establishing exactly what we want our AI agent to accomplish. Vague goals lead to inconsistent outputs and wasted iteration time.

**What to define:**
- **Primary objective**: The main task or problem to solve
- **Target audience**: Who will consume the output (technical users, business stakeholders, end customers)
- **Output format**: The structure and type of response needed (JSON, markdown, natural language, structured data)
- **Success metrics**: How we will measure if the prompt works (accuracy, completeness, consistency, response time)

**Example:**
```python
from langchain.prompts import PromptTemplate

# Basic prompt (too vague)
basic_prompt = "Analyze this customer feedback"

# Better: Clear goal
goal_prompt = PromptTemplate(
    input_variables=["feedback_data"],
    template="""
    Analyze the following customer feedback and identify:
    1. Overall sentiment (positive/negative/neutral)
    2. Top 3 recurring themes
    3. Critical issues requiring immediate attention
    
    Feedback: {feedback_data}
    """
)
```

Write down the goals before touching any code. This document becomes our reference point during iteration and helps us avoid scope creep.

#### Step 2: Establish role, persona, and expertise
Now that you have a goal, give our agent a professional identity. This shapes how it thinks and responds.

**What to define:**
- **Professional role**: The expert persona the agent should embody
- **Domain expertise**: Specific knowledge areas and skills
- **Experience level**: Junior analyst, senior expert, specialist consultant
- **Perspective**: The lens through which the agent should view problems
- **Tone and style**: Professional, friendly, technical, educational

**Implementation pattern:**
```python
role_definition_prompt = PromptTemplate(
    input_variables=["task", "data"],
    template="""
    You are a Senior Customer Experience Analyst with 10+ years of expertise in:
    - Sentiment analysis and customer behavior patterns
    - Identifying actionable insights from feedback data
    - Prioritizing issues by business impact
    
    Your professional approach:
    - Data-driven insights backed by evidence
    - Balanced perspective considering multiple stakeholders
    - Clear communication tailored to non-technical audiences
    - Proactive identification of patterns and anomalies
    - Actionable recommendations grounded in industry best practices
    
    Task: {task}
    Data: {data}
    """
)
```

Role definition prevents generic responses. An agent acting as a "Senior Customer Experience Analyst" will provide more nuanced, professional insights than one with no defined expertise.

#### Step 3: Provide comprehensive context and background
With role established, add context that helps the agent understand your specific situation. The more relevant context we provide, the better our agent can adapt its reasoning to our specific situation.

**Context layers to include:**
- **Business context**: Company goals, market position, current challenges
- **Historical context**: Previous analysis, existing knowledge, past decisions
- **Situational context**: Current conditions, constraints, stakeholder concerns
- **Domain context**: Industry-specific knowledge, regulations, standards
- **Temporal context**: Time periods, deadlines, seasonal factors

**Implementation pattern:**
```python
contextual_prompt = PromptTemplate(
    input_variables=["task", "data", "business_context", "historical_context"],
    template="""
    You are a Senior Customer Experience Analyst with 10+ years of expertise in:
    - Sentiment analysis and natural language processing
    - Customer journey mapping
    - Quantitative and qualitative data analysis
    - UX research methodologies
    
    BUSINESS CONTEXT:
    {business_context}
    
    HISTORICAL CONTEXT:
    {historical_context}
    
    Your professional approach:
    - Data-driven insights backed by evidence
    - Balanced perspective considering multiple stakeholders
    - Clear communication tailored to non-technical audiences
    - Proactive identification of patterns and anomalies
    - Actionable recommendations grounded in industry best practices
    
    Current Task: {task}
    Data to Analyze: {data}
    """
)

# Usage with context
chain = contextual_prompt | llm
result = chain.invoke({
    "task": "Analyze Q4 customer feedback for our mobile app",
    "data": feedback_data,
    "business_context": "We're planning a major redesign in Q1 2026. Leadership wants to understand current pain points and prioritize features.",
    "historical_context": "Q3 analysis showed 65% satisfaction rate. We launched new checkout flow in September which received mixed feedback."
})
```

Context should be relevant and concise. Too much irrelevant context can confuse the agent; too little leaves it guessing.

#### Step 4: Define decision-making framework and reasoning steps
Now that context is clear, guide our agent's thinking process by providing an explicit reasoning framework. This ensures consistent, logical analysis rather than ad-hoc responses that vary in quality.

**Framework components:**
- **Step-by-step reasoning**: The logical sequence for approaching the task
- **Decision criteria**: Rules for making judgments or classifications
- **Reasoning methodology**: How to think about problems (first principles, comparative analysis, risk assessment)
- **Prioritization logic**: How to weight different factors
- **Edge case handling**: How to deal with ambiguous or unusual situations

**Implementation pattern:**
```python
framework_prompt = PromptTemplate(
    input_variables=["task", "data", "business_context", "historical_context"],
    template="""
    You are a Senior Customer Experience Analyst with 10+ years of expertise in:
    - Sentiment analysis and natural language processing
    - Customer journey mapping
    - Quantitative and qualitative data analysis
    - UX research methodologies
    
    BUSINESS CONTEXT:
    {business_context}
        
    ANALYSIS FRAMEWORK:
    
    Step 1 - Initial Assessment:
    - Review the complete dataset for overall patterns
    - Note the volume and distribution of feedback
    - Flag any obvious anomalies
    
    Step 2 - Sentiment Analysis:
    - Categorize each entry: Positive, Neutral, Negative, Mixed
    - Calculate sentiment distribution percentages
    - Identify sentiment trends over time if timestamps available
    
    Step 3 - Theme Identification:
    - Group feedback by common topics
    - Count frequency of each theme
    - Prioritize by: frequency × sentiment severity × business impact
    
    Step 4 - Critical Issues:
    - Identify issues blocking core functionality
    - Assess severity: Critical > High > Medium > Low
    - Estimate user impact (how many affected?)
    
    Step 5 - Recommendations:
    - Prioritize issues using impact assessment
    - Suggest specific, actionable next steps
    - Identify quick wins vs. long-term improvements

    REASONING GUIDELINES:
    - Support every claim with evidence from the data
    - Acknowledge uncertainty when patterns are ambiguous
    - Consider alternative interpretations of data
    - Think about cause-and-effect relationships
    - Connect insights to business objectives
    
    Current Task: {task}
    Data to Analyze: {data}
    
    Execute the analysis framework systematically, showing your reasoning at each phase.
    """
)
```

The framework shouldn't be rigid - allow the agent flexibility to adapt based on what it discovers. Phrases like "if relevant" or "when applicable" give the agent room to exercise judgment.

#### Step 5: Build input validation and data quality checks
With structure in place, add safeguards to handle real-world problems like missing data or ambiguous requests. Building validation directly into prompts prevents garbage-in-garbage-out scenarios and makes failures transparent.

**Validation categories:**
- **Completeness**: All required data fields are present
- **Format**: Data is in the expected structure and type
- **Quality**: Data meets minimum quality standards (no corruption, reasonable values)
- **Scope**: Data is sufficient for the requested analysis
- **Consistency**: Data doesn't contain obvious contradictions

**Implementation pattern:**
```python
validated_prompt = PromptTemplate(
    input_variables=["task", "data", "business_context", "historical_context"],
    template="""
    You are a Senior Customer Experience Analyst with 10+ years of expertise in:
    - Sentiment analysis and natural language processing
    - Customer journey mapping
    - Quantitative and qualitative data analysis
    - UX research methodologies
    
    BUSINESS CONTEXT:
    {business_context}
    
    HISTORICAL CONTEXT:
    {historical_context}

    [ANALYSIS FRAMEWORK - as defined in step 4]
    
    CRITICAL - VALIDATION REQUIREMENTS:
    Before proceeding with analysis, validate the following:
    
    1. DATA COMPLETENESS CHECK:
       - Confirm presence of: feedback text, timestamps, user IDs, ratings
       - Verify minimum sample size (at least 50 feedback entries)
       - Check for required metadata fields
    
    2. DATA QUALITY CHECK:
       - Identify any corrupted or unreadable entries
       - Flag feedback entries that are too short (<10 characters)
       - Note any missing or null critical fields
    
    3. DATA SCOPE CHECK:
       - Verify data covers the requested time period
       - Confirm data is from the correct product/platform
       - Check if sample is representative (not skewed to one user segment)
    
    4. DATA CONSISTENCY CHECK:
       - Ensure timestamps are logical and sequential
       - Verify rating scales are consistent
       - Check for obvious data entry errors
    
    VALIDATION PROTOCOL:
    - If validation fails, STOP and report specific issues
    - List exactly what data is missing or problematic
    - Suggest what information is needed to proceed
    - Do NOT make assumptions to fill gaps
    - Do NOT proceed with partial or uncertain data
    
    Current Task: {task}
    Data to Analyze: {data}
    
    Step 1: Perform validation checks above
    Step 2: Report validation results
    Step 3: Only if validation passes, proceed with analysis
    """
)
```

Validation failures caught early save time and prevent unreliable outputs. An agent that says "I need complete timestamp data to perform temporal analysis" is far more valuable than one that produces questionable insights from incomplete data.

#### Step 6: Implement comprehensive error handling
Real-world systems fail in unexpected ways. Our prompts must anticipate common failure modes and provide clear protocols for handling them gracefully.

**Error categories to handle:**
- **Data errors**: Missing, corrupted, or invalid data
- **Scope errors**: Task is outside agent's capabilities or knowledge
- **Ambiguity errors**: Request is unclear or has multiple valid interpretations
- **Constraint violations**: Task exceeds token limits, time constraints, or other boundaries
- **Logic errors**: Requested analysis is impossible or contradictory

**Implementation pattern:**
```python
robust_prompt = PromptTemplate(
    input_variables=["task", "data", "business_context", "historical_context"],
    template="""
    You are a Senior Customer Experience Analyst with 10+ years of expertise in:
    - Sentiment analysis and natural language processing
    - Customer journey mapping
    - Quantitative and qualitative data analysis
    - UX research methodologies
    
    BUSINESS CONTEXT:
    {business_context}
    
    HISTORICAL CONTEXT:
    {historical_context}
    
    [ANALYSIS FRAMEWORK - as defined in step 4]
    [VALIDATION SECTION - as defined in step 5]
    
    ERROR HANDLING PROTOCOLS:
    
    1. INSUFFICIENT DATA ERROR:
       If validation reveals data is incomplete or insufficient:
       - State clearly: "Cannot complete analysis due to insufficient data"
       - List specifically what data is missing
       - Explain why this data is needed for the requested analysis
       - Suggest: "To proceed, please provide: [specific requirements]"
       - Offer: "I can perform a limited analysis with caveats if you need preliminary insights"
    
    2. AMBIGUOUS REQUEST ERROR:
       If the task or requirements are unclear:
       - State: "The request needs clarification to ensure accurate analysis"
       - Ask specific questions: "Do you want analysis by: time period, user segment, or feature area?"
       - Propose: "I can interpret this as [Option A] or [Option B]. Which aligns with your needs?"
       - Do NOT guess at intent—always seek clarification
    
    3. SCOPE LIMITATION ERROR:
       If the task exceeds your capabilities:
       - State honestly: "This task requires [specific capability] which is outside my current scope"
       - Explain: "I can analyze customer feedback patterns, but cannot [specific limitation]"
       - Suggest alternatives: "I recommend [alternative approach or tool]"
       - Offer partial help: "I can assist with [related capability you do have]"
    
    4. DATA QUALITY ERROR:
       If data quality is questionable but analysis is possible:
       - Proceed with analysis but clearly flag quality concerns
       - State: "Analysis completed with the following data quality caveats:"
       - List each quality issue and its potential impact on results
       - Recommend: "Consider these findings preliminary pending data validation"
    
    5. CONTRADICTORY DATA ERROR:
       If data contains logical contradictions:
       - Highlight: "The following contradictions were identified in the data:"
       - List specific contradictions with examples
       - State: "Cannot provide reliable conclusions due to data inconsistencies"
       - Suggest: "Please review and resolve these contradictions before analysis"
    
    CONFIDENCE REPORTING:
    Label every insight with a confidence level:
    
    - HIGH CONFIDENCE: Based on complete, validated data with clear patterns
      Example: "High confidence: 73% of feedback mentions checkout issues (n=432 entries)"
    
    - MEDIUM CONFIDENCE: Based on partial data or moderate pattern strength
      Example: "Medium confidence: Loading speed appears problematic based on 45% of relevant feedback, though sample size is limited"
    
    - LOW CONFIDENCE: Based on limited data, weak patterns, or significant uncertainty
      Example: "Low confidence: Possible correlation with time of day, but data is too sparse to confirm"
    
    CRITICAL RULES:
    - Never fabricate data or make assumptions to fill gaps
    - Always acknowledge limitations and uncertainties
    - Prioritize accuracy over completeness
    - When in doubt, ask for clarification rather than guessing
    - If analysis cannot be completed reliably, say so explicitly
    
    Current Task: {task}
    Data to Analyze: {data}
    
    Begin with validation, apply the analysis framework, and handle any errors according to these protocols.
    """
)
```

Good error handling makes failures informative rather than frustrating. An agent that says "I need complete Q4 data to compare with Q3 baseline" is helping us solve the problem, not just failing silently.

#### Step 7: Add examples and demonstrations (few-shot learning)
Show our agent what excellent output looks like through concrete examples. Few-shot learning dramatically improves response quality by demonstrating patterns rather than just describing them.

**When to use examples:**
- Complex tasks with subjective judgment calls
- Domain-specific terminology or classification schemes
- Nuanced tone or style requirements
- Tasks where output quality varies significantly

**Implementation pattern:**
```python
few_shot_prompt = PromptTemplate(
    input_variables=["task", "data"],
    template="""
    [ROLE, CONTEXT, VALIDATION, FRAMEWORK sections as defined previously]
    
    EXAMPLE DEMONSTRATIONS:
    
    Study these examples of high-quality analysis to understand expected output:
    
    Example 1 - Identifying a Critical Theme:
    
    Input Feedback: "App crashes every time I try to complete checkout. Lost my cart twice now. Extremely frustrating."
    
    Quality Analysis:
    {{
      "theme_name": "Checkout Flow Crash - iOS Critical",
      "frequency": 47,
      "sentiment": "negative",
      "severity": "critical",
      "user_impact": "Blocks core purchase functionality, affecting estimated 12-15% of iOS users",
      "example_quotes": [
        "App crashes every time I try to complete checkout",
        "Can't finish paying, app closes immediately",
        "Lost three orders because app crashes at payment"
      ],
      "confidence": "high"
    }}
    
    Why this is good:
    - Specific theme name identifies platform and severity
    - Quantifies frequency (47 occurrences)
    - Clearly states user impact with scope estimate
    - Includes representative quotes showing pattern
    - Appropriate high confidence due to clear, repeated reports
    
    Example 2 - Identifying a Medium-Priority Theme:
    
    Input Feedback: "The search function could be better. Sometimes I can't find items I know you have."
    
    Quality Analysis:
    {{
      "theme_name": "Search Relevance - Incomplete Results",
      "frequency": 23,
      "sentiment": "negative",
      "severity": "medium",
      "user_impact": "Impacts product discovery, may lead to lost sales, affects subset of searches",
      "example_quotes": [
        "Search doesn't find items I know exist",
        "Search results miss obvious matches",
        "Have to browse categories because search fails"
      ],
      "confidence": "medium"
    }}
    
    Why this is good:
    - Descriptive but appropriately less urgent than critical issues
    - Moderate frequency (23 vs 47) reflected in severity
    - User impact explains business consequence
    - Medium confidence acknowledges this is less clear-cut than crashes
    
    Example 3 - Handling Ambiguous Feedback:
    
    Input Feedback: "App is slow sometimes"
    
    Quality Analysis:
    {{
      "theme_name": "Performance Issues - Unspecified",
      "frequency": 8,
      "sentiment": "negative",
      "severity": "low",
      "user_impact": "Unclear scope - insufficient detail to assess true impact",
      "example_quotes": [
        "App is slow sometimes",
        "Takes a while to load occasionally"
      ],
      "confidence": "low"
    }}
    
    Caveat: "Performance complaints are too vague to diagnose. Recommend follow-up survey asking: 
    - Which specific screens are slow?
    - What device/OS are you using?
    - How often does this occur?"
    
    Why this is good:
    - Acknowledges limitation (vague feedback)
    - Low confidence reflects uncertainty
    - Low severity due to insufficient information
    - Provides actionable next step (follow-up questions)
    - Doesn't over-interpret or assume
    
    APPLY THESE PATTERNS:
    - Be specific and quantitative like Example 1
    - Calibrate severity appropriately like Example 2
    - Acknowledge uncertainty like Example 3
    - Always explain your reasoning
    - Include business impact context
    - Provide representative evidence
    
    Now analyze the following:
    
    Current Task: {task}
    Data to Analyze: {data}
    """
)
```

Use 2-4 examples that cover different scenarios (clear-cut cases, edge cases, ambiguous situations). More examples generally improve quality but increase token usage.


#### Step 8: Create a prompt registry
With working prompts, create a centralized place for all our prompts:

```python
# prompts/prompt_registry.py
from langchain.prompts import ChatPromptTemplate

class PromptRegistry:
    @staticmethod
    def get_task_planning_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are a task planning agent. Break down complex tasks into clear, actionable steps."),
            ("human", "Task: {task}\nBreak this down into steps:")
        ])
    
    @staticmethod
    def get_information_extraction_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are an information extraction agent. Extract key information from the given text."),
            ("human", "Text: {text}\nExtract: {extraction_criteria}")
        ])
    
    @staticmethod
    def get_decision_making_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "You are a decision-making agent. Analyze options and provide recommendations."),
            ("human", "Options: {options}\nCriteria: {criteria}\nRecommend the best option and explain why.")
        ])
```

#### Step 9: Organize the prompts
Create a clear file structure:
```
prompts/
├── __init__.py
├── prompt_registry.py          # Central prompt storage
└── templates/
│   ├── analysis_prompts.py     # Analysis-related prompts
│   ├── generation_prompts.py   # Content generation prompts
│   └── validation_prompts.py   # Validation and error handling
```


### Integration with LangGraph
When using LangGraph, integrate prompts into the state graph:

```python
from langgraph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    task: str
    current_step: str
    results: list

def planning_node(state: AgentState):
    prompt = PromptRegistry.get_task_planning_prompt()
    chain = prompt | llm
    result = chain.invoke({"task": state["task"]})
    
    return {
        "current_step": "planning_complete",
        "results": state["results"] + [result.content]
    }

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("planner", planning_node)
# Add more nodes...
```

## Best practices
1. **Start simple**: Begin with basic prompts and gradually add complexity
2. **Be specific**: Clear, specific prompts get better results
3. **Test regularly**: Always test prompts with real use cases
4. **Version control**: Keep track of prompt changes
5. **Reuse components**: Build a library of reusable prompt components

## Next Steps
Once we have basic prompt engineering working:
1. Experiment with different prompt styles for specific use cases
2. Build more sophisticated multi-agent workflows
3. Optimize prompts based on performance metrics
4. Create domain-specific prompt templates

Remember: Start with simple, clear prompts and build complexity gradually. Good prompt engineering is the foundation of effective AI agents.