# AI workflow patterns

This folder contains practice examples and implementations of AI workflow patterns, which define how tasks and processes are structured, sequenced, and executed in AI systems. Workflow patterns organize and coordinate the operational flow of tasks, often involving multiple steps that can be performed sequentially, in parallel, or conditionally routed based on input types.

Workflow patterns describe the designs used to break down complex tasks into manageable units, distribute the processing among specialized agents or components, and orchestrate the overall execution. They focus on task management and process automation in AI workflows. These patterns help develop robust AI pipelines with improved scalability, efficiency, and modularity by decomposing, organizing, and controlling task execution flows. They are foundational to building complex AI-driven automation and multi-agent workflows.

### Example workflow patterns
- **Prompt chaining:** Splits a complex task into a sequence of smaller prompts or sub-steps, where the output of one step feeds as input into the next. Useful for stepwise problem-solving or multi-stage reasoning.
- **Parallelization:** Divides a larger task into smaller independent subtasks that are processed simultaneously, speeding up completion when parallel execution is possible.
- **Routing:** Classifies incoming inputs and intelligently directs them to different specialized processes or agents capable of handling the specific type of input.
- **Orchestrator-worker:** Uses a central orchestrator component to break tasks into subtasks and assign them to multiple worker agents, then aggregates the worker results into a final output.
- **Evaluator-optimizer:** Runs an iterative loop where one agent generates outputs and another evaluates and provides feedback, refining results over repeated cycles for optimization.