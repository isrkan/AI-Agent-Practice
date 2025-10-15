# Prompt Engineering Techniques

This directory contains my practical explorations and implementations of various prompt engineering techniques used in LLMs. It serves as a personal learning repository where I experiment with different strategies to optimize model performance, guide generation behavior, and adapt responses to specific tasks or user intents.

The techniques covered span a wide range of prompt design principles, from basic formulations to more advanced structures for complex applications. These include methods for improving instruction clarity, injecting domain-specific context, handling multi-turn interactions, constraining output formats, and more. The goal is to develop a deeper understanding of how prompt phrasing influences model behavior across different use cases in natural language processing.

## Acknowledgements

This work is inspired by and builds upon techniques demonstrated in the excellent [Prompt Engineering Techniques Repository](https://github.com/NirDiamant/Prompt_Engineering) of Nir Diamant. I have adapted and extended the ideas for prompt engineering as part of my own learning journey to better understand and structure the principles involved in effective prompt design.

## Purpose

This folder serves as both a personal reference and practice ground for prompt engineering techniques that I have studied, applied, and in some cases, modified. It reflects an ongoing process of experimentation, iteration, and refinement in leveraging LLMs more effectively through thoughtful prompt construction.

## Summary of prompt engineering techniques

### Prompt templates and variables
This notebook introduces how to create and use prompt templates with variables to enhance interactions with AI language models. Templates make our prompts easier to manage, more adaptable, and reusable across different use cases. We will explore how to generate dynamic prompts for OpenAI's language models using two approaches: LangChain's `PromptTemplate` for simple, structured inputs, and custom Jinja2-based templates for advanced use cases like conditionals and loops.

### Zero-shot prompting
This notebook introduces zero-shot prompting, which enables language models to perform tasks without being given any examples beforehand. Zero-shot prompting allows AI models to generalize across many different tasks with minimal setup and without the need for training or fine-tuning on specific examples. It’s effective, scalable, and easy to implement.

### Few shot learning and in-context learning
In this notebook, we explore few-shot learning and in-context learning, two paradigms that allow LLMs to generalize and solve new tasks with minimal examples - often just a handful. These techniques are especially valuable in situations where labeled data is scarce, and traditional fine-tuning is impractical.

### Single-turn and multi-turn prompting
In this notebook, we explore two foundational ways of interacting with LLMs: Single-turn prompts — isolated interactions with the model and multi-turn prompts — ongoing conversations where the model maintains context between turns. Single-turn prompts are great for quick tasks, while multi-turn prompts unlock more natural, human-like interactions.

### Chain of thought prompting
This notebook introduces chain of thought (CoT) prompting - a technique that guides LLMs to solve problems by reasoning through them step-by-step. Much like how a human would “show their work” in math or logic, CoT prompting encourages the model to think aloud before arriving at a final answer. This approach is especially helpful for tasks that involve multiple steps, like math problems, logic puzzles, or anything that requires reasoning beyond a simple lookup.

### Constrained guided generation
This notebook explores the concepts of constrained and guided generation using LLMs. Instead of letting the model freely generate text, we structure and restrict the generation process to produce more precise, rule-abiding, and goal-oriented outputs. This approach not only improves the usability and accuracy of the generated text, but also makes LLMs better suited for integration into structured workflows and applications where format adherence is non-negotiable.

### Handling ambiguity and improving clarity
This notebook focuses on two aspects of prompt engineering: identifying and resolving ambiguous prompts, and techniques for writing clearer prompts. Ambiguous or vague prompts often lead to suboptimal or inconsistent results, which can make AI systems harder to work with.

### Prompt formatting and structure
In this notebook, we will explore different ways to structure prompts. When interacting with LLMs, the way we phrase and structure prompts plays a critical role in shaping the model's responses. We will investigate how changes in format (e.g., Q&A, dialogue, structured instructions) and layout elements (e.g., lists, headings) influence the model's output.

### Role prompting
This notebook explores the concept of role prompting in LLMs - a technique used to steer the model’s behavior by assigning it a clearly defined role or perspective. By embedding the right context into the prompt - such as defining the model as a financial advisor, a journalist, or a middle school teacher - we can elicit outputs that are more relevant, focused, and appropriate to a given task or audience. This approach is especially useful in domains that demand domain-specific knowledge, tone control, or user-centric explanations.

### Ethical prompt engineering
As language models become increasingly central to decision-making systems, content generation, and user-facing applications, it's essential that they behave in ways that are not only intelligent but responsible. This notebook walks through how to detect and mitigate those risks by designing more inclusive, balanced, and fair prompts. We will also introduce simple evaluation techniques to audit model outputs and reflect on their fairness.

### Evaluating prompt effectiveness
Evaluating how well a prompt performs is crucial for any application that uses language models. The quality of the prompt directly affects the model's output in terms of relevance, consistency, specificity, and clarity. In this notebook, we aim to build a toolkit that allows us to evaluate prompts using both manual and automated methods.

### Instruction engineering
This notebook focuses on instruction engineering, a critical part of prompt engineering that focuses on writing clear, structured, and effective instructions for language models. The quality of instructions directly influences the relevance, accuracy, and usability of the model's output. In this notebook, we will explore techniques for designing effective instructions, balancing the level of specificity and generality, and refining prompts iteratively for optimal results.

### Negative prompting
This notebook demonstrates how to guide and constrain model output using negative prompting, a technique that involves explicitly stating what the model should avoid doing. When working with LLMs, we often think about what should be included in a response—but it's just as important to think about what shouldn't. Our goal is to influence the tone and content of LLM responses by excluding certain behaviors or topics.

### Prompt chaining and sequencing
This notebook explores the concepts of prompt chaining and sequencing in the context of working with LLMs. These approaches enable us to guide LLMs through a series of interconnected steps, where the output from one prompt becomes the input for the next. This method is particularly valuable for tasks that require nuanced reasoning, multiple stages of processing, or decisions that need to be informed by prior steps.

### Prompt length and complexity management
In this notebook, we explore how to manage prompt length and complexity when working with LLMs. Two core challenges often arise in AI interactions: ensuring prompts are clear and concise, and handling long or complex text inputs that might exceed the model's token limit with techniques such as chunking, summarization or iterative processing. By managing these factors efficiently, we can enhance the quality and relevance of model responses, and ensure that even lengthy or multifaceted queries can be processed effectively.

### Specific task prompts
In this notebook, we explore how to engineer effective prompts for specific tasks using OpenAI's GPT model. For each task, we will build prompt templates, feed sample inputs, and analyze how the prompt design impacts the model’s output. The goal is to understand how to construct prompts that guide the model toward producing useful, task-appropriate responses.

### Task decomposition in prompts
This notebook explores task decomposition in prompt engineering, a technique that breaks down complex problems into smaller, more manageable subtasks. This approach is key to effectively leveraging LLMs for multi-step reasoning tasks. By decomposing a problem into simpler parts, we improve the model's ability to produce more reliable, accurate, and interpretable results.

### Multilingual and cross-lingual prompting
In this notebook, we explore how to design prompts for handling multiple languages and enabling cross-lingual interactions using LLMs. The goal is to build prompts that can operate seamlessly across various linguistic contexts - such as greeting users in different languages, adapting responses based on the detected input language, performing translations, and handling non-Latin scripts with cultural nuance. This kind of prompting is crucial for building inclusive applications like international chatbots, educational tools, and translation services that must support users from diverse linguistic backgrounds.

### Multiple reasoning paths and self consistency
In this notebook, we explore a strategy for improving the reliability and quality of responses from LLMs: generating multiple reasoning paths and using self-consistency as a way to evaluate and refine the final answer. While large language models like GPT are powerful, they may not always give the correct or most consistent answer on the first try - especially for tasks that involve complex reasoning, multiple steps, or some ambiguity. This technique has several advantages. It helps reduce variability and randomness in outputs, exposes possible reasoning flaws, and makes the model’s thought process more transparent.

### Prompt optimization techniques
This notebook explores practical methods for improving prompt effectiveness when working with LLMs. We focus on two main techniques: A/B testing and iterative refinement. These strategies help us fine-tune prompt phrasing to yield better, more relevant responses from the model.

### Prompt security and safety
In this notebook, we explore practical strategies to secure AI language models from prompt injection attacks and to filter unsafe or inappropriate content. As AI systems are increasingly integrated into user-facing applications, safeguarding them against malicious inputs and outputs is essential. Prompt injections are attempts by users to manipulate the model's behavior by embedding harmful instructions in natural language input, while content filtering ensures that the AI's responses remain within ethical and appropriate boundaries.