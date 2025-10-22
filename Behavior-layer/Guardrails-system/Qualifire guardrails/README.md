# 🤖 AI Chatbot with Qualifire Monitoring

A Streamlit-based AI chatbot application powered by OpenAI models and safeguarded with Qualifire’s real-time evaluation framework. This project demonstrates how to integrate Qualifire guardrails into an AI app to monitor and enforce safety, reliability, and compliance in user-facing AI systems.

## 📋 Overview
This chatbot lets users:
* Chat interactively with an AI model.
* Get real-time evaluations of AI responses using Qualifire, including:
  * Prompt injection detection.
  * PII (Personally Identifiable Information) protection.
  * Hallucination checks.
  * Harmful/dangerous content detection.
  * Harassment & hate speech detection
  * Sexually explicit content detection.
  * Custom protection rules.

This app ensures safer, production-grade AI deployment by combining AI generation with automated monitoring and guardrails.

## Why Qualifire?
AI models are powerful but can sometimes make mistakes — like leaking sensitive data, giving unsafe instructions or being tricked by malicious prompts. In production, these risks can harm users and break compliance rules.

**Qualifire helps by:**
* Catching unsafe or non-compliant responses before they reach users
* Enforcing custom business rules automatically
* Giving visibility into system performance with dashboards and traces

This makes our chatbot safer, more reliable, and ready for real-world use.

## ✨ Features
* **Interactive chat interface**: Talk with an AI assistant.
* **File upload support**: Upload TXT/PDF files for analysis.
* **Contextual responses**: AI uses uploaded documents to enhance answers.
* **Qualifire guardrails**: Automatic safety checks on every AI response.
* **Custom policy enforcement**: Define custom assertions.

## 🔧 Setup

1. **Install dependencies:**
    ```bash
    pip install streamlit openai python-dotenv PyPDF2 qualifire
    ```

2. **Configure environment:**
Create a `.env` file in the root folder with the following content:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    QUALIFIRE_API_KEY=your_qualifire_api_key_here
    ```

3. **Run the application:**
    ```bash
    streamlit run app.py
    ```

4. **Interact with the chatbot:** Ask questions or upload TXT/PDF files for analysis. Each AI response will automatically be evaluated by Qualifire.

5. **Inspect detailed traces in Qualifire:** Go to [Traces](https://app.qualifire.ai/traces/) to review logs of each conversation, including user prompts, AI responses, and evaluation results (e.g., flagged PII, hallucination or policy violations). It is useful for debugging, audits and compliance reviews.

6. **Monitor system health in the dashboard:** Use [Dashboard](https://app.qualifire.ai/dashboard/) to get aggregated insights, trends and compliance metrics across all chatbot interactions.