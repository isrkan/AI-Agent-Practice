# 🤖 AI Chatbot Streamlit App
A production-ready AI chatbot application built with Streamlit that demonstrates key deployment patterns for AI agents in real-world applications.

## 📋 Overview
This is a basic, simple foundational application focused on deploying AI agents in production. This app demonstrates the practical implementation aspects of building user-facing AI applications.


## ✨ Features
- **Interactive chat interface**: Real-time conversation with OpenAI's GPT model
- **File processing**: Upload and analyze TXT/PDF documents
- **Document summarization**: AI-powered text summarization
- **Chat history**: Persistent conversation memory during session
- **Export functionality**: Download chat history as text file

## 🚀 Setup

1. **Install dependencies**:
   ```bash
   pip install streamlit openai python-dotenv PyPDF2
   ```

2. **Configure environment**:
   Create `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Run application**:
   ```bash
   streamlit run app.py
   ```