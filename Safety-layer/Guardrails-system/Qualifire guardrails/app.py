import os
import streamlit as st
import qualifire
import openai
import PyPDF2
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env
# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QUALIFIRE_API_KEY = os.getenv("QUALIFIRE_API_KEY")
# Validate keys
if not OPENAI_API_KEY or not QUALIFIRE_API_KEY:
    raise ValueError("OPENAI_API_KEY and QUALIFIRE_API_KEY must be set")


# Initialize Qualifire SDK
qualifire.init(api_key=QUALIFIRE_API_KEY)
q_client = qualifire.client.Client(api_key=QUALIFIRE_API_KEY)

# Configure OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# FILE HANDLING FUNCTIONS
# -----------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file):
    """
    Extracts text content from a PDF file.

    Args:
        pdf_file: Uploaded PDF file object

    Returns:
        str: Extracted text
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text


def process_uploaded_file(uploaded_file):
    """
    Extracts text from uploaded file (TXT or PDF).

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        str: File content or error message
    """
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        return "Unsupported file type. Please upload a TXT or PDF file."


# -----------------------------------------------------------------------------
# AI FUNCTION
# -----------------------------------------------------------------------------

# Define a function to generate a response from the AI given a user message
def generate_response(user_prompt, file_content=None):
    """
    Sends user input (and optional file content) to the AI and returns a response.

    Args:
        user_prompt (str): The user's query
        file_content (str, optional): Uploaded file content

    Returns:
        str: AI-generated response
    """
    messages = []

    # If file content is provided, add it as context
    if file_content:
        messages.append({
            "role": "system",
            "content": (
                f"The user has uploaded a file with the following content:\n\n"
                f"{file_content}\n\n"
                f"Please consider this information when responding."
            )
        })

    # Add the user's prompt
    messages.append({"role": "user", "content": user_prompt})

    # Request AI response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",  # The AI model to use
        messages=messages,  # The conversation context
    )

    # Extract the assistant's message from the response
    message_text = response.choices[0].message.content

    # Evaluate with Qualifire, triggering all guardrails
    q_client.evaluate(
        user_prompt,  # The user's original prompt or question
        message_text,  # The AI-generated response text to evaluate
        prompt_injections=True,  # Detect if the model is tricked into following malicious instructions
        pii_check=True,  # Detect potential leakage of Personally Identifiable Information
        hallucinations_check=True,  # Detect statements that are factually incorrect or fabricated
        dangerous_content_check=True,  # Detect unsafe or harmful instructions/content
        harassment_check=True,  # Detect content that could be harassing or bullying
        hate_speech_check=True,  # Detect discriminatory or hate speech
        sexual_content_check=True,  # Detect sexually explicit content
        assertions=["don\'t give medical advice", "don\'t output PII"]  # Additional assertions we want the model to respect
    )

    return message_text  # Return the assistant's reply


# -----------------------------------------------------------------------------
# STREAMLIT APP LAYOUT
# -----------------------------------------------------------------------------
# Page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
# App header
st.title("🤖 AI Chatbot Assistant")
st.markdown("**Welcome!** Ask anything or upload a file for the bot to analyze.")


# -----------------------------------------------------------------------------
# SIDEBAR: FILE UPLOAD
# -----------------------------------------------------------------------------
# File upload section in the sidebar
uploaded_file = st.sidebar.file_uploader(
    "📎 Upload a file (optional):", type=["txt", "pdf"]
)
file_content = None

if uploaded_file is not None:
    # Display file metadata
    st.sidebar.write(
        "📄 Uploaded file:", f"**{uploaded_file.name}** ({uploaded_file.size} bytes)"
    )

    # Process the file and store its content
    with st.sidebar.spinner("Processing file..."):
        file_content = process_uploaded_file(uploaded_file)

    # Show a preview of the file content
    with st.sidebar.expander("File Content Preview"):
        st.write(
            file_content[:500] + "..." if len(file_content) > 500 else file_content
        )


# -----------------------------------------------------------------------------
# CHAT INTERFACE
# -----------------------------------------------------------------------------
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Add welcome message if chat is empty
if not st.session_state.messages:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "👋 Hello! I'm here to help. Feel free to ask me anything or upload a file.",
        }
    )

# Display past chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input widget for new messages
if user_msg := st.chat_input("Type your message here..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)
    # Generate AI response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_msg = generate_response(user_msg, file_content)
            st.markdown(assistant_msg)
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})