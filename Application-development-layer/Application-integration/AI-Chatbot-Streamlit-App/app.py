import streamlit as st
import os
from dotenv import load_dotenv
import openai
import PyPDF2

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Load environment variables (for OpenAI API key)
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


# =============================================================================
# FILE HANDLING FUNCTIONS
# =============================================================================

def extract_text_from_pdf(pdf_file):
    """
    Extract text content from a PDF file.

    Args:
        pdf_file: Uploaded PDF file object

    Returns:
        str: Extracted text content
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def process_uploaded_file(uploaded_file):
    """
    Process uploaded file and extract text content based on file type.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        str: Extracted text content or error message
    """
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        return "Unsupported file type. Please upload a TXT or PDF file."


# -----------------------------------------------------------------------------
# OPENAI FUNCTIONS
# -----------------------------------------------------------------------------

def generate_response(user_prompt, file_content=None):
    """
    Generate AI response using OpenAI's chat completion API.

    Args:
        user_prompt (str): The user's input message
        file_content (str, optional): Content from uploaded file

    Returns:
        str: AI-generated response
    """
    messages = []

    # Add file content as system context if provided
    if file_content:
        messages.append({
            "role": "system",
            "content": f"The user has uploaded a file with the following content:\n\n{file_content}\n\nPlease consider this information when responding to their query."
        })

    # Add user's prompt
    messages.append({"role": "user", "content": user_prompt})

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


def summarize_text(text):
    """
    Generate a summary of the provided text.

    Args:
        text (str): Text to summarize

    Returns:
        str: Summary of the text
    """
    # Limit text to first 3000 characters to avoid token limits
    summary_prompt = f"Summarize the following text:\n\n{text[:3000]}"
    return generate_response(summary_prompt)


# -----------------------------------------------------------------------------
# CHAT UTILITIES
# -----------------------------------------------------------------------------

def download_chat(messages):
    """
    Format chat messages for download.

    Args:
        messages (list): List of chat message dictionaries

    Returns:
        str: Formatted chat log
    """
    chat_log = "\n\n".join(
        f"{msg['role'].capitalize()}:\n{msg['content']}"
        for msg in messages
    )
    return chat_log


# =============================================================================
# SIDEBAR COMPONENTS
# =============================================================================

def render_sidebar():
    """Render sidebar with file upload and chat controls."""
    st.sidebar.title("📁 Document Hub & Controls")

    # File upload section
    uploaded_file = st.sidebar.file_uploader(
        "📎 Upload a file (optional):",
        type=["txt", "pdf"],
        help="Drop a TXT or PDF file here to unlock document-powered conversations"
    )

    file_content = None

    if uploaded_file is not None:
        # Display file info
        st.sidebar.write(
            "📄 Uploaded file:",
            f"**{uploaded_file.name}** ({uploaded_file.size} bytes)"
        )

        # Process the file
        with st.spinner("Processing file..."):
            file_content = process_uploaded_file(uploaded_file)

        # Show file content preview
        with st.sidebar.expander("👀 Document Preview"):
            preview_text = (
                file_content[:500] + "..."
                if len(file_content) > 500
                else file_content
            )
            st.write(preview_text)

        # Summarize button
        if st.sidebar.button("🧠 Summarize File"):
            with st.spinner("Generating summary..."):
                summary = summarize_text(file_content)
            st.sidebar.subheader("📌 File Summary")
            st.sidebar.write(summary)

    # -------------------------
    # Chat controls
    # -------------------------
    st.sidebar.divider()
    st.sidebar.subheader("💬 Chat Controls")

    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Download chat button
    if st.session_state.messages:
        if st.sidebar.download_button(
                label="💾 Download Chat",
                data=download_chat(st.session_state.messages),
                file_name="chat_history.txt",
                use_container_width=True
        ):
            st.sidebar.success("Chat downloaded!")

    return file_content


# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

def render_chat_interface(file_content):
    """Render the main chat interface."""

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_input := st.chat_input("💭 Share your thoughts, ask questions, or request analysis..."):
        # Add user message to history and display
        user_msg = {"role": "user", "content": user_input}
        st.session_state.messages.append(user_msg)

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                assistant_response = generate_response(user_input, file_content)
                st.markdown(assistant_response)

        # Add assistant response to history
        assistant_msg = {"role": "assistant", "content": assistant_response}
        st.session_state.messages.append(assistant_msg)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""

    # Initialize session state first
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add welcome message if no messages exist
    if not st.session_state.messages:
        welcome_msg = {
            "role": "assistant",
            "content": "Hello! I'm here to help. Feel free to ask me anything or upload a file for analysis."
        }
        st.session_state.messages.append(welcome_msg)

    # App header
    st.title("🤖 AI Chatbot Assistant")
    st.markdown(
        "**Welcome!** Ask me anything or upload a file (TXT/PDF) for analysis. "
        "I'll help you with questions, document analysis, and more!"
    )
    st.divider()

    # Render sidebar and get file content
    file_content = render_sidebar()

    # Render main chat interface
    render_chat_interface(file_content)

    # Footer
    st.sidebar.divider()
    st.sidebar.markdown(
        "💡 **Tip:** Upload documents to get contextual responses based on your files!"
    )


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()