import time

import openai
import hmac
import streamlit as st
from utils import load_chain

openai.api_key = st.secrets["OPENAI_API_KEY"]


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Niepoprawne hasło")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Custom image for the app icon and the assistant's avatar
company_logo = 'assets/mbank.png'

# Configure Streamlit page
st.set_page_config(
    page_title="mBank BOT asystent 👨‍💻",
)

# Initialize LLM chain
chain = load_chain()

with st.sidebar:
    st.image(company_logo)
    st.write("***Witajcie w aplikacji mChatAI***")
    st.write("**mBank. Technologia do usług 👨‍💻**")
    for i in range(20):
        st.write("")
    st.write("*autor: Sebastian Kowalczykiewicz*")
    st.write("*kontakt: s.kowalczykiewicz@outlook.com*")

# Initialize chat history
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant",
                                     "content": "Witaj w mChatAI - W czym mogę Ci pomóc?"}]

# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat logic
if query := st.chat_input("Zadaj pytanie"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = chain({"question": query})
        response = result['answer']
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(response)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
