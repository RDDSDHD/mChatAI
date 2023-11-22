import time

import openai
import streamlit as st
from utils import load_chain

openai.api_key = st.secrets["OPENAI_API_KEY"]

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
