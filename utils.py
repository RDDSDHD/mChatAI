import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.vectorstores import FAISS


@st.cache_resource
def load_chain():
    """
The `load_chain()` function initializes and configures a conversational retrieval chain for
answering user questions.
:return: The `load_chain()` function returns a ConversationalRetrievalChain object.
"""
    api_key = "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"

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Load OpenAI chat model
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.1, model_name="gpt-4")

    # Load our local FAISS index as a retriever
    vector_store = FAISS.load_local("faiss_index", embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # Create memory 'chat_history'
    memory = ConversationBufferWindowMemory(k=10, memory_key="chat_history")

    # Create system prompt
    template = """
    You are an AI assistant for answering only in Polish questions about the mBank internal knowledge.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer only in Polish.
    If you don't know the answer, just say 'Sorry, I don't know ... 😔 (only in Polish). 
    Don't try to make up an answer.
    If the question is not about the mBank internal knowledge, politely inform them that you are tuned to only answer questions about the mBank.

    {context}
    Question: {question}
    Helpful Answer:"""

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                  retriever=retriever,
                                                  memory=memory,
                                                  get_chat_history=lambda h: h,
                                                  verbose=True)

    # Add systemp prompt to chain
    # Can only add it at the end for ConversationalRetrievalChain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain
