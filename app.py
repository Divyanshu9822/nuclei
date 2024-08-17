import os
from groq import Groq
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

st.title("Nuclei - Chat with AI")

st.sidebar.header("About")
st.sidebar.write(
    "🌟 **Nuclei** is a context-aware AI chatbot application powered by Groq's API. 🤖✨\n\n"
    "It remembers the context of your conversation, making interactions more coherent and relevant. 🧠💬\n\n"
    "Enter your API key and select a model to start chatting. 🔑🛠️"
)


st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
selected_model = st.sidebar.selectbox(
    "Select a model:",
    [
        "gemma-7b-it",
        "gemma2-9b-it",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768"
    ],
    index=2
)

st.sidebar.warning(
    "Note: Each model has its own limits on requests and tokens. If you exceed these limits, you may encounter runtime errors."
)

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

    if "groq_model" not in st.session_state:
        st.session_state["groq_model"] = selected_model

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    for message in st.session_state.chat_history:
        memory.save_context(
            {"input": message["human"]},
            {"output": message["AI"]}
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a helpful AI assistant named Nuclei."),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}")
            ]
        )

        groq_chat = ChatGroq(
            groq_api_key=api_key,
            model_name=selected_model
        )

        conversation = LLMChain(
            llm=groq_chat,
            prompt=chat_prompt,
            verbose=True,
            memory=memory
        )

        response = conversation.predict(human_input=prompt)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append({"human": prompt, "AI": response})
else:
    st.error("Please enter your API key in the sidebar to start chatting.")
