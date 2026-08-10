import nest_asyncio
nest_asyncio.apply()
import asyncio

import streamlit as st

from llm import rag_pipeline


st.set_page_config(
    page_title="bot",
    layout="wide"
)
st.markdown(
    "<h1 style='text-align:center;'>bot</h1>",
    unsafe_allow_html=True
)


if "chat" not in st.session_state:
    st.session_state.chat = []

user_msg = st.chat_input("Hi! I am your friend 😊")

if user_msg:
    st.session_state.chat.append(('user', user_msg))
    bot_msg = asyncio.run(rag_pipeline(user_query=user_msg))
    st.session_state.chat.append(('assistant',  bot_msg))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)
