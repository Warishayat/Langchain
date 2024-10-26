import os
from langchain.llms import HuggingFaceHub
import streamlit as st

# Set the OpenAI API key (ensure not to hardcode sensitive keys in production)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# Streamlit framework
st.title("Langchain Demo with HuggingFace Hub Key:")

# Use st.text_input for user input
user_input = st.text_input("Search the topic you want:")

# Initialize the OpenAI LLM model
llm = HuggingFaceHub(repo_id="microsoft/DialoGPT-medium")

# Generate a response if input_text is provided
if user_input:
    prompt = f"Answer the following question:{user_input}"
    response = llm(prompt)
    st.write(response)
