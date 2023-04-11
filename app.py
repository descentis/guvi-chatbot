# This is a sample Python script.

import streamlit as st
from streamlit_chat import message
import os
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DataFrameLoader
from langchain.document_loaders.csv_loader import CSVLoader


os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

global input_text
def clear_input_text():
    global input_text
    input_text = ""


# We will get the user's input by calling the get_text function
def get_text():
    global input_text
    input_text = st.text_input("Ask your Question", key="input", on_change=clear_input_text)
    return input_text


@st.cache_data
def get_file(file_data):
    if '.csv' in file_data.name:
        df = pd.read_csv(file_data)
    else:
        df = pd.read_excel(file_data)
    return df


# create a file uploader
uploaded_file = st.file_uploader("Choose a file")
user_input = get_text()
questions = []
if uploaded_file:
    dataframe = get_file(uploaded_file)
    with st.sidebar:
        #if st.button("Generate"):
        agent = create_pandas_dataframe_agent(OpenAI(model_name="gpt-3.5-turbo", temperature=0), dataframe, verbose=True)
        with st.spinner('Loading some  Data Analytics questions...'):
            ques_input = "Suggest around ten complex data analysis questions on this dataframe in a markdown list"
            q_output = agent.run(ques_input)
            questions.append(q_output)
            st.write("Some sample questions on this dataset:\n"+ questions[0])
if st.button("Post"):
    with st.spinner("Waiting for the response..."):
        prefix = '\nYou are working with a pandas dataframe in Python. The name of the dataframe is `df`.\nYou should use the tools below to answer the question posed of you:'
        suffix = "\nThis is the result of `print(df.head())`:\n{df}\n\nBegin!\nQuestion: {input}\n{agent_scratchpad}"
        #agent = create_pandas_dataframe_agent(OpenAI(temperature=0), dataframe, verbose=True, prefix=prefix, suffix=suffix)
        if user_input:
            output = agent.run(user_input+'. Moreover, provide a detailed analysis of your inference below Analysis header')
            # store the output
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
