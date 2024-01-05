import streamlit as st
import os
import requests
from PIL import Image
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool
from tempfile import NamedTemporaryFile

########################
### Initialize agent ###
########################
tools = [ImageCaptionTool(), ObjectDetectionTool()]
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
llm = ChatOpenAI(
    # openai_api_key=os.environ.get("OPENAI_API_KEY"), # for local usage
    openai_api_key=st.secrets['auth_key'],
    temperature=0,
    model_name='gpt-3.5-turbo'
)
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stoppy_method='generate'
)
# Set title
st.title("Ask a question to an image")
# set header
st.header("Please upload an image")

# upload file
# file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

# input image url
image_url = st.text_input('Please input the image URL')


if image_url and image_url != "":
    image_object = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # display image
    
    #st.image(file, use_column_width=True)
    st.image(image_object, use_column_width=True)

    # text input
    user_question = st.text_input('Ask a question your image:')

    ##############################
    ### Compute agent response ###
    ##############################

    # with NamedTemporaryFile(dir='.', mode='w+b') as f:
    #     f.write(file.getbuffer())
    #     image_path = f.name

    #     # write agent response
    #     if user_question and user_question != "":
    #         with st.spinner(text='In progress...'):
    #             response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
    #             st.write(response)

    #write agent response
    if user_question and user_question != "":
        with st.spinner(text='In progress...'):
            response = agent.run('{}, this is the image path: {}'.format(user_question, image_url))
            st.write(response)