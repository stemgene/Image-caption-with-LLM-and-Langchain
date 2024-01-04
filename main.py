import streamlit as st

########################
### Initialize agent ###
########################

# Set title
st.title("Ask a question to an image")

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

if file:

    # display image
    st.image(file, use_column_width=True)

    # text input
    user_question = st.text_input('Ask a question your image:')

    ##############################
    ### Compute agent response ###
    ##############################

    # write agent response
    if user_question and user_question != "":
        st.write("dummy response")