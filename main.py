import streamlit as st

########################
### Initialize agent ###
########################

# Set title
st.title("Ask a question to an image")

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", types=['jpg', 'png', 'jpeg'])

if file:

    # display image
    st.image(file, use_column)

    # text input

    ##############################
    ### Compute agent response ###
    ##############################

    # write agent response