import streamlit as st
from src.streamlit_controller import StreamlitController

controller = StreamlitController()
controller.load_model()


st.title('Spam Ham Prediction')

user_input = st.text_input("Enter your email text:")


if user_input:
    email_content = user_input
    controller.SetEmailContent(emailContent=email_content)
    controller.transform_email_content()
    spam = controller.predict_email()

    if(spam == 0):
        st.write("That text is ham")
    elif(spam ==1):
        st.write("That text is spam")

