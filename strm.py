import streamlit as st
from src.streamlit_controller import StreamlitController

controller = StreamlitController()
controller.load_model()

def process_email():
    email_content = user_text
    controller.SetEmailContent(emailContent=email_content)
    controller.transform_email_content()
    spam = controller.predict_email()

    if(spam == 0):
        st.write("That text is ham")
    elif(spam ==1):
        st.write("That text is spam")

st.title('Spam Ham Prediction')

user_text = st.text_area("Enter your email text:")

if st.button("Process Email"):
    process_email()

