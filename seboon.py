import streamlit as st
import pandas as pd
import pickle

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

st.title("Spam/Ham Classification")

st.markdown("### About this project")
st.markdown("#### This project is used to classify emails that are Spam or Ham")

review = st.text_area("Entre your message to see if it is spam or ham")

if st.button("predict"):

    review_vec = vectorizer.transform([review])

    prediction = model.predict(review_vec)

    if prediction[0] == "ham":
        st.write("This is a ham mail")
    elif prediction[0] == "spam":
        st.write("This is a spam mail")
































