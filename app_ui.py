# app_ui.py
import streamlit as st
import requests

st.title("PDF Chatbot")
st.write("Ask questions based on your PDFs!")

# Input for user query
query = st.text_input("Enter your query:")

if st.button("Submit Query"):
    if query:
        payload = {"query": query}
        try:
            response = requests.post("http://localhost:8000/query", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.write("### Response:")
                st.write(data.get("response", "No response returned."))
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")
