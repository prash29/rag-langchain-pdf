import streamlit as st
from utils import PopulateDatabase, query_rag


# Set the title of the app
st.title("Local RAG - Chat with your PDF documents!")

# Create a button
load_button = st.button("Load Documents")

# Create another button to reset the database
reset_checkbox = st.checkbox("Reset Database")

if load_button:
    db_populator = PopulateDatabase(reset=reset_checkbox)
    if db_populator.reset:
        st.write("Clearing and re-populating the database! Please wait...")
    db_populator.execute()
    st.write("Documents loaded successfully!")
    
st.divider()
query = st.text_input("Enter your question here : ")
query_button = st.button("Submit")

if query_button:
    response, sources = query_rag(query)
    st.write(f"{response}")

    # Create an expander
    with st.expander("Click to check sources"):
        st.write(f"Sources: {sources}")

