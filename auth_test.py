import streamlit as st 

st.title("Authentication")

if st.button("Authenticate"):
    st.login("google")
    # st.session_state.authenticated = True
    # st.success("You are now logged in!")