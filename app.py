import streamlit as st
import cmn_auth

###### AUTH START #####

if not cmn_auth.check_password():
   st.stop()

######  AUTH END #####

st.write("Hello world")
