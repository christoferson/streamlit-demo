import streamlit as st
import cmn_auth
from streamlit.web.server.websocket_headers import _get_websocket_headers

######

def process_aws_oidc_token():
   
   identity = "unknown"
   headers = _get_websocket_headers()
   if headers:
      if "x-amzn-oidc-identity" in headers:
         identity = headers["x-amzn-oidc-identity"]
         #accesstoken = headers["x-amzn-oidc-accesstoken"]
         #data = headers["x-amzn-oidc-data"]
      if "X-Amzn-Oidc-Identity" in headers:
         identity = headers["X-Amzn-Oidc-Identity"]
   
   return identity

session_identity = process_aws_oidc_token()
st.write('Welcome: ', session_identity)


####


###### AUTH START #####

if not cmn_auth.check_password():
   st.stop()

######  AUTH END #####

st.write("Hello world")
