import streamlit as st
import cmn_auth
from streamlit.web.server.websocket_headers import _get_websocket_headers

import jwt
from typing import Any, Dict
import json

######

def _decode_access_token(access_token: str) -> Dict[str, Any]:
    decoded = jwt.decode(access_token, options={"verify_signature": False})
    return decoded

def process_aws_oidc_token():
   
   identity = "unknown"
   access_token = "oidc_data"
   headers = _get_websocket_headers()
   if headers:
      if "x-amzn-oidc-identity" in headers:
         identity = headers["x-amzn-oidc-identity"]
         #accesstoken = headers["x-amzn-oidc-accesstoken"]
         #data = headers["x-amzn-oidc-data"]
      if "X-Amzn-Oidc-Identity" in headers:
         identity = headers["X-Amzn-Oidc-Identity"]

      if "X-Amzn-Oidc-Accesstoken" in headers:
         access_token =  headers["X-Amzn-Oidc-Accesstoken"]
         access_token_claim = _decode_access_token(access_token=access_token)
         access_token = json.dumps(access_token_claim)

   return access_token

session_identity = process_aws_oidc_token()
st.write(f"Welcome: {session_identity}")


####


###### AUTH START #####

if not cmn_auth.check_password():
   st.stop()

######  AUTH END #####

st.write("Hello world")
