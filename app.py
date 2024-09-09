import streamlit as st
import cmn_auth
from streamlit.web.server.websocket_headers import _get_websocket_headers

import jwt
from typing import Any, Dict
import json

###### Get Name from OIDC

#def _decode_access_token(access_token: str) -> Dict[str, Any]:
#    decoded = jwt.decode(access_token, options={"verify_signature": False})
#    return decoded

def process_aws_oidc_data():

   access_token = "oidc_data"
   headers = st.context.headers #_get_websocket_headers()
   if headers:
      #if "X-Amzn-Oidc-Identity" in headers:
      #   identity = headers["X-Amzn-Oidc-Identity"]
      if "X-Amzn-Oidc-Accesstoken" in headers:
         access_token_jwt =  headers["X-Amzn-Oidc-Accesstoken"]
         #access_token_claim = _decode_access_token(access_token=access_token_jwt)
         access_token_claims = jwt.decode(access_token_jwt, options={"verify_signature": False})
         #access_token = json.dumps(access_token_claims)
         if "name" in access_token_claims:
            access_token = access_token_claims["name"]
         elif "upn" in access_token_claims:
            access_token = access_token_claims["upn"]

   return access_token

session_identity = process_aws_oidc_data()
st.write(f"Welcome: {session_identity}")


####


###### AUTH START #####

#if not cmn_auth.check_password():
#   st.stop()

######  AUTH END #####

st.write("Hello world")

#st.page_link("pages/5_2_3_kb_demo.py", label="Page 1", icon="1️⃣")
