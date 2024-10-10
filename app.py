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
session_identity = process_aws_oidc_data()
st.write(f"Welcome: {session_identity}")

# Create tabs
#tab1, tab2 = st.tabs(["Main", "Settings"])

# with tab1:
#    st.title("Main Tab")
#    st.write("Hello world")
#    #st.page_link("pages/5_2_3_kb_demo.py", label="Page 1", icon="1️⃣")

# with tab2:
st.title("Settings")
# Theme settings
st.header("Theme Settings")

# Create a radio button for theme selection
theme_choice = st.radio(
    "Choose theme:",
    options=['Light', 'Dark'],
    index=0 if st.session_state.theme == 'light' else 1,
    horizontal=True
)

# Update the theme based on the radio button selection
if theme_choice == 'Light':
    st.session_state.theme = 'light'
else:
    st.session_state.theme = 'dark'

# Apply the theme


#st.write(f"Current theme: {st.session_state.theme}")

# Font size settings
st.header("Font Size")
if 'font_size' not in st.session_state:
   st.session_state.font_size = 'medium'
font_size = st.select_slider('Choose font size', options=['small', 'medium', 'large'])
if font_size != st.session_state.font_size:
   st.session_state.font_size = font_size
   #st.rerun()
st.markdown(f"<p style='font-size: {font_size};'>This text will change size based on your selection.</p>", unsafe_allow_html=True)

# Language settings
st.header("Language")
languages = ['English', 'Spanish', 'French', 'German', 'Chinese']
selected_language = st.selectbox('Select your preferred language', languages)
st.write(f"Selected language: {selected_language}")

# Notification settings
st.header("Notification Settings")
email_notifications = st.toggle('Enable email notifications')
push_notifications = st.toggle('Enable push notifications')
st.write(f"Email notifications: {'Enabled' if email_notifications else 'Disabled'}")
st.write(f"Push notifications: {'Enabled' if push_notifications else 'Disabled'}")

# Data privacy settings
st.header("Data Privacy")
data_collection = st.checkbox('Allow data collection for improving user experience')
st.write(f"Data collection: {'Allowed' if data_collection else 'Not allowed'}")

# Accessibility features
st.header("Accessibility")
high_contrast = st.toggle('High contrast mode')
screen_reader = st.toggle('Screen reader support')
st.write(f"High contrast mode: {'Enabled' if high_contrast else 'Disabled'}")
st.write(f"Screen reader support: {'Enabled' if screen_reader else 'Disabled'}")

# Custom color picker
st.header("Custom Accent Color")
accent_color = st.color_picker('Pick a custom accent color', '#00f900')
st.markdown(f"<p style='color:{accent_color};'>This text will change color based on your selection.</p>", unsafe_allow_html=True)

# Easter egg
st.header("Secret Setting")
if st.button("Don't click me"):
   st.balloons()
   st.write("You found the secret! Here's a random joke:")
   jokes = [
      "Why don't scientists trust atoms? Because they make up everything!",
      "Why did the scarecrow win an award? He was outstanding in his field!",
      "Why don't eggs tell jokes? They'd crack each other up!",
      "What do you call a fake noodle? An impasta!"
   ]
   st.write(random.choice(jokes))

# Save settings
if st.button('Save Settings'):
   st._config.set_option('theme.base', st.session_state.theme)
   st.success('Settings saved successfully!')
   st.rerun()