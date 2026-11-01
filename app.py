import streamlit as st

from cmn.view.menu import MENU_SECTIONS

home = st.Page("home.py", title="Home", icon=":material/home:", default=True)

navigation = {"": [home]}
for section, pages in MENU_SECTIONS.items():
    navigation[section] = [st.Page(page.path, title=page.title, icon=page.icon) for page in pages]

st.navigation(navigation).run()
