import re
from pathlib import Path

import streamlit as st

from cmn.view.menu import MENU_SECTIONS


def legacy_url_path(path: str) -> str:
    # Same URL the old pages/ auto-navigation used ("3_7_5_converse_tool_v3" -> "converse_tool_v3"), so old links keep working
    return re.sub(r"^[\d_]+", "", Path(path).stem)


home = st.Page("home.py", title="Home", icon=":material/home:", default=True)

navigation = {"": [home]}
for section, pages in MENU_SECTIONS.items():
    navigation[section] = [
        st.Page(page.path, title=page.title, icon=page.icon, url_path=legacy_url_path(page.path)) for page in pages
    ]

st.navigation(navigation).run()
