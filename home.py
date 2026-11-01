import streamlit as st

from cmn.bedrock_models import FOUNDATION_MODELS
from cmn.view.menu import MENU_SECTIONS

REPO_URL = "https://github.com/christoferson/streamlit-demo"

st.set_page_config(page_title="Amazon Bedrock Demos", page_icon=":material/home:", layout="wide")
st.logo(icon_image="images/logo.png", image="images/logo_text.png")

all_pages = [page for pages in MENU_SECTIONS.values() for page in pages]
listed_models = FOUNDATION_MODELS.listed_models()

st.title("Generative AI on Amazon Bedrock")
st.markdown(
    "Hands-on demos of what you can build with **Amazon Bedrock**: chat with many models, assistants that "
    "use tools, question-answering over your own documents, and image generation and editing. Every demo "
    "calls AWS live, built in Python with **Streamlit** and **boto3**."
)

demos_column, models_column, providers_column = st.columns(3)
demos_column.metric("Demos", len(all_pages), border=True)
models_column.metric("Models to choose from", len(listed_models), border=True)
providers_column.metric("Model providers", len({model.provider for model in listed_models}), border=True)

st.header("Start here")
st.caption("A good first tour. Each demo explains itself on its own page.")
featured = [page for page in all_pages if page.featured]
for row_start in range(0, len(featured), 3):
    columns = st.columns(3)
    for column, page in zip(columns, featured[row_start:row_start + 3]):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {page.icon} {page.title}")
            st.write(page.summary)
            st.page_link(page.path, label="Open demo", icon=":material/arrow_forward:")

st.header("All demos")
for section, pages in MENU_SECTIONS.items():
    with st.expander(f"**{section}** ({len(pages)})"):
        for page in pages:
            st.page_link(page.path, label=page.title, icon=page.icon)
            st.caption(page.summary)

st.header("How it's built")
left, right = st.columns(2)
with left:
    st.markdown(
        "- **One conversation core, many pages.** Streaming, the tool-calling loop and chat history live in "
        "a UI-free package (`cmn/bedrock/converse`), so the logic can be tested without a browser.\n"
        "- **A model catalog.** Each model's ID, display name, accepted settings and features (images, "
        "documents, tools) is declared once in `cmn/bedrock_models.py`. Pages only offer what a model "
        "supports, so you won't hit \"parameter not supported\" errors."
    )
with right:
    st.markdown(
        "- **Plug-in tools and renderers.** A new tool is one class; it is automatically offered to the "
        "model and described in the system prompt, and a matching renderer turns its result into a chart, "
        "table or download.\n"
        "- **Tested.** Unit tests cover the catalog, request building and stream parsing, and a script "
        "checks every catalog model still answers on Bedrock."
    )

st.divider()
st.caption(
    f"Source code: [{REPO_URL.removeprefix('https://')}]({REPO_URL}). "
    "Some demos need resources in the AWS account they run in (a knowledge base, an S3 Vectors bucket, "
    "an OpenSearch index or an AgentCore harness)."
)
