import streamlit as st

from cmn.bedrock_models import FOUNDATION_MODELS
from cmn.view.menu import MENU_SECTIONS

REPO_URL = "https://github.com/christoferson/streamlit-demo"

st.set_page_config(page_title="Amazon Bedrock Demos", page_icon="images/logo.png", layout="wide")
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

def service_badges(page) -> str:
    return " ".join(f":blue-badge[{service}]" for service in page.services)


def render_demo_details(page) -> None:
    st.markdown(f"#### {page.icon} {page.title}")
    st.markdown(page.summary)
    if page.services:
        st.markdown(service_badges(page))
    if page.details:
        st.markdown("\n".join(f"- {detail}" for detail in page.details))
    if page.try_this:
        st.markdown(f":green[**:material/play_circle: Try this:**] {page.try_this}")
    if page.needs:
        st.markdown(f":orange[**:material/build_circle: Needs setup:**] {page.needs}")
    if page.differs:
        st.markdown(f":violet[**:material/compare_arrows: How it differs:**] {page.differs}")
    st.page_link(page.path, label="Open demo", icon=":material/arrow_forward:")


st.header("Start here")
st.caption("A good first tour. Every demo calls Bedrock live, so replies take a few seconds.")
featured = [page for page in all_pages if page.featured]
for row_start in range(0, len(featured), 3):
    columns = st.columns(3)
    for column, page in zip(columns, featured[row_start:row_start + 3]):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {page.icon} {page.title}")
            st.markdown(page.summary)
            st.markdown(service_badges(page))
            if page.try_this:
                st.markdown(f":green[**Try this:**] {page.try_this}")
            st.page_link(page.path, label="Open demo", icon=":material/arrow_forward:")

st.header("All demos")
st.markdown(
    "Each demo below lists what you can do in it, and how to read the colors: "
    ":blue-badge[AWS service or API] "
    ":green[**Try this**] a one-minute test, "
    ":orange[**Needs setup**] resources it expects in the AWS account, "
    ":violet[**How it differs**] from similar-looking demos."
)
for tab, (section, pages) in zip(st.tabs([f"{section} ({len(pages)})" for section, pages in MENU_SECTIONS.items()]),
                                 MENU_SECTIONS.items()):
    with tab:
        for row_start in range(0, len(pages), 2):
            columns = st.columns(2)
            for column, page in zip(columns, pages[row_start:row_start + 2]):
                with column.container(border=True, height="stretch"):
                    render_demo_details(page)

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
