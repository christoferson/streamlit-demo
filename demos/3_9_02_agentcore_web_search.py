import json
import logging
import re
import time
from datetime import date, datetime, time as dtime, timezone
from urllib.parse import urlparse

import boto3
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

import cmn_settings
from cmn.bedrock.converse.request_builder import InferenceSettings, build_converse_request
from cmn.bedrock.converse.stream_processor import process_stream
from cmn.bedrock_models import FOUNDATION_MODELS, Feature

AWS_REGION = cmn_settings.AWS_REGION
PAGE_TITLE = "Web search API"
WEB_SEARCH_ID = "web-search.v1"
MAX_QUERY_CHARS = 200
MAX_DOMAINS = 100
PRICE_PER_QUERY = 7 / 1000
DEFAULT_ANSWER_MODEL = "global.anthropic.claude-haiku-5-5"

SAMPLES = {
    "AWS news": ("Amazon Bedrock AgentCore new features", "", "", 30),
    "One site only": ("AgentCore web search", "dev.classmethod.jp", "", None),
    "Skip a site": ("Streamlit chat input tutorial", "", "medium.com", None),
    "This week": ("generative AI announcements", "", "", 7),
}

ANSWER_SYSTEM_PROMPT = (
    "You answer the user's question using only the numbered web search results provided. "
    "Cite sources inline as [n] right after the sentence they support. If the results do not "
    "contain the answer, say so plainly instead of guessing. Keep it concise. "
    "Answer in the language of the question."
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title=PAGE_TITLE, page_icon=":material/travel_explore:", layout="wide")
st.logo(icon_image="images/logo.png", image="images/logo_text.png")


@st.cache_resource
def get_agentcore():
    return boto3.client("bedrock-agentcore", region_name=AWS_REGION)


@st.cache_resource
def get_bedrock_runtime(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def parse_domains(text: str) -> list[str]:
    """'https://a.com/, b.com' -> ['a.com', 'b.com']"""
    domains = []
    for item in re.split(r"[\s,]+", text.strip()):
        if not item:
            continue
        host = urlparse(item if "//" in item else f"//{item}").hostname or item
        if host not in domains:
            domains.append(host)
    return domains


def build_request(query: str, max_results: int | None, include: list[str], exclude: list[str],
                  after: date | None, before: date | None) -> dict:
    request = {"webSearchId": WEB_SEARCH_ID, "searchQuery": query}
    if max_results:
        request["maxResults"] = max_results
    filters = {}
    if include or exclude:
        filters["domainFilter"] = {key: value for key, value in (("include", include), ("exclude", exclude)) if value}
    dates = {}
    if after:
        dates["publishedAfter"] = datetime.combine(after, dtime.min, tzinfo=timezone.utc)
    if before:
        dates["publishedBefore"] = datetime.combine(before, dtime.max, tzinfo=timezone.utc)
    if dates:
        filters["publishedDateFilter"] = dates
    if filters:
        request["filters"] = filters
    return request


def as_python(request: dict) -> str:
    """The request as a copy-pasteable boto3 call."""
    def literal(value):
        if isinstance(value, datetime):
            return f"datetime({value.year}, {value.month}, {value.day}, tzinfo=timezone.utc)"
        if isinstance(value, dict):
            return "{" + ", ".join(f"{k!r}: {literal(v)}" for k, v in value.items()) + "}"
        return repr(value)
    args = ",\n    ".join(f"{key}={literal(value)}" for key, value in request.items())
    return (f'client = boto3.client("bedrock-agentcore", region_name="{AWS_REGION}")\n'
            f"response = client.invoke_web_search(\n    {args},\n)\n"
            'for result in response["results"]:\n'
            '    print(result.get("title"), result.get("url"), result.get("publishedAt"))')


def search(request: dict) -> dict:
    started = time.perf_counter()
    response = get_agentcore().invoke_web_search(**request)
    return {
        "request": request,
        "results": response.get("results", []),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "answer": None,
    }


def sources_prompt(question: str, results: list[dict], chars_per_result: int) -> str:
    blocks = []
    for number, result in enumerate(results, 1):
        lines = [f"[{number}] {result.get('title') or '(no title)'}", f"URL: {result.get('url', '')}"]
        if result.get("publishedAt"):
            lines.append(f"Published: {result['publishedAt']:%Y-%m-%d}")
        lines.append(result["text"][:chars_per_result])
        blocks.append("\n".join(lines))
    return "Search results:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def domain_of(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def apply_sample():
    label = st.session_state.get("ws_sample")
    if not label:
        return
    query, include, exclude, days = SAMPLES[label]
    st.session_state["ws_query"] = query
    st.session_state["ws_include"] = include
    st.session_state["ws_exclude"] = exclude
    st.session_state["ws_use_after"] = days is not None
    if days is not None:
        st.session_state["ws_after"] = date.fromordinal(date.today().toordinal() - days)


st.session_state.setdefault("ws_query", SAMPLES["AWS news"][0])
st.session_state.setdefault("ws_after", date.fromordinal(date.today().toordinal() - 30))
st.session_state.setdefault("ws_search", None)

answer_models = [m for m in FOUNDATION_MODELS.listed_models()
                 if m.is_feature_supported(Feature.CONVERSE_STREAM) and m.is_feature_supported(Feature.SYSTEM_PROMPTS)]
answer_model_ids = [m.model_id for m in answer_models]

# ── Sidebar: filters and answer settings ───────────────────────────────────────
with st.sidebar:
    st.markdown("**Search**")
    limit_results = st.toggle("Limit the number of results", value=True,
                              help="Off: let the API decide (it returned 20 in testing).")
    max_results = st.slider("Max results", 1, 25, 10) if limit_results else None
    include_text = st.text_area("Only these domains", key="ws_include", height=68,
                                placeholder="aws.amazon.com, docs.aws.amazon.com")
    exclude_text = st.text_area("Never these domains", key="ws_exclude", height=68, placeholder="example.com")
    use_after = st.checkbox("Published after", key="ws_use_after")
    published_after = st.date_input("After", key="ws_after", label_visibility="collapsed", disabled=not use_after)
    use_before = st.checkbox("Published before", key="ws_use_before")
    published_before = st.date_input("Before", value=date.today(), label_visibility="collapsed",
                                     disabled=not use_before)

    st.divider()
    st.markdown("**Answer**")
    answer_on = st.toggle("Answer from the results with a model", value=True,
                          help="Sends the results to a Bedrock model through Converse and asks for a cited answer.")
    answer_model_id = st.selectbox(
        "Model", answer_model_ids, disabled=not answer_on,
        index=answer_model_ids.index(DEFAULT_ANSWER_MODEL) if DEFAULT_ANSWER_MODEL in answer_model_ids else 0,
        format_func=lambda mid: FOUNDATION_MODELS.find(mid).display_name,
    )
    chars_per_result = st.slider("Characters per result sent to the model", 200, 3000, 1000, 100,
                                 disabled=not answer_on,
                                 help="Excerpts can be long (about 600 characters each); trimming keeps the prompt small.")
    st.caption(f"Region {AWS_REGION} · ${PRICE_PER_QUERY:.3f} per search")

include = parse_domains(include_text)
exclude = parse_domains(exclude_text)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown(f"#### :material/travel_explore: :blue[{PAGE_TITLE}]")
st.markdown(
    "Search the web from your own code with **Amazon Bedrock AgentCore**'s `InvokeWebSearch` API: one IAM-signed "
    "call, no gateway or other resources to create. Filter by domain and publish date, then optionally have a "
    "model write an answer that cites the results."
)

with st.expander(":material/new_releases: What's new (November 2026)"):
    st.markdown(
        "- :blue-badge[Direct API] Web search used to be an MCP tool behind an AgentCore Gateway. Now it is "
        "`bedrock-agentcore` → `invoke_web_search` (boto3 1.43.127 or later).\n"
        "- :blue-badge[IAM only] Needs `bedrock-agentcore:InvokeWebSearch`. Nothing to provision.\n"
        "- :blue-badge[Filters] Up to 100 included or excluded domains, and a published-after / before range.\n"
        "- :blue-badge[Up to 25 results] Ranked by relevance. Each has an excerpt (`text`); `title`, `url` and "
        "`publishedAt` only when known.\n"
        "- :blue-badge[$7 per 1,000 searches] The same price as through the Gateway.\n\n"
        ":violet[**API or Gateway?**] Use the API when your code reads the results (this page). Use the Gateway "
        "connector when an agent should call search as an MCP tool."
    )

st.pills("Try a sample", list(SAMPLES), key="ws_sample", on_change=apply_sample,
         help="Fills in the query and the filters for it.")
with st.form("ws_form", border=False):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        query = st.text_input("Search the web", key="ws_query", max_chars=MAX_QUERY_CHARS,
                              placeholder="What do you want to find?")
        submitted = st.form_submit_button("Search", type="primary", icon=":material/search:")

request = build_request(query.strip(), max_results, include[:MAX_DOMAINS], exclude[:MAX_DOMAINS],
                        published_after if use_after else None, published_before if use_before else None)
for label, domains in (("included", include), ("excluded", exclude)):
    if len(domains) > MAX_DOMAINS:
        st.warning(f"Only the first {MAX_DOMAINS} {label} domains are sent.")

if submitted and query.strip():
    try:
        with st.spinner("Searching the web..."):
            st.session_state["ws_search"] = search(request)
    except ClientError as err:
        error = err.response["Error"]
        st.session_state["ws_search"] = None
        if error.get("Code") == "AccessDeniedException":
            st.error(f"Access denied. The caller needs `bedrock-agentcore:InvokeWebSearch` in {AWS_REGION}.\n\n"
                     f"{error.get('Message', '')}")
        else:
            st.error(f"{error.get('Code')}: {error.get('Message')}")
    except BotoCoreError as err:
        st.session_state["ws_search"] = None
        st.error(str(err))

    current = st.session_state["ws_search"]
    if current and answer_on and current["results"]:
        model = FOUNDATION_MODELS.find(answer_model_id)
        messages = [{"role": "user", "content": [
            {"text": sources_prompt(query.strip(), current["results"], chars_per_result)}]}]
        converse_request = build_converse_request(model, messages, ANSWER_SYSTEM_PROMPT,
                                                  InferenceSettings(max_tokens=2048, temperature=0.2))
        with st.container(border=True):
            st.markdown(f":material/smart_toy: **Answer** · :gray[{model.display_name}]")
            placeholder = st.empty()
            streamed = []

            def show_delta(chunk: str):
                streamed.append(chunk)
                placeholder.markdown("".join(streamed) + " ▌")

            try:
                stream = get_bedrock_runtime(model.region or AWS_REGION).converse_stream(**converse_request)
                result = process_stream(stream["stream"], on_text_delta=show_delta)
                current["answer"] = {
                    "model": model.display_name,
                    "text": result.text,
                    "errors": result.errors,
                    "input_tokens": result.metrics.input_tokens,
                    "output_tokens": result.metrics.output_tokens,
                    "latency_ms": result.metrics.latency_ms,
                }
            except (ClientError, BotoCoreError) as err:
                current["answer"] = {"model": model.display_name, "text": "", "errors": [str(err)]}
        st.rerun()

current = st.session_state["ws_search"]
if not current:
    with st.expander(":material/code: The call this page makes"):
        st.code(as_python(request), language="python")
    st.stop()

results = current["results"]
excerpt_chars = sum(len(r.get("text", "")) for r in results)
with st.container(horizontal=True):
    st.metric("Results", len(results))
    st.metric("Search latency", f"{current['latency_ms']:,} ms")
    st.metric("Excerpt text", f"{excerpt_chars:,} chars")
    st.metric("With dates", f"{sum(1 for r in results if r.get('publishedAt'))} / {len(results)}")
    st.metric("Search cost", f"${PRICE_PER_QUERY:.3f}")

if not results:
    st.info("No results. Try fewer filters or a broader query.")

answer = current.get("answer")
if answer:
    with st.container(border=True):
        st.markdown(f":material/smart_toy: **Answer** · :gray[{answer['model']}]")
        for error in answer["errors"]:
            st.error(error)
        if answer["text"]:
            linked = re.sub(
                r"\[(\d+)\]",
                lambda m: f"[[{m.group(1)}]]({results[int(m.group(1)) - 1].get('url', '')})"
                if 0 < int(m.group(1)) <= len(results) and results[int(m.group(1)) - 1].get("url") else m.group(0),
                answer["text"],
            )
            st.markdown(linked)
            if answer.get("output_tokens"):
                st.caption(f"{answer['input_tokens']:,} input · {answer['output_tokens']:,} output tokens · "
                           f"{answer['latency_ms']:,} ms")

tab_results, tab_request, tab_raw = st.tabs([
    f":material/list: Results ({len(results)})", ":material/code: Request", ":material/data_object: Raw response"])

with tab_results:
    domains = sorted({domain_of(r["url"]) for r in results if r.get("url")})
    shown_domains = st.pills("Filter by site", domains, selection_mode="multi", key="ws_domain_pills") \
        if len(domains) > 1 else []
    for number, result in enumerate(results, 1):
        url = result.get("url", "")
        if shown_domains and domain_of(url) not in shown_domains:
            continue
        with st.container(border=True):
            title = (result.get("title") or url or "(no title)").replace("[", r"\[").replace("]", r"\]")
            st.markdown(f"**{number}.** [{title}]({url})" if url else f"**{number}.** {title}")
            badges = [f":blue-badge[{domain_of(url)}]"] if url else []
            if result.get("publishedAt"):
                badges.append(f":green-badge[:material/calendar_today: {result['publishedAt']:%Y-%m-%d}]")
            else:
                badges.append(":gray-badge[no date]")
            badges.append(f":gray-badge[{len(result.get('text', '')):,} chars]")
            st.markdown(" ".join(badges))
            text = result.get("text", "")
            st.markdown(text[:300] + ("…" if len(text) > 300 else ""))
            if len(text) > 300:
                with st.expander("Full excerpt"):
                    st.text(text)

with tab_request:
    st.code(as_python(current["request"]), language="python")

with tab_raw:
    st.code(json.dumps({"results": results}, indent=2, ensure_ascii=False, default=str), language="json")
