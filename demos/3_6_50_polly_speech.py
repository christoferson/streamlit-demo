import logging
import time
from datetime import datetime

import boto3
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

import cmn_settings

AWS_REGION = cmn_settings.AWS_REGION
PAGE_TITLE = "Text to speech"
MAX_CHARACTERS = 3000  # SynthesizeSpeech limit on billed characters

ENGINES = {
    "generative": "Generative",
    "long-form": "Long-form",
    "neural": "Neural",
    "standard": "Standard",
}
NEW_VOICES = {"Kenneth", "Nate", "Russell", "Curtis", "Kiara", "Arjun", "Tina", "Carlos", "Leo", "Carolina",
              "Hyunduk", "Suvi"}
# Bilingual voices list extra languages by code only
EXTRA_LANGUAGE_NAMES = {"hi-IN": "Hindi"}
OUTPUT_FORMATS = {"mp3": "audio/mpeg", "ogg_vorbis": "audio/ogg"}

# label -> (text, language code, voice); language and voice are switched too when set
SAMPLES = {
    "Money & numbers": (
        "Your total comes to $1,249.99, including a 7.5% discount and $12.40 in tax. "
        "That's about 3.2 times the 389 dollars you spent in March.",
        None, None),
    "Dates & addresses": (
        "Your appointment moved from 03/14/2026 at 9:30 AM to Friday, April 3rd at 14:00. "
        "Please bring the form to 1600 Pennsylvania Ave NW, Washington, DC 20500.",
        None, None),
    "Codes & phone numbers": (
        "Your confirmation code is X7K-29Q. Flight UA 1543 boards at gate B12. "
        "If anything changes, call 1-800-555-0199.",
        None, None),
    "Surprise": (
        "Wait, you finished the whole project already? That's incredible! "
        "I honestly thought it would take at least another month.",
        None, None),
    "Uncertainty": (
        "Hmm, I'm not completely sure. It might be on the third floor... or maybe the fourth? "
        "Let me check and get back to you.",
        None, None),
    "Hindi": (
        "नमस्ते! आज मौसम बहुत सुहावना है। क्या आप शाम को मेरे साथ चाय पीने चलेंगे?",
        "hi-IN", "Kiara"),
    "Finnish": (
        "Hei ja tervetuloa Helsinkiin! Huomenna on aurinkoista, ja lämpötila nousee kahteenkymmeneen asteeseen.",
        "fi-FI", "Suvi"),
}

logger = logging.getLogger(__name__)

st.set_page_config(page_title=PAGE_TITLE, page_icon=":material/record_voice_over:", layout="wide")
st.logo(icon_image="images/logo.png", image="images/logo_text.png")


@st.cache_resource
def get_polly():
    return boto3.client("polly", region_name=AWS_REGION)


@st.cache_data(ttl=3600, show_spinner=False)
def list_voices(engine: str) -> list[dict]:
    voices, token = [], None
    while True:
        response = get_polly().describe_voices(Engine=engine, **({"NextToken": token} if token else {}))
        voices += response["Voices"]
        token = response.get("NextToken")
        if not token:
            return voices


def languages_of(voices: list[dict]) -> dict[str, str]:
    """Language code -> name, including the extra languages of bilingual voices."""
    names = {voice["LanguageCode"]: voice["LanguageName"] for voice in voices}
    for voice in voices:
        for code in voice.get("AdditionalLanguageCodes", []):
            names.setdefault(code, EXTRA_LANGUAGE_NAMES.get(code, code))
    return dict(sorted(names.items(), key=lambda item: item[1]))


def voices_for(voices: list[dict], language: str) -> list[dict]:
    matching = [v for v in voices if language in (v["LanguageCode"], *v.get("AdditionalLanguageCodes", []))]
    return sorted(matching, key=lambda v: (v["Id"] not in NEW_VOICES, v["Id"]))


def synthesize(text: str, voice: dict, engine: str, language: str, output_format: str, is_ssml: bool) -> dict:
    request = {
        "Engine": engine,
        "VoiceId": voice["Id"],
        "OutputFormat": output_format,
        "Text": text,
        "TextType": "ssml" if is_ssml else "text",
    }
    if language != voice["LanguageCode"]:
        request["LanguageCode"] = language  # bilingual voice speaking its second language
    started = time.perf_counter()
    response = get_polly().synthesize_speech(**request)
    audio = response["AudioStream"].read()
    return {
        "voice": voice["Id"],
        "engine": engine,
        "audio": audio,
        "format": output_format,
        "characters": response.get("RequestCharacters", len(text)),
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def apply_sample():
    label = st.session_state.get("polly_sample")
    if not label:
        return
    text, language, voice = SAMPLES[label]
    st.session_state["polly_text"] = text
    if language:
        st.session_state["polly_language"] = language
        st.session_state["polly_voice"] = voice
        st.session_state["polly_only_new"] = False


st.session_state.setdefault("polly_text", SAMPLES["Money & numbers"][0])
st.session_state.setdefault("polly_language", "en-US")
st.session_state.setdefault("polly_voice", "Kenneth")
st.session_state.setdefault("polly_results", [])

# ── Sidebar: voice ─────────────────────────────────────────────────────────────
with st.sidebar:
    engine = st.segmented_control("Engine", list(ENGINES), format_func=ENGINES.get, default="generative",
                                  key="polly_engine", required=True)
    try:
        engine_voices = list_voices(engine)
    except (ClientError, BotoCoreError) as err:
        st.error(f"Could not list Polly voices in {AWS_REGION}: {err}")
        st.stop()
    if not engine_voices:
        st.warning(f"No {ENGINES[engine]} voices in {AWS_REGION}.")
        st.stop()

    only_new = st.toggle("Only voices new in the October 2026 upgrade", key="polly_only_new",
                         disabled=engine != "generative")
    shown_voices = [v for v in engine_voices if v["Id"] in NEW_VOICES] if only_new and engine == "generative" \
        else engine_voices

    languages = languages_of(shown_voices)
    if st.session_state["polly_language"] not in languages:
        st.session_state["polly_language"] = "en-US" if "en-US" in languages else next(iter(languages))
    language = st.selectbox("Language", list(languages), format_func=languages.get, key="polly_language")

    language_voices = voices_for(shown_voices, language)
    voice_ids = [v["Id"] for v in language_voices]
    if st.session_state["polly_voice"] not in voice_ids:
        st.session_state["polly_voice"] = voice_ids[0]
    voice_id = st.selectbox(
        "Voice", voice_ids, key="polly_voice",
        format_func=lambda vid: f"{vid} · {next(v['Gender'] for v in language_voices if v['Id'] == vid)}"
                                + ("  (new)" if vid in NEW_VOICES and engine == "generative" else ""),
    )
    voice = next(v for v in language_voices if v["Id"] == voice_id)

    other_engines = [e for e in voice.get("SupportedEngines", []) if e != engine and e in ENGINES]
    compare_engine = st.selectbox(
        "Compare with", [None, *other_engines],
        format_func=lambda e: "Nothing" if e is None else f"{voice_id} on the {ENGINES[e]} engine",
        help="Play the same text with this voice on another engine, to hear the difference.",
        disabled=not other_engines,
    )
    output_format = st.radio("Audio format", list(OUTPUT_FORMATS), format_func=lambda f: f.split("_")[0].upper(),
                             horizontal=True)
    st.caption(f"{len(engine_voices)} {ENGINES[engine]} voices in {AWS_REGION}")

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown(f"#### :material/record_voice_over: :blue[{PAGE_TITLE}]")
st.markdown(
    "Type text and hear it spoken by an **Amazon Polly** voice. The **Generative** engine was upgraded in "
    "October 2026: speech sounds more natural, tone follows the meaning of the text, and numbers, money, "
    "dates, addresses and codes are read correctly."
)

with st.expander(":material/new_releases: What's new in the upgrade"):
    st.markdown(
        "- :blue-badge[Expressive] The voice picks its tone from the text: surprised, unsure, calm. No markup needed.\n"
        "- :blue-badge[Accurate reading] Prices, percentages, dates, times, addresses, phone numbers and codes.\n"
        "- :blue-badge[14 new voices] in 9 locales, plus **Hindi** and **Finnish** for the first time.\n"
        "- :blue-badge[Same API] Still `synthesize_speech(Engine=\"generative\")`, at the same price.\n\n"
        "New voices: " + " ".join(f":violet-badge[{name}]" for name in sorted(NEW_VOICES))
    )

st.pills("Try a sample", list(SAMPLES), key="polly_sample", on_change=apply_sample,
         help="Fills in the text. The Hindi and Finnish samples also switch the voice.")
text = st.text_area("Text", key="polly_text", height=140, max_chars=MAX_CHARACTERS)
is_ssml = st.toggle("Text is SSML", help="For example <speak>Hello <break time=\"1s\"/> world</speak>")

with st.container(horizontal=True, vertical_alignment="center"):
    speak = st.button("Speak", type="primary", icon=":material/play_arrow:", disabled=not text.strip())
    st.caption(f":gray[{voice_id} · {ENGINES[engine]} · {languages[language]}]")

if speak:
    runs = [engine] + ([compare_engine] if compare_engine else [])
    results = []
    with st.spinner("Generating speech..."):
        for run_engine in runs:
            try:
                results.append(synthesize(text, voice, run_engine, language, output_format, is_ssml))
            except ClientError as err:
                st.error(f"{voice_id} on {ENGINES[run_engine]}: {err.response['Error']['Message']}")
            except BotoCoreError as err:
                st.error(f"{voice_id} on {ENGINES[run_engine]}: {err}")
    st.session_state["polly_results"] = results

results = st.session_state["polly_results"]
if results:
    for column, result in zip(st.columns(len(results)), results):
        with column.container(border=True):
            st.markdown(f"**{result['voice']}** · {ENGINES[result['engine']]}")
            st.audio(result["audio"], format=OUTPUT_FORMATS[result["format"]])
            with st.container(horizontal=True):
                st.metric("Characters", f"{result['characters']:,}")
                st.metric("Latency", f"{result['latency_ms']:,} ms")
                st.metric("Size", f"{len(result['audio']) / 1024:,.0f} KB")
            extension = "mp3" if result["format"] == "mp3" else "ogg"
            st.download_button(
                "Download", result["audio"], icon=":material/download:", type="tertiary",
                file_name=f"polly_{result['voice']}_{result['engine']}_{datetime.now():%Y%m%d_%H%M%S}.{extension}",
                mime=OUTPUT_FORMATS[result["format"]], key=f"download_{result['engine']}",
            )
