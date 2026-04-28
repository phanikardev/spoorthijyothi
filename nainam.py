import os
import hashlib
import streamlit as st
from gtts import gTTS
from tempfile import NamedTemporaryFile
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from audio_recorder_streamlit import audio_recorder
import openai

# ── Config ────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Sphoorthi Jyothi Foundation - AI Studio", page_icon="🎙️", layout="wide")

# ── Session State ─────────────────────────────────────────
for key, val in  {
    "last_audio_hash": None,
    "topic": None,
    "title_result": None,
    "speech_result": None,
    "audio_out": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Lookup Tables ─────────────────────────────────────────
TONES = {
    "Motivational":  "motivational and high-energy, using powerful action words",
    "Formal":        "formal and professional, using sophisticated vocabulary",
    "Casual":        "casual and conversational, like talking to a close friend",
    "Humorous":      "humorous and witty, with light jokes and clever wordplay",
    "Empathetic":    "empathetic and compassionate, deeply acknowledging emotions",
    "Political":     "persuasive and bold, like a powerful campaign rally speech",
    "Academic":      "analytical and evidence-driven, like a university lecture",
    "Inspirational": "deeply inspiring, like a world-class TED talk",
    "Storytelling":  "narrative-driven, using vivid imagery and personal anecdotes",
}

LANGUAGES = {
    "English":    "en",  "Spanish":    "es",  "French":     "fr",
    "German":     "de",  "Hindi":      "hi",  "Tamil":      "ta",
    "Telugu":     "te",  "Arabic":     "ar",  "Japanese":   "ja",
    "Chinese":    "zh",  "Portuguese": "pt",  "Italian":    "it",
    "Korean":     "ko",  "Russian":    "ru",  "Dutch":      "nl",
}

OPENAI_VOICES = {
    "Alloy  — Neutral / Balanced":        "alloy",
    "Echo   — Clear Male":                "echo",
    "Fable  — British Storyteller":       "fable",
    "Onyx   — Deep Authoritative Male":   "onyx",
    "Nova   — Warm Female":               "nova",
    "Shimmer — Soft Expressive Female":   "shimmer",
}

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Studio Settings")
    st.divider()

    st.subheader("🎭 Tone")
    tone = st.selectbox("Speech tone", list(TONES.keys()), label_visibility="collapsed")

    st.subheader("🌍 Output Language")
    language = st.selectbox("Language", list(LANGUAGES.keys()), label_visibility="collapsed")

    st.subheader("📝 Speech Length")
    speech_length = st.slider("Words", 100, 700, 350, 50, label_visibility="collapsed")

    st.divider()
    st.subheader("🔊 TTS Engine")
    tts_engine = st.radio("Engine", ["OpenAI TTS (HD)", "gTTS (Free)"], label_visibility="collapsed")

    if tts_engine == "OpenAI TTS (HD)":
        st.subheader("🎤 Voice & Accent")
        voice_label = st.selectbox("Voice", list(OPENAI_VOICES.keys()), label_visibility="collapsed")
        voice_key   = OPENAI_VOICES[voice_label]
        tts_model   = st.radio("Quality", ["tts-1-hd", "tts-1"], label_visibility="collapsed",
                               help="tts-1-hd = higher quality, slower")

    st.divider()
    if st.button("🗑️ Clear & Reset", use_container_width=True):
        for k in ["topic", "title_result", "speech_result", "audio_out", "last_audio_hash"]:
            st.session_state[k] = None
        st.rerun()

# ── Header ────────────────────────────────────────────────
st.title("🎙️ Sphoorthi Jyothi Foundation - AI Studio")
st.caption("Generate speeches from text or voice — control tone, language, voice & accent")
st.divider()

# ── Input ─────────────────────────────────────────────────
input_mode = st.radio("Input mode", ["✏️ Type a topic", "🎤 Speak your topic"], horizontal=True)

if input_mode == "✏️ Type a topic":
    st.session_state.topic = st.text_input(
        "Topic", placeholder="e.g. Climate action, Mental resilience, Future of AI ...",
        label_visibility="collapsed"
    )

else:
    st.markdown("**🔴 Click the mic — speak — click again to stop**")
    audio_bytes = audio_recorder(pause_threshold=2.5, sample_rate=16000)

    if audio_bytes:
        audio_hash = hashlib.md5(audio_bytes).hexdigest()

        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            st.audio(audio_bytes, format="audio/wav")

            with st.spinner("🔍 Transcribing with Whisper ..."):
                with NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    tmp.flush()
                    with open(tmp.name, "rb") as af:
                        transcript = openai_client.audio.transcriptions.create(
                            model="whisper-1", file=af
                        )
            st.session_state.topic = transcript.text
            # Reset downstream results for new recording
            st.session_state.title_result  = None
            st.session_state.speech_result = None
            st.session_state.audio_out     = None

        if st.session_state.topic:
            st.success(f"📝 Heard: **{st.session_state.topic}**")

# ── Generate Button ───────────────────────────────────────
if st.session_state.topic:
    st.divider()
    left, right = st.columns([4, 1])
    with left:
        st.info(f"**Topic:** {st.session_state.topic} &nbsp;|&nbsp; "
                f"**Tone:** {tone} &nbsp;|&nbsp; "
                f"**Language:** {language} &nbsp;|&nbsp; "
                f"**Words:** ~{speech_length}")
    with right:
        generate = st.button("🚀 Generate", type="primary", use_container_width=True)

    if generate:
        tone_desc = TONES[tone]

        # 1 — Title
        with st.spinner("✍️ Crafting title ..."):
            title_chain = (
                PromptTemplate(
                    input_variables=["topic", "tone", "language"],
                    template=(
                        "You are an expert speech writer. "
                        "Create one short, powerful, memorable title for a {tone} speech about: {topic}. "
                        "Write the title in {language}. Output the title only."
                    ),
                )
                | llm | StrOutputParser()
            )
            st.session_state.title_result = title_chain.invoke({
                "topic": st.session_state.topic,
                "tone":  tone_desc,
                "language": language,
            })

        # 2 — Speech body
        with st.spinner("🖊️ Writing speech ..."):
            speech_chain = (
                PromptTemplate(
                    input_variables=["title", "tone", "length", "language"],
                    template=(
                        "Write a {tone} speech of approximately {length} words for this title: '{title}'. "
                        "Write entirely in {language}. "
                        "Structure: strong opening hook, 2-3 body points, powerful closing call to action. "
                        "Do NOT add section labels."
                    ),
                )
                | llm | StrOutputParser()
            )
            st.session_state.speech_result = speech_chain.invoke({
                "title":    st.session_state.title_result,
                "tone":     tone_desc,
                "length":   speech_length,
                "language": language,
            })

        # 3 — Text-to-Speech
        with st.spinner("🔊 Rendering audio ..."):
            try:
                if tts_engine == "OpenAI TTS (HD)":
                    resp = openai_client.audio.speech.create(
                        model=tts_model,
                        voice=voice_key,
                        input=st.session_state.speech_result,
                    )
                    st.session_state.audio_out = (resp.content, "audio/mpeg", "mp3")
                else:
                    lang_code = LANGUAGES.get(language, "en")
                    tts = gTTS(text=st.session_state.speech_result, lang=lang_code)
                    with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tts.save(tmp.name)
                        tmp.seek(0)
                        data = tmp.read()
                    st.session_state.audio_out = (data, "audio/mp3", "mp3")
            except Exception as e:
                st.error(f"TTS Error: {e}")

# ── Results ───────────────────────────────────────────────
if st.session_state.title_result:
    st.divider()
    col_text, col_audio = st.columns([3, 2])

    with col_text:
        st.subheader("🪶 Title")
        st.markdown(f"## {st.session_state.title_result}")

        st.subheader("🗣️ Speech")
        with st.expander("Read full speech", expanded=True):
            st.write(st.session_state.speech_result)
            st.download_button(
                "⬇️ Download transcript (.txt)",
                data=st.session_state.speech_result,
                file_name="speech.txt",
                mime="text/plain",
            )

    with col_audio:
        st.subheader("🔊 Audio Player")
        if st.session_state.audio_out:
            data, fmt, ext = st.session_state.audio_out
            st.audio(data, format=fmt)
            st.download_button(
                f"⬇️ Download audio (.{ext})",
                data=data,
                file_name=f"speech_{language.lower()}.{ext}",
                mime=fmt,
            )

        st.subheader("📊 Stats")
        words = len(st.session_state.speech_result.split())
        st.metric("Word count",  words)
        st.metric("Language",    language)
        st.metric("Tone",        tone)
        engine_label = voice_label if tts_engine == "OpenAI TTS (HD)" else "gTTS"
        st.metric("Voice",       engine_label)
