# app.py
import os
import json
import csv
import random
from typing import List, Dict, Optional
import streamlit as st

# Azure OpenAI SDK (openai>=1.0.0)
try:
    from openai import AzureOpenAI
except ImportError:
    raise SystemExit("Install dependencies first: pip install streamlit openai>=1.0.0")

# ---------------------------
# Resource loading utilities
# ---------------------------

def load_idioms_from_csv_file(file) -> List[Dict]:
    """Load idioms from an uploaded CSV file-like object."""
    file.seek(0)
    text = file.read().decode("utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    out = []
    for row in rows:
        tags = (row.get("tags") or "")
        tags = [t.strip() for t in tags.replace("|", ",").split(",") if t.strip()]
        out.append({
            "idiom": row.get("idiom"),
            "gloss": row.get("gloss") or "",
            "tags": tags
        })
    return [x for x in out if x.get("idiom")]

def load_idioms_from_csv_path(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            tags = (row.get("tags") or "")
            tags = [t.strip() for t in tags.replace("|", ",").split(",") if t.strip()]
            out.append({
                "idiom": row.get("idiom"),
                "gloss": row.get("gloss") or "",
                "tags": tags
            })
    return [x for x in out if x.get("idiom")]

def fallback_idioms(lang: str) -> List[Dict]:
    data = {
        "Italian": [
            {"idiom": "il troppo stroppia", "gloss": "too much of anything backfires", "tags": ["excess","caution"]},
            {"idiom": "non tutte le ciambelle riescono col buco", "gloss": "not everything turns out perfect", "tags": ["imperfection","acceptance"]},
            {"idiom": "tra il dire e il fare c’è di mezzo il mare", "gloss": "saying is easier than doing", "tags": ["action","effort"]},
            {"idiom": "l’unione fa la forza", "gloss": "unity makes strength", "tags": ["teamwork","solidarity"]},
        ],
        "English": [
            {"idiom": "too many cooks spoil the broth", "gloss": "too many people hinder progress", "tags": ["teamwork","excess"]},
            {"idiom": "actions speak louder than words", "gloss": "deeds matter more than words", "tags": ["action","integrity"]},
            {"idiom": "the early bird catches the worm", "gloss": "success comes to those who start early", "tags": ["effort","discipline"]},
            {"idiom": "when in Rome do as the Romans do", "gloss": "adapt to local customs", "tags": ["culture","adaptation"]},
        ],
    }
    return data.get(lang, [])

def load_language_resource(
    culture: str,
    uploaded_csv,  # st.file_uploader object or None
    local_csv_path: Optional[str] = None
) -> List[Dict]:
    """
    Priority: uploaded CSV (if provided) -> local CSV path (if provided) -> fallback list.
    """
    idioms: List[Dict] = []
    if uploaded_csv is not None:
        try:
            idioms = load_idioms_from_csv_file(uploaded_csv)
        except Exception as e:
            st.warning(f"Could not parse uploaded CSV: {e}")
    elif local_csv_path and os.path.exists(local_csv_path):
        try:
            idioms = load_idioms_from_csv_path(local_csv_path)
        except Exception as e:
            st.warning(f"Could not load local CSV: {e}")

    if not idioms:
        idioms = fallback_idioms(culture)

    # de-duplicate
    seen = set()
    uniq = []
    for it in idioms:
        if not it.get("idiom"):
            continue
        key = (it["idiom"].strip().lower(), (it.get("gloss","") or "").strip().lower())
        if key not in seen:
            uniq.append(it)
            seen.add(key)
    return uniq

# ---------------------------
# Optional semantic filtering
# ---------------------------

def get_embeddings(client: AzureOpenAI, texts: List[str], deployment: str) -> List[List[float]]:
    resp = client.embeddings.create(model=deployment, input=texts)
    return [d.embedding for d in resp.data]

def cosine_sim(a, b) -> float:
    import math
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot/(na*nb)

def pick_candidate_idioms(
    text: str,
    idioms: List[Dict],
    client: Optional[AzureOpenAI],
    emb_deployment: Optional[str],
    k: int
) -> List[Dict]:
    if client and emb_deployment:
        try:
            query_emb = get_embeddings(client, [text], emb_deployment)[0]
            idiom_texts = [f'{it["idiom"]} :: {it.get("gloss","")}' for it in idioms]
            id_embs = get_embeddings(client, idiom_texts, emb_deployment)
            scored = [(i, cosine_sim(query_emb, e)) for i, e in enumerate(id_embs)]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [idioms[i] for i,_ in scored[:k]]
        except Exception as e:
            st.info(f"Embedding selection failed, falling back to random: {e}")
    # fallback random
    return idioms[:k] if len(idioms) <= k else random.sample(idioms, k)

# ---------------------------
# Azure OpenAI call
# ---------------------------

SYSTEM_PROMPT = """You are a careful cultural localization assistant.
Given: (a) a short title/comment/annotation text for a story, (b) a target language,
and (c) a small bank of idioms/proverbs for that culture with glosses,
adapt the text by inserting exactly ONE idiom or proverb naturally and appropriately.

Rules:
- Preserve the controlling idea and core message.
- Insert exactly one idiom/proverb; integrate it smoothly (no brackets).
- Keep the target language consistent and the tone appropriate for the content type.
- If no candidate is perfect, choose the closest and slightly rephrase the text.
- Output strictly as JSON: adapted_text, idiom_used, justification.
"""

USER_TEMPLATE = """TARGET_LANGUAGE: {lang}

CONTENT_TYPE: {ctype}

ORIGINAL_TEXT:
{original}

CANDIDATE_IDIOMS:
{idioms_block}

Instructions:
1) Pick ONE idiom that best reinforces the controlling idea.
2) Rewrite the text in {lang} integrating that idiom smoothly.
3) Return JSON only:
{{
  "adapted_text": "...",
  "idiom_used": "...",
  "justification": "Why this idiom preserves the message and fits the context."
}}
"""

def format_idioms_block(idioms: List[Dict]) -> str:
    lines = []
    for it in idioms:
        tags = ", ".join(it.get("tags", []))
        lines.append(f'- idiom: "{it["idiom"]}" | gloss: "{it.get("gloss","")}" | tags: [{tags}]')
    return "\n".join(lines)

def call_azure_chat(client: AzureOpenAI, deployment: str, system: str, user: str) -> Dict:
    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.7,
        max_tokens=600,
        response_format={"type": "json_object"}
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        if not all(k in data for k in ("adapted_text","idiom_used","justification")):
            raise ValueError("Missing keys in model output.")
        return data
    except Exception as e:
        return {
            "adapted_text": "",
            "idiom_used": "",
            "justification": f"Model did not return valid JSON. Raw: {content[:300]}... Error: {e}"
        }

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Culturally Adaptive Data Storytelling", page_icon="📚", layout="centered")

st.title("📚 Culturally Adaptive Data Storytelling")
st.caption("Insert an idiom or proverb into a title/comment/annotation using Azure OpenAI + idiom corpora.")

with st.sidebar:
    st.header("🔧 Azure OpenAI Settings")
    endpoint = st.text_input("AZURE_OPENAI_ENDPOINT", value=os.environ.get("AZURE_OPENAI_ENDPOINT",""))
    api_key = st.text_input("AZURE_OPENAI_API_KEY", type="password", value=os.environ.get("AZURE_OPENAI_API_KEY",""))
    chat_deployment = st.text_input("Chat deployment (e.g., gpt-4o-mini)", value=os.environ.get("AZURE_OPENAI_DEPLOYMENT",""))
    use_embeddings = st.checkbox("Use embeddings for idiom selection", value=False)
    emb_deployment = ""
    if use_embeddings:
        emb_deployment = st.text_input("Embeddings deployment (e.g., text-embedding-3-large)",
                                       value=os.environ.get("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",""))

st.subheader("1) Input")
content_type = st.radio("Content type", ["Title", "Comment", "Annotation"], horizontal=True)
culture = st.selectbox("Target culture / language", ["Italian", "English"])
text = st.text_area("Original text", height=120, placeholder="Paste your title/comment/annotation here...")

st.subheader("2) Idiom Corpus")
st.write("Upload a CSV with columns: `idiom,gloss,tags` (tags separated by comma or |). If not provided, a fallback set will be used.")
uploaded_csv = st.file_uploader("Upload idioms CSV (optional)", type=["csv"])

with st.expander("Optional local CSV path (if running locally)"):
    local_csv_path = st.text_input("Path to local CSV", value="")

k_candidates = st.slider("Max candidate idioms to consider", min_value=3, max_value=12, value=6, step=1)
n_outputs = st.slider("Number of alternative adaptations to generate", min_value=1, max_value=5, value=3, step=1)
random_seed = st.number_input("Random seed (optional)", min_value=0, value=0, step=1)

st.subheader("3) Generate")
go = st.button("Generate adaptations")

if go:
    if not text.strip():
        st.error("Please provide some text.")
        st.stop()
    if not endpoint or not api_key or not chat_deployment:
        st.error("Please fill Azure endpoint, API key, and chat deployment in the sidebar.")
        st.stop()

    random.seed(random_seed or None)

    # Initialize client
    client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version="2024-05-01-preview")

    # Load idioms
    idioms = load_language_resource(culture=culture, uploaded_csv=uploaded_csv, local_csv_path=local_csv_path)
    if not idioms:
        st.error(f"No idioms available for {culture}. Provide a CSV or extend the fallback list.")
        st.stop()

    # Candidate selection
    candidates = pick_candidate_idioms(
        text=text,
        idioms=idioms,
        client=client if use_embeddings and emb_deployment else None,
        emb_deployment=emb_deployment if use_embeddings else None,
        k=k_candidates
    )

    # Prepare prompt once
    idioms_block = format_idioms_block(candidates)
    user_msg = USER_TEMPLATE.format(
        lang=culture,
        ctype=content_type,
        original=text,
        idioms_block=idioms_block
    )

    st.info(f"Using {len(candidates)} idiom candidates from the corpus.")
    results = []
    for i in range(n_outputs):
        data = call_azure_chat(client, chat_deployment, SYSTEM_PROMPT, user_msg)
        results.append(data)

    st.subheader("Results")
    for idx, r in enumerate(results, start=1):
        st.markdown(f"**Alternative {idx}**")
        st.json(r)

    st.success("Done.")

st.markdown("---")
st.markdown("Tip: for best results, provide a CSV idiom corpus per culture and enable embeddings for semantic matching.")
