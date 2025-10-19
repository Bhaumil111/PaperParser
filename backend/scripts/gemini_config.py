import os
import re
import json
import asyncio
import tempfile
from typing import List, Dict
import time
from google.genai.errors import ClientError
import fitz  # PyMuPDF
from google import genai
from dotenv import load_dotenv

load_dotenv()

from voice import (
    tts_edge_single_speaker,
    merge_mp3_files,
    tts_gemini_multi_speaker,
)

# ---------- Gemini client setup ----------
# Allow either GEMINI_API_KEY or GOOGLE_API_KEY
if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables.")

# Initialize Gemini client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# ---------- Text generation helper ----------

def _gen_model_text(prompt, model="gemini-2.5-flash", temperature=0.5):
    while True:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": temperature}
            )
            return resp.text
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print("Quota hit, waiting 10s before retry...")
                time.sleep(10)
            else:
                raise


# ---------- PDF -> text ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)

# ---------- chunk ----------
def split_into_chunks(text: str, max_chars: int = 3000) -> List[str]:
    lines = text.splitlines()
    chunks, cur = [], []
    cur_len = 0
    for ln in lines:
        add = len(ln) + 1
        if cur_len + add > max_chars and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [ln], add
        else:
            cur.append(ln)
            cur_len += add
    if cur:
        chunks.append("\n".join(cur))
    return chunks

# ---------- summarize ----------
def summarize_chunks(chunks: List[str]) -> str:
    summaries = []
    for chunk in chunks:
        prompt = f"""
You are an expert science communicator explaining research papers to an educated but non-specialist audience.
Read the following section of a paper and create a teaching-style summary:
- 5–7 sentences.
- Explain technical terms briefly if needed.
- Preserve important details about definitions, methods, results, and context.
- Be clear and engaging, like explaining to graduate students or podcast listeners.

SECTION:
{chunk}
"""
        summaries.append(_gen_model_text(prompt, model="gemini-2.5-flash", temperature=0.5))

    combined = " ".join(summaries)

    compress_prompt = f"""
Combine the following summaries into a single coherent overview (15–20 sentences).
Write it as if preparing detailed notes for a podcast discussion:
- Cover definitions, motivation, methods, key results, limitations, and implications.
- Maintain clarity and logical flow.
- Avoid bullet points or list formatting; use natural paragraphs.

SUMMARIES:
{combined}
"""
    return _gen_model_text(compress_prompt, model="gemini-2.5-flash", temperature=0.4)

# ---------- conversation JSON ----------
CONVO_PROMPT_TEMPLATE = """
Return a STRICT JSON array where each element is:
{{"speaker": "Alex" or "Avery", "text": "<1–3 sentence natural utterance>"}}

Rules:
- Alternate speakers naturally and evenly.
- Keep style friendly but informed; no stage directions.
- Use only facts from the paper summary provided.
- Do NOT include metadata, headings, Markdown, or prose outside the JSON.
- The text must be ONLY what the speaker says (no "Alex:" prefixes in text).

Paper summary to base the conversation on:
{summary}
"""

def _try_parse_json_array(s: str) -> List[Dict[str, str]]:
    data = json.loads(s)
    if not isinstance(data, list):
        raise ValueError("Root is not a list.")
    # minimal validation
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sp = item.get("speaker", "").strip()
        tx = item.get("text", "").strip()
        if sp in ("Alex", "Avery") and tx:
            out.append({"speaker": sp, "text": tx})
    if not out:
        raise ValueError("No valid items found.")
    return out

def build_conversation_json(summary: str) -> List[Dict[str, str]]:
    # 1st attempt
    raw = _gen_model_text(CONVO_PROMPT_TEMPLATE.format(summary=summary), model="gemini-2.5-flash", temperature=0.5)
    try:
        return _try_parse_json_array(raw)
    except Exception:
        pass

    # Try to extract JSON array substring
    m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if m:
        try:
            return _try_parse_json_array(m.group(0))
        except Exception:
            pass

    # Last-chance repair prompt
    repair_prompt = f"""
You produced JSON-like text but it was invalid.
Rewrite the following so that it is ONLY a valid JSON array of objects with exactly:
  - "speaker": "Alex" or "Avery"
  - "text": string
No commentary. No Markdown. Just the JSON array.

INPUT:
{raw}
"""
    repaired = _gen_model_text(repair_prompt, model="gemini-2.5-flash", temperature=0.0)
    return _try_parse_json_array(repaired)

def save_conversation_json(conversation: List[Dict[str, str]], json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)

# ---------- audio generation ----------
async def generate_low_quality_audio(conversation: List[Dict[str, str]],
                                     Alex_voice_label: str,
                                     Avery_voice_label: str,
                                     output_mp3_path: str) -> None:
    """
    Low-quality path:
    - Make one MP3 per dialogue line (speaker speaks ONLY their text).
    - Merge into a single MP3.
    """
    tmpdir = tempfile.mkdtemp(prefix="podcast_segments_")
    seg_paths: List[str] = []
    try:
        idx = 1
        for turn in conversation:
            speaker = turn["speaker"]
            text = turn["text"]
            voice_label = Alex_voice_label if speaker == "Alex" else Avery_voice_label
            seg_path = os.path.join(tmpdir, f"seg_{idx:04d}.mp3")
            await tts_edge_single_speaker(text, voice_label, seg_path)
            seg_paths.append(seg_path)
            idx += 1

        merge_mp3_files(seg_paths, output_mp3_path)
    finally:
        # clean temp files
        for p in seg_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(tmpdir)
        except Exception:
            pass

async def generate_audio_high(conversation: List[Dict[str, str]],
                              Alex_voice_label: str,
                              Avery_voice_label: str,
                              output_wav_path: str) -> None:
    """
    High-quality path (Gemini multi-speaker, single shot):
    - Build 'Alex: ...\\nAvery: ...' transcript (labels route to correct voice).
    - Produce a WAV at output_wav_path.
    NOTE: Ensure your Node layer passes a '.wav' path for high quality.
    """
    convo_text = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in conversation)
    await tts_gemini_multi_speaker(convo_text, Alex_voice_label, Avery_voice_label, output_wav_path)
