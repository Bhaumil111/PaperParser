import os
import sys
import asyncio

from gemini_config import (
    extract_text_from_pdf,
    split_into_chunks,
    summarize_chunks,
    build_conversation_json,
    save_conversation_json,
    generate_low_quality_audio,
    generate_audio_high,
)

def main():
    if len(sys.argv) != 6:
        print("Usage: generate_podcast.py <pdf_path> <output_path> <AlexVoice> <AveryVoice> <quality>")
        sys.exit(1)

    pdf_path, output_path, Alex_voice, Avery_voice, quality = sys.argv[1:6]
    print(sys.argv[1:6])
    quality = quality.lower().strip()

    # 1) Extract
    paper_text = extract_text_from_pdf(pdf_path)

    # 2) Chunk
    chunks = split_into_chunks(paper_text, max_chars=3000)

    # 3) Summarize with Gemini 2.5 Flash
    summary = summarize_chunks(chunks)

    # 4) Build strict JSON conversation (Alex/Avery alternating)
    conversation = build_conversation_json(summary)

    # 5) Save conversation.json next to output file
    base, ext = os.path.splitext(output_path)
    json_path = base + ".conversation.json"
    save_conversation_json(conversation, json_path)

    # 6) Audio
    if quality == "low":
        # MP3
        asyncio.run(generate_low_quality_audio(conversation, Alex_voice, Avery_voice, output_path))
        print(f"[OK] Low quality MP3 created at: {output_path}")
    elif quality == "high":
        # WAV (please pass a '.wav' path from Node for correctness)
        if not output_path.lower().endswith(".wav"):
            # We still write WAV bytes to the given path to avoid breaking your pipeline.
            # For best results, pass a .wav filename for high quality.
            print("[WARN] High quality expects .wav output. Writing WAV bytes to the provided path anyway.")
        asyncio.run(generate_audio_high(conversation, Alex_voice, Avery_voice, output_path))
        print(f"[OK] High quality WAV created at: {output_path}")
    else:
        print("Error: <quality> must be 'low' or 'high'")
        sys.exit(1)

if __name__ == "__main__":
    main()
