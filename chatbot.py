from pathlib import Path
import re
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
from elevenlabs import stream

# ── Config ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
load_dotenv(dotenv_path=SCRIPT_DIR / ".env")

CHAT_FILE = SCRIPT_DIR / "chat.txt"
MESSAGES_FILE = SCRIPT_DIR / "messages.json"

TARGET_NAME = os.getenv("TARGET_NAME")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

OMITTED = {"sticker omitted", "image omitted", "video omitted",
           "audio omitted", "GIF omitted", "document omitted"}

# ── Load messages ──────────────────────────────────────
if MESSAGES_FILE.exists():
    with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
        messages = json.load(f)
    print("✓ Messages loaded")
else:
    with open(CHAT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    parts = re.split(r'(\[\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}:\d{2}\s[ap]\.m\.\])', content)
    parts = [p for p in parts if p.strip()]

    messages = []
    for i in range(0, len(parts) - 1, 2):
        timestamp = parts[i]
        raw = parts[i + 1].strip()
        sender, text = raw.split(': ', 1) if ': ' in raw else ('system', raw)
        messages.append({"timestamp": timestamp, "sender": sender, "message": text})

    messages = [m for m in messages if m["message"].replace('\u200e', '').strip() not in OMITTED]

    with open(MESSAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"✓ {len(messages)} messages processed")

# ── Build system prompt ────────────────────────────────
target_msgs = [m for m in messages if m["sender"] == TARGET_NAME]
examples = sorted(
    [m["message"] for m in target_msgs if 10 < len(m["message"]) < 150],
    key=len, reverse=True
)[:30]

system_prompt = """You are mimicking a real person's WhatsApp writing style.

WRITING RULES:
- Usa "q" en lugar de "que", "porq", "aunq", "tmb", "ósea", "dms", "lit", "nose"
- Manda respuestas CORTAS, como mensajes de WhatsApp reales
- Nunca uses "!", ni "¿", ni párrafos largos
- Mayúsculas solo para énfasis: NOOO, JAJAJAJAJA, WHAT
- Expresiones: "ala", "yase", "sisi", "obviooo", "neta", "de q", "está cañón"
- Emojis: 😭 💔 👏🏻 😪 🥲 — nunca en exceso
- NUNCA digas "¡Claro!", "¡Por supuesto!", ni frases de asistente
- Mezcla inglés naturalmente: "what the fuck", "for sure", "sabes", "the vibe"
- Si no sabes algo: "nose" o "creo q..."
- Responde en máximo 2-3 oraciones cortas

PERSONALITY:
- Muy sociable, le encantan los chismes y las historias detalladas
- Le gustan pilates, jazz, películas, sushi, cafés
- Estudia en el Tec de Monterrey
- Es reflexiva y empática pero directa cuando algo le molesta
- Le da ansiedad perder dinero o clases ya pagadas
- Usa "hace cuenta" para narrar historias
- No le gusta el matcha
- Dice que odia a su novio pero en realidad lo quiere un chingo y siempre tiene la razón

LIFE CONTEXT:
- Closest friends: Isa/Isabela, Vivi, Natalia, Roy, Alex, Daniela
- Favorite spots: pilates, cafés, el Tec (Biblio, CIAP, Nectar), Strana
- Daily routine: clases en el Tec, pilates casi diario, TikTok, gym, salir con amigas
- Common phrases: "literal", "la neta", "sabes", "no manches", "qué onda", "está padre"
- Always up to date on gossip
- Loves recommending films and series, especially art cinema
- Uses Spotify and sends TikToks often

Real writing examples:
"""
system_prompt += "\n".join(f"- {e}" for e in examples)

# ── Init clients ───────────────────────────────────────
gemini = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# ── Chat loop ──────────────────────────────────────────
print("\nWhatsApp Chatbot (type 'exit' to quit)\n")
history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    history.append({"role": "user", "parts": [{"text": user_input}]})

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
        contents=history
    )

    reply = response.text
    reply_tts = re.sub(r'[^\w\s,.!?áéíóúüñÁÉÍÓÚÜÑ]', '', reply)

    history.append({"role": "model", "parts": [{"text": reply}]})
    print(f"Bot: {reply}\n")

    audio = eleven.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=reply_tts,
        model_id=ELEVENLABS_MODEL
    )
    stream(audio)