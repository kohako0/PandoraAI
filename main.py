
import lib.config

import requests
import re
import time
import threading
import html

import numpy as np
import sounddevice as sd
import soundfile as sf

from supertonic import TTS
from pathlib import Path
from datetime import datetime


# =====================================
# PANDORA - LLAMA.CPP + SUPERTONIC
# =====================================

API_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = lib.config.SYSTEMPROMPT

# Llama.cpp
TEMPERATURE = 0.7
MAX_TOKENS = 2048
TIMEOUT = 600

# =====================================
# STIMME
# =====================================

VOICE = "F4"
LANGUAGE = "de"

# Nahe an der natürlichen Modellgeschwindigkeit
SPEED = 1.0

# 8 Schritte als Qualitäts-/Tempo-Kompromiss
STEPS = 8

VOLUME = 0.80
SAMPLE_RATE = 44100

SAVE_AUDIO = False
PLAY_AUDIO = True

# Kurze Abschnitte gegen Ausspracheprobleme
MAX_CHARS = 160

# Pause zwischen den Abschnitten
PAUSE = 0.18

OUTPUT_DIR = Path("audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

speech_lock = threading.Lock()


# =====================================
# TEXT FÜR SPRACHAUSGABE BEREINIGEN
# =====================================

def clean_tts_text(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    # Codeblöcke nicht vorlesen
    text = re.sub(
        r"```[\s\S]*?```",
        " Codebeispiel. ",
        text
    )

    # Markdown-Links: nur Bezeichnung sprechen
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # URLs entfernen
    text = re.sub(
        r"https?://\S+",
        "Link",
        text
    )

    # Markdown-Überschriften
    text = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s*",
        "",
        text
    )

    # Aufzählungszeichen
    text = re.sub(
        r"(?m)^\s*[-*•]\s+",
        "",
        text
    )

    # Markdown-Sonderzeichen
    text = re.sub(
        r"[*_`~]",
        "",
        text
    )

    # Whitespace normalisieren
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================
# AUDIO VERARBEITEN
# =====================================

def process_audio(wav):
    audio = np.asarray(
        wav,
        dtype=np.float32
    ).reshape(-1).copy()

    if len(audio) == 0:
        return audio

    # Lautstärke anpassen
    audio *= VOLUME

    # Übersteuerung verhindern
    audio = np.clip(audio, -1.0, 1.0)

    # Sanftes Ein- und Ausblenden
    fade = min(
        int(SAMPLE_RATE * 0.012),
        len(audio) // 2
    )

    if fade > 0:
        audio[:fade] *= np.linspace(
            0.0, 1.0, fade
        )

        audio[-fade:] *= np.linspace(
            1.0, 0.0, fade
        )

    return audio


# =====================================
# TTS INITIALISIEREN
# =====================================

print("\nLade Pandora TTS...")

tts = TTS(auto_download=True)

style = tts.get_voice_style(
    voice_name=VOICE
)

print("Supertonic bereit!")
print(f"Stimme: {VOICE}")
print(f"Sprache: {LANGUAGE}")
print(f"Geschwindigkeit: {SPEED}")


# =====================================
# SPRACHAUSGABE
# =====================================

def speak(text):
    text = clean_tts_text(text)

    if not text:
        return

    with speech_lock:
        start = time.perf_counter()

        print("\nPandora generiert Sprache...")

        try:
            # EIN EINZIGER Syntheseaufruf
            # Supertonic teilt lange Texte intern auf.
            wav, duration = tts.synthesize(
                text=text,
                voice_style=style,
                lang=LANGUAGE,
                speed=SPEED,
                total_steps=STEPS,
                max_chunk_length=MAX_CHARS,
                silence_duration=PAUSE
            )

            audio = process_audio(wav)

            if len(audio) == 0:
                print("Keine Audiodaten generiert.")
                return

            # Optional speichern
            if SAVE_AUDIO:
                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                path = OUTPUT_DIR / (
                    f"pandora_{timestamp}.wav"
                )

                sf.write(
                    str(path),
                    audio,
                    SAMPLE_RATE
                )

                print(f"Audio gespeichert: {path.resolve()}")

            # Genau einmal abspielen
            if PLAY_AUDIO:
                sd.play(
                    audio,
                    samplerate=SAMPLE_RATE
                )
                sd.wait()

            elapsed = time.perf_counter() - start

            print(
                f"\nSprachausgabe fertig: {elapsed:.2f}s\n"
            )

        except Exception as error:
            sd.stop()
            print(f"\nTTS-Fehler: {error}")


# =====================================
# LLAMA.CPP CHAT
# =====================================

def chat(user_input):
    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = requests.post(
            API_URL,
            json={
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False
            },
            timeout=TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        answer = data["choices"][0]["message"]["content"]

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "Das Modell hat keinen gültigen Antworttext geliefert."
            )

        answer = answer.strip()

        messages.append({
            "role": "assistant",
            "content": answer
        })

        return answer

    except (
        requests.exceptions.RequestException,
        KeyError,
        IndexError,
        ValueError,
        TypeError
    ) as error:

        # Fehlgeschlagene Nachricht entfernen
        messages.pop()

        print(f"\nChat-Fehler: {error}")

        return None


# =====================================
# TEST DER SPRACHAUSGABE
# =====================================

def test_voice():
    test_text = (
        "Hallo! Ich bin Pandora. "
        "Schön, dass du wieder da bist. "
        "Wie war dein Tag bisher? "
        "Ich hoffe, du hattest einen schönen Tag! "
        "Erzähl mir doch, was du heute gemacht hast."
    )

    print("\nStarte Sprachtest...")
    speak(test_text)


# =====================================
# HAUPTSCHLEIFE
# =====================================

def main():
    print("\n" + "=" * 50)
    print("          PANDORA - LOCAL AI")
    print("=" * 50)

    print("llama.cpp + Supertonic")
    print()
    print("/exit       - Beenden")
    print("/clear      - Chat zurücksetzen")
    print("/mute       - Sprache umschalten")
    print("/testvoice  - Stimme testen")
    print()

    voice_enabled = True

    while True:
        try:
            user_input = input("Du: ").strip()

            if not user_input:
                continue

            command = user_input.lower()

            if command == "/exit":
                break

            if command == "/clear":
                messages[:] = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    }
                ]

                print(
                    "Gesprächsverlauf zurückgesetzt.\n"
                )
                continue

            if command == "/mute":
                voice_enabled = not voice_enabled

                print(
                    "Sprache:",
                    "AN" if voice_enabled else "AUS",
                    "\n"
                )
                continue

            if command == "/testvoice":
                test_voice()
                continue

            print("\nPandora: ", end="", flush=True)

            answer = chat(user_input)

            if answer is None:
                print()
                continue

            print(answer)
            print()

            # Antwort genau einmal sprechen
            if voice_enabled:
                speak(answer)

        except KeyboardInterrupt:
            sd.stop()
            print("\nChat beendet.")
            break

        except Exception as error:
            sd.stop()
            print(f"\nFehler: {error}")

    print("Pandora beendet.")


if __name__ == "__main__":
    main()
