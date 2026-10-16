
from supertonic import TTS
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
import textwrap
import re
import time

from num2words import num2words


# =====================================
# PANDORA  - NATÜRLICHE ANIME STIMME
# =====================================

VOICE = "F4"
LANGUAGE = "de"

SPEED = 0.90
PITCH = 1.09
STEPS = 8
VOLUME = 0.80

SAMPLE_RATE = 44100

SAVE_AUDIO = True
PLAY_AUDIO = True

# Maximale Textlänge pro Sprachgenerierung
MAX_CHARS = 300

# Pause zwischen Textabschnitten
PAUSE = 0.15

OUTPUT_DIR = Path("audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# ZAHLEN IN WÖRTER UMWANDELN
# =====================================

def number_words(number):
    """Wandelt eine Zahl in deutsche Wörter um."""
    return num2words(int(number), lang="de")


def year_words(year):
    """Spricht Jahreszahlen natürlich auf Deutsch aus."""
    year = int(year)

    if 1100 <= year <= 1999:
        first = year // 100
        last = year % 100

        if last == 0:
            return number_words(first) + "hundert"

        return (
            number_words(first)
            + "hundert"
            + number_words(last)
        )

    return number_words(year)


# =====================================
# DATUMSANGABEN
# =====================================

MONTHS = {
    1: "Jänner",
    2: "Februar",
    3: "März",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Dezember",
}

MONTH_PATTERN = "|".join(MONTHS.values())

ORDINALS = {
    1: "erster",
    2: "zweiter",
    3: "dritter",
    4: "vierter",
    5: "fünfter",
    6: "sechster",
    7: "siebter",
    8: "achter",
    9: "neunter",
    10: "zehnter",
    11: "elfter",
    12: "zwölfter",
    13: "dreizehnter",
    14: "vierzehnter",
    15: "fünfzehnter",
    16: "sechzehnter",
    17: "siebzehnter",
    18: "achtzehnter",
    19: "neunzehnter",
    20: "zwanzigster",
    21: "einundzwanzigster",
    22: "zweiundzwanzigster",
    23: "dreiundzwanzigster",
    24: "vierundzwanzigster",
    25: "fünfundzwanzigster",
    26: "sechsundzwanzigster",
    27: "siebenundzwanzigster",
    28: "achtundzwanzigster",
    29: "neunundzwanzigster",
    30: "dreißigster",
    31: "einunddreißigster",
}


def make_date(day, month, year):
    """Erstellt ein natürlich gesprochenes deutsches Datum."""
    day = int(day)
    year = int(year)

    if not 1 <= day <= 31:
        return None

    if isinstance(month, str):
        month_name = month
    else:
        month = int(month)

        if month not in MONTHS:
            return None

        month_name = MONTHS[month]

    return (
        f"{ORDINALS[day]} "
        f"{month_name} "
        f"{year_words(year)}"
    )


# =====================================
# TEXT FÜR DIE SPRACHGENERIERUNG
# =====================================

def text_fuer_tts(text):

    # Datumsformat: 19.10.2026
    def numeric_date(match):
        day, month, year = match.groups()

        try:
            datetime(int(year), int(month), int(day))
        except ValueError:
            return match.group(0)

        return make_date(day, month, year)

    text = re.sub(
        r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)",
        numeric_date,
        text,
    )

    # Datumsformat: 15. Mai 1955
    def written_date(match):
        day, month_name, year = match.groups()

        month_lookup = {
            name.lower(): name for name in MONTHS.values()
        }

        month_name = month_lookup.get(
            month_name.lower(), month_name
        )

        return make_date(day, month_name, year)

    text = re.sub(
        rf"\b(\d{{1,2}})\.\s*({MONTH_PATTERN})\s+(\d{{4}})\b",
        written_date,
        text,
        flags=re.IGNORECASE,
    )


    def convert_year(match):
        return year_words(match.group(0))

    text = re.sub(
        r"(?<!\d)(1[1-9]\d{2}|20\d{2})(?!\d)",
        convert_year,
        text,
    )

    def convert_percent(match):
        value = match.group(1).replace(",", ".")

        if "." in value:
            whole, decimal = value.split(".", 1)
            spoken = (
                number_words(whole)
                + " Komma "
                + " ".join(number_words(d) for d in decimal)
            )
        else:
            spoken = number_words(value)

        return spoken + " Prozent"

    text = re.sub(
        r"(?<!\w)(\d+(?:[.,]\d+)?)\s*%",
        convert_percent,
        text,
    )

    # Einfache Brüche: 1/3 -> eins von drei
    text = re.sub(
        r"(?<!\d)(\d+)/(\d+)(?!\d)",
        lambda m: (
            number_words(m.group(1))
            + " von "
            + number_words(m.group(2))
        ),
        text,
    )

    # Dezimalzahlen: 3,5 -> drei Komma fünf
    def convert_decimal(match):
        whole = match.group(1)
        decimal = match.group(2)

        return (
            number_words(whole)
            + " Komma "
            + " ".join(number_words(d) for d in decimal)
        )

    text = re.sub(
        r"(?<!\w)(\d+)[,.](\d+)(?!\w)",
        convert_decimal,
        text,
    )

    # Alle übrigen ganzen Zahlen
    text = re.sub(
        r"(?<!\w)\d+(?!\w)",
        lambda m: number_words(m.group(0)),
        text,
    )

    # Mathematische Zeichen
    text = text.replace("×", " mal ")
    text = text.replace("=", " gleich ")
    text = text.replace("/", " durch ")

    # Häufige Abkürzungen für eine klarere Aussprache
    abbreviations = {
        "BIP": "B I P",
        "BNE": "B N E",
        "VPI": "V P I",
        "HVPI": "H V P I",
        "AMS": "A M S",
        "ILO": "I L O",
        "EZB": "E Z B",
        "VGR": "V G R",
        "EU": "E U",
        "USA": "U S A",
        "NS": "N S",
    }

    for abbreviation, spoken in abbreviations.items():
        text = re.sub(
            rf"\b{abbreviation}\b",
            spoken,
            text,
            flags=re.IGNORECASE,
        )

    # Leerzeichen bereinigen
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================
# AUDIO VERARBEITEN
# =====================================

def process_audio(wav):
    audio = np.asarray(wav, dtype=np.float32).reshape(-1)

    # Leicht höhere Stimme
    if PITCH != 1.0:
        positions = np.arange(0, len(audio), PITCH)

        audio = np.interp(
            positions,
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    # Lautstärke
    audio *= VOLUME

    # Sanftes Ein- und Ausblenden
    fade = min(
        int(SAMPLE_RATE * 0.015),
        len(audio) // 2,
    )

    if fade > 0:
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)

    return audio


# =====================================
# MEHRZEILIGEN TEXT EINLESEN
# =====================================

def read_text():
    print("\nText eingeben oder einfügen.")
    print("Zum Vorlesen ENDE in eine neue Zeile schreiben.")
    print("Mit exit beenden.\n")

    lines = []

    while True:
        line = input()

        if line.strip().upper() == "ENDE":
            break

        if line.strip().lower() in ("exit", "quit"):
            if not lines:
                return None

        lines.append(line)

    return "\n".join(lines).strip()


# =====================================
# LANGEN TEXT IN ABSCHNITTE AUFTEILEN
# =====================================

def split_text(text):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Möglichst an Satzenden trennen
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    chunks = []
    current = ""

    for sentence in sentences:

        # Sehr lange Sätze zusätzlich aufteilen
        parts = textwrap.wrap(
            sentence,
            width=MAX_CHARS,
            break_long_words=True,
            break_on_hyphens=False,
        )

        for part in parts:
            candidate = (
                f"{current} {part}".strip()
                if current else part
            )

            if len(candidate) <= MAX_CHARS:
                current = candidate
            else:
                if current:
                    chunks.append(current)

                current = part

    if current:
        chunks.append(current)

    return chunks


# =====================================
# MODELL LADEN
# =====================================

print("Lade Kohakolina...")

tts = TTS(auto_download=True)
style = tts.get_voice_style(voice_name=VOICE)

print("\nKohakolina ist bereit!")
print("Stimme:", VOICE)
print("Geschwindigkeit:", SPEED)
print("Tonhöhe:", PITCH)


# =====================================
# TEXT ZU SPRACHE
# =====================================

def speak(text):

    # Zahlen und Abkürzungen automatisch vorbereiten
    text = text_fuer_tts(text)

    chunks = split_text(text)

    if not chunks:
        return

    print(f"\nTextlänge: {len(text)} Zeichen")
    print(f"Abschnitte: {len(chunks)}\n")

    start = time.perf_counter()

    saved_audio = [] if SAVE_AUDIO else None

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"[{index}/{len(chunks)}] "
            "Generiere Sprache..."
        )

        wav, duration = tts.synthesize(
            text=chunk,
            voice_style=style,
            lang=LANGUAGE,
            speed=SPEED,
            total_steps=STEPS,
        )

        audio = process_audio(wav)

        # Aktuellen Abschnitt abspielen
        if PLAY_AUDIO:
            sd.play(audio, SAMPLE_RATE)
            sd.wait()

        # Für eine gemeinsame Audiodatei sammeln
        if SAVE_AUDIO:
            saved_audio.append(audio)

        # Kleine Pause zwischen Abschnitten
        if index < len(chunks):

            if SAVE_AUDIO:
                silence = np.zeros(
                    int(SAMPLE_RATE * PAUSE),
                    dtype=np.float32,
                )

                saved_audio.append(silence)

            if PLAY_AUDIO:
                time.sleep(PAUSE)

    # Gesamten Text als eine WAV-Datei speichern
    if SAVE_AUDIO and saved_audio:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        path = OUTPUT_DIR / (
            f"kohakolina_{timestamp}.wav"
        )

        full_audio = np.concatenate(saved_audio)

        sf.write(
            path,
            full_audio,
            SAMPLE_RATE,
        )

        print(f"\nGespeichert: {path.resolve()}")

    elapsed = time.perf_counter() - start

    print(f"\nFertig! Gesamtzeit: {elapsed:.2f}s")


# =====================================
# HAUPTSCHLEIFE
# =====================================

while True:
    try:
        text = read_text()

        if text is None:
            print("Kohakolina beendet.")
            break

        if not text:
            print("Kein Text eingegeben.")
            continue

        speak(text)

    except KeyboardInterrupt:
        sd.stop()
        print("\nAbgebrochen.")
        break

    except Exception as e:
        sd.stop()
        print(f"Fehler: {e}")
