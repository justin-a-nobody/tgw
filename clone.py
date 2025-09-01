#!/usr/bin/env python3
# winsammi.py — Windows SMS bot using SIM7600 (AT commands), privacy-safe (no personal info learning)
# -*- coding: utf-8 -*-

import os, re, json, time, shutil, requests, sys
import serial
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple, Set

# =========================================================
# Bootstrapping & Config
# =========================================================

def ensure_dirs():
    # Create all necessary folders on first run
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(MEM_DIR, exist_ok=True)
    os.makedirs(PERSONA_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "user_data", "characters"), exist_ok=True)
    os.makedirs(ME_ARCHIVE_DIR, exist_ok=True)
    os.makedirs(ME_EXPORT_DIR, exist_ok=True)
    # Character YAML auto-create if missing
    if not os.path.exists(CHARACTER_PATH):
        default_yaml = f"""name: {CHARACTER_NAME}
context: ""
greeting: ""
"""
        with open(CHARACTER_PATH, "w", encoding="utf-8") as f:
            f.write(default_yaml)

# Load .env first to compute BASE_DIR etc.
load_dotenv()

# Serial modem (SIM7600) config
MODEM_AT_PORT = os.getenv("MODEM_AT_PORT", "COM4")
MODEM_BAUD = int(os.getenv("MODEM_BAUD", "115200"))
SER_TIMEOUT = float(os.getenv("SER_TIMEOUT", "3"))

# Polling
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

# Persona / files
CHARACTER_NAME = os.getenv("CHARACTER_NAME", "Character")
BASE_DIR = os.getenv("BASE_DIR", os.path.join(os.path.expanduser("~"), "chat"))
CHARACTER_PATH = os.getenv("CHARACTER_PATH", os.path.join(BASE_DIR, f"user_data/characters/{CHARACTER_NAME}.yaml"))

# LLM endpoint (Oobabooga OpenAI-compatible completions)
TGUI_API_URL = os.getenv("TGUI_API_URL", "http://127.0.0.1:5000/v1/completions")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "55"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Logging & per-number rolling history (no personal facts)
LOG_FILE = os.getenv("LOG_FILE", os.path.join(BASE_DIR, "chat_log.txt"))
MEM_DIR = os.getenv("MEM_DIR", os.path.join(BASE_DIR, "sms_conversations"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "25"))
INACTIVITY_RESET = int(os.getenv("INACTIVITY_RESET", "600"))  # seconds

# Persona dir (kept for compatibility; we won’t store personal info)
PERSONA_DIR = os.getenv("PERSONA_DIR", os.path.join(BASE_DIR, "persona"))
PROFILE_PATH = os.path.join(PERSONA_DIR, "profile.json")  # unused for personal data
ME_NOTES_FILE = os.path.join(PERSONA_DIR, "notes_me.jsonl")  # not written in privacy mode
ME_ARCHIVE_DIR = os.path.join(PERSONA_DIR, "archive")
ME_EXPORT_DIR = os.path.join(PERSONA_DIR, "weekly_exports")
ME_LAST_EXPORT = os.path.join(ME_EXPORT_DIR, "last_export.txt")

# Compaction/export cadence
COMPACT_INTERVAL = int(os.getenv("COMPACT_INTERVAL", str(60 * 60)))
_last_compact_ts = 0

# Training / “my number” (used only to detect your own test messages)
TRAIN_NUMBER = os.getenv("TRAIN_NUMBER", "+10000000000")

# Text limits
SANITIZE_LOWER_NO_PUNCT = os.getenv("SANITIZE_LOWER_NO_PUNCT", "false").lower() in ("1","true","yes")
MAX_SMS_CHARS = int(os.getenv("MAX_SMS_CHARS", "222"))

# Privacy mode — blocks all personal info learning by default
PRIVACY_MODE = os.getenv("PRIVACY_MODE", "1").lower() in ("1","true","yes")

# Prepare folders now that paths are known
ensure_dirs()

# =========================================================
# Utilities
# =========================================================
def split_sms(text: str, chunk_size: int = 153) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _mem_path(number: str) -> str:
    safe = re.sub(r"[^\dA-Za-z]+", "_", number or "")
    return os.path.join(MEM_DIR, f"{safe}.jsonl")

def append_memory(number: str, role: str, text: str):
    os.makedirs(MEM_DIR, exist_ok=True)
    entry = {"ts": _now_iso(), "role": role, "text": text}
    with open(_mem_path(number), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_recent_history(number: str) -> List[Dict]:
    path = _mem_path(number)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
    except Exception:
        return []
    if lines:
        try:
            last_ts = lines[-1]["ts"]
            last_dt = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last_dt).total_seconds() > INACTIVITY_RESET:
                return []
        except Exception:
            pass
    return lines[-HISTORY_LIMIT:]

def log_message(sender: str, text: str, reply: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] FROM: {sender}\nTEXT: {text}\nREPLY: {reply}\n\n")

def _norm_number(n: str) -> str:
    return re.sub(r"\D", "", n or "")[-10:]

# =========================================================
# Character prompt
# =========================================================
def load_character_prompt(char_path: str) -> Tuple[str, str]:
    try:
        import yaml
        with open(char_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        name = data.get("name", CHARACTER_NAME)
        context = data.get("context", "")
        greeting = data.get("greeting", "")
        style = (
            f"{name} texts like a real person: natural, brief, friendly.\n"
            f"- React to the user's topic first.\n"
            f"- At most ONE short, casual question; weave it in naturally.\n"
            f"- No lists of questions. No invitations. No fandom talk.\n"
            f"- If user is annoyed, ask ZERO questions and switch topics.\n"
            f"- Keep all texts under {MAX_SMS_CHARS} chars.\n"
            f"- Never break character or say you're a bot.\n"
            f"- Never repeat a question that was ignored or declined."
        )
        prompt = f"{context}\n{style}\n{name}: {greeting}\n"
        return prompt, name
    except Exception as e:
        print(f"[WARN] Failed to load character YAML: {e}")
        return "", CHARACTER_NAME

character_prompt, character_name = load_character_prompt(CHARACTER_PATH)

# =========================================================
# STOP & spam
# =========================================================
def should_force_stop(text: str) -> bool:
    t = (text or "").lower()
    yes_patterns = [
        r'\breply\s+yes\b', r'\btext\s+yes\b', r'\bsend\s+yes\b',
        r'\brespond\s+yes\b', r'\bresponse\s+yes\b',
        r'\bconfirm\s+with\s+yes\b', r'\breply\s+with\s+yes\b',
        r'\bwould you like to keep .*?\breply\s+yes\b',
    ]
    stop_option_patterns = [
        r'\breply\s+stop\b', r'\btext\s+stop\b', r'\bsend\s+stop\b',
        r'\bunsubscribe\b', r'\bopt[-\s]*out\b', r'\bstop to\b',
        r'\bto (end|cancel|stop).*?\bstop\b',
    ]
    patt = '(' + '|'.join(yes_patterns + stop_option_patterns) + ')'
    return re.search(patt, t) is not None

def canonicalize_stop(text: str) -> str:
    if re.search(r'\bstop\b', (text or "").strip().lower()):
        return "STOP"
    return (text or "").strip()

def is_spam(text: str) -> bool:
    t = (text or "").lower()
    for kw in ["win money","free prize","click here","limited offer","congratulations","lottery","loan","promo","unsubscribe"]:
        if kw in t: return True
    return False

# =========================================================
# Reply generation (Oobabooga /v1/completions)
# =========================================================
def sanitize_reply_text(raw: str) -> str:
    if SANITIZE_LOWER_NO_PUNCT:
        tbl = str.maketrans('', '', ''.join([c for c in map(chr, range(0x20,0x7F)) if not c.isalnum() and not c.isspace()]))
        return (raw or "").translate(tbl).lower().strip() or "ok"
    return (raw or "").strip() or "ok"

BANNED_INVITES = re.compile(r"\b(grab a beer|meet up|come over|hang out|let'?s meet)\b", re.I)
BANNED_FANDOM  = re.compile(r"\b(i('| a)m|i am)\s+(a\s+)?(fan|supporter)\b.*", re.I)

def _postprocess_reply(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = BANNED_INVITES.sub("", t).strip()
    t = BANNED_FANDOM.sub("", t).strip()
    if t.count("?") > 1:
        first = t.split("?")[0].strip()
        t = (first + "?") if first else t
    limit = max(1, int(os.getenv("MAX_SMS_CHARS", str(MAX_SMS_CHARS))))
    if len(t) > limit:
        t = t[:max(0, limit - 3)].rstrip() + "..."
    return t or "ok"

def build_prompt_with_history(history: List[Dict], user_text: str, extra_ctx: str = "") -> str:
    seq = [h for h in (history or []) if h.get("role") in ("user", "assistant")]
    if seq and seq[0]["role"] != "user":
        seq = seq[1:]
    fixed = []
    last = None
    for h in seq:
        if h["role"] == last:
            continue
        fixed.append(h); last = h["role"]

    blocks = []
    for h in fixed:
        if h["role"] == "user":
            blocks.append(f"[INST] {h.get('text','')} [/INST]")
        else:
            blocks.append(f"{h.get('text','')}</s>")

    style = (
        "You are texting like a real person: natural, brief, friendly. "
        "React to their topic first. Ask at most ONE short, casual question, "
        "woven into the reply. No lists of questions, no invites, no fandom talk. "
        "If the user seems annoyed, ask ZERO questions and switch topics. "
        "Never repeat a question that was ignored or declined."
    )
    injected_user = ((extra_ctx.strip() + "\n") if extra_ctx else "") + style + "\n" + (user_text or "")
    blocks.append(f"[INST] {injected_user.strip()} [/INST]")
    return "".join(blocks)

def generate_reply(user_input: str, history: Optional[List[Dict]] = None, extra_ctx: str = "") -> str:
    payload = {
        "prompt": build_prompt_with_history(history or [], user_input, extra_ctx=character_prompt),
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stop": ["</s>"],
        "echo": False,
    }
    try:
        r = requests.post(TGUI_API_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        raw = data["choices"][0].get("text", "")
        return sanitize_reply_text(_postprocess_reply(raw))
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "sorry im having trouble thinking right now"

# =========================================================
# SIM7600 AT helpers
# =========================================================
class Modem:
    def __init__(self, port: str, baud: int, timeout: float):
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
            print(f"[MODEM] Opened modem on {port} @ {baud}")
        except Exception as e:
            print(f"[MODEM ERROR] Could not open {port}: {e}")
            sys.exit(1)
        self.init_text_mode()

    def at(self, cmd: str, delay: float = 0.5) -> str:
        if not cmd.endswith("\r"):
            cmd += "\r"
        self.ser.write(cmd.encode("utf-8", errors="ignore"))
        time.sleep(delay)
        out = self.ser.read_all().decode(errors="ignore")
        return out

    def init_text_mode(self):
        self.at("AT")
        self.at("ATE0")
        self.at("AT+CMGF=1")           # text mode
        self.at('AT+CSCS="GSM"')       # character set
        self.at('AT+CNMI=2,1,0,0,0')   # new message indications to TE
        print("[MODEM] Initialized text mode")

    def list_unread_sms_ids(self) -> List[int]:
        out = self.at('AT+CMGL="REC UNREAD"', delay=0.8)
        return [int(x) for x in re.findall(r"\+CMGL:\s*(\d+)", out)]

    def read_sms(self, idx: int) -> Tuple[Optional[str], Optional[str]]:
        out = self.at(f"AT+CMGR={idx}", delay=0.8)
        # Typical: +CMGR: "REC UNREAD","+15551234567","","24/08/31,11:33:12-20"\r\nBody\r\n
        m = re.search(r'\+CMGR:\s*".*?",\s*"([^"]+)"[^\n]*\n(.*)\r', out, flags=re.S)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # fallback: try to find a number and first text line
        n = re.search(r'([+]\d{8,15})', out)
        b = re.search(r'\n([^\r]+)\r', out)
        return (n.group(1).strip() if n else None, b.group(1).strip() if b else None)

    def delete_sms(self, idx: int):
        self.at(f"AT+CMGD={idx}", delay=0.3)

    def send_sms_text(self, number: str, text: str):
        for i, chunk in enumerate(split_sms(text, 153), start=1):
            self.at("AT+CMGF=1")
            self.ser.write(f'AT+CMGS="{number}"\r'.encode("utf-8"))
            time.sleep(0.5)
            self.ser.write(chunk.encode("utf-8", errors="ignore") + b"\x1A")
            time.sleep(3.0)
            print(f"[SENT] Part {i} to {number}: {chunk[:60]}{'...' if len(chunk)>60 else ''}")

# =========================================================
# Maintenance (log compaction / weekly export)
# =========================================================
def compact_persona_if_needed(force: bool=False):
    global _last_compact_ts
    now=time.time()
    if not force and (now-_last_compact_ts)<COMPACT_INTERVAL:
        return
    if os.path.exists(ME_NOTES_FILE):
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        try:
            shutil.copy2(ME_NOTES_FILE, os.path.join(ME_ARCHIVE_DIR, f"notes_me_{ts}.jsonl"))
        except Exception:
            pass
    # In privacy mode we do not rewrite notes; just keep archive step
    _last_compact_ts=now
    print("[CLEANUP] compacted")

def export_weekly_conversations():
    today = datetime.now(timezone.utc)
    if today.weekday()!=6:
        return
    today_str=today.strftime("%Y-%m-%d")
    if os.path.exists(ME_LAST_EXPORT):
        try:
            if open(ME_LAST_EXPORT,"r",encoding="utf-8").read().strip()==today_str:
                return
        except Exception:
            pass

    export={"exported_at":_now_iso(),"log_file":LOG_FILE,"conversations":{}}
    for fname in os.listdir(MEM_DIR):
        if not fname.endswith(".jsonl"): continue
        path=os.path.join(MEM_DIR,fname)
        conv=[]
        try:
            with open(path,"r",encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try: conv.append(json.loads(line))
                    except Exception: pass
        except Exception:
            pass
        export["conversations"][fname[:-6]]=conv

    chats=[]
    try:
        if os.path.exists(LOG_FILE):
            raw=open(LOG_FILE,"r",encoding="utf-8").read()
            blocks=raw.strip().split("\n\n")
            for b in blocks:
                ts_m=re.search(r"^\[(.*?)\]\s*FROM:\s*(.+)$",b,re.M)
                txt_m=re.search(r"TEXT:\s*(.+)$",b,re.M)
                rep_m=re.search(r"REPLY:\s*(.+)$",b,re.M)
                if ts_m and txt_m and rep_m:
                    chats.append({"ts":ts_m.group(1),"from":ts_m.group(2).strip(),
                                  "text":txt_m.group(1).strip(),"reply":rep_m.group(1).strip()})
    except Exception:
        pass
    export["chat_log"]=chats

    out_path=os.path.join(ME_EXPORT_DIR,f"{today_str}_conversations.json")
    json.dump(export, open(out_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    open(ME_LAST_EXPORT,"w",encoding="utf-8").write(today_str)
    print(f"[EXPORT] Weekly export: {out_path}")

# =========================================================
# SMS send wrapper
# =========================================================
def send_sms_in_chunks(modem: "Modem", number: str, full_text: str):
    for chunk in split_sms(full_text):
        safe_text = chunk.replace("\n"," ").replace("\r"," ")
        modem.send_sms_text(number, safe_text)

# =========================================================
# Main loop
# =========================================================
def poll_and_reply(modem: "Modem"):
    seen: Set[int] = set()
    print(f"[BOT] SMS->LLM bot ({CHARACTER_NAME}) starting. Poll interval: {POLL_INTERVAL}s, port: {MODEM_AT_PORT}")

    while True:
        try: compact_persona_if_needed()
        except Exception as e: print(f"[WARN] compact error: {e}")

        try: export_weekly_conversations()
        except Exception as e: print(f"[WARN] weekly_export error: {e}")

        try:
            ids = modem.list_unread_sms_ids()
        except Exception as e:
            print(f"[ERR] listing SMS: {e}")
            ids = []

        for sms_id in ids:
            if sms_id in seen:
                continue

            try:
                sender, body = modem.read_sms(sms_id)
            except Exception as e:
                print(f"[ERR] reading SMS {sms_id}: {e}")
                modem.delete_sms(sms_id)
                continue

            seen.add(sms_id)

            if not sender or not body:
                modem.delete_sms(sms_id)
                continue

            print(f"[IN] From {sender}: {body}")

            history = load_recent_history(sender)
            is_me = (_norm_number(sender) == _norm_number(TRAIN_NUMBER))

            extra_ctx = ""  # privacy: no personal context
            if (not is_me) and should_force_stop(body):
                print("[POLICY] YES/STOP match. Forcing STOP.")
                reply = "STOP"
            elif (not is_me) and is_spam(body):
                print("[SPAM] Detected spam-like content.")
                reply = "STOP"
            else:
                reply = generate_reply(body, history, extra_ctx=extra_ctx)

            reply = canonicalize_stop(reply)
            print(f"[OUT] To {sender}: {reply}")

            try:
                send_sms_in_chunks(modem, sender, reply)
            except Exception as e:
                print(f"[ERR] sending SMS: {e}")

            try:
                log_message(sender, body, reply)
            except Exception as e:
                print(f"[WARN] logging failed: {e}")

            append_memory(sender, "user", body)
            append_memory(sender, "assistant", reply)

            try:
                modem.delete_sms(sms_id)
            except Exception as e:
                print(f"[WARN] delete SMS {sms_id} failed: {e}")

        time.sleep(POLL_INTERVAL)

# =========================================================
# Reset / Purge helpers (for redistribution)
# =========================================================
def purge_all_personal():
    # Remove all existing logs and per-number chats to ship a clean copy
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    except Exception:
        pass
    try:
        if os.path.isdir(MEM_DIR):
            for fn in os.listdir(MEM_DIR):
                if fn.endswith(".jsonl"):
                    try: os.remove(os.path.join(MEM_DIR, fn))
                    except Exception: pass
    except Exception:
        pass
    # Do not delete character YAML; it is generic

# =========================================================
# Entrypoint
# =========================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Windows SIM7600 SMS bot (privacy-safe)")
    ap.add_argument("--reset", action="store_true", help="Delete logs and chat histories, keep character file")
    ap.add_argument("--port", type=str, default=MODEM_AT_PORT, help="Override COM port (e.g., COM4)")
    args = ap.parse_args()

    if args.reset:
        purge_all_personal()
        print("[RESET] Cleared logs and per-number histories.")

    try:
        _last_compact_ts = 0
        time.sleep(1)
        compact_persona_if_needed(force=True)
    except Exception as e:
        print(f"[WARN] initial compact error: {e}")

    try:
        m = Modem(args.port, MODEM_BAUD, SER_TIMEOUT)
        poll_and_reply(m)
    except KeyboardInterrupt:
        print("\nExiting on Ctrl-C.")
