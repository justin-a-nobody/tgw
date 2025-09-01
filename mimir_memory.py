# mimir_memory.py — Hierarchical Memory with Retrieval and Summarization
# -*- coding: utf-8 -*-

# [Insert the full mimir_memory.py source provided earlier in the canvas]
#!/usr/bin/env python3
"""
Hierarchical memory module for character/agent chat systems.

Goals:
- Avoid fragmented retrieval from raw message vectors by storing concise, human-like summaries.
- Maintain three tiers of memory:
  1) Short-term: recent raw turns (rolling window)
  2) Medium-term: summarized segments stored in a vector index
  3) Long-term: consolidated, higher-level memories and character traits/values
- Retrieval blends semantic similarity, recency bias, and emotional salience.
- Summarization and trait updates can be driven by a local LLM endpoint (same format as the user's TGUI API).
- Disk persistence using JSONL/JSON so it is portable and easy to inspect.

Dependencies:
- Standard library only for core. If you want stronger embeddings, optionally install scikit-learn for TF-IDF.
  This module implements a built-in lightweight TF-IDF with cosine similarity. If scikit-learn is present,
  it will be used automatically (but is not required).

Usage (minimal):
    from mimir_memory import MimirMemory
    mem = MimirMemory()
    # Observe a conversation turn
    mem.observe_turn(user_text, assistant_text)
    # Build retrieval context for a new prompt
    ctx = mem.build_context(query="helpful context for next reply")
    # Persist
    mem.flush()

Integration with your SMS bot:
    1) Create a single MimirMemory instance at startup.
    2) On each incoming SMS, call mem.observe_turn(incoming_text, reply_text) after you generate the reply.
    3) Before calling your LLM to generate a reply, call mem.build_context(query=incoming_text) and prepend that
       to the LLM prompt.

Environment variables (optional):
- MIMIR_DIR: base directory for memory files (default %APPDATA%/chat/mimir on Windows, ~/.cache/mimir elsewhere)
- MIMIR_RECENT_WINDOW: number of short-term turns to keep (default 12)
- MIMIR_SUMMARY_EVERY_N: summarize after this many observed turns (default 8)
- MIMIR_SUMMARY_IDLE_SECS: or if idle gap exceeds this, summarize (default 900)
- MIMIR_HALF_LIFE_DAYS: recency half-life in days (default 14)
- MIMIR_W_SIM, MIMIR_W_REC, MIMIR_W_SAL: weights for similarity/recency/salience (defaults 0.6/0.25/0.15)
- TGUI_API_URL: local LLM completions endpoint (same shape as your existing server)
- MIMIR_MAX_TOKENS: max tokens for summary calls (default 256)
- MIMIR_TEMPERATURE: temp for summary calls (default 0.3)

Files written under MIMIR_DIR:
- recent.jsonl             (short-term raw turns)
- summaries.jsonl          (medium-term summarized segments)
- longterm.json            (long-term consolidated knowledge, traits/values)
- index.json               (vector index metadata)

"""
from __future__ import annotations
import os, json, time, math, re, threading, hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

# ---------- Paths ----------

def _win_base():
    return os.environ.get("APPDATA") or os.path.expanduser("~")

DEFAULT_DIR = (
    os.path.join(_win_base(), "chat", "mimir") if os.name == "nt" else os.path.join(os.path.expanduser("~/.cache"), "mimir")
)
MIMIR_DIR = os.environ.get("MIMIR_DIR", DEFAULT_DIR)
os.makedirs(MIMIR_DIR, exist_ok=True)

RECENT_FILE = os.path.join(MIMIR_DIR, "recent.jsonl")
SUMMARIES_FILE = os.path.join(MIMIR_DIR, "summaries.jsonl")
LONGTERM_FILE = os.path.join(MIMIR_DIR, "longterm.json")
INDEX_FILE = os.path.join(MIMIR_DIR, "index.json")

# ---------- Config ----------

RECENT_WINDOW = int(os.environ.get("MIMIR_RECENT_WINDOW", "12"))
SUMMARY_EVERY_N = int(os.environ.get("MIMIR_SUMMARY_EVERY_N", "8"))
SUMMARY_IDLE_SECS = int(os.environ.get("MIMIR_SUMMARY_IDLE_SECS", "900"))
HALF_LIFE_DAYS = float(os.environ.get("MIMIR_HALF_LIFE_DAYS", "14"))

W_SIM = float(os.environ.get("MIMIR_W_SIM", "0.6"))
W_REC = float(os.environ.get("MIMIR_W_REC", "0.25"))
W_SAL = float(os.environ.get("MIMIR_W_SAL", "0.15"))

TGUI_API_URL = os.environ.get("TGUI_API_URL", "http://127.0.0.1:5000/v1/completions")
MIMIR_MAX_TOKENS = int(os.environ.get("MIMIR_MAX_TOKENS", "256"))
MIMIR_TEMPERATURE = float(os.environ.get("MIMIR_TEMPERATURE", "0.3"))

# ---------- Utility ----------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _age_days(ts_iso: str) -> float:
    try:
        dt = datetime.strptime(ts_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    except Exception:
        return 0.0

def _recency_weight(ts_iso: str, half_life_days: float = HALF_LIFE_DAYS) -> float:
    # Exponential decay with half-life
    age = _age_days(ts_iso)
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (age / half_life_days)

# ---------- Data classes ----------

@dataclass
class Turn:
    ts: str
    role: str  # 'user' or 'assistant'
    text: str
    meta: Optional[Dict[str, Any]] = None

@dataclass
class Summary:
    id: str
    ts: str
    text: str
    src_range: Tuple[int, int]  # indices into recent.jsonl sequence snapshot at time of summary
    salience: float  # 0..1
    tags: List[str]

@dataclass
class LongTerm:
    ts: str
    knowledge: List[str]  # consolidated memories (bullets)
    traits: Dict[str, Any]  # evolving profile: values, preferences, goals

# ---------- Storage ----------

class JSONLStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            open(path, "a", encoding="utf-8").close()
        self._lock = threading.Lock()

    def append(self, obj: Dict[str, Any]):
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        with self._lock, open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

class JSONStore:
    def __init__(self, path: str, default_obj: Dict[str, Any]):
        self.path = path
        self.default = default_obj
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self.write(default_obj)
        self._lock = threading.Lock()

    def read(self) -> Dict[str, Any]:
        with self._lock:
            try:
                return json.load(open(self.path, "r", encoding="utf-8"))
            except Exception:
                return json.loads(json.dumps(self.default))

    def write(self, obj: Dict[str, Any]):
        with self._lock, open(self.path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------- Lightweight TF-IDF Vector Index ----------

try:
    # optional speedup if scikit-learn exists
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SKLEARN_OK = True
except Exception:
    import math as _m
    SKLEARN_OK = False

class LocalVectorIndex:
    """A simple vector index over summary texts.
    If scikit-learn is available, uses TfidfVectorizer; otherwise uses a tiny TF-IDF implementation.
    """
    def __init__(self, index_path: str = INDEX_FILE):
        self.index_path = index_path
        self.docs: List[Dict[str, Any]] = []  # each has: id, text, ts, salience
        self._tfidf = None
        self._matrix = None
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            try:
                data = json.load(open(self.index_path, "r", encoding="utf-8"))
                self.docs = data.get("docs", [])
            except Exception:
                self.docs = []
        else:
            self._persist()
        self._rebuild()

    def _persist(self):
        tmp = {"docs": self.docs}
        json.dump(tmp, open(self.index_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def _rebuild(self):
        texts = [d["text"] for d in self.docs]
        if SKLEARN_OK:
            self._tfidf = TfidfVectorizer(max_features=8192, ngram_range=(1,2))
            try:
                self._matrix = self._tfidf.fit_transform(texts) if texts else None
            except Exception:
                self._matrix = None
        else:
            # minimal: build df
            self._vocab = {}
            df = {}
            for t in texts:
                seen = set()
                for tok in self._tokenize(t):
                    if tok not in seen:
                        df[tok] = df.get(tok, 0) + 1
                        seen.add(tok)
            self._df = df
            self._N = max(1, len(texts))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_#@]+", (text or "").lower())

    def _tfidf_vec(self, text: str) -> Dict[str, float]:
        # Only used in no-sklearn branch
        tf = {}
        toks = self._tokenize(text)
        for tok in toks:
            tf[tok] = tf.get(tok, 0) + 1
        if not toks:
            return {}
        max_tf = max(tf.values())
        vec = {}
        for tok, f in tf.items():
            df = self._df.get(tok, 0)
            idf = math.log((self._N + 1) / (df + 1)) + 1.0
            vec[tok] = (0.5 + 0.5 * (f / max_tf)) * idf
        return vec

    @staticmethod
    def _cosine_sparse(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0
        # dot
        keys = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[k] * v2[k] for k in keys)
        n1 = math.sqrt(sum(x*x for x in v1.values()))
        n2 = math.sqrt(sum(x*x for x in v2.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def add(self, sid: str, text: str, ts: str, salience: float):
        with self._lock:
            self.docs = [d for d in self.docs if d["id"] != sid]
            self.docs.append({"id": sid, "text": text, "ts": ts, "salience": float(max(0.0, min(1.0, salience)))})
            self._persist()
            self._rebuild()

    def search(self, query: str, top_k: int = 6) -> List[Tuple[float, Dict[str, Any]]]:
        if not self.docs:
            return []
        if SKLEARN_OK and self._matrix is not None:
            import numpy as np
            qv = self._tfidf.transform([query])
            sims = (qv @ self._matrix.T).toarray().ravel()
            scored = []
            for i, d in enumerate(self.docs):
                s_sim = float(sims[i])
                s_rec = _recency_weight(d.get("ts", _now_iso()))
                s_sal = float(d.get("salience", 0.0))
                score = W_SIM * s_sim + W_REC * s_rec + W_SAL * s_sal
                scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:top_k]
        else:
            # naive branch
            qv = self._tfidf_vec(query)
            scored = []
            for d in self.docs:
                dv = self._tfidf_vec(d["text"]) if hasattr(self, "_df") else {}
                s_sim = self._cosine_sparse(qv, dv)
                s_rec = _recency_weight(d.get("ts", _now_iso()))
                s_sal = float(d.get("salience", 0.0))
                score = W_SIM * s_sim + W_REC * s_rec + W_SAL * s_sal
                scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:top_k]

# ---------- LLM Helpers ----------

import requests

class Summarizer:
    def __init__(self, api_url: str = TGUI_API_URL, max_tokens: int = MIMIR_MAX_TOKENS, temperature: float = MIMIR_TEMPERATURE):
        self.api_url = api_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": stop or ["</s>"],
            "echo": False,
        }
        try:
            r = requests.post(self.api_url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0].get("text", "").strip()
        except Exception as e:
            print(f"[WARN] LLM summarize error: {e}")
            return ""

    def summarize_segment(self, turns: List[Turn]) -> Tuple[str, float, List[str]]:
        # Produce a tight, factual summary and an estimated salience (0..1) and tags
        raw = "\n".join([f"{t.role.upper()}: {t.text}" for t in turns])
        prompt = (
            "You condense chat logs into concise memory cards.\n"
            "Rules: Write 3-6 bullet points. Keep names, facts, preferences, decisions.\n"
            "Ignore pleasantries and chit-chat.\n"
            "Also output an approximate emotional salience score 0..1 (0 neutral, 1 intense),\n"
            "and 3-6 comma-separated tags.\n\n"
            f"CHAT:\n{raw}\n\n"
            "Return as JSON with keys: bullets (list of strings), salience (number), tags (list of strings)."
        )
        text = self._call(prompt)
        bullets, salience, tags = [], 0.2, []
        try:
            j = json.loads(text)
            bullets = [b.strip() for b in j.get("bullets", []) if b.strip()]
            salience = float(j.get("salience", 0.2))
            tags = [t.strip() for t in j.get("tags", []) if t.strip()]
        except Exception:
            # fallback: naive bulletization
            bullets = _fallback_bullets(raw)
            salience = _heuristic_salience(raw)
            tags = _heuristic_tags(raw)
        return ("\n".join(f"- {b}" for b in bullets[:8]), max(0.0, min(1.0, salience)), tags[:8])

    def update_traits(self, existing_traits: Dict[str, Any], turns: List[Turn]) -> Dict[str, Any]:
        raw = "\n".join([f"{t.role.upper()}: {t.text}" for t in turns])
        prompt = (
            "You maintain a character sheet. Given chat excerpts and current traits, update persistent traits/values.\n"
            "Traits JSON contains keys such as values, preferences, goals, dislikes, likes, relationships.\n"
            "Preserve what is not contradicted. Merge new info. Keep it terse.\n\n"
            f"CURRENT_TRAITS = {json.dumps(existing_traits, ensure_ascii=False)}\n\n"
            f"CHAT = {raw}\n\n"
            "Return only JSON for updated traits."
        )
        text = self._call(prompt)
        try:
            j = json.loads(text)
            return j
        except Exception:
            return existing_traits

# ---------- Heuristics ----------

def _fallback_bullets(raw: str) -> List[str]:
    # Very crude extraction: sentences with colon/decision/likes
    sents = re.split(r"(?<=[.!?])\s+", raw)
    picks = []
    for s in sents:
        if any(k in s.lower() for k in [":", " prefer", " like", " plan", " decided", " will "]):
            picks.append(s.strip())
    if not picks:
        picks = sents[:4]
    return [re.sub(r"^USER:|^ASSISTANT:", "", p).strip() for p in picks]

def _heuristic_salience(raw: str) -> float:
    # Simple features: exclamation, all-caps, keyword cues
    ex = raw.count("!")
    caps = sum(1 for w in re.findall(r"\b[A-Z]{3,}\b", raw) if w not in ("USER", "ASSISTANT"))
    words = raw.lower()
    cues = sum(1 for k in ["urgent", "love", "hate", "angry", "excited", "worried", "great", "terrible"] if k in words)
    score = 1 - math.exp(-(ex * 0.5 + caps * 0.4 + cues * 0.8))
    return max(0.0, min(1.0, score))

def _heuristic_tags(raw: str) -> List[str]:
    # Pull hashtags and simple nouns
    tags = set(re.findall(r"#[A-Za-z0-9_]+", raw))
    for k in ["pricing", "cards", "football", "mlb", "family", "work", "plan", "goal", "food"]:
        if k in raw.lower():
            tags.add(k)
    return list(tags) or ["general"]

# ---------- MimirMemory ----------

class MimirMemory:
    def __init__(self):
        self.recent = JSONLStore(RECENT_FILE)
        self.summaries = JSONLStore(SUMMARIES_FILE)
        self.longterm = JSONStore(LONGTERM_FILE, {"ts": _now_iso(), "knowledge": [], "traits": {}})
        self.index = LocalVectorIndex(INDEX_FILE)
        self.summarizer = Summarizer()
        # state for segmenting
        self._turn_count = 0
        self._last_turn_ts = time.time()

    # ----- observe -----
    def observe_turn(self, user_text: str, assistant_text: str):
        t1 = Turn(ts=_now_iso(), role="user", text=user_text)
        t2 = Turn(ts=_now_iso(), role="assistant", text=assistant_text)
        self.recent.append(asdict(t1))
        self.recent.append(asdict(t2))
        self._turn_count += 2
        now = time.time()
        idle = now - self._last_turn_ts
        self._last_turn_ts = now
        # Summarize if threshold reached
        if self._turn_count >= SUMMARY_EVERY_N or idle >= SUMMARY_IDLE_SECS:
            self._summarize_recent()
            self._turn_count = 0

    # ----- summarize -----
    def _summarize_recent(self):
        turns = [Turn(**o) for o in self.recent.read_all()]
        if not turns:
            return
        # Segment: last N*2 turns or since last summary marker
        # For simplicity, summarize the last min(2*SUMMARY_EVERY_N, len(turns)) turns
        seg_sz = min(2 * SUMMARY_EVERY_N, len(turns))
        segment = turns[-seg_sz:]
        text, salience, tags = self.summarizer.summarize_segment(segment)
        sid = hashlib.sha1((segment[0].ts + segment[-1].ts + text).encode()).hexdigest()[:16]
        rec = Summary(id=sid, ts=_now_iso(), text=text, src_range=(len(turns) - seg_sz, len(turns) - 1), salience=salience, tags=tags)
        self.summaries.append(asdict(rec))
        self.index.add(sid, text, rec.ts, salience)
        # Update longterm traits occasionally
        if salience >= 0.4:
            self._update_traits_from(segment)
        # Consolidate knowledge every few summaries
        if len(self._read_summaries()) % 5 == 0:
            self._consolidate_longterm()

    def _read_summaries(self) -> List[Summary]:
        return [Summary(**o) for o in self.summaries.read_all()]

    def _update_traits_from(self, turns: List[Turn]):
        lt = self.longterm.read()
        traits = lt.get("traits", {})
        new_traits = self.summarizer.update_traits(traits, turns)
        lt["traits"] = new_traits
        lt["ts"] = _now_iso()
        self.longterm.write(lt)

    # ----- consolidate -----
    def _consolidate_longterm(self):
        """Merge similar medium-term memories into durable knowledge bullets."""
        lt = self.longterm.read()
        knowledge: List[str] = lt.get("knowledge", [])
        summaries = self._read_summaries()[-20:]  # recent window for consolidation
        # Pick top salient items, then merge similar ones
        salient = sorted(summaries, key=lambda s: s.salience, reverse=True)[:10]
        merged: List[str] = []
        for s in salient:
            added = False
            for i, k in enumerate(merged):
                if _roughly_similar(s.text, k):
                    merged[i] = _merge_bullets(k, s.text)
                    added = True
                    break
            if not added:
                merged.append(s.text)
        # Flatten bullets and dedupe
        flat = _dedupe_bullets("\n".join(merged))
        # Keep size bounded
        knowledge = _dedupe_bullets("\n".join(knowledge + flat)).splitlines()[:200]
        lt["knowledge"] = knowledge
        lt["ts"] = _now_iso()
        self.longterm.write(lt)

    # ----- retrieve -----
    def build_context(self, query: Optional[str] = None, k_recent: int = 8, k_medium: int = 6, k_long: int = 6) -> str:
        turns = [Turn(**o) for o in self.recent.read_all()]
        recent_block = _format_recent(turns[-k_recent*2:]) if turns else ""
        medium_block = ""
        if query:
            hits = self.index.search(query, top_k=k_medium)
            if hits:
                medium_block = "\n".join([f"# Memory: score={score:.3f}\n{d['text']}" for score, d in hits])
        lt = self.longterm.read()
        long_block = "\n".join(f"- {b}" for b in lt.get("knowledge", [])[:k_long])
        traits_block = json.dumps(lt.get("traits", {}), ensure_ascii=False)
        out = []
        if traits_block.strip() != "{}":
            out.append("[TRAITS] " + traits_block)
        if long_block.strip():
            out.append("[LONG-TERM]\n" + long_block)
        if medium_block.strip():
            out.append("[MEDIUM-TERM]\n" + medium_block)
        if recent_block.strip():
            out.append("[RECENT]\n" + recent_block)
        return "\n\n".join(out).strip()

    # ----- maintenance -----
    def flush(self):
        # All writes are synchronous; provided for API symmetry
        pass

# ---------- helpers for consolidate/format ----------

def _roughly_similar(a: str, b: str) -> bool:
    """Cheap string similarity to decide merge."
    a2 = re.sub(r"\W+", " ", a.lower()).split()
    b2 = re.sub(r"\W+", " ", b.lower()).split()
    if not a2 or not b2:
        return False
    overlap = len(set(a2) & set(b2))
    return overlap >= max(3, min(len(a2), len(b2)) // 3)

def _merge_bullets(a: str, b: str) -> str:
    return "\n".join(_dedupe_bullets(a + "\n" + b))

def _dedupe_bullets(text: str) -> List[str]:
    seen = set()
    out = []
    for line in text.splitlines():
        t = line.strip().lstrip("- ")
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append("- " + t)
    return out

def _format_recent(turns: List[Turn]) -> str:
    return "\n".join([f"{t.role.upper()}: {t.text}" for t in turns])

# ---------- CLI demo ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mimir hierarchical memory demo")
    ap.add_argument("--query", type=str, default="", help="Query to retrieve context for")
    ap.add_argument("--observe", type=str, nargs=2, metavar=("USER", "ASSISTANT"), help="Observe a single turn pair", default=None)
    ap.add_argument("--summarize-now", action="store_true", help="Force a summary of recent turns")
    args = ap.parse_args()

    mem = MimirMemory()
    if args.observe:
        u, a = args.observe
        mem.observe_turn(u, a)
        print("Observed.")
    if args.summarize_now:
        mem._summarize_recent()
        print("Summarized recent.")
    ctx = mem.build_context(query=args.query)
    if ctx:
        print("\n=== RETRIEVED CONTEXT ===\n")
        print(ctx)
    else:
        print("No context yet. Try observing some turns.")
