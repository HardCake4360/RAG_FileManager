#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server_full.py
Flask-based RAG file search server with:
- Text/PDF indexing (SentenceTransformer + FAISS)
- Incremental /index/scan with directory diff
- Per-file metadata cache (files_db) with 1-line description
- LLM backed one-liner via Ollama (fallback to rule-based)
- Persistent save/load of index/meta/cache/files_db
"""

import os
import io
import json
import time
import socket
import urllib.request
from typing import Dict, Any, Iterable, List, Tuple

import faiss
import orjson
import chardet
import pdfplumber
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from PIL import Image
import hashlib
from flask import send_file

# --- image caption (soft import) ---
try:
    from transformers import pipeline
    _captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    IMAGE_CAPTION_AVAILABLE = True
except Exception:
    _captioner = None
    IMAGE_CAPTION_AVAILABLE = False

# 이미지 캡션을 텍스트 인덱스에도 넣을지 여부 (기본 True)
IMAGE_ADD_CAPTION_TO_TEXT = True


# --------------------------
# App / Model / Index setup
# --------------------------
app = Flask(__name__)

MODEL_NAME = "BAAI/bge-m3"   # 1024-d multilingual embedding model
EMBED_DIM = 1024
NORMALIZE = True

# Load embedding model once at startup
text_model = SentenceTransformer(MODEL_NAME)

# In-memory FAISS index + metadata
faiss_index = faiss.IndexFlatIP(EMBED_DIM)  # cosine ~ inner product with normalized embeddings
meta: List[Dict[str, Any]] = []             # per-chunk metadata (id -> chunk info)
id_counter: int = 0

# --- 이미지 의미 임베딩용 모델 추가 ---
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}

IMG_INDEX_DIR = os.path.join("indexes", "image")
os.makedirs(IMG_INDEX_DIR, exist_ok=True)
IMG_INDEX_PATH = os.path.join(IMG_INDEX_DIR, "faiss.index")
IMG_META_PATH  = os.path.join(IMG_INDEX_DIR, "meta.jsonl")

#썸네일 생성용
THUMB_DIR = os.path.join("indexes", "thumbs")
os.makedirs(THUMB_DIR, exist_ok=True)


# CLIP 계열 멀티모달 모델 (텍스트/이미지 공용 임베딩)
IMG_MODEL_NAME = "clip-ViT-L-14"  # 또는 "clip-ViT-B-32"
img_model = SentenceTransformer(IMG_MODEL_NAME)
D_IMG = None                 # 이미지 임베딩 차원은 처음 벡터 뽑을 때 결정
faiss_index_img = None       # 아직 인덱스 만들지 않음
meta_img: List[Dict[str, Any]] = []


# --------------------------
# Paths for persistence
# --------------------------
INDEX_DIR = os.path.join("indexes", "text")
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.jsonl")
CACHE_PATH = os.path.join(INDEX_DIR, "cache.json")    # path -> fingerprint
FILES_PATH = os.path.join(INDEX_DIR, "files.json")    # path -> per-file dict

# Per-file dict cache and fingerprint cache
files_db: Dict[str, Dict[str, Any]] = {}   # path -> {name, path, ext, tags, description, size, mtime, first_indexed, last_indexed}
fp_cache: Dict[str, Dict[str, Any]] = {}   # path -> {size, mtime}

# --------------------------
# File type filters
# --------------------------
TEXT_EXT = {".txt", ".md", ".log", ".csv", ".json", ".xml", ".yml", ".ini",
            ".cs", ".py", ".js", ".ts", ".cpp", ".h", ".java", ".shader"}
PDF_EXT = {".pdf"}
SCAN_ALLOWED_EXTS = TEXT_EXT | PDF_EXT

# --------------------------
# LLM settings (Ollama)
# --------------------------
LLM_PROVIDER = "ollama"         # "ollama" | "none"
LLM_MODEL = "gemma3:latest"
LLM_TIMEOUT_SEC = 1.8           # short timeout for snappy UX
LLM_MAX_CHARS = 140             # limit 1-liner length
LLM_DESC_ENABLED_DEFAULT = True # default behavior in /index/scan


# --------------------------
# Utils
# --------------------------
def make_thumbnail(image_path: str, size: int = 256) -> str:
    """
    Create (or reuse) a thumbnail for the image.
    Returns thumbnail file path.
    """
    h = hashlib.md5(image_path.encode("utf-8")).hexdigest()
    thumb_path = os.path.join(THUMB_DIR, f"{h}.jpg")

    if os.path.isfile(thumb_path):
        return thumb_path

    try:
        im = Image.open(image_path).convert("RGB")
        im.thumbnail((size, size))
        im.save(thumb_path, "JPEG", quality=85)
        return thumb_path
    except Exception:
        return ""

def describe_image_with_llm(pil_image, max_chars: int = 240) -> str:
    """
    Generate a short caption for the image using BLIP (transformers pipeline).
    Falls back to empty string if captioning unavailable.
    """
    if not IMAGE_CAPTION_AVAILABLE or _captioner is None:
        return ""
    try:
        # pipeline returns [{"generated_text": "..."}]
        out = _captioner(pil_image, max_new_tokens=64)
        cap = (out[0].get("generated_text") or "").strip()
        if not cap:
            return ""
        if len(cap) > max_chars:
            cap = cap[:max_chars].rstrip() + "..."
        return cap
    except Exception:
        return ""

def add_caption_chunk_to_text_index(caption: str, origin_path: str) -> None:
    """
    Embed caption with the existing TEXT embedding model and add it into the TEXT FAISS index
    as a 'virtual chunk' pointing back to the image file (origin_path).
    """
    if not caption:
        return
    # 1) embed caption with your TEXT model
    #    (adjust 'text_model' to your actual variable, e.g., 'model' or 'emb_model')
    vec = text_model.encode([caption], normalize_embeddings=True)  # shape: (1, D_txt)

    # 2) add to TEXT faiss index (adjust 'faiss_index' to your variable)
    faiss_index.add(vec)

    # 3) push a meta row (adjust 'meta' to your meta list variable)
    sid = len(meta)
    meta.append({
        "id": sid,
        "path": origin_path,          # NOTE: link back to the image file
        "name": os.path.basename(origin_path),
        "tags": ["image", "caption"],
        "snippet": caption,           # shown in text search results
        "chunk_idx": 0,               # virtual single-chunk
        "kind": "image_caption"       # helpful tag for filtering
    })


def ensure_image_index_initialized(vec_dim: int):
    """Create the image FAISS index on first use."""
    from builtins import int as _int
    global faiss_index_img, D_IMG
    if faiss_index_img is None:
        D_IMG = _int(vec_dim)          # SWIG에 안전하게 파이썬 int로 변환
        faiss_index_img = faiss.IndexFlatIP(D_IMG)

def embed_images(pil_list: List[Image.Image]):
    # SentenceTransformer는 PIL Image 리스트도 인자로 받아 encode 가능
    return img_model.encode(pil_list, normalize_embeddings=True)

from PIL import Image

def add_image_to_index(path: str, add_caption_to_text: bool = True) -> Tuple[int, int]:
    """
    Index a single image into the IMAGE (CLIP) FAISS index.
    Additionally, if add_caption_to_text=True, generate an LLM caption and add it into the TEXT index as a chunk.
    Returns (added_file_count, added_chunks_dummy).
    """
    thumb_path = make_thumbnail(path)

    ext = os.path.splitext(path)[1].lower()
    if ext not in IMAGE_EXT:
        return 0, 0

    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return 0, 0

    # 1) Visual embedding (CLIP) → image FAISS
    vec = embed_images([im])  # (1, D_img)
    ensure_image_index_initialized(int(vec.shape[1]))
    faiss_index_img.add(vec)

    # 2) Update image meta
    sid = len(meta_img)
    base = os.path.basename(path)
    thumb_path = make_thumbnail(path)

    meta_img.append({
        "id": sid,
        "path": path,
        "name": base,
        "thumb": thumb_path,   # ← 썸네일 경로 저장
        "tags": ["image"]
    })
    
    # files_db bookkeeping (optional)
    info = files_db.get(path)
    if not info:
        fp = file_fingerprint(path) or {}
        files_db[path] = {
            "name": base, "path": path, "ext": ext,
            "tags": ["image"],
            "description": base,
            "thumbnail": thumb_path,
            "size": fp.get("size", 0),
            "mtime": fp.get("mtime", 0.0),
            "first_indexed": time.time(), "last_indexed": time.time()
        }
    else:
        info["tags"] = sorted(set(info.get("tags", [])) | {"image"})
        info["last_indexed"] = time.time()

    # 3) Image → caption (LLM) → add into TEXT index as a chunk
    if add_caption_to_text and IMAGE_ADD_CAPTION_TO_TEXT:
        caption = describe_image_with_llm(im, max_chars=240)
        if caption:
            # Optional: store/merge into files_db description for better UX
            fi = files_db.get(path)
            if fi:
                # Prepend caption if description is only filename
                if not fi.get("description") or fi["description"] == base:
                    fi["description"] = caption
                else:
                    # Keep both (short)
                    fi["description"] = f"{caption} | {fi['description']}"
            # Add a "virtual chunk" to TEXT index
            add_caption_chunk_to_text_index(caption, origin_path=path)

    return 1, 1


def iter_files_under(root: str, allowed_exts: Iterable[str]) -> Iterable[str]:
    """Yield all files under root whose extension is in allowed_exts."""
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if allowed_exts and ext not in allowed_exts:
                continue
            yield os.path.join(dirpath, fname)


def file_fingerprint(path: str):
    """Return a lightweight fingerprint: size + mtime."""
    try:
        st = os.stat(path)
        return {"size": int(st.st_size), "mtime": float(st.st_mtime)}
    except Exception:
        return None


def read_text_any(path: str, max_kb: int = 512) -> str:
    """Read text from a supported file with encoding detection and simple PDF text extraction."""
    ext = os.path.splitext(path)[1].lower()
    if ext in PDF_EXT:
        try:
            with pdfplumber.open(path) as pdf:
                pages = []
                acc = 0
                for p in pdf.pages:
                    t = (p.extract_text() or "")
                    if not t:
                        continue
                    sz = len(t.encode("utf-8")) // 1024
                    if acc + sz > max_kb:
                        break
                    pages.append(t)
                    acc += sz
                return "\n".join(pages)
        except Exception:
            return ""
    # Plain text-like
    try:
        with open(path, "rb") as f:
            raw = f.read(max_kb * 1024)
        enc = chardet.detect(raw)["encoding"] or "utf-8"
        return raw.decode(enc, errors="ignore")
    except Exception:
        return ""


def simple_chunk(text: str, max_tokens: int = 800, overlap: int = 150) -> List[str]:
    """Very simple whitespace 'token' chunking; no external tokenizer needed."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_tokens])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def embed_texts(texts: List[str]):
    """Embed a batch of texts using SentenceTransformer."""
    return text_model.encode(texts, normalize_embeddings=NORMALIZE)


def ext_to_tags(ext: str) -> List[str]:
    """Map extension to coarse tags for hybrid filtering/explanation."""
    ext = ext.lower()
    if ext in {".cs", ".py", ".js", ".ts", ".cpp", ".h", ".java", ".shader"}:
        return ["code"]
    if ext in {".txt", ".md", ".log"}:
        return ["text"]
    if ext == ".pdf":
        return ["document", "pdf"]
    if ext in {".json", ".xml", ".csv", ".yml", ".ini"}:
        return ["data"]
    return []


def make_short_description(snippet: str, fallback_name: str = "") -> str:
    """Simple first-sentence extractor with hard length limit."""
    if not snippet:
        return fallback_name or ""
    s = snippet.strip().replace("\n", " ").replace("\r", " ")
    enders = [". ", "? ", "! ", "… ", "。", "？", "！"]
    cut = len(s)
    for e in enders:
        idx = s.find(e)
        if idx != -1:
            cut = min(cut, idx + len(e.strip()))
    s = s[:cut]
    return s[:LLM_MAX_CHARS] + ("…" if len(s) > LLM_MAX_CHARS else "")


# --------------------------
# LLM one-liner via Ollama
# --------------------------
def call_ollama(prompt: str, model: str, timeout: float) -> str:
    """Call Ollama HTTP API (non-streaming). Returns empty string on failure/timeout."""
    url = "http://127.0.0.1:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
            return (payload.get("response") or "").strip()
    except socket.timeout:
        return ""
    except Exception:
        print(f"[LLM] call_ollama failed: {Exception}")   
        return ""


def llm_one_liner(text: str,
                  max_chars: int = LLM_MAX_CHARS,
                  timeout_s: float = LLM_TIMEOUT_SEC) -> str:
    """Generate a concise 1-line summary in Korean. Fallback handled by caller."""
    if not text:
        return ""
    src = text.strip().replace("\r", " ").replace("\n", " ")
    if len(src) > 2000:
        src = src[:2000]
    prompt = (
        "다음 내용을 한국어로 핵심만 담은 1문장(140자 이내)으로 요약해줘.\n"
        "군더더기 표현 없이, 파일의 목적이나 주제를 분명하게.\n"
        f"내용: {src}\n"
        "출력: 한 문장."
    )
    if LLM_PROVIDER == "ollama":
        s = call_ollama(prompt, LLM_MODEL, timeout_s)
    else:
        s = ""
    if not s:
        return ""
    return (s[:max_chars] + "…") if len(s) > max_chars else s


# --------------------------
# Persistence
# --------------------------
def save_index_and_meta():
    """Persist FAISS index, chunk meta, fingerprint cache, and per-file dict."""
    faiss.write_index(faiss_index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(orjson.dumps(m).decode("utf-8") + "\n")
    with open(CACHE_PATH, "wb") as f:
        f.write(orjson.dumps(fp_cache))
    with open(FILES_PATH, "wb") as f:
        f.write(orjson.dumps(files_db))


def load_index_and_meta_if_exists():
    """Load persisted index/meta/caches if present."""
    global faiss_index, meta, id_counter, fp_cache, files_db
    if os.path.isfile(INDEX_PATH):
        faiss_index = faiss.read_index(INDEX_PATH)
    if os.path.isfile(META_PATH):
        meta.clear()
        with open(META_PATH, "rb") as f:
            for line in f:
                meta.append(orjson.loads(line))
        if meta:
            id_counter = max(m.get("id", -1) for m in meta) + 1
    if os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            fp_cache = orjson.loads(f.read())
    if os.path.isfile(FILES_PATH):
        with open(FILES_PATH, "rb") as f:
            files_db = orjson.loads(f.read())

def save_image_index_and_meta():
    if faiss_index_img is None:
        return
    faiss.write_index(faiss_index_img, IMG_INDEX_PATH)
    with open(IMG_META_PATH, "w", encoding="utf-8") as f:
        for m in meta_img:
            f.write(orjson.dumps(m).decode("utf-8") + "\n")

def load_image_index_and_meta_if_exists():
    global faiss_index_img, meta_img, D_IMG
    if os.path.isfile(IMG_INDEX_PATH):
        faiss_index_img = faiss.read_index(IMG_INDEX_PATH)
        D_IMG = int(faiss_index_img.d)     # 로드된 인덱스에서 차원 복구
    if os.path.isfile(IMG_META_PATH):
        meta_img.clear()
        with open(IMG_META_PATH, "rb") as f:
            for line in f:
                meta_img.append(orjson.loads(line))


# --------------------------
# Per-file info upsert (LLM)
# --------------------------
def upsert_file_info(path: str, ext: str, tags: List[str], snippet: str, use_llm_desc: bool = True):
    """Insert/update per-file record in files_db with 1-line description (LLM + fallback)."""
    base = os.path.basename(path)
    fp = file_fingerprint(path) or {"size": 0, "mtime": 0.0}
    now = time.time()

    rule_desc = make_short_description(snippet, base)
    llm_desc = llm_one_liner(snippet) if use_llm_desc else ""
    desc = llm_desc or rule_desc

    rec = files_db.get(path)
    if rec is None:
        files_db[path] = {
            "name": base,
            "path": path,
            "ext": ext,
            "tags": tags,
            "description": desc,
            "size": fp["size"],
            "mtime": fp["mtime"],
            "first_indexed": now,
            "last_indexed": now
        }
    else:
        rec.update({
            "name": base,
            "ext": ext,
            "tags": tags,
            "size": fp["size"],
            "mtime": fp["mtime"],
            "last_indexed": now
        })
        # Refresh description if changed
        if desc and desc != rec.get("description", ""):
            rec["description"] = desc


# --------------------------
# Indexing
# --------------------------
def add_file_to_index(path: str, opts: Dict[str, Any]) -> Tuple[int, int]:
    """Index a single file into FAISS and update meta/files_db."""
    global id_counter
    ext = os.path.splitext(path)[1].lower()
    max_kb = int(opts.get("max_text_kb", 512))
    do_chunk = bool(opts.get("chunk", True))
    use_llm_desc = bool(opts.get("llm_desc", LLM_DESC_ENABLED_DEFAULT))

    if ext not in SCAN_ALLOWED_EXTS:
        return 0, 0

    text = read_text_any(path, max_kb=max_kb)
    if not text.strip():
        return 0, 0

    chunks = simple_chunk(text) if do_chunk else [text]
    vecs = embed_texts(chunks)
    faiss_index.add(vecs)
    added = vecs.shape[0]

    base = os.path.basename(path)
    tags = ext_to_tags(ext)

    # Record per-chunk meta
    for ci, chunk in enumerate(chunks):
        snippet = (chunk[:200] + "…") if len(chunk) > 200 else chunk
        meta.append({
            "id": id_counter,
            "path": path,
            "name": base,
            "ext": ext,
            "tags": tags,
            "chunk_idx": ci,
            "snippet": snippet
        })
        id_counter += 1

    # Update per-file info once (use first chunk for description source)
    if chunks:
        upsert_file_info(path, ext, tags, chunks[0], use_llm_desc)

    return 1, added


def full_rebuild_from_root(root: str, options: Dict[str, Any]) -> Tuple[int, int, int]:
    """Rebuild index/meta/cache/files_db from scratch for given root."""
    global faiss_index, meta, id_counter, fp_cache, files_db
    use_llm_desc = bool(options.get("llm_desc", LLM_DESC_ENABLED_DEFAULT))

    faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    meta.clear()
    id_counter = 0
    fp_cache = {}
    files_db = {}

    f_cnt = c_cnt = scanned = 0
    for path in iter_files_under(root, SCAN_ALLOWED_EXTS):
        scanned += 1
        fp = file_fingerprint(path)
        if not fp:
            continue
        fc, cc = add_file_to_index(path, options)
        if fc:
            fp_cache[path] = fp
        f_cnt += fc
        c_cnt += cc
    return scanned, f_cnt, c_cnt


# --------------------------
# Search
# --------------------------
def search_text(query: str, top_k: int = 20, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Semantic search over the FAISS index and aggregate by file."""
    if faiss_index.ntotal == 0:
        return []
    qv = embed_texts([query])
    scores, ids = faiss_index.search(qv, top_k * 5)  # over-sample then group
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    results_by_file: Dict[str, Dict[str, Any]] = {}
    for sid, sc in zip(ids, scores):
        if sid < 0 or sid >= len(meta):
            continue
        m = meta[sid]
        if filters:
            exts = set(filters.get("ext", []))
            if exts and m["ext"] not in exts:
                continue
        path = m["path"]
        info = files_db.get(path)
        if not info:
            continue
        cur = results_by_file.get(path)
        if (not cur) or (sc > cur["score"]):
            results_by_file[path] = {
                "name": info["name"],
                "path": info["path"],
                "tags": info.get("tags", []),
                "score": float(sc),
                "description": info.get("description", ""),
                "thumbnail": (
                    f"/thumbnail?path={info['thumbnail']}"
                    if info.get("thumbnail") else None
                )
            }

    out = sorted(results_by_file.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return out


# --------------------------
# API Routes
# --------------------------

@app.get("/thumbnail")
def get_thumbnail():
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        return "", 404
    return send_file(path, mimetype="image/jpeg")


@app.post("/index/scan")
def index_scan():
    """
    폴더를 순회하며 텍스트/문서/이미지를 인덱싱한다.
    payload 예시:
    {
      "root": "D:/workspace/assets",
      "options": {
        "max_text_kb": 512,
        "chunk": true,
        "llm_desc": true,
        "scan_images": true
      }
    }
    """
    t0 = time.time()
    payload = request.get_json(force=True) or {}

    root = (payload.get("root") or "").strip()
    if not root or not os.path.isdir(root):
        return jsonify({"ok": False, "msg": f"bad root: {root}"}), 400

    options = payload.get("options") or {}
    # 텍스트/문서 인덱싱 옵션(기존 add_file_to_index가 사용한다면 전달)
    max_text_kb = int(options.get("max_text_kb", 512))
    chunk       = bool(options.get("chunk", True))
    llm_desc    = bool(options.get("llm_desc", True))

    # 이미지 의미 인덱싱 on/off
    scan_images = bool(options.get("scan_images", True))
    llm_desc    = bool(options.get("llm_desc", True))   # 이미지 캡션도 이 옵션을 같이 사용

    # 1) 텍스트/문서 대상 목록(이미지는 제외)
    #    기존 SCAN_ALLOWED_EXTS 집합에서 IMAGE_EXT를 빼서 텍스트/문서만 순회
    textlike_exts = (SCAN_ALLOWED_EXTS - IMAGE_EXT) if isinstance(SCAN_ALLOWED_EXTS, set) else SCAN_ALLOWED_EXTS
    text_paths = list(iter_files_under(root, textlike_exts))

    # 2) 이미지 대상 목록
    img_paths = list(iter_files_under(root, IMAGE_EXT)) if scan_images else []

    # 3) 텍스트/문서 인덱싱
    f_added_text = 0  # 파일 수
    c_added_text = 0  # 생성된 청크 수(기존 add_file_to_index가 반환하면 반영, 아니면 파일 수만 집계)

    for p in text_paths:
        try:
            # add_file_to_index가 (file_count, chunk_count) 형태를 반환한다고 가정
            ret = add_file_to_index(p, {
                "max_text_kb": max_text_kb,
                "chunk": chunk,
                "llm_desc": llm_desc
            })
            if isinstance(ret, tuple) and len(ret) == 2:
                f, c = ret
                f_added_text += int(f)
                c_added_text += int(c)
            else:
                # 반환값 정의가 다르면 파일 1개 추가로만 집계
                f_added_text += 1
        except Exception as e:
            # 개별 파일 실패는 넘어가고 계속
            print(f"[index_scan][text] skip {p}: {e}")

    # 4) 이미지 의미 인덱싱
    f_added_img = 0
    if scan_images:
        for p in img_paths:
            try:
                f, _ = add_image_to_index(p)  # (added_file_count, dummy)
                f_added_img += int(f)
            except Exception as e:
                print(f"[index_scan][image] skip {p}: {e}")

    # 5) 인덱스/메타 저장
    try:
        save_index_and_meta()       # 텍스트/문서용
    except Exception as e:
        print(f"[index_scan] save_index_and_meta error: {e}")
    try:
        save_image_index_and_meta() # 이미지용
    except Exception as e:
        print(f"[index_scan] save_image_index_and_meta error: {e}")

    elapsed = round(time.time() - t0, 3)

    # 6) 응답 요약
    return jsonify({
        "ok": True,
        "root": root,
        "options": {
            "max_text_kb": max_text_kb,
            "chunk": chunk,
            "llm_desc": llm_desc,
            "scan_images": scan_images
        },
        "summary": {
            "textlike": {
                "scanned": len(text_paths),
                "added_files": f_added_text,
                "added_chunks": c_added_text
            },
            "image": {
                "scanned": len(img_paths),
                "added_files": f_added_img
            },
        },
        "files_total": len(files_db),
        "elapsed_sec": elapsed
    }), 200


@app.post("/search")
def search():
    payload = request.get_json(force=True) or {}
    query = payload.get("query", "").strip()
    top_k = int(payload.get("top_k", 20))
    filters = payload.get("filters", None)
    
    # 인덱스 비어있음 감지 + 로그 + 안내 응답
    if faiss_index.ntotal == 0 or len(meta) == 0:
        app.logger.warning("[SEARCH] index empty: run /index/scan first.")
        # 클라이언트가 UI에 표시할 수 있도록 플래그와 메시지 동봉
        return jsonify({
            "results": [],
            "count": 0,
            "need_index": True,
            "msg": "Index is empty. Please call /index/scan with a valid root path."
        }), 200
    
    if not query:
        return jsonify({"results": [], "msg": "empty query"}), 200
    
    results = search_text(query, top_k=top_k, filters=filters)
    return jsonify({"results": results, "count": len(results)})


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "index_size": faiss_index.ntotal,
        "meta_size": len(meta),
        "files_db": (faiss_index.ntotal > 0 and len(meta) > 0)
    })


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    load_index_and_meta_if_exists()
    load_image_index_and_meta_if_exists()
    app.run(host="0.0.0.0", port=5001, debug=True)
