# server.py
# Flask 기반 텍스트 RAG 최소 서버
# 엔드포인트:
#  - GET  /health
#  - POST /index/files  { "files": [{"path": "..."}, ...], "options": {"chunk": true, "max_text_kb": 512} }
#  - POST /search       { "query": "...", "top_k": 20, "filters": {"ext": [".txt",".md"]} }

import os
import io
import time
import faiss
import orjson
import chardet
import pdfplumber
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

# --------------------------
# 간단한 전역 상태 (파일 저장/버전관리 생략)
# --------------------------
app = Flask(__name__)
MODEL_NAME = "BAAI/bge-m3"   # 1024d, 다국어 의미검색 강함
EMBED_DIM  = 1024
NORMALIZE  = True

text_model = SentenceTransformer(MODEL_NAME)
faiss_index = faiss.IndexFlatIP(EMBED_DIM)  # cosine 유사도 = 내적 + 정규화
meta = []  # id -> {path, ext, labels, chunk_idx, snippet}
id_counter = 0

# --------------------------
# 유틸: 텍스트 로딩/청크/임베딩
# --------------------------
TEXT_EXT = {".txt", ".md", ".log", ".csv", ".json", ".xml", ".yml", ".ini", ".cs", ".py", ".js", ".ts", ".cpp", ".h", ".java", ".shader"}
PDF_EXT  = {".pdf"}

def read_text_any(path, max_kb=512):
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
                    sz = len(t.encode("utf-8"))//1024
                    if acc + sz > max_kb:
                        break
                    pages.append(t)
                    acc += sz
                return "\n".join(pages)
        except Exception:
            return ""
    # 일반 텍스트 계열
    try:
        with open(path, "rb") as f:
            raw = f.read(max_kb * 1024)
        enc = chardet.detect(raw)["encoding"] or "utf-8"
        return raw.decode(enc, errors="ignore")
    except Exception:
        return ""

def simple_chunk(text, max_tokens=800, overlap=150):
    # 토큰라이저 없이 단어 단위 근사 분할
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max_tokens - overlap if max_tokens > overlap else max_tokens
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+max_tokens])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def embed_texts(texts):
    vecs = text_model.encode(texts, normalize_embeddings=NORMALIZE)
    return vecs

def ext_to_labels(ext):
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

# --------------------------
# 인덱싱
# --------------------------
def add_file_to_index(path, opts):
    global id_counter
    ext = os.path.splitext(path)[1].lower()
    max_kb = int(opts.get("max_text_kb", 512))
    do_chunk = bool(opts.get("chunk", True))

    # 텍스트화 가능한 확장자만 처리
    if ext not in TEXT_EXT and ext not in PDF_EXT:
        return 0, 0

    text = read_text_any(path, max_kb=max_kb)
    if not text.strip():
        return 0, 0

    chunks = simple_chunk(text) if do_chunk else [text]
    vecs = embed_texts(chunks)

    faiss_index.add(vecs)
    added = vecs.shape[0]

    # 메타 저장
    base = os.path.basename(path)
    labels = ext_to_labels(ext)
    for ci, chunk in enumerate(chunks):
        snippet = (chunk[:200] + "…") if len(chunk) > 200 else chunk
        meta.append({
            "id": id_counter,
            "path": path,
            "name": base,
            "ext": ext,
            "labels": labels,
            "chunk_idx": ci,
            "snippet": snippet
        })
        id_counter += 1
    return 1, added

# --------------------------
# 검색
# --------------------------
def search_text(query, top_k=20, filters=None):
    qv = embed_texts([query])
    scores, ids = faiss_index.search(qv, top_k * 5)  # 넉넉히 뽑은 뒤 파일 단위로 묶기
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    # 필터 적용 및 파일 단위 집계
    results_by_file = {}
    for sid, sc in zip(ids, scores):
        if sid < 0 or sid >= len(meta):
            continue
        m = meta[sid]
        # 확장자 필터 등
        if filters:
            exts = set(filters.get("ext", []))
            if exts and m["ext"] not in exts:
                continue
        path = m["path"]
        cur = results_by_file.get(path)
        if not cur or sc > cur["score"]:
            results_by_file[path] = {
                "path": path,
                "name": m["name"],
                "ext": m["ext"],
                "labels": m["labels"],
                "score": float(sc),
                "snippet": m["snippet"]
            }

    # 상위 top_k 정렬
    out = sorted(results_by_file.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return out

# --------------------------
# API
# --------------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "index_size": faiss_index.ntotal,
        "meta_size": len(meta)
    })

@app.post("/index/files")
def index_files():
    payload = request.get_json(force=True) or {}
    files = payload.get("files", [])
    options = payload.get("options", {"chunk": True, "max_text_kb": 512})
    t0 = time.time()
    f_cnt = 0
    c_cnt = 0
    for f in files:
        p = f.get("path")
        if not p or not os.path.isfile(p):
            continue
        fc, cc = add_file_to_index(p, options)
        f_cnt += fc
        c_cnt += cc
    dt = time.time() - t0
    return jsonify({
        "indexed_files": f_cnt,
        "indexed_chunks": c_cnt,
        "index_size": faiss_index.ntotal,
        "elapsed_sec": round(dt, 3)
    })

@app.post("/search")
def search():
    payload = request.get_json(force=True) or {}
    query = payload.get("query", "").strip()
    top_k = int(payload.get("top_k", 20))
    filters = payload.get("filters", None)
    if not query:
        return jsonify({"results": [], "msg": "empty query"}), 200
    results = search_text(query, top_k=top_k, filters=filters)
    return jsonify({"results": results, "count": len(results)})

if __name__ == "__main__":
    # 개발용 서버
    app.run(host="0.0.0.0", port=5001, debug=True)
