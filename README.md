# Disney Reviews QA — Quick Run

This README is **minimal** for the home assignment. All heavy preprocessing (embeddings, clustering, index building) has already been done. To run the app, you just need to install requirements, fetch the prebuilt artifacts, and start the backend + frontend.

---

## 1. Clone the Repo

```bash
git clone https://github.com/dorgol/stampli.git
cd stampli
```

---

## 2. Install Requirements

Create and activate a virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel
pip install -r requirements.txt
```

---

## 3. Fetch Prebuilt Artifacts

Artifacts (Chroma index + labeled reviews parquet) are hosted on GitHub Releases. Run this once:

```powershell
PowerShell -ExecutionPolicy Bypass -File .\scripts\fetch_artifacts.ps1
```

This will create:

```
clustering_out/reviews_with_clusters_labeled.parquet
index/chroma_db/
```

---

## 4. Configure Environment Variables

Set the paths so the backend knows where to look:

```powershell
$env:CHROMA_PATH = "index/chroma_db"
$env:CHROMA_COLLECTION = "reviews"
$env:LABELED_PARQUET = "clustering_out/reviews_with_clusters_labeled.parquet"
$env:OPENAI_API_KEY = "sk-..."   # required for query-time embeddings
```

---

## 5. Run the App

**Backend (FastAPI):**

```powershell
uvicorn src.qa_server:app --host 0.0.0.0 --port 8001 --reload
```

**Frontend (Streamlit):** (new terminal, same venv)

```powershell
streamlit run src/app_frontend.py
```

---

## 6. Usage

Open [http://localhost:8501](http://localhost:8501) in your browser. The Streamlit UI will call the backend at `http://localhost:8001/ask` and show:

* Synthesized answer (+ citations)
* Parsed filters
* Aggregate stats (count, avg rating, positive share, monthly/seasonal trends, clusters, topics)
* Retrieved review snippets

---

## 7. Troubleshooting

* **Index not found** → Did you run `scripts/fetch_artifacts.ps1`?
* **Port already in use** → Change `--port` for uvicorn or close the process on 8001.
* **Missing OPENAI\_API\_KEY** → Set it as shown above, unless you patched retrieval to work offline.

---

## 8. Repo Layout (relevant parts)

```
src/
  qa_server.py        # FastAPI backend (/ask)
  app_frontend.py     # Streamlit frontend
scripts/
  fetch_artifacts.ps1 # downloads and unzips prebuilt index
clustering_out/
  reviews_with_clusters_labeled.parquet   # (downloaded)
index/
  chroma_db/                               # (downloaded)
```

---

✅ That’s it: after running the fetch script, the only commands you need are:

```powershell
uvicorn src.qa_server:app --port 8001 --reload
streamlit run src/app_frontend.py
```
