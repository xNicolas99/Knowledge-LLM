import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Depends, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app import config
from app import db
from app import qdrant_store
from app import pipeline
from app import websearch
from app import tavily
from app import watcher

# --- App Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite DB
    db.init_db()

    # Init Qdrant Collections
    await qdrant_store.init_collections()

    # Start Watcher background task
    asyncio.create_task(watcher.watch_loop())

    yield
    # Shutdown logic if needed

app = FastAPI(title="RAGgate Ingest Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependencies (Auth) ---

async def verify_api_key(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    # Try header Authorization: Bearer <key>
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        if token == config.API_KEY:
            return token

    # Try header X-API-Key
    if x_api_key and x_api_key == config.API_KEY:
        return x_api_key

    # Try body for Tavily requests
    if request.method == "POST":
        try:
            body = await request.json()
            if body.get("api_key") == config.API_KEY:
                return body["api_key"]
        except Exception:
            pass # Not JSON or body already read

    raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")

async def verify_read_access(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    request: Request = None
):
    if not config.REQUIRE_AUTH_FOR_READ:
        return True
    return await verify_api_key(request, authorization, x_api_key)

# --- API Endpoints: System & Stats ---

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/stats", dependencies=[Depends(verify_read_access)])
async def get_stats():
    return await qdrant_store.get_stats()

@app.get("/categories", dependencies=[Depends(verify_read_access)])
async def get_categories():
    return {"categories": config.KNOWLEDGE_CATEGORIES}

# --- API Endpoints: Knowledge Base ---

@app.get("/search", dependencies=[Depends(verify_read_access)])
async def search(q: str, top_k: int = 5, category: Optional[str] = None):
    results = await qdrant_store.search(q, category, top_k)
    return {"results": results}

class UpdateSourceBody(BaseModel):
    source: str
    change: str

@app.post("/update-source", dependencies=[Depends(verify_api_key)])
async def update_source(body: UpdateSourceBody):
    return await pipeline.update_source_document(body.source, body.change)

@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_file(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None)
):
    tmp_path = f"/tmp/{file.filename}"
    try:
        # Save temp file
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        text = await pipeline.extract_text(tmp_path)
        result_status = await pipeline.enrich(text, source=file.filename, force_category=category)

        return {"filename": file.filename, "status": result_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

class EnrichRequest(BaseModel):
    text: str
    source: str
    category: Optional[str] = None

@app.post("/enrich", dependencies=[Depends(verify_api_key)])
async def enrich_endpoint(req: EnrichRequest):
    result_status = await pipeline.enrich(req.text, req.source, req.category)
    return {"status": result_status}

# --- API Endpoints: Review Queue ---

@app.get("/review", dependencies=[Depends(verify_api_key)])
async def get_review_queue():
    items = db.get_pending_reviews()
    return {"items": items}

class ReviewDecision(BaseModel):
    id: int
    decision: str # "approved" or "rejected"

@app.post("/review/decision", dependencies=[Depends(verify_api_key)])
async def review_decision(decision: ReviewDecision):
    if decision.decision not in ["approved", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid decision")

    item = db.resolve_review(decision.id, decision.decision)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if decision.decision == "approved":
        # Process and upsert
        chunks = pipeline.chunk_text(item["text"])
        processed_chunks = []
        for c in chunks:
            cleaned = await pipeline.clean_chunk(c)
            processed_chunks.append({
                "text": cleaned,
                "source": item["source"],
                "category": item["category"],
                "tags": [] # Tags could be stored in DB if needed
            })
        await qdrant_store.upsert(processed_chunks)
        return {"status": "approved_and_indexed"}

    return {"status": "rejected"}

# --- API Endpoints: Web Search (Direct) ---

@app.get("/websearch", dependencies=[Depends(verify_read_access)])
async def get_websearch(q: str, top_k: int = 5, evaluate: bool = config.WEBSEARCH_EVALUATE):
    results = await websearch.websearch(q, top_k, evaluate)
    return {"results": results}

# --- API Endpoints: Tavily Adapter ---

@app.post("/tavily/search", dependencies=[Depends(verify_api_key)])
async def post_tavily_search(req: tavily.TavilySearchRequest):
    return await tavily.tavily_search(req)

@app.post("/tavily/extract", dependencies=[Depends(verify_api_key)])
async def post_tavily_extract(req: tavily.TavilyExtractRequest):
    return await tavily.tavily_extract(req)

# --- API Endpoints: Re-Index ---

@app.post("/reindex", dependencies=[Depends(verify_api_key)])
async def trigger_reindex(clean: bool = False):
    return pipeline.start_reindex(clean)

@app.get("/reindex/status", dependencies=[Depends(verify_api_key)])
async def reindex_status():
    return pipeline.get_reindex_status()

# --- API Endpoints: Prompts ---

class PromptUpdate(BaseModel):
    name: str
    content: str

@app.get("/prompt", dependencies=[Depends(verify_api_key)])
async def get_prompts():
    return {
        "clean_text": db.get_prompt("clean_text"),
        "gatekeeper": db.get_prompt("gatekeeper"),
        "update_document": db.get_prompt("update_document")
    }

@app.post("/prompt", dependencies=[Depends(verify_api_key)])
async def update_prompt(req: PromptUpdate):
    db.update_prompt(req.name, req.content)
    return {"status": "updated"}

@app.post("/prompt/reset", dependencies=[Depends(verify_api_key)])
async def reset_prompts():
    db.init_db() # This will re-insert defaults if missing, though it won't overwrite existing unless we modify db.py. Let's just return success for MVP
    return {"status": "reset triggered (defaults re-initialized if missing)"}

# --- WEB UI (Inline HTML) ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Simple TailwindCSS Dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAGgate Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            // Store API Key for requests
            const API_KEY = "{config.API_KEY}";
            const HEADERS = {{
                "Authorization": `Bearer ${{API_KEY}}`,
                "Content-Type": "application/json"
            }};

            async function fetchStats() {{
                const res = await fetch("/stats", {{ headers: HEADERS }});
                const data = await res.json();
                document.getElementById('stats').innerText = JSON.stringify(data, null, 2);
            }}

            async function fetchReviewQueue() {{
                const res = await fetch("/review", {{ headers: HEADERS }});
                const data = await res.json();
                const container = document.getElementById('review-queue');
                container.innerHTML = "";

                if (data.items.length === 0) {{
                    container.innerHTML = "<p class='text-gray-500'>No pending reviews.</p>";
                    return;
                }}

                data.items.forEach(item => {{
                    const div = document.createElement('div');
                    div.className = "border p-4 mb-4 rounded bg-white shadow";
                    div.innerHTML = `
                        <div class="flex justify-between">
                            <strong>[${{item.category}}] Source: ${{item.source}}</strong>
                            <span class="text-sm text-red-500">Reason: ${{item.reason}}</span>
                        </div>
                        <div class="mt-2 mb-4 text-sm text-gray-700 bg-gray-50 p-2 rounded max-h-32 overflow-y-auto">
                            ${{item.text}}
                        </div>
                        <div class="flex gap-2">
                            <button onclick="resolveReview(${{item.id}}, 'approved')" class="bg-green-500 text-white px-4 py-1 rounded hover:bg-green-600">Approve</button>
                            <button onclick="resolveReview(${{item.id}}, 'rejected')" class="bg-red-500 text-white px-4 py-1 rounded hover:bg-red-600">Reject</button>
                        </div>
                    `;
                    container.appendChild(div);
                }});
            }}

            async function resolveReview(id, decision) {{
                await fetch("/review/decision", {{
                    method: "POST",
                    headers: HEADERS,
                    body: JSON.stringify({{ id, decision }})
                }});
                fetchReviewQueue();
            }}

            async function uploadFile(event) {{
                event.preventDefault();
                const fileInput = document.getElementById('file-upload');
                const catInput = document.getElementById('cat-upload');
                if (!fileInput.files[0]) return;

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                if (catInput.value) formData.append("category", catInput.value);

                const statusEl = document.getElementById('upload-status');
                statusEl.innerText = "Uploading...";

                try {{
                    const res = await fetch("/upload", {{
                        method: "POST",
                        headers: {{ "Authorization": `Bearer ${{API_KEY}}` }}, // FormData handles content-type
                        body: formData
                    }});
                    const data = await res.json();
                    statusEl.innerText = `Status: ${{data.status}}`;
                    fileInput.value = "";
                    fetchStats();
                    fetchReviewQueue();
                }} catch (e) {{
                    statusEl.innerText = `Error: ${{e}}`;
                }}
            }}

            async function loadPrompts() {{
                const res = await fetch("/prompt", {{ headers: HEADERS }});
                const data = await res.json();
                document.getElementById('prompt-clean').value = data.clean_text || "";
                document.getElementById('prompt-gate').value = data.gatekeeper || "";
                document.getElementById('prompt-update').value = data.update_document || "";
            }}

            async function savePrompt(name, elementId) {{
                const content = document.getElementById(elementId).value;
                const statusEl = document.getElementById(`status-${{elementId}}`);
                statusEl.innerText = "Saving...";

                await fetch("/prompt", {{
                    method: "POST",
                    headers: HEADERS,
                    body: JSON.stringify({{ name, content }})
                }});
                statusEl.innerText = "Saved!";
                setTimeout(() => statusEl.innerText="", 2000);
            }}

            window.onload = () => {{
                fetchStats();
                fetchReviewQueue();
                loadPrompts();
            }};
        </script>
    </head>
    <body class="bg-gray-100 text-gray-800 font-sans p-8">
        <div class="max-w-6xl mx-auto space-y-8">
            <h1 class="text-3xl font-bold border-b pb-2">RAGgate Dashboard</h1>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <!-- Upload & Stats -->
                <div class="space-y-8">
                    <div class="bg-white p-6 rounded shadow">
                        <h2 class="text-xl font-semibold mb-4">Stats</h2>
                        <pre id="stats" class="text-sm bg-gray-50 p-4 rounded overflow-x-auto"></pre>
                        <button onclick="fetchStats()" class="mt-4 text-blue-500 hover:underline">Refresh</button>
                    </div>

                    <div class="bg-white p-6 rounded shadow">
                        <h2 class="text-xl font-semibold mb-4">Upload Document</h2>
                        <form onsubmit="uploadFile(event)" class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium">Category (optional)</label>
                                <select id="cat-upload" class="border p-2 rounded w-full">
                                    <option value="">-- Auto Detect --</option>
                                    {"".join(f'<option value="{c}">{c}</option>' for c in config.KNOWLEDGE_CATEGORIES)}
                                </select>
                            </div>
                            <div>
                                <label class="block text-sm font-medium">File</label>
                                <input type="file" id="file-upload" class="border p-2 rounded w-full" required>
                            </div>
                            <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">Upload & Process</button>
                            <p id="upload-status" class="text-sm font-medium text-blue-600"></p>
                        </form>
                    </div>
                </div>

                <!-- Review Queue -->
                <div class="bg-white p-6 rounded shadow flex flex-col h-[600px]">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-semibold">Review Queue</h2>
                        <button onclick="fetchReviewQueue()" class="text-blue-500 hover:underline">Refresh</button>
                    </div>
                    <div id="review-queue" class="overflow-y-auto flex-1 bg-gray-50 p-4 rounded border">
                        <!-- Items populated via JS -->
                    </div>
                </div>
            </div>

            <!-- Prompts -->
            <div class="bg-white p-6 rounded shadow space-y-6">
                <h2 class="text-xl font-semibold">Prompt Editor</h2>

                <div>
                    <div class="flex justify-between items-center mb-2">
                        <label class="font-medium">Text Cleaning Prompt</label>
                        <span id="status-prompt-clean" class="text-sm text-green-600"></span>
                    </div>
                    <textarea id="prompt-clean" rows="6" class="w-full border p-2 rounded font-mono text-sm"></textarea>
                    <button onclick="savePrompt('clean_text', 'prompt-clean')" class="mt-2 bg-gray-800 text-white px-4 py-2 rounded hover:bg-black">Save Cleaning Prompt</button>
                </div>

                <div>
                    <div class="flex justify-between items-center mb-2">
                        <label class="font-medium">Gatekeeper Prompt</label>
                        <span id="status-prompt-gate" class="text-sm text-green-600"></span>
                    </div>
                    <textarea id="prompt-gate" rows="6" class="w-full border p-2 rounded font-mono text-sm"></textarea>
                    <button onclick="savePrompt('gatekeeper', 'prompt-gate')" class="mt-2 bg-gray-800 text-white px-4 py-2 rounded hover:bg-black">Save Gatekeeper Prompt</button>
                </div>

                <div>
                    <div class="flex justify-between items-center mb-2">
                        <label class="font-medium">Update Document Prompt</label>
                        <span id="status-prompt-update" class="text-sm text-green-600"></span>
                    </div>
                    <textarea id="prompt-update" rows="6" class="w-full border p-2 rounded font-mono text-sm"></textarea>
                    <button onclick="savePrompt('update_document', 'prompt-update')" class="mt-2 bg-gray-800 text-white px-4 py-2 rounded hover:bg-black">Save Update Document Prompt</button>
                </div>
            </div>

        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
