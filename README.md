# RAGgate: Self-Hosted Web-Research & Knowledge-Base Stack

**RAGgate** ist ein vollständiger, eigenständiger und sofort deploybarer Docker-Compose-Stack für Web-Recherche und Wissensmanagement. Er integriert Semantische Suche, LLM-gestützte Textbereinigung, Websuche (SearXNG + Crawl4AI) und eine Freigabe-Warteschlange ("Review Queue") für neues Wissen.

## 1. Architektur-Überblick

```text
                 ┌──────────────────────────────────────────┐
                 │  ingest-service (FastAPI)                │
                 │  - Web-UI (Upload, Review, Prompt-Editor)│
                 │  - /websearch (SearXNG→Crawl4AI→LLM)     │
                 │  - /tavily/search + /tavily/extract      │
                 │  - Knowledge-Base (Qdrant)               │
                 │  - Selbst-Anreicherung + Review-Queue    │
                 └───┬───────────┬───────────┬───────────┬────┘
                     │           │           │           │
              ┌──────▼───┐ ┌─────▼─────┐ ┌───▼────┐ ┌────▼─────────┐
              │ SearXNG  │ │ Crawl4AI  │ │ Qdrant │ │ ext. OpenAI- │
              │ (+Redis  │ │ (Headless │ │(Vektor)│ │ kompat. LLM  │
              │  +Tor)   │ │  Chromium)│ │        │ │ + Embedding  │
              └──────────┘ └───────────┘ └────────┘ └──────────────┘
                                                       (vom Nutzer)
        ┌─────────────────────────────────────────────────────────┐
        │ OpenWebUI (optional, im Compose)                        │
        │  - eigenes Vektor-DB (Chroma)                           │
        │  - "Knowledge & Research" Tool für /search, /websearch  │
        └─────────────────────────────────────────────────────────┘
```

RAGgate ist darauf ausgelegt, komplett autark zu laufen (ohne Cloud-Abhängigkeiten), **außer** einem vom Nutzer bereitgestellten OpenAI-kompatiblen LLM- und Embedding-Endpunkt (z.B. Ollama, LM Studio, vLLM oder OpenAI).

---

## 2. Voraussetzungen

- **Docker & Docker Compose** installiert.
- Ein **OpenAI-kompatibler LLM-Endpunkt** (z.B. `http://host.docker.internal:11434/v1` via Ollama).
- Ein **OpenAI-kompatibler Embedding-Endpunkt** (z.B. über dasselbe Ollama-Setup).

---

## 3. Schnellstart

1. **Repository klonen**
   ```bash
   git clone <repo-url> raggate
   cd raggate
   ```

2. **Umgebungsvariablen konfigurieren**
   Kopiere die Beispiel-Konfiguration und trage deine API-Keys/URLs ein:
   ```bash
   cp .env.example .env
   # Bearbeite die .env (WICHTIG: Setze API_KEY und LLM_BASE_URL)
   nano .env
   ```

3. **Stack starten**
   ```bash
   docker compose up -d
   ```

4. **Healthcheck**
   Das Web-Dashboard des Ingest-Service ist unter `http://localhost:8000` erreichbar. Die API-Docs findest du unter `http://localhost:8000/docs`.

---

## 4. Konfiguration (`.env`)

| Variable | Bedeutung | Beispiel/Default |
|---|---|---|
| `LLM_BASE_URL` | Basis-URL für den Chat-Endpunkt | `http://host.docker.internal:11434/v1` |
| `LLM_API_KEY` | Key für Chat-Endpunkt | `dummy` |
| `LLM_MODEL` | Modellname für Chat | `llama3` |
| `EMBEDDING_BASE_URL` | Basis-URL für Embeddings | `http://host.docker.internal:11434/v1` |
| `EMBEDDING_API_KEY` | Key für Embeddings | `dummy` |
| `EMBEDDING_MODEL` | Modellname für Embeddings | `nomic-embed-text` |
| `EMBEDDING_DIM` | Dimension der Vektoren | `768` |
| `KNOWLEDGE_CATEGORIES`| Kommagetrennte Kategorien | `it,science,biology,business,general` |
| `API_KEY` | **Pflicht**: Schutz für schreibende API-Aufrufe | `mein-geheimer-key` |
| `REQUIRE_AUTH_FOR_READ`| Ob auch Lese-Aufrufe den Key brauchen | `true` |
| `SEARXNG_SECRET`| Wichtig: Basis-Verschlüsselung für SearXNG | Automatisch generiert falls leer |
| `CLEAN_BLOCK_CHARS`| Blockgröße für LLM-Bereinigungs-Aufrufe | `6000` |
| `CRAWL_CONCURRENCY`| Parallele Chromium-Tabs | `3` |
| `CONFLICT_CHECK_ALL_CATEGORIES`| Sucht über alle Kategorien nach Duplikaten | `false` |

*(Siehe `.env.example` für alle Variablen wie Chunker, Crawler-Tokens, etc.)*

---

## 5. Komponenten & Dienste

- **ingest-service**: Der Kern (FastAPI). Übernimmt Text-Extraktion, LLM-Bereinigung, Semantische Suche, Websuche-Koordination und stellt das Web-UI bereit.
- **Qdrant**: Die Vektor-Datenbank, welche die Dokument-Chunks für die Wissensbasis (aufgeteilt in Kategorien/Collections) vorhält.
- **Docling**: Ein Service (`docling-serve`), um komplexe Binär- und Office-Dokumente (PDFs, PPTX) in lesbaren Text zu extrahieren.
- **Crawl4AI**: Ein Headless Chromium Container (gepinnt auf `0.8.9` wegen Stabilität), der rohe URLs in sauberes Markdown rendert (`shm_size: 1gb` wichtig für Chrome).
- **SearXNG + Tor + Redis**: Eine Meta-Suchmaschine. **Wichtig:** Wir fokussieren die Engine-Liste in der `searxng/settings.yml.template` auf allgemeine Web-Engines (DuckDuckGo, Brave, Startpage, Wikipedia, etc.) und deaktivieren Spezial-Engines, um Müll-Treffer im JSON-Format zu vermeiden. Google und Bing werden bewusst über einen minimalistischen (Alpine) Tor-Container geleitet, um IP-Sperren zu umgehen. Die SearXNG Instanz nutzt einen dynamischen Entrypoint um fehlende API Secrets zur Laufzeit per `sed` sicher zu generieren.

---

## 6. Wissensspeicher (Knowledge Base)

Das Wissen wird in getrennte **Qdrant-Collections** basierend auf `KNOWLEDGE_CATEGORIES` unterteilt. Es gibt 3 Wege, neues Wissen hinzuzufügen:
1. **Web-UI**: Upload via `http://localhost:8000/`.
2. **Watch-Ordner**: Lege Dokumente in den `./watch/`-Ordner im Repo. Der Watcher verarbeitet sie automatisch und verschiebt sie nach `/watch/processed/`.
3. **API / OpenWebUI Tool**: Nutzung des `/enrich`-Endpoints.

**Freigabe-Workflow & Halluzinationsschutz**:
Neu eingereichtes Wissen durchläuft einen "Türsteher" (LLM entscheidet über Relevanz und ordnet Kategorien zu). Bei Ähnlichkeit mit bestehendem Wissen (> `CONFLICT_THRESHOLD`) wird es nicht blind überschrieben, sondern landet in der **Review Queue** in der Web-UI.
Wird ein Text verarbeitet, durchläuft er eine LLM-gestützte Bereinigung. Ein Längenschutz (`CLEAN_MIN_RATIO`, `CLEAN_MAX_RATIO`) verhindert, dass das LLM Texte komplett erfindet oder drastisch kürzt.

---

## 7. Websuche (`/websearch`)

Die Websuche koordiniert `SearXNG` (findet URLs) und `Crawl4AI` (liest den Inhalt als Markdown). Optional bewertet das LLM die Relevanz der gefundenen Inhalte.

```bash
curl "http://localhost:8000/websearch?q=Docker%20Release&top_k=3" \
  -H "Authorization: Bearer mein-geheimer-key"
```

*Troubleshooting:* Wenn keine Treffer kommen, überprüfe ob SearXNG im JSON-Format antwortet (die `settings.yml` aktiviert dies) und ob Google/Bing über Tor blockiert werden (die direkten Engines wie Brave und DuckDuckGo sind der zuverlässige Fallback).

---

## 8. Tavily-Adapter

RAGgate stellt Endpunkte bereit, die exakt das **Tavily API-Schema** nachbilden. Dadurch können Coding-Agenten oder LLM-Clients, die Tavily erwarten, nahtlos an RAGgate angebunden werden.
- `/tavily/search` (bei `search_depth="advanced"` wird die LLM-Relevanzbewertung aktiviert)
- `/tavily/extract`

Setze RAGgate als Tavily-Basis-URL und nutze deinen `API_KEY` als Tavily-Key.

---

## 9. OpenWebUI Integration & Das "Knowledge & Research" Tool

OpenWebUI kann optional per Profil gestartet werden:
```bash
docker compose --profile with-openwebui up -d
```
Es läuft unter `http://localhost:3000`.

**Installation des Tools:**
Im Hauptverzeichnis liegt die Datei `raggate_tool.py`.
1. Öffne OpenWebUI im Browser.
2. Gehe zu Workspace -> Tools -> "+" (Import Tool).
3. Lade die Datei `raggate_tool.py` hoch.
4. Setze unter den Tool-Valves die Variablen `RAGGATE_API_URL` (z.B. `http://ingest:8000` wenn beide im Docker-Netzwerk sind, sonst die externe URL) und `RAGGATE_API_KEY`.
5. Jetzt können deine OpenWebUI-Chats direkt auf die RAGgate-Wissensdatenbank und die Websuche zugreifen!

---

## 10. Reverse Proxy (TLS / Eigene Domain)

Um den Ingest-Service und den Tavily-Adapter sicher im Internet bereitzustellen, gibt es ein Caddy-Profil.

```bash
docker compose --profile with-proxy up -d
```
Passe vorher in der `.env` die Variable `PUBLIC_HOSTNAME=research.example.com` an und vergewissere dich, dass deine Domain auf den Server zeigt. Caddy kümmert sich automatisch um die TLS/HTTPS-Zertifikate.

---

## 11. Betrieb & Troubleshooting

- **Daten-Persistenz:** Die SQLite-DB (`app/data`), Qdrant-Vektoren, SearXNG-Configs und Caddy-Zertifikate sind in Docker-Volumes gespeichert. Ein Neustart verliert keine Daten.
- **Re-Index:** Wenn Embeddings oder LLM-Modelle gewechselt werden, kann über `POST /reindex` ein echter Hintergrundprozess angestoßen werden. Hierbei werden **alle** gespeicherten Payloads aus Qdrant geladen, die Collection wird komplett neu angelegt (wichtig wenn sich die `EMBEDDING_DIM` der Modelle ändert!) und die Texte werden frisch eingebettet/upserted. **Achtung**: Alle alten Vektoren werden hierbei gelöscht und mit neuen Vektoren des aktuellen Modells ersetzt. Der Status ist unter `GET /reindex/status` abfragbar.
- **Häufige Fehler:**
  - *Ingest erreicht Qdrant nicht*: Prüfe die Start-Reihenfolge. `ingest` wartet via Healthcheck auf `qdrant`.
  - *Crawl4AI liefert leeren Text*: Stelle sicher, dass `shm_size: 1gb` in der `docker-compose.yml` für Crawl4AI greift (Headless-Chrome stürzt sonst bei großen Seiten ab).
  - *LLM liefert kein JSON*: Das Framework nutzt Regex und Try-Catch, um Markdown-Code-Blöcke (` ```json `) sicher abzufangen. Prüfe die Logs des LLMs.
