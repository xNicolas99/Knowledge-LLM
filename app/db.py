import sqlite3
import json
from typing import Any, Dict, List, Optional
import os
from app import config

# Ensure DB directory exists
os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize SQLite tables for Prompts and Review Queue."""
    with _get_db() as conn:
        cursor = conn.cursor()

        # Table for prompts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                name TEXT PRIMARY KEY,
                content TEXT NOT NULL
            )
        """)

        # Table for review queue
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', -- pending, approved, rejected
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert default prompts if they don't exist
        default_clean_prompt = (
            "Du bist ein strikter Text-Bereinigungs-Assistent. "
            "Deine Aufgabe ist es, den folgenden Textblock von Formatierungsfehlern, "
            "überflüssigen Menüpunkten, Werbung und irrelevanter Navigation zu säubern.\n\n"
            "WICHTIGE REGELN:\n"
            "1. Nur entfernen, niemals erfinden!\n"
            "2. Im Zweifel den Text wortgleich belassen.\n"
            "3. Keine zusammenfassungen schreiben.\n\n"
            "TEXT:\n{text}"
        )

        default_gatekeeper_prompt = (
            "Analysiere das folgende Dokument und die Quelle.\n"
            "Entscheide, ob es relevant und wissenswert ist.\n"
            "Weise es einer der folgenden Kategorien zu: {categories}\n\n"
            "Antworte AUSSCHLIESSLICH mit einem JSON-Objekt:\n"
            "{\"keep\": bool, \"summary\": \"Kurze Zusammenfassung\", \"tags\": [\"tag1\", \"tag2\"], \"category\": \"gewählte_kategorie\"}\n\n"
            "DOKUMENT:\n{text}"
        )

        default_update_prompt = (
            "Du bearbeitest ein Dokument aus einer technischen Wissensdatenbank.\n"
            "Wende AUSSCHLIESSLICH die folgende Änderung an:\n\n"
            "ÄNDERUNG: {change}\n\n"
            "STRENGE REGELN:\n"
            "- Wende NUR die beschriebene Änderung an, sonst nichts.\n"
            "- Erfinde KEINE neuen Fakten und füge KEINE Inhalte hinzu.\n"
            "- Ändere NICHT die Struktur, Reihenfolge oder Formatierung.\n"
            "- Entferne KEINE Abschnitte, Zeilen oder Absätze.\n"
            "- Bei Unsicherheit lieber NICHTS ändern und den Text unverändert zurückgeben.\n"
            "- Gib NUR den vollständigen, bearbeiteten Dokumenttext zurück – keine Erklärungen, keine Markierungen, kein Kommentar.\n\n"
            "DOKUMENT:\n"
            "{document}"
        )

        cursor.execute("INSERT OR IGNORE INTO prompts (name, content) VALUES (?, ?)", ("clean_text", default_clean_prompt))
        cursor.execute("INSERT OR IGNORE INTO prompts (name, content) VALUES (?, ?)", ("gatekeeper", default_gatekeeper_prompt))
        cursor.execute("INSERT OR IGNORE INTO prompts (name, content) VALUES (?, ?)", ("update_document", default_update_prompt))

        conn.commit()

# --- Prompts Management ---

def get_prompt(name: str) -> Optional[str]:
    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM prompts WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            return row["content"]
    return None

def update_prompt(name: str, content: str):
    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE prompts SET content = ? WHERE name = ?", (content, name))
        if cursor.rowcount == 0:
            cursor.execute("INSERT INTO prompts (name, content) VALUES (?, ?)", (name, content))
        conn.commit()

# --- Review Queue Management ---

def add_to_review(text: str, source: str, category: str, reason: str = "") -> int:
    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO review_queue (text, source, category, reason) VALUES (?, ?, ?, ?)",
            (text, source, category, reason)
        )
        conn.commit()
        return cursor.lastrowid

def get_pending_reviews() -> List[Dict[str, Any]]:
    with _get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM review_queue WHERE status = 'pending' ORDER BY created_at ASC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def resolve_review(item_id: int, decision: str) -> Optional[Dict[str, Any]]:
    """decision should be 'approved' or 'rejected'"""
    if decision not in ("approved", "rejected"):
        raise ValueError("Decision must be 'approved' or 'rejected'")

    with _get_db() as conn:
        cursor = conn.cursor()
        # Get the item before updating it
        cursor.execute("SELECT * FROM review_queue WHERE id = ?", (item_id,))
        row = cursor.fetchone()

        if not row:
            return None

        cursor.execute("UPDATE review_queue SET status = ? WHERE id = ?", (decision, item_id))
        conn.commit()
        return dict(row)
