"""
title: RAGgate Knowledge & Research
author: Jules
author_url: https://github.com/
version: 1.1.0
description: Integrates the self-hosted RAGgate stack (ingest-service) to provide semantic knowledge base search, web search via SearXNG, and the ability to propose new knowledge updates to the review queue.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import requests
import os
import json

class Tools:
    def __init__(self):
        # We try to load configuration from the environment, falling back to typical local defaults if used inside the same compose stack
        self.raggate_url = os.getenv("RAGGATE_API_URL", "http://ingest:8000")
        self.api_key = os.getenv("RAGGATE_API_KEY", "change-me")

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def search_knowledge_base(self, query: str, category: str = None) -> str:
        """
        Searches the internal knowledge base for information.

        :param query: The question or topic to search for.
        :param category: (Optional) A specific category to restrict the search to (e.g., 'it', 'science'). If not provided, searches the general collection.
        """
        try:
            url = f"{self.raggate_url}/search"
            params = {"q": query, "top_k": 5}
            if category:
                params["category"] = category

            response = requests.get(url, params=params, headers=self._headers(), timeout=15)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            if not results:
                return f"No relevant information found in the knowledge base for '{query}'."

            output = []
            for r in results:
                src = r.get('source', 'Unknown Source')
                txt = r.get('text', '')
                score = r.get('score', 0)
                output.append(f"[Source: {src} | Relevance: {score:.2f}]\n{txt}")

            return "\n\n---\n\n".join(output)

        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    def search_web(self, query: str) -> str:
        """
        Searches the live web for current information using the RAGgate web search pipeline (SearXNG -> Crawl4AI -> Evaluation).
        Use this when the knowledge base does not have the answer or you need up-to-date internet facts.

        :param query: The search query to execute on the web.
        """
        try:
            url = f"{self.raggate_url}/websearch"
            params = {"q": query, "top_k": 5}

            response = requests.get(url, params=params, headers=self._headers(), timeout=45)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            if not results:
                return f"No relevant web search results found for '{query}'."

            output = []
            for r in results:
                title = r.get('title', '')
                link = r.get('url', '')
                content = r.get('content', '')
                kp = r.get('key_points', [])

                block = f"Title: {title}\nURL: {link}\n"
                if kp:
                    block += "Key Points:\n" + "\n".join([f"- {k}" for k in kp])
                else:
                    block += f"Content Snippet:\n{content[:500]}..."
                output.append(block)

            return "\n\n---\n\n".join(output)

        except Exception as e:
            return f"Error executing web search: {str(e)}"

    def suggest_knowledge_update(self, text: str, source: str, category: str = None) -> str:
        """
        Suggests new information to be added to the knowledge base. It will be enriched, checked for duplicates, and put into a review queue if it conflicts.

        :param text: The detailed information or document text to add.
        :param source: The origin of this information (e.g., a URL, a user's name, or 'Chat Interaction').
        :param category: (Optional) The category to file this under (e.g., 'it', 'science').
        """
        try:
            url = f"{self.raggate_url}/enrich"
            payload = {
                "text": text,
                "source": source
            }
            if category:
                payload["category"] = category

            response = requests.post(url, json=payload, headers=self._headers(), timeout=30)
            response.raise_for_status()

            data = response.json()
            status = data.get("status", "UNKNOWN")

            if status == "NEW":
                return "Successfully processed and added new knowledge directly to the base."
            elif status == "DUPLICATE":
                return "This information was flagged as a duplicate or not worth keeping by the gatekeeper."
            elif status == "REVIEW":
                return "This information conflicts with or updates existing knowledge. It has been added to the Review Queue for human approval."
            else:
                return f"Processed with status: {status}"

        except Exception as e:
            return f"Error suggesting knowledge update: {str(e)}"

    def update_knowledge_document(self, source: str, change_description: str) -> str:
        """
        Korrigiert ein bestehendes Dokument in der Wissensdatenbank. Nutze dies NUR,
        wenn der Nutzer explizit bestehenden Inhalt ändern will, eine konkrete Quelldatei
        nennt (z.B. "README.md") und eine klare Änderungsbeschreibung vorliegt.
        Beispiel: "Ändere in README.md überall Python 3.10 auf 3.13".

        :param source: The original document source name.
        :param change_description: The description of what should be changed.
        """
        try:
            url = f"{self.raggate_url}/update-source"
            payload = {
                "source": source,
                "change": change_description
            }

            response = requests.post(url, json=payload, headers=self._headers(), timeout=60)

            # For 404 or others we want to catch the JSON message if possible
            if not response.ok:
                try:
                    data = response.json()
                    return f"Fehler bei der Änderung: {data.get('message', response.text)}"
                except:
                    response.raise_for_status()

            data = response.json()
            status = data.get("status", "UNKNOWN")

            if status == "updated":
                return f"Dokument '{source}' wurde erfolgreich aktualisiert (alte Chunks: {data.get('old_chunks')}, neue Chunks: {data.get('new_chunks')})."
            elif status == "not_found":
                return f"Fehler: Quelldatei '{source}' wurde nicht in der Wissensdatenbank gefunden."
            elif status == "error":
                return f"Änderung abgebrochen: {data.get('message', 'Unbekannter Fehler')}."
            else:
                return f"Unbekannter Status: {status}"

        except Exception as e:
            return f"Error updating knowledge document: {str(e)}"
