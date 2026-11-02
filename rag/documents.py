"""Document loading and chunking for uploaded files.

Extracted from main.py unchanged apart from imports; the `.eml` loader is the
only non-obvious part (it flattens HTML parts and skips attachments).
"""

from __future__ import annotations

import html
import logging
from email import message_from_binary_file
from email.policy import default
from html.parser import HTMLParser
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

logger = logging.getLogger("rag.documents")


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )


class _TextExtractor(HTMLParser):
    """Flatten HTML to text, skipping script/style content."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str):
        if self._skip:
            return
        text = data.strip()
        if text:
            self.parts.append(text)


def load_eml_file(file_path: str) -> list[Document]:
    """Load an email's headers and text body."""
    with open(file_path, "rb") as fh:
        msg = message_from_binary_file(fh, policy=default)

    body_parts: list[str] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type not in {"text/plain", "text/html"}:
            continue
        try:
            content = part.get_content()
        except Exception:
            # get_payload(decode=True) is typed as returning str | bytes | None
            # and does return str for parts that are not a simple decoded leaf.
            # Calling .decode() on that raised AttributeError, so the fallback
            # path could fail on exactly the malformed messages it exists for.
            payload = part.get_payload(decode=True) or b""
            if isinstance(payload, bytes):
                charset = part.get_content_charset() or "utf-8"
                content = payload.decode(charset, errors="replace")
            else:
                content = str(payload)
        if not isinstance(content, str):
            continue
        if content_type == "text/html":
            extractor = _TextExtractor()
            extractor.feed(content)
            content = html.unescape(" ".join(extractor.parts))
        if content.strip():
            body_parts.append(content.strip())

    headers = [
        f"Subject: {msg.get('Subject', '(no subject)')}",
        f"From: {msg.get('From', 'Unknown')}",
        f"To: {msg.get('To', 'Unknown')}",
        f"Date: {msg.get('Date', 'Unknown')}",
    ]
    text = "\n".join(headers) + "\n\n" + ("\n".join(body_parts) if body_parts else "(empty body)")
    return [Document(page_content=text, metadata={"source": Path(file_path).name, "page": 0})]


def process_and_chunk_file(file_path: str, extension: str) -> list[Document]:
    if extension == ".pdf":
        documents = PyPDFLoader(file_path).load()
    elif extension == ".txt":
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        documents = [Document(page_content=text, metadata={"page": 0})]
    elif extension == ".docx":
        documents = Docx2txtLoader(file_path).load()
    elif extension == ".eml":
        documents = load_eml_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return make_splitter().split_documents(documents)
