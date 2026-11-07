"""Document loading and chunking for uploaded files.

Extracted from main.py unchanged apart from imports; the `.eml` loader is the
only non-obvious part. Two long-standing defects in it are fixed here:

* **multipart/alternative was indexed twice.** Those messages carry the same
  body as both `text/plain` and `text/html`, and the loader appended both. Every
  ordinary email therefore got double the chunks and near-duplicate retrieval
  hits. The plain part now wins when one exists.
* **entities were decoded twice.** `HTMLParser` is constructed with
  `convert_charrefs=True` (the default), so `handle_data` already receives
  decoded text. The `html.unescape()` applied afterwards decoded it a second
  time, turning a literal `&amp;lt;` into `<`.
"""

from __future__ import annotations

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
        # convert_charrefs defaults to True, so handle_data receives decoded
        # text. Do not unescape again afterwards.
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


def _part_text(part) -> str | None:
    """Decoded text of one MIME part, or None if it has no usable body.

    `get_content()` is typed as returning str | bytes | None and does return a
    str for parts that are not a simple decoded leaf. The old fallback called
    `.decode()` on that str, so it raised AttributeError on exactly the
    malformed messages it existed to handle.
    """
    try:
        content = part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        if isinstance(payload, bytes):
            charset = part.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
        return str(payload)

    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        return content.decode(part.get_content_charset() or "utf-8", errors="replace")
    return None


def _html_to_text(content: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(content)
    return " ".join(extractor.parts).strip()


def _body_parts(msg) -> list[str]:
    """Body text to index, preferring the plain alternative.

    multipart/alternative carries the same content twice; appending both
    doubled every email. Plain wins, and HTML is only flattened when the
    message has no plain part at all.
    """
    plain: list[str] = []
    html_parts: list[str] = []

    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type not in {"text/plain", "text/html"}:
            continue
        content = _part_text(part)
        if not content or not content.strip():
            continue
        if content_type == "text/plain":
            plain.append(content.strip())
        else:
            html_parts.append(content)

    if plain:
        return plain

    flattened: list[str] = []
    for content in html_parts:
        text = _html_to_text(content)
        if text:
            flattened.append(text)
    return flattened


def load_eml_file(file_path: str) -> list[Document]:
    """Load an email's headers and text body."""
    with open(file_path, "rb") as fh:
        msg = message_from_binary_file(fh, policy=default)

    body_parts = _body_parts(msg)

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
