"""
ingest.py — Incremental vault ingestion pipeline.

Key behaviours
--------------
* Maintains a manifest.json alongside chroma_db to track each file's
  last-modified timestamp and the chunk-ids written for it.
* On each run only re-embeds files that are new or changed.
* For changed files: deletes ALL existing Chroma chunks whose
  metadata["source"] matches the file's relative path BEFORE inserting
  new chunks, so stale chunks from old heading structures never linger.
* Deleted files: same metadata-filter delete + manifest cleanup.
* --full flag: wipes the entire collection and re-ingests from scratch.
* run_incremental_sync(config, embeddings, db) is an importable function
  that returns a summary dict, so main.py's /sync endpoint can call it.
"""

import os
import sys
import json
import logging
import hashlib
import argparse
from dataclasses import dataclass
from typing import Generator, List, Dict, Tuple, Optional, Any

from dotenv import load_dotenv
import frontmatter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding import (with fallback)
# ---------------------------------------------------------------------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
    except ImportError:
        logger.error("Required embedding packages are missing. Install langchain-huggingface.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexerConfig:
    """Holds configuration parameters securely."""
    vault_path: str
    db_directory: str
    collection_name: str
    embedding_model: str
    batch_size: int
    min_char_length: int

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.db_directory, "manifest.json")

    @classmethod
    def load_from_env(cls) -> "IndexerConfig":
        load_dotenv()
        vault = (os.getenv("VAULT_PATH") or "./sample_vault").strip()
        if not vault or not os.path.exists(vault) or not os.path.isdir(vault):
            raise ValueError(f"Invalid or missing vault directory: '{vault}'")
        return cls(
            vault_path=vault,
            db_directory=os.getenv("DB_DIR", "./chroma_db"),
            collection_name=os.getenv("COLLECTION_NAME", "vault_collection"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            batch_size=int(os.getenv("BATCH_SIZE", "100")),
            min_char_length=int(os.getenv("MIN_CHAR_LENGTH", "50")),
        )


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def load_manifest(path: str) -> Dict[str, Any]:
    """Return the manifest dict, or an empty dict if it doesn't exist yet."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read manifest, starting fresh: {e}")
    return {}


def save_manifest(path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Vault scanner
# ---------------------------------------------------------------------------
class VaultScanner:
    """Yields all .md files in the vault, skipping hidden dirs."""

    def __init__(self, vault_path: str):
        self.vault_path = vault_path

    def iter_markdown_files(self) -> Generator[str, None, None]:
        for root, dirs, files in os.walk(self.vault_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != ".obsidian"]
            for file in files:
                if file.endswith(".md"):
                    yield os.path.join(root, file)

    def scan(self) -> Dict[str, float]:
        """Return {rel_path: mtime} for every markdown file in the vault."""
        result: Dict[str, float] = {}
        for abs_path in self.iter_markdown_files():
            rel = os.path.relpath(abs_path, self.vault_path).replace("\\", "/")
            result[rel] = os.path.getmtime(abs_path)
        return result


# ---------------------------------------------------------------------------
# Note parser
# ---------------------------------------------------------------------------
class NoteParser:
    """Transforms raw markdown files into LangChain Documents with chunk IDs."""

    def __init__(self, config: IndexerConfig):
        self.config = config
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
                ("######", "Header 6"),
            ]
        )

    def _parse_metadata_field(self, field: Any) -> str:
        if isinstance(field, list):
            return ", ".join(str(i) for i in field)
        return str(field) if field is not None else ""

    def process_file(self, filepath: str) -> List[Tuple[str, Document]]:
        """Parse a single file → list of (chunk_id, Document)."""
        rel_path = os.path.relpath(filepath, self.config.vault_path).replace("\\", "/")
        note_title = os.path.splitext(os.path.basename(filepath))[0]

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Skipping {rel_path}: cannot read file — {e}")
            return []

        try:
            post = frontmatter.loads(content)
            body = post.content.strip()
            fm_meta = post.metadata
        except Exception as e:
            logger.warning(f"Skipping {rel_path}: frontmatter parse error — {e}")
            return []

        if len(body) < self.config.min_char_length:
            logger.info(f"Skipping {rel_path}: below character threshold.")
            return []

        base_metadata = {
            "source": rel_path,
            "title": note_title,
            "tags": self._parse_metadata_field(fm_meta.get("tags")),
            "aliases": self._parse_metadata_field(fm_meta.get("aliases")),
            "created": self._parse_metadata_field(fm_meta.get("created")),
            "last_modified": os.path.getmtime(filepath) if os.path.exists(filepath) else 0.0,
        }

        splits = self.splitter.split_text(body)
        if not splits:
            splits = [Document(page_content=body, metadata={})]

        chunks: List[Tuple[str, Document]] = []
        header_mapping = [
            ("Header 1", "#"),
            ("Header 2", "##"),
            ("Header 3", "###"),
            ("Header 4", "####"),
            ("Header 5", "#####"),
            ("Header 6", "######"),
        ]

        for idx, split in enumerate(splits):
            meta = base_metadata.copy()
            meta.update(split.metadata)

            # Reconstruct exact Markdown header hierarchy (# Header 1, ## Header 2, etc.)
            header_lines = []
            for h_key, prefix in header_mapping:
                if h_key in split.metadata and split.metadata[h_key]:
                    header_lines.append(f"{prefix} {split.metadata[h_key]}")

            if header_lines:
                headers_block = "\n".join(header_lines)
                content_with_header = f"{headers_block}\n\n{split.page_content}"
            else:
                content_with_header = split.page_content

            chunk_id = hashlib.sha256(f"{rel_path}_{idx}".encode()).hexdigest()
            chunks.append((chunk_id, Document(page_content=content_with_header, metadata=meta)))

        return chunks


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------
def delete_chunks_for_source(db: Chroma, source_rel_path: str) -> int:
    """
    Delete every chunk in the collection whose metadata['source'] matches
    source_rel_path. Returns the number of chunks deleted.
    """
    try:
        result = db.get(where={"source": source_rel_path})
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
            db.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for source: {source_rel_path}")
        return len(ids_to_delete)
    except Exception as e:
        logger.error(f"Failed to delete chunks for {source_rel_path}: {e}")
        return 0


def upsert_chunks(db: Chroma, chunks: List[Tuple[str, Document]], batch_size: int) -> None:
    """Batch-insert chunks into Chroma."""
    pending = list(chunks)
    while pending:
        batch = pending[:batch_size]
        pending = pending[batch_size:]
        ids, docs = zip(*batch)
        try:
            db.add_documents(documents=list(docs), ids=list(ids))
            logger.info(f"Upserted batch of {len(docs)} chunks.")
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")


# ---------------------------------------------------------------------------
# Core incremental sync — importable function
# ---------------------------------------------------------------------------
def run_incremental_sync(
    config: IndexerConfig,
    embeddings: Optional[SentenceTransformerEmbeddings] = None,
    db: Optional[Chroma] = None,
    full: bool = False,
) -> Dict[str, int]:
    """
    Perform an incremental (or full) sync of the vault into Chroma.

    Parameters
    ----------
    config    : IndexerConfig
    embeddings: Pre-built embeddings instance (created here if None).
    db        : Pre-built Chroma instance (created here if None).
    full      : If True, wipe the collection and re-embed everything.

    Returns
    -------
    dict with keys: added, changed, deleted, unchanged
    """
    # Build shared resources if not provided
    if embeddings is None:
        embeddings = SentenceTransformerEmbeddings(model_name=config.embedding_model)
    if db is None:
        db = Chroma(
            collection_name=config.collection_name,
            embedding_function=embeddings,
            persist_directory=config.db_directory,
        )

    parser = NoteParser(config)
    scanner = VaultScanner(config.vault_path)

    # --full: wipe everything and rebuild from scratch
    if full:
        logger.info("--full flag set: wiping entire collection before re-ingestion.")
        try:
            db.delete_collection()
            # Re-open collection after deletion
            db = Chroma(
                collection_name=config.collection_name,
                embedding_function=embeddings,
                persist_directory=config.db_directory,
            )
        except Exception as e:
            logger.error(f"Failed to wipe collection: {e}")
        manifest: Dict[str, Any] = {}
    else:
        manifest = load_manifest(config.manifest_path)

    # Scan current vault files
    current_files: Dict[str, float] = scanner.scan()  # {rel_path: mtime}
    manifest_files = set(manifest.keys())
    vault_files = set(current_files.keys())

    # Categorise
    new_files = vault_files - manifest_files
    deleted_files = manifest_files - vault_files
    existing_files = vault_files & manifest_files

    changed_files = {
        rel for rel in existing_files
        if current_files[rel] != manifest[rel].get("mtime", -1)
    }
    unchanged_files = existing_files - changed_files

    summary = {
        "added": len(new_files),
        "changed": len(changed_files),
        "deleted": len(deleted_files),
        "unchanged": len(unchanged_files),
    }

    # ------------------------------------------------------------------
    # 1. Delete chunks for removed files
    # ------------------------------------------------------------------
    for rel in deleted_files:
        delete_chunks_for_source(db, rel)
        del manifest[rel]
        logger.info(f"Removed from manifest: {rel}")

    # ------------------------------------------------------------------
    # 2. Changed files — delete old chunks first, then re-embed
    # ------------------------------------------------------------------
    for rel in changed_files:
        abs_path = os.path.join(config.vault_path, rel.replace("/", os.sep))
        delete_chunks_for_source(db, rel)
        chunks = parser.process_file(abs_path)
        if chunks:
            upsert_chunks(db, chunks, config.batch_size)
            chunk_ids = [cid for cid, _ in chunks]
            manifest[rel] = {"mtime": current_files[rel], "chunk_ids": chunk_ids}
            logger.info(f"Re-indexed changed file: {rel} ({len(chunks)} chunks)")
        else:
            # File became too short / unreadable — remove from manifest
            manifest.pop(rel, None)

    # ------------------------------------------------------------------
    # 3. New files — embed and add to manifest
    # ------------------------------------------------------------------
    for rel in new_files:
        abs_path = os.path.join(config.vault_path, rel.replace("/", os.sep))
        chunks = parser.process_file(abs_path)
        if chunks:
            upsert_chunks(db, chunks, config.batch_size)
            chunk_ids = [cid for cid, _ in chunks]
            manifest[rel] = {"mtime": current_files[rel], "chunk_ids": chunk_ids}
            logger.info(f"Indexed new file: {rel} ({len(chunks)} chunks)")
        else:
            logger.info(f"Skipped new file (no usable content): {rel}")
            summary["added"] -= 1  # didn't actually index it

    # ------------------------------------------------------------------
    # 4. Unchanged files — nothing to do
    # ------------------------------------------------------------------
    if unchanged_files:
        logger.info(f"Skipped {len(unchanged_files)} unchanged file(s).")

    # Persist updated manifest
    save_manifest(config.manifest_path, manifest)

    logger.info("=" * 40)
    logger.info("SYNC SUMMARY")
    logger.info(f"  Added:     {summary['added']}")
    logger.info(f"  Changed:   {summary['changed']}")
    logger.info(f"  Deleted:   {summary['deleted']}")
    logger.info(f"  Unchanged: {summary['unchanged']}")
    logger.info("=" * 40)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Vault RAG — incremental ingestion")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Wipe the entire Chroma collection and re-embed everything from scratch.",
    )
    args = parser.parse_args()

    try:
        config = IndexerConfig.load_from_env()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    summary = run_incremental_sync(config, full=args.full)

    print("\nSync complete:")
    print(f"  Files added:     {summary['added']}")
    print(f"  Files changed:   {summary['changed']}")
    print(f"  Files deleted:   {summary['deleted']}")
    print(f"  Files unchanged: {summary['unchanged']}")


if __name__ == "__main__":
    main()