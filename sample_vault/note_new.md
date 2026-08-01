---
tags: [incremental, testing]
created: 2025-01-10
---
# Incremental Sync Testing

This is a brand new note added to verify that the incremental ingestion pipeline correctly detects and indexes new files without re-processing unchanged notes.

## How It Works

The manifest.json file records the last-modified timestamp of every indexed markdown file. On each run, the pipeline compares current file mtimes to the manifest and only processes files that are new or changed.

## Expected Behaviour

Files that have not changed since the last ingest are skipped entirely. This keeps re-indexing fast even for large vaults.
