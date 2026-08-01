---
tags: [test, dummy]
created: 2025-01-20
---
# Dummy Test Note

This note exists solely to verify that the Sync Vault button in the UI correctly detects new files and reports them in its summary. It can be deleted after testing.

## Why This Exists

The incremental sync pipeline reads each file's last-modified timestamp and compares it to the manifest. A file absent from the manifest is categorised as new and embedded on the next sync.
