---
tags: [setup, configuration]
aliases: [Setup, Config]
created: 2025-01-03
---
# Local Setup Configuration

This note tracks the environment details for running the vault assistant entirely on my own laptop. It has been expanded with more detail.

## Environment Variables

The vault path and the Anthropic API key are stored in a file named dot env, which is never committed to version control. The embedding model itself needs no API key since it runs locally.

## Where Things Live

The vector database is a folder called chroma db sitting next to the python scripts. It is not a server, just files on disk that get opened directly by whichever script needs them. See [[note2]] for how those files get populated in the first place.

## Model Details

The embedding model used is all-MiniLM-L6-v2 from the sentence-transformers library. It maps text to 384-dimensional dense vectors, runs entirely on CPU, and requires no internet connection after the initial download.

## Startup Sequence

On first run, ingest.py must be executed before query.py. After the initial full ingest, subsequent runs use the incremental sync which only re-processes changed or new files.