# Project Context: vault-rag

Read this file before doing anything else in this project.

This is a personal RAG (Retrieval-Augmented Generation) assistant over an Obsidian vault. It is designed to run locally, allowing the user to index and query their own markdown notes using vector database technology and LLMs.

## Hard Constraints
- **Vector Storage**: Must use a local Chroma instance persisted at `./chroma_db` (never a cloud vector DB).
- **Embeddings**: Generated locally via `sentence-transformers` (never a paid embedding API).
- **Generative LLM**: The only paid API call in this project is the final answer generation via the Anthropic Claude API.
- **Vault Location**: The Obsidian vault path must be read from the `.env` file as `VAULT_PATH`.
- **Exclusions**: The `.obsidian` folder and any dotfiles/folders inside the vault must always be excluded from processing.
- **File Types**: Only `.md` files are processed; other vault files (such as images, PDFs, `.canvas` files, etc.) must be skipped.

## Environment & Python Version
- **Python Version**: 3.11
- **Virtual Environment**: Use a `.venv` folder for virtual environment packages.
