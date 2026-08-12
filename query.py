import re
import os
import sys
import logging
from typing import List, Tuple, Set

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from llmProvider import LLMConfig, get_chat_model, generate_answer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_wikilinks(text: str) -> List[str]:
    """Extract unique Obsidian [[WikiLink]] target note titles from text."""
    pattern = r"\[\[([^\]\|#]+)(?:#[^\]\|]+)?(?:\|[^\]]+)?\]\]"
    matches = re.findall(pattern, text)
    seen: Set[str] = set()
    result: List[str] = []
    for match in matches:
        cleaned = match.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def expand_linked_documents(
    db: Chroma, primary_results: List[Tuple[Document, float]], max_linked: int = 10
) -> List[Tuple[Document, float]]:
    """
    Scan primary search results for [[WikiLinks]] and fetch the target notes
    from Chroma. Appends resolved documents to the context list.
    """
    existing_titles = {doc.metadata.get("title", "") for doc, _ in primary_results}
    linked_results: List[Tuple[Document, float]] = []

    for doc, _score in primary_results:
        parent_title = doc.metadata.get("title", "unknown")
        links = extract_wikilinks(doc.page_content)

        for link_target in links:
            if link_target in existing_titles:
                continue
            existing_titles.add(link_target)

            try:
                found = db.get(where={"title": link_target})
                if found and found.get("documents"):
                    docs_list = found.get("documents", [])
                    metas_list = found.get("metadatas", [])
                    for c_text, c_meta in zip(docs_list, metas_list):
                        meta_copy = dict(c_meta) if c_meta else {}
                        meta_copy["via_link_from"] = parent_title
                        linked_results.append((Document(page_content=c_text, metadata=meta_copy), 0.0))
                        if len(linked_results) >= max_linked:
                            break
            except Exception as e:
                logger.warning(f"Could not resolve linked note '{link_target}': {e}")

            if len(linked_results) >= max_linked:
                break

    if linked_results:
        logger.info(f"WikiLink resolution expanded search with {len(linked_results)} linked chunk(s).")

    return primary_results + linked_results


def load_env() -> Tuple[str, str, str, LLMConfig]:
    """Load environment variables and LLM configuration."""
    load_dotenv()

    db_dir = os.getenv("DB_DIR", "./chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "vault_collection")
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    llm_config = LLMConfig.load_from_env()

    return db_dir, collection_name, embedding_model, llm_config


def get_vector_store(db_dir: str, collection_name: str, embeddings) -> Chroma:
    """Initialize connection with provided embeddings and validate."""
    if not os.path.isdir(db_dir) or not os.listdir(db_dir):
        raise RuntimeError("No data directory found — run ingest.py first")

    db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=db_dir)

    try:
        db_data = db.get()
        if not db_data or not db_data.get("ids"):
            raise RuntimeError("Vector database collection is empty — run ingest.py first")
    except Exception as e:
        logger.error(f"Deep database storage validation failed: {e}")
        raise

    return db


def similarity_search(
    db: Chroma, query: str, k: int = 5, expand_links: bool = True
) -> List[Tuple[Document, float]]:
    """Query the vector database and return matching items with optional WikiLink expansion."""
    try:
        results = db.similarity_search_with_score(query, k=k)
        if not results:
            logger.info("No results returned from vector store.")
            return []

        if expand_links:
            results = expand_linked_documents(db, results)

        return results
    except Exception as e:
        logger.error(f"Vector store query execution failed: {e}")
        return []


def build_prompt_messages(question: str, excerpts: List[Tuple[Document, float]]):
    """Isolates operational rules natively inside the LLM's System frame."""
    excerpt_texts = []
    for doc, _score in excerpts:
        title = doc.metadata.get("title", "(untitled)")
        source = doc.metadata.get("source", "unknown")
        via_link = doc.metadata.get("via_link_from")
        excerpt = doc.page_content.strip()

        if via_link:
            header_info = f"Title: {title} (Source: {source}, Resolved via [[WikiLink]] from: {via_link})"
        else:
            header_info = f"Title: {title} (Source: {source})"

        excerpt_texts.append(f"{header_info}\nExcerpt:\n{excerpt}")

    excerpts_block = "\n---\n".join(excerpt_texts)

    system_prompt = (
        "You are a helpful personal assistant. Answer the user's question **using ONLY** the provided note excerpts.\n"
        "Strictly adhere to the following operational boundaries:\n"
        "1. Do not use outside knowledge or extrapolate past details not explicitly mentioned in the excerpts.\n"
        "2. If the excerpts do not contain enough specific details to confidently answer, reply EXACTLY with: "
        "'I couldn't find anything in your notes about that.'\n"
        "3. If you can answer, keep it concise. At the absolute end of your response, create a dedicated section "
        "labeled 'Sources Used:' followed by a clean bulleted list showing only the unique note titles you referenced.\n"
        "4. Critical: If you output the refusal string because the notes are irrelevant, do NOT output a sources section.\n\n"
        f"Context Note Excerpts:\n{excerpts_block}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    return prompt_template.format_messages(question=question)


def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\" [k]")
        sys.exit(1)

    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # Load configuration and create a single embeddings instance & chat model instance
    db_dir, collection_name, embedding_model, llm_config = load_env()
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    try:
        db = get_vector_store(db_dir, collection_name, embeddings)
    except RuntimeError as e:
        print(str(e))
        sys.exit(0)

    results = similarity_search(db, question, k=k)
    if not results:
        print("I couldn't find anything in your notes about that.")
        sys.exit(0)

    # 1. Print raw developer diagnostics (Under the hood monitoring)
    print(f"\nQuery: {question}\n{'-' * 60}")
    for idx, (doc, score) in enumerate(results, 1):
        title = doc.metadata.get("title", "(untitled)")
        source = doc.metadata.get("source", "unknown")
        print(f"{idx}. Title: {title} (Source: {source})")
        print(f"   Similarity Distance Score: {score:.4f}")
        print(f"   Excerpt: {doc.page_content.strip()[:200]}...\n")

    # 2. Build Chat Messages and execute model pipeline via llmProvider
    prompt_messages = build_prompt_messages(question, results)
    chat_model = get_chat_model(llm_config)
    answer = generate_answer(chat_model, prompt_messages, llm_config.provider, llm_config.model)

    print("\n" + "="*20 + f" Answer from {llm_config.provider.upper()} ({llm_config.model}) " + "="*20 + "\n")
    print(answer)
    print("\n" + "="*60)


if __name__ == "__main__":
    main()