from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import sys
import traceback
import unicodedata
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Mapping


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
LOGS_DIR = PROJECT_DIR / "logs"
COGNEE_INTERNAL_LOGS_DIR = LOGS_DIR / "cognee"
VENDOR_DIR = PROJECT_DIR / "vendor"
ENV_FILE = PROJECT_DIR / ".env"
TIKTOKEN_CACHE_DIR = PROJECT_DIR / ".tiktoken_cache"
COGNEE_WORK_DIR = PROJECT_DIR / "cognee_workspace"
COGNEE_SYSTEM_DIR = COGNEE_WORK_DIR / ".cognee_system"
COGNEE_DATA_DIR = COGNEE_WORK_DIR / ".data_storage"
COGNEE_CACHE_DIR = COGNEE_WORK_DIR / ".cognee_cache"
COGNEE_DATABASES_DIR = COGNEE_SYSTEM_DIR / "databases"
PETROBRAS_CERT_BUNDLE = PROJECT_DIR / "petrobras-cert-bundle.pem"
DATASET_SLUG = "hover"
DATASET_DISPLAY_NAME = "HoVer"

REQUIRED_ENV = (
    "LLM_MODEL",
    "LLM_ENDPOINT",
    "LLM_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_ENDPOINT",
    "EMBEDDING_API_KEY",
    "EMBEDDING_DIMENSIONS",
)

CSV_FIELD_ALIASES = {
    "id": ("id", "claim_id"),
    "split": ("split",),
    "claim": ("claim", "alegacao"),
    "evidencia": ("evidencia", "evidence_text", "evidence"),
    "label": ("label", "rotulo"),
}

OPTIONAL_CSV_FIELD_ALIASES = {
    "evidence_metadata": ("evidence",),
    "evidence_text_raw": ("evidence_text",),
    "evidence_annotation_id": ("evidence_annotation_id",),
    "evidence_id": ("evidence_id",),
    "evidence_wiki_url": ("evidence_wiki_url",),
    "evidence_sentence_id": ("evidence_sentence_id",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one JSON file per HoVer CSV row with Cognee graphs.",
    )
    parser.add_argument(
        "--input",
        dest="input_csv",
        type=Path,
        default=None,
        help="CSV file. Defaults to the only .csv found in input/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where one JSON file per item will be written.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="First CSV data row to process. Row 1 is the first row after the header.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to process.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate JSON files that already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an error payload for failed rows and continue processing.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per row before marking it as failed. Default: 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and show planned files without importing Cognee.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional Cognee chunk size override.",
    )
    parser.add_argument(
        "--chunks-per-batch",
        type=int,
        default=1,
        help="Number of chunks per Cognee cognify batch.",
    )
    parser.add_argument(
        "--data-per-batch",
        type=int,
        default=1,
        help="Number of data items per Cognee pipeline batch.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N processed rows.",
    )
    parser.add_argument(
        "--no-clean-between-graphs",
        action="store_true",
        help="Reuse Cognee state instead of pruning before each claim/evidence graph.",
    )
    parser.add_argument(
        "--preserve-cognee-workspace",
        action="store_true",
        help="Leave temporary Cognee databases in cognee_workspace/ after finishing.",
    )
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hover_cognee")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        LOGS_DIR / "gerar_grafos_hover.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def normalize_header(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def resolve_input_csv(input_csv: Path | None) -> Path:
    if input_csv is not None:
        path = input_csv
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        return path.resolve()

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv file found in {INPUT_DIR}")
    if len(csv_files) > 1:
        names = ", ".join(path.name for path in csv_files)
        raise RuntimeError(f"More than one CSV found in input/: {names}. Use --input.")
    return csv_files[0].resolve()


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def build_field_map(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise RuntimeError("CSV header is empty.")

    normalized_to_original = {normalize_header(name): name for name in fieldnames}
    field_map: dict[str, str] = {}

    for canonical_name, aliases in CSV_FIELD_ALIASES.items():
        for alias in aliases:
            original = normalized_to_original.get(normalize_header(alias))
            if original is not None:
                field_map[canonical_name] = original
                break

    missing = [name for name in CSV_FIELD_ALIASES if name not in field_map]
    if missing:
        available = ", ".join(fieldnames)
        raise RuntimeError(
            f"Missing expected CSV columns: {', '.join(missing)}. Available: {available}"
        )

    return field_map


def build_optional_field_map(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}

    normalized_to_original = {normalize_header(name): name for name in fieldnames}
    field_map: dict[str, str] = {}

    for canonical_name, aliases in OPTIONAL_CSV_FIELD_ALIASES.items():
        for alias in aliases:
            original = normalized_to_original.get(normalize_header(alias))
            if original is not None:
                field_map[canonical_name] = original
                break

    return field_map


def parse_json_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(string_list(item))
        return values
    return [json.dumps(value, ensure_ascii=False, sort_keys=True)]


def hover_metadata_from_row(
    row: Mapping[str, Any],
    optional_field_map: Mapping[str, str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    evidence_field = optional_field_map.get("evidence_metadata")
    if evidence_field:
        evidence_payload = parse_json_value(row.get(evidence_field))
        if isinstance(evidence_payload, Mapping):
            for key in ("supporting_facts", "num_hops"):
                value = evidence_payload.get(key)
                if value not in (None, ""):
                    metadata[key] = value
        elif isinstance(evidence_payload, list):
            metadata["evidence"] = evidence_payload

    for key in (
        "evidence_annotation_id",
        "evidence_id",
        "evidence_wiki_url",
        "evidence_sentence_id",
    ):
        field = optional_field_map.get(key)
        if not field:
            continue
        parsed = parse_json_value(row.get(field))
        if parsed not in (None, ""):
            metadata[key] = parsed

    return metadata


def _evidence_sets_from_value(value: str) -> list[tuple[Any, list[str]]]:
    """Parse evidence_text while preserving set boundaries and sentence order."""
    parsed = parse_json_value(value)

    if isinstance(parsed, list):
        result: list[tuple[Any, list[str]]] = []
        for index, evidence_set in enumerate(parsed):
            if isinstance(evidence_set, Mapping):
                sentences = string_list(
                    evidence_set.get("text")
                    or evidence_set.get("sentences")
                    or evidence_set.get("evidence")
                )
                result.append((evidence_set.get("set_id", index), sentences))
            else:
                result.append((index, string_list(evidence_set)))
        return result

    if isinstance(parsed, Mapping):
        sentences = string_list(
            parsed.get("text")
            or parsed.get("sentences")
            or parsed.get("evidence_text")
            or parsed.get("evidence")
        )
        return [(parsed.get("set_id", 0), sentences)]

    text = (value or "").strip()
    return [(0, [text] if text else [])]


def _supporting_fact_pages(value: Any) -> list[str]:
    """Return page titles from HoVer supporting_facts in their original order."""
    pages: list[str] = []
    if not isinstance(value, list):
        return pages

    for fact in value:
        if not isinstance(fact, (list, tuple)) or len(fact) < 2:
            continue
        page = str(fact[0] or "").strip()
        if page:
            pages.append(page)
    return pages


def evidence_text_to_plain_text(
    value: str,
    supporting_facts: Any = None,
) -> str:
    """Linearize HoVer evidence while keeping the Wikipedia page per sentence.

    The HoVer CSV stores resolved evidence sentences in the same order as
    ``supporting_facts``.  When those two sequences have the same length, each
    sentence sent to Cognee is prefixed with its gold source page:

        Page: <title>. Sentence: <text>

    This preserves page boundaries and reduces cross-page coreference errors.
    If the sequences do not align, the old sentence-only linearization is used
    rather than assigning a potentially wrong page to a sentence.
    """
    evidence_sets = _evidence_sets_from_value(value)
    pages = _supporting_fact_pages(supporting_facts)

    all_sentences: list[str] = []
    for _, sentences in evidence_sets:
        all_sentences.extend(sentence.strip() for sentence in sentences if sentence.strip())

    if not all_sentences:
        return (value or "").strip()

    if pages and len(pages) == len(all_sentences):
        rendered_sets: list[str] = []
        offset = 0

        for set_id, sentences in evidence_sets:
            clean_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            lines: list[str] = []

            for sentence in clean_sentences:
                page = pages[offset]
                lines.append(f"Page: {page}. Sentence: {sentence}")
                offset += 1

            if not lines:
                continue

            if len(evidence_sets) > 1:
                rendered_sets.append(f"Evidence set {set_id}:\n" + "\n".join(lines))
            else:
                rendered_sets.extend(lines)

        return "\n".join(rendered_sets).strip()

    # Safe fallback: preserve the evidence text but do not fabricate page links.
    sections: list[str] = []
    for index, (set_id, sentences) in enumerate(evidence_sets):
        joined = " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()
        if not joined:
            continue
        if len(evidence_sets) > 1:
            sections.append(f"Evidence set {set_id if set_id is not None else index}: {joined}")
        else:
            sections.append(joined)

    return "\n\n".join(sections)


def build_dataset_item(
    row: Mapping[str, str],
    field_map: Mapping[str, str],
    optional_field_map: Mapping[str, str],
) -> dict[str, Any]:
    item: dict[str, Any] = {
        canonical: (row.get(original) or "").strip()
        for canonical, original in field_map.items()
    }

    item["dataset"] = DATASET_SLUG

    # Parse metadata first because supporting_facts carries the page title for
    # each resolved HoVer sentence in evidence_text.
    hover_metadata = hover_metadata_from_row(row, optional_field_map)
    supporting_facts = hover_metadata.get("supporting_facts", [])

    item["evidencia"] = evidence_text_to_plain_text(
        item["evidencia"],
        supporting_facts=supporting_facts,
    )
    item["evidence_text"] = item["evidencia"]

    if hover_metadata:
        item["hover_metadata"] = hover_metadata
        if "num_hops" in hover_metadata:
            item["num_hops"] = hover_metadata["num_hops"]

    return item


def iter_dataset_rows(
    csv_path: Path,
    *,
    start_row: int,
    limit: int | None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    configure_csv_field_limit()

    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        field_map = build_field_map(reader.fieldnames)
        optional_field_map = build_optional_field_map(reader.fieldnames)

        yielded = 0
        for row_number, row in enumerate(reader, start=1):
            if row_number < start_row:
                continue

            item = build_dataset_item(row, field_map, optional_field_map)

            yield row_number, item
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def safe_stem(value: Any, *, default: str = "sem_id", max_length: int = 80) -> str:
    text = str(value or "").strip()
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9_.-]+", "_", ascii_text).strip("._-")
    if not ascii_text:
        ascii_text = default
    return ascii_text[:max_length]


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def output_path_for_item(output_dir: Path, row_number: int, item_id: str) -> Path:
    return output_dir / f"{row_number:07d}_{safe_stem(item_id)}.json"


def dataset_name(row_number: int, kind: str, text: str) -> str:
    return f"{DATASET_SLUG}_{kind}_{row_number:07d}_{short_hash(text)}"


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        raise FileNotFoundError(f".env not found: {env_file}")

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_env_file_fallback(env_file)
        return

    load_dotenv(env_file, override=True)


def load_env_file_fallback(env_file: Path) -> None:
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def configure_environment() -> str:
    if VENDOR_DIR.exists():
        vendor_path = str(VENDOR_DIR)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)

    load_env_file(ENV_FILE)

    for directory in (
        COGNEE_WORK_DIR,
        COGNEE_SYSTEM_DIR,
        COGNEE_DATA_DIR,
        COGNEE_CACHE_DIR,
        COGNEE_DATABASES_DIR,
        TIKTOKEN_CACHE_DIR,
        COGNEE_INTERNAL_LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    apply_project_runtime_env()

    return configure_ssl()


def apply_project_runtime_env() -> None:
    os.environ["DISABLE_AIOHTTP_TRANSPORT"] = "True"
    os.environ["SYSTEM_ROOT_DIRECTORY"] = str(COGNEE_SYSTEM_DIR)
    os.environ["DATA_ROOT_DIRECTORY"] = str(COGNEE_DATA_DIR)
    os.environ["CACHE_ROOT_DIRECTORY"] = str(COGNEE_CACHE_DIR)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)
    os.environ["COGNEE_LOGS_DIR"] = str(COGNEE_INTERNAL_LOGS_DIR)

    os.environ["DB_PROVIDER"] = "sqlite"
    os.environ["GRAPH_DATABASE_PROVIDER"] = "kuzu"
    os.environ["GRAPH_DATASET_DATABASE_HANDLER"] = "kuzu"
    os.environ["GRAPH_DATABASE_SUBPROCESS_ENABLED"] = "false"
    os.environ["VECTOR_DB_PROVIDER"] = "lancedb"
    os.environ["VECTOR_DATASET_DATABASE_HANDLER"] = "lancedb"
    os.environ["STORAGE_BACKEND"] = "local"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["REQUIRE_AUTHENTICATION"] = "false"


def configure_ssl() -> str:
    certificate_env_vars = (
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTPX_CA_BUNDLE",
    )
    for variable in certificate_env_vars:
        os.environ.pop(variable, None)

    if PETROBRAS_CERT_BUNDLE.exists():
        bundle = str(PETROBRAS_CERT_BUNDLE)
        for variable in certificate_env_vars:
            os.environ[variable] = bundle
        return "petrobras-cert-bundle.pem"

    try:
        import truststore
    except ImportError:
        return "python-default"

    truststore.inject_into_ssl()
    return "windows-certificate-store"


def validate_runtime_environment() -> None:
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing required variables in .env: " + ", ".join(missing))

    if not VENDOR_DIR.joinpath("cognee").exists():
        raise RuntimeError(f"Vendored Cognee package not found: {VENDOR_DIR / 'cognee'}")


def import_cognee(logger: logging.Logger):
    ssl_mode = configure_environment()
    validate_runtime_environment()

    try:
        import tiktoken

        tiktoken.get_encoding("cl100k_base")
    except ImportError as error:
        raise RuntimeError("tiktoken is required. Run pip install -r requirements.txt") from error

    import cognee

    apply_project_runtime_env()
    cognee.config.system_root_directory(str(COGNEE_SYSTEM_DIR))
    cognee.config.data_root_directory(str(COGNEE_DATA_DIR))
    cognee.config.set_graph_database_provider("kuzu")
    cognee.config.set_graph_database_subprocess_enabled(False)
    cognee.config.set_vector_db_provider("lancedb")

    logger.info("Cognee loaded from %s", Path(cognee.__file__).resolve())
    logger.info("SSL mode: %s", ssl_mode)
    logger.info("Cognee workspace: %s", COGNEE_WORK_DIR)
    return cognee


async def reset_cognee_state(cognee, logger: logging.Logger) -> None:
    logger.info("Pruning temporary Cognee state")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def graph_for_empty_text(reason: str = "empty text") -> dict[str, Any]:
    return {"nodes": [], "edges": []}


def value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def first_text_value(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = value_to_text(payload.get(key))
        if value:
            return value
    return ""


def endpoint_id(value: Any) -> str:
    if isinstance(value, Mapping):
        return value_to_text(value.get("id") or value.get("original_id") or value.get("name"))
    return value_to_text(value)


def compact_graph(nodes: list[Mapping[str, Any]], edges: list[Mapping[str, Any]]) -> dict[str, Any]:
    id_map: dict[str, int] = {}
    compact_nodes: list[dict[str, Any]] = []

    def add_node(original_id: str, raw_node: Mapping[str, Any] | None = None) -> int:
        if original_id in id_map:
            return id_map[original_id]

        node_index = len(compact_nodes)
        id_map[original_id] = node_index

        raw_node = raw_node or {}
        node_text = first_text_value(
            raw_node,
            ("name", "text", "content", "title", "description"),
        )
        if not node_text:
            node_text = original_id

        compact_nodes.append(
            {
                "id": node_index,
                "original_id": original_id,
                "text": node_text,
                "type": first_text_value(raw_node, ("type", "node_type", "label")) or "Unknown",
            }
        )
        return node_index

    for raw_node in nodes:
        original_id = first_text_value(raw_node, ("id", "node_id", "uuid"))
        if not original_id:
            original_id = f"node_{len(compact_nodes)}"
        add_node(original_id, raw_node)

    compact_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[int, int, str]] = set()

    for raw_edge in edges:
        source_original_id = endpoint_id(
            raw_edge.get("source_id")
            or raw_edge.get("source")
            or raw_edge.get("from")
            or raw_edge.get("start")
        )
        target_original_id = endpoint_id(
            raw_edge.get("target_id")
            or raw_edge.get("target")
            or raw_edge.get("to")
            or raw_edge.get("end")
        )
        if not source_original_id or not target_original_id:
            continue

        source = add_node(source_original_id)
        target = add_node(target_original_id)
        edge_type = first_text_value(raw_edge, ("relationship", "type", "label")) or "related_to"
        edge_identity = (source, target, edge_type)
        if edge_identity in seen_edges:
            continue
        seen_edges.add(edge_identity)

        compact_edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
            }
        )

    return {"nodes": compact_nodes, "edges": compact_edges}


def snapshot_to_graph(snapshot: Any) -> dict[str, Any]:
    payload = snapshot.model_dump(mode="json")
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []
    return compact_graph(nodes, edges)


async def generate_graph(
    cognee,
    *,
    text: str,
    dataset: str,
    logger: logging.Logger,
    clean_before: bool,
    chunk_size: int | None,
    chunks_per_batch: int,
    data_per_batch: int,
) -> dict[str, Any]:
    if not text.strip():
        return graph_for_empty_text()

    if clean_before:
        await reset_cognee_state(cognee, logger)

    logger.info("Adding dataset %s", dataset)
    await cognee.add(
        text,
        dataset_name=dataset,
        incremental_loading=False,
        data_per_batch=data_per_batch,
    )

    cognify_kwargs: dict[str, Any] = {
        "datasets": [dataset],
        "chunks_per_batch": chunks_per_batch,
        "data_per_batch": data_per_batch,
    }
    if chunk_size is not None:
        cognify_kwargs["chunk_size"] = chunk_size

    logger.info("Cognifying dataset %s", dataset)
    await cognee.cognify(**cognify_kwargs)

    logger.info("Exporting dataset %s", dataset)
    snapshot = await cognee.export(dataset, format="pydantic")
    return snapshot_to_graph(snapshot)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_exception_text(error: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(type(current).__name__)
        parts.append(str(current))
        parts.extend(str(arg) for arg in getattr(current, "args", ()))
        current = current.__cause__ or current.__context__

    return "\n".join(part for part in parts if part)


def content_filter_details(error: BaseException) -> list[dict[str, str]]:
    text = collect_exception_text(error)
    details: list[dict[str, str]] = []
    categories = ("hate", "jailbreak", "self_harm", "sexual", "violence")

    for category in categories:
        filtered_pattern = (
            rf"['\"]{category}['\"]\s*:\s*\{{[^}}]*"
            rf"['\"]filtered['\"]\s*:\s*True"
        )
        if not re.search(filtered_pattern, text, flags=re.IGNORECASE):
            continue

        severity = ""
        severity_match = re.search(
            rf"['\"]{category}['\"]\s*:\s*\{{[^}}]*"
            rf"['\"]severity['\"]\s*:\s*['\"]([^'\"]+)['\"]",
            text,
            flags=re.IGNORECASE,
        )
        if severity_match:
            severity = severity_match.group(1)

        details.append({"category": category, "severity": severity})

    return details


def is_content_filter_error(error: BaseException) -> bool:
    text = collect_exception_text(error).lower()
    markers = (
        "content_filter",
        "responsibleaipolicyviolation",
        "content management policy",
    )
    return any(marker in text for marker in markers)


def error_reason(error: BaseException) -> str:
    if not is_content_filter_error(error):
        return str(error)

    details = content_filter_details(error)
    if not details:
        return "Azure OpenAI content_filter bloqueou o item."

    categories = []
    for detail in details:
        if detail["severity"]:
            categories.append(f"{detail['category']} severity={detail['severity']}")
        else:
            categories.append(detail["category"])

    return "Azure OpenAI content_filter bloqueou o item: " + ", ".join(categories)


def base_payload_for_item(item: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": item.get("dataset") or DATASET_SLUG,
        "id": item["id"],
        "split": item["split"],
        "claim": item["claim"],
        "evidencia": item["evidencia"],
        "evidence_text": item.get("evidence_text") or item["evidencia"],
        "label": item["label"],
    }

    for key in ("num_hops", "hover_metadata"):
        value = item.get(key)
        if value not in (None, ""):
            payload[key] = value

    return payload


def build_error_payload(
    item: Mapping[str, Any],
    row_number: int,
    error: BaseException,
    attempts: int,
) -> dict[str, Any]:
    is_content_filter = is_content_filter_error(error)
    payload = base_payload_for_item(item)
    payload.update(
        {
            "grafo_claim": graph_for_empty_text("row failed before claim graph was completed"),
            "grafo_evidencia": graph_for_empty_text(
                "row failed before evidence graph was completed"
            ),
        }
    )
    payload["erro"] = {
        "row_number": row_number,
        "attempts": attempts,
        "category": "content_filter" if is_content_filter else "processing_error",
        "motivo": error_reason(error),
        "content_filter": {
            "blocked": True,
            "details": content_filter_details(error),
        }
        if is_content_filter
        else None,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    return payload


async def process_item(
    cognee,
    *,
    item: Mapping[str, Any],
    row_number: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, Any]:
    clean_between_graphs = not args.no_clean_between_graphs

    claim_graph = await generate_graph(
        cognee,
        text=item["claim"],
        dataset=dataset_name(row_number, "claim", item["claim"]),
        logger=logger,
        clean_before=clean_between_graphs,
        chunk_size=args.chunk_size,
        chunks_per_batch=args.chunks_per_batch,
        data_per_batch=args.data_per_batch,
    )

    evidence_graph = await generate_graph(
        cognee,
        text=item["evidencia"],
        dataset=dataset_name(row_number, "evidencia", item["evidencia"]),
        logger=logger,
        clean_before=clean_between_graphs,
        chunk_size=args.chunk_size,
        chunks_per_batch=args.chunks_per_batch,
        data_per_batch=args.data_per_batch,
    )

    payload = base_payload_for_item(item)
    payload.update(
        {
            "grafo_claim": claim_graph,
            "grafo_evidencia": evidence_graph,
        }
    )
    return payload


async def process_item_with_retries(
    cognee,
    *,
    item: Mapping[str, Any],
    row_number: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, Any]:
    for attempt in range(1, args.max_attempts + 1):
        try:
            if attempt > 1:
                logger.info(
                    "Retrying row %s id=%s attempt=%s/%s",
                    row_number,
                    item["id"],
                    attempt,
                    args.max_attempts,
                )
            return await process_item(
                cognee,
                item=item,
                row_number=row_number,
                args=args,
                logger=logger,
            )
        except Exception as error:
            setattr(error, "attempts", attempt)
            if is_content_filter_error(error):
                logger.warning(
                    "Content filter blocked row %s id=%s attempt=%s/%s; not retrying. Reason: %s",
                    row_number,
                    item["id"],
                    attempt,
                    args.max_attempts,
                    error_reason(error),
                    exc_info=True,
                )
                raise

            if attempt >= args.max_attempts:
                raise

            logger.warning(
                "Attempt %s/%s failed for row %s id=%s; cleaning Cognee state before retry",
                attempt,
                args.max_attempts,
                row_number,
                item["id"],
                exc_info=True,
            )
            try:
                await reset_cognee_state(cognee, logger)
            except Exception:
                logger.warning(
                    "Could not clean Cognee state after failed attempt for row %s id=%s",
                    row_number,
                    item["id"],
                    exc_info=True,
                )

    raise RuntimeError("Retry loop finished unexpectedly.")


async def process_csv(
    *,
    csv_path: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    cognee = import_cognee(logger)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0
    content_filtered = 0

    try:
        for row_number, item in iter_dataset_rows(
            csv_path,
            start_row=args.start_row,
            limit=args.limit,
        ):
            output_path = output_path_for_item(output_dir, row_number, item["id"])
            if output_path.exists() and not args.overwrite:
                skipped += 1
                logger.info("Skipping row %s because output exists: %s", row_number, output_path)
                continue

            logger.info("Processing row %s id=%s", row_number, item["id"])

            try:
                payload = await process_item_with_retries(
                    cognee,
                    item=item,
                    row_number=row_number,
                    args=args,
                    logger=logger,
                )
                write_json(output_path, payload)
                processed += 1
                logger.info("Wrote %s", output_path)
            except Exception as error:
                failed += 1
                attempts = int(getattr(error, "attempts", args.max_attempts))
                if is_content_filter_error(error):
                    content_filtered += 1
                    logger.warning(
                        "Saving content_filter error for row %s id=%s after %s attempt(s): %s",
                        row_number,
                        item["id"],
                        attempts,
                        error_reason(error),
                    )
                    write_json(
                        output_path,
                        build_error_payload(item, row_number, error, attempts),
                    )
                    continue

                logger.exception(
                    "Failed row %s id=%s after %s attempts",
                    row_number,
                    item["id"],
                    attempts,
                )
                if not args.continue_on_error:
                    raise
                write_json(
                    output_path,
                    build_error_payload(item, row_number, error, attempts),
                )

            total_done = processed + skipped + failed
            if args.progress_every > 0 and total_done % args.progress_every == 0:
                logger.info(
                    "Progress: processed=%s skipped=%s failed=%s content_filtered=%s",
                    processed,
                    skipped,
                    failed,
                    content_filtered,
                )
    finally:
        if not args.preserve_cognee_workspace:
            await reset_cognee_state(cognee, logger)

    logger.info(
        "Finished: processed=%s skipped=%s failed=%s content_filtered=%s output=%s",
        processed,
        skipped,
        failed,
        content_filtered,
        output_dir,
    )


def dry_run(
    *,
    csv_path: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    logger.info("Dry run only. Cognee will not be imported.")
    logger.info("Dataset: %s", DATASET_DISPLAY_NAME)
    logger.info("Project: %s", PROJECT_DIR)
    logger.info("Input CSV: %s", csv_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Vendored Cognee present: %s", VENDOR_DIR.joinpath("cognee").exists())
    logger.info(".env present: %s", ENV_FILE.exists())
    logger.info("PEM files: %s", len(list(PROJECT_DIR.glob("*.pem"))))

    preview_limit = args.limit if args.limit is not None else 3
    rows = list(
        islice(
            iter_dataset_rows(csv_path, start_row=args.start_row, limit=preview_limit),
            preview_limit,
        )
    )
    if not rows:
        logger.info("No rows selected.")
        return

    for row_number, item in rows:
        logger.info(
            "Would write row=%s id=%s split=%s label=%s claim_chars=%s evidence_chars=%s num_hops=%s path=%s",
            row_number,
            item["id"],
            item["split"],
            item["label"],
            len(item["claim"]),
            len(item["evidencia"]),
            item.get("num_hops", ""),
            output_path_for_item(output_dir, row_number, item["id"]),
        )


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    csv_path = resolve_input_csv(args.input_csv)

    if args.start_row < 1:
        raise ValueError("--start-row must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be >= 1")

    if args.dry_run:
        dry_run(csv_path=csv_path, args=args, logger=logger)
        return

    asyncio.run(process_csv(csv_path=csv_path, args=args, logger=logger))


if __name__ == "__main__":
    main()
