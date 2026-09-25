from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_DIR / ".env"
INPUT_DIR = PROJECT_DIR / "input"
DEFAULT_INPUT_DIR = PROJECT_DIR / "output"
DEFAULT_OUTPUT_BASE_DIR = PROJECT_DIR / "exports"
DEFAULT_DATASET_CSV = INPUT_DIR / "hover_dataset_full.csv"

EXPORT_PROCESS_ENV = "EXPORT_PROCESS"
DEFAULT_EXPORT_PROCESS = "export limpo"

GRAPH_SIDES = (
    ("claim", "grafo_claim"),
    ("evidencia", "grafo_evidencia"),
)
QUALITY_REMOVED_NODE_TYPES = {
    "TextSummary",
    "DocumentChunk",
    "TextDocument",
}

CSV_FIELD_ALIASES = {
    "id": ("id", "claim_id"),
    "split": ("split",),
}
UNKNOWN_SPLIT = "unknown"


@dataclass(frozen=True)
class ProcessConfig:
    name: str
    slug: str
    removed_node_types: frozenset[str]


PROCESS_CONFIGS = {
    "export total": ProcessConfig(
        name="export total",
        slug="export-total",
        removed_node_types=frozenset(),
    ),
    "export com textdocument": ProcessConfig(
        name="export com textDocument",
        slug="export-com-textdocument",
        removed_node_types=frozenset({"TextSummary", "DocumentChunk"}),
    ),
    "export limpo": ProcessConfig(
        name="export limpo",
        slug="export-limpo",
        removed_node_types=frozenset(
            {"TextSummary", "DocumentChunk", "TextDocument"}
        ),
    ),
}

PROCESS_ALIASES = {
    "total": "export total",
    "export_total": "export total",
    "exporttotal": "export total",
    "com_textdocument": "export com textdocument",
    "textdocument": "export com textdocument",
    "comtextdocument": "export com textdocument",
    "export_com_textdocument": "export com textdocument",
    "exportcomtextdocument": "export com textdocument",
    "limpo": "export limpo",
    "export_limpo": "export limpo",
    "exportlimpo": "export limpo",
}

GRAPH_SCHEMA = pa.schema(
    [
        ("graph_id", pa.string()),
        ("json_file", pa.string()),
        ("row_number", pa.int64()),
        ("id", pa.string()),
        ("claim", pa.string()),
        ("evidencia", pa.string()),
        ("label", pa.string()),
        ("num_hops", pa.int64()),
        ("split", pa.string()),
        ("claim_num_nodes", pa.int64()),
        ("claim_num_edges", pa.int64()),
        ("evidencia_num_nodes", pa.int64()),
        ("evidencia_num_edges", pa.int64()),
    ]
)

NODE_SCHEMA = pa.schema(
    [
        ("graph_id", pa.string()),
        ("json_file", pa.string()),
        ("row_number", pa.int64()),
        ("id", pa.string()),
        ("label", pa.string()),
        ("split", pa.string()),
        ("graph_side", pa.string()),
        ("node_id", pa.int64()),
        ("node_original_id", pa.string()),
        ("node_type", pa.string()),
        ("text", pa.string()),
    ]
)

EDGE_SCHEMA = pa.schema(
    [
        ("graph_id", pa.string()),
        ("json_file", pa.string()),
        ("row_number", pa.int64()),
        ("id", pa.string()),
        ("label", pa.string()),
        ("split", pa.string()),
        ("graph_side", pa.string()),
        ("edge_id", pa.int64()),
        ("source", pa.int64()),
        ("target", pa.int64()),
        ("edge_type", pa.string()),
    ]
)


@dataclass(frozen=True)
class CsvSplitRow:
    item_id: str
    split: str


@dataclass(frozen=True)
class CsvSplitIndex:
    by_row: dict[int, CsvSplitRow]
    by_id: dict[str, str]
    ambiguous_ids: set[str]
    row_count: int
    rows_without_split: int


@dataclass(frozen=True)
class PreparedGraph:
    nodes: list[tuple[dict[str, Any], int]]
    edges: list[tuple[dict[str, Any], int, int]]
    removed_nodes: int
    removed_edges: int
    invalid_nodes: int
    invalid_edges: int


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    reasons: tuple[str, ...]


@dataclass
class ExportCounters:
    files_seen: int = 0
    files_exported: int = 0
    json_copied: int = 0
    graphs: int = 0
    nodes: int = 0
    edges: int = 0
    removed_nodes: int = 0
    removed_edges: int = 0
    invalid_nodes: int = 0
    invalid_edges: int = 0
    invalid_json: int = 0
    quality_rejected: int = 0
    unknown_split: int = 0
    split_from_csv_row: int = 0
    split_from_csv_id: int = 0
    split_from_json: int = 0
    csv_row_id_mismatch: int = 0
    split_counts: Counter[str] | None = None
    quality_rejection_reasons: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.split_counts is None:
            self.split_counts = Counter()
        if self.quality_rejection_reasons is None:
            self.quality_rejection_reasons = Counter()


class ParquetWriters:
    def __init__(self, output_dir: Path, compression: str | None) -> None:
        self.output_dir = output_dir
        self.compression = compression
        self.schemas = {
            "graphs": GRAPH_SCHEMA,
            "nodes": NODE_SCHEMA,
            "edges": EDGE_SCHEMA,
        }
        self.paths = {
            "graphs": output_dir / "graphs.parquet",
            "nodes": output_dir / "nodes.parquet",
            "edges": output_dir / "edges.parquet",
        }
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.row_counts = Counter()

    def __enter__(self) -> "ParquetWriters":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, schema in self.schemas.items():
            self.writers[name] = pq.ParquetWriter(
                self.paths[name],
                schema=schema,
                compression=self.compression,
            )
        return self

    def write(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=self.schemas[name])
        self.writers[name].write_table(table)
        self.row_counts[name] += table.num_rows

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if exc_type is None:
                for name, writer in self.writers.items():
                    if self.row_counts[name] == 0:
                        empty_table = pa.Table.from_pylist([], schema=self.schemas[name])
                        writer.write_table(empty_table)
        finally:
            for writer in self.writers.values():
                writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta JSONs HoVer aprovados na qualidade para JSON e Parquet."
    )
    parser.add_argument(
        "--process",
        default=None,
        help=(
            "Processo de exportacao. Se omitido, usa EXPORT_PROCESS da .env. "
            'Opcoes: "export total", "export com textDocument", "export limpo".'
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Pasta com os JSONs gerados. Padrao: output/.",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE_DIR,
        help="Pasta base das execucoes. Padrao: exports/.",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=DEFAULT_DATASET_CSV,
        help="CSV HoVer com a coluna split. Padrao: input/hover_dataset_full.csv.",
    )
    parser.add_argument(
        "--split-column",
        default="split",
        help="Nome da coluna de split no CSV. Padrao: split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Quantidade de JSONs processados por row group. Padrao: 1000.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
        help="Compressao dos Parquets. Padrao: zstd.",
    )
    parser.add_argument(
        "--fail-on-unknown-split",
        action="store_true",
        help="Falha se algum JSON aprovado ficar com split=unknown.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Mostra progresso a cada N JSONs lidos. Use 0 para silenciar. Padrao: 1000.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def configure_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_header(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def normalize_process_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_split(value: Any) -> str:
    split = normalize_text(value).strip().lower()
    if split == "validation":
        return "val"
    return split


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_DIR / path).resolve()


def resolve_dataset_csv(path: Path) -> Path:
    resolved = resolve_project_path(path)
    if resolved.exists():
        return resolved

    if resolved == DEFAULT_DATASET_CSV.resolve():
        csv_files = sorted(INPUT_DIR.glob("*.csv"))
        if len(csv_files) == 1:
            return csv_files[0].resolve()

    raise FileNotFoundError(f"CSV do dataset nao encontrado: {resolved}")


def resolve_process(value: str | None) -> ProcessConfig:
    raw_value = (value or "").strip() or DEFAULT_EXPORT_PROCESS
    process_key = normalize_process_key(raw_value)
    canonical_key = PROCESS_ALIASES.get(process_key, process_key.replace("_", " "))
    config = PROCESS_CONFIGS.get(canonical_key)
    if config is not None:
        return config

    valid = ", ".join(config.name for config in PROCESS_CONFIGS.values())
    raise ValueError(
        f"Processo de exportacao invalido: {raw_value!r}. Valores validos: {valid}."
    )


def csv_field(fieldnames: list[str] | None, canonical_name: str, aliases: tuple[str, ...]) -> str:
    if not fieldnames:
        raise RuntimeError("CSV header is empty.")

    normalized_to_original = {normalize_header(name): name for name in fieldnames}
    for alias in aliases:
        original = normalized_to_original.get(normalize_header(alias))
        if original is not None:
            return original

    available = ", ".join(fieldnames)
    raise RuntimeError(
        f"Missing expected CSV column for {canonical_name}: {', '.join(aliases)}. "
        f"Available: {available}"
    )


def load_csv_split_index(
    csv_path: Path,
    *,
    wanted_rows: set[int],
    split_column: str,
) -> CsvSplitIndex:
    configure_csv_field_limit()
    by_row: dict[int, CsvSplitRow] = {}
    id_to_splits: dict[str, set[str]] = defaultdict(set)
    rows_without_split = 0
    row_count = 0
    store_all_rows = not wanted_rows

    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        id_field = csv_field(reader.fieldnames, "id", CSV_FIELD_ALIASES["id"])
        split_field = csv_field(reader.fieldnames, "split", (split_column,))

        for row_number, row in enumerate(reader, start=1):
            row_count = row_number
            item_id = normalize_id(row.get(id_field))
            split = normalize_split(row.get(split_field))

            if not split:
                rows_without_split += 1

            if store_all_rows or row_number in wanted_rows:
                by_row[row_number] = CsvSplitRow(item_id=item_id, split=split)

            if item_id and split:
                id_to_splits[item_id].add(split)

    by_id = {
        item_id: next(iter(splits))
        for item_id, splits in id_to_splits.items()
        if len(splits) == 1
    }
    ambiguous_ids = {
        item_id for item_id, splits in id_to_splits.items() if len(splits) > 1
    }
    return CsvSplitIndex(
        by_row=by_row,
        by_id=by_id,
        ambiguous_ids=ambiguous_ids,
        row_count=row_count,
        rows_without_split=rows_without_split,
    )


def parse_row_number(json_file: Path) -> int | None:
    prefix = json_file.stem.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def payload_evidence_text(payload: dict[str, Any]) -> str:
    return normalize_text(payload.get("evidencia") or payload.get("evidence_text"))


def payload_num_hops(payload: dict[str, Any]) -> int | None:
    num_hops = parse_int(payload.get("num_hops"))
    if num_hops is not None:
        return num_hops

    hover_metadata = payload.get("hover_metadata")
    if isinstance(hover_metadata, dict):
        return parse_int(hover_metadata.get("num_hops"))

    return None


def list_json_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Pasta de JSONs nao encontrada: {input_dir}")
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"Nenhum arquivo *.json encontrado em {input_dir}")
    return json_files


def load_json(json_file: Path) -> dict[str, Any]:
    with json_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON precisa conter um objeto no topo: {json_file}")
    return payload


def graph_items(payload: dict[str, Any], graph_key: str) -> tuple[list[Any], list[Any]]:
    graph = payload.get(graph_key)
    if not isinstance(graph, dict):
        return [], []

    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    return nodes, edges


def prepare_graph(
    nodes: list[Any],
    edges: list[Any],
    *,
    removed_node_types: set[str] | frozenset[str],
    remove_orphan_edges: bool,
) -> PreparedGraph:
    prepared_nodes: list[tuple[dict[str, Any], int]] = []
    valid_node_ids: set[int] = set()
    removed_nodes = 0
    invalid_nodes = 0

    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            invalid_nodes += 1
            removed_nodes += 1
            continue

        node_type = normalize_text(node.get("type"))
        if node_type in removed_node_types:
            removed_nodes += 1
            continue

        node_id = parse_int(node.get("id"))
        if node_id is None:
            node_id = node_index

        prepared_nodes.append((node, node_id))
        valid_node_ids.add(node_id)

    prepared_edges: list[tuple[dict[str, Any], int, int]] = []
    removed_edges = 0
    invalid_edges = 0

    for edge in edges:
        if not isinstance(edge, dict):
            invalid_edges += 1
            removed_edges += 1
            continue

        source = parse_int(edge.get("source"))
        target = parse_int(edge.get("target"))
        if source is None or target is None:
            invalid_edges += 1
            removed_edges += 1
            continue

        if remove_orphan_edges and (
            source not in valid_node_ids or target not in valid_node_ids
        ):
            removed_edges += 1
            continue

        prepared_edges.append((edge, source, target))

    return PreparedGraph(
        nodes=prepared_nodes,
        edges=prepared_edges,
        removed_nodes=removed_nodes,
        removed_edges=removed_edges,
        invalid_nodes=invalid_nodes,
        invalid_edges=invalid_edges,
    )


def quality_check(payload: dict[str, Any]) -> QualityResult:
    reasons: list[str] = []

    if not normalize_text(payload.get("claim")).strip():
        reasons.append("sem_claim")
    if not payload_evidence_text(payload).strip():
        reasons.append("sem_evidencia")

    for graph_side, graph_key in GRAPH_SIDES:
        graph = payload.get(graph_key)
        if not isinstance(graph, dict):
            reasons.append(f"sem_{graph_key}")
            continue

        nodes, edges = graph_items(payload, graph_key)
        prepared = prepare_graph(
            nodes,
            edges,
            removed_node_types=QUALITY_REMOVED_NODE_TYPES,
            remove_orphan_edges=True,
        )
        if not prepared.nodes:
            reasons.append(f"{graph_side}_sem_nos_uteis")

    return QualityResult(passed=not reasons, reasons=tuple(reasons))


def append_prepared_graph_rows(
    *,
    prepared: PreparedGraph,
    graph_side: str,
    context: dict[str, Any],
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    counters: ExportCounters,
) -> tuple[int, int]:
    counters.removed_nodes += prepared.removed_nodes
    counters.removed_edges += prepared.removed_edges
    counters.invalid_nodes += prepared.invalid_nodes
    counters.invalid_edges += prepared.invalid_edges

    for node, node_id in prepared.nodes:
        node_rows.append(
            {
                **context,
                "graph_side": graph_side,
                "node_id": node_id,
                "node_original_id": normalize_text(node.get("original_id")),
                "node_type": normalize_text(node.get("type")),
                "text": normalize_text(node.get("text")),
            }
        )

    for edge_index, (edge, source, target) in enumerate(prepared.edges):
        edge_rows.append(
            {
                **context,
                "graph_side": graph_side,
                "edge_id": edge_index,
                "source": source,
                "target": target,
                "edge_type": normalize_text(edge.get("type")),
            }
        )

    return len(prepared.nodes), len(prepared.edges)


def split_for_payload(
    *,
    json_file: Path,
    row_number: int | None,
    item_id: str,
    payload: dict[str, Any],
    split_index: CsvSplitIndex,
    counters: ExportCounters,
) -> str:
    row = split_index.by_row.get(row_number) if row_number is not None else None
    if row is not None:
        if row.item_id and item_id and row.item_id != item_id:
            counters.csv_row_id_mismatch += 1
        elif row.split:
            counters.split_from_csv_row += 1
            return row.split

    if item_id in split_index.by_id:
        counters.split_from_csv_id += 1
        return split_index.by_id[item_id]

    split = normalize_split(payload.get("split"))
    if split:
        counters.split_from_json += 1
        return split

    counters.unknown_split += 1
    logging.debug("Split nao encontrado para %s", json_file.name)
    return UNKNOWN_SPLIT


def make_context(
    json_file: Path,
    payload: dict[str, Any],
    split_index: CsvSplitIndex,
    counters: ExportCounters,
) -> dict[str, Any]:
    row_number = parse_row_number(json_file)
    item_id = normalize_id(payload.get("id"))
    split = split_for_payload(
        json_file=json_file,
        row_number=row_number,
        item_id=item_id,
        payload=payload,
        split_index=split_index,
        counters=counters,
    )

    return {
        "graph_id": json_file.stem,
        "json_file": json_file.name,
        "row_number": row_number,
        "id": item_id,
        "label": normalize_text(payload.get("label")),
        "split": split,
    }


def create_run_dir(output_base_dir: Path, process: ProcessConfig) -> Path:
    output_base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{timestamp}_{process.slug}"
    run_dir = output_base_dir / base_name

    suffix = 2
    while run_dir.exists():
        run_dir = output_base_dir / f"{base_name}-{suffix}"
        suffix += 1

    run_dir.mkdir()
    return run_dir


def copy_accepted_json(json_file: Path, json_output_dir: Path) -> None:
    json_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_file, json_output_dir / json_file.name)


def export_batch(
    json_files: list[Path],
    *,
    process: ProcessConfig,
    split_index: CsvSplitIndex,
    writers: ParquetWriters,
    json_output_dir: Path,
    counters: ExportCounters,
) -> None:
    graph_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for json_file in json_files:
        counters.files_seen += 1
        try:
            payload = load_json(json_file)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            counters.invalid_json += 1
            logging.warning("JSON ignorado por erro de leitura: %s (%s)", json_file, error)
            continue

        quality = quality_check(payload)
        if not quality.passed:
            counters.quality_rejected += 1
            assert counters.quality_rejection_reasons is not None
            counters.quality_rejection_reasons.update(quality.reasons)
            logging.debug(
                "JSON reprovado na qualidade: %s (%s)",
                json_file.name,
                ", ".join(quality.reasons),
            )
            continue

        copy_accepted_json(json_file, json_output_dir)
        counters.json_copied += 1

        context = make_context(json_file, payload, split_index, counters)
        split = context["split"]
        assert counters.split_counts is not None
        counters.split_counts[split] += 1

        side_counts: dict[str, tuple[int, int]] = {}
        for graph_side, graph_key in GRAPH_SIDES:
            nodes, edges = graph_items(payload, graph_key)
            prepared = prepare_graph(
                nodes,
                edges,
                removed_node_types=process.removed_node_types,
                remove_orphan_edges=bool(process.removed_node_types),
            )
            side_counts[graph_side] = append_prepared_graph_rows(
                prepared=prepared,
                graph_side=graph_side,
                context=context,
                node_rows=node_rows,
                edge_rows=edge_rows,
                counters=counters,
            )

        claim_counts = side_counts["claim"]
        evidencia_counts = side_counts["evidencia"]
        graph_rows.append(
            {
                **context,
                "claim": normalize_text(payload.get("claim")),
                "evidencia": payload_evidence_text(payload),
                "claim_num_nodes": claim_counts[0],
                "claim_num_edges": claim_counts[1],
                "num_hops": payload_num_hops(payload),
                "evidencia_num_nodes": evidencia_counts[0],
                "evidencia_num_edges": evidencia_counts[1],
            }
        )

    writers.write("graphs", graph_rows)
    writers.write("nodes", node_rows)
    writers.write("edges", edge_rows)

    counters.files_exported += len(graph_rows)
    counters.graphs += len(graph_rows)
    counters.nodes += len(node_rows)
    counters.edges += len(edge_rows)


def batched(values: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size < 1:
        raise ValueError("--batch-size precisa ser maior que zero.")
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def main() -> int:
    load_dotenv(ENV_FILE, override=True)
    configure_logging()
    args = parse_args()

    process_value = args.process or os.getenv(EXPORT_PROCESS_ENV)
    process = resolve_process(process_value)
    compression = None if args.compression == "none" else args.compression

    input_dir = resolve_project_path(args.input_dir)
    output_base_dir = resolve_project_path(args.output_base_dir)
    dataset_csv = resolve_dataset_csv(args.dataset_csv)

    json_files = list_json_files(input_dir)
    wanted_rows = {
        row_number
        for row_number in (parse_row_number(json_file) for json_file in json_files)
        if row_number is not None
    }
    split_index = load_csv_split_index(
        dataset_csv,
        wanted_rows=wanted_rows,
        split_column=args.split_column,
    )
    missing_rows = len(wanted_rows.difference(split_index.by_row))

    run_dir = create_run_dir(output_base_dir, process)
    json_output_dir = run_dir / "json"
    parquet_output_dir = run_dir / "parquet"
    json_output_dir.mkdir(parents=True, exist_ok=True)
    parquet_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Processo selecionado: %s", process.name)
    logging.info("Pasta da execucao: %s", run_dir)
    logging.info(
        "CSV de split carregado: %s linhas, %s rows selecionadas, %s ids ambiguos, %s rows sem split",
        split_index.row_count,
        len(split_index.by_row),
        len(split_index.ambiguous_ids),
        split_index.rows_without_split,
    )
    if missing_rows:
        logging.warning(
            "%s row_numbers dos JSONs nao existem no CSV; usando fallback por id/JSON quando possivel",
            missing_rows,
        )

    counters = ExportCounters()
    logging.info("Lendo %s JSONs de %s", len(json_files), input_dir)
    logging.info(
        "Qualidade minima: claim, evidencia, grafo_claim e grafo_evidencia com nos uteis apos remover %s",
        sorted(QUALITY_REMOVED_NODE_TYPES),
    )
    logging.info("Tipos removidos no Parquet: %s", sorted(process.removed_node_types))

    with ParquetWriters(parquet_output_dir, compression=compression) as writers:
        for batch_index, batch in enumerate(batched(json_files, args.batch_size), start=1):
            export_batch(
                batch,
                process=process,
                split_index=split_index,
                writers=writers,
                json_output_dir=json_output_dir,
                counters=counters,
            )
            if args.progress_every and counters.files_seen % args.progress_every == 0:
                logging.info(
                    "Lidos %s JSONs em %s lotes; aprovados=%s; reprovados=%s",
                    counters.files_seen,
                    batch_index,
                    counters.files_exported,
                    counters.quality_rejected,
                )

    if args.fail_on_unknown_split and counters.unknown_split:
        raise ValueError(f"{counters.unknown_split} JSONs ficaram com split=unknown.")

    logging.info(
        "Exportacao concluida: lidos=%s, aprovados=%s, reprovados_qualidade=%s, invalidos=%s",
        counters.files_seen,
        counters.files_exported,
        counters.quality_rejected,
        counters.invalid_json,
    )
    logging.info("JSONs aprovados copiados em %s: %s", json_output_dir, counters.json_copied)
    logging.info(
        "Parquets gerados em %s: graphs=%s linhas, nodes=%s linhas, edges=%s linhas",
        parquet_output_dir,
        counters.graphs,
        counters.nodes,
        counters.edges,
    )
    logging.info("Distribuicao split: %s", dict(sorted(counters.split_counts.items())))
    logging.info(
        "Origem do split: csv_row=%s, csv_id=%s, json=%s, unknown=%s",
        counters.split_from_csv_row,
        counters.split_from_csv_id,
        counters.split_from_json,
        counters.unknown_split,
    )
    if counters.quality_rejection_reasons:
        logging.info(
            "Motivos de reprovacao: %s",
            dict(sorted(counters.quality_rejection_reasons.items())),
        )
    if counters.csv_row_id_mismatch:
        logging.warning(
            "%s JSONs tinham row_number com id diferente no CSV; use --overwrite no gerador se quiser atualizar os JSONs",
            counters.csv_row_id_mismatch,
        )
    if counters.removed_nodes or counters.removed_edges:
        logging.info(
            "Itens removidos no Parquet: nodes=%s, edges=%s",
            counters.removed_nodes,
            counters.removed_edges,
        )
    if counters.invalid_nodes or counters.invalid_edges:
        logging.warning(
            "Itens invalidos ignorados no Parquet: nodes=%s, edges=%s",
            counters.invalid_nodes,
            counters.invalid_edges,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
