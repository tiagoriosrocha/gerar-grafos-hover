"""Exporta registros JSON validados para JSON Lines com auditoria."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


EXPORTER_VERSION = "2.1.0"
MAX_JSONL_BYTES = 99_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exports"
VALID_SPLITS = {"train", "validation", "test", "pool"}
DATASET_LABELS = {
    "fever": {"SUPPORTS", "REFUTES"},
    "feverous": {"SUPPORTS", "REFUTES"},
    "factkg": {"SUPPORTS", "REFUTES"},
    "hover": {"SUPPORTED", "NOT_SUPPORTED"},
    "scifact": {"SUPPORT", "CONTRADICT"},
}
SOURCE_AUDIT_FIELDS = (
    "id",
    "dataset",
    "claim",
    "evidencia/evidence_text",
    "label",
    "split",
    "evaluation_role",
    "group_id",
    "synthetic",
    "evidence_complete",
    "grafo_claim",
    "grafo_evidencia",
    "num_hops",
    "types",
    "evidence_kind",
)


class RecordInvalid(ValueError):
    """Erro de validação de um único registro."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta JSONs de output/ para JSONL validado e auditado."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Fallback para DATASET_NAME: fever, feverous, factkg, hover ou scifact.",
    )
    parser.add_argument(
        "--strict-files",
        action="store_true",
        help="Interrompe somente quando um arquivo inteiro não puder ser lido/analisado.",
    )
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dotenv_value(root: Path, name: str) -> tuple[bool, str | None]:
    """Lê uma variável simples do .env do projeto sem alterar o ambiente do processo."""
    dotenv_path = root / ".env"
    if not dotenv_path.is_file():
        return False, None
    try:
        lines = dotenv_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise OSError(f"não foi possível ler {dotenv_path}: {exc}") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return True, value
    return False, None


def canonical_dataset(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordInvalid("dataset ausente ou vazio")
    dataset = value.strip().lower()
    if dataset not in DATASET_LABELS:
        raise RecordInvalid(f"dataset inválido: {value!r}")
    return dataset


def nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordInvalid(f"{field} ausente, vazio ou não é string")
    return value


def identifier_key(value: Any) -> str:
    """Chave estável para IDs JSON, distinguindo tipos quando necessário."""
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return repr(value)


def validate_graph(value: Any, field: str) -> tuple[dict[str, list[Any]], bool, bool]:
    """Valida um grafo e retorna (grafo, vazio, estruturalmente_inválido)."""
    errors: list[str] = []
    empty = False
    if not isinstance(value, dict):
        raise RecordInvalid(f"{field} deve ser um objeto")

    if "nodes" not in value:
        errors.append("nodes ausente")
    if "edges" not in value:
        errors.append("edges ausente")
    nodes = value.get("nodes")
    edges = value.get("edges")
    if not isinstance(nodes, list):
        errors.append("nodes não é lista")
        nodes = []
    if not isinstance(edges, list):
        errors.append("edges não é lista")
        edges = []
    empty = not nodes or not edges
    if not nodes:
        errors.append("grafo sem nós")
    if not edges:
        errors.append("grafo sem arestas")

    node_ids: set[str] = set()
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nó {node_index} não é objeto")
            continue
        id_field = "id" if "id" in node else "node_id" if "node_id" in node else None
        node_id = node.get(id_field) if id_field else None
        if id_field is None or node_id is None or (
            isinstance(node_id, str) and not node_id.strip()
        ):
            errors.append(f"nó {node_index} sem ID")
        else:
            key = identifier_key(node_id)
            if key in node_ids:
                errors.append(f"nó duplicado: {node_id!r}")
            node_ids.add(key)
        text = next((node.get(name) for name in ("text", "name", "label") if name in node), None)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"nó {node_index} sem texto não vazio")

    for edge_index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"aresta {edge_index} não é objeto")
            continue
        if "source" not in edge or "target" not in edge:
            errors.append(f"aresta {edge_index} sem source/target")
        else:
            if identifier_key(edge.get("source")) not in node_ids:
                errors.append(f"aresta {edge_index} aponta para source inexistente")
            if identifier_key(edge.get("target")) not in node_ids:
                errors.append(f"aresta {edge_index} aponta para target inexistente")
        relation = next(
            (edge.get(name) for name in ("type", "relation", "edge_type") if name in edge),
            None,
        )
        if not isinstance(relation, str) or not relation.strip():
            errors.append(f"aresta {edge_index} sem relação")

    if errors:
        raise RecordInvalid(f"{field}: " + "; ".join(errors))
    return {"nodes": nodes, "edges": edges}, empty, False


def normalized_claim(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def missing_source_fields(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return list(SOURCE_AUDIT_FIELDS)
    missing = [
        field
        for field in SOURCE_AUDIT_FIELDS
        if field != "evidencia/evidence_text" and field not in item
    ]
    if "evidencia" not in item and "evidence_text" not in item:
        missing.append("evidencia/evidence_text")
    return missing


def validate_record(
    item: Any,
    *,
    expected_dataset: str | None,
) -> tuple[dict[str, Any], int, int]:
    if not isinstance(item, dict):
        raise RecordInvalid("registro não é um objeto")

    for field in ("id", "claim", "label", "grafo_claim", "grafo_evidencia"):
        if field not in item:
            raise RecordInvalid(f"campo obrigatório ausente: {field}")

    record_id = nonempty_string(item["id"], "id")
    if "dataset" in item and canonical_dataset(item["dataset"]) != expected_dataset:
        raise RecordInvalid(
            f"dataset incompatível: esperado {expected_dataset!r}, recebido {item['dataset']!r}"
        )
    dataset = expected_dataset
    claim = nonempty_string(item["claim"], "claim")
    evidence_key = "evidencia" if "evidencia" in item else "evidence_text"
    if evidence_key not in item:
        raise RecordInvalid("campo obrigatório ausente: evidencia ou evidence_text")
    evidence = nonempty_string(item[evidence_key], evidence_key)
    label = nonempty_string(item["label"], "label")
    if label not in DATASET_LABELS[dataset]:
        allowed = ", ".join(sorted(DATASET_LABELS[dataset]))
        raise RecordInvalid(f"label inválido para {dataset}: {label!r}; aceitos: {allowed}")
    for field in ("synthetic", "evidence_complete"):
        if field in item and not isinstance(item[field], bool):
            raise RecordInvalid(f"{field} deve ser booleano quando presente")
    if "types" in item and not isinstance(item["types"], list):
        raise RecordInvalid("types deve ser lista quando presente")

    empty_graphs = 0
    invalid_graphs = 0
    graphs: dict[str, dict[str, list[Any]]] = {}
    graph_errors: list[str] = []
    for field in ("grafo_claim", "grafo_evidencia"):
        try:
            graph, is_empty, _ = validate_graph(item[field], field)
        except RecordInvalid as exc:
            invalid_graphs += 1
            value = item[field]
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("nodes"), list)
                or not value.get("nodes")
                or not isinstance(value.get("edges"), list)
                or not value.get("edges")
            ):
                empty_graphs += 1
            graph_errors.append(str(exc))
            continue
        graphs[field] = graph
        empty_graphs += int(is_empty)

    if graph_errors:
        raise RecordInvalid("; ".join(graph_errors))

    record = {
        "id": record_id,
        "dataset": dataset,
        "claim": claim,
        "evidencia": evidence,
        "label": label,
        "grafo_claim": graphs["grafo_claim"],
        "grafo_evidencia": graphs["grafo_evidencia"],
    }
    for field in (
        "synthetic",
        "evidence_complete",
        "split",
        "evaluation_role",
        "group_id",
        "num_hops",
        "types",
        "evidence_kind",
    ):
        if field in item:
            record[field] = item[field]
    return record, empty_graphs, invalid_graphs


def safe_json_loads(data: bytes) -> Any:
    return json.loads(
        data.decode("utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"constante JSON não permitida: {value}")
        ),
    )


def read_input_files(
    input_dir: Path,
) -> tuple[list[tuple[Path, int, Any]], list[dict[str, Any]], list[str]]:
    """Lê arquivos e retorna candidatos (arquivo, índice, item), auditoria e erros."""
    if not input_dir.exists():
        raise FileNotFoundError(f"pasta de entrada não encontrada: {input_dir}")
    files = sorted(
        (path for path in input_dir.glob("*.json") if path.is_file()),
        key=lambda p: p.name,
    )
    candidates: list[tuple[Path, int, Any]] = []
    file_audits: list[dict[str, Any]] = []
    file_errors: list[str] = []
    for path in files:
        entry: dict[str, Any] = {
            "path": str(path),
            "size_bytes": None,
            "sha256": None,
            "records_read": 0,
            "errors": [],
        }
        try:
            data = path.read_bytes()
            entry["size_bytes"] = len(data)
            entry["sha256"] = hashlib.sha256(data).hexdigest()
            value = safe_json_loads(data)
            if isinstance(value, dict):
                candidates.append((path, 0, value))
                entry["records_read"] = 1
            elif isinstance(value, list):
                entry["top_level_format"] = "array_processed_as_records"
                entry["records_read"] = len(value)
                candidates.extend((path, i, item) for i, item in enumerate(value))
            else:
                raise ValueError("top-level JSON deve ser objeto ou array")
        except Exception as exc:  # noqa: BLE001 - erro de arquivo deve ser auditado
            message = f"{type(exc).__name__}: {exc}"
            entry["errors"].append(message)
            file_errors.append(f"{path}: {message}")
        file_audits.append(entry)
    return candidates, file_audits, file_errors


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for counter in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"não foi possível gerar nome único para {path}")


def output_base_path_for(
    output_dir: Path,
    requested: Path | None,
    dataset: str | None,
    count: int,
    timestamp: datetime,
) -> Path:
    if requested is not None:
        path = requested if requested.is_absolute() else output_dir / requested
        if path.suffix:
            path = path.with_suffix("")
    else:
        name = dataset or "unknown"
        path = output_dir / f"{name}_{count}_{timestamp:%Y%m%d_%H%M%S}"
    return path


def validate_jsonl_file(
    jsonl_path: Path,
    *,
    expected_records: int | None = None,
) -> int:
    """Reimporta um JSONL e garante um objeto JSON completo por linha."""
    records_read = 0
    with jsonl_path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordInvalid(
                    f"JSONL inválido na linha {line_number}: "
                    f"{exc.msg} (coluna {exc.colno})"
                ) from exc
            if not isinstance(value, dict):
                raise RecordInvalid(
                    f"JSONL inválido na linha {line_number}: "
                    f"esperado objeto JSON, recebido {type(value).__name__}"
                )
            records_read += 1

    if expected_records is not None and records_read != expected_records:
        raise RecordInvalid(
            f"contagem JSONL divergente em {jsonl_path}: "
            f"escritos={expected_records}, relidos={records_read}"
        )
    return records_read


def write_jsonl_parts_atomic(
    records: Iterable[dict[str, Any]],
    output_base: Path,
) -> tuple[list[Path], list[int]]:
    """Escreve partes JSONL sem quebrar registros e valida cada parte."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    part_counts: list[int] = []
    stream: Any | None = None
    temp_path: Path | None = None
    part_number = 0
    current_count = 0
    current_bytes = 0

    def open_part() -> None:
        nonlocal stream, temp_path, part_number, current_count, current_bytes
        part_number += 1
        stream = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_base.parent,
            prefix=f".{output_base.name}_part_{part_number}.",
            suffix=".tmp",
            delete=False,
        )
        temp_path = Path(stream.name)
        current_count = 0
        current_bytes = 0

    def finish_part() -> None:
        nonlocal stream, temp_path, current_count, current_bytes
        if stream is None or temp_path is None:
            return
        try:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            validate_jsonl_file(temp_path, expected_records=current_count)
            final_path = unique_path(
                output_base.with_name(
                    f"{output_base.name}_part_{part_number}.jsonl"
                )
            )
            os.replace(temp_path, final_path)
            part_paths.append(final_path)
            part_counts.append(current_count)
            temp_path = None
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            stream = None
            temp_path = None
            current_count = 0
            current_bytes = 0

    try:
        for record in records:
            json_text = json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            line = json_text + "\n"
            line_bytes = len(line.encode("utf-8"))
            if stream is None:
                open_part()
            elif current_count and current_bytes + line_bytes > MAX_JSONL_BYTES:
                finish_part()
                open_part()
            stream.write(line)
            current_count += 1
            current_bytes += line_bytes

        if stream is None:
            open_part()
        finish_part()
    except BaseException:
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return part_paths, part_counts


def execute_export(
    *,
    input_dir: Path,
    output_dir: Path,
    output_file: Path | None = None,
    dataset: str | None = None,
    strict_files: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    input_dir = resolve_path(input_dir)
    output_dir = resolve_path(output_dir)
    dotenv_has_dataset, dotenv_dataset = load_dotenv_value(PROJECT_ROOT, "DATASET_NAME")
    environment_dataset = os.environ.get("DATASET_NAME")
    if dotenv_has_dataset:
        dataset_value = dotenv_dataset
        dataset_source = ".env"
    elif environment_dataset is not None:
        dataset_value = environment_dataset
        dataset_source = "DATASET_NAME"
    else:
        dataset_value = dataset
        dataset_source = "--dataset"
    if dataset_value is None:
        raise RecordInvalid("DATASET_NAME não definido no .env nem no ambiente e nenhum --dataset foi informado")
    forced_dataset = canonical_dataset(dataset_value)
    started = datetime.now().astimezone()
    candidates, input_files, file_errors = read_input_files(input_dir)
    if strict_files and file_errors:
        raise OSError("erros de leitura/análise de arquivo: " + " | ".join(file_errors))

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    duplicate_ids: list[dict[str, Any]] = []
    duplicate_claims: list[dict[str, Any]] = []
    labels: set[str] = set()
    splits: set[str] = set()
    roles: set[str] = set()
    accepted_ids: dict[str, dict[str, Any]] = {}
    claim_roles: defaultdict[str, set[str]] = defaultdict(set)
    selected_dataset = forced_dataset
    empty_graphs = 0
    invalid_graphs = 0
    missing_field_counts: Counter[str] = Counter()
    missing_fields_by_record: list[dict[str, Any]] = []

    def reject(
        path: Path,
        index: int,
        item: Any,
        reason: str,
        missing_fields: list[str],
    ) -> None:
        rejection_reasons[reason.split(":", 1)[0]] += 1
        entry = {
            "file": str(path),
            "index": index,
            "id": item.get("id") if isinstance(item, dict) else None,
            "reason": reason,
        }
        if missing_fields:
            entry["missing_fields"] = missing_fields
        rejected.append(entry)

    for path, index, item in candidates:
        missing_fields = missing_source_fields(item)
        if missing_fields:
            missing_fields_by_record.append(
                {
                    "file": str(path),
                    "index": index,
                    "id": item.get("id") if isinstance(item, dict) else None,
                    "fields": missing_fields,
                }
            )
            missing_field_counts.update(missing_fields)
        if isinstance(item, dict):
            for key, target in (
                ("label", labels),
                ("split", splits),
                ("evaluation_role", roles),
            ):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    target.add(value)
        try:
            record, empty_count, invalid_count = validate_record(
                item, expected_dataset=selected_dataset
            )
            empty_graphs += empty_count
            invalid_graphs += invalid_count
            record_id_key = identifier_key(record["id"])
            if record_id_key in accepted_ids:
                previous = accepted_ids[record_id_key]
                duplicate = {
                    "id": record["id"],
                    "first_file": previous["file"],
                    "first_index": previous["index"],
                    "duplicate_file": str(path),
                    "duplicate_index": index,
                }
                duplicate_ids.append(duplicate)
                reject(
                    path,
                    index,
                    item,
                    f"id duplicado; primeiro registro em {previous['file']}:{previous['index']}",
                    missing_fields,
                )
                continue
            claim_key = normalized_claim(record["claim"])
            prior_roles = claim_roles[claim_key]
            current_role = record.get("evaluation_role")
            if (
                prior_roles
                and isinstance(current_role, str)
                and current_role.strip()
                and current_role not in prior_roles
            ):
                conflict = {
                    "claim_normalized": claim_key,
                    "roles_already_seen": sorted(prior_roles),
                    "new_role": current_role,
                    "file": str(path),
                    "index": index,
                    "id": record["id"],
                }
                duplicate_claims.append(conflict)
                reject(
                    path,
                    index,
                    item,
                    "claim duplicada entre evaluation_role diferentes; registro posterior ignorado",
                    missing_fields,
                )
                continue
            accepted_ids[record_id_key] = {"file": str(path), "index": index}
            if isinstance(current_role, str) and current_role.strip():
                claim_roles[claim_key].add(current_role)
            records.append(record)
        except RecordInvalid as exc:
            reject(path, index, item, str(exc), missing_fields)

    output_base = output_base_path_for(
        output_dir, output_file, forced_dataset, len(records), started
    )
    audit_path = unique_path(
        output_base.with_name(f"{output_base.name}_audit.json")
    )
    audit = {
        "dataset": forced_dataset,
        "dataset_source": dataset_source,
        "timestamp": started.isoformat(),
        "exporter_version": EXPORTER_VERSION,
        "python": sys.version,
        "input_files": input_files,
        "input_file_hashes_sha256": {
            entry["path"]: entry["sha256"] for entry in input_files
        },
        "total_read": sum(entry["records_read"] for entry in input_files),
        "exported": len(records),
        "rejected": len(rejected),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "rejected_records": rejected,
        "missing_field_counts": dict(sorted(missing_field_counts.items())),
        "missing_fields_by_record": missing_fields_by_record,
        "labels_found": sorted(labels),
        "splits_found": sorted(splits),
        "evaluation_roles_found": sorted(roles),
        "duplicate_ids": duplicate_ids,
        "duplicate_claims_between_roles": duplicate_claims,
        "empty_graph_count": empty_graphs,
        "structurally_invalid_graph_count": invalid_graphs,
        "jsonl_file": None,
        "jsonl_files": [],
        "part_count": 0,
        "max_jsonl_bytes": MAX_JSONL_BYTES,
        "part_sizes_bytes": [],
        "part_records": [],
        "lines_written": len(records),
        "lines_reloaded": None,
        "continuation_policy": "registros inválidos são auditados e ignorados; campos ausentes são registrados sem receber defaults; split/evaluation_role/group_id só são exportados quando presentes; o processamento continua; --strict-files vale apenas para erros de arquivo inteiro",
        "ignored_files_by_error": file_errors,
        "status": (
            "no_valid_records"
            if not records
            else "completed_with_rejections"
            if rejected or file_errors
            else "completed"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths, part_counts = write_jsonl_parts_atomic(records, output_base)
    lines_written = sum(part_counts)
    lines_reloaded = 0
    part_sizes_bytes: list[int] = []
    for output_path, expected_count in zip(output_paths, part_counts):
        lines_reloaded += validate_jsonl_file(
            output_path,
            expected_records=expected_count,
        )
        part_sizes_bytes.append(output_path.stat().st_size)
    audit["lines_written"] = lines_written
    audit["lines_reloaded"] = lines_reloaded
    audit["jsonl_file"] = output_paths[0].name if len(output_paths) == 1 else None
    audit["jsonl_files"] = [path.name for path in output_paths]
    audit["part_count"] = len(output_paths)
    audit["part_sizes_bytes"] = part_sizes_bytes
    audit["part_records"] = part_counts
    if lines_written != lines_reloaded:
        raise RecordInvalid(
            f"contagem final divergente: escritos={lines_written}, "
            f"relidos={lines_reloaded}"
        )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_paths[0], audit_path, audit


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_path, audit_path, audit = execute_export(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            output_file=args.output_file,
            dataset=args.dataset,
            strict_files=args.strict_files,
        )
    except (FileNotFoundError, OSError, RecordInvalid) as exc:
        raise SystemExit(f"Erro: {exc}") from exc
    print(
        f"Exportados {audit['exported']} registros em "
        f"{audit['part_count']} parte(s)"
    )
    for filename in audit["jsonl_files"]:
        print(f"JSONL: {output_path.parent / filename}")
    print(f"Auditoria salva em {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
