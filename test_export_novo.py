from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src").exists() else HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import export_novo  # noqa: E402


_DATASET_FOUND, DEFAULT_DATASET = export_novo.load_dotenv_value(ROOT, "DATASET_NAME")
DEFAULT_DATASET = DEFAULT_DATASET or "scifact"
LABELS = {
    "fever": "SUPPORTS",
    "feverous": "SUPPORTS",
    "factkg": "SUPPORTS",
    "hover": "SUPPORTED",
    "scifact": "SUPPORT",
}


def graph() -> dict[str, list[dict[str, object]]]:
    return {
        "nodes": [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}],
        "edges": [{"source": "a", "target": "b", "type": "relates"}],
    }


def large_graph(prefix: str) -> dict[str, list[dict[str, object]]]:
    nodes = [
        {
            "id": f"{prefix}-{index}",
            "text": f"Node {index}\n" + ("long text " * 30),
        }
        for index in range(180)
    ]
    edges = [
        {"source": f"{prefix}-{index}", "target": f"{prefix}-{index + 1}", "type": "next"}
        for index in range(179)
    ]
    return {"nodes": nodes, "edges": edges}


def record(
    *,
    record_id: str = "id-1",
    claim: str = "A claim",
    role: str = "train",
    dataset: str = DEFAULT_DATASET,
) -> dict[str, object]:
    return {
        "id": record_id,
        "dataset": dataset,
        "synthetic": False,
        "claim": claim,
        "evidencia": "Evidence text",
        "label": LABELS[dataset],
        "split": role,
        "evaluation_role": role,
        "group_id": "group-1",
        "evidence_complete": True,
        "grafo_claim": graph(),
        "grafo_evidencia": graph(),
        "num_hops": 1,
        "types": ["single-hop"],
        "evidence_kind": "text",
    }


class ExportNovoTests(unittest.TestCase):
    def run_export(self, values: list[object], files: dict[str, object] | None = None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root
            output_dir = root
            if files is None:
                files = {"000.json": values}
            for name, value in files.items():
                path = input_dir / name
                if isinstance(value, str):
                    path.write_text(value, encoding="utf-8")
                else:
                    path.write_text(json.dumps(value), encoding="utf-8")
            output_path, audit_path, audit = export_novo.execute_export(
                input_dir=input_dir, output_dir=output_dir, dataset=DEFAULT_DATASET
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()
            return lines, json.loads(audit_path.read_text(encoding="utf-8")), audit

    def test_valid_record_and_audit(self):
        lines, audit, _ = self.run_export([record()])
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["dataset"], DEFAULT_DATASET)
        self.assertEqual(audit["status"], "completed")
        self.assertEqual(audit["lines_written"], 1)
        self.assertEqual(audit["labels_found"], [LABELS[DEFAULT_DATASET]])
        self.assertTrue(audit["input_files"][0]["sha256"])

    def test_required_and_field_validation_matrix(self):
        variants: dict[str, dict[str, object]] = {}
        missing = record()
        del missing["claim"]
        variants["missing required field"] = missing
        empty_claim = record()
        empty_claim["claim"] = "   "
        variants["empty claim"] = empty_claim
        bad_label = record()
        bad_label["label"] = "INVALID_LABEL"
        variants["invalid label"] = bad_label
        no_nodes = record()
        no_nodes["grafo_claim"] = {"nodes": [], "edges": []}
        variants["graph without nodes"] = no_nodes
        no_edges = record()
        no_edges["grafo_claim"] = {"nodes": graph()["nodes"], "edges": []}
        variants["graph without edges"] = no_edges
        duplicate_node = record()
        duplicate_node["grafo_claim"] = {
            "nodes": [{"id": "a", "text": "one"}, {"id": "a", "text": "two"}],
            "edges": [{"source": "a", "target": "a", "type": "loop"}],
        }
        variants["duplicate node"] = duplicate_node
        dangling = record()
        dangling["grafo_claim"] = {
            "nodes": [{"id": "a", "text": "one"}],
            "edges": [{"source": "a", "target": "missing", "type": "relates"}],
        }
        variants["dangling edge"] = dangling
        no_relation = record()
        no_relation["grafo_claim"] = {
            "nodes": graph()["nodes"],
            "edges": [{"source": "a", "target": "b"}],
        }
        variants["missing relation"] = no_relation
        bad_synthetic = record()
        bad_synthetic["synthetic"] = "false"
        variants["bad synthetic"] = bad_synthetic
        bad_complete = record()
        bad_complete["evidence_complete"] = "true"
        variants["bad evidence_complete"] = bad_complete

        for name, invalid in variants.items():
            with self.subTest(name=name):
                lines, audit, _ = self.run_export([invalid, record(record_id="valid")])
                self.assertEqual(len(lines), 1)
                self.assertEqual(audit["rejected"], 1)

    def test_mixed_dataset_is_rejected(self):
        other_dataset = next(name for name in LABELS if name != DEFAULT_DATASET)
        lines, audit, _ = self.run_export(
            [record(), record(record_id="other", dataset=other_dataset)]
        )
        self.assertEqual(len(lines), 1)
        self.assertIn("dataset incompatível", audit["rejected_records"][0]["reason"])

    def test_missing_raw_metadata_is_allowed_and_recorded(self):
        value = record()
        for field in ("dataset", "split", "evaluation_role", "group_id"):
            del value[field]
        lines, audit, _ = self.run_export([value])
        exported = json.loads(lines[0])
        self.assertEqual(exported["dataset"], DEFAULT_DATASET)
        self.assertNotIn("split", exported)
        self.assertNotIn("evaluation_role", exported)
        self.assertNotIn("group_id", exported)
        self.assertEqual(audit["missing_field_counts"]["split"], 1)
        self.assertEqual(audit["missing_field_counts"]["evaluation_role"], 1)
        self.assertEqual(audit["missing_field_counts"]["group_id"], 1)

    def test_project_dotenv_defines_dataset(self):
        found, value = export_novo.load_dotenv_value(ROOT, "DATASET_NAME")
        self.assertTrue(found)
        self.assertEqual(value, DEFAULT_DATASET)

    def test_dotenv_defines_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text(
                "# project configuration\nexport DATASET_NAME=\"hover\"\n",
                encoding="utf-8",
            )
            found, value = export_novo.load_dotenv_value(root, "DATASET_NAME")
            self.assertTrue(found)
            self.assertEqual(value, "hover")

    def test_duplicate_id_keeps_first(self):
        lines, audit, _ = self.run_export([record(), record(record_id="id-1", claim="Other")])
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(audit["duplicate_ids"]), 1)

    def test_claim_leakage_between_roles_keeps_first(self):
        lines, audit, _ = self.run_export(
            [record(), record(record_id="test-1", role="test")]
        )
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(audit["duplicate_claims_between_roles"]), 1)

    def test_invalid_file_is_followed_by_valid_file(self):
        lines, audit, _ = self.run_export(
            [],
            {"000_invalid.json": "{not json", "001_valid.json": record(record_id="valid")},
        )
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(audit["ignored_files_by_error"]), 1)
        self.assertEqual(audit["status"], "completed_with_rejections")

    def test_consecutive_invalid_records_do_not_stop_processing(self):
        bad = record()
        del bad["grafo_evidencia"]
        lines, audit, _ = self.run_export([bad, {}, {"not": "a record"}, record(record_id="last")])
        self.assertEqual(len(lines), 1)
        self.assertEqual(audit["rejected"], 3)

    def test_no_valid_records_creates_empty_jsonl_and_audit(self):
        invalid = record()
        invalid["claim"] = ""
        lines, audit, _ = self.run_export([invalid])
        self.assertEqual(lines, [])
        self.assertEqual(audit["status"], "no_valid_records")
        self.assertEqual(audit["lines_written"], 0)

    def test_output_name_collision_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root
            output_dir = root
            (input_dir / "000.json").write_text(json.dumps(record()), encoding="utf-8")
            first, _, _ = export_novo.execute_export(
                input_dir=input_dir,
                output_dir=output_dir,
                output_file=Path("fixed.jsonl"),
                dataset=DEFAULT_DATASET,
            )
            second, _, _ = export_novo.execute_export(
                input_dir=input_dir,
                output_dir=output_dir,
                output_file=Path("fixed.jsonl"),
                dataset=DEFAULT_DATASET,
            )
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_large_multiline_record_is_one_parseable_jsonl_line(self):
        value = record(record_id="large")
        value["claim"] = "Claim first line\n" + ("claim text " * 1200)
        value["evidencia"] = "Evidence first line\n" + ("evidence text\n" * 1200)
        value["grafo_claim"] = large_graph("claim")
        value["grafo_evidencia"] = large_graph("evidence")

        lines, audit, _ = self.run_export([value])

        self.assertEqual(len(lines), 1)
        self.assertGreater(len(lines[0]), 10_000)
        reimported = json.loads(lines[0])
        self.assertIsInstance(reimported, dict)
        self.assertEqual(reimported, {**value, "dataset": DEFAULT_DATASET})
        self.assertEqual(audit["lines_written"], 1)
        self.assertEqual(audit["lines_reloaded"], 1)

        for graph_name in ("grafo_claim", "grafo_evidencia"):
            graph_value = reimported[graph_name]
            node_ids = {node["id"] for node in graph_value["nodes"]}
            self.assertTrue(
                all(
                    edge["source"] in node_ids and edge["target"] in node_ids
                    for edge in graph_value["edges"]
                )
            )

    def test_jsonl_is_split_into_complete_parts(self):
        original_limit = export_novo.MAX_JSONL_BYTES
        export_novo.MAX_JSONL_BYTES = 2_500
        values = []
        for index in range(3):
            value = record(record_id=f"part-{index}")
            value["claim"] = "Claim line 1\\n" + ("c" * 700)
            value["evidencia"] = "Evidence line 1\\n" + ("e" * 700)
            values.append(value)

        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "000.json").write_text(
                    json.dumps(values), encoding="utf-8"
                )
                first_path, audit_path, audit = export_novo.execute_export(
                    input_dir=root,
                    output_dir=root,
                    dataset=DEFAULT_DATASET,
                )
                self.assertTrue(first_path.exists())
                self.assertGreater(audit["part_count"], 1)
                self.assertEqual(audit["lines_written"], 3)
                self.assertEqual(audit["lines_reloaded"], 3)

                reimported = []
                for filename in audit["jsonl_files"]:
                    part_path = root / filename
                    self.assertLessEqual(
                        part_path.stat().st_size,
                        export_novo.MAX_JSONL_BYTES,
                    )
                    physical_lines = part_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    self.assertTrue(physical_lines)
                    for line in physical_lines:
                        parsed = json.loads(line)
                        self.assertIsInstance(parsed, dict)
                        for graph_name in ("grafo_claim", "grafo_evidencia"):
                            graph_value = parsed[graph_name]
                            node_ids = {node["id"] for node in graph_value["nodes"]}
                            self.assertTrue(
                                all(
                                    edge["source"] in node_ids
                                    and edge["target"] in node_ids
                                    for edge in graph_value["edges"]
                                )
                            )
                        reimported.append(parsed)

                self.assertEqual(
                    reimported,
                    [{**value, "dataset": DEFAULT_DATASET} for value in values],
                )
                json.loads(audit_path.read_text(encoding="utf-8"))
        finally:
            export_novo.MAX_JSONL_BYTES = original_limit


if __name__ == "__main__":
    unittest.main()
