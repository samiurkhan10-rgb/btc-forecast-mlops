"""Static checks of the orchestration files.

CI does not install Airflow or kfp (too heavy), so the DAG is checked with `ast` and the
compiled KFP YAML is checked as text. The real runtime checks are the Airflow run in
docker compose and `python kubeflow/btc_kfp_pipeline.py` (see docs/report_notes.md).
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAG_FILE = ROOT / "dags" / "btc_etl_pipeline.py"
KFP_YAML = ROOT / "kubeflow" / "btc_kfp_pipeline.yaml"


def _dag_decorator_kwargs() -> dict:
    tree = ast.parse(DAG_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for deco in node.decorator_list:
                if isinstance(deco, ast.Call) and getattr(deco.func, "id", None) == "dag":
                    return {kw.arg: kw.value for kw in deco.keywords}
    raise AssertionError("no @dag(...) decorator found")


def test_dag_schedule_catchup_and_retries():
    kw = _dag_decorator_kwargs()
    assert ast.literal_eval(kw["schedule"]) == "@daily"
    assert ast.literal_eval(kw["catchup"]) is False
    assert ast.literal_eval(kw["max_active_runs"]) == 1
    tree = ast.parse(DAG_FILE.read_text())
    default_args = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "default_args"
    )
    keys = [ast.literal_eval(k) for k in default_args.keys]
    assert "retries" in keys


def test_dag_has_all_tasks_and_short_circuit_gate():
    src = DAG_FILE.read_text()
    task_ids = re.findall(r'task_id="([a-z_]+)"', src)
    assert task_ids == [
        "fetch_data",
        "build_features",
        "train_and_evaluate",
        "quality_gate",
        "register_model",
        "reload_api",
    ]
    # count real decorator lines only (the module docstring also mentions the decorator)
    assert len(re.findall(r"^\s+@task\.short_circuit\(", src, re.M)) == 1
    assert len(re.findall(r"^\s+@task\.external_python\(", src, re.M)) == 3


def test_kfp_yaml_has_three_container_components_with_artifacts():
    text = KFP_YAML.read_text()
    for comp in ("comp-ingest-data", "comp-generate-features", "comp-train-model"):
        assert comp in text
    for artifact in ("system.Dataset", "system.Model", "system.Metrics"):
        assert artifact in text
    images = set(re.findall(r"image: (\S+)", text))
    assert len(images) == 1, images
    image = images.pop()
    assert image.startswith("docker.io/samiurk/btc-forecast-train:")
    assert not image.endswith(":latest"), "pipeline must be pinned to an immutable SHA tag"
    for cmd in ("- fetch", "- features", "- train"):
        assert cmd in text
