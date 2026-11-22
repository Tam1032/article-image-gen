import pytest

from app.services.workflows import WorkflowError, WorkflowRegistry


def test_registry_loads_flux_workflow(workflows_dir):
    registry = WorkflowRegistry(workflows_dir)
    assert "flux_txt2img" in registry.names()


def test_build_applies_params_and_defaults(workflows_dir):
    wf = WorkflowRegistry(workflows_dir).get("flux_txt2img")
    graph = wf.build({"prompt": "a red tram", "seed": 42, "width": 1376, "height": 720})

    assert graph["6"]["inputs"]["text"] == "a red tram"
    assert graph["31"]["inputs"]["seed"] == 42
    assert graph["27"]["inputs"]["width"] == 1376
    assert graph["31"]["inputs"]["steps"] == 20  # default from manifest
    assert graph["35"]["inputs"]["guidance"] == 3.5


def test_build_does_not_mutate_template(workflows_dir):
    wf = WorkflowRegistry(workflows_dir).get("flux_txt2img")
    wf.build({"prompt": "x", "seed": 1, "width": 512, "height": 512})
    assert wf.graph["6"]["inputs"]["text"] == ""


def test_build_rejects_unknown_and_missing_params(workflows_dir):
    wf = WorkflowRegistry(workflows_dir).get("flux_txt2img")
    with pytest.raises(WorkflowError, match="unknown"):
        wf.build({"prompt": "x", "seed": 1, "width": 512, "height": 512, "foo": 1})
    with pytest.raises(WorkflowError, match="missing"):
        wf.build({"prompt": "x"})


def test_unknown_workflow_name(workflows_dir):
    with pytest.raises(WorkflowError, match="Unknown workflow"):
        WorkflowRegistry(workflows_dir).get("does-not-exist")
