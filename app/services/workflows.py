"""ComfyUI workflow templates.

A workflow is a ComfyUI graph exported in API format (``*.json``) plus a
manifest (``*.yaml``) that maps friendly parameter names to node inputs::

    name: flux_txt2img
    workflow: flux_txt2img.json
    params:
      prompt: {node: "6", input: text}
      steps:  {node: "31", input: steps, default: 20}

Code only ever talks about ``prompt``/``steps``; node ids stay in the manifest,
so a workflow can be re-exported from the ComfyUI editor without code changes.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class WorkflowError(ValueError):
    pass


@dataclass(frozen=True)
class ParamBinding:
    node: str
    input: str


@dataclass
class Workflow:
    name: str
    description: str
    graph: dict[str, Any]
    params: dict[str, ParamBinding]
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, manifest_path: Path) -> "Workflow":
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        graph_path = manifest_path.parent / manifest["workflow"]
        graph = json.loads(graph_path.read_text(encoding="utf-8"))

        params: dict[str, ParamBinding] = {}
        defaults: dict[str, Any] = {}
        for name, spec in manifest.get("params", {}).items():
            binding = ParamBinding(node=str(spec["node"]), input=spec["input"])
            node = graph.get(binding.node)
            if node is None or binding.input not in node.get("inputs", {}):
                raise WorkflowError(
                    f"{manifest_path.name}: param '{name}' points to missing "
                    f"node/input {binding.node}.{binding.input}"
                )
            params[name] = binding
            if "default" in spec:
                defaults[name] = spec["default"]

        return cls(
            name=manifest["name"],
            description=manifest.get("description", ""),
            graph=graph,
            params=params,
            defaults=defaults,
        )

    def build(self, values: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the graph with the given parameter values applied."""
        unknown = set(values) - set(self.params)
        if unknown:
            raise WorkflowError(f"{self.name}: unknown params {sorted(unknown)}")
        merged = {**self.defaults, **values}
        missing = set(self.params) - set(merged)
        if missing:
            raise WorkflowError(f"{self.name}: missing params {sorted(missing)}")

        graph = copy.deepcopy(self.graph)
        for name, value in merged.items():
            binding = self.params[name]
            graph[binding.node]["inputs"][binding.input] = value
        return graph


class WorkflowRegistry:
    def __init__(self, directory: Path):
        self._workflows: dict[str, Workflow] = {}
        for manifest_path in sorted(Path(directory).glob("*.yaml")):
            wf = Workflow.load(manifest_path)
            self._workflows[wf.name] = wf
        if not self._workflows:
            raise WorkflowError(f"No workflow manifests found in {directory}")

    def get(self, name: str) -> Workflow:
        try:
            return self._workflows[name]
        except KeyError:
            raise WorkflowError(
                f"Unknown workflow '{name}'. Available: {self.names()}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._workflows)
