"""Assign unique names to subgraphs so that they can be uniquely identified."""

from model_explorer_onnx import utils
from onnxscript import ir


def _assign_subgraph_names(graph: ir.Graph | ir.Function, namespace: str) -> None:
    for node in graph:
        for attr in node.attributes.values():
            if attr.type != ir.AttributeType.GRAPH:
                continue
            if isinstance(attr, ir.RefAttr):
                continue
            assert isinstance(attr.value, ir.Graph)
            attr.name = f"{namespace}/{node.name}/{attr.name}"
            _assign_subgraph_names(attr.value, attr.name)


def assign_subgraph_names(model: ir.Model) -> ir.Model:
    """Assign unique names to subgraphs so that they can be uniquely identified."""
    if model.graph.name is None:
        model.graph.name = "<main>"
    _assign_subgraph_names(model.graph, model.graph.name)
    for function in model.functions.values():
        function_graph_name = utils.get_function_graph_name(function.identifier())
        _assign_subgraph_names(function, function_graph_name)

    return model
