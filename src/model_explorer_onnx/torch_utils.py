"""Utilities for PyTorch"""

from __future__ import annotations
import os
from typing import Collection, TYPE_CHECKING

from model_explorer import node_data_builder as ndb
from onnxscript import ir
import logging

if TYPE_CHECKING:
    # TODO: Change the import when it is exposed to public
    from torch.onnx._internal.exporter._verification import VerificationInfo


logger = logging.getLogger(__name__)


def _create_value_mapping(graph: ir.Graph) -> dict[str, ir.Value]:
    """Return a dictionary mapping names to values in the graph.

    The mapping does not include values from subgraphs.

    Args:
        graph: The graph to extract the mapping from.

    Returns:
        A dictionary mapping names to values.
    """
    values = {}
    values.update(graph.initializers)
    # The names of the values can be None or "", which we need to exclude
    for input in graph.inputs:
        if not input.name:
            continue
        values[input.name] = input
    for node in graph:
        for value in node.outputs:
            if not value.name:
                continue
            values[value.name] = value
    return values


def save_node_data_from_verification_info(
    verification_infos: Collection[VerificationInfo],
    onnx_model: ir.Model,
    directory: str = "",
    model_name: str = "model",
):
    """Saves the node data for model explorer.

    Example::

        onnx_program = torch.onnx.export(
            model,
            args,
            dynamo=True
        )

        onnx_program.save("model.onnx")

        from torch.onnx.verification import VerificationInterpreter

        interpreter = VerificationInterpreter(onnx_program)
        interpreter.run(*args)

        from model_explorer_onnx.torch_utils import save_node_data_from_verification_info

        save_node_data_from_verification_info(
            interpreter.verification_infos, onnx_program.model, model_name="model"
        )

    You can then use Model Explorer to visualize the results by loading the generated node data files.

    Args:
        verification_infos: The verification information objects.
        node_names: The names of the nodes each VerificationInfo corresponds to.
        model_name: The name of the model, used for constructing the file names.
    """
    values = _create_value_mapping(onnx_model.graph)
    node_names = []
    for info in verification_infos:
        if info.name in values:
            node_names.append(values[info.name].producer().name)
        else:
            node_names.append(info.name)
            logger.warning(
                "The name %s is not found in the graph. Please ensure the model provided matches the "
                "verification information.",
                info.name,
            )
    for field in ("max_abs_diff", "max_rel_diff"):
        # Populate values for the main graph in a model.
        main_graph_results: dict[str, ndb.NodeDataResult] = {}
        for info, node_name in zip(verification_infos, node_names):
            if (
                values.get(info.name) is not None
                and values[info.name].is_graph_output()
            ):
                main_graph_results[f"[value] {info.name}"] = ndb.NodeDataResult(
                    value=getattr(info, field)
                )
            else:
                main_graph_results[node_name] = ndb.NodeDataResult(
                    value=getattr(info, field)
                )

        thresholds: list[ndb.ThresholdItem] = [
            ndb.ThresholdItem(value=0.00001, bgColor="#388e3c"),
            ndb.ThresholdItem(value=0.0001, bgColor="#8bc34a"),
            ndb.ThresholdItem(value=0.001, bgColor="#c8e6c9"),
            ndb.ThresholdItem(value=0.01, bgColor="#ffa000"),
            ndb.ThresholdItem(value=1, bgColor="#ff5722"),
            ndb.ThresholdItem(value=100, bgColor="#d32f2f"),
        ]

        # Construct the data for the main graph.
        main_graph_data = ndb.GraphNodeData(
            results=main_graph_results, thresholds=thresholds
        )

        # Construct the data for the model.
        # "main_graph" is the default graph name produced by the exporter.
        model_data = ndb.ModelNodeData(graphsData={"main_graph": main_graph_data})

        model_data.save_to_file(os.path.join(directory, f"{model_name}_{field}.json"))
