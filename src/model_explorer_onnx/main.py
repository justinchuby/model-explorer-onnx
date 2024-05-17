from __future__ import annotations

from typing import Any, Mapping, Sequence
import model_explorer
from model_explorer import graph_builder
import onnx
from onnxscript import ir
import logging

logger = logging.getLogger(__name__)


def add_inputs_metadata(onnx_node: ir.Node, node: graph_builder.GraphNode):
    for i, input_value in enumerate(onnx_node.inputs):
        metadata = graph_builder.MetadataItem(id=str(i), attrs=[])
        if input_value is None:
            metadata.attrs.append(graph_builder.KeyValue(key="__tensor_tag", value=""))
        else:
            metadata.attrs.append(
                graph_builder.KeyValue(key="__tensor_tag", value=input_value.name or "")
            )
        node.inputsMetadata.append(metadata)


def add_outputs_metadata(onnx_node: ir.Node, node: graph_builder.GraphNode):
    for output in onnx_node.outputs:
        metadata = graph_builder.MetadataItem(id=str(output.index()), attrs=[])
        dtype = str(output.dtype)
        type_str = str(output.type)
        shape = str(output.shape)
        metadata.attrs.append(
            graph_builder.KeyValue(key="__tensor_tag", value=output.name or "")
        )
        metadata.attrs.append(
            graph_builder.KeyValue(key="tensor_shape", value=dtype + shape)
        )
        metadata.attrs.append(graph_builder.KeyValue(key="type", value=type_str))
        node.outputsMetadata.append(metadata)


def add_node_attrs(onnx_node: ir.Node, node: graph_builder.GraphNode):
    for attr in onnx_node.attributes.values():
        if isinstance(attr, ir.Attr):
            node.attrs.append(
                graph_builder.KeyValue(key=attr.name, value=str(attr.value))
            )
        elif isinstance(attr, ir.RefAttr):
            node.attrs.append(
                graph_builder.KeyValue(
                    key=attr.name, value=f"Ref({attr.ref_attr_name})"
                )
            )


def add_incoming_edges(
    onnx_node: ir.Node,
    node: graph_builder.GraphNode,
    graph_inputs: Mapping[ir.Value, int],
):
    for target_input_id, input_value in enumerate(onnx_node.inputs):
        if input_value is None:
            continue
        if input_value in graph_inputs:
            # The input is a graph input. Create an input edge.
            node.incomingEdges.append(
                graph_builder.IncomingEdge(
                    sourceNodeId=input_value.name, # type: ignore
                    sourceNodeOutputId=str(graph_inputs[input_value]),
                    targetNodeInputId=str(target_input_id),
                )
            )
            continue
        input_node = input_value.producer()
        if input_node is None:
            logger.warning(
                "Input value %s does not have a producer. Skipping incoming edge.",
                input_value,
            )
            continue
        if not input_node.name:
            logger.debug(
                "Node %s does not have a name. Skipping incoming edge.", input_node
            )
            continue
        node.incomingEdges.append(
            graph_builder.IncomingEdge(
                sourceNodeId=input_node.name,
                sourceNodeOutputId=str(input_value.index()),
                targetNodeInputId=str(target_input_id),
            )
        )


def create_op_label(domain: str, op_type: str) -> str:
    if domain in {"", "ai.onnx"}:
        return op_type
    return f"{domain}::{op_type}"


def create_node(
    onnx_node: ir.Node, graph_inputs: Mapping[ir.Value, int]
) -> graph_builder.GraphNode | None:
    if onnx_node.name is None:
        logger.warning("Node does not have a name. Skipping node %s.", onnx_node)
        return None
    node = graph_builder.GraphNode(
        id=onnx_node.name,
        label=create_op_label(onnx_node.domain, onnx_node.op_type),
        # namespace=None,
    )
    add_incoming_edges(onnx_node, node, graph_inputs)
    add_node_attrs(onnx_node, node)
    add_inputs_metadata(onnx_node, node)
    add_outputs_metadata(onnx_node, node)
    return node


def add_graph_io(graph: graph_builder.Graph, input_or_outputs: Sequence[ir.Value]):
    for value in input_or_outputs:
        graph.nodes.append(
            graph_builder.GraphNode(
                id=value.name,  # type: ignore
                label=value.name,  # type: ignore
            )
        )


def create_graph(onnx_graph: ir.Graph | ir.Function) -> graph_builder.Graph:
    graph = graph_builder.Graph(id="graph", nodes=[])
    graph_inputs: dict[ir.Value, int] = {
        input_value: i for i, input_value in enumerate(onnx_graph.inputs)
    }
    if isinstance(onnx_graph, ir.Graph):
        graph_inputs.update(
            {
                input_value: i
                for i, input_value in enumerate(onnx_graph.initializers.values())
            }
        )
    for onnx_node in onnx_graph:
        node = create_node(onnx_node, graph_inputs)
        if node is None:
            continue
        graph.nodes.append(node)
    add_graph_io(graph, list(graph_inputs.keys()))
    add_graph_io(graph, onnx_graph.outputs)
    return graph


class ONNXAdapter(model_explorer.Adapter):
    metadata = model_explorer.AdapterMetadata(
        id="onnx_adapter",
        name="ONNX adapter",
        description="ONNX adapter for Model Explorer",
        source_repo="https://github.com/justinchuby/model_explorer_onnx",
        fileExts=["onnx", "onnxtext", "onnxtxt"],
    )

    def convert(
        self, model_path: str, settings: dict[str, Any]
    ) -> model_explorer.ModelExplorerGraphs:
        onnx_model = onnx.load(model_path)
        model = ir.serde.deserialize_model(onnx_model)
        graphs = [create_graph(model.graph)]
        for function in model.functions.values():
            graphs.append(create_graph(function))
        return {"graphs": graphs}
