from __future__ import annotations

import logging
import os
from typing import Any, Literal, Sequence

import model_explorer
import onnx
from model_explorer import graph_builder
from onnxscript import ir
import numpy as np
import ml_dtypes

logger = logging.getLogger(__name__)

_TENSOR_DISPLAY_LIMIT = 1024


def display_tensor(tensor: ir.TensorProtocol | None) -> str:
    if tensor is None:
        return "Data not available"
    if tensor.size < _TENSOR_DISPLAY_LIMIT:
        try:
            array = tensor.numpy()
            if tensor.dtype == ir.DataType.BFLOAT16:
                array.astype(ml_dtypes.bfloat16)
            elif tensor.dtype == ir.DataType.FLOAT8E4M3FN:
                array.astype(ml_dtypes.float8_e4m3fn)
            elif tensor.dtype == ir.DataType.FLOAT8E4M3FNUZ:
                array.astype(ml_dtypes.float8_e4m3fnuz)
            elif tensor.dtype == ir.DataType.FLOAT8E5M2:
                array.astype(ml_dtypes.float8_e5m2)
            elif tensor.dtype == ir.DataType.FLOAT8E5M2FNUZ:
                array.astype(ml_dtypes.float8_e5m2fnuz)
            return np.array2string(array, separator=",")
        except Exception as e:
            logger.warning("Failed to display tensor: %s", e)
            return str(tensor)
    return str(tensor)


def format_shape(shape: ir.ShapeProtocol | None) -> str:
    return str(shape) if shape is not None else "[?]"


def format_type(type: ir.TypeProtocol | None) -> str:
    return str(type) if type is not None else "?"


def format_tensor_shape(value: ir.Value | ir.TensorProtocol) -> str:
    if isinstance(value, ir.Value):
        return f"{format_type(value.type)}{format_shape(value.shape)}"
    return f"{value.dtype or '?'}{format_shape(value.shape)}"


def get_graph_io_node_name(value: ir.Value) -> str:
    return f"[io]{value.name}"


def get_initializer_node_name(value: ir.Value) -> str:
    return f"[initializer]{value.name}"


def add_inputs_metadata(onnx_node: ir.Node, node: graph_builder.GraphNode):
    for i, input_value in enumerate(onnx_node.inputs):
        metadata = graph_builder.MetadataItem(id=str(i), attrs=[])
        if input_value is None:
            metadata.attrs.append(graph_builder.KeyValue(key="__tensor_tag", value=""))
        else:
            metadata.attrs.append(
                graph_builder.KeyValue(key="__tensor_tag", value=input_value.name or "")
            )
            # tensor_shape is a special key that is used to display the type and shape of the tensor
            metadata.attrs.append(
                graph_builder.KeyValue(
                    key="tensor_shape", value=format_tensor_shape(input_value)
                )
            )
        node.inputsMetadata.append(metadata)


def add_outputs_metadata(onnx_node: ir.Node, node: graph_builder.GraphNode):
    for output in onnx_node.outputs:
        metadata = graph_builder.MetadataItem(id=str(output.index()), attrs=[])
        metadata.attrs.append(
            graph_builder.KeyValue(key="__tensor_tag", value=output.name or "")
        )
        # tensor_shape is a special key that is used to display the type and shape of the tensor
        metadata.attrs.append(
            graph_builder.KeyValue(
                key="tensor_shape", value=format_tensor_shape(output)
            )
        )
        node.outputsMetadata.append(metadata)


def add_node_attrs(onnx_node: ir.Node, node: graph_builder.GraphNode):
    for attr in onnx_node.attributes.values():
        if isinstance(attr, ir.Attr):
            if attr.type == ir.AttributeType.TENSOR:
                attr_value = display_tensor(attr.value)
            else:
                attr_value = str(attr.value)
            node.attrs.append(graph_builder.KeyValue(key=attr.name, value=attr_value))
        elif isinstance(attr, ir.RefAttr):
            node.attrs.append(
                graph_builder.KeyValue(
                    key=attr.name, value=f"Ref({attr.ref_attr_name})"
                )
            )


def add_incoming_edges(
    onnx_node: ir.Node,
    node: graph_builder.GraphNode,
    graph_inputs: set[ir.Value],
):
    for target_input_id, input_value in enumerate(onnx_node.inputs):
        if input_value is None:
            continue
        if input_value in graph_inputs:
            # The input is a graph input. Create an input edge.
            source_node_id = get_graph_io_node_name(input_value)
            source_node_output_id = "0"
        else:
            input_node = input_value.producer()
            if input_node is None:
                logger.debug(
                    "Input value %s does not have a producer. Treating as initializer.",
                    input_value,
                )
                source_node_id = get_initializer_node_name(input_value)
                source_node_output_id = "0"
            elif not input_node.name:
                logger.debug(
                    "Node %s does not have a name. Skipping incoming edge.", input_node
                )
                continue
            else:
                source_node_id = input_node.name
                source_node_output_id = str(input_value.index())
        assert source_node_id is not None
        node.incomingEdges.append(
            graph_builder.IncomingEdge(
                sourceNodeId=source_node_id,
                sourceNodeOutputId=source_node_output_id,
                targetNodeInputId=str(target_input_id),
            )
        )


def create_op_label(domain: str, op_type: str) -> str:
    if domain in {"", "ai.onnx"}:
        return op_type
    return f"{domain}::{op_type}"


def create_node(
    onnx_node: ir.Node, graph_inputs: set[ir.Value], namespace: str = ""
) -> graph_builder.GraphNode | None:
    if onnx_node.name is None:
        logger.warning("Node does not have a name. Skipping node %s.", onnx_node)
        return None

    embedded_namespace = onnx_node.name.lstrip("/").split("/")[0:-1]
    embedded_namespace = [ns or "<anonymous>" for ns in embedded_namespace]
    if embedded_namespace:
        namespace = namespace + "/" + "/".join(embedded_namespace)
    node = graph_builder.GraphNode(
        id=onnx_node.name,
        label=create_op_label(onnx_node.domain, onnx_node.op_type),
        namespace=namespace,
    )
    add_incoming_edges(onnx_node, node, graph_inputs)
    add_node_attrs(onnx_node, node)
    add_inputs_metadata(onnx_node, node)
    add_outputs_metadata(onnx_node, node)
    return node


def add_graph_io(
    graph: graph_builder.Graph,
    input_or_outputs: Sequence[ir.Value],
    type: Literal["Input", "Output"],
    all_nodes: dict[str, graph_builder.GraphNode],
):
    for value in input_or_outputs:
        node = graph_builder.GraphNode(
            id=get_graph_io_node_name(value),
            label=value.name,  # type: ignore
        )
        producer = value.producer()
        if producer is not None and producer.name is not None:
            node.incomingEdges.append(
                graph_builder.IncomingEdge(
                    sourceNodeId=producer.name,
                    sourceNodeOutputId=str(value.index()),
                    targetNodeInputId="0",
                )
            )
        if type == "Input":
            metadata = graph_builder.MetadataItem(id="0", attrs=[])
            metadata.attrs.append(
                graph_builder.KeyValue(key="__tensor_tag", value=value.name or "")
            )
            # tensor_shape is a special key that is used to display the type and shape of the tensor
            metadata.attrs.append(
                graph_builder.KeyValue(
                    key="tensor_shape", value=format_tensor_shape(value)
                )
            )
            node.outputsMetadata.append(metadata)
        node.attrs.append(graph_builder.KeyValue(key="type", value=type))
        graph.nodes.append(node)
        # Record nodes for quick lookup
        all_nodes[node.id] = node


def add_initializers(
    graph: graph_builder.Graph,
    initializers: Sequence[ir.Value],
    namespace: str,
    all_nodes: dict[str, graph_builder.GraphNode],
):
    for initializer in initializers:
        if initializer.name in all_nodes:
            # The initializer is also a graph input. Fill in the missing metadata.
            node = all_nodes[initializer.name]
            metadata = node.outputsMetadata[0]
            metadata.attrs.append(
                graph_builder.KeyValue(
                    key="value", value=display_tensor(initializer.const_value)
                )
            )
            continue
        node = graph_builder.GraphNode(
            id=get_initializer_node_name(initializer),
            label=initializer.name,  # type: ignore
            namespace=namespace,
        )
        # Annotate the initializer node as an initializer
        node.attrs.append(graph_builder.KeyValue(key="type", value="Initializer"))
        # Add metadata for the output tensor
        if initializer.const_value is None:
            logger.warning(
                "Initializer %s does not have a const value. Skipping.", initializer
            )
            graph.nodes.append(node)
            continue
        metadata = graph_builder.MetadataItem(id="0", attrs=[])
        metadata.attrs.append(
            graph_builder.KeyValue(key="__tensor_tag", value=initializer.name or "")
        )
        # tensor_shape is a special key that is used to display the type and shape of the tensor
        metadata.attrs.append(
            graph_builder.KeyValue(
                key="tensor_shape",
                value=f"{initializer.const_value.dtype}{format_shape(initializer.const_value.shape)}",
            )
        )
        metadata.attrs.append(
            graph_builder.KeyValue(
                key="value", value=display_tensor(initializer.const_value)
            )
        )
        node.outputsMetadata.append(metadata)
        graph.nodes.append(node)


def create_graph(onnx_graph: ir.Graph | ir.Function) -> graph_builder.Graph | None:
    if onnx_graph.name is None:
        logger.warning("Graph does not have a name. skipping graph: %s", onnx_graph)
        return None
    graph = graph_builder.Graph(id=onnx_graph.name or "graph", nodes=[])
    graph_inputs = set(onnx_graph.inputs)
    all_nodes = {}
    add_graph_io(graph, onnx_graph.inputs, type="Input", all_nodes=all_nodes)

    for onnx_node in onnx_graph:
        node = create_node(onnx_node, graph_inputs, namespace=onnx_graph.name)  # type: ignore
        if node is None:
            continue
        graph.nodes.append(node)
        all_nodes[node.id] = node

    # Add initializers
    if isinstance(onnx_graph, ir.Graph):
        add_initializers(
            graph,
            list(onnx_graph.initializers.values()),
            onnx_graph.name,
            all_nodes=all_nodes,
        )
    add_graph_io(graph, onnx_graph.outputs, type="Output", all_nodes=all_nodes)
    return graph


class ONNXAdapter(model_explorer.Adapter):
    """Adapter for ONNX models."""

    metadata = model_explorer.AdapterMetadata(
        id="onnx_adapter",
        name="ONNX adapter",
        description="ONNX adapter for Model Explorer",
        source_repo="https://github.com/justinchuby/model_explorer_onnx",
        fileExts=["onnx", "onnxtext", "onnxtxt", "textproto", "onnxjson", "json"],
    )

    def convert(
        self, model_path: str, settings: dict[str, Any]
    ) -> model_explorer.ModelExplorerGraphs:
        del settings  # Unused

        onnx_model = onnx.load(model_path, load_external_data=False)
        try:
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        except Exception as e:
            logger.warning(
                "Failed to infer shapes. Continue with the original model. Error: %s", e
            )

        # Load external data after shape inference
        model_filepath = os.path.abspath(model_path)
        base_dir = os.path.dirname(model_filepath)
        onnx.load_external_data_for_model(onnx_model, base_dir)

        # Convert to ONNX IR
        model = ir.serde.deserialize_model(onnx_model)
        main_graph = create_graph(model.graph)
        graphs = []
        main_graph = create_graph(model.graph)
        assert main_graph is not None
        graphs.append(main_graph)

        # TODO: Better support functions and subgraphs
        for function in model.functions.values():
            function_graph = create_graph(function)
            assert function_graph is not None
            graphs.append(function_graph)
        return {"graphs": graphs}
