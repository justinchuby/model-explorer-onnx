from __future__ import annotations

from typing import Any, Mapping, Sequence
import model_explorer
from model_explorer import graph_builder
import onnx
from onnxscript import ir
import logging

logger = logging.getLogger(__name__)

_TENSOR_DISPLAY_LIMIT = 1024


def display_tensor(tensor: ir.TensorProtocol) -> str:
    if tensor.size < _TENSOR_DISPLAY_LIMIT:
        try:
            return str(tensor.numpy())
        except Exception as e:
            logger.warning("Failed to display tensor: %s", e)
            return str(tensor)
    return str(tensor)


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
        type_str = str(output.type)
        shape_text = str(output.shape) if output.shape is not None else "[?]"

        metadata.attrs.append(
            graph_builder.KeyValue(key="__tensor_tag", value=output.name or "")
        )
        # tensor_shape is a special key that is used to display the type and shape of the tensor
        metadata.attrs.append(
            graph_builder.KeyValue(key="tensor_shape", value=f"{type_str}{shape_text}")
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
    graph_inputs: Mapping[ir.Value, int],
):
    for target_input_id, input_value in enumerate(onnx_node.inputs):
        if input_value is None:
            continue
        if input_value in graph_inputs:
            # The input is a graph input. Create an input edge.
            node.incomingEdges.append(
                graph_builder.IncomingEdge(
                    sourceNodeId=input_value.name,  # type: ignore
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
    onnx_node: ir.Node, graph_inputs: Mapping[ir.Value, int], namespace: str = ""
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


def add_graph_io(graph: graph_builder.Graph, input_or_outputs: Sequence[ir.Value]):
    for value in input_or_outputs:
        node = graph_builder.GraphNode(
            id=value.name,  # type: ignore
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

        graph.nodes.append(node)


def add_initializers(
    graph: graph_builder.Graph, initializers: Sequence[ir.Value], namespace: str
):
    for initializer in initializers:
        node = graph_builder.GraphNode(
            id=initializer.name,  # type: ignore
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
        shape_text = (
            str(initializer.const_value.shape)
            if initializer.const_value.shape is not None
            else "[?]"
        )
        metadata.attrs.append(
            graph_builder.KeyValue(key="__tensor_tag", value=initializer.name or "")
        )
        # tensor_shape is a special key that is used to display the type and shape of the tensor
        metadata.attrs.append(
            graph_builder.KeyValue(key="tensor_shape", value=f"{initializer.const_value.dtype}{shape_text}")
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
    graph_inputs: dict[ir.Value, int] = {
        input_value: i for i, input_value in enumerate(onnx_graph.inputs)
    }
    for onnx_node in onnx_graph:
        node = create_node(onnx_node, graph_inputs, namespace=onnx_graph.name)
        if node is None:
            continue
        graph.nodes.append(node)
    add_graph_io(graph, onnx_graph.inputs)
    # Add initializers
    if isinstance(onnx_graph, ir.Graph):
        add_initializers(graph, list(onnx_graph.initializers.values()), onnx_graph.name)
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
        # TODO: Add shape inference and tensor display size as settings
        onnx_model = onnx.load(model_path)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        model = ir.serde.deserialize_model(onnx_model)
        main_graph = create_graph(model.graph)
        graphs = []
        main_graph = create_graph(model.graph)
        assert main_graph is not None
        graphs.append(main_graph)
        for function in model.functions.values():
            function_graph = create_graph(function)
            assert function_graph is not None
            graphs.append(function_graph)
        return {"graphs": graphs}
