from __future__ import annotations

import logging
import os
from typing import Any, Literal, Sequence

import ml_dtypes
import model_explorer
import numpy as np
import onnx
from model_explorer import graph_builder
from onnxscript import ir

logger = logging.getLogger(__name__)

_TENSOR_DISPLAY_LIMIT = 1024
_DEFAULT_OPSET_VERSION = 18


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


def get_function_graph_name(identifier: ir.OperatorIdentifier) -> str:
    name = f"[function]{identifier[0]}::{identifier[1]}"
    if identifier[2]:
        name += f"::{identifier[2]}"
    return name


def get_node_input_param_name(
    schema: onnx.defs.OpSchema, input_index: int
) -> str | None:
    """Get the name of the input parameter of the node from OpSchema."""
    try:
        if len(schema.inputs) == 0:
            # Invalid schema.
            return None
        if input_index < len(schema.inputs):
            return schema.inputs[input_index].name
        if (
            schema.inputs[-1].option
            == onnx.defs.OpSchema.FormalParameterOption.Variadic
        ):
            # The last input is variadic. Return the name of the last input.
            return schema.inputs[-1].name
        return None
    except Exception as e:
        logger.warning("Failed to get input schema name: %s", e)
    return None


def get_node_output_param_name(
    schema: onnx.defs.OpSchema, output_index: int
) -> str | None:
    """Get the name of the output parameter of the node from OpSchema."""
    try:
        if len(schema.outputs) == 0:
            # Invalid schema. Return the output index as a fallback.
            return None
        if output_index < len(schema.outputs):
            return schema.outputs[output_index].name
        if (
            schema.outputs[-1].option
            == onnx.defs.OpSchema.FormalParameterOption.Variadic
        ):
            # The last input is variadic. Return the name of the last input.
            return schema.outputs[-1].name
        return None
    except Exception as e:
        logger.warning("Failed to get output schema name: %s", e)


def add_inputs_metadata(
    onnx_node: ir.Node, node: graph_builder.GraphNode, opset_version: int
):
    if onnx.defs.has(onnx_node.op_type, max_inclusive_version=opset_version):
        schema = onnx.defs.get_schema(
            onnx_node.op_type, max_inclusive_version=opset_version
        )
    else:
        schema = None
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
        if schema is not None:
            if (param_name := get_node_input_param_name(schema, i)) is not None:
                metadata.attrs.append(
                    graph_builder.KeyValue(key="param_name", value=param_name)
                )
        node.inputsMetadata.append(metadata)


def add_outputs_metadata(
    onnx_node: ir.Node, node: graph_builder.GraphNode, opset_version: int
):
    if onnx.defs.has(onnx_node.op_type, max_inclusive_version=opset_version):
        schema = onnx.defs.get_schema(
            onnx_node.op_type, max_inclusive_version=opset_version
        )
    else:
        schema = None
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
        if schema is not None:
            output_index = output.index()
            assert output_index is not None
            if (
                param_name := get_node_output_param_name(schema, output_index)
            ) is not None:
                metadata.attrs.append(
                    graph_builder.KeyValue(key="param_name", value=param_name)
                )
        node.outputsMetadata.append(metadata)


def add_node_attrs(onnx_node: ir.Node, node: graph_builder.GraphNode):
    for attr in onnx_node.attributes.values():
        if isinstance(attr, ir.Attr):
            if attr.type == ir.AttributeType.TENSOR:
                attr_value = display_tensor(attr.value)
            elif onnx_node.op_type == "Cast" and attr.name == "to":
                attr_value = str(ir.DataType(attr.value))
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
            else:
                assert input_node.name, "Bug: Node name is required"
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
    onnx_node: ir.Node,
    graph_inputs: set[ir.Value],
    namespace: str,
    all_function_ids: set[ir.OperatorIdentifier],
    opset_version: int,
) -> graph_builder.GraphNode | None:
    """Create a GraphNode from an ONNX node.

    Args:
        onnx_node: The ONNX node to convert.
        graph_inputs: The set of graph inputs.
        namespace: The namespace of the node.
        all_function_ids: The set of all function identifiers.
        opset_version: The current ONNX opset version.
    """
    assert onnx_node.name, "Bug: Node name is required"

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
    add_inputs_metadata(onnx_node, node, opset_version=opset_version)
    add_outputs_metadata(onnx_node, node, opset_version=opset_version)
    if onnx_node.op_identifier() in all_function_ids:
        node.subgraphIds.append(get_function_graph_name(onnx_node.op_identifier()))
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
        if producer is not None:
            assert producer.name, "Bug: Node name is required"
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
        initializer_node_name = get_initializer_node_name(initializer)
        if initializer_node_name in all_nodes:
            # The initializer is also a graph input. Fill in the missing metadata.
            node = all_nodes[initializer_node_name]
            metadata = node.outputsMetadata[0]
            metadata.attrs.append(
                graph_builder.KeyValue(
                    key="value", value=display_tensor(initializer.const_value)
                )
            )
            continue
        node = graph_builder.GraphNode(
            id=initializer_node_name,
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


def create_graph(
    onnx_graph: ir.Graph | ir.Function,
    all_function_ids: set[ir.OperatorIdentifier],
    opset_version: int,
) -> graph_builder.Graph | None:
    if isinstance(onnx_graph, ir.Function):
        graph_name = get_function_graph_name(onnx_graph.identifier())
    elif onnx_graph.name is None:
        logger.warning("Graph does not have a name. skipping graph: %s", onnx_graph)
        return None
    else:
        graph_name = onnx_graph.name
    graph = graph_builder.Graph(id=graph_name, nodes=[])
    graph_inputs = set(onnx_graph.inputs)
    all_nodes = {}
    add_graph_io(graph, onnx_graph.inputs, type="Input", all_nodes=all_nodes)

    for i, onnx_node in enumerate(onnx_graph):
        if not onnx_node.name:
            onnx_node.name = f"<node_{i}>"
        node = create_node(
            onnx_node,
            graph_inputs,  # type: ignore
            namespace=graph_name,
            all_function_ids=all_function_ids,
            opset_version=opset_version,
        )  # type: ignore
        if node is None:
            continue
        graph.nodes.append(node)
        all_nodes[node.id] = node

    # Add initializers
    if isinstance(onnx_graph, ir.Graph):
        add_initializers(
            graph,
            list(onnx_graph.initializers.values()),
            graph_name,
            all_nodes=all_nodes,
        )

    # Add outputs
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
        all_function_ids = set(model.functions)
        graphs = []
        opset_version = model.opset_imports.get("", _DEFAULT_OPSET_VERSION)
        # TODO: Better support subgraphs in nodes
        main_graph = create_graph(
            model.graph, all_function_ids, opset_version=opset_version
        )
        assert main_graph is not None
        graphs.append(main_graph)

        for function in model.functions.values():
            function_graph = create_graph(
                function, all_function_ids, opset_version=opset_version
            )
            assert function_graph is not None
            graphs.append(function_graph)
        return {"graphs": graphs}
