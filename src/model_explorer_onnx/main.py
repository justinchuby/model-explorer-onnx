from __future__ import annotations

import json
import logging
from typing import Any, Literal, Sequence

import ml_dtypes
import model_explorer
import numpy as np
import onnx
from model_explorer import graph_builder as gb
from onnxscript import ir

logger = logging.getLogger(__name__)

_DEFAULT_OPSET_VERSION = 18


class Settings:
    def __init__(self, const_element_count_limit: int = 1024, **_: Any):
        self.const_element_count_limit: int = const_element_count_limit


def _tensor_to_numpy(tensor: ir.TensorProtocol) -> np.ndarray:
    array = tensor.numpy()
    if tensor.dtype == ir.DataType.BFLOAT16:
        array = array.view(ml_dtypes.bfloat16)
    elif tensor.dtype == ir.DataType.FLOAT8E4M3FN:
        array = array.view(ml_dtypes.float8_e4m3fn)
    elif tensor.dtype == ir.DataType.FLOAT8E4M3FNUZ:
        array = array.view(ml_dtypes.float8_e4m3fnuz)
    elif tensor.dtype == ir.DataType.FLOAT8E5M2:
        array = array.view(ml_dtypes.float8_e5m2)
    elif tensor.dtype == ir.DataType.FLOAT8E5M2FNUZ:
        array = array.view(ml_dtypes.float8_e5m2fnuz)
    return array


def display_tensor_repr(tensor: ir.TensorProtocol | None) -> str:
    if tensor is None:
        return "Data not available"
    return str(tensor)


def can_display_tensor_json(
    tensor: ir.TensorProtocol | None, settings: Settings
) -> bool:
    """Check if the tensor can be displayed as JSON."""
    del settings  # Unused
    if tensor is None:
        return False
    if isinstance(tensor, ir.ExternalTensor):
        return False
    return True


def display_tensor_json(
    tensor: ir.TensorProtocol | np.ndarray, settings: Settings
) -> str:
    try:
        if isinstance(tensor, np.ndarray):
            array = tensor
        else:
            array = _tensor_to_numpy(tensor)
        size_limit = settings.const_element_count_limit
        if size_limit < 0 or size_limit >= array.size:
            # Use separators=(',', ':') to remove spaces
            return json.dumps(array.tolist(), separators=(",", ":"))
        # Show the first `size_limit` elements if the tensor is too large
        return json.dumps(
            (array.flatten())[:size_limit].tolist(), separators=(",", ":")
        )
    except Exception as e:
        logger.warning("Failed to display tensor (%s): %s", tensor, e)
    return ""


def format_shape(shape: ir.ShapeProtocol | None) -> str:
    return str(shape) if shape is not None else "[?]"


def format_type(type_: ir.TypeProtocol | None) -> str:
    return str(type_) if type_ is not None else "?"


def format_tensor_shape(value: ir.Value | ir.TensorProtocol) -> str:
    if isinstance(value, ir.Value):
        return f"{format_type(value.type)}{format_shape(value.shape)}"
    return f"{value.dtype or '?'}{format_shape(value.shape)}"


def get_value_node_name(value: ir.Value) -> str:
    """Create name for node that is created from a value, that is for visualization only. E.g. Input."""
    return f"[value] {value.name}"


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


def set_attr(obj: gb.GraphNode | gb.MetadataItem, key: str, value: str) -> None:
    """Set an attribute on a GraphNode or MetadataItem."""
    obj.attrs.append(gb.KeyValue(key=key, value=value))


def set_type_shape_metadata(
    metadata: gb.MetadataItem, value: ir.Value | ir.TensorProtocol
) -> None:
    # tensor_shape is a special key that is used to display the type and shape of the tensor
    set_attr(metadata, "tensor_shape", format_tensor_shape(value))


def set_metadata_props(metadata: gb.MetadataItem, value: ir.Value) -> None:
    for prop_key, prop_value in value.metadata_props.items():
        set_attr(metadata, f"[metadata] {prop_key}", prop_value)


def add_inputs_metadata(
    onnx_node: ir.Node, node: gb.GraphNode, opset_version: int
) -> None:
    if onnx.defs.has(onnx_node.op_type, max_inclusive_version=opset_version):
        schema = onnx.defs.get_schema(
            onnx_node.op_type, max_inclusive_version=opset_version
        )
    else:
        schema = None
    for i, input_value in enumerate(onnx_node.inputs):
        metadata = gb.MetadataItem(id=str(i), attrs=[])
        if input_value is None:
            set_attr(metadata, "__tensor_tag", "None")
        else:
            set_attr(metadata, "__tensor_tag", input_value.name or "None")
            set_type_shape_metadata(metadata, input_value)
            set_metadata_props(metadata, input_value)
        if schema is not None:
            if (param_name := get_node_input_param_name(schema, i)) is not None:
                set_attr(metadata, "param_name", param_name)
        node.inputsMetadata.append(metadata)


def add_outputs_metadata(onnx_node: ir.Node, node: gb.GraphNode, opset_version: int):
    if onnx.defs.has(onnx_node.op_type, max_inclusive_version=opset_version):
        schema = onnx.defs.get_schema(
            onnx_node.op_type, max_inclusive_version=opset_version
        )
    else:
        schema = None
    for output_value in onnx_node.outputs:
        metadata = gb.MetadataItem(id=str(output_value.index()), attrs=[])
        set_attr(metadata, "__tensor_tag", output_value.name or "None")
        set_type_shape_metadata(metadata, output_value)
        set_metadata_props(metadata, output_value)
        if len(output_value.uses()) == 0 and not output_value.is_graph_output():
            # The output is unused. Add a flag to indicate that.
            set_attr(metadata, "unused", "True")
        if schema is not None:
            output_index = output_value.index()
            assert output_index is not None
            if (
                param_name := get_node_output_param_name(schema, output_index)
            ) is not None:
                set_attr(metadata, "param_name", param_name)
        node.outputsMetadata.append(metadata)


def add_node_attrs(onnx_node: ir.Node, node: gb.GraphNode, settings: Settings) -> None:
    for attr in onnx_node.attributes.values():
        if isinstance(attr, ir.Attr):
            if attr.type == ir.AttributeType.TENSOR:
                if onnx_node.op_type in {"Constant", "Initializer"}:
                    if can_display_tensor_json(attr.value, settings=settings):
                        assert attr.value is not None
                        set_attr(
                            node,
                            "__value",
                            display_tensor_json(attr.value, settings=settings),
                        )
                    attr_value = display_tensor_repr(attr.value)
                else:
                    if can_display_tensor_json(attr.value, settings=settings):
                        attr_value = display_tensor_json(attr.value, settings=settings)
                    else:
                        attr_value = display_tensor_repr(attr.value)
            elif onnx_node.op_type == "Constant" and attr.name in {
                "value_float",
                "value_int",
                "value_string",
                "value_floats",
                "value_ints",
                "value_strings",
            }:
                set_attr(
                    node,
                    "__value",
                    display_tensor_json(np.array(attr.value), settings=settings),
                )
                attr_value = str(attr.value)
            elif onnx_node.op_type == "Cast" and attr.name == "to":
                attr_value = str(ir.DataType(attr.value))
            else:
                attr_value = str(attr.value)
            set_attr(node, attr.name, attr_value)
        elif isinstance(attr, ir.RefAttr):
            set_attr(node, attr.name, f"@{attr.ref_attr_name}")

    if onnx_node.doc_string:
        set_attr(node, "[metadata] doc_string", onnx_node.doc_string)
    for prop_key, prop_value in onnx_node.metadata_props.items():
        set_attr(node, f"[metadata] {prop_key}", str(prop_value))


def add_incoming_edges(
    onnx_node: ir.Node,
    node: gb.GraphNode,
    graph_inputs: set[ir.Value],
) -> None:
    for target_input_id, input_value in enumerate(onnx_node.inputs):
        if input_value is None:
            continue
        if input_value in graph_inputs:
            # The input is a graph input. Create an input edge.
            source_node_id = get_value_node_name(input_value)
            source_node_output_id = "0"
        else:
            input_node = input_value.producer()
            if input_node is None:
                logger.debug(
                    "Input value %s does not have a producer. Treating as initializer.",
                    input_value,
                )
                source_node_id = get_value_node_name(input_value)
                source_node_output_id = "0"
            else:
                assert input_node.name, "Bug: Node name is required"
                source_node_id = input_node.name
                source_node_output_id = str(input_value.index())
        assert source_node_id is not None
        node.incomingEdges.append(
            gb.IncomingEdge(
                sourceNodeId=source_node_id,
                sourceNodeOutputId=source_node_output_id,
                targetNodeInputId=str(target_input_id),
            )
        )


def create_op_label(domain: str, op_type: str) -> str:
    if domain in {"", "ai.onnx"}:
        return op_type
    return f"{domain}::{op_type}"


def _parse_namespace(node_name: str) -> list[str]:
    """Parse the namespace from the node name if it is in the format of /namespace/node_name."""
    split = node_name.lstrip("/").rstrip("/").split("/")
    return [ns or "<anonymous>" for ns in split]


def get_node_namespace(node: ir.Node) -> list[str]:
    """Get the namespace from the node."""
    if (metadata_namespace := node.metadata_props.get("namespace")) is not None:
        return _parse_namespace(metadata_namespace)
    if node.name:
        # Remove the last part of the node name to get the namespace
        return _parse_namespace(node.name)[:-1]
    return []


def create_node(
    onnx_node: ir.Node,
    graph_inputs: set[ir.Value],
    namespace: str,
    all_function_ids: set[ir.OperatorIdentifier],
    opset_version: int,
    settings: Settings,
) -> gb.GraphNode | None:
    """Create a GraphNode from an ONNX node.

    Args:
        onnx_node: The ONNX node to convert.
        graph_inputs: The set of graph inputs.
        namespace: The namespace of the node.
        all_function_ids: The set of all function identifiers.
        opset_version: The current ONNX opset version.
    """
    assert onnx_node.name, "Bug: Node name is required"

    if onnx_node.op_type == "Constant":
        # Move the constant closer to the user node's namespace
        namespace = get_constant_namespace(onnx_node.outputs[0], namespace)
    else:
        embedded_namespace = get_node_namespace(onnx_node)
        if embedded_namespace:
            namespace = namespace + "/" + "/".join(embedded_namespace)
    node = gb.GraphNode(
        id=onnx_node.name,
        label=create_op_label(onnx_node.domain, onnx_node.op_type),
        namespace=namespace,
    )
    add_incoming_edges(onnx_node, node, graph_inputs)
    add_node_attrs(onnx_node, node, settings=settings)
    add_inputs_metadata(onnx_node, node, opset_version=opset_version)
    add_outputs_metadata(onnx_node, node, opset_version=opset_version)
    if onnx_node.op_identifier() in all_function_ids:
        node.subgraphIds.append(get_function_graph_name(onnx_node.op_identifier()))
    return node


def add_graph_io(
    graph: gb.Graph,
    input_or_outputs: Sequence[ir.Value],
    type_: Literal["Input", "Output"],
    all_nodes: dict[str, gb.GraphNode],
) -> None:
    for i, value in enumerate(input_or_outputs):
        node = gb.GraphNode(
            id=get_value_node_name(value),
            label=type_,
        )
        producer = value.producer()
        if producer is not None:
            assert producer.name, "Bug: Node name is required"
            node.incomingEdges.append(
                gb.IncomingEdge(
                    sourceNodeId=producer.name,
                    sourceNodeOutputId=str(value.index()),
                    targetNodeInputId="0",
                )
            )
        if type_ == "Input":
            metadata = gb.MetadataItem(id="0", attrs=[])
            set_attr(metadata, "__tensor_tag", value.name or "")
            set_type_shape_metadata(metadata, value)
            set_metadata_props(metadata, value)
            node.outputsMetadata.append(metadata)
        set_attr(node, "name", value.name or "")
        set_attr(node, "index", str(i))
        graph.nodes.append(node)
        # Record nodes for quick lookup
        all_nodes[node.id] = node


def get_constant_namespace(initializer: ir.Value, root_namespace: str) -> str:
    """Move the constant/initializer closer to the user's namespace."""
    initializer_namespace = root_namespace
    # A single node can have multiple uses of the same value.
    # Here we only count the unique nodes that use the initializer to push the
    # initializer to the same namespace as much as possible.
    user_nodes = tuple(set(node for node, _ in initializer.uses()))
    if not user_nodes:
        # The initializer is not used by any node. Keep it in the root namespace.
        return initializer_namespace
    if len(user_nodes) == 1:
        # If the initializer is used by a single node, move it to the same namespace as the node
        user_node = user_nodes[0]
        user_node_namespace = get_node_namespace(user_node)
        if user_node_namespace:
            initializer_namespace = (
                initializer_namespace + "/" + "/".join(user_node_namespace)
            )
    else:
        # If there are multiple user nodes, find the common namespace
        common_namespace = get_node_namespace(user_nodes[0])
        for user_node in user_nodes:
            user_node_namespace = get_node_namespace(user_node)
            for i, (name_a, name_b) in enumerate(
                zip(common_namespace, user_node_namespace)
            ):
                if name_a != name_b:
                    # That's the end of the common namespace
                    common_namespace = common_namespace[:i]
                    break
        if common_namespace:
            initializer_namespace = (
                initializer_namespace + "/" + "/".join(common_namespace)
            )
    return initializer_namespace


def add_initializers(
    graph: gb.Graph,
    initializers: Sequence[ir.Value],
    namespace: str,
    all_nodes: dict[str, gb.GraphNode],
    settings: Settings,
) -> None:
    for initializer in initializers:
        if not initializer.name:
            logger.warning(
                "Initializer does not have a name. Skipping: %s", initializer
            )
            continue
        initializer_node_name = get_value_node_name(initializer)
        if initializer_node_name in all_nodes:
            # The initializer is also a graph input.
            # Convert it into an InitializedInput and fill in the missing metadata
            node = all_nodes[initializer_node_name]
            node.label = "InitializedInput"
            # Push the initializer closer to the user node's namespace
            node.namespace = get_constant_namespace(initializer, namespace)
            # Display the constant value
            if can_display_tensor_json(initializer.const_value, settings=settings):
                assert initializer.const_value is not None
                set_attr(
                    node,
                    "__value",
                    display_tensor_json(initializer.const_value, settings=settings),
                )
            # Set output metadata
            metadata = node.outputsMetadata[0]
            set_attr(metadata, "value", display_tensor_repr(initializer.const_value))
            continue
        node = gb.GraphNode(
            id=initializer_node_name,
            label="Initializer",
            namespace=get_constant_namespace(initializer, namespace),
        )
        # Add metadata for the output tensor
        if initializer.const_value is None:
            logger.warning(
                "Initializer %s does not have a const value. Skipping.", initializer
            )
            graph.nodes.append(node)
            continue
        metadata = gb.MetadataItem(id="0", attrs=[])
        set_attr(metadata, "__tensor_tag", initializer.name or "")
        set_type_shape_metadata(metadata, initializer.const_value)
        if can_display_tensor_json(initializer.const_value, settings=settings):
            assert initializer.const_value is not None
            set_attr(
                node,
                "__value",
                display_tensor_json(initializer.const_value, settings=settings),
            )
        set_attr(metadata, "value", display_tensor_repr(initializer.const_value))
        set_metadata_props(metadata, initializer)
        # Note if the initializer is unused
        if not initializer.uses():
            set_attr(metadata, "unused", "True")
        node.outputsMetadata.append(metadata)
        graph.nodes.append(node)


def create_graph(
    onnx_graph: ir.Graph | ir.Function,
    all_function_ids: set[ir.OperatorIdentifier],
    opset_version: int,
    settings: Settings,
    attrs: dict[str, Any],
) -> gb.Graph | None:
    if isinstance(onnx_graph, ir.Function):
        graph_name = get_function_graph_name(onnx_graph.identifier())
    elif onnx_graph.name is None:
        logger.warning("Graph does not have a name. skipping graph: %s", onnx_graph)
        return None
    else:
        graph_name = onnx_graph.name
    graph = gb.Graph(
        id=graph_name,
        nodes=[],
        groupNodeAttributes={
            "": {key: str(value) for key, value in attrs.items()},
            graph_name: {
                f"[metadata] {key}": str(value)
                for key, value in onnx_graph.metadata_props.items()
            },
        },
    )
    graph_inputs = set(onnx_graph.inputs)
    all_nodes = {}
    add_graph_io(graph, onnx_graph.inputs, type_="Input", all_nodes=all_nodes)

    for i, onnx_node in enumerate(onnx_graph):
        if not onnx_node.name:
            onnx_node.name = f"<node_{i}>"
        node = create_node(
            onnx_node,
            graph_inputs,  # type: ignore
            namespace=graph_name,
            all_function_ids=all_function_ids,
            opset_version=opset_version,
            settings=settings,
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
            settings=settings,
        )

    # Add outputs
    add_graph_io(graph, onnx_graph.outputs, type_="Output", all_nodes=all_nodes)
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
        parsed_settings = Settings(**settings)

        # Do not load external data because the model file is copied to a temporary location
        # and the external data paths are not valid anymore.
        onnx_model = onnx.load(model_path, load_external_data=False)
        try:
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)
        except Exception as e:
            logger.warning(
                "Failed to infer shapes. Continue with the original model. Error: %s", e
            )

        # Convert to ONNX IR
        model = ir.serde.deserialize_model(onnx_model)
        all_function_ids = set(model.functions)
        graphs = []
        opset_version = model.opset_imports.get("")
        if opset_version is None:
            opset_version = model.opset_imports.get("ai.onnx")
        if opset_version is None:
            opset_version = _DEFAULT_OPSET_VERSION
        # TODO: Better support subgraphs in nodes
        if model.graph.name is None:
            model.graph.name = "<main>"
            logger.warning(
                "Main graph of ONNX file '%s' does not have a name. Set name to '<main>'",
                model_path,
            )
        main_graph = create_graph(
            model.graph,
            all_function_ids,
            opset_version=opset_version,
            settings=parsed_settings,
            attrs={
                "opset_imports": model.opset_imports,
                "producer_name": model.producer_name,
                "producer_version": model.producer_version,
                "domain": model.domain,
                "model_version": model.model_version,
                "doc_string": model.doc_string,
                **{
                    f"[metadata] {key}": value
                    for key, value in model.metadata_props.items()
                },
            },
        )
        assert main_graph is not None, "Bug: Main graph should not be None"
        graphs.append(main_graph)

        for function in model.functions.values():
            function_graph = create_graph(
                function,
                all_function_ids,
                opset_version=opset_version,
                settings=parsed_settings,
                attrs={
                    "opset_imports": model.opset_imports,
                    "attributes": function.attributes,
                    "doc_string": function.doc_string,
                    **{
                        f"[metadata] {key}": value
                        for key, value in function.metadata_props.items()
                    },
                },
            )
            assert function_graph is not None
            graphs.append(function_graph)
        return {"graphs": graphs}
