# NOTE: This file is web-adapted and maintained alongside source adapter.
# SOURCE_SYNC_FILES: src/model_explorer_onnx/main.py, src/model_explorer_onnx/_passes.py
# SOURCE_SYNC_SHA256: 2071ea119e5cf81cad278f42a8de33a8b23962279d35fde9c16f178c0e438005

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Sequence

import numpy as np
import onnx
import onnx_ir as ir
import onnx_ir.passes.common as common_passes

logger = logging.getLogger(__name__)
_DEFAULT_OPSET_VERSION = 18


class Settings:
    def __init__(self, const_element_count_limit: int = 1024, **_: Any):
        self.const_element_count_limit: int = const_element_count_limit


def kv(key: str, value: str) -> dict[str, str]:
    return {"key": key, "value": value}


def metadata_item(item_id: str) -> dict[str, Any]:
    return {"id": item_id, "attrs": []}


def graph_node(node_id: str, label: str, namespace: str = "") -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "namespace": namespace,
        "attrs": [],
        "incomingEdges": [],
        "inputsMetadata": [],
        "outputsMetadata": [],
        "subgraphIds": [],
    }


def set_attr(obj: dict[str, Any], key: str, value: str) -> None:
    obj["attrs"].append(kv(key, value))


def display_tensor_repr(tensor: ir.TensorProtocol | None) -> str:
    if tensor is None:
        return "Data not available"
    return str(tensor)


def can_display_tensor_json(
    tensor: ir.TensorProtocol | None, settings: Settings
) -> bool:
    del settings
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
            array = tensor.numpy()
        size_limit = settings.const_element_count_limit
        if size_limit < 0 or size_limit >= array.size:
            return json.dumps(array.tolist(), separators=(",", ":"))
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
    return f"[value] {value.name}"


def get_function_graph_name(identifier: ir.OperatorIdentifier) -> str:
    name = f"[function]{identifier[0]}::{identifier[1]}"
    if identifier[2]:
        name += f"::{identifier[2]}"
    return name


def get_subgraph_name(node: ir.Node, attr_name: str) -> str:
    return f"{node.name}/{attr_name}"


def _parse_namespace(node_name: str) -> list[str]:
    split = node_name.lstrip("/").rstrip("/").split("/")
    return [ns for ns in split if ns != ""]


def get_node_namespace(node: ir.Node) -> list[str]:
    if (metadata_namespace := node.metadata_props.get("namespace")) is not None:
        return _parse_namespace(metadata_namespace)
    if node.name:
        ns = _parse_namespace(node.name)
        if not ns:
            return []
        return ns[:-1]
    return []


def set_type_shape_metadata(
    metadata: dict[str, Any], value: ir.Value | ir.TensorProtocol
) -> None:
    set_attr(metadata, "tensor_shape", format_tensor_shape(value))


def set_metadata_props(metadata: dict[str, Any], value: ir.Value) -> None:
    for prop_key, prop_value in value.metadata_props.items():
        set_attr(metadata, f"[metadata] {prop_key}", prop_value)


def get_node_input_param_name(
    schema: onnx.defs.OpSchema, input_index: int
) -> str | None:
    try:
        if len(schema.inputs) == 0:
            return None
        if input_index < len(schema.inputs):
            return schema.inputs[input_index].name
        if (
            schema.inputs[-1].option
            == onnx.defs.OpSchema.FormalParameterOption.Variadic
        ):
            return schema.inputs[-1].name
        return None
    except Exception as e:
        logger.warning("Failed to get input schema name: %s", e)
    return None


def get_node_output_param_name(
    schema: onnx.defs.OpSchema, output_index: int
) -> str | None:
    try:
        if len(schema.outputs) == 0:
            return None
        if output_index < len(schema.outputs):
            return schema.outputs[output_index].name
        if (
            schema.outputs[-1].option
            == onnx.defs.OpSchema.FormalParameterOption.Variadic
        ):
            return schema.outputs[-1].name
        return None
    except Exception as e:
        logger.warning("Failed to get output schema name: %s", e)
    return None


def add_inputs_metadata(
    onnx_node: ir.Node, node: dict[str, Any], opset_version: int
) -> None:
    if onnx.defs.has(onnx_node.op_type, max_inclusive_version=opset_version):
        schema = onnx.defs.get_schema(
            onnx_node.op_type, max_inclusive_version=opset_version
        )
    else:
        schema = None
    for i, input_value in enumerate(onnx_node.inputs):
        metadata = metadata_item(str(i))
        if input_value is None:
            set_attr(metadata, "__tensor_tag", "None")
        else:
            set_attr(metadata, "__tensor_tag", input_value.name or "None")
            set_type_shape_metadata(metadata, input_value)
            set_metadata_props(metadata, input_value)
        if schema is not None:
            if (param_name := get_node_input_param_name(schema, i)) is not None:
                set_attr(metadata, "param_name", param_name)
        node["inputsMetadata"].append(metadata)


def add_outputs_metadata(onnx_node: ir.Node, node: dict[str, Any], opset_version: int):
    if onnx.defs.has(onnx_node.op_type, max_inclusive_version=opset_version):
        schema = onnx.defs.get_schema(
            onnx_node.op_type, max_inclusive_version=opset_version
        )
    else:
        schema = None
    for output_value in onnx_node.outputs:
        metadata = metadata_item(str(output_value.index()))
        set_attr(metadata, "__tensor_tag", output_value.name or "None")
        set_type_shape_metadata(metadata, output_value)
        set_metadata_props(metadata, output_value)
        if len(output_value.uses()) == 0 and not output_value.is_graph_output():
            set_attr(metadata, "unused", "True")
        if schema is not None:
            output_index = output_value.index()
            assert output_index is not None
            if (
                param_name := get_node_output_param_name(schema, output_index)
            ) is not None:
                set_attr(metadata, "param_name", param_name)
        node["outputsMetadata"].append(metadata)


def add_node_attrs(
    onnx_node: ir.Node, node: dict[str, Any], settings: Settings
) -> None:
    for attr in onnx_node.attributes.values():
        if isinstance(attr, ir.Attr) and attr.value is not None:
            if attr.type == ir.AttributeType.TENSOR:
                if onnx_node.op_type in {"Constant", "Initializer"}:
                    if can_display_tensor_json(attr.value, settings=settings):
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
        else:
            set_attr(node, attr.name, f"@{attr.ref_attr_name}")
    if onnx_node.doc_string:
        set_attr(node, "[metadata] doc_string", onnx_node.doc_string)
    for prop_key, prop_value in onnx_node.metadata_props.items():
        set_attr(node, f"[metadata] {prop_key}", str(prop_value))


def add_incoming_edges(
    onnx_node: ir.Node,
    node: dict[str, Any],
    graph_inputs: set[ir.Value],
) -> None:
    for target_input_id, input_value in enumerate(onnx_node.inputs):
        if input_value is None:
            continue
        if input_value in graph_inputs:
            source_node_id = get_value_node_name(input_value)
            source_node_output_id = "0"
        else:
            input_node = input_value.producer()
            if input_node is None:
                source_node_id = get_value_node_name(input_value)
                source_node_output_id = "0"
            else:
                assert input_node.name
                source_node_id = input_node.name
                source_node_output_id = str(input_value.index())
        node["incomingEdges"].append(
            {
                "sourceNodeId": source_node_id,
                "sourceNodeOutputId": source_node_output_id,
                "targetNodeInputId": str(target_input_id),
            }
        )


def create_op_label(domain: str, op_type: str) -> str:
    if domain in {"", "ai.onnx"}:
        return op_type
    return f"{domain}::{op_type}"


def get_constant_namespace(initializer: ir.Value, root_namespace: str) -> str:
    initializer_namespace = root_namespace
    user_nodes = tuple(set(node for node, _ in initializer.uses()))
    if not user_nodes:
        return initializer_namespace
    if len(user_nodes) == 1:
        user_node_namespace = get_node_namespace(user_nodes[0])
        if user_node_namespace:
            initializer_namespace = "/".join(
                (initializer_namespace, *user_node_namespace)
            )
    else:
        common_namespace = get_node_namespace(user_nodes[0])
        for user_node in user_nodes:
            user_node_namespace = get_node_namespace(user_node)
            for i, (name_a, name_b) in enumerate(
                zip(common_namespace, user_node_namespace)
            ):
                if name_a != name_b:
                    common_namespace = common_namespace[:i]
                    break
            else:
                common_namespace = common_namespace[: len(user_node_namespace)]
        if common_namespace:
            initializer_namespace = "/".join((initializer_namespace, *common_namespace))
    return initializer_namespace


def create_node(
    onnx_node: ir.Node,
    graph_inputs: set[ir.Value],
    namespace: str,
    all_function_ids: set[ir.OperatorIdentifier],
    opset_version: int,
    settings: Settings,
) -> dict[str, Any] | None:
    assert onnx_node.name
    if onnx_node.op_type == "Constant":
        namespace = get_constant_namespace(onnx_node.outputs[0], namespace)
    else:
        embedded_namespace = get_node_namespace(onnx_node)
        if embedded_namespace:
            namespace = "/".join((namespace, *embedded_namespace))
    node = graph_node(
        node_id=onnx_node.name,
        label=create_op_label(onnx_node.domain, onnx_node.op_type),
        namespace=namespace,
    )
    add_incoming_edges(onnx_node, node, graph_inputs)
    add_node_attrs(onnx_node, node, settings=settings)
    add_inputs_metadata(onnx_node, node, opset_version=opset_version)
    add_outputs_metadata(onnx_node, node, opset_version=opset_version)
    if onnx_node.op_identifier() in all_function_ids:
        node["subgraphIds"].append(get_function_graph_name(onnx_node.op_identifier()))
    for attr in onnx_node.attributes.values():
        if attr.type == ir.AttributeType.GRAPH:
            node["subgraphIds"].append(get_subgraph_name(onnx_node, attr.name))
    return node


def add_graph_io(
    graph: dict[str, Any],
    input_or_outputs: Sequence[ir.Value],
    type_: Literal["Input", "Output"],
    all_nodes: dict[str, dict[str, Any]],
) -> None:
    for i, value in enumerate(input_or_outputs):
        node = graph_node(node_id=get_value_node_name(value), label=type_)
        producer = value.producer()
        if producer is not None:
            assert producer.name
            node["incomingEdges"].append(
                {
                    "sourceNodeId": producer.name,
                    "sourceNodeOutputId": str(value.index()),
                    "targetNodeInputId": "0",
                }
            )
        if type_ == "Input":
            metadata = metadata_item("0")
            set_attr(metadata, "__tensor_tag", value.name or "")
            set_type_shape_metadata(metadata, value)
            set_metadata_props(metadata, value)
            node["outputsMetadata"].append(metadata)
        set_attr(node, "name", value.name or "")
        set_attr(node, "index", str(i))
        graph["nodes"].append(node)
        all_nodes[node["id"]] = node


def add_initializers(
    graph: dict[str, Any],
    initializers: Sequence[ir.Value],
    namespace: str,
    all_nodes: dict[str, dict[str, Any]],
    settings: Settings,
) -> None:
    for initializer in initializers:
        if not initializer.name:
            continue
        initializer_node_name = get_value_node_name(initializer)
        if initializer_node_name in all_nodes:
            node = all_nodes[initializer_node_name]
            node["label"] = "InitializedInput"
            node["namespace"] = get_constant_namespace(initializer, namespace)
            if can_display_tensor_json(initializer.const_value, settings=settings):
                set_attr(
                    node,
                    "__value",
                    display_tensor_json(initializer.const_value, settings=settings),
                )
            metadata = node["outputsMetadata"][0]
            set_attr(metadata, "value", display_tensor_repr(initializer.const_value))
            continue

        node = graph_node(
            node_id=initializer_node_name,
            label="Initializer",
            namespace=get_constant_namespace(initializer, namespace),
        )
        if initializer.const_value is None:
            graph["nodes"].append(node)
            continue
        metadata = metadata_item("0")
        set_attr(metadata, "__tensor_tag", initializer.name or "")
        set_type_shape_metadata(metadata, initializer.const_value)
        if can_display_tensor_json(initializer.const_value, settings=settings):
            set_attr(
                node,
                "__value",
                display_tensor_json(initializer.const_value, settings=settings),
            )
        set_attr(metadata, "value", display_tensor_repr(initializer.const_value))
        set_metadata_props(metadata, initializer)
        if not initializer.uses():
            set_attr(metadata, "unused", "True")
        node["outputsMetadata"].append(metadata)
        graph["nodes"].append(node)


def create_graph(
    onnx_graph: ir.Graph | ir.Function,
    all_function_ids: set[ir.OperatorIdentifier],
    opset_version: int,
    settings: Settings,
    attrs: dict[str, Any],
) -> dict[str, Any] | None:
    if isinstance(onnx_graph, ir.Function):
        graph_name = get_function_graph_name(onnx_graph.identifier())
    elif onnx_graph.name is None:
        return None
    else:
        graph_name = onnx_graph.name

    graph = {
        "id": graph_name,
        "nodes": [],
        "groupNodeAttributes": {
            "": {key: str(value) for key, value in attrs.items()},
            graph_name: {
                f"[metadata] {key}": str(value)
                for key, value in onnx_graph.metadata_props.items()
            },
        },
    }
    graph_inputs = set(onnx_graph.inputs)
    all_nodes: dict[str, dict[str, Any]] = {}
    add_graph_io(graph, onnx_graph.inputs, type_="Input", all_nodes=all_nodes)

    for i, onnx_node in enumerate(onnx_graph):
        if not onnx_node.name:
            onnx_node.name = f"<node_{i}>"
        node = create_node(
            onnx_node,
            graph_inputs,
            namespace=graph_name,
            all_function_ids=all_function_ids,
            opset_version=opset_version,
            settings=settings,
        )
        if node is None:
            continue
        graph["nodes"].append(node)
        all_nodes[node["id"]] = node

    if isinstance(onnx_graph, ir.Graph):
        add_initializers(
            graph,
            list(onnx_graph.initializers.values()),
            graph_name,
            all_nodes=all_nodes,
            settings=settings,
        )

    add_graph_io(graph, onnx_graph.outputs, type_="Output", all_nodes=all_nodes)
    return graph


class AssignNodeNamespacePass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            if "namespace" in node.metadata_props:
                continue
            if node.name:
                node.metadata_props["namespace"] = "/".join(get_node_namespace(node))
                modified = True
        return ir.passes.PassResult(model, modified=modified)


class ImplicitUseAnalysisPass(ir.passes.InPlacePass):
    def _iterate_subgraphs(self, node: ir.Node, implicit_usages, graph_stack):
        def process_node(sub_node: ir.Node, subgraph: ir.Graph):
            for inp in sub_node.inputs:
                if inp is not None and inp.graph is not subgraph:
                    for g in reversed(graph_stack):
                        if g is inp.graph:
                            break
                        implicit_usages[g].append(inp)

        for attr in node.attributes.values():
            if not isinstance(attr, ir.Attr):
                continue
            if attr.type == ir.AttributeType.GRAPH:
                subgraph = attr.as_graph()
                graph_stack.append(subgraph)
                if subgraph not in implicit_usages:
                    implicit_usages[subgraph] = []
                for sub_node in subgraph:
                    process_node(sub_node, subgraph)
                    self._iterate_subgraphs(sub_node, implicit_usages, graph_stack)
                graph_stack.pop()
            elif attr.type == ir.AttributeType.GRAPHS:
                for subgraph in attr.as_graphs():
                    graph_stack.append(subgraph)
                    if subgraph not in implicit_usages:
                        implicit_usages[subgraph] = []
                    for sub_node in subgraph:
                        process_node(sub_node, subgraph)
                        self._iterate_subgraphs(sub_node, implicit_usages, graph_stack)
                    graph_stack.pop()

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph_stack: list[ir.Graph] = []
        implicit_usages: dict[ir.Graph, list[ir.Value]] = {}
        for node in model.graph:
            self._iterate_subgraphs(node, implicit_usages, graph_stack)

        for graph, used_values in implicit_usages.items():
            seen = set()
            closed_values = []
            for val in used_values:
                if val not in seen:
                    seen.add(val)
                    closed_values.append(val)
            graph.meta["implicit_uses"] = closed_values
        return ir.passes.PassResult(model, modified=False)


class AddCaptureNodePass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            if not any(
                attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
                for attr in node.attributes.values()
            ):
                continue
            if node.op_type == "If" and node.domain == "":
                continue

            all_closed_values: list[ir.Value] = []
            for attr in node.attributes.values():
                if attr.type == ir.AttributeType.GRAPH:
                    subgraphs = [attr.as_graph()]
                elif attr.type == ir.AttributeType.GRAPHS:
                    subgraphs = attr.as_graphs()
                else:
                    continue
                for subgraph in subgraphs:
                    closed_values = subgraph.meta.get("implicit_uses", [])
                    all_closed_values.extend(closed_values)

            unique_closed_values: list[ir.Value] = []
            seen_values: set[ir.Value] = set()
            for value in all_closed_values:
                if value in seen_values:
                    continue
                seen_values.add(value)
                unique_closed_values.append(value)

            if not unique_closed_values:
                continue
            capture_node = ir.node(
                "(Capture)",
                inputs=unique_closed_values,
                name=f"{node.name}_capture",
            )
            capture_node.outputs[0].name = f"{node.name}_captured_output"
            graph = node.graph
            if graph is not None:
                graph.insert_before(node, capture_node)
                original_input_count = len(node.inputs)
                node.resize_inputs(original_input_count + 1)
                node.replace_input_with(original_input_count, capture_node.outputs[0])
                modified = True
        return ir.passes.PassResult(model, modified=modified)


class EmbedIfPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            if node.op_type != "If" or node.domain != "":
                continue
            modified = True
            subgraphs: list[tuple[str, ir.Graph]] = []
            parent_namespace = get_node_namespace(node)
            parent_namespace.append(f"{node.op_type} <{node.name}>")
            node.metadata_props["namespace"] = "/".join(parent_namespace)
            for name, attr in node.attributes.items():
                if attr.type == ir.AttributeType.GRAPH:
                    subgraphs.append((name, attr.as_graph()))

            if not subgraphs:
                continue

            outputs = []
            last_node = None
            for attr_name, subgraph in subgraphs:
                for sub_node in subgraph:
                    sub_namespace = get_node_namespace(sub_node)
                    sub_node.metadata_props["namespace"] = "/".join(
                        (*parent_namespace, attr_name, *sub_namespace)
                    )
                    sub_node.name = f"{node.name}/{attr_name}/{sub_node.name}"
                    for output in sub_node.outputs:
                        output.name = f"{node.name}/{attr_name}/{output.name}"
                for idx, g_output in enumerate(subgraph.outputs):
                    g_output.metadata_props["graph_output_index"] = str(idx)
                node.attributes.pop(attr_name)
                outputs.extend(subgraph.outputs)
                subgraph.outputs.clear()
                sub_nodes = tuple(subgraph)
                subgraph.remove(sub_nodes)
                node.append(sub_nodes)
                last_node = sub_nodes[-1]
            assert last_node is not None
            phi_node = ir.node(
                "(Phi)",
                inputs=(*([None] * len(node.outputs)), *[out for out in outputs]),
                num_outputs=len(node.outputs),
                name=f"{node.name}_phi",
                metadata_props=node.metadata_props.copy(),
            )
            last_node.append(phi_node)
            for i, output in enumerate(node.outputs):
                phi_output = phi_node.outputs[i]
                phi_output.name = output.name
                phi_output.type = output.type
                phi_output.shape = output.shape
                output.replace_all_uses_with(phi_output, replace_graph_outputs=True)
                output.name = f"{output.name}_if_"
            for i, output in enumerate(node.outputs):
                phi_node.replace_input_with(i, output)
        return ir.passes.PassResult(model, modified=modified)


def process_model(model: ir.Model) -> None:
    passes = ir.passes.PassManager(
        [
            common_passes.NameFixPass(),
            AssignNodeNamespacePass(),
            ImplicitUseAnalysisPass(),
            AddCaptureNodePass(),
            EmbedIfPass(),
        ]
    )
    passes(model)


def convert_onnx_file(model_path: str, settings: dict[str, Any]) -> dict[str, Any]:
    parsed_settings = Settings(**settings)
    onnx_model = onnx.load(model_path, load_external_data=False)
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)
    except Exception as e:
        logger.warning("Failed to infer shapes. Continue with original model: %s", e)

    model = ir.serde.deserialize_model(onnx_model)
    process_model(model)
    all_function_ids = set(model.functions)
    graphs = []

    opset_version = model.opset_imports.get("")
    if opset_version is None:
        opset_version = model.opset_imports.get("ai.onnx")
    if opset_version is None:
        opset_version = _DEFAULT_OPSET_VERSION
    if model.graph.name is None:
        model.graph.name = "<main>"

    main_graph = create_graph(
        model.graph,
        all_function_ids=all_function_ids,
        opset_version=opset_version,
        settings=parsed_settings,
        attrs={
            "opset_imports": model.opset_imports,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "domain": model.domain,
            "model_version": model.model_version,
            "doc_string": model.doc_string,
            **{f"[metadata] {k}": v for k, v in model.metadata_props.items()},
        },
    )
    assert main_graph is not None
    graphs.append(main_graph)

    for node in model.graph:
        for attr in node.attributes.values():
            if attr.type == ir.AttributeType.GRAPH:
                attr.value.name = get_subgraph_name(node, attr.name)
                subgraph = create_graph(
                    attr.value,
                    all_function_ids=all_function_ids,
                    opset_version=opset_version,
                    settings=parsed_settings,
                    attrs={
                        "opset_imports": model.opset_imports,
                        "attributes": node.attributes,
                        "doc_string": node.doc_string,
                        **{
                            f"[metadata] {k}": v for k, v in node.metadata_props.items()
                        },
                    },
                )
                if subgraph is not None:
                    graphs.append(subgraph)

    for function in model.functions.values():
        function_graph = create_graph(
            function,
            all_function_ids=all_function_ids,
            opset_version=opset_version,
            settings=parsed_settings,
            attrs={
                "opset_imports": model.opset_imports,
                "attributes": function.attributes,
                "doc_string": function.doc_string,
                **{f"[metadata] {k}": v for k, v in function.metadata_props.items()},
            },
        )
        assert function_graph is not None
        graphs.append(function_graph)

    return {"graphs": graphs}
