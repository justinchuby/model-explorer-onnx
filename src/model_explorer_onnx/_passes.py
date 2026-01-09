from __future__ import annotations

import logging
import onnx_ir as ir
import onnx_ir.passes.common as common_passes


logger = logging.getLogger(__name__)


def _parse_namespace(node_name: str) -> list[str]:
    """Parse the namespace from the node name if it is in the format of /namespace/node_name."""
    split = node_name.lstrip("/").rstrip("/").split("/")
    return [ns for ns in split if ns != ""]


def get_node_namespace(node: ir.Node) -> list[str]:
    """Get the namespace from the node."""
    if (metadata_namespace := node.metadata_props.get("namespace")) is not None:
        return _parse_namespace(metadata_namespace)
    if node.name:
        ns = _parse_namespace(node.name)
        if not ns:
            return []
        # Remove the last part of the node name to get the namespace
        return ns[:-1]
    return []


class AssignNodeNamespacePass(ir.passes.InPlacePass):
    """Assign namespace to nodes based on their containing graph."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            if "namespace" in node.metadata_props:
                continue
            if node.name:
                # Remove the last part of the node name to get the namespace
                node.metadata_props["namespace"] = "/".join(get_node_namespace(node))
                modified = True
        return ir.passes.PassResult(model, modified=modified)


class EmbedIfPass(ir.passes.InPlacePass):
    """Convert the model to embed If directly within nodes."""

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False
        for node in model.graph.all_nodes():
            if node.op_type != "If" or node.domain != "":
                continue
            modified = True
            # Handle single graph attributes only
            subgraphs: list[tuple[str, ir.Graph]] = []
            parent_namespace = get_node_namespace(node)
            parent_namespace.append(f"{node.op_type} <{node.name}>")
            node.metadata_props["namespace"] = "/".join(parent_namespace)
            for name, attr in node.attributes.items():
                if attr.type == ir.AttributeType.GRAPH:
                    subgraphs.append((name, attr.as_graph()))

            if not subgraphs:
                logger.warning("IF node %s has no subgraphs", node.name)
                continue

            outputs = []
            last_node = None
            for attr_name, subgraph in subgraphs:
                # Assign namespace to all nodes in the subgraph
                for sub_node in subgraph:
                    sub_namespace = get_node_namespace(sub_node)
                    sub_node.metadata_props["namespace"] = "/".join(
                        (*parent_namespace, attr_name, *sub_namespace)
                    )
                    sub_node.name = f"{node.name}/{attr_name}/{sub_node.name}"
                    for idx, output in enumerate(sub_node.outputs):
                        output.name = f"{node.name}/{attr_name}/{output.name}"
                for idx, g_output in enumerate(subgraph.outputs):
                    g_output.metadata_props["graph_output_index"] = str(idx)
                # Remove the attribute from the node
                node.attributes.pop(attr_name)
                outputs.extend(subgraph.outputs)
                # Clear the subgraph and move the nodes to the parent graph
                subgraph.outputs.clear()
                sub_nodes = tuple(subgraph)
                subgraph.remove(sub_nodes)
                node.append(sub_nodes)
                last_node = sub_nodes[-1]
            assert last_node is not None
            # Create a phi node to merge outputs
            phi_node = ir.node(
                "(Phi)",
                inputs=(*([None] * len(node.outputs)), *[out for out in outputs]),
                num_outputs=len(node.outputs),
                name=f"{node.name}_phi",
                metadata_props=node.metadata_props.copy(),
            )
            last_node.append(phi_node)
            # Copy all information from the If node to the Phi node
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
    """Process the model with the defined passes."""
    passes = ir.passes.PassManager(
        [
            common_passes.NameFixPass(),
            AssignNodeNamespacePass(),
            EmbedIfPass(),
        ]
    )
    passes(model)
