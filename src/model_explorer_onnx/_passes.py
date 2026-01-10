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


class ImplicitUseAnalysisPass(ir.passes.InPlacePass):
    """Find all closed variables for each node with subgraphs."""

    def _iterate_subgraphs(self, node: ir.Node, implicit_usages, graph_stack):
        def process_node(node: ir.Node, subgraph: ir.Graph):
            for inp in node.inputs:
                if inp is not None and inp.graph is not subgraph:
                    # This is a closed variable, add to implicit usages of all parent graphs
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
                for node in subgraph:
                    process_node(node, subgraph)
                    self._iterate_subgraphs(node, implicit_usages, graph_stack)
                graph_stack.pop()
            elif attr.type == ir.AttributeType.GRAPHS:
                for subgraph in attr.as_graphs():
                    graph_stack.append(subgraph)
                    if subgraph not in implicit_usages:
                        implicit_usages[subgraph] = []
                    for node in subgraph:
                        process_node(node, subgraph)
                        self._iterate_subgraphs(node, implicit_usages, graph_stack)
                    graph_stack.pop()

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph_stack: list[ir.Graph] = []
        implicit_usages: dict[ir.Graph, list[ir.Value]] = {}
        for node in model.graph:
            self._iterate_subgraphs(node, implicit_usages, graph_stack)

        for graph, used_values in implicit_usages.items():
            # Remove duplicates while preserving order
            seen = set()
            closed_values = []
            for val in used_values:
                if val not in seen:
                    seen.add(val)
                    closed_values.append(val)
            graph.meta["implicit_uses"] = closed_values
        return ir.passes.PassResult(model, modified=False)


class AddCaptureNodePass(ir.passes.InPlacePass):
    """Add a (Capture) node to nodes with subgraphs to visualize closed variables.

    For nodes with subgraphs (e.g., If, Loop, Scan), find all values that are captured
    from the outer graph (closed variables) and create a special (Capture) node that
    takes these values as inputs. The Capture node is added as an input to the parent node.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:
        modified = False

        for node in model.graph.all_nodes():
            if not any(
                attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
                for attr in node.attributes.values()
            ):
                continue
            if node.op_type == "If" and node.domain == "":
                # Skip If nodes as they are handled in EmbedIfPass
                continue

            # Get the closed variables for this node's graph(s)
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

            # Deduplicate closed values while preserving order, to avoid redundant
            # inputs when the same value is used in multiple subgraphs.
            unique_closed_values: list[ir.Value] = []
            seen_values: set[ir.Value] = set()
            for value in all_closed_values:
                if value in seen_values:
                    continue
                seen_values.add(value)
                unique_closed_values.append(value)

            if not unique_closed_values:
                continue

            # Create a Capture node
            capture_node = ir.node(
                "(Capture)",
                inputs=unique_closed_values,
                name=f"{node.name}_capture",
            )

            capture_node.outputs[0].name = f"{node.name}_captured_output"

            # Find the graph containing the node and insert the capture node before it
            graph = node.graph
            if graph is not None:
                graph.insert_before(node, capture_node)

                # Add the capture node's output as an input to the current node
                # Use resize_inputs to add a new input slot
                original_input_count = len(node.inputs)
                node.resize_inputs(original_input_count + 1)
                node.replace_input_with(original_input_count, capture_node.outputs[0])

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
            ImplicitUseAnalysisPass(),
            AddCaptureNodePass(),
            EmbedIfPass(),
        ]
    )
    passes(model)
