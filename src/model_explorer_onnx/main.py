from __future__ import annotations

from typing import Any
import model_explorer
from model_explorer import graph_builder
import onnx
from onnxscript import ir
import logging

logger = logging.getLogger(__name__)

# class PytorchExportedProgramAdapterImpl:

#   def __init__(self, ep: torch.export.ExportedProgram):
#     self.ep = ep
#     self.gm = self.ep.graph_module
#     self.inputs_map = self.get_inputs_map()

#   def legacy_graph_module_flat_inputs(
#       self, ep: torch.export.ExportedProgram, args, kwargs
#   ):
#     """Transform args, kwargs of __call__ to args for graph_module.

#     self.graph_module takes stuff from state dict as inputs.
#     The invariant is for ep: ExportedProgram is
#     ep(args, kwargs) ==
#       ep.postprocess(ep.graph_module(ep.graph_module_flat_inputs(args, kwargs)))
#     """
#     if args is None:
#       args = tuple()
#     if kwargs is None:
#       kwargs = {}

#     flat_args = args
#     if (in_spec := ep.call_spec.in_spec) is not None:
#       if (
#           in_spec.type == tuple
#           and len(in_spec.children_specs) == 2
#           and in_spec.children_specs[0].type == tuple
#           and in_spec.children_specs[1].type == dict
#       ):
#         # NOTE: this is the case where in_spec is for both args and kwargs
#         flat_args = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
#       else:
#         flat_args = fx_pytree.tree_flatten_spec(args, in_spec)

#     param_buffer_keys = (
#         ep.graph_signature.parameters + ep.graph_signature.buffers
#     )
#     param_buffer_values = tuple(ep.state_dict[key] for key in param_buffer_keys)

#     if hasattr(ep.graph_signature, 'lifted_tensor_constants'):
#       ordered_tensor_constants = tuple(
#           ep.tensor_constants[name]
#           for name in ep.graph_signature.lifted_tensor_constants
#       )
#     else:
#       ordered_tensor_constants = tuple()

#     return (*param_buffer_values, *flat_args, *ordered_tensor_constants)

#   def get_inputs_map(self):
#     inputs_map = {}
#     if not self.ep.example_inputs:
#       print(
#           'WARNING: no ExportedProgram.example_inputs found. Cannot show'
#           ' constant tensor values in Model Explorer.'
#       )
#       return inputs_map

#     input_tensors = None
#     if hasattr(self.ep, '_graph_module_flat_inputs'):
#       input_tensors = self.ep._graph_module_flat_inputs(*self.ep.example_inputs)
#     else:
#       # Backward compatibility with torch 2.2.x
#       input_tensors = self.legacy_graph_module_flat_inputs(
#           self.ep, *self.ep.example_inputs
#       )
#     for input_spec, tensor in zip(
#         self.ep.graph_signature.input_specs, input_tensors
#     ):
#       inputs_map[input_spec.arg.name] = [input_spec.target, tensor]
#     return inputs_map

#   def is_arg_node(self, fx_node: torch.fx.node.Node):
#     return fx_node.op == 'placeholder'

#   def is_getitem_node(self, fx_node: torch.fx.node.Node):
#     return isinstance(fx_node.target, types.BuiltinFunctionType)

#   def class_fullname(self, klass):
#     module = klass.__module__
#     if module == 'builtins':
#       return klass.__qualname__
#     return module + '.' + klass.__qualname__

#   def get_label(self, fx_node: torch.fx.node.Node):
#     if hasattr(fx_node.target, 'overloadpacket'):
#       return str(fx_node.target.overloadpacket)
#     if self.is_getitem_node(fx_node):
#       return 'getitem'
#     return str(fx_node.target)

#   def get_hierachy(self, fx_node: torch.fx.node.Node):
#     # Stores all arg and input nodes to `inputs` namespace.
#     if self.is_arg_node(fx_node):
#       return 'inputs'

#     stack_traces = fx_node.meta.get('nn_module_stack', {})
#     layers = []
#     for name, layer in stack_traces.values():
#       iid = '' if not name else '_' + name.split('.')[-1]
#       layer_str = (
#           layer if isinstance(layer, str) else self.class_fullname(layer)
#       )
#       layers.append(layer_str + iid)
#     hierachy_str = '/'.join(layers)
#     return hierachy_str

#   def add_incoming_edges(self, fx_node: torch.fx.node.Node, node: GraphNode):
#     for target_input_id, input_fx_node in enumerate(fx_node.all_input_nodes):
#       source_node_output_id = '0'  # default to the first output
#       for idx, user in enumerate(input_fx_node.users):
#         if user == fx_node:
#           source_node_output_id = str(idx)
#           break
#       node.incomingEdges.append(
#           IncomingEdge(
#               sourceNodeId=input_fx_node.name,
#               sourceNodeOutputId=source_node_output_id,
#               targetNodeInputId=str(target_input_id),
#           )
#       )

#   def print_tensor(self, tensor: torch.Tensor, size_limit: int = 16):
#     shape = tensor.shape
#     total_size = 1
#     for dim in shape:
#       total_size *= dim
#     if size_limit < 0 or size_limit > total_size:
#       return json.dumps(tensor.detach().numpy().tolist())

#     return json.dumps((tensor.detach().numpy().flatten())[:size_limit].tolist())

#   def add_node_attrs(self, fx_node: torch.fx.node.Node, node: GraphNode):
#     if hasattr(fx_node.target, '_schema'):
#       for idx, arg in enumerate(fx_node.target._schema.arguments):
#         if idx < len(fx_node.args):
#           node.attrs.append(
#               KeyValue(key=arg.name, value=str(fx_node.args[idx]))
#           )
#         else:
#           val = fx_node.kwargs.get(arg.name, arg.default_value)
#           node.attrs.append({'key': arg.name, 'value': str(val)})

#     if self.is_arg_node(fx_node):
#       tensor_spec = self.inputs_map.get(fx_node.name)
#       if tensor_spec:
#         node.attrs.append(KeyValue(key='target', value=str(tensor_spec[0])))
#         node.attrs.append(
#             KeyValue(key='__value', value=self.print_tensor(tensor_spec[1]))
#         )

#   def add_outputs_metadata(self, fx_node: torch.fx.node.Node, node: GraphNode):
#     out_vals = fx_node.meta.get('val')
#     if out_vals is None:
#       return

#     if isinstance(out_vals, (tuple, list)):
#       for idx, val in enumerate(out_vals):
#         metadata = MetadataItem(id=str(idx), attrs=[])
#         if val is None:
#           continue
#         dtype = str(val.dtype)
#         shape = json.dumps(val.shape)
#         metadata.attrs.append(KeyValue(key='tensor_shape', value=dtype + shape))
#         node.outputsMetadata.append(metadata)
#     else:
#       dtype = str(out_vals.dtype)
#       shape = json.dumps(out_vals.shape)
#       metadata = MetadataItem(
#           id='0', attrs=[KeyValue(key='tensor_shape', value=dtype + shape)]
#       )
#       node.outputsMetadata.append(metadata)

#   def create_node(self, fx_node: torch.fx.node.Node):
#     node = GraphNode(
#         id=fx_node.name,
#         label=self.get_label(fx_node),
#         namespace=self.get_hierachy(fx_node),
#     )
#     self.add_incoming_edges(fx_node, node)
#     self.add_node_attrs(fx_node, node)
#     self.add_outputs_metadata(fx_node, node)
#     return node

#   def create_graph(self):
#     graph = Graph(id='graph', nodes=[])
#     for node in self.gm.graph.nodes:
#       graph.nodes.append(self.create_node(node))
#     return graph

#   def convert(self) -> ModelExplorerGraphs:
#     return {'graphs': [self.create_graph()]}

def add_outputs_metadata(onnx_node: ir.Node, node: graph_builder.GraphNode):
    for output in onnx_node.outputs:
        metadata = graph_builder.MetadataItem(id=str(output.index()), attrs=[])
        dtype = str(output.dtype)
        type_str = str(output.type)
        shape = str(output.shape)
        metadata.attrs.append(graph_builder.KeyValue(key='tensor_shape', value=dtype + shape))
        metadata.attrs.append(graph_builder.KeyValue(key='type', value=type_str))
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

def add_incoming_edges(onnx_node: ir.Node, node: graph_builder.GraphNode):
    for target_input_id, input_value in enumerate(onnx_node.inputs):
        if input_value is None:
            continue
        input_node = input_value.producer()
        if input_node is None:
            logger.debug(
                "Node %s does not have a producer. Skipping incoming edge.",
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

def create_node(onnx_node: ir.Node) -> graph_builder.GraphNode:
    if onnx_node.name is None:
        logger.warning("Node does not have a name. Skipping node %s.", onnx_node)
    node = graph_builder.GraphNode(
        id=onnx_node.name,
        label=create_op_label(onnx_node.domain, onnx_node.op_type),
        # namespace=None,
    )
    add_incoming_edges(onnx_node, node)
    add_node_attrs(onnx_node, node)
    add_outputs_metadata(onnx_node, node)
    return node

def create_graph(onnx_graph: ir.Graph) -> graph_builder.Graph:
    graph = graph_builder.Graph(id="graph", nodes=[])
    for node in onnx_graph:
        graph.nodes.append(create_node(node))
    return graph




class ONNXAdapter(model_explorer.Adapter):
    metadata = model_explorer.AdapterMetadata(
        id="my_adapter",
        name="ONNX adapter",
        description="My first adapter!",
        source_repo="https://github.com/user/my_adapter",
        fileExts=["onnx", "onnxtext", "onnxtxt"],
    )

    # Required.
    def __init__(self):
        super().__init__()

    def convert(
        self, model_path: str, settings: dict[str, Any]
    ) -> model_explorer.ModelExplorerGraphs:
        return {"graphs": []}
