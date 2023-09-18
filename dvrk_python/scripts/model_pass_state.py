# from public shared github gist of @skottmckay

import argparse
import copy
import typing
import onnx
import onnxruntime as ort
import os
import pathlib
import onnx
import itertools

from onnx import shape_inference
from onnxruntime.tools import onnx_model_utils

# get map of value name to ValueInfo for each model value
def get_value_infos(model: onnx.ModelProto):
    # create annotated copy of model with ValueInfo for model values
    m = onnx.shape_inference.infer_shapes(model)

    all_values = itertools.chain(m.graph.input, m.graph.output, m.graph.value_info)
    value_infos = { vi.name : vi for vi in all_values }

    return value_infos


def _node_has_existing_input(node: onnx.NodeProto, input_name, graph_inputs, initializers, expanded_initializers):
    if input_name in graph_inputs:
        print(f"Skipping {node.op_type} node with name '{node.name}' as it already has a graph input for {input_name}.")
        return True

    if input_name not in initializers and input_name not in expanded_initializers.keys():
        # a node output is providing the initial state, and it isn't the output of an Expand node broadcasting an
        # initializer
        print(f"Skipping {node.op_type} node with name '{node.name}' " +
              "as the initial_h input is provided by another node.")
        return True

    return False


def _wire_up_state(graph: onnx.GraphProto,
                   node: onnx.NodeProto,
                   value_info: typing.Dict[str, onnx.ValueInfoProto],
                   state_type: str,  # 'h' for hidden (all RNN nodes), 'c' for cell (LSTM only)
                   rnn_idx: int,
                   elem_type: int,
                   layout: int,
                   hidden_size: int,
                   directions: int,
                   expanded_initializers: typing.Dict[str, str],
                   graph_outputs: typing.Set[str]):

    input_num, output_num = (5, 1) if state_type == 'h' else (6, 2)
    input_name = node.input[input_num] if len(node.input) >= input_num + 1 else ''
    output_name = node.output[output_num] if len(node.output) >= output_num + 1 else ''

    input_name = None
    if not input_name:
        # create name for new state input and add to node. TODO: add checks to ensure new name is unique
        input_name = f'{node.op_type}_{rnn_idx}_initial_{state_type}'

        # add any missing optional inputs so that we're guaranteed to have enough
        while len(node.input) < input_num + 1:
            node.input.append('')

        node.input[input_num] = input_name

    if not output_name:
        # create name for new output and add to node. TODO: add checks to ensure unique
        output_name = f'{node.op_type}_{rnn_idx}_Y_{state_type}'
        while len(node.output) < output_num + 1:
            node.output.append('')

        node.output[output_num] = output_name

    # add output to graph outputs if necessary
    if output_name in graph_outputs:
        output_vi = value_info[output_name]
    else:
        batch_dim_value = None #batch_dim.dim_value if batch_dim.HasField('dim_value') \
        #    else batch_dim.dim_param if batch_dim.HasField('dim_param') \
        #    else None
        dims = (directions, batch_dim_value, hidden_size) if layout == 0 \
            else (batch_dim_value, directions, hidden_size)

        output_vi = onnx.helper.make_tensor_value_info(output_name, elem_type, dims)
        graph.output.append(output_vi)

    # handle special case where an Expand node broadcasts an initializer based on batch size.
    # we need the graph input to match the name of that initializer. when the state is provided as a graph input it
    # most likely has entries for all items in the batch, so the Expand should be a no-op in that case (will cost a
    # copy of the data though as we don't know it's a no-op until runtime).
    input_is_expanded_initializer = input_name in expanded_initializers.keys()
    graph_input_name = expanded_initializers[input_name] if input_is_expanded_initializer else input_name
    input_vi = copy.copy(output_vi)
    input_vi.name = graph_input_name
    graph.input.append(input_vi)

    return graph_input_name, output_name


def _update_model(input_model: pathlib.Path, output_model: pathlib.Path):
    input_path = str(input_model.resolve(strict=True))
    print(f'Processing {input_path}...')
    m = onnx.load(input_path)

    value_info = get_value_infos(m)

    graph_inputs = set([i.name for i in m.graph.input])
    graph_outputs = set([o.name for o in m.graph.output])

    print([value_info[i] for i in graph_inputs])
    print()
    print()
    print([value_info[o] for o in graph_inputs])

    initializers = set([i.name for i in m.graph.initializer])

    # Constant nodes are equivalent to initializers
    # for node in m.graph.node:
    #     if (node.domain == '' or node.domain == 'ai.onnx') and node.op_type == 'Constant':
    #         initializers.add(node.output[0])

    expanded_initializers = {}  # map of output name to initializer
    for node in m.graph.node:
        # Expand can be used to broadcast an initializer with the default value to the batch size. Find Expand nodes
        # that match that usage so that we can allow for that when checking if the initial value has a default provided
        # by an initializer.
        if (node.domain == '' or node.domain == 'ai.onnx') and node.op_type == 'Expand' \
                and node.input[0] in initializers:
            expanded_initializers[node.output[0]] = node.input[0]

    rnn_idx = 0

    for node in m.graph.node:
        if (node.domain == '' or node.domain == 'ai.onnx') and \
                (node.op_type == 'RNN' or node.op_type == 'LSTM' or node.op_type == 'GRU'):
            # for all 3 RNN operators initial_h is input 6 and Y_h is output 2. both are optional.
            # a missing optional input/output has name == ''
            is_lstm = node.op_type == 'LSTM'
            initial_h = node.input[5] if len(node.input) >= 6 else ''
            initial_c = node.input[6] if len(node.input) >= 7 and is_lstm else ''

            # if initial_h:
            #     if _node_has_existing_input(node, initial_h, graph_inputs, initializers, expanded_initializers):
            #         # validate assumption that both initial_h and initial_c have existing inputs. not sure how
            #         # the model would work if you can only provide state for one of those.
            #         if is_lstm:
            #             assert(_node_has_existing_input(node, initial_c, graph_inputs, initializers, expanded_initializers))
            #         continue

            # if initial_c:
            #     if _node_has_existing_input(node, initial_c, graph_inputs, initializers, expanded_initializers):
            #         continue

            # read attribute values if present or use default if not
            layout = next((attr.i for attr in node.attribute if attr.name == 'layout'), 0)
            hidden_size = next(attr.i for attr in node.attribute if attr.name == 'hidden_size')
            directions = next((2 for attr in node.attribute if attr.name == 'direction' and attr.s == b'bidirectional'), 1)

            # get batch dim from the first input
            X = value_info[node.input[0]]

            # not sure this is useful if the sequence length is a fixed value.
            # I guess in theory you could split a large sequence into fixed size chunks...
            # commented out for now to support that usage. uncomment if that seems pointless in your model
            # if seq_dim.HasField('dim_value'):
            #     print(f"Skipping {node.op_type} node with name '{node.name}' as it has a fixed sequence length.")
            #     continue
            elem_type = X.type.tensor_type.elem_type
            initial_h, y_h, = _wire_up_state(m.graph, node, value_info, 'h', rnn_idx, elem_type,
                                             layout, hidden_size, directions,
                                             expanded_initializers, graph_outputs)

            initial_c, y_c = _wire_up_state(m.graph, node, value_info, 'c', rnn_idx, elem_type,
                                            layout, hidden_size, directions,
                                            expanded_initializers, graph_outputs) if is_lstm else (None, None)

            initial_h_optional = initial_h in initializers or initial_h in expanded_initializers.keys()

            msg = f"Updated {node.op_type} node with name '{node.name}'.\n" \
                  f"\tinitial_h value can be passed in using {'(optional)' if initial_h_optional else ''} " \
                  f"graph input named {initial_h}. "\
                  f"The latest state can be retrieved via the graph output named {y_h}. "

            if is_lstm:
                initial_c_optional = initial_c in initializers or initial_c in expanded_initializers.keys()
                msg += f"\n\tinitial_c value can be passed in using {'(optional)' if initial_c_optional else ''} " \
                       f"graph input named {initial_c}. " \
                       f"The latest state can be retrieved via the graph output named {y_c}. "

            print(msg)
            rnn_idx += 1

    if rnn_idx != 0:
        # we updated at least one node
        # run checker in case we broke something
        onnx.checker.check_model(m, full_check=True)

        output_path = str(output_model.resolve())
        print(f'Writing updated model to {output_path}')
        onnx.save(m, output_path)
    else:
        print('Model was not updated.')


def make_rnn_state_graph_input():
    parser = argparse.ArgumentParser(f'{os.path.basename(__file__)}',
                                     description='''
                                     Update an ONNX model to add graph outputs and matching graph inputs for passing
                                     the hidden state from RNN/LSTM/GRU nodes between executions of the model.
                                     
                                     NOTES: Does not handle nodes in subgraphs from Scan/Loop/If nodes currently. 
                                     That would require additional logic to wire any new subgraph output back through 
                                     all ancestor graphs so it is also a main graph output. 
                                     ''')

    parser.add_argument('input_model', type=pathlib.Path, help='Provide path to ONNX model to update.')
    parser.add_argument('output_model', type=pathlib.Path, help='Provide path to write updated ONNX model to.')

    args = parser.parse_args()

    # 'optimize' to temporary model. this just converts Constant nodes to initializers to simplify some things
    tmp_file = args.input_model.resolve(strict=True).with_suffix('.tmp.onnx')
    onnx_model_utils.optimize_model(args.input_model, tmp_file, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    _update_model(tmp_file, args.output_model)
    opt_file = args.output_model.resolve(strict=True).with_suffix('.opt.onnx')
    onnx_model_utils.optimize_model(args.output_model, opt_file, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    os.remove(tmp_file)

if __name__ == '__main__':
    make_rnn_state_graph_input()