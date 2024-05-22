import json
from collections import defaultdict
from typing import Sequence


def rgb(r: int, g: int, b: int) -> str:
    return "#%02x%02x%02x" % (r, g, b)


# https://github.com/lutzroeder/netron/blob/0408dc7fb856f1e97cc23d8d91000b1f5bd487ac/source/grapher.css#L130
NETRON_THEME = {
    "layer": rgb(51, 85, 136),
    "activation": rgb(75, 27, 22),
    "pool": rgb(51, 85, 51),
    "normalization": rgb(51, 85, 68),
    "dropout": rgb(69, 71, 112),
    "shape": rgb(108, 79, 71),
    "tensor": rgb(89, 66, 59),
    "transform": rgb(51, 85, 68),
    "data": rgb(85, 85, 85),
    "quantization": rgb(80, 40, 0),
    "attention": rgb(100, 50, 0),
}

NETRON_CATEGORIES = {
    "Attention": "Attention",
    "AveragePool": "Pool",
    "BatchNormalization": "Normalization",
    "Clip": "Activation",
    "Concat": "Tensor",
    "Conv": "Layer",
    "ConvInteger": "Layer",
    "ConvTranspose": "Layer",
    "Crop": "Data",
    "DecoderAttention": "Attention",
    "DecoderMaskedMultiHeadAttention": "Attention",
    "DecoderMaskedSelfAttention": "Attention",
    "Dropout": "Dropout",
    "Elu": "Activation",
    "Flatten": "Shape",
    "FusedConv": "Layer",
    "GRU": "Layer",
    "Gather": "Transform",
    "Gemm": "Layer",
    "GlobalAveragePool": "Pool",
    "GlobalLpPool": "Pool",
    "GlobalMaxPool": "Pool",
    "GroupQueryAttention": "Attention",
    "HardSigmoid": "Activation",
    "ImageScaler": "Data",
    "InstanceNormalization": "Normalization",
    "LRN": "Normalization",
    "LSTM": "Layer",
    "LayerNormalization": "Normalization",
    "LeakyRelu": "Activation",
    "LogSoftmax": "Activation",
    "LongformerAttention": "Attention",
    "LpNormalization": "Normalization",
    "LpPool": "Pool",
    "MaxPool": "Pool",
    "MaxRoiPool": "Pool",
    "MultiHeadAttention": "Attention",
    "PRelu": "Activation",
    "PackedAttention": "Attention",
    "PackedMultiHeadAttention": "Attention",
    "Pad": "Tensor",
    "QAttention": "Attention",
    "QOrderedAttention": "Attention",
    "QOrderedLongformerAttention": "Attention",
    "RNN": "Layer",
    "Relu": "Activation",
    "Reshape": "Shape",
    "RotaryEmbedding": "Transform",
    "Selu": "Activation",
    "Sigmoid": "Activation",
    "SimplifiedLayerNormalization": "Normalization",
    "SkipSimplifiedLayerNormalization": "Normalization",
    "Slice": "Tensor",
    "Softmax": "Activation",
    "Softplus": "Activation",
    "Softsign": "Activation",
    "SparseAttention": "Attention",
    "Split": "Tensor",
    "Squeeze": "Transform",
    "Tanh": "Activation",
    "ThresholdedRelu": "Activation",
    "Tile": "Shape",
    "Transpose": "Transform",
    "Unsqueeze": "Transform",
    "Upsample": "Data",
}


def color_rule(names: Sequence[str], color: str, text_color: str = "#ffffff"):
    return {
        "queries": [
            {"type": "node_type", "nodeType": "op_nodes"},
            {"type": "regex", "queryRegex": "|".join(names), "matchTypes": ["title"]},
        ],
        "nodeType": "op_nodes",
        "styles": {
            "node_bg_color": {"id": "node_bg_color", "value": color},
            "node_text_color": {"id": "node_text_color", "value": text_color},
        },
        "version": "v2",
    }


def construct_theme():
    category_op_map = defaultdict(list)
    for op, category in NETRON_CATEGORIES.items():
        category_op_map[category].append(op)
    rules = [
        color_rule(ops, NETRON_THEME[category.lower()])
        for category, ops in category_op_map.items()
    ]
    rules.append(
        color_rule(
            ["Initializer", "Constant"],
            "#ffffff",
            text_color="#000000",
        )
    )
    rules.append(
        color_rule(
            ["Input", "Output"],
            rgb(238, 238, 238),
            text_color="#000000",
        )
    )
    rules.append(
        color_rule(
            [".*"],
            "#000000",
            text_color="#ffffff",
        )
    )
    return rules


def main():
    rules = construct_theme()
    with open("netron.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)


if __name__ == "__main__":
    main()
