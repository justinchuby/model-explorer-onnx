# Model Explorer ONNX Adapter

[![PyPI - Version](https://img.shields.io/pypi/v/model-explorer-onnx.svg)](https://pypi.org/project/model-explorer-onnx) [![PyPI - Downloads](https://img.shields.io/pypi/dm/model-explorer-onnx)](https://pypi.org/project/model-explorer-onnx) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

ONNX Adapter for [google-ai-edge/model-explorer](https://github.com/google-ai-edge/model-explorer)

## Installation

```bash
pip install --upgrade model-explorer-onnx
```

## Usage

```bash
model-explorer --extensions=model_explorer_onnx

# Or as a shortcut
onnxvis

# Supply model path
onnxvis model.onnx
```

> [!NOTE]
> Model Explorer only supports WSL on Windows.

Read more on the [Model Explorer User Guide](https://github.com/google-ai-edge/model-explorer/wiki/2.-User-Guide).

## Notes on representation

Graph input/output/initializers in ONNX are values (edges), not nodes. A node is displayed here for visualization. Graph inputs that are initialized by initializers are displayed as `InitializedInput`, and are displayed closer to nodes that use them.

## Color Themes

Get node color themes [here](./themes)

## Screenshots

<img width="1294" alt="image" src="https://github.com/justinchuby/model-explorer-onnx/assets/11205048/ed7e1eee-a693-48bd-811d-b384f784ef9b">

<img width="1291" alt="image" src="https://github.com/justinchuby/model-explorer-onnx/assets/11205048/b266d8e9-9760-4860-a0a7-eda1de31e1a1">

<img width="1285" alt="image" src="https://github.com/justinchuby/model-explorer-onnx/assets/11205048/b772aa13-4cc3-4034-a729-f134fa3cf818">

<img width="1292" alt="image" src="https://github.com/justinchuby/model-explorer-onnx/assets/11205048/dabbad76-0ec7-43b1-b253-13b81e7dc658">

<img width="1293" alt="image" src="https://github.com/justinchuby/model-explorer-onnx/assets/11205048/fbf2fa05-bd29-4938-93d1-709690d9f9c6">

<img width="1301" alt="image" src="https://github.com/justinchuby/model-explorer-onnx/assets/11205048/a68f7ecd-1fa1-4eac-9e1f-8e9a5bbf9fe3">
