from __future__ import annotations

from typing import Any
import model_explorer


class ONNXAdapter(model_explorer.Adapter):
    metadata = model_explorer.AdapterMetadata(
        id="my_adapter",
        name="My first adapter",
        description="My first adapter!",
        source_repo="https://github.com/user/my_adapter",
        fileExts=["test"],
    )

    # Required.
    def __init__(self):
        super().__init__()

    def convert(
        self, model_path: str, settings: dict[str, Any]
    ) -> model_explorer.ModelExplorerGraphs:
        return {"graphs": []}
