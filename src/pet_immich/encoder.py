from __future__ import annotations

from pathlib import Path

import torch


def export_encoder(source: str, destination: Path) -> None:
    import onnx
    from transformers import AutoImageProcessor, AutoModel

    if destination.exists():
        raise FileExistsError(destination)
    processor = AutoImageProcessor.from_pretrained(source, trust_remote_code=False)
    if list(processor.image_mean) != [0.485, 0.456, 0.406] or list(processor.image_std) != [0.229, 0.224, 0.225]:
        raise ValueError("Encoder requires different normalization; adapt extraction before exporting")
    model = AutoModel.from_pretrained(source, trust_remote_code=False, attn_implementation="eager").eval()
    if model.config.model_type != "dinov2":
        raise ValueError("This exporter supports DINOv2 encoders")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    torch.set_num_threads(4)
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 224, 224),),
        destination,
        input_names=["input"],
        output_names=["tokens", "embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}, "tokens": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(destination)
    del graph.graph.output[0]
    onnx.checker.check_model(graph)
    onnx.save(graph, destination)


__all__ = ["export_encoder"]
