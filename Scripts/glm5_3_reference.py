#!/usr/bin/env python3
"""FP32 reference activations for GLM-5.3-Flash prefix layers.

Runs the upstream transformers `Glm5NextTextModel` (main branch, glm5_next) on the
first N decoder layers of the original FP8 checkpoint, dequantized to FP32. Writes
`tokens.json`, `layer_<i>.f32` residual streams ([T, 4, hidden]) after each layer,
and `logits.f32` ([T, vocab]) computed from the last converted layer.

With `--image`, the prompt becomes `<|begin_of_image|>`, one `<|image|>` per merged
patch and `<|end_of_image|>` followed by the prompt text. The official image processor
and FP32 vision tower produce `pixel_values.f32`, `vision.f32` and `grid` in tokens.json,
and the vision output replaces the placeholder embeddings as in Glm5NextModel.

Requires torch, safetensors and transformers with glm5_next (plus torchvision and Pillow
for --image).
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel, Glm5NextVisionModel


class Checkpoint:
    def __init__(self, directory: Path):
        self.directory = directory
        self.weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}

    def tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        if filename not in self.files:
            self.files[filename] = safe_open(self.directory / filename, framework="pt")
        value = self.files[filename].get_tensor(name)
        if value.dtype == torch.float8_e4m3fn:
            # 128x128 blocks of FP32 weight_scale_inv.
            scale = self.tensor(name + "_scale_inv")
            rows, columns = value.shape
            scale = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :columns]
            return value.float() * scale
        return value.float()


def state_dict(checkpoint: Checkpoint, model: Glm5NextTextModel) -> dict:
    prefix = "model.language_model."
    result = {}
    for name, parameter in model.state_dict().items():
        source = prefix + name
        if ".attn_hc." in name or ".ffn_hc." in name:
            layer, kind, field = name.split(".")[1], name.split(".")[2][:-3], name.split(".")[3]
            source = f"{prefix}layers.{layer}.hc_{kind}_{field}"
            value = checkpoint.tensor(source)
        elif ".self_attn.conv1d.weight" in name:
            stem = prefix + name[: -len("conv1d.weight")]
            value = torch.cat([checkpoint.tensor(f"{stem}{kind}_conv1d.weight") for kind in "qkv"])
        elif ".forget_gate." in name:
            value = checkpoint.tensor(source.replace(".forget_gate.", "."))
        elif name.endswith(".mlp.experts.gate_up_proj"):
            stem = prefix + name[: -len("gate_up_proj")]
            value = torch.stack([
                torch.cat([checkpoint.tensor(f"{stem}{i}.gate_proj.weight"),
                           checkpoint.tensor(f"{stem}{i}.up_proj.weight")])
                for i in range(parameter.shape[0])
            ])
        elif name.endswith(".mlp.experts.down_proj"):
            stem = prefix + name[: -len("down_proj")]
            value = torch.stack([checkpoint.tensor(f"{stem}{i}.down_proj.weight") for i in range(parameter.shape[0])])
        else:
            value = checkpoint.tensor(source)
        if tuple(value.shape) != tuple(parameter.shape):
            raise ValueError(f"{name}: {tuple(value.shape)} != {tuple(parameter.shape)}")
        result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    text = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        text[key] = text[key][: args.layers]
    text["num_hidden_layers"] = args.layers
    config = Glm5NextTextConfig(**text)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Glm5NextTextModel(config)
    checkpoint = Checkpoint(args.checkpoint)
    model.load_state_dict(state_dict(checkpoint, model), strict=True, assign=True)
    model = model.float().eval()
    head = checkpoint.tensor("lm_head.weight")

    full_config = json.loads((args.checkpoint / "config.json").read_text())
    tokens = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json")).encode(args.prompt).ids
    meta = {"prompt": args.prompt}
    image_embeds = None
    if args.image is not None:
        from PIL import Image
        from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor

        image = Image.open(args.image).convert("RGB")
        processed = Glm5NextImageProcessor.from_pretrained(args.checkpoint)(
            images=[image], return_tensors="pt")
        # Raw RGB in [0, 1], [height, width, 3], to check a port of the preprocessing.
        (torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.height, image.width, 3) / 255
         ).numpy().tofile(args.output / "image_rgb.f32")
        meta["image_size"] = [image.height, image.width]
        pixel_values, grid = processed["pixel_values"].float(), processed["image_grid_thw"]
        vision_config = Glm5NextVisionConfig(**full_config["vision_config"])
        vision_config._attn_implementation = "eager"
        # Built on CPU: the rotary inverse frequencies are a non-persistent buffer.
        visual = Glm5NextVisionModel(vision_config)
        visual.load_state_dict(
            {name: checkpoint.tensor("model.visual." + name) for name in visual.state_dict()},
            strict=True)
        visual = visual.float().eval()
        image_embeds = visual(pixel_values, grid_thw=grid).pooler_output
        pixel_values.numpy().tofile(args.output / "pixel_values.f32")
        image_embeds.numpy().tofile(args.output / "vision.f32")
        count = image_embeds.shape[0]
        tokens = ([full_config["image_start_token_id"]] + [full_config["image_token_id"]] * count
                  + [full_config["image_end_token_id"]] + tokens)
        meta["grid"] = grid[0].tolist()
        meta["image_tokens"] = [1, 1 + count]
        print(f"image grid {grid[0].tolist()}: {count} tokens, vision rms={image_embeds.square().mean().sqrt():.4f}")
    meta["tokens"] = tokens
    (args.output / "tokens.json").write_text(json.dumps(meta))
    outputs = {}
    for index, layer in enumerate(model.layers):
        layer.register_forward_hook(
            lambda module, inputs, output, index=index: outputs.__setitem__(index, output[0][0].contiguous()))
    embeds = model.embed_tokens(torch.tensor([tokens]))
    if image_embeds is not None:
        mask = (torch.tensor([tokens]) == full_config["image_token_id"]).unsqueeze(-1)
        embeds = embeds.masked_scatter(mask, image_embeds)
    hidden = model(inputs_embeds=embeds, use_cache=False).last_hidden_state[0]
    for index, value in sorted(outputs.items()):
        value.numpy().tofile(args.output / f"layer_{index}.f32")
        print(f"layer {index}: {tuple(value.shape)} rms={value.square().mean().sqrt():.4f}")
    logits = hidden @ head.T
    logits.numpy().tofile(args.output / "logits.f32")
    print("tokens", tokens)
    print("top-1", logits.argmax(-1).tolist())


if __name__ == "__main__":
    main()
