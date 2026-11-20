#!/usr/bin/env python3
"""FP32 reference activations for Inkling 2 Small prefix layers.

Runs the upstream transformers `Inkling2TextModel` (inkling2 on transformers main) on the first N
decoder layers of the original NVFP4 checkpoint, dequantized to FP32. Writes `tokens.json`,
`layer_<i>.f32` partial blocks ([T, hidden]) after each layer, and `logits.f32` ([T, unpadded
vocab]) from the final block mixture of the converted layers, softcapped as the model does.

Routed experts are dequantized on demand, one expert at a time, from the checkpoint's NVFP4
(E2M1 low nibble first, an E4M3 scale per 16 values, amax / outer_scale_denominator per expert) or
MXFP8 (E4M3, an E8M0 scale per 32 values) banks; w13 rows interleave gate and up.

`--position-offset` shifts the position ids, which only the full-attention log scaling reads.
`--emulate-q8` rounds each expert matrix as s4nnc's rowwise Q8_0 (i8x without an imatrix) does, so
a store converted with `--expert-codec i8x` can be compared without its quantization error.
`--emulate-a8` also rounds each expert input row to int8 (absmax / 127), as the segmented int8 GEMM
does in prefill chunks of more than 48 tokens (smaller chunks and decode keep float activations).

With `--image`, the prompt holds one `<|unused_200054|>`, which becomes one placeholder per image
token as Inkling2Processor does. The official image processor and FP32 vision tower write
`image_rgb.f32`, `pixel_values.f32` (the patches the patch embedding reads, `[4 * tokens, 1200]`),
`vision.f32` and the grid in tokens.json, and the vision output replaces the placeholder embeddings as in Inkling2Model. `--layers 0` writes only these, so
the checkpoint needs just the vision shard, `config.json`, `processor_config.json` and the
tokenizer.

Requires torch, safetensors and transformers with inkling2 (plus torchvision and Pillow for
--image).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from torch import nn
from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
from transformers.models.inkling2.modeling_inkling2 import Inkling2TextModel

FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


class Checkpoint:
    def __init__(self, directory: Path):
        self.directory = directory
        self.weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}

    def file(self, name: str):
        filename = self.weight_map[name]
        if filename not in self.files:
            self.files[filename] = safe_open(self.directory / filename, framework="pt")
        return self.files[filename]

    def tensor(self, name: str) -> torch.Tensor:
        return self.file(name).get_tensor(name).float()

    def expert(self, name: str, expert: int) -> torch.Tensor:
        """One expert's FP32 matrix from a routed bank."""
        weight = self.file(name).get_slice(name)[expert]
        scale = self.file(name + "_scale").get_slice(name + "_scale")[expert]
        rows = weight.shape[0]
        if weight.dtype == torch.uint8:
            values = torch.stack([FP4[(weight & 15).long()], FP4[(weight >> 4).long()]], -1).reshape(rows, -1)
            blocks = values.reshape(rows, -1, 16) * scale.float()[..., None]
            amax = self.file(name + "_amax").get_slice(name + "_amax")[expert]
            denominator = self.tensor(name + "_outer_scale_denominator")[0]
            return blocks.reshape(rows, -1) * (amax / denominator)
        scale = torch.pow(2.0, scale.int().float() - 127)
        return (weight.float().reshape(rows, -1, 32) * scale[..., None]).reshape(rows, -1)


def quantize_a8(rows: torch.Tensor) -> torch.Tensor:
    """int8_matmul_quantize_activation: per row, rint(x * 127 / absmax) clamped to [-127, 127]."""
    max_abs = rows.abs().amax(1, keepdim=True)
    inverse = torch.where(max_abs > 0, 127 / max_abs, torch.full_like(max_abs, 127.0))
    q = torch.round(rows * inverse).clamp(-127, 127)
    return q * torch.where(max_abs > 0, max_abs / 127, torch.full_like(max_abs, 1 / 127))


def quantize_q8(matrix: torch.Tensor) -> torch.Tensor:
    """ccv_nnc_quantize_8i_rowwise in FP32: per row, int8 values in [-127, 127] (lrint) with an FP32
    scale refit by least squares from absmax / 127, keeping the scale of least squared error."""
    v = matrix.double()
    scale = v.abs().amax(1) / 127
    best_scale = torch.zeros_like(scale)
    best_sse = torch.full_like(scale, float("inf"))
    done = ~(scale > 0)
    for _ in range(8):
        stored = scale.float().double()
        done |= ~(stored > 0)
        q = torch.round(v / stored[:, None]).clamp(-127, 127)
        d = v - stored[:, None] * q
        sse = (d * d).sum(1)
        better = ~done & (sse < best_sse)
        best_sse = torch.where(better, sse, best_sse)
        best_scale = torch.where(better, stored, best_scale)
        sum_qx = (q * v).sum(1)
        sum_qq = (q * q).sum(1)
        following = sum_qx / sum_qq
        done |= ~(sum_qq > 0) | ~(sum_qx > 0) | (following.float() == stored.float())
        scale = torch.where(done, scale, following)
    best = best_scale.float()
    q = torch.round(v / best.double()[:, None]).clamp(-127, 127)
    return torch.where(best[:, None] > 0, q.float() * best[:, None], torch.zeros_like(matrix))


class LazyExperts(nn.Module):
    """Inkling2Experts with each routed expert dequantized when a token selects it."""

    def __init__(
        self, checkpoint: Checkpoint, layer: int, config: Inkling2TextConfig, emulate_q8: bool, emulate_a8: bool
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.emulate_q8 = emulate_q8
        self.emulate_a8 = emulate_a8
        self.prefix = f"model.llm.layers.{layer}.mlp.experts."
        self.num_experts = config.n_routed_experts
        self.gate_softcap = config.mlp_gate_softcap
        self.up_softcap = config.mlp_up_softcap

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert in top_k_index.unique().tolist():
            token, slot = torch.where(top_k_index == expert)
            w13 = self.checkpoint.expert(self.prefix + "w13_weight", expert)
            gate, up = w13[0::2], w13[1::2]
            down = self.checkpoint.expert(self.prefix + "w2_weight", expert)
            if self.emulate_q8:
                gate, up, down = quantize_q8(gate), quantize_q8(up), quantize_q8(down)
            x = hidden_states[token]
            if self.emulate_a8:
                x = quantize_a8(x)
            g, u = x @ gate.T, x @ up.T
            h = torch.tanh(g / self.gate_softcap) * self.gate_softcap * torch.sigmoid(g) * (
                torch.tanh(u / self.up_softcap) * self.up_softcap)
            # Routing weights scale rows before the down projection, as the s4nnc model does.
            h = h * top_k_weights[token, slot, None]
            if self.emulate_a8:
                h = quantize_a8(h)
            final.index_add_(0, token, h @ down.T)
        return final.to(hidden_states.dtype)


def source_name(name: str) -> str:
    llm = "model.llm."
    if name == "embed_tokens.weight":
        return llm + "embed.weight"
    if name == "embed_tokens.embed_norm.weight":
        return llm + "embed_norm.weight"
    if name == "norm.weight":
        return llm + "norm.weight"
    if name == "final_input.query":
        return llm + "attn_res.output_query"
    parts = name.split(".")
    layer = int(parts[1])
    rest = ".".join(parts[2:])
    if rest == "attention_input.query":
        return f"{llm}attn_res.attn_queries.{layer - 1}"
    if rest == "mlp_input.query":
        return f"{llm}attn_res.mlp_queries.{layer}"
    renames = {
        "self_attn.q_proj.weight": "attn.wq_du.weight",
        "self_attn.k_proj.weight": "attn.wk_dv.weight",
        "self_attn.v_proj.weight": "attn.wv_dv.weight",
        "self_attn.r_proj.weight": "attn.wr_du.weight",
        "self_attn.o_proj.weight": "attn.wo_ud.weight",
        "self_attn.gate_proj.weight": "attn.gate_attn.weight",
        "self_attn.sink_proj.weight": "attn.attn_sink_proj.weight",
        "self_attn.q_l2_clip.scale": "attn.q_l2_clip.scale",
        "self_attn.rel_l2_clip.scale": "attn.rel_l2_clip.scale",
        "self_attn.rel_logits_proj.proj": "attn.rel_logits_proj.proj",
        "self_attn.k_sconv.conv1d.weight": "attn.k_sconv.weight",
        "self_attn.v_sconv.conv1d.weight": "attn.v_sconv.weight",
        "self_attn.k_sconv.static_weight": "attn.k_sconv.static_w",
        "self_attn.v_sconv.static_weight": "attn.v_sconv.static_w",
        "self_attn.k_sconv.weight_proj.weight": "attn.k_sconv.weight_proj",
        "self_attn.v_sconv.weight_proj.weight": "attn.v_sconv.weight_proj",
        "input_layernorm.weight": "attn_norm.weight",
        "q_input_layernorm.weight": "q_input_norm.weight",
        "k_input_layernorm.weight": "k_input_norm.weight",
        "v_input_layernorm.weight": "v_input_norm.weight",
        "post_attention_layernorm.weight": "mlp_norm.weight",
        "post_attention_convolution.conv1d.weight": "attn_sconv.weight",
        "post_mlp_convolution.conv1d.weight": "mlp_sconv.weight",
        "mlp.gate.weight": "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias": "mlp.gate.bias",
    }
    return f"{llm}layers.{layer}.{renames[rest]}"


def vision_source_name(name: str) -> str:
    """Checkpoint name of an Inkling2VisionModel parameter."""
    if name.startswith("merger."):
        return name.replace("merger.linear_1.", "model.visual.proj_in.").replace(
            "merger.linear_2.", "model.visual.proj_out.")
    return "model.visual.backbone." + (
        name.replace("position_embedding.weight", "position_embedding")
        .replace("pre_layernorm.", "ln_pre.").replace("layernorm_before.", "ln_1.")
        .replace("layernorm_after.", "ln_2.").replace("mlp.fc1.", "mlp.c_fc.")
        .replace("mlp.fc2.", "mlp.c_proj.").replace("lambda_1", "ls_1.gamma")
        .replace("lambda_2", "ls_2.gamma"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument("--emulate-q8", action="store_true")
    parser.add_argument("--emulate-a8", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layers == 0 and args.image is None:
        parser.error("--layers 0 compares only the vision tower and needs --image")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    checkpoint = Checkpoint(args.checkpoint)
    full_config = json.loads((args.checkpoint / "config.json").read_text())
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
    tokenizer = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    tokens = tokenizer.encode(prompt).ids
    if args.max_tokens:
        tokens = tokens[: args.max_tokens]
    meta = {"prompt": prompt, "tokens": tokens, "position_offset": args.position_offset}
    image_embeds = None
    if args.image is not None:
        import numpy as np
        from PIL import Image
        from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
        from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
        from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

        processor_config = json.loads((args.checkpoint / "processor_config.json").read_text())
        settings = dict(processor_config["image_processor"])
        settings.pop("image_processor_type")
        image = Image.open(args.image)
        processed = Inkling2ImageProcessor(**settings)(images=[image], return_tensors="pt")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        rgb.tofile(args.output / "image_rgb.f32")
        meta["image_size"] = [image.height, image.width]
        pixel_values, grid = processed["pixel_values"].float(), processed["image_grid_thw"]
        vision_config = Inkling2VisionConfig(**full_config["vision_config"])
        # Written as the patch embedding reads them: block-major patches, (row, column, channel).
        merge, patch = vision_config.spatial_merge_size, vision_config.patch_size
        pixel_values.reshape(-1, merge, patch, merge, patch, 3).transpose(2, 3).reshape(
            -1, 3 * patch * patch).numpy().tofile(args.output / "pixel_values.f32")
        vision_config._attn_implementation = "eager"
        visual = Inkling2VisionModel(vision_config)
        visual.load_state_dict(
            {name: checkpoint.tensor(vision_source_name(name)) for name in visual.state_dict()}, strict=True)
        visual = visual.float().eval()
        image_embeds = visual(pixel_values, image_grid_thw=grid).pooler_output
        image_embeds.numpy().tofile(args.output / "vision.f32")
        image_token = tokenizer.token_to_id(processor_config["image_token"])
        if tokens.count(image_token) != 1:
            parser.error(f"the prompt needs one {processor_config['image_token']} where the image goes")
        start, count = tokens.index(image_token), image_embeds.shape[0]
        tokens = tokens[:start] + [image_token] * count + tokens[start + 1:]
        meta.update(tokens=tokens, grid=grid[0, 1:].tolist(), image_tokens=[start, start + count])
        print(f"image grid {grid[0].tolist()}: {count} tokens, vision rms={image_embeds.square().mean().sqrt():.4f}")
    (args.output / "tokens.json").write_text(json.dumps(meta))
    if args.layers == 0:
        return

    text = full_config["text_config"]
    renames = {"intermediate_size": "moe_intermediate_size", "sconv_kernel_size": "conv_kernel_size"}
    text_fields = {renames.get(key, key): value for key, value in text.items()}
    config = Inkling2TextConfig(
        **{key: value for key, value in text_fields.items() if key in Inkling2TextConfig.__dataclass_fields__})
    config.num_hidden_layers = args.layers
    config.layer_types = config.layer_types[: args.layers]
    expected = ["hybrid" if i not in text["local_layer_ids"] else "hybrid_sliding" for i in range(args.layers)]
    assert config.layer_types == expected, (config.layer_types, expected)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Inkling2TextModel(config)
    for index, layer in enumerate(model.layers):
        layer.mlp.experts = LazyExperts(checkpoint, index, config, args.emulate_q8, args.emulate_a8)
    state = {}
    for name, parameter in model.state_dict().items():
        value = checkpoint.tensor(source_name(name))
        if value.shape != parameter.shape:
            value = value.reshape(parameter.shape)
        state[name] = value
    model.load_state_dict(state, strict=True, assign=True)
    model = model.float().eval()
    head = checkpoint.tensor("model.llm.unembed.weight")[: config.unpadded_vocab_size]

    outputs = {}
    for index, layer in enumerate(model.layers):
        layer.register_forward_hook(
            lambda module, inputs, output, index=index: outputs.__setitem__(index, output[0].contiguous()))
    positions = torch.arange(len(tokens))[None] + args.position_offset
    embeds = model.embed_tokens(torch.tensor([tokens]))
    if image_embeds is not None:
        embeds[0, start:start + count] = image_embeds
    hidden = model(inputs_embeds=embeds, position_ids=positions, use_cache=False).last_hidden_state[0]
    for index, value in sorted(outputs.items()):
        value.numpy().tofile(args.output / f"layer_{index}.f32")
        print(f"layer {index}: {tuple(value.shape)} rms={value.square().mean().sqrt():.4f}")
    logits = torch.tanh(hidden @ head.T / config.final_logit_softcapping) * config.final_logit_softcapping
    logits.numpy().tofile(args.output / "logits.f32")
    print("tokens", len(tokens), tokens[:32])
    print("top-1", logits.argmax(-1).tolist()[:64])


if __name__ == "__main__":
    main()
