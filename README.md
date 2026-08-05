<div align="center">

<a href="https://arxiv.org/abs/2606.10671">
  <img src="assets/logo-variants/fademem-logo-tight.png" alt="FadeMem logo" width="560">
</a>

### Distance-Aware Memory Consolidation for Autoregressive Video Diffusion

[Yu Lu](https://yulu.net.cn/)<sup>1,&#42;</sup> ·
[Junjie Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Junjie)<sup>1,&#42;</sup> ·
[Piotr Koniusz](https://www.koniusz.com/)<sup>2,3</sup> ·
[YuXin Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+YuXin)<sup>4</sup> ·
[Yi Yang](https://reler.net/people/yi_yang/index.html)<sup>1</sup>

<sup>1</sup>Zhejiang University · <sup>2</sup>UNSW · <sup>3</sup>Data61/CSIRO · <sup>4</sup>Baidu Inc  
<sup>&#42;</sup>Equal contribution

[![Paper](https://img.shields.io/badge/arXiv-2606.10671-b31b1b.svg)](https://arxiv.org/abs/2606.10671)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FadeMem--FT-FFD21E.svg)](https://huggingface.co/sanity2025/FadeMem-FT)
[![License](https://img.shields.io/badge/License-Apache--2.0-4c1.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab.svg)](https://www.python.org/)

**A fixed-budget KV memory that stays dense nearby and becomes progressively coarser into the past.**

</div>

FadeMem is a distance-aware KV memory consolidation mechanism that organizes historical KV blocks into a temporal hierarchy under a fixed cache budget. Within a single unified memory, it keeps recent history fine-grained while progressively consolidating older adjacent entries, yielding a dense-near, sparse-far memory within one cache.

## Highlights

- **Dense-near, sparse-far memory.** Recent history retains temporal detail; older history is progressively consolidated.
- **Fixed cache budget.** Memory remains bounded as the generated video grows.
- **Architecture-preserving design.** FadeMem supports inference-time use and lightweight fine-tuning without modifying the backbone architecture.

## Installation

The code is tested on Linux with Python 3.10, PyTorch 2.8.0, CUDA 12.8, and FlashAttention 2.8.3.

```bash
conda create -n fademem python=3.10 -y
conda activate fademem

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

Adjust the PyTorch installation command if your CUDA version differs.

## Model Weights

First, download the [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and [LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B) checkpoints required for inference:

```bash
bash scripts/download_models.sh inference
```

Then download the released [FadeMem-FT](https://huggingface.co/sanity2025/FadeMem-FT) LoRA checkpoint:

```bash
hf download sanity2025/FadeMem-FT \
  --local-dir fademem_models/FadeMem-FT
```

FadeMem-FT is a LoRA checkpoint rather than a standalone text-to-video model. Inference still requires the LongLive base checkpoint. The relevant files should be organized as follows:

```text
FadeMem/
├── fademem_models/FadeMem-FT/model.pt
├── longlive_models/models/longlive_base.pt
└── wan_models/Wan2.1-T2V-1.3B/
```

For training, also download the [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) teacher:

```bash
bash scripts/download_models.sh training
```

This release follows the [LongLive v1.0](https://github.com/NVlabs/LongLive/tree/v1.0) model format.

## Inference

The released inference path targets Wan2.1-T2V-1.3B at 480 x 832 resolution with three-frame latent blocks. Add one text prompt per line to `prompts/example.txt`.

Run inference with the released FadeMem-FT checkpoint:

```bash
bash scripts/infer.sh configs/inference_ft.yaml
```

The configuration keeps the LongLive base checkpoint unchanged and loads FadeMem-FT as the LoRA checkpoint:

```yaml
generator_ckpt: longlive_models/models/longlive_base.pt
lora_ckpt: fademem_models/FadeMem-FT/model.pt
```

Generated videos are saved under `outputs/inference/`. Low-memory mode is selected automatically and can be overridden with `FADEMEM_LOW_MEMORY=0` or `FADEMEM_LOW_MEMORY=1`.

Prompt-level multi-GPU inference is supported through `NUM_GPUS`:

```bash
NUM_GPUS=2 bash scripts/infer.sh configs/inference_ft.yaml
```

Use at least one prompt per GPU and make the prompt count divisible by `NUM_GPUS`.

## Interactive and Multi-Prompt Generation

This release focuses on the standard long-video generation setting and does not provide a maintained interactive or multi-shot inference entry point. FadeMem preserves the LongLive model architecture and causal generation interface, and the underlying pipeline interfaces for sequential prompt switching are retained for further extension.

Users interested in interactive or multi-prompt generation may adapt the [LongLive v1.0 interactive inference workflow](https://github.com/NVlabs/LongLive/tree/v1.0) to FadeMem. Prompt-switch scheduling and cache handling should be validated for the intended setting, as these modes are not part of the officially evaluated release.

## Training

Prepare the paired LongLive prompt files and launch the released eight-GPU recipe:

```bash
cp longlive_models/prompts/vidprom_filtered_extended.txt prompts/
cp longlive_models/prompts/vidprom_filtered_extended_switch.txt prompts/

NUM_GPUS=8 bash scripts/train_long.sh
```

Training instantiates the generator together with real- and fake-score networks and is therefore memory intensive. Checkpoints are written to `outputs/train_long/`.

## Acknowledgements

This repository builds upon [Wan2.1](https://github.com/Wan-Video/Wan2.1), [LongLive](https://github.com/NVlabs/LongLive), and [Self Forcing](https://github.com/guandeh17/Self-Forcing). We thank the authors for releasing their work. See [Third-Party Notices](THIRD_PARTY_NOTICES.md) for source-level attributions.

## Citation

If you find FadeMem useful, please cite:

```bibtex
@article{lu2026fademem,
  title   = {FadeMem: Distance-Aware Memory Consolidation for Autoregressive Video Diffusion},
  author  = {Lu, Yu and Yang, Junjie and Koniusz, Piotr and Song, YuXin and Yang, Yi},
  journal = {arXiv preprint arXiv:2606.10671},
  year    = {2026}
}
```

## License

Released under the [Apache License 2.0](LICENSE). Please also follow the licenses of the upstream models and datasets used with this code.
