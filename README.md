<div align="center">

<img src="https://raw.githubusercontent.com/Studio1HQ/Studio1HQ/main/assets/banners/gpt-oss-finetuning.webp" alt="Fine-Tune GPT-OSS-20B with Unsloth: Fine-tune GPT-OSS-20B on a free Colab T4, with data from Bright Data" width="100%" />

<br />
<br />

# Fine-Tune GPT-OSS-20B with Unsloth

### Fine-tune GPT-OSS-20B on a free Colab T4, with data from Bright Data

A step-by-step pipeline and Colab notebook that scrapes training text with Bright Data, formats it, and fine-tunes unsloth/gpt-oss-20b with LoRA using Unsloth.

<br />

[![Built by Studio1](https://img.shields.io/badge/Built_by_Studio1-f97316?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTYwIiBmaWxsPSJub25lIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTAgMzVDMCAxNi42ODA3IDE0LjA3NDQgMS42NDg0NCAzMiAwLjEyNjk1M1YxNDFINTEuMDUyM0M1Mi40MzA4IDEzNy4zNDQgNTQuNzQ0NiAxMzMuNjg5IDU3Ljk5MzggMTMwLjAzM0M2MS4xNDQ2IDEyNi41MDggNjQuNjg5MiAxMjMuNDQgNjguNjI3NyAxMjAuODI5QzcyLjU2NjIgMTE4LjM0OSA3Ni4zNTY5IDExNi43ODIgODAgMTE2LjEyOVY4Ni43NTQ0Qzc1LjQ3MDggODcuOTI5MiA3MS4zODQ2IDg5LjY5MTkgNjcuNzQxNSA5Mi4wNDE1QzY0IDk0LjUyMjUgNjAuNzUwOCA5Ny40NiA1Ny45OTM4IDEwMC44NTRDNTYuNTM0NSAxMDIuNzIgNTUuMjU0NSAxMDQuNjU5IDU0LjE1MzkgMTA2LjY3MVYwSDEyNUMxNDQuMzMgMCAxNjAgMTUuNjY5OSAxNjAgMzVWMTI1QzE2MCAxNDQuMzMgMTQ0LjMzIDE2MCAxMjUgMTYwSDEyNFYxOEgxMDQuOTQ4QzEwMy41NjkgMjEuNjgxNiAxMDEuMjU1IDI1LjM2MjggOTguMDA2MiAyOS4wNDQ0Qzk0Ljg1NTQgMzIuNTk0MiA5MS4zMTA4IDM1LjY4NDEgODcuMzcyMyAzOC4zMTRDODMuNDMzOCA0MC44MTIgNzkuNjQzMSA0Mi4zODk2IDc2IDQzLjA0NzRWNzIuNjMwNEM4MC41MjkyIDcxLjQ0NzMgODQuNjE1NCA2OS42NzI0IDg4LjI1ODUgNjcuMzA1N0M5MiA2NC44MDc2IDk1LjI0OTIgNjEuODQ5MSA5OC4wMDYyIDU4LjQzMDdDOTkuNDY1NSA1Ni41NTEzIDEwMC43NDUgNTQuNTk4NiAxMDEuODQ2IDUyLjU3MjNWMTYwSDM1QzE1LjY3IDE2MCAwIDE0NC4zMyAwIDEyNVYzNVoiIGZpbGw9IiNmZmZmZmYiLz48L3N2Zz4K)](https://studio1hq.com)
[![Built with Bright Data](https://img.shields.io/badge/Built_with_Bright_Data-141414?style=for-the-badge)](https://brightdata.com)

[Bright Data docs](https://docs.brightdata.com) &nbsp;·&nbsp; [Bright Data](https://brightdata.com) &nbsp;·&nbsp; [Unsloth](https://unsloth.ai) &nbsp;·&nbsp; [More examples](https://github.com/Studio1HQ) &nbsp;·&nbsp; [studio1hq.com](https://studio1hq.com)

</div>

<br />

## Overview

A modular pipeline for fine-tuning `unsloth/gpt-oss-20b` with [Unsloth](https://unsloth.ai) and LoRA. It fits on a free Google Colab T4 GPU. Training data is collected by scraping documentation pages as Markdown with the Bright Data SDK (`brightdata-sdk`, `bdclient.scrape`), then turned into instruction and response pairs.

Each step is a separate script under `src/`, and `main.py` runs them in order. The same flow is available as a Colab notebook in `notebooks/GPT_OSS_Fine_tuning_Complete.ipynb`.

## Features

- **Faster training**: Uses Unsloth for roughly 2x faster fine-tuning.
- **Free GPU friendly**: Runs on a Colab T4 with 4-bit loading.
- **LoRA**: Trains under 1% of the parameters (`r=8`, `alpha=16`).
- **Reasoning effort control**: The tester checks low, medium and high reasoning levels.
- **Automatic data collection**: Bright Data scrapes the URLs listed in `config.py`.
- **Run any step**: Each script can run alone, and `main.py` supports `--skip-data-collection` and `--max-steps`.

## Tech stack

| Layer | Tool |
| --- | --- |
| Fine-tuning | [Unsloth](https://unsloth.ai), LoRA, TRL |
| Base model | [`unsloth/gpt-oss-20b`](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) |
| Data collection | [Bright Data](https://brightdata.com) (`brightdata-sdk`) |
| Libraries | PyTorch, Transformers, Datasets, bitsandbytes, Accelerate |
| Model hub | [Hugging Face](https://huggingface.co) |

## Getting started

### Prerequisites

- GPU with 16 GB or more VRAM (T4, V100 or A100)
- Python 3.8+
- CUDA 11.8+
- A [Bright Data](https://brightdata.com) API token
- A [Hugging Face](https://huggingface.co/settings/tokens) token

### Installation

```bash
git clone https://github.com/Studio1HQ/gpt-oss-finetuning.git
cd gpt-oss-finetuning
pip install -r requirements.txt
# Unsloth needs a special install
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
cp .env.example .env
```

### Environment variables

| Variable | Description |
| --- | --- |
| `BRIGHTDATA_API_TOKEN` | Bright Data API token. |
| `HF_TOKEN` | Hugging Face token, used to load and save models. |
| `MODEL_NAME` | Base model. Default `unsloth/gpt-oss-20b`. |
| `OUTPUT_DIR` | Output directory. Default `./outputs`. |

### Run

```bash
# Full pipeline
python main.py

# Reuse cached data
python main.py --skip-data-collection

# Train longer
python main.py --max-steps 100
```

Run single steps:

```bash
python src/08_model_tester.py   # test the model only
python src/09_model_saver.py    # save the model only
```

To use Colab instead, open `notebooks/GPT_OSS_Fine_tuning_Complete.ipynb`.

## How it works

```text
src/
├── 01_setup.py           # Environment check
├── 02_model_loader.py    # Load GPT-OSS
├── 03_lora_config.py     # LoRA setup
├── 04_data_collector.py  # Bright Data scraping
├── 05_data_formatter.py  # Format for training
├── 06_trainer_setup.py   # Configure trainer
├── 07_training.py        # Train
├── 08_model_tester.py    # Test with reasoning levels
└── 09_model_saver.py     # Save to disk or Hugging Face Hub
```

`config.py` holds the settings: sequence length 1024, 4-bit loading, batch size 2 with gradient accumulation 4, 60 steps, learning rate 2e-4, and the list of Python documentation URLs to scrape. The collector strips Markdown and links, filters short or navigation text, and pairs consecutive sentences as instruction and response.

Reported results from the original README (not re-measured here):

| Metric | Value |
| --- | --- |
| Training speed | about 30 tokens/sec |
| Memory usage | 12-14 GB |
| Training time | 10-15 min for 1000 examples |
| Trainable parameters | under 1% (LoRA) |

## Resources

- [Unsloth docs](https://docs.unsloth.ai): Fine-tuning guides and the gpt-oss notebooks.
- [Bright Data](https://brightdata.com/solutions/ai): Web data for AI.
- [Bright Data docs](https://docs.brightdata.com): API and SDK documentation.
- [GPT-OSS on Hugging Face](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4): Model collection.

## License

Released under the [MIT License](LICENSE).

<br />

---

<div align="center">

<a href="https://studio1hq.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Studio1HQ/website/main/brand/studio1-logo-on-dark.svg" />
    <img src="https://raw.githubusercontent.com/Studio1HQ/website/main/brand/studio1-logo-on-light.svg" alt="Studio1" width="160" />
  </picture>
</a>

Built by **[Studio1](https://studio1hq.com)**, the technical content and DevRel partner for devtool teams. Made with [Bright Data](https://brightdata.com).<br />
Browse more open-source examples at [github.com/Studio1HQ](https://github.com/Studio1HQ).

[![Website](https://img.shields.io/badge/studio1hq.com-0b0807?style=for-the-badge&logo=googlechrome&logoColor=white)](https://studio1hq.com)
[![Book a call](https://img.shields.io/badge/Book_a_strategy_call-f97316?style=for-the-badge&logo=googlecalendar&logoColor=white)](https://cal.com/studio1/collab)
[![X](https://img.shields.io/badge/%40Studio1HQ-141414?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Studio1HQ)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0a66c2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/studio1hq/)

<sub><a href="https://studio1hq.com">studio1hq.com</a> · Technical content, docs, video and DevRel for devtools</sub>

</div>
