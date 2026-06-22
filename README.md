<div align="center">
  <h1>TailorMind</h1>
  <p><strong>From user behavior to tailored multimodal content.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/SSLRec-WSDM%202024-6C5CE7?style=for-the-badge" alt="SSLRec">
    <img src="https://img.shields.io/badge/License-MIT-00B894?style=for-the-badge" alt="MIT License">
  </p>

  <img src="./assets/framework.png" alt="TailorMind framework" width="100%">
</div>

## Overview

TailorMind is a personalized AIGC framework that connects recommendation systems, multimodal understanding, iterative user profiling, and content generation. It learns from real user behavior on social/content platforms, builds preference-aware user profiles, and generates tailored outputs such as social posts, image-text notes, and video ideas.

The framework is designed around a simple loop:

```text
User behavior -> SSL recommendation -> multimodal item understanding
              -> user profile refinement -> tailored content generation
              -> evaluation and reflection -> refined profile
```

## Highlights

| Capability | What it does |
| --- | --- |
| Multimodal behavior modeling | Builds user-item interaction signals from text, images, and videos. |
| SSL recommendation engine | Integrates the SSLRec framework for self-supervised recommendation across multiple model families. |
| Multimodal analysis agents | Uses text, image, and video analysts to convert raw content into structured item profiles. |
| Iterative profile optimization | Refines user profiles through validation/test feedback, hit analysis, and reflection records. |
| Personalized content generation | Converts refined user profiles into post ideas, image-text content, and video-oriented creative directions. |
| Platform-aware data pipeline | Supports Bilibili, Xiaohongshu/RedBook, Hupu, and Douban-oriented data flows. |

## Architecture

TailorMind contains four major stages:

1. Data collection: crawlers and realtime collectors gather platform interactions and media assets.
2. Recommendation: SSLRec ranks candidate items and prepares recommended, historical, validation, and test sets.
3. Profiling: multimodal agents summarize items, build user profiles, and refine them with iterative feedback.
4. Generation: profile-to-idea and product-generation agents create personalized text-image posts and video concepts.

## Repository Layout

```text
.
├── agents/                 # LLM/VLM agents for analysis, profiling, reflection, and generation
├── assets/
│   └── framework.png       # TailorMind framework figure used by this README
├── SSLRec/                 # Integrated self-supervised recommendation framework
├── tools/                  # Pipeline scripts for recommendation, analysis, realtime data, and evaluation
├── main.py                 # Top-level orchestration entry
├── bilibli_spider.py       # Bilibili crawler
├── redbookSpider.py        # Xiaohongshu/RedBook crawler
├── requirements.txt
└── LICENSE
```

## Core Modules

| Module | Main files | Purpose |
| --- | --- | --- |
| Recommendation | `tools/sslrec.py`, `SSLRec/` | Trains/runs SSLRec models and prepares recommendation outputs. |
| Offline analysis | `tools/analyze.py` | Analyzes downloaded user/item media and builds initial item/user profiles. |
| Enhanced analysis | `tools/enhanced_analyze.py` | Performs hit analysis, validation/test reflection, and profile refinement. |
| Realtime analysis | `tools/realtime_analyze.py`, `tools/enhanced_realtime_analyze.py` | Processes newly collected platform data. |
| Content generation | `tools/product.py`, `agents/profile2idea.py`, `agents/itproduct_generator.py`, `agents/vproduct_generator.py` | Generates ideas, image-text content, and video-oriented products. |
| Evaluation | `tools/stat_ndcg_from_json.py`, `tools/evaluate_post_quality.py`, `tools/evaluate_html_post.py`, `tools/stat_post_evaluations.py` | Computes recommendation/profile/content quality statistics. |

## Quick Start

### 1. Clone

```bash
git clone https://github.com/iLearn-Lab/TailorMind.git
cd TailorMind
```

### 2. Create Environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install PyTorch according to your CUDA version first. For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the core runtime packages used by TailorMind:

```bash
pip install openai python-dotenv pandas numpy requests beautifulsoup4 selenium aiohttp pillow moviepy yt-dlp tenacity scipy pyyaml tqdm tensorboard
```

SSLRec models may require extra packages such as `dgl`, depending on the selected model.

### 4. Configure Environment Variables

Create a `.env` file in the repository root:

```env
# Dataset: bilibili, redbook, hupu, douban
DATASET=bilibili

# SSLRec
MODEL=SGL
SKIP_DOWNLOAD=false

# Chat / multimodal model API
CHAT_API_KEY=your_api_key
CHAT_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o
VIDEO_MODEL=gpt-4o

# Image generation, optional
IMAGE_API_KEY=your_image_api_key
IMAGE_BASE_URL=your_image_base_url
IMAGE_MODEL=your_image_model

# Reflection / retrieval helpers, optional
SEARCH_MODEL=gpt-4o
VISION_MODEL=gpt-4o
TEST_EACH_ROUND=true

# Platform cookies, when needed
REDBOOK_COOKIE=
```

### 5. Prepare Data

TailorMind expects platform data and downloaded media to follow this convention:

```text
dataset/{DATASET}/{real_user_id}/
download/{DATASET}/{real_user_id}/
├── historical/{item_id}/
├── recommended/{item_id}/
├── validation/{item_id}/
└── test/{item_id}/
```

SSLRec mappings and matrices are expected under:

```text
SSLRec/datasets/general_cf/{DATASET}/
├── user_map.json
├── item_map.json
├── valid_matrix.pkl
└── test_matrix.pkl
```

Each item folder can contain `.txt`, `.jpg`/`.png`, and `.mp4` files. TailorMind will write outputs such as `analysis.json`, `item_profiles.txt`, `user_profile.txt`, `product_ideas.json`, `valid_reflection_results.json`, and `test_reflection_results.json`.

## Running the Pipeline

The current `main.py` is an orchestration script with several stages available as commented blocks. By default, it runs enhanced analysis:

```bash
python main.py
```

For modular runs, call individual tools:

```bash
# Run SSLRec recommendation
python -c "from dotenv import load_dotenv; load_dotenv(); from tools.sslrec import run_sslrec; run_sslrec()"

# Build multimodal item profiles and user profiles
python -c "from dotenv import load_dotenv; load_dotenv(); from tools.analyze import analyze; analyze(max_workers=15)"

# Run iterative profile refinement
python -c "from dotenv import load_dotenv; load_dotenv(); from tools.enhanced_analyze import enhanced_analyze; enhanced_analyze(max_concurrent_users=4)"

# Generate personalized products from user profiles
python -c "from dotenv import load_dotenv; load_dotenv(); from tools.product import product; product()"
```

## Realtime Collection

Realtime collectors can be used after historical data has been prepared:

```python
from tools.bilibili_realtime_download import BilibiliRealTime
from tools.redbook_realtime import RedBookRealtime
from tools.hupu_realtime import HupuRealTime

bilibili = BilibiliRealTime()
bilibili.process_all_users(save_data=True, download_videos=True)

redbook = RedBookRealtime()
redbook.process_all_users(save_data=True, download_media=True, max_users=200, parallel=3)

hupu = HupuRealTime()
hupu.process_all_users(save_data=True, download_media=True, max_users=200)
```

Platform collectors may require cookies, browser drivers, and compliance with each platform's terms of service.

## Outputs

| Output | Location | Description |
| --- | --- | --- |
| Item analysis | `download/{DATASET}/{user}/{split}/{item}/analysis.json` | Multimodal summary of text/images/videos. |
| Item profile bundle | `download/{DATASET}/{user}/item_profiles.txt` | Aggregated item-level evidence for profile generation. |
| User profile | `download/{DATASET}/{user}/user_profile.txt` | Preference profile generated from item evidence. |
| Product ideas | `download/{DATASET}/{user}/product_ideas.json` | Profile-conditioned creative ideas. |
| Reflection records | `valid_reflection_results.json`, `test_reflection_results.json` | Iterative validation/test feedback. |
| Global stats | `download/{DATASET}/global_reflection_statistics.json` | Aggregated NDCG/Hit Rate statistics across users. |

## Evaluation

Useful evaluation scripts:

```bash
# Aggregate NDCG and hit statistics from reflection JSON files
python tools/stat_ndcg_from_json.py --dataset bilibili

# Evaluate generated post quality
python tools/evaluate_post_quality.py

# Evaluate HTML post outputs
python tools/evaluate_html_post.py

# Summarize post evaluation versions
python tools/stat_post_evaluations.py
```

## Notes

- The top-level project is released under the MIT License.
- The `SSLRec/` directory is an integrated recommendation component; see `SSLRec/README.md` and `SSLRec/LICENSE.txt` for its original documentation and license.
- Large datasets, downloaded media, model checkpoints, cookies, and generated artifacts should usually stay out of version control.
- For reproducible experiments, record `DATASET`, `MODEL`, model API names, and reflection settings alongside generated results.

## Citation

If you use the integrated SSLRec component, please also cite:

```bibtex
@inproceedings{ren2024sslrec,
  title={SSLRec: A Self-Supervised Learning Framework for Recommendation},
  author={Ren, Xubin and Xia, Lianghao and Yang, Yuhao and Wei, Wei and Wang, Tianle and Cai, Xuheng and Huang, Chao},
  booktitle={Proceedings of the 17th ACM International Conference on Web Search and Data Mining},
  pages={567--575},
  year={2024}
}
```
