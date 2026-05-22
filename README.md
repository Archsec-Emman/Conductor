# Conductor

[![Language](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2508.02739-b31b1b.svg)](https://arxiv.org/abs/2508.02739)

**Conductor: A Foundation Model for the Language of Financial Markets**  
The first open‑source foundation model for financial candlesticks (K‑lines), pre‑trained on data from over 45 global exchanges.

[Live Demo](https://Archsec-Emman.github.io/Conductor-demo/) • [Paper](https://arxiv.org/abs/2508.02739) • [Hugging Face](https://huggingface.co/NeoQuasar)

---

## 📖 Overview

Conductor is a family of decoder‑only foundation models designed specifically for the unique, high‑noise, and non‑stationary characteristics of financial K‑line (OHLCV) data. Unlike general‑purpose time‑series foundation models (TSFMs), which often underperform on financial data, Conductor leverages a two‑stage framework:

1. **Specialised Tokenizer** – Quantises continuous, multi‑dimensional K‑line data (open, high, low, close, volume, amount) into hierarchical discrete tokens while preserving both price dynamics and trade activity patterns.
2. **Autoregressive Transformer** – Pre‑trained on a massive, multi‑market corpus of **over 12 billion K‑line records from 45 global exchanges**, enabling nuanced temporal and cross‑asset representations in a zero‑shot setting.

## ✨ Key Features

### Core Capabilities
- **Price Series Forecasting** – 93% improvement in RankIC over leading TSFMs on benchmark datasets.
- **Volatility Prediction** – 9% lower MAE compared to non‑pre‑trained baselines.
- **Synthetic Data Generation** – 22% improvement in generative fidelity for realistic K‑line sequences.

### Technical Highlights
- **Multi‑Asset Support** – Works with equities, crypto, FX, and commodities across 45 global exchanges.
- **Flexible Input** – Accepts OHLCV data; volume and amount columns are optional.
- **Batch Prediction** – Parallel inference on multiple time series simultaneously with GPU acceleration.
- **Fine‑Tuning Pipeline** – Full support for adapting Conductor to custom datasets using Qlib (A‑share market example) or plain CSV files.
- **Web UI** – Intuitive graphical interface with K‑line charts, parameter tuning, and multi‑device support (CPU/CUDA/MPS).

### Model Zoo

| Model | Tokenizer | Context Length | Params | Open‑Source |
|-------|-----------|----------------|--------|-------------|
| **Conductor‑mini** | Conductor‑Tokenizer‑2k | 2048 | 4.1M | ✅ |
| **Conductor‑small** | Conductor‑Tokenizer‑base | 512 | 24.7M | ✅ |
| **Conductor‑base** | Conductor‑Tokenizer‑base | 512 | 102.3M | ✅ |
| Conductor‑large | Conductor‑Tokenizer‑base | 512 | 499.2M | ❌ |

All open‑source models are available on the [Hugging Face Hub](https://huggingface.co/NeoQuasar).

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch (with CUDA/MPS support as needed)
- pandas & numpy
- transformers (Hugging Face)
- qlib (optional, for A‑share fine‑tuning example)
- Flask (for web UI)

---

## 📈 Making Forecasts

Conductor provides a simple `ConductorPredictor` class that handles preprocessing, normalisation, prediction, and inverse normalisation.

### Step 1: Load Model & Tokenizer
```python
from model import Conductor, ConductorTokenizer, ConductorPredictor

tokenizer = ConductorTokenizer.from_pretrained("NeoQuasar/Conductor-Tokenizer-base")
model = Conductor.from_pretrained("NeoQuasar/Conductor-small")
```

### Step 2: Initialise Predictor
```python
predictor = ConductorPredictor(model, tokenizer, max_context=512)
```

### Step 3: Prepare Input Data
Your DataFrame must include `['open', 'high', 'low', 'close']` columns; `volume` and `amount` are optional.

```python
import pandas as pd

df = pd.read_csv("./data/your_data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120

x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
```

### Step 4: Generate Forecasts
```python
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # Temperature for sampling
    top_p=0.9,      # Nucleus sampling probability
    sample_count=1  # Number of forecast paths (averaged)
)

print(pred_df.head())  # Contains open, high, low, close, volume, amount
```

### Batch Prediction (Multiple Time Series)
```python
df_list = [df1, df2, df3]
x_timestamp_list = [x_ts1, x_ts2, x_ts3]
y_timestamp_list = [y_ts1, y_ts2, y_ts3]

pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)
```

**Requirements for batch prediction:**
- All series must have the same lookback window length and prediction length (`pred_len`).
- Each DataFrame must include the required OHLC columns.

---

## 🖥️ Web UI

Conductor includes a complete web‑based interface for interactive forecasting.

### Quick Start
```bash
cd webui
python run.py
# or
./start.sh
# or directly
python app.py
```

Then open `http://localhost:7070` in your browser.

### Features
- **Multi‑format Data Support** – CSV, Feather, and other financial formats.
- **Smart Time Window** – Fixed 400+120 data point slider selection.
- **Real Model Prediction** – Integrated Conductor models (mini, small, base).
- **Prediction Quality Control** – Adjustable temperature (0.1–2.0), nucleus sampling (top_p 0.1–1.0), and sample count (1–5).
- **Multi‑Device Support** – CPU, CUDA (NVIDIA), MPS (Apple Silicon).
- **Comparison Analysis** – Detailed error analysis and prediction quality assessment.
- **K‑Line Chart Display** – Professional financial candle charts powered by Plotly.js.

### Recommended Parameters
| Parameter | Range | Recommendation |
|-----------|-------|----------------|
| Temperature (T) | 0.1–2.0 | 1.2–1.5 for better quality |
| Nucleus Sampling (top_p) | 0.1–1.0 | 0.95–1.0 to consider more possibilities |
| Sample Count | 1–5 | 2–3 samples to improve quality |

---

## 🔧 Fine‑Tuning on Custom Data

Conductor provides complete pipelines for adapting pre‑trained models to your own datasets. Two approaches are available:

### Option A: Fine‑Tune with Qlib (A‑Share Market Example)

This pipeline demonstrates fine‑tuning on Chinese A‑share market data using Microsoft's Qlib library.

**Prerequisites:**
- Install `pyqlib` and prepare Qlib data following the [official guide](https://github.com/microsoft/qlib).
- Daily frequency data is assumed.

**Step 1 – Configure Experiment:**  
Edit `finetune/config.py` to set:
- `qlib_data_path`: Path to your local Qlib data.
- `dataset_path`: Where processed pickle files will be saved.
- `save_path`: Base directory for model checkpoints.
- `backtest_result_path`: Directory for backtesting results.
- `pretrained_tokenizer_path` and `pretrained_predictor_path`: Starting models (local or Hugging Face names).

**Step 2 – Prepare Dataset:**  
```bash
python finetune/qlib_data_preprocess.py
```
This loads raw market data, splits into training/validation/test sets, and saves them as `train_data.pkl`, `val_data.pkl`, and `test_data.pkl`.

**Step 3 – Fine‑Tune (Multi‑GPU Supported):**  
```bash
# Fine‑tune tokenizer
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py

# Fine‑tune predictor
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```

**Step 4 – Backtesting Evaluation:**  
```bash
python finetune/qlib_test.py --device cuda:0
```
Outputs performance metrics and generates a cumulative return plot comparing your strategy against the benchmark.

### Option B: Fine‑Tune on Custom CSV Data

The `finetune_csv/` module provides a flexible pipeline for any CSV‑formatted financial data.

**Required CSV Columns:**
| Column | Description |
|--------|-------------|
| `timestamps` | DateTime stamps for each data point |
| `open` | Opening price |
| `high` | Highest price |
| `low` | Lowest price |
| `close` | Closing price |
| `volume` | Trading volume |
| `amount` | Trading amount (can be 0 if not available) |

**Configuration:**  
Edit `finetune_csv/configs/config_ali09988_candle-5min.yaml` (or create your own) to set data paths, lookback window, prediction window, and training parameters.

**Sequential Training (Recommended):**  
```bash
# Complete training (tokenizer + predictor)
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml

# Skip existing models
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-existing

# Train tokenizer only
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-basemodel

# Train predictor only
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-tokenizer
```

**Distributed Training (DDP):**  
```bash
DIST_BACKEND=nccl torchrun --standalone --nproc_per_node=8 train_sequential.py --config configs/config_ali09988_candle-5min.yaml
```

**Training Outputs:**
- **Tokenizer Checkpoints:** `{base_save_path}/{exp_name}/tokenizer/best_model/`
- **Predictor Checkpoints:** `{base_save_path}/{exp_name}/basemodel/best_model/`
- **Logs:** Console output + detailed log files

---

## 📁 Project Structure

```
Conductor/
├── model/                  # Core Conductor model implementation
│   ├── conductor.py        # Main model classes
│   ├── module.py           # Transformer modules
│   └── __init__.py
├── finetune/               # Fine‑tuning pipeline (Qlib A‑share example)
│   ├── config.py           # Configuration
│   ├── dataset.py          # Data loading utilities
│   ├── qlib_data_preprocess.py
│   ├── qlib_test.py        # Backtesting evaluation
│   ├── train_tokenizer.py
│   ├── train_predictor.py
│   └── utils/
├── finetune_csv/           # Fine‑tuning on custom CSV data
│   ├── train_sequential.py # Complete training pipeline
│   ├── finetune_tokenizer.py
│   ├── finetune_base_model.py
│   └── configs/            # YAML configuration files
├── webui/                  # Flask web interface
│   ├── app.py              # Main application
│   ├── run.py              # Launcher script
│   ├── start.sh
│   └── requirements.txt
├── examples/               # Example scripts
│   ├── prediction_example.py
│   ├── prediction_batch_example.py
│   ├── prediction_cn_markets_day.py
│   ├── get_akshare_date_2024-2025_x.py
│   └── ...
├── tests/                  # Unit tests
├── figures/                # Documentation images
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## 📊 Performance Benchmarks

Conductor has been rigorously evaluated on standard financial benchmarks:

| Task | Improvement vs. Leading TSFM |
|------|------------------------------|
| Price Series Forecasting (RankIC) | **+93%** |
| Price Series Forecasting (vs. best non‑pre‑trained baseline) | **+87%** |
| Volatility Forecasting (MAE) | **-9%** |
| Synthetic K‑line Generation (Fidelity) | **+22%** |

These results establish Conductor as a robust, versatile foundation model for end‑to‑end financial time‑series analysis.

---

## 🧪 Live Demo

A live demo is available that visualises Conductor's forecasting results for the **BTC/USDT** trading pair over the next 24 hours.

👉 **[Access the Live Demo Here](https://Archsec-Emman.github.io/Conductor-demo/)**

---

## 📝 Important Notes

### Context Length Limitations
- `Conductor-small` and `Conductor-base` have a maximum context length of **512**.
- For optimal performance, ensure your input `lookback` length does not exceed this limit.
- The `ConductorPredictor` will automatically truncate longer contexts.

### AI‑Generated Comments
Many code comments in the `finetune/` directory were generated by an AI assistant (Gemini 2.5 Pro) for explanatory purposes. While they aim to be helpful, they may contain inaccuracies. Treat the code itself as the definitive source of logic.

### From Demo to Production
The fine‑tuning pipeline is a demonstration, not a production‑ready quantitative trading system. Robust strategies require:
- Portfolio optimisation to neutralise exposure to common risk factors (market beta, size, value).
- Meticulous modelling of transaction costs, slippage, and market impact.
- More complex logic for portfolio construction, dynamic position sizing, and risk management (stop‑loss/take‑profit rules).

---

## 🙏 Citation

If you use Conductor in your research, please cite our paper:

```bibtex
@misc{shi2025conductor,
  title={Conductor: A Foundation Model for the Language of Financial Markets},
  author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
  year={2025},
  eprint={2508.02739},
  archivePrefix={arXiv},
  primaryClass={q-fin.ST},
  url={https://arxiv.org/abs/2508.02739},
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Areas for Contribution
- Additional data connectors (more exchange APIs, alternative data sources)
- New evaluation metrics and benchmarks
- Optimised inference backends (ONNX, TensorRT)
- Production‑ready deployment examples (Docker, Kubernetes)
- More fine‑tuning examples (crypto, FX, commodities)

---

## 📬 Contact & Support

- **GitHub Issues**: [https://github.com/Archsec-Emman/Conductor/issues](https://github.com/Archsec-Emman/Conductor/issues)
- **Author**: [Archsec-Emman](https://github.com/Archsec-Emman)

---

*If you find Conductor useful in your research or trading workflows, please consider giving the repository a star.* ⭐
```
