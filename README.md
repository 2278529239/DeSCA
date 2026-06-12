<div align="center">

# 🔌 DeSCA

### Streaming Spatio-Temporal Prediction: A Decoupled and Selective Continual Adaptation Framework

<p><em>Paper under review</em></p>

</div>

<div align="center">

⭐ DeSCA is a decoupled and selective continual adaptation framework for streaming spatio-temporal prediction.

</div>

## Updates/News

🚩 **News (Jun. 2026):** DeSCA has been released as an open-source framework for streaming spatio-temporal prediction and continual adaptation.
## 📖 Introduction

Spatio-temporal data are continuously collected in ubiquitous cyber-physical systems, often exhibiting spatio-temporal non-stationarity and continuous distribution shifts. This requires spatio-temporal models to continually adapt to streaming spatio-temporal data.

Existing methods that update backbone parameters are limited in both efficiency and effectiveness, incurring substantial training costs and risking catastrophic forgetting. Recent methods improve efficiency by freezing the backbone and updating lightweight plug-in parameters. However, most lightweight adaptation methods apply a rigid adaptation strategy to spatial and temporal changes, overlooking that heterogeneous spatio-temporal non-stationarity may manifest as distinct temporal shifts, spatial shifts, or joint shifts. This often introduces unnecessary updates and limits adaptation effectiveness.

To address these limitations, we propose **DeSCA**, a **Decoupled and Selective Continual Adaptation** framework for streaming spatio-temporal prediction.

Specifically:

- A **Decoupled Prompt Mapping Module** constructs spatial and temporal prompts to provide separate adaptation feeds.
- A **Prototype-aware Shift Detection Module** derives spatial and temporal deviations to identify temporal-dominant, spatial-dominant, and joint shift patterns.
- A **Selective Gated Updating Module** filters insignificant deviations and activates only the prompt parameters indicated by effective deviations, reducing unnecessary updates and improving adaptation stability.

Extensive experiments on three real-world streaming spatio-temporal datasets demonstrate that DeSCA consistently improves four representative backbone architectures, achieving an average performance gain of **8.56%** across all backbones and evaluation metrics while maintaining **linear time and space complexity**.

<p align="center">
    <img src="fig/structure.png" alt="DeSCA Framework" align="center" width="800px" />
</p>

## 📊 Datasets

The framework supports the following streaming spatio-temporal datasets:

| Dataset | Description | Scenario |
|----------|-------------|----------|
| **PEMS03-Stream** | Streaming traffic data derived from PEMS03 | Traffic forecasting |
| **PEMS04-Stream** | Streaming traffic data derived from PEMS04 | Traffic forecasting |
| **Air-Stream** | Streaming air quality monitoring data | Air quality forecasting |

### Dataset Download
- **PEMS03-Stream** and **AIR-Stream**: Available from the [EAC repository](https://github.com/Onedean/EAC)
- **PEMS04-Stream**: Available from the [TEAM repository](https://github.com/kvmduc/TEAM-topo-evo-traffic-forecasting)

Please download the datasets and place them in the `data/` directory.

**Directory names should be kept as follows:**

```text
data/
├── PEMS03/
├── PEMS04/
└── AIR/

## 🚀 Getting Started

### Installation

Create and activate the environment:

```bash
conda env create -f environment.yaml
conda activate stg
```

### Quick Start

DeSCA is a plug-and-play continual adaptation framework that can be integrated into different streaming spatio-temporal forecasting backbones.

#### Example 1: DeSCA + EAC

```bash
python main_pre.py \
    --conf conf/PEMS03/eac_DeSCA.json \
    --gpuid 0 \
    --seed 43
```

#### Example 2: DeSCA + STBP

```bash
python mainSTBP.py \
    --conf conf/PEMS03/STBP_DeSCA.json \
    --gpuid 0 \
    --seed 43
```

#### Example 3: DeSCA + DCRNN

```bash
python main_pre.py \
    --conf conf/PEMS03/DCRNN_DeSCA.json \
    --gpuid 0 \
    --seed 43
```

#### Example 4: DeSCA + PDFormer

```bash
python main_pre.py \
    --conf conf/PEMS03/PDFormer_DeSCA.json \
    --gpuid 0 \
    --seed 43
```

#### Run All Experiments

To reproduce all experiments reported in the paper:

```bash
bash scripts/run_all.sh
```

| Backbone | Model Family | Variant |
|-----------|-------------|----------|
| DCRNN | RNN-based | DCRNN w/ DeSCA |
| PDFormer | Transformer-based | PDFormer w/ DeSCA |
| EAC | STGNN-based | EAC w/ DeSCA |
| STBP | Frequency-based | STBP w/ DeSCA |

DeSCA is not a standalone forecasting model.

Instead, it functions as a decoupled and selective continual adaptation framework that can be seamlessly integrated into diverse streaming spatio-temporal forecasting backbones, including RNN-based, Transformer-based, STGNN-based, and frequency-based architectures.


## 🔗 Acknowledgement

We gratefully acknowledge the following open-source projects, whose codebases, datasets, and research insights have supported the development of this work:

- [EAC](https://github.com/Onedean/EAC)
- [STBP](https://github.com/Aoyu-Liu/STBP)
- [DCRNN](https://github.com/liyaguang/DCRNN)
- [PDFormer](https://github.com/BUAABIGSCity/PDFormer)
- [TrafficStream](https://github.com/AprLie/TrafficStream)
- [STKEC](https://github.com/wangbinwu13116175205/STKEC)

We thank the authors for their valuable contributions to the spatio-temporal forecasting and continual learning communities.