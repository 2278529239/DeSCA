<div align="center">
  <h2><b><big>🔌 DeSCA</big>

 <u>De</u>viation-aware <u>S</u>patio-temporal <u>C</u>ontinual <u>A</u>daptation: A Plug-and-Play Framework for Streaming Spatio-Temporal Prediction</b></h2>
  <p><em>（论文名：还没确定 | 论文链接：还没确定）</em></p>
</div>
<div align="center">
未知
</div>
<div align="center">
> ⭐ DeSCA is a plug-and-play framework for streaming spatio-temporal prediction, enabling continual adaptation to evolving graph structures and distribution shifts.
</div>

## Updates/News:
🚩 **News** (Jun. 2026): DeSCA framework is now fully open source with support for streaming spatio-temporal prediction!

## 📖 Introduction
DeSCA is a **plug-and-play plugin module** designed to enhance existing spatio-temporal prediction models through innovative mechanisms. Unlike standalone models, DeSCA can be seamlessly integrated into various spatio-temporal forecasting backbones such as STID, GWNET, DCRNN, and PDFormer.

**Core Mechanisms:**
- **Spatio-temporal Feature Decoupling**: Separates spatial and temporal features to better capture their distinct patterns and dynamics, enabling more precise modeling of complex spatio-temporal relationships.
- **Deviation Detection with Gated Update**: Incorporates a gating mechanism that detects distribution shifts in streaming data and adaptively updates model parameters, enabling continual adaptation without catastrophic forgetting.

By integrating DeSCA into your spatio-temporal forecasting pipeline, you can effectively improve prediction accuracy and robustness when faced with evolving graph structures and distribution shifts in streaming scenarios.

<p align="center">
    <img src="fig/structure.png" alt="DeSCA Framework" align="center" width="800px" />
</p>

## 📊 Datasets
The framework supports the following datasets:

| Dataset | Description | Scenarios |
|---|---|---|
| **PEMS-Stream** | Traffic flow data from California highways | Traffic forecasting |
| **AIR-Stream** | Air quality monitoring data | Air quality forecasting |
| **PEMS04** | Traditional PEMS dataset for baseline comparison | Traffic forecasting |

### Dataset Download
- **PEMS-Stream** and **AIR-Stream**: Available from the [EAC repository](https://github.com/Onedean/EAC)
- **PEMS04**: Available from the [TEAM repository](https://github.com/kvmduc/TEAM-topo-evo-traffic-forecasting)

Please download the datasets and place them in the `data/` directory.

## 🚀 Getting Started

### Installation
1. Create and activate the environment:
```shell
conda env create -f environment.yaml
conda activate stg
```

### Quick Start
Run the main training script with different configurations:

#### EAC Method
```shell
python main_pre.py --conf conf/PEMS/eac.json --gpuid 0 --seed 42
```

#### STBP Method
```shell
python mainSTBP.py --conf conf/PEMS/STBP_PEMS.json --gpuid 0 --seed 42
```

#### Baseline Methods
```shell
# DCRNN baseline
python main_pre.py --conf conf/PEMS/DCRNNpre.json --gpuid 0 --seed 42

# PDFormer baseline
python main_pre.py --conf conf/PEMS/PDFormerpre.json --gpuid 0 --seed 42
```

### Supported Methods
| Method | Type | Description |
|---|---|---|
| **EAC** | Prompt tuning | Expand and Compress for continual learning |
| **STBP** | Pattern bank | Spatio-temporal backbone with pattern bank |
| **DCRNNpre** | Baseline | DCRNN with incremental adaptation |
| **PDFormerpre** | Baseline | PDFormer with incremental adaptation |
| **STID** | Baseline | Spatial-Temporal Identity network |
| **GWNET** | Baseline | Graph WaveNet |
| **STKEC** | Baseline | Spatio-temporal knowledge embedding |
| **TrafficStream** | Baseline | Traffic stream learning |

## 📁 Project Structure
```
CLST/
├── conf/                  # Configuration files
│   ├── PEMS/              # PEMS-Stream dataset configs
│   ├── AIR/               # AIR-Stream dataset configs
│   ├── PEMS04/            # PEMS04 dataset configs
│   └── ...
├── src/                   # Source code
│   ├── model/             # Model definitions
│   │   ├── modelpre.py    # EAC/DCRNN/PDFormer models
│   │   └── modelSTBP.py   # STBP model
│   ├── trainer/           # Training logic
│   │   ├── default_trainerpre.py
│   │   └── default_trainerSTBP.py
│   └── ...
├── utils/                 # Utility functions
├── main_pre.py            # Main entry for EAC/DCRNN/PDFormer methods
├── mainSTBP.py            # Main entry for STBP method
└── environment.yaml       # Environment configuration
```

## 🎯 Experimental Results

后补

## 📝 Configuration
All experiment configurations are stored in the `conf/` directory. Each dataset has its own subdirectory with configuration files for different methods. Key configuration parameters:

- `method`: Method name (eac, stbp, dcrnnpre, etc.)
- `year`: Current training year
- `begin_year`: Starting year for incremental training
- `end_year`: Ending year
- `batch_size`: Batch size
- `lr`: Learning rate
- `epochs`: Number of epochs
- `patience`: Early stopping patience

## 📚 References

⚠️ **论文名：还没确定** | **论文链接：还没确定**

If you use this framework in your research, please consider citing the following papers:

```
@inproceedings{chen2025eac,
  title={Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting},
  author={Wei Chen and Yuxuan Liang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

```
@inproceedings{liu2026stbp,
  title={A General Spatio-Temporal Backbone with Scalable Contextual Pattern Bank for Urban Continual Forecasting},
  author={Aoyu Liu and others},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## 🙏 Acknowledgement
We greatly appreciate the following GitHub repositories for their valuable code and contributions:

- [EAC](https://github.com/Onedean/EAC) - Expand and Compress framework
- [STBP](https://github.com/Aoyu-Liu/STBP) - Spatio-temporal pattern bank
- [TrafficStream](https://github.com/AprLie/TrafficStream) - Traffic stream learning
- [STKEC](https://github.com/wangbinwu13116175205/STKEC) - Spatio-temporal knowledge embedding


## 📄 License
This project is licensed under the Apache-2.0 License.

---

*Built with ❤️ for continual spatio-temporal learning research*
