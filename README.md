# SL-AD: Detection of Electromagnetic Temporal Anomalies Associated with Major Earthquakes via a Two-Stage Deep Learning Framework

<p align="center">
  Zining Yu<sup>1</sup>, Xilong Jing<sup>1</sup>, Jiarui Zhang<sup>1</sup>,
  Minglin Yang<sup>1</sup>, Shanzhi Dong<sup>1</sup>, and
  Haiyong Zheng<sup>1,2,3,*</sup>, Senior Member, IEEE
</p>

<p align="center">
  <sup>1</sup>College of Electronic Engineering, Ocean University of China, Qingdao 266404, China<br>
  <sup>2</sup>Shandong Key Laboratory of Intelligent Sensing Chips and Systems,
  Ocean University of China, Qingdao, China<br>
  <sup>3</sup>Shenzhen Research Institute of Ocean University of China,
  Shenzhen 518100, China
</p>

<p align="center">
  <sup>*</sup>Corresponding author:
  <a href="mailto:zhenghaiyong@ouc.edu.cn">Haiyong Zheng</a>
</p>

## 📑 Table of Contents

<details open>
<summary><b>Click to expand/collapse navigation</b></summary>

- **[📌 Highlights](#-highlights)**
- **[🔬 Abstract](#-abstract)**
- **[📡 Dataset and Study Region](#-dataset-and-study-region)**
- **[🏗️ Methodology](#%EF%B8%8F-methodology)**
  - [Sequence Learning Stage](#sequence-learning-stage)
  - [Temporal Feature Compression](#temporal-feature-compression)
  - [Autoformer-Based Feature Learning](#autoformer-based-feature-learning)
  - [Guided Local Adaptive Feature Filtering](#guided-local-adaptive-feature-filtering)
  - [Anomaly Detection Stage](#anomaly-detection-stage)
- **[📊 Experimental Results](#-experimental-results)**
  - [Overall Performance](#overall-performance)
  - [Earthquake Case Analysis](#earthquake-case-analysis)
  - [Model Comparison](#model-comparison)
  - [Interpretability Analysis](#interpretability-analysis)
- **[💻 Code](#-code-coming-soon)**
- **[📦 Data](#-data)**
- **[🙏 Acknowledgment](#-acknowledgment)**
- **[📧 Contact](#-contact)**

</details>

---

## 📌 Highlights

### Key Contributions

1. A two-stage deep learning framework, termed **SL-AD
   (Sequence Learning–Anomaly Detection)**, is proposed for
   electromagnetic earthquake precursor detection.

2. A large-scale upstream sequence-learning task is introduced to
   learn transferable temporal representations before downstream
   earthquake anomaly classification.

3. A multi-scale temporal representation architecture combines
   **PCA-based feature compression, 1-D convolution, Autoformer,
   and Guided Local Adaptive Feature Filtering (GLAFF)**.

4. A downstream **BiLSTM-based anomaly classifier** transfers
   the learned temporal representations to highly imbalanced
   earthquake precursor detection.

5. Attention visualization and feature-encoding correlation analysis
   are introduced to investigate the temporal characteristics learned
   from earthquake-related electromagnetic signals.

### Performance Highlights

On the naturally imbalanced anomaly-detection test set, SL-AD achieves:

| Metric | Performance |
|---|---:|
| ROC-AUC | **0.824** |
| Accuracy | **0.872** |
| Recall | **0.619** |
| FNR | **0.381** |
| FPR | **0.124** |

The results demonstrate improved discrimination of rare
earthquake-related electromagnetic anomalies compared with several
representative time-series baselines.

---

## 🔬 Abstract

Earthquake precursor detection remains challenging because
electromagnetic observations exhibit complex temporal variations,
while major-earthquake samples are extremely sparse and highly
imbalanced.

To address these challenges, we propose **SL-AD
(Sequence Learning–Anomaly Detection)**, a two-stage deep learning
framework for detecting electromagnetic temporal anomalies associated
with major earthquakes.

In the upstream Sequence Learning stage, the model learns temporal
patterns from large-scale electromagnetic observations by extracting
and modeling multi-scale temporal components. In the downstream
Anomaly Detection stage, the learned representations are transferred
to a BiLSTM-based classifier to identify possible pre-earthquake
anomalies.

Experiments using electromagnetic observations from southwestern
China demonstrate that SL-AD achieves an ROC-AUC of 0.824 on a
highly imbalanced test set. Case studies of eight earthquakes during
2022–2023 further show that detected electromagnetic anomalies tend
to accumulate several days before earthquake occurrence.

Additional ablation, visualization, and feature-correlation analyses
demonstrate the importance of both the two-stage learning strategy
and multi-scale temporal representation for electromagnetic precursor
analysis.

## 📡 Dataset and Study Region

### Study Region

The study focuses on southwestern China within approximately:

```text
Latitude : 22°N – 34°N
Longitude: 98°E – 107°E
```
The region contains several major active fault systems and is one of the most tectonically active regions in China.

### AETA Electromagnetic Observations
Electromagnetic observations are obtained from the Acoustic and Electromagnetics to Artificial Intelligence (AETA) monitoring system.

The dataset contains observations from:
```text
Observation period: 2017-01-01 to 2023-05-05
Sampling interval: 10 minutes
```
Three low-frequency electromagnetic bands are used:

```text
0–5 Hz
5–10 Hz
10–15 Hz
```
### Earthquake Dataset
Earthquakes with:
```text
Ms >= 5.0
```
within the study region are considered.

Eight major earthquakes occurring during 2022–2023 are used for case-based evaluation.

---

## 🏗️ Methodology

SL-AD consists of two major stages:

```text
Electromagnetic Time Series
          │
          ▼
┌─────────────────────────────┐
│ Stage I: Sequence Learning  │
│                             │
│ TFC → AFL → GLAFF           │
└─────────────┬───────────────┘
              │
       Learned Temporal
        Representations
              │
              ▼
┌─────────────────────────────┐
│ Stage II: Anomaly Detection │
│                             │
│ Encoder → BiLSTM → Classifier│
└─────────────┬───────────────┘
              │
              ▼
       EQ / NEQ Probability
```

### Sequence Learning Stage

The upstream Sequence Learning stage learns general temporal representations from continuous electromagnetic observations.

It contains three major components:

```text
TFC   : Temporal Feature Compression
AFL   : Autoformer-Based Feature Learning
GLAFF : Guided Local Adaptive Feature Filtering
```

The model uses the previous seven days of electromagnetic observations to predict the subsequent seven days.

### Temporal Feature Compression

The **TFC module** reduces the dimensionality and temporal redundancy of long electromagnetic sequences.

It consists of:

```text
PCA
 ↓
1-D Convolution
 ↓
Hierarchical Temporal Features
```

Three sequential one-dimensional convolutional layers are used to progressively compress the temporal dimension while preserving important temporal patterns.

### Autoformer-Based Feature Learning

The **AFL module** adopts the Autoformer architecture to model long-range temporal dependencies.

The Auto-Correlation mechanism identifies periodic dependencies by discovering dominant temporal delays in the frequency domain.

The decomposition architecture separates electromagnetic sequences into:

```text
Periodic components
+
High-frequency components
```

allowing the model to characterize electromagnetic variations at different temporal scales.

### Guided Local Adaptive Feature Filtering

The **GLAFF module** combines:

```text
Local temporal prediction
+
Global timestamp-guided periodic prediction
```

through adaptive fusion weights.

The global prediction provides a stable periodic reference, while the local Autoformer prediction captures flexible temporal variations.

The resulting adaptive combination improves robustness against irregular fluctuations and temporal shifts.

### Anomaly Detection Stage

The downstream AD stage reuses the temporal representation learned by the upstream SL stage.

The encoded temporal features are passed through a two-layer Bidirectional LSTM:

```text
Temporal Representation
        ↓
     BiLSTM
        ↓
Temporal Average Pooling
        ↓
Fully Connected Layers
        ↓
     Softmax
        ↓
     EQ / NEQ
```

The classifier determines whether an input electromagnetic sequence contains temporal patterns associated with an earthquake occurring within the subsequent seven days.

---

## 📊 Experimental Results

### Overall Performance

The complete SL-AD framework achieves:

```text
ROC-AUC : 0.824
ACC     : 0.872
Recall  : 0.619
FNR     : 0.381
FPR     : 0.124
```

These results indicate that SL-AD can retain sensitivity to rare earthquake-related anomaly samples while controlling false alarms in a strongly imbalanced test environment.

### Dataset Statistics

| Dataset | Training Set | Test Set |
|---|---:|---:|
| Sequence Learning Dataset | 67,000 | 7,000 |
| Anomaly Detection — NEQ (0) | 212,827 | 7,173 |
| Anomaly Detection — EQ (1) | 680 | 100 |

The downstream anomaly-detection task is therefore characterized by severe class imbalance, motivating the use of upstream temporal representation learning before earthquake anomaly classification.

### Earthquake Case Analysis

The model is evaluated on eight earthquakes occurring during 2022–2023.

For each event, the predicted probabilities of associated monitoring stations are aggregated to analyze their temporal evolution.

The case studies show that:

- predicted anomaly probabilities generally increase before earthquake occurrence;
- cumulative anomaly counts begin to increase several days before the events;
- anomaly counts reach their highest levels around earthquake occurrence;
- anomaly activity decreases after the earthquake.

The cumulative analysis indicates a noticeable increase approximately **five days before the earthquakes**.

### Model Comparison

SL-AD is compared with representative time-series models including:

```text
TimesNet
Flowformer
Autoformer
FEDformer
SCINet
SSL-EF
```

The experiments demonstrate that direct end-to-end classification models are strongly affected by the extreme imbalance of the earthquake dataset.

In contrast, the two-stage learning strategy provides more transferable temporal representations and improves downstream precursor anomaly detection.

### Interpretability Analysis

#### Attention Visualization

The Autoformer encoder attention behavior is visualized to investigate temporal dependencies learned during sequence modeling.

The analysis shows that the model captures correlations between similar phases of periodic electromagnetic variations.

For downstream anomaly detection, gradient-based temporal importance analysis is used to compare:

```text
Quiet-period negative samples
Pre-earthquake positive samples
Post-earthquake negative samples
```

Pre-earthquake samples exhibit stronger and more concentrated temporal importance than quiet-period samples.

#### Encoding Correlation Analysis

The encoder representations of earthquake-related and non-earthquake samples are further analyzed using Pearson correlation.

Results show:

- high similarity among many earthquake-related feature encodings;
- substantially lower correlations between earthquake and non-earthquake samples;
- more diverse representations among normal electromagnetic samples.

These findings indicate that SL-AD learns a distinguishable feature space for earthquake-associated electromagnetic anomalies.

---

## 💻 Code

### 🚧 Repository Under Construction

The implementation will be released after publication.

### Planned Repository Structure

```text
SL-AD/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── configs/
│   ├── sequence_learning.yaml
│   ├── anomaly_detection.yaml
│   ├── evaluation.yaml
│   └── visualization.yaml
│
├── data_preprocessing/
│   ├── __init__.py
│   ├── load_aeta.py
│   ├── interpolation.py
│   ├── normalization.py
│   ├── frequency_selection.py
│   ├── sequence_segmentation.py
│   └── anomaly_labeling.py
│
├── models/
│   ├── __init__.py
│   ├── sl_ad.py
│   ├── tfc.py
│   ├── autoformer.py
│   ├── glaff.py
│   └── bilstm_classifier.py
│
├── training/
│   ├── train_sl.py
│   ├── train_ad.py
│   └── losses.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   ├── roc_analysis.py
│   └── earthquake_case_analysis.py
│
├── interpretability/
│   ├── attention_visualization.py
│   ├── temporal_importance.py
│   └── encoding_correlation.py
│
├── baselines/
│   ├── timesnet/
│   ├── autoformer/
│   ├── fedformer/
│   ├── scinet/
│   └── ssl_ef/
│
├── utils/
│   ├── io.py
│   ├── seed.py
│   ├── logger.py
│   └── plotting.py
│
├── scripts/
│   ├── preprocess.py
│   ├── train_sequence_learning.py
│   ├── train_anomaly_detection.py
│   ├── evaluate_sl_ad.py
│   └── reproduce_main_results.py
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── earthquake_catalog/
│       └── .gitkeep
│
├── checkpoints/
│   └── .gitkeep
│
└── results/
    ├── metrics/
    ├── predictions/
    ├── figures/
    └── interpretability/
```

---

## 📦 Data

The electromagnetic observations used in this study were obtained from the **AETA (Acoustic and Electromagnetics to Artificial Intelligence)** platform.

The earthquake catalog and associated seismic information were supported by the China Earthquake Networks Center and the National Earthquake Data Center.

Due to data ownership and redistribution requirements, the original raw observation files may not be redistributed directly through this repository.

Data preprocessing procedures and dataset construction scripts will be provided to facilitate reproducibility where permitted.

---

## ⚙️ Expected Dependencies

```txt
# Deep Learning
torch

# Scientific Computing
numpy
scipy
pandas
scikit-learn

# Visualization
matplotlib
seaborn

# Utilities
PyYAML
tqdm
```

> Exact package versions will be provided together with the released implementation.

---

## 🔁 Reproducibility

The experimental workflow consists of:

```text
1. AETA data preprocessing
2. Electromagnetic sequence construction
3. SL-stage sequence learning
4. Transfer of learned temporal representations
5. AD-stage fine-tuning
6. Earthquake precursor anomaly detection
7. ROC and classification evaluation
8. Earthquake case analysis
9. Attention and encoding-correlation analysis
```

The released code will include configuration files and random-seed settings required to reproduce the main experiments.

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{yu2026slad,
  title   = {Detection of Electromagnetic Temporal Anomalies Associated with Major Earthquakes in Western China via a Two-Stage Deep Learning Framework},
  author  = {Yu, Zining and Jing, Xilong and Zhang, Jiarui and Yang, Minglin and Dong, Shanzhi and Zheng, Haiyong},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

> Citation information will be updated after publication.

---

## 🙏 Acknowledgment

### Funding Support

This work was supported in part by:

- **National Natural Science Foundation of China**, Grant **42204005**
- **TaiShan Scholar Youth Expert Program of Shandong Province**, Grant **tsqn202306096**

### Data Support

The authors acknowledge data support from:

- **China Earthquake Networks Center**
- **National Earthquake Data Center**
- **AETA platform**

---

## 📧 Contact

For questions related to this project, please contact the corresponding author.

### Corresponding Author

**Haiyong Zheng, Ph.D.**  
Professor  
College of Electronic Engineering  
Ocean University of China  
Qingdao, China  
📧 Email: [zhenghaiyong@ouc.edu.cn](mailto:zhenghaiyong@ouc.edu.cn)

### Project Authors

**Zining Yu, Ph.D.**  
College of Electronic Engineering  
Ocean University of China  
Qingdao, China  
📧 Email: [yuzining@ouc.edu.cn](mailto:yuzining@ouc.edu.cn)

---

<div align="center">

**This repository is maintained for academic research and reproducibility of electromagnetic earthquake precursor analysis.**

</div>

