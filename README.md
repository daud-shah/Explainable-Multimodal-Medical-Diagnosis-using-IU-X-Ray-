# 🩺 Multimodal Chest X-Ray Diagnosis with Explainable AI

**Detect 14 diseases using X-ray images + radiology reports**  
**ViT + ClinicalBERT Fusion | Grad-CAM Explainability | AUC = 0.862**

---

## 🌍 Overview

This project presents a **multimodal deep learning pipeline** for automatic chest X-ray diagnosis using both **images** and **radiology reports**.  
It integrates **Vision Transformers (ViT-Base)** for visual features and **Bio_ClinicalBERT** for textual understanding, connected through a **cross-attention fusion mechanism**.

**Features:**
- ViT + ClinicalBERT multimodal fusion  
- Grad-CAM explainability on transformer patches  
- Complete training and evaluation pipeline  
- Ablation study: Vision-only vs Text-only vs Fusion  
- Ready-to-use inference with overlay heatmaps  
- Structured for deployment and reproducibility  

---

## 📈 Performance

| Model Type      | Mean AUC |
|------------------|----------|
| Vision-only      | 0.590    |
| Text-only        | 0.859    |
| **Multimodal**   | **0.862**|

> **+27.2% improvement over vision-only** | **+0.3% over text-only**

---

## 🧠 Dataset

- **Source:** [IU X-Ray (Indiana University)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)  
- **Total Samples:** 3,851 image–report pairs  
- **Labels:** 14 chest pathologies (multi-label)  
- **Split:** Train / Validation / Test (stratified)

---

## ⚙️ Model Architecture

```

[ X-ray Image ] → ViT-Base → [768-dim]
↓
[Radiology Report] → ClinicalBERT → [768-dim]
↓
Cross-Attention Fusion
↓
MLP Classifier → 14 sigmoid outputs
↓
Grad-CAM on ViT patches

```

---

## 📁 Directory Structure

```

├── model_week3.py              # Model definition
├── best_model.pt               # Trained weights
├── config.pkl                  # Class names & settings
├── phase3_outputs/             # Grad-CAM visualizations
├── phase4_evaluation/          # Metrics and ablation
├── thesis_figures/             # ROC, PR, and confusion plots
├── custom_test/                # Inference demo outputs
├── test_custom.py              # Inference script
└── README.md

````

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/multimodal-chest-xray-ai.git
cd multimodal-chest-xray-ai
````

### 2. Install Dependencies

```bash
pip install torch torchvision transformers opencv-python pillow pandas scikit-learn matplotlib
```

### 3. Download Dataset

```bash
kaggle datasets download -d raddar/chest-xrays-indiana-university
unzip chest-xrays-indiana-university.zip -d data/
```

### 4. Run Inference

```bash
python test_custom.py \
  --image_path "data/images/test/00001234_001.png" \
  --report "Frontal chest X-ray shows mild cardiomegaly. Lungs clear."
```

**Outputs:**

* `overlay.png` → Grad-CAM visualization
* `report.txt` → Prediction with explanation

---

## 🔍 Explainability (Grad-CAM)

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

target_layer = model.vision_encoder.vit.encoder.layer[-1].layernorm_after
gradcam = GradCAM(model=wrapper, target_layers=[target_layer])
```

Generates clinically aligned heatmaps (e.g., highlights cardiac region for cardiomegaly).

---

## 📊 Results & Figures

| Figure                              | Description                         |
| ----------------------------------- | ----------------------------------- |
| `thesis_figures/roc_curves.png`     | ROC Curves for all classes          |
| `thesis_figures/pr_curves.png`      | Precision–Recall plots              |
| `thesis_figures/ablation_study.png` | Vision vs Text vs Fusion comparison |
| `phase3_outputs/overlay_*.png`      | Grad-CAM examples                   |

---

## 🔁 Reproducibility

1. Load class configuration with `config.pkl`
2. Load weights using:

   ```python
   model.load_state_dict(torch.load('best_model.pt', map_location='cuda'))
   ```
3. Preprocessing:

   * Image size: 224×224
   * Normalization: ImageNet mean/std
   * Text max length: 128 tokens

---

## 📚 Citation

```bibtex
@misc{yourname2025multimodal,
  author = {Daud Shah},
  title = {Multimodal Fusion of Chest X-rays and Reports with Explainable Grad-CAM},
  year = {2025},
  publisher = {daud-shah},
  howpublished = {\url{https://github.com/daud-shah/Explainable Multimodal Medical Diagnosis using IU X-Ray}}
}
```

---

## ⚖️ License

This project is released under the **MIT License**.

---



