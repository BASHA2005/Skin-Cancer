# Skin Lesion Classifier: Clinical Deployment Package

Clinical-grade dermoscopy classification system with dual-trunk **EfficientNetV2-M + EfficientNet-B4** backbones and a 5-node hierarchical decision tree, trained on the ISIC 2019 dataset across 7 diagnostic classes (80.99% held-out balanced test accuracy).

## Key Capabilities & Features
- **Preprocessing & Scanning Engine**: Multi-stage optical inspection pipeline featuring morphological hair detection (Blackhat filtering), DullRazor Telea inpainting, median noise filtering, and Shades-of-Gray illumination normalization.
- **Clinical ABCDE Morphometric Analysis**: Automated quantitative lesion analysis mirroring clinical dermatology screening criteria:
  - **A (Asymmetry)**: Horizontal and vertical axis symmetry score (0–100%).
  - **B (Border Irregularity)**: Compactness ratio ($P^2 / 4\pi A$) and contour roughness.
  - **C (Color Variegation)**: Dispersion across RGB channels and pigment variegation.
  - **D (Diameter & Coverage)**: Estimated physical diameter (mm) and image area coverage (%).
  - **E (Evolution / Composite Risk)**: Integrated morphologic risk score.
- **Interactive Decision Tree**: Visual traversal of the 5 hierarchical nodes (`root`, `mal`, `epi`, `ben`, `ker`).
- **Complete 7-Class Distribution**: Posterior probabilities with risk tiering (Malignant, Precancerous, Benign).
- **1-Click Clinical Demo Samples**: Instant testing with built-in Melanoma, Nevus, and Benign Keratosis dermoscopy images.
- **Patient Metadata Fusion**: Optional incorporation of age, biological sex, and anatomical body site.
- **Consultation Report Generator**: Instant downloadable summary report (`.txt`) for clinical documentation.

## Files
| File | Purpose |
|---|---|
| `app.py` | Modern Streamlit web application with scanning engine and clinical dashboard |
| `pipeline_tools.py` | Preprocessing stage extractor, lesion segmentation, ABCDE metrics, class atlas, report generator |
| `inference.py` | Model definition, dual-trunk backbone, hierarchical routing, and TTA inference |
| `config.json` | Model parameters, class names, hierarchy nodes, metadata encoding, and normalization config |
| `model.weights.h5` | Trained weights for EfficientNetV2-M + EfficientNet-B4 dual trunk |
| `sample_*.jpg` | Built-in clinical dermoscopy test images for quick one-click demo testing |
| `requirements.txt` | Dependencies for TensorFlow, Streamlit, OpenCV, NumPy, and Pandas |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running from Python (API / Service Integration)
```python
import cv2
import inference as inf
import pipeline_tools as pt

# 1. Load model weights
model = inf.load_model()

# 2. Preprocess image through DullRazor + Shades-of-Gray pipeline
bgr = cv2.imread("sample_melanoma.jpg")
pipeline_data = pt.run_pipeline_stages(bgr)

# 3. Compute ABCDE clinical features
abcde = pt.compute_abcde_features(pipeline_data['final_tensor'], pipeline_data['segmentation'])

# 4. Classify with optional patient metadata
patient_meta = inf.meta_vector(age=55, sex='male', site='anterior torso')
res = inf.classify(model, pipeline_data['final_tensor'], meta=patient_meta, routing='hard')

print(f"Predicted Diagnosis: {res['label']} ({res['label_name']})")
print(f"Malignancy Likelihood: {res['malignant_probability']:.1%}")
print(f"ABCDE Risk Rating: {abcde['abc_level']}")
```

## Clinical Notice
This system is an investigational research prototype trained on ISIC 2019. It is not an FDA/CE-cleared medical device and must not be used as a standalone diagnostic tool. All findings must be interpreted by a board-certified dermatologist.
