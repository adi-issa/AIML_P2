# VibeCode Pathology Portal

A local Streamlit portal that classifies a histopathology patch with both
fine-tuned models from the original notebook:

- ResNet50V2 (`resnet50v2_Tuning.keras`)
- DenseNet169 (`densenet169_Tuning.keras`)

For each model the portal reports the predicted class, predicted tissue group
(Lung or Breast), confidence percentage,
confidence level, recorded test-set accuracy, all class probabilities, and a
Grad-CAM overlay/heatmap/highlighted region.

## Start the portal on Windows

Double-click `run_portal.bat`.

The first run creates a private Python 3.11 virtual environment and installs
TensorFlow and Streamlit. It can take several minutes. When ready, the portal
opens in your browser. Later starts reuse the same environment.

Alternatively, open PowerShell in the cloned `VibeCode Portal` folder and run:

```powershell
.\setup_and_run.ps1
```

## Models and class order

The trained models are **not included in this Git repository**. Their sizes
(about 481 MB and 259 MB) exceed GitHub's normal file limit. Obtain the two
`.keras` files from the project owner and put them in an `Orginal Code` folder
beside the cloned `VibeCode Portal` folder:

```text
project/
├── Orginal Code/
│   ├── resnet50v2_Tuning.keras
│   └── densenet169_Tuning.keras
└── VibeCode Portal/
    └── app.py
```

Alternatively, set `VIBECODE_MODEL_DIR` to the folder containing both files
before starting the app. For example, in PowerShell:

```powershell
$env:VIBECODE_MODEL_DIR = "C:\path\to\models"
.\setup_and_run.ps1
```

The class order is preserved exactly from training:

1. Breast InSitu
2. Breast Benign
3. Breast Invasive
4. SCC
5. LUGAD

After uploading a patch, set the ROI threshold above **Analyse with both models**.

## Meaning of the numbers

- **Test accuracy** is a dataset-level metric recorded in the original notebook
  on 15,000 held-out patches (ResNet50V2: 90.94%; DenseNet169: 92.89%).
- **Confidence** is the model's softmax probability for the predicted class on
  the uploaded patch. It is not the same as accuracy.

## Important

This project is for research and educational use. A patch-level classifier and
Grad-CAM visualisation are not a medical diagnosis and must not be used alone
for clinical decisions.
