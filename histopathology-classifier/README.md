# Histopathology Cancer Classifier

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

Alternatively, open PowerShell in the cloned `histopathology-classifier` folder and run:

```powershell
.\setup_and_run.ps1
```

## Run online in GitHub Codespaces

1. [Create a Codespace for AIML_P2](https://codespaces.new/adi-issa/AIML_P2) on the `main` branch.
2. In its terminal, run:

   ```bash
   bash histopathology-classifier/run_online.sh
   ```

3. When port **8501** appears in the **Ports** tab, open its forwarded URL.

The script downloads both models from the private release, verifies their
SHA-256 checksums, installs dependencies, and starts Streamlit. The forwarded
port is private by default, so sign in to GitHub to open it. A Codespace is a
temporary development environment; the app stops when the Codespace stops.

## Deploy the public app on Streamlit Community Cloud

1. Connect Streamlit Community Cloud to `adi-issa/AIML_P2`.
2. In the deployment form, select branch `main` and main file path
   `histopathology-classifier/app.py`.
3. Deploy. If the repository is public, the app downloads the `models-v1`
   release assets without a token. If the repository is private, create a
   fine-grained GitHub personal access token for only `AIML_P2` with
   **Contents: Read-only** permission. Add it under **Advanced settings →
   Secrets** (or the deployed app's **Settings → Secrets**) as:

   ```toml
   GITHUB_TOKEN = "YOUR_FINE_GRAINED_TOKEN"
   ```

   Never add the real token to the GitHub repository or share it in a message.
4. Set **App settings → Sharing → This app is public and searchable** if it
   is not already public.

On first analysis, the app downloads the models into its server cache and checks
their SHA-256 hashes. This can take several minutes after a restart or when a
sleeping free app wakes. Subsequent analyses reuse the cached files and loaded
models while the server remains active. The free tier may hit its memory limit;
verify both models work with a sample patch before sharing the link.

## Models and class order

The trained models are **not in Git history** because each exceeds GitHub's
normal file limit. Download both individual files from the private
[models-v1 release](https://github.com/adi-issa/AIML_P2/releases/tag/models-v1):

- [resnet50v2_Tuning.keras](https://github.com/adi-issa/AIML_P2/releases/download/models-v1/resnet50v2_Tuning.keras)
- [densenet169_Tuning.keras](https://github.com/adi-issa/AIML_P2/releases/download/models-v1/densenet169_Tuning.keras)

Keep their filenames unchanged. Put them in an `Orginal Code` folder beside
the cloned `histopathology-classifier` folder:

```text
project/
├── Orginal Code/
│   ├── resnet50v2_Tuning.keras
│   └── densenet169_Tuning.keras
└── histopathology-classifier/
    └── app.py
```

Alternatively, set `HISTOPATHOLOGY_MODEL_DIR` to the folder containing both files
before starting the app. For example, in PowerShell:

```powershell
$env:HISTOPATHOLOGY_MODEL_DIR = "C:\path\to\models"
.\setup_and_run.ps1
```

The release asset SHA-256 checksums are:

```text
resnet50v2_Tuning.keras  515589e9330ef4ad8250724dd9574da437f0b131a5c1cb06576b9987139e14e3
densenet169_Tuning.keras  3a20ca66c183cd11b33a4f4eeb190c33a707fc088365ee37ca8d852792642f55
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
