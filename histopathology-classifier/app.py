"""Streamlit portal for side-by-side pathology patch classification."""

from __future__ import annotations

import base64
import html
import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import (
    MODEL_SPECS,
    ModelSpec,
    heatmap_image,
    load_model,
    predict_with_gradcam,
)
from model_store import ensure_model_file


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DEFAULT_MODEL_DIR = Path(
    os.environ.get("HISTOPATHOLOGY_MODEL_DIR", str(PROJECT_DIR / "Orginal Code"))
).expanduser()
TISSUE_BY_CLASS = {
    "Breast InSitu": "Breast",
    "Breast Benign": "Breast",
    "Breast Invasive": "Breast",
    "SCC": "Lung",
    "LUGAD": "Lung",
}


st.set_page_config(
    page_title="Histopathology Cancer Classifier",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .app-version {
        position: fixed; right: 20px; bottom: 20px; z-index: 999;
        padding: 8px 15px; border-radius: 999px;
        background: #140f50; color: #ffffff; border: 1px solid #a7b8ff;
        font: 700 14px/1.2 system-ui, sans-serif;
        box-shadow: 0 3px 12px rgba(20, 15, 80, .18);
    }
    </style>
    <span class="app-version" title="App V2 — updated Grad-CAM calculation"
          aria-label="App version 2 with updated Grad-CAM">V2</span>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
<style>
    @font-face {
        font-family:'National 2 Condensed'; font-style:normal; font-weight:500; font-display:swap;
        src:url('https://adelaide.edu.au/etc.clientlibs/adelaideuniversity/clientlibs/clientlib-common/resources/fonts/National/national-2-condensed-medium.woff2') format('woff2');
    }
    @font-face {
        font-family:'National 2 Condensed'; font-style:normal; font-weight:700; font-display:swap;
        src:url('https://adelaide.edu.au/etc.clientlibs/adelaideuniversity/clientlibs/clientlib-common/resources/fonts/National/national-2-condensed-bold.woff2') format('woff2');
    }
    @font-face {
        font-family:'Roboto Serif'; font-style:normal; font-weight:400; font-display:swap;
        src:url('https://adelaide.edu.au/etc.clientlibs/adelaideuniversity/clientlibs/clientlib-common/resources/fonts/Roboto/RobotoSerif-Regular.ttf') format('truetype');
    }
    @font-face {
        font-family:'Roboto Serif'; font-style:normal; font-weight:600; font-display:swap;
        src:url('https://adelaide.edu.au/etc.clientlibs/adelaideuniversity/clientlibs/clientlib-common/resources/fonts/Roboto/RobotoSerif-SemiBold.ttf') format('truetype');
    }
    :root {
        --ink: #140f50;
        --muted: #495266;
        --teal: #1448ff;
        --teal-dark: #0c2b99;
        --mint: #e6f4f0;
        --cream: #fafafb;
        --line: #d4dade;
        --coral: #e56b58;
    }
    .stApp { background: #fafafb; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1420px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    .portal-kicker { color: var(--teal); font-size: .78rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    .portal-title { color: var(--ink); font-size: clamp(2.2rem, 5vw, 4.2rem); line-height: .98; font-weight: 800; max-width: 950px; margin: .55rem 0 1rem; letter-spacing: -.055em; }
    .portal-subtitle { color: var(--muted); font-size: 1.05rem; max-width: 820px; line-height: 1.65; margin-bottom: 1.4rem; }
    .model-chip { display: inline-flex; align-items: center; gap: .45rem; padding: .42rem .75rem; margin: 0 .45rem .45rem 0; border: 1px solid #c9dfd8; border-radius: 999px; background: rgba(255,255,255,.78); color: var(--teal-dark); font-weight: 700; font-size: .86rem; }
    .model-chip::before { content: ''; width: .48rem; height: .48rem; border-radius: 50%; background: #2ca58d; box-shadow: 0 0 0 4px rgba(44,165,141,.12); }
    .upload-note { color: var(--muted); font-size: .88rem; line-height: 1.5; }
    .result-card { background: rgba(255,255,255,.86); border: 1px solid var(--line); border-radius: 20px; padding: 1.25rem 1.35rem; box-shadow: 0 10px 35px rgba(26,69,64,.07); margin-bottom: 1rem; }
    .result-model { color: var(--teal); font-size: .78rem; font-weight: 800; text-transform: uppercase; letter-spacing: .12em; }
    .result-class { color: var(--ink); font-size: 1.75rem; font-weight: 800; margin: .2rem 0 .8rem; }
    .metric-grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: .65rem; }
    .metric { background: var(--cream); border: 1px solid #ece9df; border-radius: 13px; padding: .72rem .78rem; }
    .metric-label { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
    .metric-value { color: var(--ink); font-size: 1.08rem; font-weight: 800; margin-top: .16rem; }
    .prob-row { display: grid; grid-template-columns: 118px 1fr 54px; align-items: center; gap: .55rem; margin: .5rem 0; font-size: .78rem; color: var(--muted); }
    .prob-track { height: 8px; border-radius: 999px; background: #e9efed; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #1448ff, #1448ff); }
    .agreement { border: 1px solid #bfe0d7; background: #e8f6f1; color: #145f55; border-radius: 14px; padding: .9rem 1rem; font-weight: 700; margin: .5rem 0 1rem; }
    .disagreement { border-color: #f2c5bc; background: #fff0ec; color: #983d2f; }
    .fine-print { color: var(--muted); font-size: .76rem; line-height: 1.5; }
    [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.82); border: 1.5px dashed #7a88ab; border-radius: 16px; }
    .stButton > button { border-radius: 12px; min-height: 3rem; font-weight: 800; border: 0; background: var(--teal); color: white; }
    .stButton > button:hover { background: var(--teal-dark); color: white; }
    @media (max-width: 760px) { .metric-grid { grid-template-columns: 1fr; } .prob-row { grid-template-columns: 104px 1fr 48px; } }
.project-heading {margin: 1rem 0; padding: clamp(20px,4vw,48px); background:#140f50; border-left: 6px solid #1448ff;}
    .project-heading {position:relative; isolation:isolate; overflow:hidden;
        border:0; border-radius:4px 36px 4px 4px;
        background:linear-gradient(104deg,#140f50 30%,#29236a 70%,#41318a);}
    .slide-art {position:absolute; right:-5%; top:-55%; height:210%; aspect-ratio:1;
        z-index:-1; pointer-events:none; opacity:.88;
        background:repeating-radial-gradient(ellipse at 70% 45%, transparent 0 13px,
          #e493c4 14px 16px, #9e81ec 17px 20px, transparent 21px 36px);
        border-radius:42% 58% 64% 36%; transform:rotate(-22deg);
        mask-image:linear-gradient(100deg,transparent 12%,#000 62%);}
    .project-heading h1 {max-width:76%; text-shadow:0 2px 8px #140f50;}
    .specimen {--field:#293bcd; --type:#fff; --accent:#d3ff75;
        background:var(--field); color:var(--type); container-type:inline-size;
        overflow:hidden; border-radius:3px 32px 3px 3px; margin-bottom:12px;}
    .specimen.densenet169 {--field:#ffd1b8; --type:#421439; --accent:#932958;}
    .specimen-body {padding:clamp(18px,5cqi,34px);}
    .plate-id,.plate-caption {font:600 .78rem/1.4 Consolas,monospace; letter-spacing:.045em;}
    .plate-id {display:flex; align-items:center; gap:9px;}
    .plate-id::before {content:''; width:21px; height:7px; background:var(--accent);}
    .plate-face {display:grid; grid-template-columns:minmax(0,1fr) 33%;
        align-items:center; gap:5%; min-height:165px;}
    .plate-face h3 {color:inherit; padding:.25em 0; margin:0; font:700 clamp(2rem,12cqi,5rem)/.94 Arial,sans-serif;
        letter-spacing:-.06em; overflow-wrap:anywhere;}
    .plate-lens {margin:0; text-align:center;}
    .plate-lens img {display:block; width:100%; aspect-ratio:1; border-radius:50%;
        border:4px solid var(--field); outline:2px solid var(--accent);}
    .plate-lens figcaption {font: .72rem/1.3 Consolas,monospace; margin-top:8px;}
    .plate-score {display:flex; flex-wrap:wrap; align-items:baseline; gap:5px 10px;}
    .plate-score strong {font:600 clamp(1.55rem,7cqi,2.8rem)/1 Arial,sans-serif; letter-spacing:-.055em;}
    .plate-meter {height:5px; margin:12px 0 0; background:rgba(128,128,128,.4);}
    .plate-meter span {display:block; height:100%; background:var(--accent);}
    .plate-foot {background:#15123d; color:#fff; padding:14px 5%; display:flex;
        align-items:center; justify-content:space-between; flex-wrap:wrap; gap:6px 16px; font-size:.82rem;}
    .densenet169 .plate-foot {background:#421439;}
    @media (prefers-reduced-motion:no-preference) {
        .plate-meter span {transform-origin:left; animation:show-score .65s ease-out both;}
        .plate-lens {transition:transform .25s ease;}
        .specimen:hover .plate-lens {transform:translateY(-3px);}
    }
    [data-testid="stFileUploaderDropzone"] {background:#f1f4ff; border-color:#637cc7;}
    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploaderDropzone"]:focus-within {background:#e7edff; border-color:#1448ff;}
    @media (prefers-reduced-motion:no-preference) {
        .slide-art {animation:focus-slide 1.5s ease-out both;}
        @keyframes focus-slide {from {transform:translateX(12%) rotate(-12deg); opacity:.25;}}
        .prob-fill {transform-origin:left; animation:show-score .55s ease-out both;}
        @keyframes show-score {from {transform:scaleX(0);}}
        .stButton > button {transition:transform .16s ease;}
        .stButton > button:hover {transform:translateY(-2px);}
    }
    @media (max-width:650px) {
        .project-heading h1 {max-width:100%;}
        .slide-art {opacity:.18; right:-45%;}
    }
    .project-heading h1 {color:white; font: 600 clamp(1.85rem,4vw,3.5rem)/1.08 Georgia,serif; padding:0; overflow-wrap:break-word;}
    .project-heading em {color:#dce4ff; font-weight:400;}
    /* Specimen studio: the uploaded tissue, model views and explanations share one visual language. */
    .stApp {
        background:
            radial-gradient(circle at 95% 8%, #efcff2 0, transparent 24rem),
            radial-gradient(circle at 0 45%, #d7e6ff 0, transparent 27rem),
            #f7f8fc;
        font-family:'Roboto Serif', Georgia, serif;
    }
    [data-testid="stMain"], [data-testid="stMain"] button,
    [data-testid="stMain"] input, [data-testid="stMain"] label {
        font-family:'Roboto Serif', Georgia, serif;
    }
    .block-container { max-width: 1320px; padding-top: 1.25rem; }
    .block-container > div > div > [data-testid="stImage"] { margin-bottom: 1.25rem; }
    .project-heading {
        display:flex; flex-direction:column; justify-content:space-between;
        min-height:clamp(340px,45vw,540px); padding:clamp(26px,5vw,66px);
        margin:0 0 1.5rem; border-radius:30px 30px 90px 30px;
        background:linear-gradient(120deg,#12113f 0%,#252184 54%,#8f3fbd 100%);
        box-shadow:0 28px 60px -38px #161363;
    }
    .project-heading::after {content:none;}
    .hero-micrograph {
        position:absolute; right:-13%; bottom:-34%; z-index:-1;
        width:min(48%,500px); aspect-ratio:1; overflow:hidden; border-radius:50%;
        border:12px solid rgba(247,172,219,.8);
        box-shadow:0 0 0 18px rgba(251,175,220,.21),0 0 0 42px rgba(251,175,220,.1),
                   0 24px 50px rgba(17,11,57,.35);
        transform:rotate(-12deg); pointer-events:none;
    }
    .hero-micrograph img {width:100%; height:100%; object-fit:cover; filter:saturate(1.25) contrast(1.12);}
    .hero-micrograph::after {content:""; position:absolute; inset:0;
        background:linear-gradient(130deg,rgba(58,24,119,.15),rgba(50,24,116,.52));}
    .slide-art {display:none;}
    .hero-eyebrow {
        color:#c9e4ff; font:700 1rem/1.2 'National 2 Condensed','Arial Narrow',sans-serif;
        letter-spacing:.08em; text-transform:uppercase;
    }
    .project-heading h1 {
        max-width:80%; margin:auto 0; color:#fff;
        font:700 clamp(3.1rem,6.8vw,6.8rem)/.88 'National 2 Condensed','Arial Narrow',sans-serif;
        letter-spacing:-.012em; text-shadow:none;
    }
    .project-heading h1 em {color:#ffd1e7; font-family:'National 2 Condensed','Arial Narrow',sans-serif; font-style:normal; font-weight:500;}
    .hero-bottom {display:flex; justify-content:space-between; align-items:end; gap:1rem;
        color:#fff; font-size:.91rem; max-width:75%;}
    .hero-bottom p {max-width:40rem; margin:0; line-height:1.55;}
    .hero-model-line {display:flex; align-items:center; gap:.5rem; flex:none;
        white-space:nowrap; font:700 1.1rem 'National 2 Condensed','Arial Narrow',sans-serif;
        letter-spacing:.025em; text-shadow:0 2px 8px #18134f,0 1px 2px #18134f;}
    .hero-model-line b {color:#d4f96b; font-weight:700;}
    .hero-model-line b:last-child {color:#ffd1e7;}
    .hero-model-line i {color:#fff; font-style:normal; opacity:.85;}
    [data-testid="stMain"] h3 {font-family:'National 2 Condensed','Arial Narrow',sans-serif; font-weight:700; font-size:clamp(2rem,3vw,2.7rem); letter-spacing:-.01em;}
    [data-testid="stMain"] button {font-family:'National 2 Condensed','Arial Narrow',sans-serif; font-size:1.1rem; letter-spacing:.01em;}
    [data-testid="stFileUploaderDropzone"] {
        min-height:155px; border:2px dashed #455fec; border-radius:24px;
        background:linear-gradient(110deg,#e7edff,#f9ecf7);
        transition:background .25s ease, border-color .25s ease, transform .25s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploaderDropzone"]:focus-within {
        background:linear-gradient(110deg,#dce6ff,#f6deef);
        border-color:#8d3fad; transform:translateY(-2px);
    }
    .stButton > button[kind="primary"] {
        border-radius:100px; min-height:3.6rem; padding-inline:1.8rem;
        background:#2447e8; box-shadow:0 10px 24px -12px #2447e8;
    }
    .stButton > button[kind="primary"]:hover {background:#132eae;}
    .specimen {box-shadow:0 24px 42px -28px #171349; border-radius:26px 26px 56px 26px;}
    .specimen.resnet50v2 {--field:#2040c7; --accent:#dcff73;}
    .specimen.densenet169 {--field:#ffd0e1; --type:#421439; --accent:#aa2f70;}
    .plate-face {min-height:195px;}
    .plate-id,.plate-caption {font-family:'National 2 Condensed','Arial Narrow',sans-serif; letter-spacing:.025em;}
    .plate-id {font-size:1.2rem; font-weight:700;}
    .plate-caption {font-size:1.05rem; font-weight:500;}
    .plate-class {display:flex; align-items:center; flex-wrap:wrap; gap:.55rem .8rem;}
    .plate-organ {display:inline-block; padding:.3rem .6rem; border-radius:999px;
        background:var(--accent); color:var(--field);
        font:700 1rem/1 'National 2 Condensed','Arial Narrow',sans-serif;
        letter-spacing:.045em; text-transform:uppercase;}
    .plate-face h3 {font-family:'National 2 Condensed','Arial Narrow',sans-serif; font-weight:700; font-size:clamp(3.5rem,13cqi,6rem); letter-spacing:-.02em;}
    .plate-lens figcaption {font-family:'Roboto Serif',Georgia,serif; font-size:.82rem;}
    .plate-score strong {font-family:'National 2 Condensed','Arial Narrow',sans-serif; font-weight:700;}
    .plate-lens img {box-shadow:0 0 0 7px color-mix(in srgb,var(--accent),transparent 75%);}
    .plate-score strong {font-size:clamp(2rem,8cqi,3.5rem);}
    .agreement {border:0; background:#dcf6e9; color:#124c38; border-radius:16px;}
    .agreement.disagreement {background:#ffebeb; color:#822b4a;}
    .prob-row {font-size:.9rem; margin:.65rem 0;}
    .prob-fill {background:linear-gradient(90deg,#304af4,#b744a0);}
    @media (max-width:760px) {
        .project-heading {min-height:390px; border-radius:22px 22px 60px 22px;}
        .project-heading h1 {max-width:100%; font-size:clamp(2.7rem,11vw,4.3rem);}
        .hero-micrograph {width:84%; opacity:.28; right:-39%; bottom:-9%;}
        .hero-bottom {max-width:100%; align-items:start; flex-direction:column;}
        .plate-face {min-height:155px;}
    }
    @media (prefers-reduced-motion:no-preference) {
        .hero-micrograph {animation:hero-reveal 1.1s ease-out both;}
        @keyframes hero-reveal {from {opacity:0; transform:translate(3%,5%) scale(.92);}}
        .specimen {animation:card-arrive .45s ease-out both;}
        @keyframes card-arrive {from {opacity:0; transform:translateY(12px);}}
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_model(spec: ModelSpec, path: str, modified_ns: int):
    # modified_ns is intentionally part of the cache key so replacing a model
    # at the same path does not reuse stale weights.
    del modified_ns
    return load_model(spec, path)


def probability_bars(probabilities: dict[str, float]) -> str:
    rows = []
    for label, probability in sorted(
        probabilities.items(), key=lambda item: item[1], reverse=True
    ):
        safe_label = html.escape(label)
        percent = probability * 100
        rows.append(
            f'<div class="prob-row"><span>{safe_label}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{percent:.2f}%"></div></div>'
            f'<strong>{percent:.1f}%</strong></div>'
        )
    return "".join(rows)


def predicted_tissue(class_name: str) -> str:
    return TISSUE_BY_CLASS.get(class_name, "Unknown")


def result_summary(result, spec: ModelSpec) -> str:
    buffer = io.BytesIO()
    result.overlay.save(buffer, format="PNG")
    overlay = base64.b64encode(buffer.getvalue()).decode("ascii")
    activation_label = "Grad-CAM" if getattr(result, "has_activation", True) else "No positive activation"
    return f"""
    <article class="specimen {spec.architecture}">
      <div class="specimen-body">
        <div class="plate-id">{html.escape(result.model_name)}</div>
        <div class="plate-face">
          <div><span class="plate-caption">Predicted class</span><div class="plate-class"><h3>{html.escape(result.predicted_class)}</h3><span class="plate-organ" aria-label="Predicted tissue: {predicted_tissue(result.predicted_class)}">{predicted_tissue(result.predicted_class)}</span></div></div>
          <figure class="plate-lens"><img src="data:image/png;base64,{overlay}" alt="{html.escape(result.model_name)} {activation_label}"><figcaption>{activation_label}</figcaption></figure>
        </div>
        <div class="plate-score"><strong>{result.confidence:.2%}</strong><span>{result.confidence_level} confidence</span></div>
        <div class="plate-meter" aria-hidden="true"><span style="width:{result.confidence:.2%}"></span></div>
      </div>
      <div class="plate-foot"><span><b>{spec.test_accuracy:.2%}</b> test accuracy · 15,000 patches</span><span>{result.inference_seconds:.2f} s incl. Grad-CAM</span></div>
    </article>
    """


hero_tissue = base64.b64encode((APP_DIR / "hero-tissue.jpg").read_bytes()).decode("ascii")
st.image(str(APP_DIR / "adelaide-university.svg"), width=210)
st.markdown(f"""
<section class="project-heading">
  <div class="hero-micrograph" aria-hidden="true"><img src="data:image/jpeg;base64,{hero_tissue}" alt=""></div>
  <div class="hero-eyebrow">Histopathology analysis / Two-model comparison</div>
  <h1>Histopathology Cancer Classification<br><em>with Transfer Learning</em></h1>
  <div class="hero-bottom"><p>Upload a tissue patch. Compare two trained networks, examine class confidence and explore the regions behind each prediction.</p><div class="hero-model-line"><b>RESNET50V2</b><i>+</i><b>DENSENET169</b></div></div>
</section>
""", unsafe_allow_html=True)

def clear_results():
    for key in ("analysis_results", "analysis_errors"):
        st.session_state.pop(key, None)


st.subheader("Upload a patch")
image_upload = st.file_uploader(
    "Drop a JPG, PNG, BMP, or TIFF image here",
    type=("jpg", "jpeg", "png", "bmp", "tif", "tiff"),
    on_change=clear_results,
    label_visibility="collapsed",
)

image = None
if image_upload is not None:
    try:
        image = Image.open(image_upload).convert("RGB")
        preview_col, detail_col = st.columns((1, 2), gap="large")
        with preview_col:
            st.image(image, caption=image_upload.name, use_container_width=True)
        with detail_col:
            st.markdown("### Ready for analysis")
            st.markdown(
                f"**Image:** {image.width} × {image.height} px  \n"
                "The models will receive a resized 224 × 224 RGB copy. "
                "The original upload is not modified."
            )
            roi_threshold = st.slider(
                "ROI threshold", 0.30, 0.95, 0.50, step=0.05,
                help="Higher values highlight only the strongest Grad-CAM regions.",
                on_change=clear_results,
            )
            run_analysis = st.button(
                "Analyse with both models",
                type="primary",
                use_container_width=True,
            )
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"The uploaded file could not be read as an image: {exc}")
        run_analysis = False
else:
    st.info("Upload one histopathology patch to begin.")
    run_analysis = False


if run_analysis and image is not None:
    results = {}
    errors = {}
    progress = st.progress(0, text="Preparing models…")

    for index, spec in enumerate(MODEL_SPECS):
        progress.progress(
            index / len(MODEL_SPECS),
            text=f"Preparing {spec.display_name}…",
        )
        try:
            try:
                github_token = os.environ.get("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
            except FileNotFoundError:
                github_token = os.environ.get("GITHUB_TOKEN")
            model_path = ensure_model_file(spec.filename, DEFAULT_MODEL_DIR, github_token)
            loaded = cached_model(
                spec,
                str(model_path.resolve()),
                model_path.stat().st_mtime_ns,
            )
            progress.progress(
                (index + 0.5) / len(MODEL_SPECS),
                text=f"Running {spec.display_name} and Grad-CAM…",
            )
            results[spec.architecture] = predict_with_gradcam(
                loaded,
                image,
                roi_threshold=roi_threshold,
            )
        except Exception as exc:  # Keep the other model usable if one fails.
            errors[spec.architecture] = str(exc)

    progress.progress(1.0, text="Analysis complete")
    progress.empty()
    st.session_state["analysis_results"] = results
    st.session_state["analysis_errors"] = errors


results = st.session_state.get("analysis_results", {})
errors = st.session_state.get("analysis_errors", {})


if image is not None and (results or errors):
    st.divider()
    st.subheader("Model comparison")
    if len(results) == 2:
        predictions = [result.predicted_class for result in results.values()]
        tissues = [predicted_tissue(class_name) for class_name in predictions]
        agrees = len(set(predictions)) == 1
        if agrees:
            st.markdown(
                f'<div class="agreement">Both models predict {html.escape(tissues[0])} tissue · {html.escape(predictions[0])}.</div>',
                unsafe_allow_html=True,
            )
        else:
            tissue_note = (
                f'Both models predict {html.escape(tissues[0])} tissue, but disagree on the class.'
                if len(set(tissues)) == 1
                else f'The models predict different tissue groups: {html.escape(tissues[0])} and {html.escape(tissues[1])}.'
            )
            st.markdown(
                f'<div class="agreement disagreement">{tissue_note} Review both '
                'confidence distributions and Grad-CAM regions before interpreting the patch.</div>',
                unsafe_allow_html=True,
            )

    columns = st.columns(2, gap="large")
    for column, spec in zip(columns, MODEL_SPECS):
        with column:
            result = results.get(spec.architecture)
            if result is None:
                st.error(
                    f"{spec.display_name} could not run: "
                    f"{errors.get(spec.architecture, 'Unknown error')}"
                )
                continue

            st.markdown(
                result_summary(result, spec),
                unsafe_allow_html=True,
            )
            has_activation = getattr(result, "has_activation", True)
            if not has_activation:
                st.info(
                    f"{spec.display_name} has no positive Grad-CAM activation for "
                    "this prediction. The image above is the original patch, "
                    "and the ROI contains no highlighted region."
                )
            tabs = st.tabs(("Grad-CAM overlay", "Heatmap", "Highlighted ROI"))
            with tabs[0]:
                st.image(
                    result.overlay,
                    caption=(f"{spec.display_name} Grad-CAM overlay" if has_activation
                             else "Original patch — no positive activation"),
                    use_container_width=True,
                )
            with tabs[1]:
                if has_activation:
                    st.image(
                        heatmap_image(result.heatmap),
                        caption="Blue = lower influence · red = higher influence",
                        use_container_width=True,
                    )
                else:
                    st.info("No positive Grad-CAM activation is available for this prediction.")
            with tabs[2]:
                st.image(
                    result.roi,
                    caption=f"Regions at or above {roi_threshold:.0%} activation",
                    use_container_width=True,
                )
            st.markdown("#### Class probabilities")
            st.markdown(
                probability_bars(result.probabilities),
                unsafe_allow_html=True,
            )

    st.warning(
        "Research use only. These outputs are model predictions for image patches and "
        "must not be used as a standalone diagnosis or treatment decision."
    )
