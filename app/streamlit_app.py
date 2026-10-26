"""
AI Chest X-Ray Research Platform — Streamlit Research Dashboard.
Communicates strictly via REST API with the FastAPI backend.
"""
import os
import io
import json
import base64
from pathlib import Path
import streamlit as st
import requests
from PIL import Image

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Chest X-Ray Research Platform",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 5px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        color: #92400e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-card p {
        margin: 0;
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pred-badge-pneumonia {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #f87171;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }
    .pred-badge-normal {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #4ade80;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Configuration & Backend Connection Helper
# ---------------------------------------------------------
DEFAULT_BACKEND_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/x-ray.png", width=64)
    st.title("Platform Controls")

    backend_url = st.text_input(
        "FastAPI Backend URL",
        value=DEFAULT_BACKEND_URL,
        help="Address of the running FastAPI inference server",
    ).rstrip("/")

    st.markdown("---")
    navigation = st.radio(
        "Navigation",
        [
            "🔬 Diagnostic Inference & Grad-CAM",
            "📊 Model Evaluation & Metrics",
            "🗂️ Benchmark Dataset Explorer",
            "🏗️ Architecture & Model Card",
        ],
    )

    st.markdown("---")
    st.markdown("### System Status")


@st.cache_data(ttl=5)
def check_backend_status(url: str):
    """Check health and readiness of the FastAPI server."""
    try:
        health_resp = requests.get(f"{url}/health", timeout=2.5)
        ready_resp = requests.get(f"{url}/ready", timeout=2.5)
        if health_resp.status_code == 200 and ready_resp.status_code == 200:
            return True, ready_resp.json()
        return False, {"message": f"Server status code: {health_resp.status_code}"}
    except Exception as e:
        return False, {"message": str(e)}


is_online, ready_data = check_backend_status(backend_url)

with st.sidebar:
    if is_online:
        st.success(f"Backend: Online ({ready_data.get('device', 'cpu').upper()})")
        if ready_data.get("ready"):
            st.info("Model Engine: Ready")
        else:
            st.warning("Model Engine: Not Loaded")
    else:
        st.error("Backend: Disconnected")
        st.caption("Start backend with: `python -m uvicorn app.api.main:app --port 8000`")

    st.markdown("---")
    st.caption("AI Chest X-Ray Research Platform v1.0.0")

# ---------------------------------------------------------
# Top Header & Medical Disclaimer Banner
# ---------------------------------------------------------
st.markdown('<div class="main-header">AI Chest X-Ray Research Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Interpretability and Diagnostic Exploration on PneumoniaMNIST</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Clinical Research Disclaimer:</strong> Educational research demo only.
        Not intended for clinical use, patient diagnosis, treatment planning, or medical decision-making.
        Always consult certified radiologists and physicians for medical evaluations.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Tab 1: Diagnostic Inference & Grad-CAM
# ---------------------------------------------------------
if navigation == "🔬 Diagnostic Inference & Grad-CAM":
    st.subheader("Interactive Radiograph Analysis")
    st.write(
        "Select an official held-out test sample or upload a grayscale chest radiograph to evaluate the CNN prediction and Grad-CAM visual attention."
    )

    if not is_online:
        st.error(
            f"Cannot connect to the FastAPI backend at `{backend_url}`. "
            "Please ensure the backend server is running."
        )
        st.stop()

    input_mode = st.radio("Choose Input Source:", ["Benchmark Test Sample", "Custom Image Upload"], horizontal=True)

    selected_raw_image_bytes = None
    sample_index_selected = None

    if input_mode == "Benchmark Test Sample":
        col_sel, col_desc = st.columns([1, 2])
        try:
            samples_resp = requests.get(f"{backend_url}/samples", timeout=5)
            if samples_resp.status_code == 200:
                samples_data = samples_resp.json().get("samples", [])
                options = [
                    f"Sample #{s['index']} — Reference: {s['true_label_name']}"
                    for s in samples_data
                ]
                with col_sel:
                    selected_option = st.selectbox("Select Test Image:", options)
                    sample_index_selected = int(selected_option.split("#")[1].split(" ")[0])
                    current_sample = next(s for s in samples_data if s["index"] == sample_index_selected)
                with col_desc:
                    st.caption(f"Ground Truth Reference Label: **{current_sample['true_label_name']}**")
                    st.image(current_sample["preview_base64"], width=128, caption=f"28x28 Raw Radiograph (#{sample_index_selected})")
            else:
                st.warning("Could not fetch pre-loaded samples from backend.")
        except Exception as e:
            st.error(f"Error loading samples: {e}")

    else:
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray Image (PNG, JPEG, WebP):",
            type=["png", "jpg", "jpeg", "webp"],
            help="Grayscale or RGB images will be converted to 28x28 single-channel input",
        )
        if uploaded_file is not None:
            selected_raw_image_bytes = uploaded_file.getvalue()
            st.image(uploaded_file, width=150, caption="Uploaded Radiograph Preview")

    st.markdown("---")
    run_btn = st.button("🚀 Analyze Radiograph with Grad-CAM", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Executing neural inference and computing gradient activation maps..."):
            try:
                if input_mode == "Benchmark Test Sample" and sample_index_selected is not None:
                    resp = requests.post(
                        f"{backend_url}/predict/sample/{sample_index_selected}?explain=true",
                        timeout=10,
                    )
                elif selected_raw_image_bytes:
                    files = {"file": ("upload.png", selected_raw_image_bytes, "image/png")}
                    resp = requests.post(f"{backend_url}/predict?explain=true", files=files, timeout=10)
                else:
                    st.warning("Please provide or select an image first.")
                    resp = None

                if resp and resp.status_code == 200:
                    data = resp.json()
                    pred = data["prediction"]
                    expl = data.get("explanation", {})

                    st.success(f"Analysis completed in {data['inference_time_ms']:.1f} ms")

                    # Display results in 3 columns
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("#### 1. Input Radiograph")
                        st.image(data["image_preview_base64"], use_container_width=True)
                        st.caption("Standardized 28x28 Grayscale Input")

                    with col2:
                        st.markdown("#### 2. Grad-CAM Heatmap")
                        if expl and expl.get("heatmap_base64"):
                            st.image(expl["heatmap_base64"], use_container_width=True)
                            st.caption(f"Gradient Activation Map ({expl.get('target_class')})")
                        else:
                            st.info("Grad-CAM heatmap not available.")

                    with col3:
                        st.markdown("#### 3. Anatomical Overlay")
                        if expl and expl.get("overlay_base64"):
                            st.image(expl["overlay_base64"], use_container_width=True)
                            st.caption("Alpha-Blended Radiograph Attention")
                        else:
                            st.info("Overlay visualization not available.")

                    st.markdown("---")
                    st.markdown("### Model Classification Output")

                    pred_class = pred["class_name"]
                    badge_class = "pred-badge-pneumonia" if pred_class == "Pneumonia" else "pred-badge-normal"
                    icon = "🦠" if pred_class == "Pneumonia" else "🫁"

                    st.markdown(
                        f'Prediction: <span class="{badge_class}">{icon} {pred_class}</span>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Model Scores")
                    st.caption(pred["score_note"])

                    scores = pred["model_scores"]
                    for cname, score_val in scores.items():
                        c_col1, c_col2 = st.columns([1, 4])
                        with c_col1:
                            st.write(f"**{cname}**")
                        with c_col2:
                            st.progress(float(score_val), text=f"{score_val * 100:.2f}%")

                    with st.expander("🔍 Interpretability & Technical Note"):
                        st.markdown(
                            """
                            **How Grad-CAM Works in this Architecture:**
                            - Gradients of the predicted class score are backpropagated to the final convolutional layer (`conv3`, 128 feature channels).
                            - Spatial Global Average Pooling calculates the positive importance weight of each feature map.
                            - The rectified weighted combination highlights regions contributing positively to the selected class prediction.
                            - Red/Yellow highlights represent high neural excitation; Blue represents low excitation.
                            """
                        )

                elif resp:
                    st.error(f"Inference request failed: {resp.status_code} — {resp.text}")

            except Exception as e:
                st.error(f"Failed to communicate with prediction service: {e}")

# ---------------------------------------------------------
# Tab 2: Model Evaluation & Metrics
# ---------------------------------------------------------
elif navigation == "📊 Model Evaluation & Metrics":
    st.subheader("Official Benchmark Evaluation Metrics")
    st.write(
        "Evaluation strictly performed on the untouched official **PneumoniaMNIST test split (624 samples)**."
    )

    metrics_file = Path("reports/metrics.json")
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        m = metrics_data["metrics"]
        meta = metrics_data["metadata"]

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{m["accuracy"]*100:.1f}%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{m["recall_sensitivity"]*100:.1f}%</h3><p>Sensitivity (Recall)</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>{m["specificity"]*100:.1f}%</h3><p>Specificity</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h3>{m["precision"]*100:.1f}%</h3><p>Precision</p></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-card"><h3>{m["roc_auc"]:.3f}</h3><p>ROC-AUC</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Diagnostic Plots")

        plot_c1, plot_c2 = st.columns(2)
        cm_path = Path("reports/confusion_matrix.png")
        roc_path = Path("reports/roc_curve.png")
        loss_path = Path("reports/loss_curve.png")

        with plot_c1:
            if cm_path.exists():
                st.image(str(cm_path), caption="Test Split Confusion Matrix", use_container_width=True)
        with plot_c2:
            if roc_path.exists():
                st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC)", use_container_width=True)

        if loss_path.exists():
            st.markdown("### Training & Validation Loss Curves")
            st.image(str(loss_path), caption="Training vs. Validation Loss and Accuracy Progression", use_container_width=True)

        with st.expander("📋 Full Machine-Readable Metrics JSON"):
            st.json(metrics_data)

    else:
        st.warning("Evaluation report not found. Run evaluation with `python scripts/run_evaluation.py`.")

# ---------------------------------------------------------
# Tab 3: Benchmark Dataset Explorer
# ---------------------------------------------------------
elif navigation == "🗂️ Benchmark Dataset Explorer":
    st.subheader("MedMNIST v2 PneumoniaMNIST Overview")
    st.markdown(
        """
        PneumoniaMNIST is a benchmark dataset derived from the pediatric chest radiograph cohort
        (Kermany et al., 2018), preprocessed into standardized 28x28 grayscale images by the MedMNIST v2 team.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>4,708</h3><p>Training Images</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>524</h3><p>Validation Images</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>624</h3><p>Held-Out Test Images</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset Verification Command")
    st.code("python scripts/verify_dataset.py", language="bash")

# ---------------------------------------------------------
# Tab 4: Architecture & Model Card
# ---------------------------------------------------------
elif navigation == "🏗️ Architecture & Model Card":
    st.subheader("Architecture & Model Card")
    st.markdown(
        """
        ### ChestXRayMiniCNN Architecture
        - **Input Layer**: Single channel grayscale image (1, 28, 28)
        - **Conv Block 1**: Conv2D(1 -> 32, 3x3) + BatchNorm2D + ReLU + MaxPool2D(2x2) -> (32, 14, 14)
        - **Conv Block 2**: Conv2D(32 -> 64, 3x3) + BatchNorm2D + ReLU + MaxPool2D(2x2) -> (64, 7, 7)
        - **Conv Block 3 (Grad-CAM Target)**: Conv2D(64 -> 128, 3x3) + BatchNorm2D + ReLU + MaxPool2D(2x2) -> (128, 3, 3)
        - **Classifier Head**: AdaptiveAvgPool2d((3, 3)) + Flatten + Linear(1152 -> 128) + BatchNorm1D + ReLU + Dropout(0.25) + Linear(128 -> 2)
        - **Parameters**: ~240,000 trainable weights (< 1MB serialized checkpoint)

        ---

        ### Intended vs Excluded Uses
        - **Intended Use**: Educational demonstrations, algorithmic interpretability research, and AI imaging pedagogy.
        - **Strictly Excluded**: Clinical diagnosis, emergency triage, autonomous patient evaluation, or therapeutic decision-making.
        """
    )
