import os
import re
import math
import html
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import streamlit as st
import requests

# ============================================================
# LOGGING & CORE CONFIGURATION
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PlantCareAI")

st.set_page_config(
    page_title="PlantCare AI — AI-Based Plant Disease Detection & Health Hub",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Active Root & Storage Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
FARMER_IMAGES_DIR = IMAGES_DIR / "farmers"
AD_IMAGES_DIR = IMAGES_DIR / "advertisements"
CROP_IMAGES_DIR = IMAGES_DIR / "crops"

for directory in [DATA_DIR, IMAGES_DIR, FARMER_IMAGES_DIR, AD_IMAGES_DIR, CROP_IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. MODEL CLASSES REGISTRY (AI INFERENCE SOURCE OF TRUTH)
# ============================================================
DEFAULT_MODEL_CLASSES = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
]

# Standard 35 Vegetable Crops Directory
DEFAULT_35_CROPS = {
    "Solanaceae": [
        {"id": "sol_tomato", "name": "Tomato", "scientific_name": "Solanum lycopersicum", "status": "AI Detection Available", "ai_supported": True, "description": "High-value commercial crop. Full AI diagnosis active for 10 pathological conditions."},
        {"id": "sol_potato", "name": "Potato", "scientific_name": "Solanum tuberosum", "status": "AI Detection Available", "ai_supported": True, "description": "Staple tuber crop. Full AI diagnosis active for Early Blight, Late Blight, and Healthy Foliage."},
        {"id": "sol_capsicum", "name": "Capsicum / Bell Pepper", "scientific_name": "Capsicum annuum", "status": "AI Detection Available", "ai_supported": True, "description": "Sweet pepper crop. Full AI diagnosis active for Bacterial Spot and Healthy Foliage."},
        {"id": "sol_brinjal", "name": "Brinjal / Eggplant", "scientific_name": "Solanum melongena", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "sol_chilli", "name": "Chilli", "scientific_name": "Capsicum frutescens", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."}
    ],
    "Cucurbit Vegetables": [
        {"id": "cuc_pumpkin", "name": "Pumpkin", "scientific_name": "Cucurbita moschata", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_cucumber", "name": "Cucumber", "scientific_name": "Cucumis sativus", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_bottle_gourd", "name": "Bottle Gourd / Lauki", "scientific_name": "Lagenaria siceraria", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_bitter_gourd", "name": "Bitter Gourd / Karela", "scientific_name": "Momordica charantia", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_ridge_gourd", "name": "Ridge Gourd / Turai", "scientific_name": "Luffa acutangula", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_sponge_gourd", "name": "Sponge Gourd / Gilki", "scientific_name": "Luffa aegyptiaca", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_pointed_gourd", "name": "Pointed Gourd / Parwal", "scientific_name": "Trichosanthes dioica", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_ash_gourd", "name": "Ash Gourd / Petha", "scientific_name": "Benincasa hispida", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_zucchini", "name": "Zucchini", "scientific_name": "Cucurbita pepo", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_snake_gourd", "name": "Snake Gourd / Chichinda", "scientific_name": "Trichosanthes cucumerina", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "cuc_ivy_gourd", "name": "Ivy Gourd / Kundru", "scientific_name": "Coccinia grandis", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."}
    ],
    "Common Indian Vegetables": [
        {"id": "com_okra", "name": "Okra / Lady Finger / Bhindi", "scientific_name": "Abelmoschus esculentus", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "com_french_bean", "name": "French Bean", "scientific_name": "Phaseolus vulgaris", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "com_green_bean", "name": "Green Bean", "scientific_name": "Phaseolus vulgaris var.", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "com_peas", "name": "Peas / Matar", "scientific_name": "Pisum sativum", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "com_sweet_corn", "name": "Sweet Corn", "scientific_name": "Zea mays var. saccharata", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "com_carrot", "name": "Carrot", "scientific_name": "Daucus carota", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "com_radish", "name": "Radish / Mooli", "scientific_name": "Raphanus sativus", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "com_beetroot", "name": "Beetroot", "scientific_name": "Beta vulgaris", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "com_turnip", "name": "Turnip / Shalgam", "scientific_name": "Brassica rapa subsp. rapa", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."}
    ],
    "Cole Vegetables": [
        {"id": "col_cabbage", "name": "Cabbage", "scientific_name": "Brassica oleracea var. capitata", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "col_cauliflower", "name": "Cauliflower", "scientific_name": "Brassica oleracea var. botrytis", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "col_broccoli", "name": "Broccoli", "scientific_name": "Brassica oleracea var. italica", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."}
    ],
    "Leafy Vegetables": [
        {"id": "lea_spinach", "name": "Spinach / Palak", "scientific_name": "Spinacia oleracea", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "lea_fenugreek", "name": "Fenugreek / Methi", "scientific_name": "Trigonella foenum-graecum", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "lea_coriander", "name": "Coriander / Dhaniya", "scientific_name": "Coriandrum sativum", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "lea_lettuce", "name": "Lettuce", "scientific_name": "Lactuca sativa", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "lea_amaranth", "name": "Amaranth / Chaulai", "scientific_name": "Amaranthus cruentus", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."}
    ],
    "Bulb Vegetables": [
        {"id": "bul_onion", "name": "Onion", "scientific_name": "Allium cepa", "status": "Knowledge Available / Training Planned", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."},
        {"id": "bul_garlic", "name": "Garlic", "scientific_name": "Allium sativum", "status": "Coming Soon", "ai_supported": False, "description": "Crop information available — AI leaf disease detection is being calibrated."}
    ]
}

# ============================================================
# 2. DATA ACCESS & PERSISTENCE LAYER
# ============================================================
def load_json_file(file_path: Path, fallback_data):
    if file_path.is_file():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"Error reading {file_path}: {exc}")
            return fallback_data
    return fallback_data

def save_json_file(file_path: Path, data):
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        logger.error(f"Error writing to {file_path}: {exc}")
        return False

def get_diseases_database():
    candidates = [
        DATA_DIR / "diseases.json",
        DATA_DIR / "disease_database.json",
        DATA_DIR / "plant_database.json",
    ]
    for c in candidates:
        if c.is_file():
            data = load_json_file(c, {})
            if isinstance(data, dict) and any("Pepper" in k or "Tomato" in k for k in data.keys()):
                return data
    return {}

def get_crops_database():
    candidates = [
        DATA_DIR / "plant_database.json",
        DATA_DIR / "crop_database.json",
        DATA_DIR / "crops.json",
    ]
    for c in candidates:
        if c.is_file():
            data = load_json_file(c, {})
            if isinstance(data, dict) and any("Solanaceae" in k or "Cucurbit" in k for k in data.keys()):
                return data
    return DEFAULT_35_CROPS

def get_advisory_database():
    candidates = [
        DATA_DIR / "advisory_rules.json",
        DATA_DIR / "advisory_database.json",
    ]
    for c in candidates:
        if c.is_file():
            return load_json_file(c, {})
    return {}

def get_farmer_stories():
    return load_json_file(DATA_DIR / "farmer_stories.json", [])

def get_advertisements():
    return load_json_file(DATA_DIR / "advertisements.json", [])

def get_disease_detail(condition_name: str):
    db = get_diseases_database()
    if db and condition_name in db:
        return db[condition_name]
    
    return {
        "crop": "Vegetable Crop",
        "scientific_crop": "",
        "category": "General Agronomic Health",
        "severity": "Moderate",
        "status": "Analyzed",
        "badge": "status-warning",
        "overview": "Comprehensive pathological details for this condition are cataloged in the PlantCare AI Agronomic Hub.",
        "etiology": "Pathological inoculation favored by high foliar wetness and microclimate fluctuations.",
        "symptoms": "Visible chlorotic spotting, necrosis, lesions, or vascular wilting across foliar/fruit tissue.",
        "causes": "Excess canopy humidity, pathogen inoculums, or physiological nutritional imbalance.",
        "chemical_treatment": "Apply locally registered protective fungicides or bactericides strictly according to label directions.",
        "organic_treatment": "Foliar application of bio-antagonists (Bacillus subtilis or Trichoderma) and neem oil extract.",
        "prevention": "Ensure clean seed stock, proper row ventilation, balanced fertigation, and field sanitation.",
        "fertilizer": "Maintain balanced N-P-K ratios; supplement with Calcium and Potassium.",
        "pest_control": "Monitor insect vectors (whiteflies, thrips, aphids) regularly.",
        "farmer_tips": "Conduct field inspections in early morning while symptoms are crisp.",
        "ideal_climate": "Humid, warm canopy microclimate",
        "economic_threshold": "Initiate corrective action upon observing 5% foliar damage."
    }

def save_uploaded_asset(uploaded_file, target_folder: Path, prefix: str = "img") -> str:
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"
    safe_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{ext}"
    target_path = target_folder / safe_name
    
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    img.save(target_path, quality=85, optimize=True)
    return str(target_path.relative_to(BASE_DIR)).replace("\\", "/")

# ============================================================
# 3. ULTRA-PREMIUM DYNAMIC ANIMATED & RESPONSIVE CSS
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --primary: #059669;
        --primary-dark: #064e3b;
        --primary-deep: #022c22;
        --primary-light: #ecfdf5;
        --primary-border: #a7f3d0;
        --accent-emerald: #10b981;
        --text-main: #091e14;
        --text-muted: #4a6356;
        --card-bg: rgba(255, 255, 255, 0.90);
        --card-border: rgba(226, 236, 230, 0.88);
        --shadow-sm: 0 4px 14px rgba(6, 78, 59, 0.04);
        --shadow-md: 0 12px 32px rgba(6, 78, 59, 0.07);
        --shadow-lg: 0 22px 55px rgba(6, 78, 59, 0.12);
    }

    /* Ambient Moving Gradient Background */
    @keyframes ambientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: linear-gradient(-45deg, #f0fdf4, #ecfdf5, #f7faf8, #e6fcf0);
        background-size: 400% 400%;
        animation: ambientShift 20s ease infinite;
        color: var(--text-main);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Modern Luxury Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #02261d 0%, #043628 45%, #064e3b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15);
    }
    [data-testid="stSidebar"] * {
        color: #f0fdf4 !important;
    }

    /* Auto-Responsive Container Layout */
    .block-container {
        max-width: 1340px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Fluid Responsive Hero Banner */
    .hero-banner {
        padding: 3.4rem 3.2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #022c22 0%, #044433 35%, #065f46 70%, #047857 100%);
        color: white;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 2rem;
        width: 100%;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        width: 440px;
        height: 440px;
        right: -120px;
        top: -140px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.28) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        left: -80px;
        bottom: -100px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(12px);
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #d1fae5;
        margin-bottom: 1.1rem;
        border: 1px solid rgba(255, 255, 255, 0.22);
    }
    .hero-title {
        font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;
        font-size: clamp(2.2rem, 4.2vw, 3.8rem);
        line-height: 1.15;
        margin: 0.3rem 0 0.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .hero-desc {
        max-width: 720px;
        font-size: clamp(1rem, 1.8vw, 1.15rem);
        line-height: 1.7;
        color: #e6fcf0;
        margin-bottom: 1.5rem;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.2rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        font-size: 0.88rem;
        font-weight: 700;
        color: #a7f3d0;
    }

    /* Glassmorphic Luxury Cards */
    .product-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 22px;
        padding: 1.8rem 1.9rem;
        margin: 0.95rem 0;
        box-shadow: var(--shadow-sm);
        backdrop-filter: blur(14px);
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
    }
    .product-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-border);
        transform: translateY(-2px);
    }
    .product-card h3 {
        margin: 0 0 0.6rem;
        color: var(--primary-dark);
        font-size: 1.25rem;
        font-weight: 800;
    }
    .card-muted {
        color: var(--text-muted);
        line-height: 1.68;
        font-size: 0.96rem;
    }

    /* Metric Visualizers */
    .metric-container {
        background: #ffffff;
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 1.35rem 1.3rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        width: 100%;
        margin-bottom: 0.75rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 850;
        color: var(--primary);
    }
    .metric-label {
        color: var(--text-muted);
        font-size: 0.82rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.35rem;
    }

    /* Results Banner */
    .result-panel {
        border-radius: 26px;
        padding: 2.2rem;
        background: #ffffff;
        border: 1px solid var(--card-border);
        box-shadow: var(--shadow-md);
        margin: 1.2rem 0;
        width: 100%;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 1.1rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.82rem;
    }
    .status-healthy {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .status-warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .status-danger {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    .confidence-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 750;
        background: #f1f5f9;
        color: #334155;
        margin-left: 0.5rem;
    }

    /* Probability Grid */
    .prob-grid-row {
        display: grid;
        grid-template-columns: 260px 1fr 90px;
        gap: 16px;
        align-items: center;
        margin: 0.85rem 0;
    }
    .prob-label {
        font-weight: 750;
        font-size: 0.94rem;
        color: var(--text-main);
    }
    .prob-track {
        height: 12px;
        background: #e8f1ec;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        border-radius: 999px;
    }
    .prob-pct {
        text-align: right;
        font-weight: 850;
        color: var(--primary-dark);
        font-size: 0.96rem;
    }

    /* 2x2 Info Grid */
    .info-layout-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.35rem;
        margin-top: 1.1rem;
    }
    .info-box {
        background: #ffffff;
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 1.45rem;
    }
    .info-box-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: var(--primary-dark);
        margin-bottom: 0.55rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .info-box-text {
        font-size: 0.93rem;
        color: var(--text-main);
        line-height: 1.65;
        margin: 0;
    }

    /* Sponsored Card */
    .ad-card {
        background: #ffffff;
        border: 1px solid #bbf7d0;
        border-radius: 24px;
        padding: 1.8rem;
        margin: 1.4rem 0;
        box-shadow: var(--shadow-md);
        width: 100%;
    }
    .ad-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
    }
    .ad-badge {
        font-size: 0.74rem;
        font-weight: 850;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        background: #ecfdf5;
        color: #065f46;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        border: 1px solid #a7f3d0;
    }
    .ad-title {
        font-size: 1.35rem;
        font-weight: 850;
        color: #064e3b;
        margin: 0.4rem 0;
    }

    .disclaimer-card {
        font-size: 0.86rem;
        color: #64748b;
        background: #ffffff;
        border-left: 4px solid #059669;
        padding: 0.95rem 1.25rem;
        border-radius: 0 14px 14px 0;
        margin-top: 1.6rem;
        line-height: 1.65;
        box-shadow: var(--shadow-sm);
    }

    .app-footer-bar {
        margin-top: 4.5rem;
        padding: 2.2rem 0.5rem 1.8rem 0.5rem;
        border-top: 1px solid var(--card-border);
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: var(--text-muted);
        font-size: 0.92rem;
        flex-wrap: wrap;
        gap: 1.2rem;
    }
    .footer-brand {
        font-weight: 850;
        color: var(--primary-dark);
        font-size: 1.05rem;
    }

    /* ============================================================
       MOBILE VIEWPORT ADAPTIVE RESPONSIVENESS (< 850px)
       ============================================================ */
    @media (max-width: 850px) {
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            padding-top: 1rem !important;
        }
        .hero-banner {
            padding: 2rem 1.4rem !important;
            border-radius: 20px !important;
        }
        .hero-title {
            font-size: 2rem !important;
        }
        .product-card {
            padding: 1.4rem 1.25rem !important;
            border-radius: 18px !important;
        }
        .prob-grid-row {
            grid-template-columns: 1fr !important;
            gap: 6px !important;
        }
        .prob-pct {
            text-align: left !important;
        }
        .info-layout-grid {
            grid-template-columns: 1fr !important;
            gap: 0.9rem !important;
        }
        .result-panel {
            padding: 1.5rem 1.25rem !important;
            border-radius: 20px !important;
        }
        .app-footer-bar {
            flex-direction: column !important;
            text-align: center !important;
            gap: 0.8rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 4. DEEP LEARNING MODEL ENGINE & INFERENCE
# ============================================================
def find_model_path():
    candidates = [
        BASE_DIR / "plant_disease_model.h5",
        BASE_DIR / "models" / "plant_disease_model.h5",
        BASE_DIR / "model" / "plant_disease_model.h5",
        Path.cwd() / "plant_disease_model.h5",
        Path.cwd() / "models" / "plant_disease_model.h5",
        BASE_DIR.parent / "plant_disease_model.h5",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None

@st.cache_resource(show_spinner=False)
def load_screening_model():
    path = find_model_path()
    if path is None:
        return None, "Model file 'plant_disease_model.h5' not found in workspace."
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(path), compile=False)
        return model, str(path)
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        return None, f"Model load error: {exc}"

def inspect_model_dimensions(model):
    dimension_candidates = []
    try:
        in_shape = model.input_shape
        if isinstance(in_shape, list):
            in_shape = in_shape[0]
        if len(in_shape) >= 3 and isinstance(in_shape[1], int) and isinstance(in_shape[2], int):
            if in_shape[1] > 0 and in_shape[2] > 0:
                dimension_candidates.append(int(in_shape[1]))
    except Exception:
        pass

    try:
        for layer in model.layers:
            if hasattr(layer, "weights") and layer.weights:
                w_shape = layer.weights[0].shape
                if len(w_shape) == 2:
                    in_features = int(w_shape[0])
                    for channel in [3, 16, 32, 64, 128, 256, 512, 1024]:
                        if in_features % channel == 0:
                            spatial = in_features // channel
                            sq = int(math.isqrt(spatial))
                            if sq * sq == spatial:
                                for pool_factor in [1, 2, 4, 8, 16, 32]:
                                    candidate = sq * pool_factor
                                    if 64 <= candidate <= 512 and candidate not in dimension_candidates:
                                        dimension_candidates.append(candidate)
                    break
    except Exception:
        pass

    for std_size in [224, 128, 112, 160, 192, 96, 256]:
        if std_size not in dimension_candidates:
            dimension_candidates.append(std_size)

    return dimension_candidates

def validate_image_quality(img):
    stat = ImageStat.Stat(img)
    r, g, b = stat.mean[:3]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    warnings = []
    if brightness < 40:
        warnings.append("The image appears dark. Ensure adequate illumination for reliable confidence.")
    elif brightness > 225:
        warnings.append("The image appears overexposed. Ensure leaf/fruit texture is clearly visible.")
    var = stat.var
    avg_var = sum(var[:3]) / 3.0
    if avg_var < 100:
        warnings.append("The image appears soft in focus. A sharper, focused photo is recommended.")
    return warnings

def validate_and_load_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        img = img.convert("RGB")
        return img, None
    except Exception:
        return None, "Unable to read the uploaded image. Please provide a valid JPG, JPEG, PNG, or WEBP file."

def prepare_tensor(image, size):
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def normalize_probabilities(raw_output):
    import tensorflow as tf
    probs = np.asarray(raw_output).squeeze().astype(np.float32)
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    if probs.size == 0:
        raise ValueError("Model produced an empty probability vector.")
    if np.any(probs < 0) or np.max(probs) > 1.0 or not np.isclose(float(probs.sum()), 1.0, atol=0.05):
        probs = tf.nn.softmax(probs).numpy()
    return probs

def execute_adaptive_prediction(model, image, model_classes):
    if model is None:
        raise RuntimeError("AI model is currently offline. Please ensure plant_disease_model.h5 is in the application directory.")

    target_resolutions = inspect_model_dimensions(model)

    for target_res in target_resolutions:
        try:
            tensor = prepare_tensor(image, target_res)
            raw_pred = model.predict(tensor, verbose=0)
            probs = normalize_probabilities(raw_pred)

            if probs.size < 2:
                continue

            class_names = [
                model_classes[i] if i < len(model_classes) else f"Class {i+1}"
                for i in range(probs.size)
            ]

            sorted_indices = np.argsort(probs)[::-1]
            top_predictions = [
                (class_names[int(idx)], float(probs[int(idx)]) * 100.0)
                for idx in sorted_indices[:5]
            ]

            primary_condition = top_predictions[0][0]
            confidence_score = top_predictions[0][1]

            return primary_condition, confidence_score, top_predictions, target_res
        except Exception:
            continue

    raise RuntimeError("Unable to complete screening for this image. Please upload a clear leaf/fruit photo and retry.")

# ============================================================
# 5. WEATHER & SPRAY ADVISORY ENGINE
# ============================================================
def get_live_weather_data(latitude: float, longitude: float):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m&timezone=auto"
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        return {
            "temperature_c": current.get("temperature_2m", 24.0),
            "relative_humidity_pct": current.get("relative_humidity_2m", 70.0),
            "rain_probability_pct": current.get("precipitation_probability", 10.0),
            "wind_speed_kmh": current.get("wind_speed_10m", 8.0),
            "source": "Open-Meteo Satellite API"
        }, None
    except Exception as exc:
        return None, f"Could not connect to live weather service: {exc}"

def calculate_spray_advisory(temp, humidity, rain_prob, wind_speed, rules):
    spray_limits = rules.get("spray_rules", {
        "max_wind_speed_kmh": 15.0,
        "max_rain_probability_pct": 30.0,
        "min_temperature_c": 10.0,
        "max_temperature_c": 35.0,
        "max_relative_humidity_pct": 88.0
    })

    reasons = []
    suitable = True

    if wind_speed > spray_limits["max_wind_speed_kmh"]:
        suitable = False
        reasons.append(f"High wind speed ({wind_speed:.1f} km/h > {spray_limits['max_wind_speed_kmh']} km/h) creates significant spray drift hazard.")
    if rain_prob > spray_limits["max_rain_probability_pct"]:
        suitable = False
        reasons.append(f"High rain chance ({rain_prob:.0f}% > {spray_limits['max_rain_probability_pct']}%) risks chemical wash-off before foliar uptake.")
    if temp > spray_limits["max_temperature_c"]:
        suitable = False
        reasons.append(f"Elevated temperature ({temp:.1f}°C) may cause foliar scorch and fast droplet evaporation.")
    elif temp < spray_limits["min_temperature_c"]:
        suitable = False
        reasons.append(f"Low temperature ({temp:.1f}°C) slows systemic chemical absorption.")
    if humidity > spray_limits["max_relative_humidity_pct"]:
        reasons.append(f"High humidity ({humidity:.0f}%) extends drying time and may encourage spore germination.")

    status = "Favorable (Safe Application Window)" if suitable else "Unfavorable (Postpone Spraying)"
    return status, reasons

def calculate_disease_risks(temp, humidity, rules):
    thresholds = rules.get("disease_risk_thresholds", {
        "late_blight": {"min_temp": 10.0, "max_temp": 23.0, "min_humidity": 85.0, "risk_label": "Critical Risk (Late Blight / Oomycete Pressure)"},
        "early_blight": {"min_temp": 22.0, "max_temp": 30.0, "min_humidity": 70.0, "risk_label": "High Risk (Fungal Blight & Leaf Mold Spore Germination)"},
        "bacterial_spot": {"min_temp": 24.0, "max_temp": 34.0, "min_humidity": 78.0, "risk_label": "Elevated Risk (Bacterial Foliar Invasion Pressure)"},
        "spider_mites": {"min_temp": 27.0, "max_temp": 45.0, "max_humidity": 50.0, "risk_label": "Elevated Risk (Hot, Dry Microclimate Encouraging Spider Mites)"}
    })
    active_risks = []
    for _, config in thresholds.items():
        min_t = config.get("min_temp", -999)
        max_t = config.get("max_temp", 999)
        min_h = config.get("min_humidity", 0)
        max_h = config.get("max_humidity", 100)
        if (min_t <= temp <= max_t) and (min_h <= humidity <= max_h):
            active_risks.append(config.get("risk_label"))
    if not active_risks:
        active_risks.append("Baseline Pathogen Risk (Normal Atmospheric Conditions)")
    return active_risks

# ============================================================
# 6. GEOSPATIAL MAP RESOURCE QUERY
# ============================================================
def maps_query_url(lat, lon, query):
    return f"https://www.google.com/maps/search/{quote_plus(query)}/@{lat},{lon},14z"

def query_nearby_plant_care(lat, lon, limit=8):
    query = f"""
    [out:json][timeout:20];
    (
      nwr(around:8000,{lat},{lon})["shop"="garden_centre"];
      nwr(around:8000,{lat},{lon})["shop"="farm"];
      nwr(around:8000,{lat},{lon})["shop"="agrarian"];
      nwr(around:8000,{lat},{lon})["shop"="doityourself"]["name"];
      nwr(around:8000,{lat},{lon})["craft"="gardener"];
      nwr(around:8000,{lat},{lon})["amenity"="garden_centre"];
      nwr(around:8000,{lat},{lon})["name"]["shop"];
    );
    out center tags;
    """
    headers = {"User-Agent": "PlantCareAI/8.0 (Commercial Agritech AI)"}
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    data = None
    last_err = None
    for ep in endpoints:
        try:
            resp = requests.post(ep, data=query.encode("utf-8"), headers=headers, timeout=22)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            last_err = exc

    if data is None:
        raise RuntimeError(f"Geospatial service error: {last_err}")

    results = []
    seen = set()
    for item in data.get("elements", []):
        tags = item.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        norm_key = re.sub(r"\s+", " ", name.strip().lower())
        if norm_key in seen:
            continue
        seen.add(norm_key)

        center = item.get("center", {})
        slat = item.get("lat", center.get("lat"))
        slon = item.get("lon", center.get("lon"))
        if slat is None or slon is None:
            continue

        street = tags.get("addr:street", "")
        city = tags.get("addr:city", "")
        addr = ", ".join([x for x in [street, city] if x])
        raw_type = tags.get("shop") or tags.get("amenity") or tags.get("craft") or "agricultural"
        cat_title = "🌱 Nursery / Garden Center" if "garden" in raw_type else "🌾 Agricultural Supplies & Seeds"

        results.append({
            "name": name,
            "type": cat_title,
            "address": addr or "Address available in Google Maps view",
            "lat": float(slat),
            "lon": float(slon),
            "maps": maps_query_url(float(slat), float(slon), name),
        })
        if len(results) >= limit:
            break

    return results

# ============================================================
# 7. EXPORTABLE PLANT HEALTH REPORT GENERATOR
# ============================================================
def generate_plain_text_report(p, info):
    top5_formatted = "\n".join([f"  {i}. {n} — {v:.2f}%" for i, (n, v) in enumerate(p["top5"], 1)])
    
    return f"""======================================================================
PLANTCARE AI — PLANT HEALTH SCREENING DOSSIER
Powered by SEA AUTO
Developed by Madhav Kumar
======================================================================
Screening Date & Time : {p["timestamp"]}
Crop Name             : {p["plant"]}
Diagnosed Condition   : {p["condition"]}
Condition Category    : {p["category"]}
Health Status         : {info.get("status", "Analyzed")}
Confidence Level      : {p["confidence"]:.2f}%

----------------------------------------------------------------------
1. CLINICAL OVERVIEW & DESCRIPTION
----------------------------------------------------------------------
{info.get("overview", "N/A")}

----------------------------------------------------------------------
2. PATHOGEN ETIOLOGY & LIFE CYCLE
----------------------------------------------------------------------
{info.get("etiology", "N/A")}

----------------------------------------------------------------------
3. VISUAL SYMPTOMS & PATHOLOGICAL CAUSES
----------------------------------------------------------------------
Symptoms:
{info.get("symptoms", "N/A")}

Possible Causes:
{info.get("causes", "N/A")}

----------------------------------------------------------------------
4. TREATMENT & MANAGEMENT REGIMES
----------------------------------------------------------------------
{info.get("chemical_treatment", info.get("treatment", "N/A"))}

Organic Alternative:
{info.get("organic_treatment", "N/A")}

----------------------------------------------------------------------
5. PREVENTATIVE AGRONOMIC SCHEDULE
----------------------------------------------------------------------
{info.get("prevention", "N/A")}

----------------------------------------------------------------------
6. SMART FERTILIZER & PEST CONTROL GUIDANCE
----------------------------------------------------------------------
Fertilizer Guidance:
{info.get("fertilizer", "N/A")}

Pest Management:
{info.get("pest_control", "N/A")}

Farmer / Grower Tips:
{info.get("farmer_tips", "N/A")}

----------------------------------------------------------------------
7. AI PROBABILITY DISTRIBUTION (TOP 5)
----------------------------------------------------------------------
{top5_formatted}

======================================================================
Disclaimer: AI-assisted visual screening is intended as an initial 
assessment. Use locally approved products according to label directions 
and consult an agriculture professional when needed.
Developed by Madhav Kumar | © 2026 PlantCare AI. All rights reserved.
======================================================================
"""

# ============================================================
# INITIALIZE SESSION STATE & MODEL
# ============================================================
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "user_coords" not in st.session_state:
    st.session_state.user_coords = None
if "nearby_shops" not in st.session_state:
    st.session_state.nearby_shops = None

MODEL_OBJ, MODEL_LOG = load_screening_model()

# ============================================================
# SPONSORED PARTNER COMPONENT
# ============================================================
def render_sponsored_partner_card():
    ads = get_advertisements()
    active_ads = []
    today = datetime.now().strftime("%Y-%m-%d")

    for ad in ads:
        if ad.get("status") == "active":
            start_date = ad.get("start_date", "")
            end_date = ad.get("end_date", "")
            if start_date and start_date > today:
                continue
            if end_date and end_date < today:
                continue
            active_ads.append(ad)

    if not active_ads:
        return

    active_ads.sort(key=lambda x: x.get("priority", 99))
    top_ad = active_ads[0]

    st.markdown(f"""
    <div class="ad-card">
        <div class="ad-header">
            <span class="ad-badge">✦ Featured Partner</span>
            <span style="font-size: 0.82rem; color: #64748b; font-weight: 700;">{top_ad.get('company', 'SEA AUTO Ecosystem')}</span>
        </div>
        <div class="ad-title">{html.escape(top_ad.get('title', ''))}</div>
        <div style="font-size: 0.96rem; color: #334155; line-height: 1.65; margin-bottom: 0.85rem;">
            {html.escape(top_ad.get('description', ''))}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if top_ad.get("image"):
        img_p = BASE_DIR / top_ad["image"]
        if img_p.is_file():
            st.image(str(img_p), use_container_width=True)

    if top_ad.get("button_url") and top_ad.get("button_text"):
        st.link_button(top_ad["button_text"], top_ad["button_url"])

# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1.2rem;">
            <div style="font-size: 2.3rem;">🌿</div>
            <div style="font-size: 1.6rem; font-weight: 850; letter-spacing: -0.02em; font-family: 'Space Grotesk', sans-serif;">PlantCare AI</div>
            <div style="opacity: 0.82; font-size: 0.82rem; margin-top: 0.25rem;">
                AI-Powered Plant Health Screening
            </div>
        </div>
        """, unsafe_allow_html=True)

        selected_page = st.radio(
            "Navigation Menu",
            [
                "🏠 Home",
                "🔬 Disease Detection",
                "📄 Plant Health Report",
                "🌱 Explore Crops",
                "📚 Disease Knowledge Hub",
                "🌾 Farmer Stories",
                "🌦️ Weather & Spray Advisory",
                "📍 Nearby Plant Care",
                "⚙️ Content Manager",
                "ℹ️ About PlantCare AI"
            ],
            label_visibility="collapsed",
            key="navigation_page_selector"
        )

        st.markdown("---")
        st.markdown("**AI MODEL STATUS**")
        if MODEL_OBJ is not None:
            st.markdown("🟢 **Model Online & Loaded**")
        else:
            st.markdown("🟡 **Model Offline (Demo Mode)**")
            st.caption(f"Status: {MODEL_LOG}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="padding: 1.1rem; border-radius: 18px; background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.12); box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.75;">DEVELOPER</div>
            <div style="font-weight: 800; font-size: 1.05rem; margin-top: 0.2rem; color: #ffffff;">Madhav Kumar</div>
            <div style="font-size: 0.82rem; color: #a7f3d0; font-weight: 750; margin-top: 0.45rem;">✦ Powered by SEA AUTO</div>
        </div>
        """, unsafe_allow_html=True)

        return selected_page

# ============================================================
# PAGE 1: HOME
# ============================================================
def render_home_page():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-kicker">✦ SMART AGRITECH INTELLIGENCE</div>
        <div class="hero-title">PlantCare AI</div>
        <div style="font-size: 1.38rem; font-weight: 700; color: #a7f3d0; margin-bottom: 0.6rem;">
            AI-Powered Plant Health Screening & Diagnostic Engine
        </div>
        <div class="hero-desc">
            Upload plant leaf or fruit imagery to receive objective AI-assisted health assessments, 
            clinical pathology insights, and practical crop-care guidance across 35 agricultural crops.
        </div>
        <span class="hero-pill">🌿 Powered by SEA AUTO</span>
    </div>
    """, unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.4, 2.2])
    with btn_col1:
        if st.button("Analyze Leaf / Fruit", type="primary", use_container_width=True):
            st.session_state["navigation_page_selector"] = "🔬 Disease Detection"
            st.rerun()
    with btn_col2:
        if st.button("Explore 35 Crops", use_container_width=True):
            st.session_state["navigation_page_selector"] = "🌱 Explore Crops"
            st.rerun()
    with btn_col3:
        if st.button("Disease Knowledge Hub", use_container_width=True):
            st.session_state["navigation_page_selector"] = "📚 Disease Knowledge Hub"
            st.rerun()

    st.write("")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="product-card">
            <h3>🔬 Multi-Organ Vision Screening</h3>
            <div class="card-muted">
                Fast, objective visual assessment from leaf, fruit, or tuber imagery with transparent neural network confidence distributions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="product-card">
            <h3>📊 35-Crop Pathology Compendium</h3>
            <div class="card-muted">
                Exhaustive disease descriptions, life cycles, stage-by-stage symptoms, and verified chemical and biological regimens.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="product-card">
            <h3>🌱 Dynamic Weather & Care Guidance</h3>
            <div class="card-muted">
                Live meteorological spray feasibility windows, disease pressure indexes, and localized plant-care discovery.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📢 Featured Products & Services")
    render_sponsored_partner_card()

# ============================================================
# PAGE 2: DISEASE DETECTION
# ============================================================
def render_detection_page():
    st.markdown("## 🔬 Disease Detection")
    st.caption("Upload a clear photo of a plant leaf, fruit, or tuber to receive an AI-assisted health assessment and practical care guidance.")

    model_classes = DEFAULT_MODEL_CLASSES

    uploaded_file = st.file_uploader(
        "Upload plant specimen image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image, err_msg = validate_and_load_image(uploaded_file)
        if err_msg:
            st.error(f"❌ Analysis could not be completed: {err_msg}")
            return

        quality_warnings = validate_image_quality(image)
        for warn in quality_warnings:
            st.warning(f"💡 {warn}")

        col_preview, col_action = st.columns([1, 1.25], gap="large")

        with col_preview:
            st.image(image, caption="Uploaded Specimen Preview", use_container_width=True)

        with col_action:
            st.markdown("""
            <div class="product-card">
                <h3>Ready for Neural Screening</h3>
                <div class="card-muted">
                    Click the button below to initiate multi-resolution neural network classification against verified pathological datasets.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔬 Analyze Specimen", type="primary", use_container_width=True):
                with st.spinner("Analyzing plant specimen..."):
                    try:
                        condition, confidence, top5, used_res = execute_adaptive_prediction(MODEL_OBJ, image, model_classes)
                        info = get_disease_detail(condition)

                        st.session_state.prediction_data = {
                            "image": image,
                            "condition": condition,
                            "confidence": confidence,
                            "top5": top5,
                            "plant": info.get("crop", "Solanaceous Crop"),
                            "category": info.get("category", "General Condition"),
                            "severity": info.get("severity", "Moderate"),
                            "resolution": used_res,
                            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                        }
                        st.success("Screening completed successfully.")
                    except Exception as exc:
                        st.session_state.prediction_data = None
                        st.error(f"❌ Analysis could not be completed: {exc}")

    if st.session_state.prediction_data is not None:
        p = st.session_state.prediction_data
        info = get_disease_detail(p["condition"])

        st.write("")
        conf_tag = (
            '<span class="confidence-tag">🟢 High confidence</span>'
            if p["confidence"] >= 70.0
            else '<span class="confidence-tag">🟡 Moderate / Low confidence</span>'
        )

        st.markdown(f"""
        <div class="result-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
                <div style="font-size: 0.82rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b;">
                    Screening Result
                </div>
                <div>
                    <span class="status-badge {info.get('badge', 'status-warning')}">● {info.get('status', 'Analyzed')}</span>
                    {conf_tag}
                </div>
            </div>
            <div style="font-size: 1.95rem; font-weight: 850; color: #0d1f17; margin-bottom: 0.5rem; font-family: 'Space Grotesk', sans-serif;">
                {p['condition']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if p["confidence"] < 60.0:
            st.info("💡 **Low-confidence result:** Try uploading a clearer specimen image with good natural lighting and minimal background clutter.")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="font-size: 1.4rem;">{p['plant']}</div>
                <div class="metric-label">Plant</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="font-size: 1.4rem;">{p['category']}</div>
                <div class="metric-label">Category</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="font-size: 1.4rem;">{p['severity']}</div>
                <div class="metric-label">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="font-size: 1.4rem;">{p['confidence']:.2f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown(f"""
        <div class="product-card">
            <h3>Disease Information & Clinical Overview</h3>
            <div class="info-layout-grid">
                <div class="info-box">
                    <div class="info-box-title">📖 Description</div>
                    <p class="info-box-text">{html.escape(info.get('overview', 'N/A'))}</p>
                </div>
                <div class="info-box">
                    <div class="info-box-title">🤒 Symptoms</div>
                    <p class="info-box-text">{html.escape(info.get('symptoms', 'N/A'))}</p>
                </div>
                <div class="info-box">
                    <div class="info-box-title">⚠️ Causes</div>
                    <p class="info-box-text">{html.escape(info.get('causes', 'N/A'))}</p>
                </div>
                <div class="info-box">
                    <div class="info-box-title">💊 Treatment / Management</div>
                    <p class="info-box-text">{html.escape(info.get('chemical_treatment', info.get('treatment', 'N/A')))}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🌾 Smart Recommendations")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="product-card">
                <h3>🌱 Fertilizer Guidance</h3>
                <div class="card-muted">{html.escape(info.get('fertilizer', 'N/A'))}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="product-card">
                <h3>🐛 Pest / Disease Control</h3>
                <div class="card-muted">{html.escape(info.get('pest_control', 'N/A'))}</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="product-card">
                <h3>👨‍🌾 Farmer Tips</h3>
                <div class="card-muted">{html.escape(info.get('farmer_tips', 'N/A'))}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Top 5 AI Predictions")
        st.caption("Actual model probability distribution")
        for name, prob in p["top5"]:
            safe_name = html.escape(name)
            bar_w = max(0.0, min(100.0, prob))
            st.markdown(f"""
            <div class="prob-grid-row">
                <div class="prob-label">{safe_name}</div>
                <div class="prob-track"><div class="prob-fill" style="width: {bar_w:.2f}%"></div></div>
                <div class="prob-pct">{prob:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-card">
            AI-assisted visual screening is intended as an initial assessment. Image quality, lighting, 
            plant variety and environmental conditions may affect the result. For important agricultural decisions, 
            consult a qualified agricultural professional.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 3: PLANT HEALTH REPORT
# ============================================================
def render_report_page():
    st.markdown("## 📄 Plant Health Report")
    st.caption("Complete diagnostic dossier ready for review and local export.")

    p = st.session_state.prediction_data
    if not p:
        st.markdown("""
        <div class="product-card" style="text-align: center; padding: 3.5rem 2rem;">
            <div style="font-size: 2.8rem; margin-bottom: 0.6rem;">📋</div>
            <h3>No Active Screening Record</h3>
            <div class="card-muted">
                Please analyze a plant leaf or fruit in the 'Disease Detection' section first to view and download your clinical health report.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Disease Detection", type="primary"):
            st.session_state["navigation_page_selector"] = "🔬 Disease Detection"
            st.rerun()
        return

    info = get_disease_detail(p["condition"])

    col_rep_img, col_rep_meta = st.columns([1, 2], gap="large")
    with col_rep_img:
        st.image(p["image"], caption="Screened Specimen", use_container_width=True)
    with col_rep_meta:
        st.markdown(f"""
        <div class="hero-banner" style="padding: 2rem 2.2rem; margin-bottom: 1rem;">
            <div class="hero-kicker">HEALTH SCREENING DOSSIER</div>
            <div class="hero-title" style="font-size: 1.9rem;">{html.escape(p['condition'])}</div>
            <div>Confidence: <strong>{p['confidence']:.2f}%</strong> | Status: <strong>{info.get('status', 'Analyzed')}</strong></div>
            <div style="margin-top: 0.6rem; font-size: 0.88rem; color: #d1fae5;">Developed by Madhav Kumar | Powered by SEA AUTO</div>
        </div>
        """, unsafe_allow_html=True)

    report_sections = [
        ("🌱 Plant", p["plant"]),
        ("🦠 Detected Condition", p["condition"]),
        ("📊 Diagnostic Category", p["category"]),
        ("⚠️ Risk Level", p["severity"]),
        ("📖 Description", info.get("overview", "N/A")),
        ("🤒 Symptoms", info.get("symptoms", "N/A")),
        ("⚠️ Causes", info.get("causes", "N/A")),
        ("💊 Treatment / Management", info.get("chemical_treatment", info.get("treatment", "N/A"))),
        ("🛡️ Prevention", info.get("prevention", "N/A")),
        ("🌱 Fertilizer Guidance", info.get("fertilizer", "N/A")),
        ("🐛 Pest & Disease Management", info.get("pest_control", "N/A")),
        ("👨‍🌾 Farmer Tips", info.get("farmer_tips", "N/A")),
    ]

    for title, content in report_sections:
        st.markdown(f"""
        <div class="product-card">
            <h3>{html.escape(title)}</h3>
            <div class="card-muted">{html.escape(content)}</div>
        </div>
        """, unsafe_allow_html=True)

    txt_dossier = generate_plain_text_report(p, info)

    st.download_button(
        "📥 Download Plant Health Report (.txt)",
        txt_dossier,
        file_name=f"PlantCare_Report_{p['condition'].replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ============================================================
# PAGE 4: EXPLORE CROPS (35 Crops Catalog)
# ============================================================
def render_crop_directory():
    st.markdown("## 🌱 Explore Crops")
    st.caption("Complete directory of 35 vegetable crops across standard botanical categories.")

    plant_db = get_crops_database()

    for cat_name, crops in plant_db.items():
        with st.expander(f"{cat_name} ({len(crops)} Crops)", expanded=True):
            cols = st.columns(3)
            for idx, crop in enumerate(crops):
                col = cols[idx % 3]
                with col:
                    if crop.get("ai_supported"):
                        badge_style = "status-healthy"
                    elif "Planned" in crop.get("status", ""):
                        badge_style = "status-warning"
                    else:
                        badge_style = "status-danger"

                    st.markdown(f"""
                    <div style="background:#ffffff; border:1px solid #e2ece6; border-radius:16px; padding:1.2rem; margin-bottom:0.95rem; box-shadow:0 3px 8px rgba(0,0,0,0.02);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <strong style="color:#064e3b; font-size:1.05rem;">{crop['name']}</strong>
                            <span class="status-badge {badge_style}" style="font-size:0.68rem; padding:0.25rem 0.6rem;">{crop['status']}</span>
                        </div>
                        <div style="font-size:0.78rem; color:#64748b; font-style:italic; margin-bottom:0.45rem;">{crop.get('scientific_name', '')}</div>
                        <p style="font-size:0.88rem; color:#52665a; margin:0; line-height:1.55;">{crop['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================
# PAGE 5: DYNAMIC 35-CROP DISEASE KNOWLEDGE HUB
# ============================================================
def render_knowledge_hub():
    st.markdown("## 📚 Disease Knowledge Hub")
    st.caption("Advanced Clinical Pathology Dossiers & Integrated Disease Management (IDM) Compendium.")

    diseases_db = get_diseases_database()
    crops_catalog = get_crops_database()

    # Dynamic extraction of all 35 crops
    crop_names_set = set()
    for cat, crop_list in crops_catalog.items():
        for c in crop_list:
            crop_names_set.add(c["name"])

    for d_name, d_info in diseases_db.items():
        if d_info.get("crop"):
            crop_names_set.add(d_info.get("crop"))

    all_crop_options = ["All Crops"] + sorted(list(crop_names_set))

    col_crop_sel, col_disease_sel = st.columns([1, 1.5])
    with col_crop_sel:
        crop_filter = st.selectbox(
            "Filter by Crop",
            all_crop_options,
            key="crop_knowledge_filter"
        )

    # Filter diseases dynamically
    if crop_filter == "All Crops":
        filtered_diseases = list(diseases_db.keys())
    else:
        filtered_diseases = [
            d_name for d_name, d_info in diseases_db.items()
            if d_info.get("crop") == crop_filter or crop_filter.lower() in d_info.get("crop", "").lower()
        ]

    if not filtered_diseases:
        st.info(f"💡 **{crop_filter}:** Detailed disease pathology profiles are currently being compiled for this crop. Check back soon or view general crop details in 'Explore Crops'.")
        return

    with col_disease_sel:
        selected_condition = st.selectbox(
            "Select Pathological Condition to Inspect",
            filtered_diseases,
            key="condition_knowledge_selector"
        )

    info = get_disease_detail(selected_condition)

    st.markdown(f"""
    <div class="result-panel">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <span style="font-size: 0.88rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b;">
                    Botanical Specimen: <em>{info.get('scientific_crop', info.get('crop', 'Crop'))}</em>
                </span>
                <div style="font-size: 0.88rem; color: #047857; font-weight: 750; margin-top: 0.25rem;">
                    Pathogen Taxon: {info.get('pathogen', 'N/A')}
                </div>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="status-badge {info.get('badge', 'status-warning')}">● {info.get('category', 'Category')}</span>
                <span class="status-badge status-warning" style="background:#f1f5f9; color:#334155; border:1px solid #cbd5e1;">Severity: {info.get('severity', 'Moderate')}</span>
            </div>
        </div>
        <div style="font-size: 2.1rem; font-weight: 850; color: #0d1f17; margin-bottom: 0.65rem; font-family: 'Space Grotesk', sans-serif;">
            {selected_condition}
        </div>
        <div style="font-size: 1rem; color: #334155; line-height: 1.72;">
            {info.get('overview', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Etiology & Symptoms",
        "💊 Integrated Chemical & Bio-Control",
        "🛡️ Prevention & Cultural Control",
        "🌱 Nutrition & Thresholds"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="product-card">
                <h3>🧬 Pathogen Etiology & Life Cycle</h3>
                <div class="card-muted">{info.get('etiology', info.get('causes', 'N/A'))}</div>
            </div>
            <div class="product-card">
                <h3>⚠️ Environmental Pre-disposing Factors</h3>
                <div class="card-muted">
                    {info.get('causes', 'N/A')}
                    <br><br>
                    <strong>Optimal Climate Conditions:</strong><br>
                    <span style="color:#064e3b; font-weight:750;">{info.get('ideal_climate', 'Warm, humid weather with prolonged foliar moisture')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="product-card">
                <h3>🤒 Symptomatology & Stage Diagnosis</h3>
                <div class="card-muted" style="white-space: pre-line;">{info.get('symptoms', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"""
            <div class="product-card">
                <h3>🧪 Chemical Formulations & Doses</h3>
                <div class="card-muted" style="white-space: pre-line;">{info.get('chemical_treatment', info.get('treatment', 'N/A'))}</div>
                <div style="font-size:0.82rem; color:#b91c1c; margin-top:0.85rem; font-weight:600;">
                    * Follow local agrochemical regulations and strictly respect Pre-Harvest Intervals (PHI).
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="product-card">
                <h3>🌿 Biological & Bio-Pesticide Regimes</h3>
                <div class="card-muted" style="white-space: pre-line;">{info.get('organic_treatment', 'Maintain foliar sprays of certified biological antagonists such as Bacillus subtilis or Trichoderma species.')}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        c5, c6 = st.columns(2)
        with c5:
            st.markdown(f"""
            <div class="product-card">
                <h3>🛡️ Preventative Agronomic Protocol</h3>
                <div class="card-muted" style="white-space: pre-line;">{info.get('prevention', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="product-card">
                <h3>🐛 Vector & Alternate Host Management</h3>
                <div class="card-muted">{info.get('pest_control', 'N/A')}</div>
            </div>
            <div class="product-card">
                <h3>👨‍🌾 Operational Field Recommendations</h3>
                <div class="card-muted">{info.get('farmer_tips', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        c7, c8 = st.columns(2)
        with c7:
            st.markdown(f"""
            <div class="product-card">
                <h3>🌱 Nutrient Modulation & Soil Health</h3>
                <div class="card-muted">{info.get('fertilizer', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c8:
            st.markdown(f"""
            <div class="product-card">
                <h3>📊 Economic Threshold Level (ETL)</h3>
                <div class="card-muted">
                    {info.get('economic_threshold', 'Initiate chemical interventions upon observing 5% foliar canopy damage across the field.')}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE 6: FARMER STORIES
# ============================================================
def render_farmer_stories():
    st.markdown("## 🌾 Farmer Stories")
    st.caption("Real crop-protection experiences and field management lessons shared by farmers.")

    stories = get_farmer_stories()
    active_stories = [s for s in stories if s.get("status") == "active"]
    active_stories.sort(key=lambda x: x.get("priority", 99))

    if not active_stories:
        st.info("No active farmer stories available. You can add new stories from the Content Manager.")
        return

    cols = st.columns(2, gap="large")
    for idx, st_item in enumerate(active_stories):
        col = cols[idx % 2]
        with col:
            st.markdown(f"""
            <div class="product-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <div>
                        <h3 style="margin-bottom: 0.2rem;">{html.escape(st_item.get('farmer_name', 'Farmer'))}</h3>
                        <div style="font-size: 0.88rem; color: #047857; font-weight: 750;">
                            📍 {html.escape(st_item.get('location', ''))}, {html.escape(st_item.get('state', ''))}
                        </div>
                    </div>
                    <span class="status-badge status-healthy" style="font-size: 0.76rem;">
                        {html.escape(st_item.get('crop', 'Crop'))}
                    </span>
                </div>
                <div style="font-size: 0.94rem; color: #334155; line-height: 1.65; margin-bottom: 0.85rem;">
                    {html.escape(st_item.get('short_description', ''))}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st_item.get("image"):
                img_path = BASE_DIR / st_item["image"]
                if img_path.is_file():
                    st.image(str(img_path), use_container_width=True)

            with st.expander(f"📖 Read Full Experience ({st_item.get('farmer_name')})"):
                st.markdown(st_item.get("story", "No detailed story recorded."))
                if st_item.get("contact_cta"):
                    st.caption(f"Contact / Network: {st_item.get('contact_cta')}")

# ============================================================
# PAGE 7: WEATHER & SPRAY ADVISORY
# ============================================================
def render_weather_advisory():
    st.markdown("## 🌦️ Weather & Spray Advisory")
    st.caption("Dynamic microclimate assessment calculating disease infection risks and spraying windows.")

    advisory_rules = get_advisory_database()

    st.markdown("""
    <div class="product-card">
        <h3>Location & Microclimate Configuration</h3>
        <div class="card-muted">
            Enable your location or enter coordinates to fetch live weather parameters and automatically recalculate advisories.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_w1, c_w2 = st.columns([1, 2], gap="large")

    with c_w1:
        st.markdown("#### Coordinates Input")
        use_live = st.checkbox("Fetch Live Weather API", value=True)
        lat = st.number_input("Latitude", value=25.5941, format="%.4f")
        lon = st.number_input("Longitude", value=85.1376, format="%.4f")

    with c_w2:
        weather_data = None
        if use_live:
            with st.spinner("Connecting to live meteorological service..."):
                weather_data, err = get_live_weather_data(lat, lon)
                if err:
                    st.warning(f"⚠️ {err}. Using default parameters.")

        if not weather_data:
            st.info("Operating in manual parameter mode.")
            t_col, h_col, w_col = st.columns(3)
            with t_col:
                temp_in = st.slider("Temperature (°C)", 5, 45, 25)
            with h_col:
                hum_in = st.slider("Humidity (%)", 10, 100, 75)
            with w_col:
                wind_in = st.slider("Wind Speed (km/h)", 0, 50, 10)
            weather_data = {
                "temperature_c": temp_in,
                "relative_humidity_pct": hum_in,
                "rain_probability_pct": 15.0,
                "wind_speed_kmh": wind_in,
                "source": "Manual Input / Default Fallback"
            }

        st.markdown(f"**Data Source:** `{weather_data['source']}`")
        m_w1, m_w2, m_w3, m_w4 = st.columns(4)
        with m_w1:
            st.metric("Temperature", f"{weather_data['temperature_c']:.1f}°C")
        with m_w2:
            st.metric("Humidity", f"{weather_data['relative_humidity_pct']:.0f}%")
        with m_w3:
            st.metric("Rain Chance", f"{weather_data['rain_probability_pct']:.0f}%")
        with m_w4:
            st.metric("Wind Speed", f"{weather_data['wind_speed_kmh']:.1f} km/h")

    st.markdown("---")
    st.markdown("### 📊 Dynamic Agronomic Recalculation")

    spray_status, spray_reasons = calculate_spray_advisory(
        weather_data["temperature_c"],
        weather_data["relative_humidity_pct"],
        weather_data["rain_probability_pct"],
        weather_data["wind_speed_kmh"],
        advisory_rules
    )

    disease_risks = calculate_disease_risks(
        weather_data["temperature_c"],
        weather_data["relative_humidity_pct"],
        advisory_rules
    )

    c_adv1, c_adv2 = st.columns(2)
    with c_adv1:
        st.markdown(f"""
        <div class="product-card">
            <h3>Spray Suitability Status</h3>
            <div style="font-size: 1.22rem; font-weight: 850; color: {'#064e3b' if 'Favorable' in spray_status else '#991b1b'}; margin-bottom: 0.5rem;">
                {spray_status}
            </div>
            <div class="card-muted">
                {'<br>'.join(['• ' + r for r in spray_reasons]) if spray_reasons else 'All microclimate thresholds (wind, rain chance, temperature, humidity) are currently within safe application limits.'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_adv2:
        st.markdown(f"""
        <div class="product-card">
            <h3>Microclimate Disease Pressures</h3>
            <div class="card-muted">
                {'<br>'.join(['• ' + r for r in disease_risks])}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 8: NEARBY PLANT CARE
# ============================================================
def render_nearby_page():
    st.markdown("## 📍 Nearby Plant Care")
    st.caption("Discover nearby agricultural stores, seed centers, and plant nurseries.")

    st.markdown("""
    <div class="product-card">
        <h3>Location-Based Resource Discovery</h3>
        <div class="card-muted">
            Enable your location to receive local weather, spray advisory and nearby plant-care recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_loc_ctrl, c_loc_view = st.columns([1, 1.3], gap="large")

    with c_loc_ctrl:
        st.markdown("""
        <div class="product-card">
            <h3>Location Settings</h3>
            <div class="card-muted" style="margin-bottom: 1rem;">
                Set coordinates to discover agricultural services in your immediate radius.
            </div>
        """, unsafe_allow_html=True)

        use_loc = st.checkbox("Enable Location Access", value=st.session_state.get("loc_enabled", False))
        st.session_state["loc_enabled"] = use_loc

        if use_loc:
            c_lat, c_lon = st.columns(2)
            with c_lat:
                lat = st.number_input("Latitude", value=25.5941, format="%.4f", key="near_lat")
            with c_lon:
                lon = st.number_input("Longitude", value=85.1376, format="%.4f", key="near_lon")
            st.session_state.user_coords = {"lat": lat, "lon": lon}
            st.success("Location connected.")
        else:
            st.session_state.user_coords = None
            st.info("Location access is disabled. You can enable it above or search nearby services manually.")

        st.markdown("</div>", unsafe_allow_html=True)

    with c_loc_view:
        coords = st.session_state.user_coords
        if coords:
            lat = coords["lat"]
            lon = coords["lon"]

            if st.button("🔎 Discover Nearby Agri-Centers", type="primary", use_container_width=True):
                with st.spinner("Connecting to geospatial mapping directory..."):
                    try:
                        st.session_state.nearby_shops = query_nearby_plant_care(lat, lon, limit=8)
                    except Exception:
                        st.session_state.nearby_shops = []

            if st.session_state.nearby_shops is not None:
                shops = st.session_state.nearby_shops
                if shops:
                    st.markdown("### Nearby Verified Centers")
                    for s in shops:
                        st.markdown(f"""
                        <div class="product-card">
                            <h3 style="font-size: 1.15rem; margin-bottom: 0.35rem;">🏪 {html.escape(s['name'])}</h3>
                            <div class="card-muted">
                                <strong>Category:</strong> {html.escape(s['type'])}<br>
                                <strong>Address:</strong> {html.escape(s['address'])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.link_button("🗺️ Open in Google Maps", s["maps"])
                else:
                    st.warning("No mapped agricultural stores found in the immediate search radius.")
                    st.link_button(
                        "🔎 Open in Google Maps Search",
                        maps_query_url(lat, lon, "plant nursery agricultural supply fertilizer seeds pesticide"),
                    )
        else:
            st.markdown("""
            <div class="product-card" style="text-align: center; padding: 2.8rem 1.5rem;">
                <div style="font-size: 2.8rem; margin-bottom: 0.6rem;">🗺️</div>
                <h3>Locator Ready</h3>
                <div class="card-muted">
                    Enable location access on the left to locate certified suppliers.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE 9: CONTENT MANAGER (Admin UI)
# ============================================================
def render_content_manager():
    st.markdown("## ⚙️ Content Manager")
    st.caption("Manage Farmer Stories and Sponsored Ads dynamically without modifying application code.")

    tab_story, tab_ads = st.tabs(["🌾 Manage Farmer Stories", "📢 Manage Advertisements"])

    with tab_story:
        st.markdown("### Add New Farmer Story")
        with st.form("add_farmer_story_form", clear_on_submit=True):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                f_name = st.text_input("Farmer Name", placeholder="e.g., Sanjay Singh")
                f_loc = st.text_input("Location / Village", placeholder="e.g., Kaithma")
                f_state = st.text_input("State", placeholder="e.g., Bihar, India")
            with col_f2:
                f_crop = st.text_input("Crop", placeholder="e.g., Tomato (Solanaceae)")
                f_priority = st.number_input("Display Priority (1 = Top)", min_value=1, value=1)
                f_status = st.selectbox("Status", ["active", "inactive"])

            f_short = st.text_area("Short Summary", placeholder="Brief highlight of the outcome...")
            f_story = st.text_area("Full Experience / Story", placeholder="Complete background, treatment, and results...")
            f_cta = st.text_input("Contact / Network Label (Optional)", placeholder="e.g., Farmer Producer Org Contact")
            f_img_file = st.file_uploader("Upload Farmer Photo (Optional)", type=["jpg", "jpeg", "png", "webp"], key="f_img_up")

            submitted_story = st.form_submit_button("Save Farmer Story", type="primary")

            if submitted_story:
                if not f_name or not f_short or not f_story:
                    st.error("Please fill in the required fields (Name, Short Summary, Story).")
                else:
                    img_rel_path = ""
                    if f_img_file is not None:
                        img_rel_path = save_uploaded_asset(f_img_file, FARMER_IMAGES_DIR, prefix="farmer")

                    new_story = {
                        "id": f"story_{uuid.uuid4().hex[:6]}",
                        "farmer_name": f_name,
                        "location": f_loc,
                        "state": f_state,
                        "crop": f_crop,
                        "image": img_rel_path,
                        "short_description": f_short,
                        "story": f_story,
                        "contact_cta": f_cta,
                        "status": f_status,
                        "priority": int(f_priority),
                        "date": datetime.now().strftime("%Y-%m-%d")
                    }

                    current_stories = get_farmer_stories()
                    current_stories.append(new_story)
                    save_json_file(DATA_DIR / "farmer_stories.json", current_stories)
                    st.success("Farmer story saved successfully! The UI will update automatically.")

        st.markdown("---")
        st.markdown("### Existing Stories")
        stories = get_farmer_stories()
        for idx, s in enumerate(stories):
            with st.expander(f"{s.get('farmer_name')} ({s.get('crop')}) — Status: {s.get('status')}"):
                st.write(s)
                if st.button(f"Delete Story #{idx+1}", key=f"del_story_{idx}"):
                    stories.pop(idx)
                    save_json_file(DATA_DIR / "farmer_stories.json", stories)
                    st.rerun()

    with tab_ads:
        st.markdown("### Add New Advertisement / Partner")
        with st.form("add_ad_form", clear_on_submit=True):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                ad_title = st.text_input("Title / Heading", placeholder="e.g., Smart Solar Drip Kit")
                ad_company = st.text_input("Company / Brand", placeholder="e.g., SEA AUTO")
                ad_category = st.text_input("Category", placeholder="e.g., Agri-Tech")
            with col_a2:
                ad_btn_text = st.text_input("Button Text", value="Learn More")
                ad_btn_url = st.text_input("Button URL", value="https://example.com")
                ad_priority = st.number_input("Display Priority (1 = Highest)", min_value=1, value=1, key="ad_prio")

            ad_desc = st.text_area("Description", placeholder="Clear summary of the sponsored service...")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                ad_status = st.selectbox("Status", ["active", "inactive"], key="ad_stat")
            with col_d2:
                ad_start = st.date_input("Start Date", value=datetime.today())
            with col_d3:
                ad_end = st.date_input("End Date", value=datetime(2026, 12, 31))

            ad_img_file = st.file_uploader("Upload Ad Banner Image (Optional)", type=["jpg", "jpeg", "png", "webp"], key="ad_img_up")
            submitted_ad = st.form_submit_button("Save Advertisement", type="primary")

            if submitted_ad:
                if not ad_title or not ad_desc:
                    st.error("Please fill in required fields (Title, Description).")
                else:
                    img_rel_path = ""
                    if ad_img_file is not None:
                        img_rel_path = save_uploaded_asset(ad_img_file, AD_IMAGES_DIR, prefix="ad")

                    new_ad = {
                        "id": f"ad_{uuid.uuid4().hex[:6]}",
                        "title": ad_title,
                        "company": ad_company,
                        "image": img_rel_path,
                        "description": ad_desc,
                        "category": ad_category,
                        "button_text": ad_btn_text,
                        "button_url": ad_btn_url,
                        "status": ad_status,
                        "priority": int(ad_priority),
                        "start_date": ad_start.strftime("%Y-%m-%d"),
                        "end_date": ad_end.strftime("%Y-%m-%d")
                    }

                    current_ads = get_advertisements()
                    current_ads.append(new_ad)
                    save_json_file(DATA_DIR / "advertisements.json", current_ads)
                    st.success("Advertisement saved successfully! Active ads will appear dynamically on the Home page.")

        st.markdown("---")
        st.markdown("### Existing Advertisements")
        all_ads = get_advertisements()
        for idx, a in enumerate(all_ads):
            with st.expander(f"{a.get('title')} ({a.get('company')}) — Status: {a.get('status')}"):
                st.write(a)
                if st.button(f"Delete Ad #{idx+1}", key=f"del_ad_{idx}"):
                    all_ads.pop(idx)
                    save_json_file(DATA_DIR / "advertisements.json", all_ads)
                    st.rerun()

# ============================================================
# PAGE 10: ABOUT PLANTCARE AI
# ============================================================
def render_about_page():
    st.markdown("""
    <div class="product-card">
        <div class="hero-kicker" style="background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0;">
            ✦ ABOUT PLANTCARE AI
        </div>
        <h1 style="color: #064e3b; font-size: 2.3rem; margin: 0.5rem 0 0.85rem; font-family: 'Space Grotesk', sans-serif;">About PlantCare AI</h1>
        <p class="card-muted" style="font-size: 1.08rem;">
            PlantCare AI is an AI-powered plant health screening and crop protection engine developed by <strong>Madhav Kumar</strong> under <strong>SEA AUTO</strong> to help growers, gardeners, 
            and agricultural specialists understand visible plant-health problems from leaf and fruit images and receive actionable agronomic guidance.
        </p>
        <p class="card-muted">
            The platform provides AI-assisted visual screening, comprehensive clinical disease profiles across 35 agricultural crops, preventative cultural schedules, 
            and real-time meteorological spray advisories.
        </p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.35rem; margin-top: 1.2rem;">
        <div class="product-card">
            <h3>Product Mission</h3>
            <div class="card-muted">
                To make early visual plant disease assessment accessible, instant, and straightforward for farmers, growers, and agronomists worldwide.
            </div>
        </div>
        <div class="product-card">
            <h3>AI-Assisted Vision</h3>
            <div class="card-muted">
                High-precision deep learning image screening delivering transparent probability metrics and immediate management steps.
            </div>
        </div>
        <div class="product-card">
            <h3>SEA AUTO Ecosystem</h3>
            <div class="card-muted">
                PlantCare AI is part of SEA AUTO's technology ecosystem focused on building scalable, practical real-world agricultural solutions.
            </div>
        </div>
    </div>

    <div class="product-card" style="margin-top: 1.4rem; text-align: center;">
        <div style="font-size: 1.35rem; font-weight: 850; color: #064e3b; margin-bottom: 0.35rem;">
            Developed by Madhav Kumar
        </div>
        <div style="font-size: 1rem; font-weight: 750; color: #059669; margin-bottom: 0.25rem;">
            ✦ Powered by SEA AUTO
        </div>
        <div style="font-size: 0.88rem; color: #64748b;">
            Committed to accessible, intelligent agricultural technology solutions.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APPLICATION ROUTER
# ============================================================
def main():
    inject_custom_css()
    current_page = render_sidebar()

    if current_page == "🏠 Home":
        render_home_page()
    elif current_page == "🔬 Disease Detection":
        render_detection_page()
    elif current_page == "📄 Plant Health Report":
        render_report_page()
    elif current_page == "🌱 Explore Crops":
        render_crop_directory()
    elif current_page == "📚 Disease Knowledge Hub":
        render_knowledge_hub()
    elif current_page == "🌾 Farmer Stories":
        render_farmer_stories()
    elif current_page == "🌦️ Weather & Spray Advisory":
        render_weather_advisory()
    elif current_page == "📍 Nearby Plant Care":
        render_nearby_page()
    elif current_page == "⚙️ Content Manager":
        render_content_manager()
    elif current_page == "ℹ️ About PlantCare AI":
        render_about_page()

    # Consumer Footer
    st.markdown("""
    <div class="app-footer-bar">
        <div>
            <span class="footer-brand">PlantCare AI</span>
            <span>· Powered by SEA AUTO</span>
        </div>
        <div>Developed by <strong>Madhav Kumar</strong> &nbsp;|&nbsp; © 2026 PlantCare AI. All rights reserved.</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()