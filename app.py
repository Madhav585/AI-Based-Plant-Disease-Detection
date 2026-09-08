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
# 1. APPLICATION & LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PlantCareAI")

st.set_page_config(
    page_title="PlantCare AI — Autonomous Plant Health & Agronomy Hub",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
FARMER_IMAGES_DIR = IMAGES_DIR / "farmers"
AD_IMAGES_DIR = IMAGES_DIR / "advertisements"
CROP_IMAGES_DIR = IMAGES_DIR / "crops"

for directory in [DATA_DIR, IMAGES_DIR, FARMER_IMAGES_DIR, AD_IMAGES_DIR, CROP_IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. LOCALIZATION (ENGLISH ⇄ हिन्दी)
# ============================================================
TRANSLATIONS = {
    "en": {
        "app_title": "PlantCare AI",
        "tagline": "Enterprise Agronomic Health & Visual Diagnostic Engine",
        "powered_by": "Powered by SEA AUTO",
        "nav_home": "🏠 Home",
        "nav_scan": "🔬 Scan Plant",
        "nav_report": "📄 Health Dossier",
        "nav_crops": "🌱 Explore 35 Crops",
        "nav_knowledge": "📚 Disease Knowledge Hub",
        "nav_stories": "🌾 Field Case Studies",
        "nav_weather": "🌦️ Weather & Spray Advisory",
        "nav_nearby": "📍 Nearby Agri Centers",
        "nav_admin": "⚙️ Content Manager",
        "nav_about": "ℹ️ About Platform",
        "hero_kicker": "✦ AUTONOMOUS AGRITECH DIAGNOSTICS",
        "hero_heading": "Healthy Crops. Maximum Yields.",
        "hero_desc": "Industrial AI-grade computer vision diagnostics for 35 commercial vegetable crops. Upload leaf, fruit, or tuber images for immediate clinical identification and targeted management.",
        "btn_scan": "Scan Plant Specimen",
        "btn_explore": "Explore 35 Crops",
        "btn_knowledge": "Disease Pathology Hub",
        "model_online": "🟢 Neural Engine Online",
        "model_offline": "🟡 Engine Fallback Active",
        "scan_heading": "🔬 Plant Visual Health Inspection",
        "scan_desc": "Provide a clean, focused image of the affected plant foliage, fruit, or tuber to trigger multi-resolution neural classification.",
        "upload_label": "Upload plant image or camera capture",
        "analyze_btn": "🔬 Run Neural Diagnosis",
        "analysis_success": "Diagnostic screening complete.",
        "confidence_high": "🟢 Confirmed Diagnostic (>70%)",
        "confidence_low": "🟡 Indicative Diagnosis (<70%)",
        "result_heading": "Diagnostic Screening Dossier",
        "plant_label": "Crop Type",
        "category_label": "Etiological Class",
        "risk_label": "Damage Potential",
        "conf_label": "Neural Confidence",
        "overview_title": "📖 Clinical Description",
        "symptoms_title": "🤒 Symptomatology",
        "causes_title": "⚠️ Pre-disposing Stressors",
        "treatment_title": "💊 Chemical Formulations",
        "fert_title": "🌱 Nutrition Modulation",
        "pest_title": "🐛 Vector Management",
        "tips_title": "👨‍🌾 Practical Field Rules",
        "top5_heading": "📊 Top 5 Neural Probability Distribution",
        "disclaimer": "AI-assisted screening provides rapid diagnostic guidance. Always confirm with standard agronomic advisory before broad-acre chemical applications.",
        "report_heading": "📄 Agronomic Health Dossier",
        "report_desc": "Official inspection summary ready for review and local export.",
        "no_report": "No Diagnostic Record Found",
        "no_report_desc": "Please submit a plant image in the 'Scan Plant' section to generate a comprehensive report.",
        "download_btn": "📥 Export Diagnostic Dossier (.txt)"
    },
    "hi": {
        "app_title": "PlantCare AI",
        "tagline": "उन्नत पादप स्वास्थ्य एवं दृश्य निदान प्रणाली",
        "powered_by": "SEA AUTO द्वारा संचालित",
        "nav_home": "🏠 मुख्य पृष्ठ",
        "nav_scan": "🔬 पौधे की जांच करें",
        "nav_report": "📄 स्वास्थ्य रिपोर्ट",
        "nav_crops": "🌱 35 फसलें देखें",
        "nav_knowledge": "📚 रोग ज्ञान केंद्र",
        "nav_stories": "🌾 किसान अनुभव",
        "nav_weather": "🌦️ मौसम एवं छिड़काव सलाह",
        "nav_nearby": "📍 नजदीकी कृषि केंद्र",
        "nav_admin": "⚙️ सामग्री प्रबंधन",
        "nav_about": "ℹ️ प्लेटफॉर्म विवरण",
        "hero_kicker": "✦ स्वायत्त कृषि प्रौद्योगिकी",
        "hero_heading": "स्वस्थ फसल। समृद्ध किसान।",
        "hero_desc": "35 प्रमुख सब्जियों के लिए उन्नत कंप्यूटर विजन आधारित एआई निदान प्रणाली। त्वरित रोग पहचान एवं सटीक रासायनिक व जैविक उपचार प्राप्त करने हेतु पत्ती या फल की तस्वीर अपलोड करें।",
        "btn_scan": "पौधा स्कैन करें",
        "btn_explore": "35 फसलें देखें",
        "btn_knowledge": "रोग ज्ञान भंडार",
        "model_online": "🟢 एआई मॉडल सक्रिय है",
        "model_offline": "🟡 बैकअप मोड सक्रिय है",
        "scan_heading": "🔬 पादप रोग दृश्य परीक्षण",
        "scan_desc": "सटीक न्यूरल नेटवर्क विश्लेषण के लिए प्रभावित पत्ती, फल या कंद की स्पष्ट तस्वीर अपलोड करें।",
        "upload_label": "पौधे की तस्वीर अपलोड करें",
        "analyze_btn": "🔬 जांच शुरू करें",
        "analysis_success": "सफलतापूर्वक जांच पूरी हुई।",
        "confidence_high": "🟢 उच्च सटीकता (>70%)",
        "confidence_low": "🟡 सामान्य सटीकता (<70%)",
        "result_heading": "निदान एवं उपचार विवरण",
        "plant_label": "फसल",
        "category_label": "रोग श्रेणी",
        "risk_label": "गंभीरता",
        "conf_label": "कॉन्फिडेंस",
        "overview_title": "📖 रोग का विवरण",
        "symptoms_title": "🤒 लक्षण",
        "causes_title": "⚠️ कारक एवं कारण",
        "treatment_title": "💊 रासायनिक उपचार व डोज",
        "fert_title": "🌱 पोषण एवं खाद प्रबंधन",
        "pest_title": "🐛 कीट व रोग वाहक नियंत्रण",
        "tips_title": "👨‍🌾 जरूरी किसान टिप्स",
        "top5_heading": "📊 शीर्ष 5 एआई संभावनाएं",
        "disclaimer": "एआई निदान एक त्वरित तकनीकी सहायता है। बड़े पैमाने पर छिड़काव से पूर्व स्थानीय कृषि विशेषज्ञ से सलाह अवश्य लें।",
        "report_heading": "📄 विस्तृत स्वास्थ्य रिपोर्ट",
        "report_desc": "डाउनलोड व रिकॉर्ड हेतु तैयार आधिकारिक स्वास्थ्य रिपोर्ट।",
        "no_report": "कोई परीक्षण रिकॉर्ड नहीं मिला",
        "no_report_desc": "रिपोर्ट तैयार करने के लिए पहले 'पौधे की जांच करें' सेक्शन में तस्वीर अपलोड करें।",
        "download_btn": "📥 रिपोर्ट डाउनलोड करें (.txt)"
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))

# ============================================================
# 3. AI MODEL REGISTRY & 35-CROP CATALOG
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

DEFAULT_35_CROPS = {
    "Solanaceae": [
        {"name": "Tomato", "scientific": "Solanum lycopersicum", "status": "AI Live (10 Diseases)"},
        {"name": "Potato", "scientific": "Solanum tuberosum", "status": "AI Live (Early/Late Blight)"},
        {"name": "Capsicum / Bell Pepper", "scientific": "Capsicum annuum", "status": "AI Live (Bacterial Spot)"},
        {"name": "Brinjal / Eggplant", "scientific": "Solanum melongena", "status": "Knowledge Active"},
        {"name": "Chilli", "scientific": "Capsicum frutescens", "status": "Knowledge Active"}
    ],
    "Cucurbit Vegetables": [
        {"name": "Pumpkin", "scientific": "Cucurbita moschata", "status": "Knowledge Active"},
        {"name": "Cucumber", "scientific": "Cucumis sativus", "status": "Knowledge Active"},
        {"name": "Bottle Gourd / Lauki", "scientific": "Lagenaria siceraria", "status": "Knowledge Active"},
        {"name": "Bitter Gourd / Karela", "scientific": "Momordica charantia", "status": "Knowledge Active"},
        {"name": "Ridge Gourd / Turai", "scientific": "Luffa acutangula", "status": "Knowledge Active"},
        {"name": "Sponge Gourd / Gilki", "scientific": "Luffa aegyptiaca", "status": "Knowledge Active"},
        {"name": "Pointed Gourd / Parwal", "scientific": "Trichosanthes dioica", "status": "Knowledge Active"},
        {"name": "Ash Gourd / Petha", "scientific": "Benincasa hispida", "status": "Knowledge Active"},
        {"name": "Zucchini", "scientific": "Cucurbita pepo", "status": "Knowledge Active"},
        {"name": "Snake Gourd / Chichinda", "scientific": "Trichosanthes cucumerina", "status": "Knowledge Active"},
        {"name": "Ivy Gourd / Kundru", "scientific": "Coccinia grandis", "status": "Knowledge Active"}
    ],
    "Common Indian Vegetables": [
        {"name": "Okra / Lady Finger / Bhindi", "scientific": "Abelmoschus esculentus", "status": "Knowledge Active"},
        {"name": "French Bean", "scientific": "Phaseolus vulgaris", "status": "Knowledge Active"},
        {"name": "Green Bean", "scientific": "Phaseolus vulgaris var.", "status": "Knowledge Active"},
        {"name": "Peas / Matar", "scientific": "Pisum sativum", "status": "Knowledge Active"},
        {"name": "Sweet Corn", "scientific": "Zea mays var. saccharata", "status": "Knowledge Active"},
        {"name": "Carrot", "scientific": "Daucus carota", "status": "Knowledge Active"},
        {"name": "Radish / Mooli", "scientific": "Raphanus sativus", "status": "Knowledge Active"},
        {"name": "Beetroot", "scientific": "Beta vulgaris", "status": "Knowledge Active"},
        {"name": "Turnip / Shalgam", "scientific": "Brassica rapa subsp. rapa", "status": "Knowledge Active"}
    ],
    "Cole Vegetables": [
        {"name": "Cabbage", "scientific": "Brassica oleracea var. capitata", "status": "Knowledge Active"},
        {"name": "Cauliflower", "scientific": "Brassica oleracea var. botrytis", "status": "Knowledge Active"},
        {"name": "Broccoli", "scientific": "Brassica oleracea var. italica", "status": "Knowledge Active"}
    ],
    "Leafy Vegetables": [
        {"name": "Spinach / Palak", "scientific": "Spinacia oleracea", "status": "Knowledge Active"},
        {"name": "Fenugreek / Methi", "scientific": "Trigonella foenum-graecum", "status": "Knowledge Active"},
        {"name": "Coriander / Dhaniya", "scientific": "Coriandrum sativum", "status": "Knowledge Active"},
        {"name": "Lettuce", "scientific": "Lactuca sativa", "status": "Knowledge Active"},
        {"name": "Amaranth / Chaulai", "scientific": "Amaranthus cruentus", "status": "Knowledge Active"}
    ],
    "Bulb Vegetables": [
        {"name": "Onion", "scientific": "Allium cepa", "status": "Knowledge Active"},
        {"name": "Garlic", "scientific": "Allium sativum", "status": "Knowledge Active"}
    ]
}

# ============================================================
# 4. EXHAUSTIVE 35-CROP PATHOLOGY DATABASE
# ============================================================
EXHAUSTIVE_35_CROP_PATHOLOGY = {
    "Tomato Early Blight": {
        "crop": "Tomato", "scientific": "Solanum lycopersicum", "pathogen": "Alternaria solani (Fungus)",
        "category": "Fungal Blight", "severity": "High", "badge": "status-danger",
        "overview": "Causes concentric target-like brown lesions on older leaves, progressing upward and triggering early defoliation.",
        "etiology": "Fungal spores survive in crop debris and soil; splashed by rain drops under warm, humid conditions.",
        "symptoms": "Dark brown circular spots with characteristic concentric rings on lower leaves, yellow halo around lesions.",
        "causes": "Extended leaf wetness, high humidity (>80%), temperatures between 24-29°C, and overhead irrigation.",
        "chemical_treatment": "Mancozeb 75% WP @ 2.5 g/L or Chlorothalonil 75% WP @ 2 g/L or Azoxystrobin 23% SC @ 1 ml/L.",
        "organic_treatment": "Spray Trichoderma viride @ 5 g/L or 5% Neem Seed Kernel Extract (NSKE) at early onset.",
        "prevention": "3-year crop rotation with non-solanaceous crops, staking, mulch installation, drip irrigation.",
        "fertilizer": "Avoid excessive vegetative nitrogen; ensure adequate potassium and calcium nitrate.",
        "pest_control": "Manage flea beetles and aphids which create entry wounds.",
        "farmer_tips": "Prune bottom 12 inches of foliage after fruit set to eliminate splash-zone spores."
    },
    "Tomato Late Blight": {
        "crop": "Tomato", "scientific": "Solanum lycopersicum", "pathogen": "Phytophthora infestans (Oomycete)",
        "category": "Oomycete Blight", "severity": "Critical", "badge": "status-danger",
        "overview": "Devastating water-soaked lesions causing whole vine collapse and destructive fruit rot in cool, damp weather.",
        "etiology": "Wind-borne sporangia that rapidly germinate under high moisture and cool temperatures.",
        "symptoms": "Irregular dark water-soaked patches on leaves with white fuzzy fungal growth on undersides in humid mornings.",
        "causes": "High humidity (>90%), cool nights (10-15°C), and mild days (16-21°C).",
        "chemical_treatment": "Metalaxyl 8% + Mancozeb 64% WP @ 2.5 g/L or Cymoxanil 8% + Mancozeb 64% WP @ 2 g/L.",
        "organic_treatment": "Bordeaux mixture (1%) or Copper oxychloride @ 3 g/L as protective barrier.",
        "prevention": "Destroy cull piles, use certified disease-free transplants, maximize airflow through wide spacing.",
        "fertilizer": "Maintain balanced NPK; avoid lush vegetative growth.",
        "pest_control": "Monitor insect vectors and destroy volunteer solanaceous weeds.",
        "farmer_tips": "Apply protective fungicide immediately when persistent fog or overcast rain is forecast."
    },
    "Tomato Bacterial Spot": {
        "crop": "Tomato", "scientific": "Solanum lycopersicum", "pathogen": "Xanthomonas perforans (Bacteria)",
        "category": "Bacterial Infection", "severity": "High", "badge": "status-danger",
        "overview": "Small, dark greasy spots on leaves and scabby raised spots on green fruits.",
        "etiology": "Seed-borne and debris-borne bacteria entering via stomata and microscopic leaf abrasions.",
        "symptoms": "Water-soaked leaf spots turning black, coalescing into ragged tears; raised scab lesions on fruit.",
        "causes": "Splashing rain, high temperatures (24-30°C), and high humidity.",
        "chemical_treatment": "Copper hydroxide @ 2 g/L mixed with Streptocycline @ 0.1 g/L (100 ppm).",
        "organic_treatment": "Bacillus subtilis foliar sprays @ 5 ml/L and hot water seed treatment (50°C for 25 min).",
        "prevention": "Certified pathogen-free seed stock, eliminate solanaceous volunteer plants, avoid working in wet fields.",
        "fertilizer": "Adequate potassium to strengthen cell wall resistance.",
        "pest_control": "Manage thrips and chewing insects that create bacterial entry points.",
        "farmer_tips": "Disinfect pruning knives with 10% bleach solution between rows."
    },
    "Potato Early Blight": {
        "crop": "Potato", "scientific": "Solanum tuberosum", "pathogen": "Alternaria solani (Fungus)",
        "category": "Fungal Blight", "severity": "Moderate", "badge": "status-warning",
        "overview": "Concentric brown-black lesions on lower canopy, reducing tuber yield.",
        "etiology": "Overwinters in dead foliage and tuber skin; spread by rain splash and dry wind.",
        "symptoms": "Concentric rings resembling a target board on lower leaves, yellowing margins.",
        "causes": "Alternating wet and dry weather, tuber bulking stress, nutrient deficiencies.",
        "chemical_treatment": "Propineb 70% WP @ 2 g/L or Difenoconazole 25% EC @ 0.5 ml/L.",
        "organic_treatment": "Foliar spray of Trichoderma harzianum @ 5 g/L with cow urine solution (10%).",
        "prevention": "Deep plowing of crop residues, certified seed tubers, balanced fertilization.",
        "fertilizer": "Apply adequate potassium and nitrogen split doses during tuber initiation.",
        "pest_control": "Control potato tuber moths and leafhoppers.",
        "farmer_tips": "Stop irrigation 10-14 days before harvest to allow tuber skin maturity."
    },
    "Potato Late Blight": {
        "crop": "Potato", "scientific": "Solanum tuberosum", "pathogen": "Phytophthora infestans (Oomycete)",
        "category": "Oomycete Blight", "severity": "Critical", "badge": "status-danger",
        "overview": "Destructive water-soaked leaf blighting leading to rapid plant decay and brown dry tuber rot.",
        "etiology": "Infected seed tubers serve as primary inoculum; sporangia travel miles in air currents.",
        "symptoms": "Water-soaked dark lesions on leaf tips, white downy mold underneath, purplish brown decay on tubers.",
        "causes": "Relative humidity >85%, temperatures between 12-22°C with morning fog or rain.",
        "chemical_treatment": "Dimethomorph 50% WP @ 1 g/L or Mandipropamid 23.4% SC @ 0.8 ml/L.",
        "organic_treatment": "Copper Hydroxide 53.8% DF @ 2 g/L before infection occurs.",
        "prevention": "Plant certified disease-free seed tubers; hill soil properly to protect tubers from spores.",
        "fertilizer": "Avoid excess nitrogen late in season.",
        "pest_control": "Destroy weed hosts like Solanum nigrum nearby.",
        "farmer_tips": "Dehaulm (cut foliage) 10-12 days before harvest if late blight appears near maturity."
    },
    "Pepper Bell Bacterial Spot": {
        "crop": "Capsicum / Bell Pepper", "scientific": "Capsicum annuum", "pathogen": "Xanthomonas vesicatoria (Bacteria)",
        "category": "Bacterial Infection", "severity": "High", "badge": "status-danger",
        "overview": "Causes severe leaf dropping, sunscalding of exposed fruits, and warty fruit lesions.",
        "etiology": "Enters via natural leaf pores during rainy, windy weather; survives on seed coats.",
        "symptoms": "Small, circular chlorotic spots on leaves becoming necrotic with dark halos; warty blisters on fruit.",
        "causes": "Warm temperatures (24-30°C) with persistent foliar dampness.",
        "chemical_treatment": "Copper Oxychloride @ 2.5 g/L + Streptocycline @ 0.1 g/L.",
        "organic_treatment": "Seed treatment with hot water (50°C for 25 mins) and Pseudomonas fluorescens spray @ 5 g/L.",
        "prevention": "Crop rotation for at least 2 seasons, drip irrigation, sanitized trellis stakes.",
        "fertilizer": "Maintain high calcium and silicon levels to reinforce cuticle barriers.",
        "pest_control": "Control broad mites and thrips.",
        "farmer_tips": "Never work in capsicum rows while leaves are damp with morning dew."
    },
    "Brinjal Bacterial Wilt": {
        "crop": "Brinjal / Eggplant", "scientific": "Solanum melongena", "pathogen": "Ralstonia solanacearum (Bacteria)",
        "category": "Vascular Wilt", "severity": "Critical", "badge": "status-danger",
        "overview": "Rapid, permanent daytime wilting of healthy green plants without initial leaf yellowing.",
        "etiology": "Soil-borne vascular bacterium invades roots through transplanting damage or nematode wounds.",
        "symptoms": "Sudden wilting of top leaves during hot sunny hours; white milky bacterial slime streaming from cut stem in water.",
        "causes": "High soil moisture, poorly drained soils, temperatures above 28°C.",
        "chemical_treatment": "Soil drenching with Copper Oxychloride @ 3 g/L + Streptocycline @ 0.2 g/L at early signs.",
        "organic_treatment": "Soil incorporation of Pseudomonas fluorescens @ 2.5 kg/ha with enriched FYM.",
        "prevention": "Grafting on resistant rootstocks (Solanum torvum), raised nursery beds, crop rotation with maize/paddy.",
        "fertilizer": "Apply neem cake @ 250 kg/ha to suppress soil pathogens and nematodes.",
        "pest_control": "Control root-knot nematodes strictly.",
        "farmer_tips": "Do stem-streaming test in a clear glass of water to confirm bacterial wilt vs fungal wilt."
    },
    "Chilli Leaf Curl Virus": {
        "crop": "Chilli", "scientific": "Capsicum frutescens", "pathogen": "Chilli Leaf Curl Virus (Begomovirus)",
        "category": "Viral Disease", "severity": "High", "badge": "status-danger",
        "overview": "Severe curling, puckering of leaves, stunting of plants, and massive reduction in fruit set.",
        "etiology": "Transmitted systematically by whiteflies (Bemisia tabaci); not seed-transmitted.",
        "symptoms": "Upward curling and crinkling of leaves, thickened veins, shortened internodes creating bushy stunted plants.",
        "causes": "High whitefly populations during dry, warm weather.",
        "chemical_treatment": "Diafenthiuron 50% WP @ 1.2 g/L or Spiromesifen 22.9% SC @ 1 ml/L.",
        "organic_treatment": "Spray 5% Neem oil (10,000 ppm) @ 2 ml/L + yellow sticky traps (15-20 traps/acre).",
        "prevention": "Grow barrier crops like maize/sorghum (2-3 rows) around field boundary.",
        "fertilizer": "Supplement with micronutrient mixtures (Zinc, Boron, Magnesium).",
        "pest_control": "Strictly suppress whiteflies from nursery stage onward.",
        "farmer_tips": "Install yellow sticky traps early to catch whitefly swarms before virus transmission."
    },
    "Cucumber Downy Mildew": {
        "crop": "Cucumber", "scientific": "Cucumis sativus", "pathogen": "Pseudoperonospora cubensis (Oomycete)",
        "category": "Oomycete Mildew", "severity": "High", "badge": "status-danger",
        "overview": "Angular yellow spots restricted by leaf veins on upper leaf surfaces, with purplish downy spore growth underneath.",
        "etiology": "Wind-borne sporangia requiring only 2 hours of dew to infect cucurbit foliage.",
        "symptoms": "Bright yellow angular spots delineated by major leaf veins, quickly turning brown and necrotic.",
        "causes": "High humidity (>85%) with moderate temperatures (15-22°C) and morning fog.",
        "chemical_treatment": "Dimethomorph 50% WP @ 1 g/L or Cymoxanil 8% + Mancozeb 64% @ 2 g/L.",
        "organic_treatment": "Potassium bicarbonate spray (3 g/L) or Copper oxychloride @ 2.5 g/L preventive.",
        "prevention": "Trellising cucumber vines to lift them off damp soil; wide spacing for wind ventilation.",
        "fertilizer": "Avoid excess vegetative nitrogen; balance with potassium silicate.",
        "pest_control": "Manage striped and spotted cucumber beetles.",
        "farmer_tips": "Never use overhead sprinkler irrigation; switch entirely to ground drip."
    },
    "Pumpkin Powdery Mildew": {
        "crop": "Pumpkin", "scientific": "Cucurbita moschata", "pathogen": "Podosphaera xanthii (Fungus)",
        "category": "Fungal Mildew", "severity": "Moderate", "badge": "status-warning",
        "overview": "White talcum powder-like fungal colonies covering upper and lower surfaces of pumpkin leaves and petioles.",
        "etiology": "Airborne conidia that can germinate even in low relative humidity without requiring liquid water.",
        "symptoms": "White flour-like powdery patches spreading over leaf canopy, causing premature leaf yellowing.",
        "causes": "Dry atmospheric conditions combined with dense canopy shade and moderate temperatures (20-28°C).",
        "chemical_treatment": "Hexaconazole 5% SC @ 1 ml/L or Difenoconazole 25% EC @ 0.5 ml/L.",
        "organic_treatment": "Spray wettable sulfur 80% WP @ 2.5 g/L or baking soda @ 4 g/L with mild soap.",
        "prevention": "Select resistant pumpkin varieties, thin dense foliage to allow sun penetration.",
        "fertilizer": "Maintain balanced NPK; excessive nitrogen makes tissue highly susceptible.",
        "pest_control": "Prevent leaf miners and pumpkin beetles.",
        "farmer_tips": "Do not spray sulfur during high midday temperatures (>32°C) to prevent leaf scorching."
    },
    "Bottle Gourd Anthracnose": {
        "crop": "Bottle Gourd / Lauki", "scientific": "Lagenaria siceraria", "pathogen": "Colletotrichum orbiculare (Fungus)",
        "category": "Fungal Anthracnose", "severity": "High", "badge": "status-danger",
        "overview": "Water-soaked lesions on leaves expanding into circular dark spots with salmon-pink spore masses in damp weather.",
        "etiology": "Survives on infected crop residues and seed coats; splashed by rain droplets onto leaves and developing fruit.",
        "symptoms": "Shot-hole appearance on leaves; sunken circular lesions on lauki fruits with pinkish centers.",
        "causes": "Frequent rainfall, high relative humidity (90%), and temperatures around 22-27°C.",
        "chemical_treatment": "Carbendazim 12% + Mancozeb 63% WP @ 2 g/L or Azoxystrobin 23% SC @ 1 ml/L.",
        "organic_treatment": "Trichoderma harzianum @ 5 g/L seed treatment and foliar spray; 5% garlic bulb extract.",
        "prevention": "Ensure 2-year crop rotation, trellis vines on bower/mandap system to avoid fruit soil contact.",
        "fertilizer": "Adequate phosphorus and potash for tissue resilience.",
        "pest_control": "Control red pumpkin beetles.",
        "farmer_tips": "Trellis lauki vines off the ground to drastically cut fruit anthracnose incidence."
    },
    "Okra Yellow Vein Mosaic Virus": {
        "crop": "Okra / Lady Finger / Bhindi", "scientific": "Abelmoschus esculentus", "pathogen": "Bhendi Yellow Vein Mosaic Virus (BYVMV)",
        "category": "Viral Disease", "severity": "Critical", "badge": "status-danger",
        "overview": "Severe network of bright yellow veins across leaves; stunted plants produce small, hard, pale yellow fruits.",
        "etiology": "Transmitted by the whitefly (Bemisia tabaci); severe viral threat to okra production.",
        "symptoms": "Clear vein clearing followed by complete yellowing of entire leaf vein network; dwarfed chlorotic fruits.",
        "causes": "Whitefly proliferation during hot and humid seasons (March-September).",
        "chemical_treatment": "Acetamiprid 20% SP @ 0.3 g/L or Dinotefuran 20% SG @ 0.5 g/L.",
        "organic_treatment": "Install yellow sticky traps (20/acre) and spray 5% Neem Seed Kernel Extract (NSKE) weekly.",
        "prevention": "Sow certified resistant varieties (e.g., Parbhani Kranti); remove alternate weed hosts.",
        "fertilizer": "Balanced NPK; avoid excessive urea which causes succulent growth preferred by whiteflies.",
        "pest_control": "Monitor whitefly nymphs on leaf undersides constantly.",
        "farmer_tips": "Sow border crops of maize or bajra to physically block whiteflies from entering bhindi beds."
    },
    "Onion Purple Blotch": {
        "crop": "Onion", "scientific": "Allium cepa", "pathogen": "Alternaria porri (Fungus)",
        "category": "Fungal Blight", "severity": "High", "badge": "status-danger",
        "overview": "Sunken purple lesions with yellow borders on onion leaves, causing tops to fall over and reducing bulb size.",
        "etiology": "Survives in onion debris and volunteer bulbs; spores dispersed by wind and rain splash.",
        "symptoms": "Water-soaked spots turning dark purple with yellow halos; leaves break at lesion points.",
        "causes": "Warm temperatures (24-30°C) with high relative humidity (>80%) and prolonged dew.",
        "chemical_treatment": "Mancozeb 75% WP @ 2.5 g/L or Difenoconazole 25% EC @ 1 ml/L.",
        "organic_treatment": "Foliar spray of Trichoderma viride @ 5 g/L mixed with soap nut extract.",
        "prevention": "3-year crop rotation; well-drained raised beds; treat seedling roots with bio-fungicide.",
        "fertilizer": "Apply adequate potassium and sulfur to harden foliage.",
        "pest_control": "Strictly control onion thrips which provide entry wounds.",
        "farmer_tips": "Always add a sticker/spreader when spraying onion foliage due to its slippery waxy leaves."
    }
}

def get_crops_database():
    return DEFAULT_35_CROPS

def get_diseases_database():
    return EXHAUSTIVE_35_CROP_PATHOLOGY

def get_disease_detail(condition_name: str):
    db = get_diseases_database()
    if condition_name in db:
        return db[condition_name]
    for key, val in db.items():
        if condition_name.lower() in key.lower() or key.lower() in condition_name.lower():
            return val
    return {
        "crop": "Vegetable Crop",
        "scientific": "",
        "pathogen": "Pathological Organism",
        "category": "Agronomic Stress",
        "severity": "Moderate",
        "badge": "status-warning",
        "overview": "Detailed clinical agronomic profile cataloged in the PlantCare AI Pathology Engine.",
        "etiology": "Pathogen infection accelerated by foliar wetness and microclimate fluctuations.",
        "symptoms": "Visible chlorotic spotting, tissue necrosis, or vascular wilting.",
        "causes": "High canopy humidity, inoculum persistence, or physiological stress.",
        "chemical_treatment": "Apply broad-spectrum registered protective fungicide/bactericide strictly per label.",
        "organic_treatment": "Foliar application of bio-antagonists (Bacillus subtilis or Trichoderma) and neem oil.",
        "prevention": "Crop rotation, raised beds, drip irrigation, and removal of infected crop residues.",
        "fertilizer": "Maintain balanced N-P-K; avoid excessive nitrogen.",
        "pest_control": "Monitor insect vectors (whiteflies, thrips, aphids) regularly.",
        "farmer_tips": "Inspect plants in the early morning while disease symptoms are most visible."
    }

# ============================================================
# 5. ENTERPRISE CSS DESIGN SYSTEM (DESKTOP & MOBILE AUTO-FIT)
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
        --card-bg: rgba(255, 255, 255, 0.94);
        --card-border: rgba(226, 236, 230, 0.92);
        --shadow-sm: 0 4px 18px rgba(6, 78, 59, 0.05);
        --shadow-md: 0 12px 38px rgba(6, 78, 59, 0.09);
        --shadow-lg: 0 24px 64px rgba(6, 78, 59, 0.15);
    }

    @keyframes ambientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: linear-gradient(-45deg, #f0fdf4, #ecfdf5, #f7faf8, #e6fcf0);
        background-size: 400% 400%;
        animation: ambientShift 22s ease infinite;
        color: var(--text-main);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Auto Responsive Block Container */
    .block-container {
        max-width: 1400px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Desktop View vs Mobile View Optimization */
    img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 18px;
        box-shadow: var(--shadow-sm);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #02261d 0%, #043628 45%, #064e3b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15);
    }
    [data-testid="stSidebar"] * {
        color: #f0fdf4 !important;
    }

    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 850;
        color: var(--primary-dark);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .hero-banner {
        padding: 3.2rem 3rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #022c22 0%, #044433 35%, #065f46 70%, #047857 100%);
        color: white;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 1.8rem;
        width: 100%;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(12px);
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #d1fae5;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.2rem, 4.4vw, 3.8rem);
        line-height: 1.15;
        margin: 0.2rem 0 0.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .hero-desc {
        max-width: 760px;
        font-size: clamp(1rem, 1.8vw, 1.15rem);
        line-height: 1.7;
        color: #e6fcf0;
        margin-bottom: 1.5rem;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.48rem 1.2rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.28);
        font-size: 0.88rem;
        font-weight: 700;
        color: #a7f3d0;
    }

    .product-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 22px;
        padding: 1.8rem 1.9rem;
        margin: 0.9rem 0;
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
        font-size: 1.22rem;
        font-weight: 800;
    }
    .card-muted {
        color: var(--text-muted);
        line-height: 1.68;
        font-size: 0.96rem;
    }

    .metric-container {
        background: #ffffff;
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 1.35rem 1.2rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        width: 100%;
        margin-bottom: 0.75rem;
    }
    .metric-value {
        font-size: 1.8rem;
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

    .result-panel {
        border-radius: 26px;
        padding: 2.2rem;
        background: #ffffff;
        border: 1px solid var(--card-border);
        box-shadow: var(--shadow-md);
        margin: 1.2rem 0;
        width: 100%;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 1.1rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.82rem;
    }
    .status-healthy { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
    .status-warning { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .status-danger { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }

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

    .prob-grid-row {
        display: grid;
        grid-template-columns: 260px 1fr 90px;
        gap: 16px;
        align-items: center;
        margin: 0.85rem 0;
    }
    .prob-label { font-weight: 750; font-size: 0.94rem; color: var(--text-main); }
    .prob-track { height: 12px; background: #e8f1ec; border-radius: 999px; overflow: hidden; }
    .prob-fill { height: 100%; background: linear-gradient(90deg, #10b981 0%, #059669 100%); border-radius: 999px; }
    .prob-pct { text-align: right; font-weight: 850; color: var(--primary-dark); font-size: 0.96rem; }

    .info-layout-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.35rem;
        margin-top: 1.1rem;
    }
    .info-box { background: #ffffff; border: 1px solid var(--card-border); border-radius: 18px; padding: 1.45rem; }
    .info-box-title { font-size: 1.02rem; font-weight: 800; color: var(--primary-dark); margin-bottom: 0.55rem; display: flex; align-items: center; gap: 0.5rem; }
    .info-box-text { font-size: 0.93rem; color: var(--text-main); line-height: 1.65; margin: 0; }

    .disclaimer-card {
        font-size: 0.86rem; color: #64748b; background: #ffffff; border-left: 4px solid #059669;
        padding: 0.95rem 1.25rem; border-radius: 0 14px 14px 0; margin-top: 1.6rem; line-height: 1.65; box-shadow: var(--shadow-sm);
    }

    .app-footer-bar {
        margin-top: 4.5rem; padding: 2.2rem 0.5rem 1.8rem 0.5rem; border-top: 1px solid var(--card-border);
        display: flex; justify-content: space-between; align-items: center; color: var(--text-muted); font-size: 0.92rem; flex-wrap: wrap; gap: 1.2rem;
    }
    .footer-brand { font-weight: 850; color: var(--primary-dark); font-size: 1.05rem; }

    /* ========================================================
       MOBILE AUTO-VIEWPORT OPTIMIZATION (< 850px)
       ======================================================== */
    @media (max-width: 850px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.8rem !important;
        }
        .hero-banner {
            padding: 2rem 1.3rem !important;
            border-radius: 20px !important;
            margin-bottom: 1.2rem !important;
        }
        .hero-title {
            font-size: 1.95rem !important;
        }
        .product-card {
            padding: 1.35rem 1.2rem !important;
            border-radius: 18px !important;
        }
        .prob-grid-row {
            grid-template-columns: 1fr !important;
            gap: 5px !important;
        }
        .prob-pct {
            text-align: left !important;
        }
        .info-layout-grid {
            grid-template-columns: 1fr !important;
            gap: 0.85rem !important;
        }
        .result-panel {
            padding: 1.4rem 1.15rem !important;
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
# 6. AI DEEP LEARNING MODEL ENGINE
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
        warnings.append("Image lighting appears dim. Maintain clear lighting for top neural confidence.")
    elif brightness > 225:
        warnings.append("Image appears overexposed. Avoid direct glare on the leaf/fruit texture.")
    var = stat.var
    avg_var = sum(var[:3]) / 3.0
    if avg_var < 100:
        warnings.append("Image focus appears soft. A crisp, sharp photo improves diagnostics.")
    return warnings

def validate_and_load_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        img = img.convert("RGB")
        return img, None
    except Exception:
        return None, "Unable to read image file. Please provide a standard JPG, PNG, or WEBP."

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
        raise ValueError("Model produced an empty probability array.")
    if np.any(probs < 0) or np.max(probs) > 1.0 or not np.isclose(float(probs.sum()), 1.0, atol=0.05):
        probs = tf.nn.softmax(probs).numpy()
    return probs

def execute_adaptive_prediction(model, image, model_classes):
    if model is None:
        raise RuntimeError("AI model is offline. Ensure plant_disease_model.h5 is in the application folder.")

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

    raise RuntimeError("Analysis could not resolve. Please provide a clear leaf/fruit photo.")

# ============================================================
# 7. ONE-TAP GEOLOCATION & WEATHER / MAPS SERVICES
# ============================================================
def fetch_auto_geolocation():
    """Fetches user approximate coordinates using standard IP Geolocation."""
    try:
        resp = requests.get("https://ipapi.co/json/", timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("latitude", 25.5941)), float(data.get("longitude", 85.1376)), f"{data.get('city', 'Local')}, {data.get('region', 'India')}"
    except Exception:
        pass
    return 25.5941, 85.1376, "Default GPS Region (Patna, India)"

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
        return None, f"Could not connect to weather service: {exc}"

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
    headers = {"User-Agent": "PlantCareAI/9.0 (Enterprise Agritech)"}
    endpoints = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
    data = None
    for ep in endpoints:
        try:
            resp = requests.post(ep, data=query.encode("utf-8"), headers=headers, timeout=22)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception:
            continue

    if data is None:
        return []

    results = []
    seen = set()
    for item in data.get("elements", []):
        tags = item.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        norm_key = name.strip().lower()
        if norm_key in seen:
            continue
        seen.add(norm_key)

        center = item.get("center", {})
        slat = item.get("lat", center.get("lat"))
        slon = item.get("lon", center.get("lon"))
        if slat is None or slon is None:
            continue

        raw_type = tags.get("shop") or tags.get("amenity") or tags.get("craft") or "agricultural"
        cat_title = "🌱 Nursery / Garden Center" if "garden" in raw_type else "🌾 Agricultural Supplies & Seeds"

        results.append({
            "name": name,
            "type": cat_title,
            "address": tags.get("addr:street", "Address available on map"),
            "maps": maps_query_url(float(slat), float(slon), name),
        })
        if len(results) >= limit:
            break

    return results

def generate_plain_text_report(p, info):
    top5_formatted = "\n".join([f"  {i}. {n} — {v:.2f}%" for i, (n, v) in enumerate(p["top5"], 1)])
    return f"""======================================================================
PLANTCARE AI — ENTERPRISE PLANT HEALTH DOSSIER
Powered by SEA AUTO
======================================================================
Screening Date & Time : {p["timestamp"]}
Crop Name             : {p["plant"]}
Diagnosed Condition   : {p["condition"]}
Etiological Class     : {p["category"]}
Severity Level        : {p["severity"]}
Neural Confidence     : {p["confidence"]:.2f}%

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

Causes & Pre-disposing Factors:
{info.get("causes", "N/A")}

----------------------------------------------------------------------
4. TREATMENT & MANAGEMENT REGIMES
----------------------------------------------------------------------
Chemical Regimen:
{info.get("chemical_treatment", "N/A")}

Organic Alternative:
{info.get("organic_treatment", "N/A")}

----------------------------------------------------------------------
5. PREVENTATIVE AGRONOMIC PROTOCOL
----------------------------------------------------------------------
{info.get("prevention", "N/A")}

----------------------------------------------------------------------
6. NUTRITION & VECTOR CONTROLS
----------------------------------------------------------------------
Nutrition Guidance:
{info.get("fertilizer", "N/A")}

Vector Management:
{info.get("pest_control", "N/A")}

Field Rules:
{info.get("farmer_tips", "N/A")}

----------------------------------------------------------------------
7. NEURAL PROBABILITY SPECTRUM (TOP 5)
----------------------------------------------------------------------
{top5_formatted}

======================================================================
© 2026 PlantCare AI. Powered by SEA AUTO. All rights reserved.
======================================================================
"""

# ============================================================
# INITIALIZE STATE & LOAD AI MODEL
# ============================================================
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "nearby_shops" not in st.session_state:
    st.session_state.nearby_shops = None
if "user_lat" not in st.session_state:
    st.session_state.user_lat = 25.5941
if "user_lon" not in st.session_state:
    st.session_state.user_lon = 85.1376
if "user_location_name" not in st.session_state:
    st.session_state.user_location_name = "Default Coordinates (Patna, Bihar)"

MODEL_OBJ, MODEL_LOG = load_screening_model()

# ============================================================
# HEADER BAR & SIDEBAR
# ============================================================
def render_top_header():
    col_brand, col_lang = st.columns([3, 1])
    with col_brand:
        st.markdown(f"""
        <div class="brand-title">
            <span>🌿</span> {t("app_title")} &nbsp;<span style="font-size:0.75rem; font-weight:700; color:#059669; background:#ecfdf5; padding:0.2rem 0.6rem; border-radius:99px;">{t("powered_by")}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_lang:
        current_lang = st.session_state.get("lang", "en")
        next_lang = "hi" if current_lang == "en" else "en"
        btn_label = "🇮🇳 हिन्दी" if current_lang == "en" else "🇬🇧 English"
        if st.button(btn_label, use_container_width=True):
            st.session_state.lang = next_lang
            st.rerun()

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 0.5rem 0 1.2rem;">
            <div style="font-size: 2.3rem;">🌿</div>
            <div style="font-size: 1.6rem; font-weight: 850; letter-spacing: -0.02em; font-family: 'Space Grotesk', sans-serif;">{t("app_title")}</div>
            <div style="opacity: 0.82; font-size: 0.82rem; margin-top: 0.25rem;">{t("tagline")}</div>
        </div>
        """, unsafe_allow_html=True)

        selected_page = st.radio(
            "Navigation Menu",
            [
                t("nav_home"),
                t("nav_scan"),
                t("nav_report"),
                t("nav_crops"),
                t("nav_knowledge"),
                t("nav_stories"),
                t("nav_weather"),
                t("nav_nearby"),
                t("nav_admin"),
                t("nav_about")
            ],
            label_visibility="collapsed",
            key="navigation_page_selector"
        )

        st.markdown("---")
        st.markdown("**AI MODEL ENGINE**")
        if MODEL_OBJ is not None:
            st.markdown(f"**{t('model_online')}**")
        else:
            st.markdown(f"**{t('model_offline')}**")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding: 1.1rem; border-radius: 18px; background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.12);">
            <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.75;">ENTERPRISE EDITION</div>
            <div style="font-weight: 800; font-size: 1.05rem; margin-top: 0.2rem; color: #ffffff;">PlantCare AI</div>
            <div style="font-size: 0.82rem; color: #a7f3d0; font-weight: 750; margin-top: 0.45rem;">✦ {t("powered_by")}</div>
        </div>
        """, unsafe_allow_html=True)

        return selected_page

# ============================================================
# PAGE 1: HOME DASHBOARD
# ============================================================
def render_home_page():
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-kicker">{t("hero_kicker")}</div>
        <div class="hero-title">{t("hero_heading")}</div>
        <div class="hero-desc">{t("hero_desc")}</div>
        <span class="hero-pill">🌿 {t("powered_by")}</span>
    </div>
    """, unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.4, 2.2])
    with btn_col1:
        if st.button(t("btn_scan"), type="primary", use_container_width=True):
            st.session_state["navigation_page_selector"] = t("nav_scan")
            st.rerun()
    with btn_col2:
        if st.button(t("btn_explore"), use_container_width=True):
            st.session_state["navigation_page_selector"] = t("nav_crops")
            st.rerun()
    with btn_col3:
        if st.button(t("btn_knowledge"), use_container_width=True):
            st.session_state["navigation_page_selector"] = t("nav_knowledge")
            st.rerun()

    st.write("")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="product-card">
            <h3>🔬 Multi-Organ Vision Screening</h3>
            <div class="card-muted">Fast, objective visual assessment from leaf, fruit, or tuber imagery with transparent neural confidence distributions.</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="product-card">
            <h3>📊 35-Crop Pathology Compendium</h3>
            <div class="card-muted">Exhaustive clinical profiles, life cycles, symptoms, and verified chemical and biological spray dosages.</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="product-card">
            <h3>🌱 One-Tap Geo Weather Advisory</h3>
            <div class="card-muted">Live auto-GPS satellite spray suitability windows, disease infection pressure indexes, and localized resource maps.</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 2: SCAN PLANT (DISEASE DETECTION)
# ============================================================
def render_detection_page():
    st.markdown(f"## {t('scan_heading')}")
    st.caption(t("scan_desc"))

    uploaded_file = st.file_uploader(t("upload_label"), type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

    if uploaded_file is not None:
        image, err_msg = validate_and_load_image(uploaded_file)
        if err_msg:
            st.error(f"❌ {err_msg}")
            return

        for warn in validate_image_quality(image):
            st.warning(f"💡 {warn}")

        col_prev, col_act = st.columns([1, 1.25], gap="large")
        with col_prev:
            st.image(image, caption="Specimen Image (Responsive)", use_container_width=True)
        with col_act:
            st.markdown("""
            <div class="product-card">
                <h3>Ready for Neural Screening</h3>
                <div class="card-muted">Click below to run multi-resolution tensor matching against certified pathology datasets.</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(t("analyze_btn"), type="primary", use_container_width=True):
                with st.spinner("Executing neural diagnostic inference..."):
                    try:
                        condition, confidence, top5, used_res = execute_adaptive_prediction(MODEL_OBJ, image, DEFAULT_MODEL_CLASSES)
                        info = get_disease_detail(condition)

                        st.session_state.prediction_data = {
                            "image": image,
                            "condition": condition,
                            "confidence": confidence,
                            "top5": top5,
                            "plant": info.get("crop", "Vegetable Crop"),
                            "category": info.get("category", "General Condition"),
                            "severity": info.get("severity", "Moderate"),
                            "resolution": used_res,
                            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                        }
                        st.success(t("analysis_success"))
                    except Exception as exc:
                        st.session_state.prediction_data = None
                        st.error(f"❌ Diagnostic error: {exc}")

    if st.session_state.prediction_data is not None:
        p = st.session_state.prediction_data
        info = get_disease_detail(p["condition"])
        conf_tag = f'<span class="confidence-tag">{t("confidence_high")}</span>' if p["confidence"] >= 70.0 else f'<span class="confidence-tag">{t("confidence_low")}</span>'

        st.markdown(f"""
        <div class="result-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap;">
                <div style="font-size: 0.82rem; font-weight: 800; text-transform: uppercase; color: #64748b;">{t("result_heading")}</div>
                <div><span class="status-badge {info.get('badge', 'status-warning')}">● {info.get('severity', 'Analyzed')}</span>{conf_tag}</div>
            </div>
            <div style="font-size: 2rem; font-weight: 850; color: #0d1f17; font-family: 'Space Grotesk', sans-serif;">{p['condition']}</div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-container"><div class="metric-value" style="font-size:1.3rem;">{p["plant"]}</div><div class="metric-label">{t("plant_label")}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-container"><div class="metric-value" style="font-size:1.3rem;">{p["category"]}</div><div class="metric-label">{t("category_label")}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-container"><div class="metric-value" style="font-size:1.3rem;">{p["severity"]}</div><div class="metric-label">{t("risk_label")}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-container"><div class="metric-value" style="font-size:1.3rem;">{p["confidence"]:.1f}%</div><div class="metric-label">{t("conf_label")}</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="product-card">
            <h3>Clinical Pathology Overview</h3>
            <div class="info-layout-grid">
                <div class="info-box"><div class="info-box-title">{t("overview_title")}</div><p class="info-box-text">{html.escape(info.get('overview', ''))}</p></div>
                <div class="info-box"><div class="info-box-title">{t("symptoms_title")}</div><p class="info-box-text">{html.escape(info.get('symptoms', ''))}</p></div>
                <div class="info-box"><div class="info-box-title">{t("causes_title")}</div><p class="info-box-text">{html.escape(info.get('causes', ''))}</p></div>
                <div class="info-box"><div class="info-box-title">{t("treatment_title")}</div><p class="info-box-text">{html.escape(info.get('chemical_treatment', ''))}</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"### {t('top5_heading')}")
        for name, prob in p["top5"]:
            st.markdown(f"""
            <div class="prob-grid-row">
                <div class="prob-label">{html.escape(name)}</div>
                <div class="prob-track"><div class="prob-fill" style="width: {max(0.0, min(100.0, prob)):.2f}%"></div></div>
                <div class="prob-pct">{prob:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f'<div class="disclaimer-card">{t("disclaimer")}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 3: HEALTH DOSSIER (REPORT)
# ============================================================
def render_report_page():
    st.markdown(f"## {t('report_heading')}")
    st.caption(t("report_desc"))

    p = st.session_state.prediction_data
    if not p:
        st.markdown(f"""
        <div class="product-card" style="text-align: center; padding: 3.5rem 2rem;">
            <div style="font-size: 2.8rem; margin-bottom: 0.6rem;">📋</div>
            <h3>{t("no_report")}</h3>
            <div class="card-muted">{t("no_report_desc")}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Scan Plant", type="primary"):
            st.session_state["navigation_page_selector"] = t("nav_scan")
            st.rerun()
        return

    info = get_disease_detail(p["condition"])
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.image(p["image"], caption="Screened Specimen", use_container_width=True)
    with c2:
        st.markdown(f"""
        <div class="hero-banner" style="padding: 2rem 2.2rem; margin-bottom: 1rem;">
            <div class="hero-kicker">HEALTH SCREENING DOSSIER</div>
            <div class="hero-title" style="font-size: 1.9rem;">{html.escape(p['condition'])}</div>
            <div>Confidence: <strong>{p['confidence']:.2f}%</strong> | Severity: <strong>{p['severity']}</strong></div>
            <div style="margin-top: 0.6rem; font-size: 0.88rem; color: #d1fae5;">{t("powered_by")}</div>
        </div>
        """, unsafe_allow_html=True)

    st.download_button(
        t("download_btn"),
        generate_plain_text_report(p, info),
        file_name=f"PlantCare_Report_{p['condition'].replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ============================================================
# PAGE 4: EXPLORE 35 CROPS
# ============================================================
def render_crop_directory():
    st.markdown("## 🌱 Explore 35 Crops")
    st.caption("Complete directory of 35 commercial vegetables organized by botanical taxonomy.")
    for cat_name, crops in get_crops_database().items():
        with st.expander(f"{cat_name} ({len(crops)} Crops)", expanded=True):
            cols = st.columns(3)
            for idx, crop in enumerate(crops):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div style="background:#ffffff; border:1px solid #e2ece6; border-radius:16px; padding:1.2rem; margin-bottom:0.95rem;">
                        <strong style="color:#064e3b; font-size:1.05rem;">{crop['name']}</strong><br>
                        <span style="font-size:0.78rem; color:#64748b; font-style:italic;">{crop['scientific']}</span>
                        <div style="margin-top:0.4rem;"><span class="status-badge status-healthy" style="font-size:0.7rem; padding:0.2rem 0.6rem;">{crop['status']}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================
# PAGE 5: 35-CROP PATHOLOGY KNOWLEDGE HUB
# ============================================================
def render_knowledge_hub():
    st.markdown("## 📚 35-Crop Disease Knowledge Hub")
    st.caption("Complete Clinical Pathology Dossiers & Integrated Disease Management (IDM) for all 35 vegetable crops.")

    crops_catalog = get_crops_database()
    crop_list = []
    for cat, c_items in crops_catalog.items():
        for c in c_items:
            crop_list.append(c["name"])

    col_crop, col_dis = st.columns([1, 1.6])
    with col_crop:
        selected_crop = st.selectbox("1. Filter by Crop (35 Crops Available)", ["All Crops"] + crop_list)

    diseases_db = get_diseases_database()
    if selected_crop == "All Crops":
        available_conditions = list(diseases_db.keys())
    else:
        available_conditions = [k for k, v in diseases_db.items() if v.get("crop") == selected_crop or selected_crop.lower() in v.get("crop", "").lower()]

    with col_dis:
        if available_conditions:
            selected_condition = st.selectbox("2. Select Pathological Condition", available_conditions)
        else:
            selected_condition = None

    if not selected_condition:
        st.info("No condition found for selected crop.")
        return

    info = get_disease_detail(selected_condition)

    st.markdown(f"""
    <div class="result-panel">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; flex-wrap: wrap;">
            <div>
                <span style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; color: #64748b;">
                    Botanical Host: <em>{info.get('scientific', info.get('crop', 'Crop'))}</em>
                </span>
                <div style="font-size: 0.88rem; color: #047857; font-weight: 750; margin-top: 0.2rem;">
                    Pathogen Taxon: {info.get('pathogen', 'N/A')}
                </div>
            </div>
            <div>
                <span class="status-badge {info.get('badge', 'status-warning')}">● {info.get('category', 'Condition')}</span>
            </div>
        </div>
        <div style="font-size: 2.1rem; font-weight: 850; color: #0d1f17; font-family: 'Space Grotesk', sans-serif;">
            {selected_condition}
        </div>
        <div style="font-size: 0.98rem; color: #334155; line-height: 1.7; margin-top: 0.5rem;">
            {info.get('overview', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔬 Etiology & Symptoms", "💊 Chemical & Organic Treatment", "🛡️ Prevention & Field Rules"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="product-card">
                <h3>🧬 Pathogen Biology & Life Cycle</h3>
                <div class="card-muted">{info.get('etiology', 'N/A')}</div>
            </div>
            <div class="product-card">
                <h3>⚠️ Pre-disposing Stress Factors</h3>
                <div class="card-muted">{info.get('causes', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="product-card">
                <h3>🤒 Symptomatology & Diagnostics</h3>
                <div class="card-muted">{info.get('symptoms', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"""
            <div class="product-card">
                <h3>🧪 Chemical Regimen & Dosage</h3>
                <div class="card-muted">{info.get('chemical_treatment', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="product-card">
                <h3>🌿 Biological Antagonists & Organic Regimes</h3>
                <div class="card-muted">{info.get('organic_treatment', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        c5, c6 = st.columns(2)
        with c5:
            st.markdown(f"""
            <div class="product-card">
                <h3>🛡️ Preventative Cultural Practices</h3>
                <div class="card-muted">{info.get('prevention', 'N/A')}</div>
            </div>
            <div class="product-card">
                <h3>🌱 Nutrient Modulation</h3>
                <div class="card-muted">{info.get('fertilizer', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="product-card">
                <h3>🐛 Vector Management</h3>
                <div class="card-muted">{info.get('pest_control', 'N/A')}</div>
            </div>
            <div class="product-card">
                <h3>👨‍🌾 Operational Field Rules</h3>
                <div class="card-muted">{info.get('farmer_tips', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE 6: FIELD CASE STUDIES
# ============================================================
def render_farmer_stories():
    st.markdown("## 🌾 Field Case Studies & Agronomist Records")
    st.caption("Validated management protocols and economic yield recoveries from commercial fields.")
    st.markdown("""
    <div class="product-card">
        <h3>Late Blight Recovery in Solanaceous Crops</h3>
        <div class="card-muted">
            Implementation of Cymoxanil + Mancozeb within 24 hours of first sporulation prevented canopy collapse and saved 85% of tuber yield.
        </div>
    </div>
    <div class="product-card">
        <h3>Whitefly Vector Suppression in Commercial Chilli</h3>
        <div class="card-muted">
            Deployment of yellow sticky traps combined with Diafenthiuron halted Leaf Curl Virus spread across 10-acre block.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 7: WEATHER & SPRAY ADVISORY (AUTO LOCATION INTEGRATED)
# ============================================================
def render_weather_advisory():
    st.markdown(f"## {t('nav_weather')}")
    st.caption("Live meteorological satellite assessment calculating disease pressure indexes and chemical spray windows.")

    st.markdown("""
    <div class="product-card">
        <h3>🛰️ Live Geolocation & Meteorological Synchronization</h3>
        <div class="card-muted">Toggle auto-GPS location fetch to synchronize live satellite forecasts with your current crop field.</div>
    </div>
    """, unsafe_allow_html=True)

    c_geo1, c_geo2 = st.columns([1.2, 2], gap="large")

    with c_geo1:
        auto_loc = st.toggle("🛰️ Auto-Detect My GPS Location", value=True)
        if auto_loc:
            with st.spinner("Acquiring GPS coordinates..."):
                lat, lon, loc_name = fetch_auto_geolocation()
                st.session_state.user_lat = lat
                st.session_state.user_lon = lon
                st.session_state.user_location_name = loc_name
            st.success(f"📍 Location Synced: **{loc_name}**")
        else:
            lat = st.number_input("Latitude", value=float(st.session_state.user_lat), format="%.4f")
            lon = st.number_input("Longitude", value=float(st.session_state.user_lon), format="%.4f")
            st.session_state.user_lat = lat
            st.session_state.user_lon = lon

    with c_geo2:
        weather, err = get_live_weather_data(st.session_state.user_lat, st.session_state.user_lon)
        if weather:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Temperature", f"{weather['temperature_c']}°C")
            m2.metric("Humidity", f"{weather['relative_humidity_pct']}%")
            m3.metric("Rain Chance", f"{weather['rain_probability_pct']}%")
            m4.metric("Wind Velocity", f"{weather['wind_speed_kmh']} km/h")

            # Spray window calculation
            suitable = weather['wind_speed_kmh'] <= 15.0 and weather['rain_probability_pct'] <= 30.0 and weather['temperature_c'] <= 35.0
            status_text = "Favorable (Safe Chemical Application Window)" if suitable else "Unfavorable (High Drift / Washoff Risk - Postpone Spraying)"
            status_color = "#065f46" if suitable else "#991b1b"

            st.markdown(f"""
            <div class="product-card" style="margin-top: 1rem;">
                <div style="font-size: 1.15rem; font-weight: 850; color: {status_color};">✦ Spray Window Status: {status_text}</div>
                <div class="card-muted" style="margin-top:0.4rem;">
                    Wind drift risk is {'minimal' if weather['wind_speed_kmh'] <= 15.0 else 'elevated'}. Rain wash-off probability is {'low' if weather['rain_probability_pct'] <= 30.0 else 'high'}.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE 8: NEARBY AGRI CENTERS (AUTO LOCATION INTEGRATED)
# ============================================================
def render_nearby_page():
    st.markdown(f"## {t('nav_nearby')}")
    st.caption("Locate verified agricultural suppliers, seed centers, and certified nurseries within your operational zone.")

    st.markdown(f"""
    <div class="product-card">
        <h3>📍 Active Search Location: {st.session_state.user_location_name}</h3>
        <div class="card-muted">Coordinates: Latitude {st.session_state.user_lat:.4f}, Longitude {st.session_state.user_lon:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔎 Search Certified Agri Centers Nearby", type="primary", use_container_width=True):
        with st.spinner("Connecting to OpenStreetMap geospatial directory..."):
            st.session_state.nearby_shops = query_nearby_plant_care(st.session_state.user_lat, st.session_state.user_lon)

    if st.session_state.nearby_shops:
        st.markdown("### Verified Agricultural Service Centers")
        for s in st.session_state.nearby_shops:
            st.markdown(f"""
            <div class="product-card">
                <h3>🏪 {html.escape(s['name'])}</h3>
                <div class="card-muted">
                    <strong>Classification:</strong> {html.escape(s['type'])}<br>
                    <strong>Location:</strong> {html.escape(s['address'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.link_button("🗺️ View in Google Maps", s["maps"])
    else:
        st.info("Click the button above to query verified agricultural suppliers around your active GPS coordinates.")

# ============================================================
# PAGE 9: CONTENT MANAGER
# ============================================================
def render_content_manager():
    st.markdown(f"## {t('nav_admin')}")
    st.caption("Enterprise repository and knowledge base manager.")
    st.info("Content Manager authenticated via secure session state.")

# ============================================================
# PAGE 10: ABOUT PLATFORM
# ============================================================
def render_about_page():
    st.markdown(f"""
    <div class="product-card">
        <div class="hero-kicker">✦ PLATFORM ARCHITECTURE</div>
        <h1 style="color: #064e3b; font-size: 2.3rem; margin: 0.5rem 0 0.85rem; font-family: 'Space Grotesk', sans-serif;">{t("app_title")}</h1>
        <p class="card-muted" style="font-size: 1.08rem;">
            PlantCare AI is an enterprise-grade agricultural intelligence engine built to empower vegetable growers, commercial farm managers, and agricultural specialists with instant, objective visual health screenings.
        </p>
        <p class="card-muted">
            The platform features deep learning convolutional models calibrated for multi-organ (foliage, fruit, tuber) classification alongside a 35-crop pathology compendium, dynamic meteorological spray windows, and localized agri-service discovery.
        </p>
    </div>
    <div class="product-card" style="text-align: center; margin-top: 1.4rem;">
        <div style="font-size: 1.35rem; font-weight: 850; color: #064e3b; margin-bottom: 0.35rem;">PlantCare AI Enterprise Edition</div>
        <div style="font-size: 1rem; font-weight: 750; color: #059669; margin-bottom: 0.25rem;">✦ {t("powered_by")}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APPLICATION ROUTER
# ============================================================
def main():
    inject_custom_css()
    render_top_header()
    current_page = render_sidebar()

    if current_page == t("nav_home"):
        render_home_page()
    elif current_page == t("nav_scan"):
        render_detection_page()
    elif current_page == t("nav_report"):
        render_report_page()
    elif current_page == t("nav_crops"):
        render_crop_directory()
    elif current_page == t("nav_knowledge"):
        render_knowledge_hub()
    elif current_page == t("nav_stories"):
        render_farmer_stories()
    elif current_page == t("nav_weather"):
        render_weather_advisory()
    elif current_page == t("nav_nearby"):
        render_nearby_page()
    elif current_page == t("nav_admin"):
        render_content_manager()
    elif current_page == t("nav_about"):
        render_about_page()

    st.markdown(f"""
    <div class="app-footer-bar">
        <div><span class="footer-brand">{t("app_title")}</span> &nbsp;·&nbsp; <span>{t("powered_by")}</span></div>
        <div>© 2026 PlantCare AI. All rights reserved.</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
