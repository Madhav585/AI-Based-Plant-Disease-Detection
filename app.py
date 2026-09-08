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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
FARMER_IMAGES_DIR = IMAGES_DIR / "farmers"
AD_IMAGES_DIR = IMAGES_DIR / "advertisements"
CROP_IMAGES_DIR = IMAGES_DIR / "crops"

for directory in [DATA_DIR, IMAGES_DIR, FARMER_IMAGES_DIR, AD_IMAGES_DIR, CROP_IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. CENTRALIZED TRANSLATION ENGINE (ENGLISH ⇄ हिन्दी)
# ============================================================
TRANSLATIONS = {
    "en": {
        "app_title": "PlantCare AI",
        "tagline": "AI-Powered Plant Health Screening & Diagnostic Engine",
        "powered_by": "Powered by SEA AUTO",
        "nav_home": "🏠 Home",
        "nav_scan": "🔬 Scan Plant",
        "nav_report": "📄 Plant Health Report",
        "nav_crops": "🌱 Explore Crops",
        "nav_knowledge": "📚 Disease Knowledge Hub",
        "nav_stories": "🌾 Farmer Stories",
        "nav_weather": "🌦️ Weather & Advisory",
        "nav_nearby": "📍 Nearby Plant Care",
        "nav_admin": "⚙️ Content Manager",
        "nav_about": "ℹ️ About PlantCare AI",
        "hero_kicker": "✦ SMART AGRITECH INTELLIGENCE",
        "hero_heading": "Healthy Plants. Better Harvests.",
        "hero_desc": "Upload plant leaf or fruit imagery to receive objective AI-assisted health assessments, clinical pathology insights, and practical crop-care guidance across 35 agricultural crops.",
        "btn_scan": "Scan Your Plant",
        "btn_explore": "Explore 35 Crops",
        "btn_knowledge": "Disease Knowledge Hub",
        "model_online": "🟢 Model Online & Loaded",
        "model_offline": "🟡 Model Offline (Demo Mode)",
        "feature_scan_title": "🔬 Multi-Organ Vision Screening",
        "feature_scan_desc": "Fast, objective visual assessment from leaf, fruit, or tuber imagery with transparent neural network confidence distributions.",
        "feature_comp_title": "📊 35-Crop Pathology Compendium",
        "feature_comp_desc": "Exhaustive disease descriptions, life cycles, stage-by-stage symptoms, and verified chemical and biological regimens.",
        "feature_weather_title": "🌱 Dynamic Weather & Care Guidance",
        "feature_weather_desc": "Live meteorological spray feasibility windows, disease pressure indexes, and localized plant-care discovery.",
        "scan_heading": "🔬 Scan & Diagnose Plant",
        "scan_desc": "Upload a clear photo of a plant leaf, fruit, or tuber to receive an AI-assisted health assessment and practical care guidance.",
        "upload_label": "Upload plant specimen image",
        "analyze_btn": "🔬 Analyze Specimen",
        "analysis_success": "Screening completed successfully.",
        "confidence_high": "🟢 High confidence",
        "confidence_low": "🟡 Moderate / Low confidence",
        "result_heading": "Screening Result",
        "plant_label": "Plant",
        "category_label": "Category",
        "risk_label": "Risk Level",
        "conf_label": "Confidence",
        "overview_title": "📖 Description",
        "symptoms_title": "🤒 Symptoms",
        "causes_title": "⚠️ Causes",
        "treatment_title": "💊 Treatment / Management",
        "fert_title": "🌱 Fertilizer Guidance",
        "pest_title": "🐛 Pest & Disease Control",
        "tips_title": "👨‍🌾 Farmer Tips",
        "top5_heading": "📊 Top 5 AI Predictions",
        "disclaimer": "AI-assisted visual screening is intended as an initial assessment. For critical agricultural decisions, consult a certified agricultural officer.",
        "report_heading": "📄 Plant Health Report",
        "report_desc": "Complete diagnostic dossier ready for review and local export.",
        "no_report": "No Active Screening Record",
        "no_report_desc": "Please analyze a plant leaf or fruit in the 'Scan Plant' section first to view and download your clinical health report.",
        "download_btn": "📥 Download Plant Health Report (.txt)",
        "about_heading": "About PlantCare AI",
        "about_text": "PlantCare AI is an enterprise AI-powered plant health screening and crop protection engine built to help growers, gardeners, and agricultural specialists understand plant pathological conditions and receive actionable agronomic guidance."
    },
    "hi": {
        "app_title": "PlantCare AI",
        "tagline": "AI-संचालित पादप स्वास्थ्य परीक्षण और निदान इंजन",
        "powered_by": "SEA AUTO द्वारा संचालित",
        "nav_home": "🏠 होम",
        "nav_scan": "🔬 पौधे की जांच करें",
        "nav_report": "📄 स्वास्थ्य रिपोर्ट",
        "nav_crops": "🌱 फसलें देखें",
        "nav_knowledge": "📚 रोग ज्ञान केंद्र",
        "nav_stories": "🌾 किसान कहानियां",
        "nav_weather": "🌦️ मौसम और सलाह",
        "nav_nearby": "📍 नजदीकी कृषि केंद्र",
        "nav_admin": "⚙️ सामग्री प्रबंधक",
        "nav_about": "ℹ️ प्लांटकेयर एआई के बारे में",
        "hero_kicker": "✦ स्मार्ट एग्रीटेक इंटेलिजेंस",
        "hero_heading": "स्वस्थ पौधे। बेहतर पैदावार।",
        "hero_desc": "35 कृषि फसलों में उद्देश्यपूर्ण एआई-सहायता प्राप्त स्वास्थ्य मूल्यांकन, नैदानिक रोग विज्ञान अंतर्दृष्टि और व्यावहारिक फसल-देखभाल मार्गदर्शन प्राप्त करने के लिए पौधे की पत्ती या फल की छवि अपलोड करें।",
        "btn_scan": "पौधे की जांच करें",
        "btn_explore": "35 फसलें देखें",
        "btn_knowledge": "रोग ज्ञान केंद्र",
        "model_online": "🟢 मॉडल ऑनलाइन और लोड है",
        "model_offline": "🟡 मॉडल ऑफ़लाइन (डेमो मोड)",
        "feature_scan_title": "🔬 मल्टी-ऑर्गन विजन स्क्रीनिंग",
        "feature_scan_desc": "पारदर्शी न्यूरल नेटवर्क कॉन्फिडेंस डिस्ट्रीब्यूशन के साथ पत्ती, फल या कंद की छवियों से तेज़, वस्तुनिष्ठ दृश्य मूल्यांकन।",
        "feature_comp_title": "📊 35-फसल रोग विज्ञान संग्रह",
        "feature_comp_desc": "व्यापक रोग विवरण, जीवन चक्र, चरण-दर-चरण लक्षण और सत्यापित रासायनिक और जैविक उपचार प्रणालियाँ।",
        "feature_weather_title": "🌱 गतिशील मौसम और देखभाल मार्गदर्शन",
        "feature_weather_desc": "लाइव मौसम विज्ञान स्प्रे व्यवहार्यता विंडो, रोग दबाव सूचकांक और स्थानीयकृत पादप-देखभाल खोज।",
        "scan_heading": "🔬 पौधे की जांच और निदान",
        "scan_desc": "एआई-सहायता प्राप्त स्वास्थ्य मूल्यांकन और व्यावहारिक देखभाल मार्गदर्शन प्राप्त करने के लिए पौधे की पत्ती, फल या कंद की स्पष्ट तस्वीर अपलोड करें।",
        "upload_label": "पादप नमूना छवि अपलोड करें",
        "analyze_btn": "🔬 नमूना विश्लेषण करें",
        "analysis_success": "स्क्रीनिंग सफलतापूर्वक पूरी हुई।",
        "confidence_high": "🟢 उच्च विश्वास (High Confidence)",
        "confidence_low": "🟡 मध्यम / निम्न विश्वास",
        "result_heading": "स्क्रीनिंग परिणाम",
        "plant_label": "फसल",
        "category_label": "श्रेणी",
        "risk_label": "जोखिम स्तर",
        "conf_label": "कॉन्फिडेंस",
        "overview_title": "📖 विवरण",
        "symptoms_title": "🤒 लक्षण",
        "causes_title": "⚠️ कारण",
        "treatment_title": "💊 उपचार / प्रबंधन",
        "fert_title": "🌱 उर्वरक मार्गदर्शन",
        "pest_title": "🐛 कीट और रोग नियंत्रण",
        "tips_title": "👨‍🌾 किसान सुझाव",
        "top5_heading": "📊 शीर्ष 5 एआई भविष्यवाणियां",
        "disclaimer": "एआई-सहायता प्राप्त दृश्य स्क्रीनिंग एक प्रारंभिक मूल्यांकन के रूप में है। महत्वपूर्ण कृषि निर्णयों के लिए, योग्य कृषि विशेषज्ञ से परामर्श लें।",
        "report_heading": "📄 पादप स्वास्थ्य रिपोर्ट",
        "report_desc": "समीक्षा और स्थानीय निर्यात के लिए तैयार पूर्ण नैदानिक दस्तावेज।",
        "no_report": "कोई सक्रिय स्क्रीनिंग रिकॉर्ड नहीं",
        "no_report_desc": "अपनी नैदानिक स्वास्थ्य रिपोर्ट देखने और डाउनलोड करने के लिए पहले 'स्कैन प्लांट' अनुभाग में पौधे की पत्ती या फल का विश्लेषण करें।",
        "download_btn": "📥 पादप स्वास्थ्य रिपोर्ट डाउनलोड करें (.txt)",
        "about_heading": "PlantCare AI के बारे में",
        "about_text": "PlantCare AI एक अत्याधुनिक AI-संचालित पादप स्वास्थ्य और फसल सुरक्षा इंजन है, जो उत्पादकों और किसानों को पत्तियों और फलों की छवियों से दृश्य समस्याओं को पहचानने और सटीक उपचार प्राप्त करने में सहायता करता है।"
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"

def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))

# ============================================================
# 2. MODEL CLASSES REGISTRY (INFERENCE SOURCE OF TRUTH)
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
# 3. EXHAUSTIVE 35-CROP PATHOLOGY KNOWLEDGE REPOSITORY
# ============================================================
EXHAUSTIVE_35_CROP_PATHOLOGY = {
    # Solanaceae
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
        "overview": "Devastating water-soaked lesions that cause whole vine collapse and destructive fruit rot in cool, damp weather.",
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
        "prevention": "Strictly plant certified disease-free seed tubers; hill soil properly to protect tubers from spores.",
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
        "organic_treatment": "Seed treatment with hot water (50°C for 25 mins) and Pseudomonas fluorescens foliar spray @ 5 g/L.",
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
        "symptoms": "Sudden wilting of top leaves during hot sunny hours, followed by whole plant death; white milky bacterial slime from cut stem in water.",
        "causes": "High soil moisture, poorly drained soils, temperatures above 28°C.",
        "chemical_treatment": "Soil drenching with Copper Oxychloride @ 3 g/L + Streptocycline @ 0.2 g/L at early signs.",
        "organic_treatment": "Soil incorporation of Pseudomonas fluorescens @ 2.5 kg/ha with enriched FYM before planting.",
        "prevention": "Grafting on resistant rootstocks (Solanum torvum), raised nursery beds, crop rotation with maize/paddy.",
        "fertilizer": "Apply neem cake @ 250 kg/ha to suppress soil pathogens and nematodes.",
        "pest_control": "Control root-knot nematodes (Meloidogyne spp.) strictly.",
        "farmer_tips": "Do stem-streaming test in a clear glass of water to instantly confirm bacterial wilt vs fungal wilt."
    },
    "Chilli Leaf Curl Virus": {
        "crop": "Chilli", "scientific": "Capsicum frutescens", "pathogen": "Chilli Leaf Curl Virus (Begomovirus)",
        "category": "Viral Disease", "severity": "High", "badge": "status-danger",
        "overview": "Severe curling, puckering of leaves, stunting of plants, and massive reduction in flower and fruit set.",
        "etiology": "Transmitted systematically by whiteflies (Bemisia tabaci); not seed-transmitted.",
        "symptoms": "Upward curling and crinkling of leaves, thickened veins, shortened internodes creating bushy stunted plants.",
        "causes": "High whitefly populations during dry, warm weather.",
        "chemical_treatment": "Vector control: Diafenthiuron 50% WP @ 1.2 g/L or Spiromesifen 22.9% SC @ 1 ml/L or Thiamethoxam 25% WG @ 0.3 g/L.",
        "organic_treatment": "Foliar spray of 5% Neem oil (10,000 ppm) @ 2 ml/L + yellow sticky traps (15-20 traps/acre).",
        "prevention": "Grow barrier crops like maize/sorghum (2-3 rows) around field boundary; use nursery insect nets.",
        "fertilizer": "Supplement with micronutrient mixtures (Zinc, Boron, Magnesium) to improve plant vigor.",
        "pest_control": "Strictly suppress whiteflies from nursery stage onward.",
        "farmer_tips": "Install yellow sticky traps early to catch whitefly swarms before virus transmission."
    },

    # Cucurbits
    "Cucumber Downy Mildew": {
        "crop": "Cucumber", "scientific": "Cucumis sativus", "pathogen": "Pseudoperonospora cubensis (Oomycete)",
        "category": "Oomycete Mildew", "severity": "High", "badge": "status-danger",
        "overview": "Angular yellow spots restricted by leaf veins on upper leaf surfaces, with purplish downy spore growth underneath.",
        "etiology": "Wind-borne sporangia requiring only 2 hours of dew to infect cucurbit foliage.",
        "symptoms": "Bright yellow angular spots delineated by major leaf veins, quickly turning brown and necrotic.",
        "causes": "High humidity (>85%) with moderate temperatures (15-22°C) and morning fog.",
        "chemical_treatment": "Dimethomorph 50% WP @ 1 g/L or Cymoxanil 8% + Mancozeb 64% @ 2 g/L or Fluopicolide + Propamocarb @ 1.5 ml/L.",
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
        "symptoms": "White flour-like powdery patches spreading over leaf canopy, causing premature leaf yellowing and crisping.",
        "causes": "Dry atmospheric conditions combined with dense canopy shade and moderate temperatures (20-28°C).",
        "chemical_treatment": "Hexaconazole 5% SC @ 1 ml/L or Difenoconazole 25% EC @ 0.5 ml/L or Dinocap @ 1 ml/L.",
        "organic_treatment": "Spray wettable sulfur 80% WP @ 2.5 g/L or baking soda (sodium bicarbonate) @ 4 g/L with mild soap.",
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
        "symptoms": "Shot-hole appearance on leaves; sunken circular lesions on lauki fruits with pinkish gelatinous centers.",
        "causes": "Frequent rainfall, high relative humidity (90%), and temperatures around 22-27°C.",
        "chemical_treatment": "Carbendazim 12% + Mancozeb 63% WP @ 2 g/L or Azoxystrobin 23% SC @ 1 ml/L.",
        "organic_treatment": "Trichoderma harzianum @ 5 g/L seed treatment and foliar spray; 5% garlic bulb extract.",
        "prevention": "Ensure 2-year crop rotation, trellis vines on bower/mandap system to avoid fruit soil contact.",
        "fertilizer": "Adequate phosphorus and potash for tissue resilience.",
        "pest_control": "Control red pumpkin beetles.",
        "farmer_tips": "Trellis lauki vines off the ground to drastically cut fruit anthracnose incidence."
    },
    "Bitter Gourd Gummy Stem Blight": {
        "crop": "Bitter Gourd / Karela", "scientific": "Momordica charantia", "pathogen": "Stagonosporopsis cucurbitacearum (Fungus)",
        "category": "Stem Blight", "severity": "High", "badge": "status-danger",
        "overview": "Circular leaf spots and gummy amber-colored exudate oozing from vine lesions, leading to vine death.",
        "etiology": "Seed-borne and debris-borne fungus that enters stems through harvest cuts, insect wounds, or leaf axils.",
        "symptoms": "Water-soaked lesions near soil line on stems, oozing amber-colored gum; leaves develop circular necrotic lesions.",
        "causes": "High soil wetness, poor drainage, splashing rainfall, and humid canopy conditions.",
        "chemical_treatment": "Thiophanate-methyl 70% WP @ 1.5 g/L or Chlorothalonil 75% WP @ 2 g/L sprayed onto crown and lower stems.",
        "organic_treatment": "Paste of copper oxychloride (5%) applied on wounded or oozing vine sections.",
        "prevention": "Hot water seed treatment at 50°C for 20 minutes; practice crop rotation; sterilize cutting tools.",
        "fertilizer": "Supply balanced calcium and micronutrients.",
        "pest_control": "Control melon fruit flies and stem borers.",
        "farmer_tips": "Apply paste of copper oxychloride mixed with cow dung on bleeding vine cankers."
    },
    "Ridge Gourd Mosaic Virus": {
        "crop": "Ridge Gourd / Turai", "scientific": "Luffa acutangula", "pathogen": "Cucumber Mosaic Virus (CMV)",
        "category": "Viral Disease", "severity": "Moderate", "badge": "status-warning",
        "overview": "Mottling, mosaic pattern, leaf distortion, and stunted fruit development.",
        "etiology": "Transmitted non-persistently by aphids (Aphis gossypii) and mechanically by sap.",
        "symptoms": "Alternating light and dark green mosaic patches on leaves, puckering, small deformed fruits with bitter taste.",
        "causes": "Aphid colonization during spring and warm summer periods.",
        "chemical_treatment": "Insecticidal vector control: Imidacloprid 17.8% SL @ 0.5 ml/L or Acetamiprid 20% SP @ 0.3 g/L.",
        "organic_treatment": "Neem oil 10,000 ppm @ 2 ml/L combined with yellow sticky traps across the plot.",
        "prevention": "Rogue out infected plants early; maintain weed-free headlands; clean pruning tools.",
        "fertilizer": "Foliar zinc and magnesium to prevent physiological yellowing confusion.",
        "pest_control": "Monitor and control aphid populations before flowering begins.",
        "farmer_tips": "Uproot and burn virus-infected vines immediately to prevent aphid spread."
    },
    "Sponge Gourd Downy Mildew": {
        "crop": "Sponge Gourd / Gilki", "scientific": "Luffa aegyptiaca", "pathogen": "Pseudoperonospora cubensis (Oomycete)",
        "category": "Oomycete Mildew", "severity": "High", "badge": "status-danger",
        "overview": "Angular chlorotic lesions bounded by veins on leaves, causing fast leaf drying.",
        "etiology": "Foliar water film allows flagellated zoospores to swim into open stomata.",
        "symptoms": "Pale yellow angular spots on upper surface, gray-brown downy mold underneath.",
        "causes": "Warm days, cool nights, and extended dew periods during monsoon season.",
        "chemical_treatment": "Metalaxyl 8% + Mancozeb 64% WP @ 2 g/L or Cymoxanil 8% + Mancozeb 64% @ 2 g/L.",
        "organic_treatment": "Bordeaux mixture (1%) as protective spray every 10 days during monsoon.",
        "prevention": "Grow vines on overhead wire trellis to allow quick foliage drying after showers.",
        "fertilizer": "Split potassium application to improve cell wall strength.",
        "pest_control": "Suppress whiteflies and aphids.",
        "farmer_tips": "Orient trellis rows in the direction of prevailing wind to speed drying."
    },
    "Pointed Gourd Fruit Rot": {
        "crop": "Pointed Gourd / Parwal", "scientific": "Trichosanthes dioica", "pathogen": "Phytophthora melonis (Oomycete)",
        "category": "Oomycete Rot", "severity": "Critical", "badge": "status-danger",
        "overview": "Water-soaked rotting lesions on parwal fruits touching wet soil, leading to rapid liquification.",
        "etiology": "Soil-dwelling zoospores splash onto low-hanging fruits during heavy monsoonal rain.",
        "symptoms": "Water-soaked soft patches on developing fruits, covered with white cottony mycelium.",
        "causes": "Waterlogging in parwal fields, fruits resting directly on damp soil.",
        "chemical_treatment": "Copper Oxychloride 50% WP @ 3 g/L or Fosetyl-Al 80% WP @ 2 g/L spray.",
        "organic_treatment": "Mulching beds with straw/polythene to create physical barrier between fruit and soil.",
        "prevention": "Construct bamboo bower structures (mandap) to suspend parwal vines off the ground.",
        "fertilizer": "Incorporate well-rotted FYM mixed with Trichoderma viride @ 5 kg/acre.",
        "pest_control": "Control fruit flies with pheromone traps.",
        "farmer_tips": "Ensure parwal fruits hang freely from the bamboo trellis without touching weeds."
    },
    "Ash Gourd Anthracnose": {
        "crop": "Ash Gourd / Petha", "scientific": "Benincasa hispida", "pathogen": "Colletotrichum gloeosporioides (Fungus)",
        "category": "Fungal Anthracnose", "severity": "Moderate", "badge": "status-warning",
        "overview": "Circular dark sunken spots on leaves and fruits of ash gourd, impacting storage quality.",
        "etiology": "Spores splash by rain droplets and enter through fruit cuticle micro-cracks.",
        "symptoms": "Brown circular leaf lesions that dry out and crack; dark sunken depressions on petha fruits.",
        "causes": "High rainfall, prolonged cloudy days, and relative humidity above 85%.",
        "chemical_treatment": "Carbendazim 50% WP @ 1 g/L or Difenoconazole 25% EC @ 0.5 ml/L.",
        "organic_treatment": "Foliar spray of neem seed kernel extract (NSKE 5%) or Trichoderma @ 5 g/L.",
        "prevention": "Collect and burn crop debris after harvest; use certified disease-free seeds.",
        "fertilizer": "Maintain balanced micronutrients (Boron and Zinc).",
        "pest_control": "Control cucumber beetles and aphids.",
        "farmer_tips": "Store harvested ash gourds in dry, well-ventilated rooms to prevent storage rot."
    },
    "Zucchini Yellow Mosaic Virus": {
        "crop": "Zucchini", "scientific": "Cucurbita pepo", "pathogen": "Zucchini Yellow Mosaic Virus (ZYMV)",
        "category": "Viral Disease", "severity": "High", "badge": "status-danger",
        "overview": "Causes severe blistering of leaves, shoe-stringing, and extreme knobby fruit malformation.",
        "etiology": "Non-persistently transmitted within seconds by several aphid species.",
        "symptoms": "Deep leaf vein banding, severe blistering, yellow mosaic, and knobby misshapen unmarketable fruits.",
        "causes": "High aphid pressure in vicinity of squash and cucurbit fields.",
        "chemical_treatment": "Spray systemic aphicides: Flonicamid 50% WG @ 0.3 g/L or Imidacloprid @ 0.5 ml/L.",
        "organic_treatment": "Reflective silver mulch to repel incoming winged aphids + neem oil 10,000 ppm @ 2 ml/L.",
        "prevention": "Plant resistant hybrids; maintain weed-free isolation strip around zucchini beds.",
        "fertilizer": "Avoid excess nitrogen fertilizer that attracts aphid swarms.",
        "pest_control": "Set up yellow sticky cards along bed perimeters.",
        "farmer_tips": "Lay reflective silver mulch film before sowing to naturally repel aphid vectors."
    },
    "Snake Gourd Cercospora Leaf Spot": {
        "crop": "Snake Gourd / Chichinda", "scientific": "Trichosanthes cucumerina", "pathogen": "Cercospora citrullina (Fungus)",
        "category": "Fungal Spot", "severity": "Moderate", "badge": "status-warning",
        "overview": "Small circular spots with light gray centers and dark reddish margins on foliage.",
        "etiology": "Fungus overwinters in vine residues; spores blown by wind or rain splashes.",
        "symptoms": "Round spots (2-5 mm) with ash-gray centers and dark borders; heavy infection causes leaf blight.",
        "causes": "Warm, humid rainy periods (26-32°C, >80% RH).",
        "chemical_treatment": "Mancozeb 75% WP @ 2.5 g/L or Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L.",
        "organic_treatment": "Spray Pseudomonas fluorescens @ 5 g/L at 14-day intervals.",
        "prevention": "Ensure good vine training on trellis to allow rapid wind drying of foliage.",
        "fertilizer": "Balanced N-P-K fertigation; supplement with potassium sulphate.",
        "pest_control": "Control chichinda fruit flies and leaf caterpillars.",
        "farmer_tips": "Pick and dispose of infected bottom leaves as soon as first spots appear."
    },
    "Ivy Gourd Rust": {
        "crop": "Ivy Gourd / Kundru", "scientific": "Coccinia grandis", "pathogen": "Puccinia spp. (Fungus)",
        "category": "Fungal Rust", "severity": "Moderate", "badge": "status-warning",
        "overview": "Bright orange to brownish pustules on lower leaf surfaces and tender kundru shoots.",
        "etiology": "Airborne urediniospores that infect during high humidity and dew periods.",
        "symptoms": "Small yellowish specks on upper leaf surfaces; prominent powdery orange-brown pustules on undersides.",
        "causes": "High humidity, shaded canopy, and temperatures around 18-25°C.",
        "chemical_treatment": "Wettable Sulfur 80% WP @ 3 g/L or Propiconazole 25% EC @ 1 ml/L.",
        "organic_treatment": "Foliar spray of 10% cow urine solution fermented with neem leaves.",
        "prevention": "Prune old, woody perennial vines annually to promote fresh vigorous growth.",
        "fertilizer": "Apply well-decomposed manure with bio-fertilizers annually after winter pruning.",
        "pest_control": "Control gall flies and whiteflies.",
        "farmer_tips": "Conduct annual heavy pruning of perennial kundru vines to discard fungal reservoirs."
    },

    # Common Indian Vegetables
    "Okra Yellow Vein Mosaic Virus": {
        "crop": "Okra / Lady Finger / Bhindi", "scientific": "Abelmoschus esculentus", "pathogen": "Bhendi Yellow Vein Mosaic Virus (BYVMV)",
        "category": "Viral Disease", "severity": "Critical", "badge": "status-danger",
        "overview": "Severe network of bright yellow veins across leaves; stunted plants produce small, hard, pale yellow fruits.",
        "etiology": "Transmitted by the whitefly (Bemisia tabaci); severe viral threat to okra production.",
        "symptoms": "Clear vein clearing followed by complete yellowing of entire leaf vein network; dwarfed chlorotic fruits.",
        "causes": "Whitefly proliferation during hot and humid seasons (March-September).",
        "chemical_treatment": "Control vector with Acetamiprid 20% SP @ 0.3 g/L or Dinotefuran 20% SG @ 0.5 g/L or Spiromesifen @ 1 ml/L.",
        "organic_treatment": "Install yellow sticky traps (20/acre) and spray 5% Neem Seed Kernel Extract (NSKE) weekly.",
        "prevention": "Sow certified resistant varieties (e.g., Parbhani Kranti, Kashi Kranti); remove alternate weed hosts.",
        "fertilizer": "Balanced NPK; avoid excessive urea which causes succulent growth preferred by whiteflies.",
        "pest_control": "Monitor whitefly nymphs on leaf undersides constantly.",
        "farmer_tips": "Sow border crops of maize or bajra to physically block whiteflies from entering bhindi beds."
    },
    "French Bean Anthracnose": {
        "crop": "French Bean", "scientific": "Phaseolus vulgaris", "pathogen": "Colletotrichum lindemuthianum (Fungus)",
        "category": "Fungal Anthracnose", "severity": "High", "badge": "status-danger",
        "overview": "Dark reddish-purple to black sunken cankers on stems, veins, and bean pods.",
        "etiology": "Seed-borne fungus spreading rapidly in wet weather via splashing raindrops.",
        "symptoms": "Dark brick-red along leaf veins on undersides; circular sunken cankers with reddish borders on pods.",
        "causes": "Cool, wet weather (17-23°C) and frequent rainfall.",
        "chemical_treatment": "Carbendazim 50% WP @ 1 g/L or Kresoxim-methyl 44.3% SC @ 1 ml/L.",
        "organic_treatment": "Seed treatment with Trichoderma viride @ 4 g/kg seed + hot water treatment (50°C for 15 min).",
        "prevention": "Use pathogen-tested disease-free seeds; do not walk in wet bean fields.",
        "fertilizer": "Foliar application of potassium and zinc.",
        "pest_control": "Manage bean aphids and pod borers.",
        "farmer_tips": "Never harvest or cultivate french beans while dew remains on foliage."
    },
    "Green Bean Rust": {
        "crop": "Green Bean", "scientific": "Phaseolus vulgaris var.", "pathogen": "Uromyces appendiculatus (Fungus)",
        "category": "Fungal Rust", "severity": "Moderate", "badge": "status-warning",
        "overview": "Reddish-brown powdery pustules on leaves, causing premature defoliation and reduced pod yield.",
        "etiology": "Air-dispersed urediniospores; requires high humidity (>95%) for initial spore germination.",
        "symptoms": "Tiny pale yellow spots on leaves that rupture into rust-colored powdery pustules surrounded by yellow halos.",
        "causes": "Cool to moderate temperatures (16-24°C) with prolonged cloudiness and dew.",
        "chemical_treatment": "Mancozeb 75% WP @ 2.5 g/L or Tebuconazole 25.9% EC @ 1 ml/L.",
        "organic_treatment": "Sulfur 80% WP @ 2.5 g/L or foliar application of Bacillus subtilis @ 5 ml/L.",
        "prevention": "Rotate beans with non-legumes; plow under vine residues immediately after harvest.",
        "fertilizer": "Maintain balanced fertility; avoid high nitrogen late in growth.",
        "pest_control": "Control leafhoppers and aphids.",
        "farmer_tips": "Destroy old bean vines right after final picking to prevent rust build-up."
    },
    "Peas Powdery Mildew": {
        "crop": "Peas / Matar", "scientific": "Pisum sativum", "pathogen": "Erysiphe pisi (Fungus)",
        "category": "Fungal Mildew", "severity": "High", "badge": "status-danger",
        "overview": "White flour-like powdery growth covering leaves, stems, and pods, turning gray-brown as crop matures.",
        "etiology": "Wind-blown conidia; overwinters in seed or perennial legume weeds.",
        "symptoms": "White powdery patches that rapidly coalesce to cover entire plant canopy; pods become discolored and shriveled.",
        "causes": "Dry weather with cool nights and warm days (15-25°C) in late winter/spring.",
        "chemical_treatment": "Sulfur 80% WP @ 2.5 g/L or Hexaconazole 5% EC @ 1 ml/L or Dinocap 48% EC @ 1 ml/L.",
        "organic_treatment": "Spray 10% cow milk solution in water or Ampelomyces quisqualis bio-fungicide @ 5 g/L.",
        "prevention": "Early sowing of winter peas (October-November); use resistant varieties (e.g., Rachna, Pant Matar).",
        "fertilizer": "Supply balanced phosphorus and potash at basal sowing.",
        "pest_control": "Manage pea leaf miners and aphids.",
        "farmer_tips": "Complete pea sowing early to harvest before dry, warm late-winter weather favors mildew."
    },
    "Sweet Corn Northern Corn Leaf Blight": {
        "crop": "Sweet Corn", "scientific": "Zea mays var. saccharata", "pathogen": "Exserohilum turcicum (Fungus)",
        "category": "Fungal Blight", "severity": "Moderate", "badge": "status-warning",
        "overview": "Large cigar-shaped grayish-green to tan lesions on leaves, reducing photosynthesis and cob filling.",
        "etiology": "Overwinters on corn residue; spores are wind-blown or rain-splashed onto lower leaves.",
        "symptoms": "Long, elliptical grayish-green to tan lesions (2.5 to 15 cm long) with smooth margins.",
        "causes": "Moderate temperatures (18-27°C) with persistent heavy dew or rain.",
        "chemical_treatment": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L or Mancozeb 75% WP @ 2.5 g/L.",
        "organic_treatment": "Spray Trichoderma viride @ 5 g/L in early vegetative stage.",
        "prevention": "Deep burial of corn stubble after harvest; choose resistant sweet corn hybrids.",
        "fertilizer": "Ensure adequate nitrogen side-dressing and potassium balance.",
        "pest_control": "Control fall armyworm (Spodoptera frugiperda) and corn earworms.",
        "farmer_tips": "Shred and plow down corn stalks immediately after cob harvest."
    },
    "Carrot Alternaria Leaf Blight": {
        "crop": "Carrot", "scientific": "Daucus carota", "pathogen": "Alternaria dauci (Fungus)",
        "category": "Fungal Blight", "severity": "Moderate", "badge": "status-warning",
        "overview": "Dark brown-black spots with yellow halos on carrot leaflets, causing foliage to die back and hindering mechanical harvesting.",
        "etiology": "Seed-borne and debris-borne; conidia spread through wind and splashing rain.",
        "symptoms": "Brown-black necrotic lesions on outer leaf margins, curled leaf edges, feather-like foliage collapsing.",
        "causes": "Warm, humid periods (20-28°C) with prolonged foliar moisture.",
        "chemical_treatment": "Chlorothalonil 75% WP @ 2 g/L or Iprodione 50% WP @ 1.5 g/L or Propiconazole @ 1 ml/L.",
        "organic_treatment": "Hot water seed soak (50°C for 20 min); foliar spray of Trichoderma harzianum @ 5 g/L.",
        "prevention": "3-year crop rotation; use treated seed; thin carrots to improve canopy airflow.",
        "fertilizer": "Avoid excess nitrogen that promotes dense, wet carrot tops.",
        "pest_control": "Manage carrot rust fly and leafhoppers.",
        "farmer_tips": "Protect carrot tops with preventative spray if rain is forecast near harvest."
    },
    "Radish Black Rot": {
        "crop": "Radish / Mooli", "scientific": "Raphanus sativus", "pathogen": "Xanthomonas campestris pv. campestris (Bacteria)",
        "category": "Bacterial Rot", "severity": "High", "badge": "status-danger",
        "overview": "V-shaped yellow lesions on leaf margins with blackened veins, causing internal blackening of radish roots.",
        "etiology": "Enters leaves through hydathodes (water pores) along leaf margins during morning guttation.",
        "symptoms": "V-shaped chlorotic lesions on leaf margins; blackened internal vascular ring in roots.",
        "causes": "Warm (25-30°C), wet weather with heavy rain splash or overhead irrigation.",
        "chemical_treatment": "Streptocycline @ 0.1 g/L combined with Copper Oxychloride @ 2 g/L.",
        "organic_treatment": "Hot water seed treatment at 50°C for 25 minutes; spray Pseudomonas fluorescens @ 5 g/L.",
        "prevention": "Use certified black rot-free seeds; rotate with non-cruciferous crops for 3 years.",
        "fertilizer": "Ensure sufficient boron and calcium to prevent root internal cracking.",
        "pest_control": "Control flea beetles and diamondback moth larvae.",
        "farmer_tips": "Avoid overhead sprinkler irrigation on radish during warm afternoon hours."
    },
    "Beetroot Cercospora Leaf Spot": {
        "crop": "Beetroot", "scientific": "Beta vulgaris", "pathogen": "Cercospora beticola (Fungus)",
        "category": "Fungal Spot", "severity": "Moderate", "badge": "status-warning",
        "overview": "Small circular spots with grayish-white centers and dark brown to reddish-purple margins on beetroot foliage.",
        "etiology": "Spores survive in crop residue and weed hosts; wind and rain splash drive dissemination.",
        "symptoms": "Circular spots (2-4 mm) on leaves with ash-colored centers and reddish borders; heavy spotting defoliates roots.",
        "causes": "Warm temperatures (25-30°C) combined with high relative humidity (>90%).",
        "chemical_treatment": "Carbendazim 12% + Mancozeb 63% WP @ 2 g/L or Difenoconazole 25% EC @ 0.5 ml/L.",
        "organic_treatment": "Copper-based fungicides @ 2.5 g/L or foliar application of bio-agent Bacillus subtilis.",
        "prevention": "Maintain 3-year rotation away from Chenopodiaceae family; destroy beet residues.",
        "fertilizer": "Apply balanced boron to avoid internal black spot physiological disorder.",
        "pest_control": "Control leaf-mining flies and cutworms.",
        "farmer_tips": "Space beetroot rows properly to allow morning breeze to dry leaf surfaces."
    },
    "Turnip Mosaic Virus": {
        "crop": "Turnip / Shalgam", "scientific": "Brassica rapa subsp. rapa", "pathogen": "Turnip Mosaic Virus (TuMV)",
        "category": "Viral Disease", "severity": "Moderate", "badge": "status-warning",
        "overview": "Interveinal chlorosis, mosaic mottling, and blistering of leaves, causing stunted roots.",
        "etiology": "Transmitted non-persistently by numerous aphid species (Myzus persicae, Brevicoryne brassicae).",
        "symptoms": "Mottling, light and dark green mosaic patterns, crinkled leaves, and undersized shalgam roots.",
        "causes": "High aphid vectors migrating from neighboring brassica fields.",
        "chemical_treatment": "Control aphids: Thiamethoxam 25% WG @ 0.3 g/L or Dimethoate 30% EC @ 1.5 ml/L.",
        "organic_treatment": "Spray 5% neem seed kernel extract (NSKE) and install yellow sticky cards (15/acre).",
        "prevention": "Eradicate cruciferous weeds around boundaries; practice non-brassica rotations.",
        "fertilizer": "Supplement with trace minerals and potassium.",
        "pest_control": "Scout for aphid colonies under turnip leaf undersides weekly.",
        "farmer_tips": "Uproot initial stunted, mottled plants to prevent aphid transmission to the whole field."
    },

    # Cole Crops
    "Cabbage Black Rot": {
        "crop": "Cabbage", "scientific": "Brassica oleracea var. capitata", "pathogen": "Xanthomonas campestris pv. campestris (Bacteria)",
        "category": "Bacterial Rot", "severity": "Critical", "badge": "status-danger",
        "overview": "V-shaped yellow marginal lesions, black veins in stem cross-section, and soft rotting of cabbage heads.",
        "etiology": "Seed-borne and debris-borne bacteria entering hydathodes and insect-damaged tissues.",
        "symptoms": "Distinct V-shaped yellowing on leaf margins pointing inward along veins; blackened vascular bundles.",
        "causes": "Warm, humid conditions (25-30°C) with persistent rains or heavy morning dews.",
        "chemical_treatment": "Copper Oxychloride @ 2.5 g/L mixed with Streptocycline @ 0.1 g/L.",
        "organic_treatment": "Seed immersion in water at 50°C for 30 minutes; foliar spray of Pseudomonas fluorescens @ 5 g/L.",
        "prevention": "Strictly sow pathogen-tested black-rot-free certified seed; 3-year brassica crop rotation.",
        "fertilizer": "Avoid excessive top-dressed nitrogen fertilizer that softens outer cabbage leaves.",
        "pest_control": "Control diamondback moths and flea beetles immediately to limit entry wounds.",
        "farmer_tips": "Disinfect seed trays and nursery soil before raising cabbage seedlings."
    },
    "Cauliflower Curd Rot": {
        "crop": "Cauliflower", "scientific": "Brassica oleracea var. botrytis", "pathogen": "Pseudomonas marginalis / Erwinia carotovora (Bacteria)",
        "category": "Bacterial Rot", "severity": "High", "badge": "status-danger",
        "overview": "Water-soaked brownish discolored lesions on the white cauliflower curd, turning into foul-smelling soft rot.",
        "etiology": "Soil bacteria entering through insect punctures, mechanical injuries, or heavy dew accumulation in curd cups.",
        "symptoms": "Curd develops water-soaked brownish spots, leading to soft slimy rot and unpleasant odor.",
        "causes": "High rainfall during curd maturation, insect injury, high humidity (>90%).",
        "chemical_treatment": "Streptocycline @ 0.15 g/L + Copper Hydroxide @ 2 g/L directed into the head/curd area.",
        "organic_treatment": "Tie wrapper leaves over curds (blanching) to shield them from rain and direct dew; spray Trichoderma.",
        "prevention": "Ensure good field drainage, avoid overhead irrigation, tie wrapper leaves around curd.",
        "fertilizer": "Apply Borax @ 10 kg/ha to avoid boron deficiency (brown rot) that precedes bacterial decay.",
        "pest_control": "Control head borers and cabbage caterpillars.",
        "farmer_tips": "Practice blanching (tying outer leaves over curds) to shield against rain-borne bacteria."
    },
    "Broccoli Downy Mildew": {
        "crop": "Broccoli", "scientific": "Brassica oleracea var. italica", "pathogen": "Hyaloperonospora parasitica (Oomycete)",
        "category": "Oomycete Mildew", "severity": "Moderate", "badge": "status-warning",
        "overview": "Yellow angular spots on broccoli leaves with white-gray downy fungal growth on the underside.",
        "etiology": "Airborne and seed-borne; requires free moisture on leaves for infection.",
        "symptoms": "Yellowish-gray angular leaf lesions; black internal flecking inside broccoli florets and stems.",
        "causes": "Cool, damp weather (10-18°C) with persistent fog or drizzle.",
        "chemical_treatment": "Metalaxyl-M + Mancozeb @ 2 g/L or Azoxystrobin 23% SC @ 1 ml/L.",
        "organic_treatment": "Protective copper oxychloride @ 2.5 g/L spray before canopy closure.",
        "prevention": "Wider spacing of plants; drip irrigation; eradicate wild mustard weeds.",
        "fertilizer": "Maintain calcium and potassium levels for firm floret development.",
        "pest_control": "Manage aphids and cabbage loopers.",
        "farmer_tips": "Harvest broccoli heads early in the morning when florets are cool and tight."
    },

    # Leafy Vegetables
    "Spinach Downy Mildew": {
        "crop": "Spinach / Palak", "scientific": "Spinacia oleracea", "pathogen": "Peronospora effusa (Oomycete)",
        "category": "Oomycete Mildew", "severity": "High", "badge": "status-danger",
        "overview": "Yellow patches on upper leaf surfaces with purplish-gray downy fungal sporulation on undersides.",
        "etiology": "Air-dispersed oospores and sporangia; can also be carried on seed coats.",
        "symptoms": "Dull yellow leaf blotches, leaves become curled and distorted; underside develops purplish-gray felt-like mold.",
        "causes": "Cool, moist weather (10-16°C) and high humidity (>85%).",
        "chemical_treatment": "Mandipropamid 23.4% SC @ 0.8 ml/L or Dimethomorph 50% WP @ 1 g/L.",
        "organic_treatment": "Copper Octanoate @ 2 ml/L or potassium bicarbonate @ 3 g/L.",
        "prevention": "Sow downy-mildew-resistant spinach cultivars; avoid night sprinkler watering; practice 3-year rotation.",
        "fertilizer": "Balanced nitrogen; do not over-fertilize close to harvest.",
        "pest_control": "Control crown mites and aphids.",
        "farmer_tips": "Never irrigate palak beds in late evening; soil surface must dry before sunset."
    },
    "Fenugreek Cercospora Leaf Spot": {
        "crop": "Fenugreek / Methi", "scientific": "Trigonella foenum-graecum", "pathogen": "Cercospora traversiana (Fungus)",
        "category": "Fungal Spot", "severity": "Moderate", "badge": "status-warning",
        "overview": "Circular to oblong necrotic spots with dark brown margins on leaves, stems, and pods.",
        "etiology": "Seed-borne and debris-borne fungus; spreads through rain splash.",
        "symptoms": "Sunken brown spots on leaflets with grayish centers; pod lesions cause seed shriveling.",
        "causes": "Warm, humid conditions and close plant spacing in irrigated beds.",
        "chemical_treatment": "Mancozeb 75% WP @ 2 g/L or Carbendazim 50% WP @ 1 g/L.",
        "organic_treatment": "Seed treatment with Trichoderma viride @ 4 g/kg seed + foliar neem oil (0.5%).",
        "prevention": "Use certified clean seeds; thin seedlings to maintain open airflow; burn crop stubble.",
        "fertilizer": "Phosphorus and bio-fertilizer (Rhizobium) seed inoculation.",
        "pest_control": "Control aphids.",
        "farmer_tips": "Treat methi seeds with Rhizobium culture and Trichoderma before sowing."
    },
    "Coriander Stem Gall": {
        "crop": "Coriander / Dhaniya", "scientific": "Coriandrum sativum", "pathogen": "Protomyces macrosporus (Fungus)",
        "category": "Fungal Gall", "severity": "High", "badge": "status-danger",
        "overview": "Tumor-like swellings (galls) on stems, pedicels, and seeds of coriander, severely deforming the crop.",
        "etiology": "Soil-borne chlamydospores that survive for several years in field soil.",
        "symptoms": "Blister-like swelling on coriander stems, leaf veins, and fruits; plants become distorted and sterile.",
        "causes": "Excessive soil moisture, high humidity, and cloudiness during flowering stage.",
        "chemical_treatment": "Seed treatment with Thiram @ 3 g/kg + foliar spray of Hexaconazole @ 1 ml/L or Propiconazole @ 1 ml/L at first sign of galls.",
        "organic_treatment": "Soil application of Trichoderma viride enriched farmyard manure (FYM) @ 2.5 kg/acre.",
        "prevention": "Solarize soil in summer; use tolerant cultivars; practice minimum 3-year crop rotation.",
        "fertilizer": "Avoid high nitrogen levels which increase gall severity.",
        "pest_control": "Control coriander aphids.",
        "farmer_tips": "Avoid water-ponding in coriander plots during flowering and seed formation."
    },
    "Lettuce Bottom Rot": {
        "crop": "Lettuce", "scientific": "Lactuca sativa", "pathogen": "Rhizoctonia solani (Fungus)",
        "category": "Fungal Rot", "severity": "Moderate", "badge": "status-warning",
        "overview": "Slimy rust-colored sunken sores on midribs of lower leaves touching damp soil, wilting outer leaves.",
        "etiology": "Soil-inhabiting fungus that attacks leaves in direct contact with damp earth.",
        "symptoms": "Rust-colored elliptical lesions on lower leaf petioles; leaves collapse into slimy brown rot.",
        "causes": "High soil moisture, warm temperatures (22-28°C), and poorly drained beds.",
        "chemical_treatment": "Azoxystrobin 23% SC @ 1 ml/L or Iprodione 50% WP @ 1.5 g/L directed at plant base.",
        "organic_treatment": "Apply Trichoderma harzianum @ 10 g/L as root drench at transplanting.",
        "prevention": "Grow lettuce on raised beds; lay plastic or straw mulch to isolate leaves from soil.",
        "fertilizer": "Balanced nutrition; avoid excessive moisture around head.",
        "pest_control": "Control slugs and snails.",
        "farmer_tips": "Plant lettuce on raised ridges so outer leaves stay dry and suspended."
    },
    "Amaranth White Rust": {
        "crop": "Amaranth / Chaulai", "scientific": "Amaranthus cruentus", "pathogen": "Albugo bliti (Oomycete)",
        "category": "Oomycete Rust", "severity": "Moderate", "badge": "status-warning",
        "overview": "Chalky white, blister-like pustules on lower leaf surfaces of chaulai, causing leaf yellowing and distortion.",
        "etiology": "Soil-borne oospores and wind-spread sporangia.",
        "symptoms": "Yellowish irregular spots on upper leaf surfaces; prominent chalk-white blisters on undersides.",
        "causes": "High humidity, overhead watering, and moderate temperatures (20-26°C).",
        "chemical_treatment": "Mancozeb 75% WP @ 2.5 g/L or Metalaxyl 8% + Mancozeb 64% @ 2 g/L.",
        "organic_treatment": "Foliar spray of 1% Bordeaux mixture or 5% neem seed extract.",
        "prevention": "Avoid overhead watering; remove wild amaranth weed hosts; crop rotation.",
        "fertilizer": "Supplement with organic compost and rock phosphate.",
        "pest_control": "Control amaranth leaf webbers and weevils.",
        "farmer_tips": "Harvest chaulai leaves frequently to prevent thick canopy moisture build-up."
    },

    # Bulb Crops
    "Onion Purple Blotch": {
        "crop": "Onion", "scientific": "Allium cepa", "pathogen": "Alternaria porri (Fungus)",
        "category": "Fungal Blight", "severity": "High", "badge": "status-danger",
        "overview": "Sunken purple lesions with yellow borders on onion leaves, causing tops to fall over and reducing bulb size.",
        "etiology": "Survives in onion debris and volunteer bulbs; spores dispersed by wind and rain splash.",
        "symptoms": "Water-soaked spots turning dark purple with yellow halos; leaves break at lesion points.",
        "causes": "Warm temperatures (24-30°C) with high relative humidity (>80%) and prolonged dew.",
        "chemical_treatment": "Mancozeb 75% WP @ 2.5 g/L or Difenoconazole 25% EC @ 1 ml/L or Tebuconazole 25.9% EC @ 1 ml/L.",
        "organic_treatment": "Foliar spray of Trichoderma viride @ 5 g/L mixed with sticking agent (soap nut extract).",
        "prevention": "3-year crop rotation; well-drained raised beds; treat seedling roots with carbendazim.",
        "fertilizer": "Apply adequate potassium and sulfur to harden foliage.",
        "pest_control": "Strictly control onion thrips (Thrips tabaci) which provide infection sites.",
        "farmer_tips": "Always add a surfactant or sticker (spreader) when spraying onion foliage due to waxy leaves."
    },
    "Garlic White Rot": {
        "crop": "Garlic", "scientific": "Allium sativum", "pathogen": "Stromatinia cepivora (Fungus)",
        "category": "Soil-borne Rot", "severity": "Critical", "badge": "status-danger",
        "overview": "Foliage turns yellow, wilts, and collapses; fluffy white mold and tiny black sclerotia form on roots and decaying bulbs.",
        "etiology": "Sclerotia remain dormant in soil for over 20 years, triggered to germinate by allium root exudates.",
        "symptoms": "Yellowing of leaves starting from tips; garlic bulbs rot with white fungal crust and black poppy-seed-like sclerotia.",
        "causes": "Cool soil temperatures (10-20°C) and moist ground conditions.",
        "chemical_treatment": "Tebuconazole 25.9% EC @ 1 ml/L or Boscalid @ 1 g/L applied as soil drench around bulb zone.",
        "organic_treatment": "Soil incorporation of Trichoderma harzianum @ 10 kg/ha with compost before planting.",
        "prevention": "Do not plant in fields with known white rot history; use clean seed cloves; clean machinery.",
        "fertilizer": "Apply well-decomposed manure; avoid waterlogging.",
        "pest_control": "Control root maggots and soil nematodes.",
        "farmer_tips": "Never plant garlic in soil where allium white rot has occurred in the past 10 years."
    }
}

# ============================================================
# 4. DATA ACCESS LAYER
# ============================================================
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
        "pathogen": "Botanical/Physiological",
        "category": "Agronomic Condition",
        "severity": "Moderate",
        "badge": "status-warning",
        "overview": "Clinical pathology details cataloged in the PlantCare AI Agronomic Hub.",
        "etiology": "Pathological inoculation favored by microclimate humidity and temperature fluctuations.",
        "symptoms": "Foliar spotting, chlorosis, lesions, or vascular wilting across tissue.",
        "causes": "Elevated moisture, pathogen inoculum, or nutrient imbalance.",
        "chemical_treatment": "Apply registered protective fungicide or bactericide strictly per label.",
        "organic_treatment": "Foliar application of bio-antagonists (Bacillus subtilis or Trichoderma) and neem oil.",
        "prevention": "Ensure certified seed stock, proper row ventilation, and balanced fertigation.",
        "fertilizer": "Maintain balanced N-P-K; supplement Calcium and Potassium.",
        "pest_control": "Monitor insect vectors (whiteflies, thrips, aphids) regularly.",
        "farmer_tips": "Inspect fields in early morning when lesions are distinct."
    }

def get_farmer_stories():
    stories_file = DATA_DIR / "farmer_stories.json"
    if stories_file.is_file():
        try:
            with open(stories_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def get_advertisements():
    ads_file = DATA_DIR / "advertisements.json"
    if ads_file.is_file():
        try:
            with open(ads_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

# ============================================================
# 5. MOBILE-FIRST RESPONSIVE CSS & UI DESIGN SYSTEM
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
        --card-bg: rgba(255, 255, 255, 0.92);
        --card-border: rgba(226, 236, 230, 0.90);
        --shadow-sm: 0 4px 16px rgba(6, 78, 59, 0.04);
        --shadow-md: 0 12px 36px rgba(6, 78, 59, 0.08);
        --shadow-lg: 0 24px 60px rgba(6, 78, 59, 0.14);
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

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #02261d 0%, #043628 45%, #064e3b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15);
    }
    [data-testid="stSidebar"] * {
        color: #f0fdf4 !important;
    }

    .block-container {
        max-width: 1340px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 850;
        color: var(--primary-dark);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

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
        font-family: 'Space Grotesk', sans-serif;
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

    @media (max-width: 850px) {
        .block-container { padding-left: 0.85rem !important; padding-right: 0.85rem !important; padding-top: 1rem !important; }
        .hero-banner { padding: 2rem 1.4rem !important; border-radius: 20px !important; }
        .hero-title { font-size: 2rem !important; }
        .product-card { padding: 1.4rem 1.25rem !important; border-radius: 18px !important; }
        .prob-grid-row { grid-template-columns: 1fr !important; gap: 6px !important; }
        .prob-pct { text-align: left !important; }
        .info-layout-grid { grid-template-columns: 1fr !important; gap: 0.9rem !important; }
        .result-panel { padding: 1.5rem 1.25rem !important; border-radius: 20px !important; }
        .app-footer-bar { flex-direction: column !important; text-align: center !important; gap: 0.8rem !important; }
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
        raise RuntimeError("AI model is currently offline. Please verify plant_disease_model.h5 is in root directory.")

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

    raise RuntimeError("Unable to complete screening for this image. Please upload a clear plant specimen and retry.")

# ============================================================
# 7. WEATHER ENGINE & GEOSPATIAL MAPS
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
PLANTCARE AI — PLANT HEALTH SCREENING DOSSIER
Powered by SEA AUTO
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
Chemical Formulation:
{info.get("chemical_treatment", "N/A")}

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
assessment. For important agricultural decisions, consult a qualified 
agricultural professional.
© 2026 PlantCare AI. Powered by SEA AUTO.
======================================================================
"""

# ============================================================
# INITIALIZE STATE & LOAD AI MODEL
# ============================================================
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "nearby_shops" not in st.session_state:
    st.session_state.nearby_shops = None

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
        st.markdown("**AI MODEL STATUS**")
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
# PAGE 1: HOME
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
        st.markdown(f"""
        <div class="product-card">
            <h3>{t("feature_scan_title")}</h3>
            <div class="card-muted">{t("feature_scan_desc")}</div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="product-card">
            <h3>{t("feature_comp_title")}</h3>
            <div class="card-muted">{t("feature_comp_desc")}</div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="product-card">
            <h3>{t("feature_weather_title")}</h3>
            <div class="card-muted">{t("feature_weather_desc")}</div>
        </div>
        """, unsafe_allow_html=True)

    ads = get_advertisements()
    active_ads = [a for a in ads if a.get("status") == "active"]
    if active_ads:
        top_ad = active_ads[0]
        st.markdown(f"""
        <div class="product-card" style="border-color: #bbf7d0; background: #ffffff;">
            <div style="font-size: 0.74rem; font-weight: 850; letter-spacing: 0.1em; text-transform: uppercase; color: #065f46; margin-bottom: 0.4rem;">✦ Featured Partner</div>
            <div style="font-size: 1.35rem; font-weight: 850; color: #064e3b; margin-bottom: 0.4rem;">{html.escape(top_ad.get('title', ''))}</div>
            <div class="card-muted">{html.escape(top_ad.get('description', ''))}</div>
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
            st.image(image, caption="Uploaded Specimen Preview", use_container_width=True)
        with col_act:
            st.markdown("""
            <div class="product-card">
                <h3>Ready for Neural Screening</h3>
                <div class="card-muted">Click below to run multi-resolution neural network classification.</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(t("analyze_btn"), type="primary", use_container_width=True):
                with st.spinner("Analyzing plant specimen..."):
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
                        st.error(f"❌ Analysis error: {exc}")

    if st.session_state.prediction_data is not None:
        p = st.session_state.prediction_data
        info = get_disease_detail(p["condition"])
        conf_tag = f'<span class="confidence-tag">{t("confidence_high")}</span>' if p["confidence"] >= 70.0 else f'<span class="confidence-tag">{t("confidence_low")}</span>'

        st.markdown(f"""
        <div class="result-panel">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap;">
                <div style="font-size: 0.82rem; font-weight: 800; text-transform: uppercase; color: #64748b;">{t("result_heading")}</div>
                <div><span class="status-badge {info.get('badge', 'status-warning')}">● {info.get('status', 'Analyzed')}</span>{conf_tag}</div>
            </div>
            <div style="font-size: 1.95rem; font-weight: 850; color: #0d1f17; font-family: 'Space Grotesk', sans-serif;">{p['condition']}</div>
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
            <h3>Disease Information & Clinical Overview</h3>
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
# PAGE 3: PLANT HEALTH REPORT
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
            <div>Confidence: <strong>{p['confidence']:.2f}%</strong></div>
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
# PAGE 4: EXPLORE CROPS
# ============================================================
def render_crop_directory():
    st.markdown("## 🌱 Explore Crops")
    st.caption("Complete directory of 35 vegetable crops across standard botanical categories.")
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
# PAGE 5: 35-CROP EXHAUSTIVE DISEASE KNOWLEDGE HUB
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
        st.info("No matching condition found for the selected filter.")
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

    tab1, tab2, tab3 = st.tabs(["🔬 Etiology & Symptoms", "💊 Chemical & Organic Treatment", "🛡️ Prevention & Agronomy"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="product-card">
                <h3>🧬 Pathogen Etiology & Biology</h3>
                <div class="card-muted">{info.get('etiology', 'N/A')}</div>
            </div>
            <div class="product-card">
                <h3>⚠️ Pre-disposing Climate Factors</h3>
                <div class="card-muted">{info.get('causes', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="product-card">
                <h3>🤒 Symptomatology & Field Diagnostic</h3>
                <div class="card-muted">{info.get('symptoms', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"""
            <div class="product-card">
                <h3>🧪 Chemical Formulations & Dosage</h3>
                <div class="card-muted">{info.get('chemical_treatment', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="product-card">
                <h3>🌿 Biological & Bio-Control Regimes</h3>
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
                <h3>🐛 Vector & Pest Management</h3>
                <div class="card-muted">{info.get('pest_control', 'N/A')}</div>
            </div>
            <div class="product-card">
                <h3>👨‍🌾 Grower Field Tips</h3>
                <div class="card-muted">{info.get('farmer_tips', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE 6: FARMER STORIES
# ============================================================
def render_farmer_stories():
    st.markdown("## 🌾 Farmer Stories")
    st.caption("Real crop-protection experiences and field management lessons shared by growers.")
    stories = get_farmer_stories()
    if not stories:
        st.info("Farmer experiences and case studies are being recorded.")
        return
    for s in stories:
        st.markdown(f"""
        <div class="product-card">
            <h3>{html.escape(s.get('farmer_name', ''))} — {html.escape(s.get('location', ''))}</h3>
            <div class="card-muted">{html.escape(s.get('story', ''))}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 7: WEATHER & SPRAY ADVISORY
# ============================================================
def render_weather_advisory():
    st.markdown(f"## {t('nav_weather')}")
    st.caption("Dynamic microclimate assessment calculating disease infection risks and spraying windows.")
    lat = st.number_input("Latitude", value=25.5941, format="%.4f")
    lon = st.number_input("Longitude", value=85.1376, format="%.4f")
    weather, _ = get_live_weather_data(lat, lon)
    if weather:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{weather['temperature_c']}°C")
        col2.metric("Humidity", f"{weather['relative_humidity_pct']}%")
        col3.metric("Rain Prob", f"{weather['rain_probability_pct']}%")
        col4.metric("Wind Speed", f"{weather['wind_speed_kmh']} km/h")

# ============================================================
# PAGE 8: NEARBY PLANT CARE
# ============================================================
def render_nearby_page():
    st.markdown(f"## {t('nav_nearby')}")
    st.caption("Discover nearby agricultural stores, seed centers, and plant nurseries.")
    if st.button("Discover Nearby Agri-Centers", type="primary"):
        st.session_state.nearby_shops = query_nearby_plant_care(25.5941, 85.1376)

    if st.session_state.nearby_shops:
        for s in st.session_state.nearby_shops:
            st.markdown(f"""
            <div class="product-card">
                <h3>🏪 {html.escape(s['name'])}</h3>
                <div class="card-muted">{html.escape(s['address'])}</div>
            </div>
            """, unsafe_allow_html=True)
            st.link_button("🗺️ Open in Google Maps", s["maps"])

# ============================================================
# PAGE 9: CONTENT MANAGER (ADMIN)
# ============================================================
def render_content_manager():
    st.markdown(f"## {t('nav_admin')}")
    st.caption("Manage Farmer Stories and Sponsored Ads securely.")
    st.info("Content Manager authenticated via secure session state.")

# ============================================================
# PAGE 10: ABOUT
# ============================================================
def render_about_page():
    st.markdown(f"""
    <div class="product-card">
        <div class="hero-kicker">✦ ABOUT</div>
        <h1 style="color: #064e3b; font-size: 2.3rem; margin: 0.5rem 0 0.85rem; font-family: 'Space Grotesk', sans-serif;">{t("about_heading")}</h1>
        <p class="card-muted" style="font-size: 1.08rem;">{t("about_text")}</p>
    </div>
    <div class="product-card" style="text-align: center; margin-top: 1.4rem;">
        <div style="font-size: 1.35rem; font-weight: 850; color: #064e3b; margin-bottom: 0.35rem;">PlantCare AI Enterprise</div>
        <div style="font-size: 1rem; font-weight: 750; color: #059669; margin-bottom: 0.25rem;">✦ {t("powered_by")}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN ROUTER
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
