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
    page_title="PlantCare AI — AI-Based Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Active root directory resolution
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
FARMER_IMAGES_DIR = IMAGES_DIR / "farmers"
AD_IMAGES_DIR = IMAGES_DIR / "advertisements"
CROP_IMAGES_DIR = IMAGES_DIR / "crops"

for directory in [DATA_DIR, IMAGES_DIR, FARMER_IMAGES_DIR, AD_IMAGES_DIR, CROP_IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. MODEL CLASSES (SINGLE SOURCE OF TRUTH)
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

# Comprehensive Clinical Knowledge Base
DEEP_DISEASE_KNOWLEDGE = {
    "Pepper Bell Bacterial Spot": {
        "crop": "Pepper (Bell)",
        "scientific_crop": "Capsicum annuum",
        "category": "Bacterial Infection",
        "pathogen": "Xanthomonas campestris pv. vesicatoria",
        "severity": "High",
        "status": "Disease Detected",
        "badge": "status-danger",
        "overview": "Bacterial Spot of Bell Pepper is a destructive pathogen complex that damages foliar tissue and fruit pods. In high-humidity conditions, it triggers rapid defoliation, exposing fruit clusters to severe solar scalding and creating entry avenues for secondary soft-rot pathogens.",
        "etiology": "Gram-negative, strictly aerobic, rod-shaped bacterium with a single polar flagellum. The pathogen overwinters internally and externally in seed embryos, volunteer nightshade weeds (Solanum nigrum), and undecomposed plant debris. Transmission occurs via wind-driven rain droplets, high-pressure sprinkler spray, and physical friction between workers/tools and damp foliage.",
        "symptoms": "• Initial Foliar Stage: Small (1-3 mm) translucent, water-soaked circular to angular lesions on leaf undersides.\n• Advanced Foliar Stage: Lesions enlarge to 4-6 mm, turning dark brownish-black with prominent chlorotic (yellow) borders. Leaves turn yellow and drop prematurely.\n• Pod / Fruit Stage: Small, blister-like raised eruptions that rupture into rough, brown, crater-like warts (5-8 mm), compromising fruit commercial value.\n• Stem Stage: Elongated, rough, cankerous dark fissures that weaken structural branches.",
        "causes": "Extended canopy wetness periods (> 4 hours), high ambient relative humidity (> 80%), daytime temperatures between 24°C and 32°C, and overhead splash dispersion.",
        "chemical_treatment": "• Protective Barrier: Fixed Copper Hydroxide (53.8% DF @ 2.0-2.5 g/L) or Copper Oxychloride (50% WP @ 3.0 g/L).\n• Anti-Resistance Mix: Tank-mix Copper Hydroxide with Mancozeb (75% WP @ 2.0 g/L) to release free cupric ions against copper-tolerant bacterial populations.\n• Systemic SAR Activator: Acibenzolar-S-methyl (Actigard 50 WG @ 25-35 g/ha) applied preventatively to trigger natural plant immune defenses.\n• Pre-Harvest Interval (PHI): 3 to 7 days depending on regional copper regulations.",
        "organic_treatment": "• Microbial Antagonists: Foliar spray of Bacillus subtilis (QST 713 strain) or Bacillus amyloliquefaciens @ 5 ml/L at 7-day intervals.\n• Botanical Shield: Cold-pressed Neem Oil (10,000 ppm Azadirachtin) @ 3-5 ml/L with an organic wetting agent.\n• Seed Sanitization: Hot water seed soak at 50°C (122°F) for precisely 25 minutes prior to nursery sowing.",
        "prevention": "1. Source exclusively certified pathogen-free seeds.\n2. Adopt subsurface drip irrigation to keep canopies dry.\n3. Enforce a minimum 2 to 3-year crop rotation avoiding all Solanaceae.\n4. Avoid all field operations while morning dew or raindrops persist on foliage.",
        "fertilizer": "Maintain balanced N:P:K (1:1:2 ratio). Strictly avoid high-nitrogen vegetative feeding (excess urea) which produces soft, succulent epidermal cell walls easily penetrated by bacteria. Ensure adequate Calcium (Ca) and Boron (B) to enhance cellular integrity.",
        "pest_control": "Control thrips (Frankliniella occidentalis) and flea beetles using bio-insecticides like Spinosad or Beauveria bassiana, as their feeding wounds serve as direct bacterial entry gates.",
        "farmer_tips": "Prune lower branches (up to 15 cm from ground level) 3 weeks after transplanting to prevent soil-splash bacterial inoculation.",
        "ideal_climate": "Temperature: 24°C - 32°C | Relative Humidity: > 80% | High splashing rain risk",
        "economic_threshold": "First appearance of 1-2 angular water-soaked lesions per plant canopy during vegetative or early flowering stage.",
    },
    "Pepper Bell Healthy": {
        "crop": "Pepper (Bell)",
        "scientific_crop": "Capsicum annuum",
        "category": "Healthy Crop",
        "pathogen": "None (Optimum Physiological State)",
        "severity": "None",
        "status": "Healthy",
        "badge": "status-healthy",
        "overview": "The pepper foliage exhibits optimal cellular turgor, uniform chlorophyll distribution, intact leaf cuticles, and active vegetative and floral development without pathological signs.",
        "etiology": "Physiologically balanced specimen. Cellular structure shows intact palisade mesophyll, active chloroplasts, robust stomatal regulation, and uncompromised vascular bundles.",
        "symptoms": "Vibrant deep emerald green color, smooth and crisp leaf margins, intact foliar cuticle layer, strong apical growth shoots, clean flower buds, and absence of necrotic spotting or wilting.",
        "causes": "Optimal soil aeration, well-regulated drip irrigation, soil pH between 6.0 and 6.8, balanced fertility, and disciplined crop hygiene.",
        "chemical_treatment": "No corrective chemical intervention is required. Continue preventative monitoring.",
        "organic_treatment": "Apply bi-weekly foliar applications of liquid seaweed (Ascophyllum nodosum extract @ 2 ml/L) or humic/fulvic acid drenches to support root cation exchange capacity.",
        "prevention": "1. Maintain scheduled weekly scouting.\n2. Maintain a 5-7 cm organic straw mulch over beds.\n3. Monitor irrigation EC (1.5-2.2 mS/cm) and soil moisture tensiometer levels.",
        "fertilizer": "Apply balanced water-soluble fertilizer (19:19:19 @ 3 g/L) during vegetative stage; transition to Potassium Nitrate (13:0:45) and Calcium Nitrate during flowering and fruit setting.",
        "pest_control": "Deploy 15-20 yellow and blue sticky traps per acre to monitor thrips and aphids before populations establish.",
        "farmer_tips": "Ensure night temperatures remain above 16°C and below 24°C to avoid flower abortion and blossom drop.",
        "ideal_climate": "Temperature: 20°C - 28°C | Relative Humidity: 55% - 70% | Well-drained loamy soil",
        "economic_threshold": "N/A (Healthy Baseline Maintenance).",
    },
    "Potato Early Blight": {
        "crop": "Potato",
        "scientific_crop": "Solanum tuberosum",
        "category": "Fungal Infection",
        "pathogen": "Alternaria solani",
        "severity": "Medium",
        "status": "Disease Detected",
        "badge": "status-warning",
        "overview": "Early Blight of Potato is an economically significant fungal disease caused by the necrotrophic pathogen Alternaria solani. It targets mature and senescing leaves, reducing functional photosynthetic area and causing secondary dry tuber rot.",
        "etiology": "Fungus producing large, dark-colored, multi-celled beaked conidia. Overwinters as dormant mycelium or chlamydospores in solanaceous crop debris and infected tubers. Disseminated primarily by air currents, wind-blown dust, rain splash, and mechanical equipment.",
        "symptoms": "• Foliar: Dark brown to black circular-to-angular lesions characterized by concentric ridges resembling a target board or tree growth rings. Lesions are bordered by narrow chlorotic yellow margins. Begins on oldest lower leaves and progresses acropetally.\n• Stems: Sunken, elongated brown lesions with concentric markings.\n• Tubers: Irregular, sunken, dark leathery lesions with underlying flesh turning brown, dry, and corky.",
        "causes": "Alternating cycles of high humidity/heavy dew and dry windy weather; plant senescence; nutrient stress (low Nitrogen/Potassium); mechanical leaf injury.",
        "chemical_treatment": "• Protectant: Chlorothalonil (75% WP @ 2.0 g/L) or Mancozeb (75% WP @ 2.5 g/L) applied prior to row closure.\n• Systemic / Translaminar: Azoxystrobin (23% SC @ 1.0 ml/L), Pyraclostrobin (20% WG @ 1.0 g/L), or Difenoconazole (25% EC @ 0.5 ml/L).\n• Premix Formulations: Fluopyram + Tebuconazole @ 1.0 ml/L.\n• PHI: 7 to 14 days depending on compound.",
        "organic_treatment": "• Potassium Bicarbonate (Armicarb @ 3.0 g/L) to alter leaf surface pH and inhibit spore germination.\n• Copper Octanoate (Liquid Copper Fungicide @ 2.5 ml/L).\n• Trichoderma harzianum soil drench (2 x 10^8 CFU/g @ 5 g/L).",
        "prevention": "1. Implement a 3-year rotation away from potato, tomato, and pepper.\n2. Hill up soil generously (at least 10-15 cm cover over tubers) to prevent washdown of fungal spores.\n3. Maintain wide row spacing (75-90 cm) to promote fast canopy drying.",
        "fertilizer": "Maintain continuous Potassium (K) availability throughout tuber bulking. Plants under nutritional stress or premature senescence are significantly more vulnerable to Alternaria invasion.",
        "pest_control": "Control Colorado potato beetles and potato flea beetles to prevent chewing damage that provides spore infection courts.",
        "farmer_tips": "Remove and burn lower yellowing leaves showing target spots before the canopy closes between adjacent rows.",
        "ideal_climate": "Temperature: 22°C - 30°C | Alternating wet and dry foliar cycles | High wind spore dispersal",
        "economic_threshold": "5% lower canopy foliar infection observed before tuber bulking phase.",
    },
    "Potato Healthy": {
        "overview": "The potato vine shows robust vegetative growth, dark green compound leaves, firm vascular stems, and active subterranean stolon and tuber development.",
        "crop": "Potato",
        "scientific_crop": "Solanum tuberosum",
        "category": "Healthy Crop",
        "pathogen": "None (Optimum Physiological State)",
        "severity": "None",
        "status": "Healthy",
        "badge": "status-healthy",
        "etiology": "Uninfected physiological state. Vascular xylem and phloem vessels are clear, stomatal conductance is optimal, and tuber initiation enzymes are functioning normally.",
        "symptoms": "Lush, dark emerald compound foliage, thick erect stems, uniform leaf margins, absence of chlorotic flecking or necrosis, clean root stolons, and firm developing tuber skins.",
        "causes": "Certified virus-free seed tubers, loose and well-aerated sandy loam soil, optimal soil moisture (65-75% field capacity), and balanced base fertilization.",
        "chemical_treatment": "No chemical treatment is indicated. Continue standard preventative crop management.",
        "organic_treatment": "Apply biological root inoculants (Glomus intraradices mycorrhizal fungi) at planting and foliar fermented plant juice every 15 days.",
        "prevention": "1. Hill up soil around potato hills every 2-3 weeks.\n2. Maintain consistent, deep irrigation intervals.\n3. Disinfect tractor tires and spray equipment between field blocks.",
        "fertilizer": "Apply N-P-K in a 1:2:2 ratio during planting; supplement with Potassium Sulfate (0:0:50 @ 5 g/L foliar) during tuber bulking to boost dry matter content and starch synthesis.",
        "pest_control": "Inspect the undersides of leaves weekly for green peach aphid (Myzus persicae) colonies to prevent viral transmission.",
        "farmer_tips": "Halt all nitrogen applications 30 days prior to harvest to encourage natural canopy maturation and tuber skin set.",
        "ideal_climate": "Temperature: 15°C - 22°C | Soil Moisture: 65% - 75% Field Capacity | Full Sun Exposure",
        "economic_threshold": "N/A (Healthy Baseline Maintenance).",
    },
    "Potato Late Blight": {
        "overview": "Late Blight of Potato, caused by the oomycete Phytophthora infestans, is an aggressive, destructive plant disease that can destroy entire fields within 7 to 10 days and cause rot in stored tubers.",
        "crop": "Potato",
        "scientific_crop": "Solanum tuberosum",
        "category": "Water Mold / Blight",
        "pathogen": "Phytophthora infestans",
        "severity": "Critical",
        "status": "Disease Detected",
        "badge": "status-danger",
        "etiology": "Oomycete (water mold) organism producing biflagellate motile zoospores inside airborne sporangia. Sporangia germinate directly at 18-24°C or produce 8-12 motile zoospores at 10-15°C. Overwinters primarily in infected tubers left in the soil or cull piles.",
        "symptoms": "• Foliar: Starts as irregular, water-soaked, pale-to-dark green lesions that turn purplish-brown/black. Under high humidity, a distinctive white, cottony downy mold appears on leaf undersides along the lesion border.\n• Stems: Dark brown to black greasy, girdling lesions causing upper canopy collapse.\n• Tubers: Irregular, sunken, brownish-red or purplish firm dry rot extending 5-15 mm into tuber flesh, often followed by foul bacterial soft rot.",
        "causes": "Cool, persistent wet weather; extended leaf wetness (> 8-10 hours); relative humidity > 90%; temperatures between 10°C and 21°C.",
        "chemical_treatment": "• Preventative: Mancozeb (75% WP @ 2.5 g/L), Propineb (70% WP @ 2.5 g/L), or Chlorothalonil (75% WP @ 2.0 g/L).\n• Curative / Anti-Oomycete Systemics: Cymoxanil + Mancozeb (Curzate @ 2.5 g/L), Dimethomorph (50% WP @ 1.0 g/L), Mandipropamid (Revus @ 0.8 ml/L), or Metalaxyl-M + Mancozeb (Ridomil Gold @ 2.5 g/L).\n• Anti-Sporulant: Fluopicolide + Propamocarb (Infinito @ 1.5 ml/L).\n• PHI: 7 to 14 days.",
        "organic_treatment": "• Preventative Fixed Copper: Bordeaux Mixture (1% copper sulfate + hydrated lime) or Copper Hydroxide @ 2.5 g/L prior to forecast rain events.\n• Biological controls have limited efficacy once late blight sporulation is active in the field.",
        "prevention": "1. Plant exclusively certified pathogen-tested seed tubers.\n2. Completely destroy all volunteer potato plants and cull piles within a 5 km radius.\n3. Utilize local weather-based Blight Decision Support Systems (DSS) to time sprays before rain fronts.",
        "fertilizer": "Avoid excessive nitrogen applications which generate dense, slow-drying foliar canopies. Ensure adequate Potassium and Silica to harden cell walls.",
        "pest_control": "Eradicate alternate solanaceous weed hosts (e.g., Solanum dulcamara, Solanum nigrum) from irrigation canals and field borders.",
        "farmer_tips": "If late blight field infection exceeds 5%, desiccate (kill) the entire canopy with approved chemical or mechanical flailing 2-3 weeks prior to harvest to prevent tuber contact with active spores.",
        "ideal_climate": "Temperature: 10°C - 21°C | Relative Humidity: > 90% | Overcast, misty, or rainy weather",
        "economic_threshold": "Zero-tolerance. Action required upon the very first confirmed leaf lesion in the regional monitoring area.",
    },
    "Tomato Bacterial Spot": {
        "overview": "Bacterial Spot of Tomato is caused by Xanthomonas perforans, X. euvesicatoria, X. gardneri, and X. vesicatoria. It causes extensive foliar necrosis, severe flower blossom drop, and raised scab-like lesions on green and ripening fruit.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Bacterial Infection",
        "pathogen": "Xanthomonas perforans / euvesicatoria",
        "severity": "High",
        "status": "Disease Detected",
        "badge": "status-danger",
        "etiology": "Gram-negative, rod-shaped bacterium transmitted internally and externally on seeds, in volunteer crop residue, and via aerosolized water droplets. Bacteria enter through natural openings (stomata, hydathodes) or mechanical micro-wounds.",
        "symptoms": "• Foliar: Small (under 3 mm), angular, dark brown to black water-soaked spots with yellow halos. Spots coalesce, causing the leaf tissue to turn brown, dry, and tear, giving the foliage a ragged, scorched appearance.\n• Fruit: Small black spots that expand into raised, rough, dark brown scabs (up to 5 mm) with sunken centers, making fruit unmarketable.\n• Stems & Pedicels: Elongated dark cankers leading to flower blossom abortion.",
        "causes": "High rainfall, overhead irrigation splash, high relative humidity (> 85%), warm temperatures (25°C to 32°C), and field traffic through wet canopies.",
        "chemical_treatment": "• Copper Hydroxide (53.8% DF @ 2.0 g/L) tank-mixed with Mancozeb (75% WP @ 2.0 g/L) to counter copper-resistant bacterial strains.\n• Plant Defense Inducer: Acibenzolar-S-methyl (Actigard @ 25-35 g/ha) applied preventatively.\n• Bacteriophage: Specific registered agricultural bacteriophage bio-treatments applied in early evening.\n• PHI: 3 to 7 days.",
        "organic_treatment": "• Streptomyces lydicus (Actinovate @ 1.5 g/L) or Bacillus subtilis (Serenade ASO @ 5 ml/L).\n• Copper Octanoate (Soap-shield @ 2.5 ml/L).\n• Hot-water seed treatment at 50°C for 25 minutes.",
        "prevention": "1. Use certified pathogen-tested seed.\n2. Install drip irrigation beneath plastic mulch.\n3. Sanitize tomato stakes, wire trellises, and harvest crates with 10% sodium hypochlorite solution between seasons.\n4. Practice a 2-year rotation away from solanaceous crops.",
        "fertilizer": "Maintain balanced N:P:K nutrition with soil pH 6.2-6.8. Avoid excess nitrogen fertilization which promotes lush, thin-walled vegetative growth.",
        "pest_control": "Control piercing-sucking insects (stink bugs, leaf-footed bugs) that puncture fruit skin and spread bacteria internally.",
        "farmer_tips": "Stake and prune indeterminate tomatoes to keep the entire leaf canopy at least 30 cm above soil level.",
        "ideal_climate": "Temperature: 25°C - 32°C | High relative humidity (> 85%) | Driving rainstorms",
        "economic_threshold": "First visual appearance of water-soaked angular leaf spots during seedling or early vegetative growth.",
    },
    "Tomato Early Blight": {
        "overview": "Early Blight of Tomato is caused by Alternaria linariae (formerly A. solani). It is a major foliar and stem disease that causes collar rot in young seedlings, progressive lower-leaf defoliation, and sunken leathery fruit rot near the stem calyx.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Fungal Infection",
        "pathogen": "Alternaria linariae (A. tomatophila)",
        "severity": "Medium",
        "status": "Disease Detected",
        "badge": "status-warning",
        "etiology": "Fungus producing large, dark, transverse- and longitudinal-septate conidia. Survives in infected solanaceous crop debris for over a year and spreads via wind, splashing rain, and mechanical contact.",
        "symptoms": "• Foliar: Dark brown to black circular lesions showing distinct concentric rings (target pattern) surrounded by chlorotic yellow halos. Progresses upward from the oldest bottom foliage.\n• Stems: Dark, sunken collar rot lesions near the soil line on seedlings; elongated target-pattern cankers on mature stems.\n• Fruit: Dark, leathery, sunken spots with concentric rings at the stem end of both green and ripe fruit.",
        "causes": "Warm, humid conditions (24°C - 29°C), frequent rainfall or heavy dew, overhead irrigation, plant senescence, and nitrogen/potassium nutrient deficiency.",
        "chemical_treatment": "• Protectants: Mancozeb (75% WP @ 2.5 g/L) or Chlorothalonil (75% WP @ 2.0 g/L).\n• Systemic / Curative Fungicides: Azoxystrobin + Difenoconazole (Amistar Top @ 1.0 ml/L), Boscalid + Pyraclostrobin (Pristine @ 1.0 g/L), or Penthiopyrad (Fontelis @ 1.2 ml/L).\n• PHI: 3 to 7 days.",
        "organic_treatment": "• Bacillus subtilis (Serenade MAX @ 3.0 g/L) applied weekly.\n• Copper Octanoate @ 2.5 ml/L.\n• Apply a 5-8 cm clean straw or compost mulch beneath plants to prevent soil-spore splash.",
        "prevention": "1. Prune off all bottom foliage up to 30 cm above the ground once plants reach 1 meter in height.\n2. Space plants at least 60 cm apart in rows 1.2 m apart.\n3. Implement a 3-year crop rotation.",
        "fertilizer": "Maintain consistent potassium and calcium fertilization. Ensure balanced nitrogen to avoid plant stress during heavy fruit load.",
        "pest_control": "Control flea beetles and tomato hornworms which cause wounds that accelerate fungal spore entry.",
        "farmer_tips": "Strip diseased lower leaves the moment target spots appear, seal them in plastic bags, and remove them from the field.",
        "ideal_climate": "Temperature: 24°C - 30°C | High humidity and frequent rainfall | Warm nights",
        "economic_threshold": "Presence of active target lesions on > 5% of lower canopy leaves before first fruit cluster harvest.",
    },
    "Tomato Healthy": {
        "overview": "The tomato plant demonstrates prime physiological health, robust dark green foliage, vigorous apical growth shoots, clean floral trusses, and absence of visual pathogen symptoms.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Healthy Crop",
        "pathogen": "None (Optimum Physiological State)",
        "severity": "None",
        "status": "Healthy",
        "badge": "status-healthy",
        "etiology": "Uncompromised biological state. High photosynthetic rate, normal stomatal conductance, optimal nutrient absorption, and uninhibited vascular transport.",
        "symptoms": "Uniform deep green leaves, firm crisp stems, robust flowering clusters, active terminal shoots, clean calyxes, and absence of chlorosis, necrosis, or foliar distortion.",
        "causes": "Fertile well-drained loamy soil (pH 6.2-6.8), 6-8 hours daily direct solar exposure, balanced drip fertigation, and disciplined preventative hygiene.",
        "chemical_treatment": "No chemical treatment is indicated. Continue standard preventative maintenance.",
        "organic_treatment": "Apply liquid kelp extract (Ascophyllum nodosum @ 2.0 ml/L) and humic acid foliar drenches every 14 days to sustain natural systemic vigor.",
        "prevention": "1. Continue routine sucker pruning on indeterminate varieties for canopy aeration.\n2. Keep organic mulch layers clean and intact.\n3. Maintain weekly scouting routines.",
        "fertilizer": "Apply low-nitrogen, high-phosphorus and potassium fertilizers (e.g., 5-10-10 or 13-0-45 Potassium Nitrate @ 3 g/L) during active fruit setting to support fruit development.",
        "pest_control": "Inspect leaf undersides weekly for whiteflies, spider mites, and tomato pinworms; deploy yellow sticky cards.",
        "farmer_tips": "Water consistently at the root zone to prevent soil moisture swings that trigger Blossom End Rot (BER) and fruit splitting.",
        "ideal_climate": "Temperature: 21°C - 28°C | Solar Exposure: 6-8 hrs/day | Soil pH 6.2 - 6.8",
        "economic_threshold": "N/A (Healthy Baseline Maintenance).",
    },
    "Tomato Late Blight": {
        "overview": "Late Blight of Tomato, caused by Phytophthora infestans, is an aggressive oomycete pathogen capable of rapidly destroying entire tomato crops, rotting foliage, stems, and fruit within days.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Water Mold / Blight",
        "pathogen": "Phytophthora infestans",
        "severity": "Critical",
        "status": "Disease Detected",
        "badge": "status-danger",
        "etiology": "Oomycete producing airborne sporangia and motile biflagellate zoospores. Spreads through air currents and cool storm fronts, surviving in volunteer potatoes and solanaceous weeds.",
        "symptoms": "• Foliar: Large, irregular, water-soaked oily greenish-brown patches that rapidly turn dark brown to black. In humid air, a delicate white fuzzy downy mold develops on the lower leaf surface along lesion margins.\n• Stems: Dark brown to black greasy girdling lesions that cause branch collapse.\n• Fruit: Large, firm, greasy olive-brown to bronze sunken lesions with a rough surface, affecting both green and ripe tomatoes.",
        "causes": "Cool, humid, overcast, and rainy weather; temperatures between 12°C and 22°C; relative humidity > 90%; prolonged canopy moisture (> 8 hours).",
        "chemical_treatment": "• Preventative Protectants: Mancozeb (75% WP @ 2.5 g/L), Chlorothalonil (75% WP @ 2.0 g/L), or Copper Oxychloride (50% WP @ 3.0 g/L).\n• Curative Systemics: Mandipropamid (Revus 250 SC @ 0.8 ml/L), Dimethomorph (50% WP @ 1.0 g/L), Cymoxanil + Mancozeb (Curzate @ 2.5 g/L), or Famoxadone + Cymoxanil (Tanos @ 1.0 g/L).\n• Anti-Sporulant: Fluopicolide + Propamocarb (Infinito @ 1.5 ml/L).\n• PHI: 3 to 7 days.",
        "organic_treatment": "• Preventative Fixed Copper (Bordeaux mixture or Copper Hydroxide @ 2.5 g/L) applied prior to forecast rain.\n• Heavily infected plants must be rogued out and destroyed immediately.",
        "prevention": "1. Plant late-blight resistant tomato cultivars (e.g., Mountain Magic, Defiant, Iron Lady).\n2. Maximize row spacing (90-120 cm).\n3. Keep high-tunnel and greenhouse ventilation fans running overnight to prevent dew condensation.",
        "fertilizer": "Avoid high-nitrogen feeding which produces dense, humid vegetative canopies.",
        "pest_control": "Eliminate wild nightshade species and cull potato piles around the field perimeter.",
        "farmer_tips": "Immediately bag and remove infected plants from the plot on a dry afternoon. Do not leave pulled infected vines on the ground.",
        "ideal_climate": "Temperature: 12°C - 22°C | Relative Humidity: > 90% | Overcast, misty, or rainy weather",
        "economic_threshold": "Zero-tolerance. Implement immediate intervention upon first confirmed regional diagnosis.",
    },
    "Tomato Leaf Mold": {
        "overview": "Tomato Leaf Mold is caused by the biotrophic fungus Passalora fulva (formerly Cladosporium fulvum). It is a major disease in greenhouses, polyhouses, and high tunnels with restricted airflow and high relative humidity.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Fungal Infection",
        "pathogen": "Passalora fulva (Cladosporium fulvum)",
        "severity": "Medium",
        "status": "Disease Detected",
        "badge": "status-warning",
        "etiology": "Fungus producing branched conidiophores with pale olive conidia. Conidia survive for months in greenhouse framework and crop debris. Spreads via air currents, splashing water, and clothing/tools.",
        "symptoms": "• Foliar: Starts as pale green to light yellow diffuse spots with indistinct borders on the upper leaf surface. The corresponding lower leaf surface develops a dense, olive-green to velvety brown mold growth. Leaves eventually turn yellow, curl, wither, and drop.\n• Blossoms & Fruit: Blossoms can abort; fruit rarely develops a smooth black stem-end rot.",
        "causes": "High relative humidity (> 85%), warm temperatures (20°C to 25°C), dense foliar canopies, and poor air exchange in protected growing environments.",
        "chemical_treatment": "• Protective Sprays: Chlorothalonil (75% WP @ 2.0 g/L) or Copper Hydroxide (53.8% DF @ 2.0 g/L).\n• Systemic / Translaminar: Azoxystrobin (23% SC @ 1.0 ml/L), Difenoconazole (25% EC @ 0.5 ml/L), or Cyprodinil + Fludioxonil (Switch @ 0.8 g/L).\n• PHI: 3 to 7 days.",
        "organic_treatment": "• Potassium Bicarbonate (MilStop @ 3.0 g/L) applied at 5-day intervals.\n• Bio-fungicide: Bacillus amyloliquefaciens (Double Nickel @ 2.5 g/L).\n• Horticultural mineral oil @ 5 ml/L.",
        "prevention": "1. Install horizontal airflow fans in high tunnels and greenhouses.\n2. Vent structures at dusk to lower nighttime relative humidity.\n3. Prune interior foliage and indeterminate suckers to allow light penetration.",
        "fertilizer": "Ensure adequate potassium and calcium to strengthen foliar cell walls against fungal hyphae penetration.",
        "pest_control": "Scout greenhouse structures for aphids and whiteflies which leave honeydew that promotes secondary sooty molds.",
        "farmer_tips": "Never water plants overhead in enclosed structures. Install drip lines under plastic mulch for complete moisture control.",
        "ideal_climate": "Temperature: 21°C - 26°C | Relative Humidity: > 85% | Stagnant protected air",
        "economic_threshold": "Visual observation of olive-green velvety sporulation on > 3% of middle canopy leaves in greenhouse crops.",
    },
    "Tomato Septoria Leaf Spot": {
        "overview": "Septoria Leaf Spot of Tomato is caused by Septoria lycopersici. It is a common and destructive foliar fungal disease that causes severe progressive lower defoliation, exposing developing fruit to sunscald and reducing total yield.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Fungal Infection",
        "pathogen": "Septoria lycopersici",
        "severity": "Medium",
        "status": "Disease Detected",
        "badge": "status-warning",
        "etiology": "Fungus producing filiform (needle-like) hyaline conidia inside dark pycnidia fruiting bodies. Overwinters on diseased solanaceous plant residue and solanaceous weeds (horsenettle). Conidia are extruded in gelatinous tendrils and spread by rain splash, wind, insects, and workers.",
        "symptoms": "• Foliar: Numerous small (2-3 mm), circular spots with dark brown margins and light gray or tan centers. Tiny black specks (pycnidia fruiting bodies) are visible within the center of mature spots.\n• Stems & Calyxes: Small, elongated, dark spots with pycnidia.\n• Fruit: Direct fruit infection is rare, but severe defoliation causes fruit sunscald.",
        "causes": "Moderate temperatures (20°C to 27°C), prolonged leaf wetness (> 8 hours), frequent rainfall, overhead irrigation, and dense ground-level foliage.",
        "chemical_treatment": "• Protectants: Chlorothalonil (75% WP @ 2.0 g/L), Mancozeb (75% WP @ 2.5 g/L), or Copper Hydroxide (53.8% DF @ 2.0 g/L).\n• Systemics: Azoxystrobin + Difenoconazole (Amistar Top @ 1.0 ml/L) or Pyraclostrobin (Cabrio @ 1.0 g/L).\n• PHI: 3 to 7 days.",
        "organic_treatment": "• Liquid Copper Octanoate @ 2.5 ml/L.\n• Bacillus subtilis (Serenade ASO @ 5 ml/L).\n• Apply clean straw or black plastic mulch to eliminate soil-splash spore transmission.",
        "prevention": "1. Maintain a strict 3-year rotation away from nightshade crops.\n2. Prune the bottom 30 cm of foliage to break splash transmission.\n3. Disinfect tomato cages and stakes between growing seasons.",
        "fertilizer": "Apply balanced organic fertilizers; avoid excessive nitrogen that increases foliar density.",
        "pest_control": "Eradicate horsenettle, groundcherry, and jimsonweed in field borders.",
        "farmer_tips": "Do not overhead water. Always inspect oldest lower leaves first when scouting for Septoria outbreaks.",
        "ideal_climate": "Temperature: 20°C - 27°C | High rainfall / Heavy dew | High humidity",
        "economic_threshold": "Presence of circular spots with black pycnidia on > 5% of lower canopy foliage during early fruiting.",
    },
    "Tomato Spider Mites": {
        "overview": "Two-Spotted Spider Mite (Tetranychus urticae) infestation is a serious pest problem in tomatoes. These microscopic arachnids puncture foliar epidermal cells to suck out plant sap, causing chlorosis, stippling, bronzing, and foliar desiccation.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Pest Infestation",
        "pathogen": "Tetranychus urticae (Two-Spotted Spider Mite)",
        "severity": "Medium",
        "status": "Disease Detected",
        "badge": "status-warning",
        "etiology": "Arachnid pest with a rapid lifecycle (egg to adult in 5-7 days at 30°C). Females lay up to 200 eggs on leaf undersides. Hot, dry, and dusty microclimates trigger exponential population growth.",
        "symptoms": "• Foliar: Fine yellow or white stippling speckles across the upper leaf surface. Under severe infestation, leaves turn bronze, dry out like paper, and fine silky webs envelope leaf axils and growing tips.",
        "causes": "High temperatures (> 28°C), low relative humidity (< 50%), dusty roadways, drought-stressed plants, and overuse of broad-spectrum insecticides that kill natural predators.",
        "chemical_treatment": "• Selective Acaricides / Miticides: Abamectin (1.9% EC @ 0.5 ml/L), Bifenazate (Floramite @ 0.8 ml/L), Spiromesifen (Oberon 240 SC @ 0.8 ml/L), or Fenpyroximate (5% EC @ 1.0 ml/L).\n• Alternate miticide modes of action (IRAC groups) to prevent resistance development.\n• PHI: 3 to 7 days.",
        "organic_treatment": "• Insecticidal Soap (Potassium salts of fatty acids @ 10 ml/L).\n• Cold-pressed Neem Oil (10,000 ppm @ 4 ml/L) with emulsifier.\n• Horticultural mineral oil (1-2% concentration) applied with high pressure.",
        "prevention": "1. Keep crops adequately hydrated to eliminate drought stress.\n2. Dampen dusty farm roadways to reduce dust that shields spider mites from predators.\n3. Release beneficial predatory mites (Phytoseiulus persimilis or Neoseiulus californicus).",
        "fertilizer": "Avoid excess nitrogen fertilization; high foliar nitrogen increases mite reproduction rates.",
        "pest_control": "Avoid broad-spectrum synthetic pyrethroid insecticides which eliminate natural beneficial mite predators (lady beetles, predatory thrips).",
        "farmer_tips": "A strong jet of clean water aimed at the undersides of leaves can physically dislodge mite colonies in small plots.",
        "ideal_climate": "Temperature: > 28°C | Relative Humidity: < 50% | Dusty, drought-stressed microclimate",
        "economic_threshold": "Presence of active mite stippling or live nymphs on > 10% of sampled mid-canopy leaves.",
    },
    "Tomato Target Spot": {
        "overview": "Target Spot of Tomato is caused by the fungal pathogen Corynespora cassiicola. It causes foliar blighting, stem lesions, and sunken circular pitted craters on mature fruit in warm, humid production regions.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Fungal Infection",
        "pathogen": "Corynespora cassiicola",
        "severity": "Medium",
        "status": "Disease Detected",
        "badge": "status-warning",
        "etiology": "Fungus producing large, multi-septate cylindrical conidia. Survives in crop residue and alternative weed hosts. Dispersed by wind, rain splash, and mechanical equipment.",
        "symptoms": "• Foliar: Pinpoint brown spots expanding into circular lesions (up to 1 cm) with light brown centers, dark margins, and subtle concentric rings. Lesions often develop yellow halos.\n• Stems: Dark brown, elongated lesions.\n• Fruit: Small brown specks that enlarge into sunken circular pits with dark centers on green or ripe fruit.",
        "causes": "Warm temperatures (25°C to 32°C), high relative humidity (> 80%), prolonged canopy wetness, and poor air movement in dense canopies.",
        "chemical_treatment": "• Protectants: Chlorothalonil (75% WP @ 2.0 g/L) or Mancozeb (75% WP @ 2.5 g/L).\n• Systemic Fungicides: Azoxystrobin (23% SC @ 1.0 ml/L), Boscalid + Pyraclostrobin (Pristine @ 1.0 g/L), or Penthiopyrad (Fontelis @ 1.2 ml/L).\n• PHI: 3 to 7 days.",
        "organic_treatment": "• Bacillus amyloliquefaciens (Double Nickel @ 2.5 g/L).\n• Liquid copper octanoate @ 2.5 ml/L.\n• Apply a 5-8 cm organic mulch layer.",
        "prevention": "1. Increase plant and row spacing to ensure rapid leaf drying.\n2. Prune indeterminate vines and stake securely.\n3. Implement a 3-year crop rotation.",
        "fertilizer": "Provide consistent calcium and potassium to maintain structural fruit and leaf cuticle integrity.",
        "pest_control": "Control alternate weed hosts and scout foliage weekly during warm, rainy weather.",
        "farmer_tips": "Harvest mature green or blushing fruit promptly to minimize exposure to fruit-lesion infections.",
        "ideal_climate": "Temperature: 25°C - 32°C | High relative humidity (> 80%) | Moderate to high rainfall",
        "economic_threshold": "Observation of active target lesions on > 5% of mid-canopy leaves during flowering.",
    },
    "Tomato Yellow Leaf Curl Virus": {
        "overview": "Tomato Yellow Leaf Curl Virus (TYLCV) is a destructive Begomovirus (family Geminiviridae). It causes severe plant stunting, upward leaf cupping, chlorosis, and near-total flower abortion, causing up to 100% crop loss if infection occurs before flowering.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Viral Disease",
        "pathogen": "Tomato Yellow Leaf Curl Geminivirus (TYLCV)",
        "severity": "Critical",
        "status": "Disease Detected",
        "badge": "status-danger",
        "etiology": "Circular single-stranded DNA virus transmitted persistently by the sweetpotato/silverleaf whitefly (Bemisia tabaci). The virus is not seed-borne or mechanically transmitted through touch.",
        "symptoms": "• Foliar: Severe upward curling and cupping of leaflets, pronounced interveinal chlorosis (yellowing), reduced leaf size, bushy stunted plant habit, and complete failure to set new fruit.\n• Whole Plant: Erect, compact, stunted habit resembling a floral rosette.",
        "causes": "High populations of viruliferous whiteflies (Bemisia tabaci); warm arid weather; proximity to older infected solanaceous fields.",
        "chemical_treatment": "• No chemical viricide exists. Management targets the whitefly insect vector:\n• Systemic Neonicotinoids: Imidacloprid (17.8% SL @ 0.5 ml/L) or Thiamethoxam (25% WG @ 0.3 g/L) applied as seedling drench.\n• Selective Feeding Blockers: Pymetrozine (50% WDG @ 0.6 g/L), Flonicamid (50% WG @ 0.4 g/L), or Cyantraniliprole (Benevia @ 1.5 ml/L).\n• PHI: 3 to 7 days.",
        "organic_treatment": "• Entomopathogenic Fungi: Beauveria bassiana (1 x 10^8 CFU/ml @ 3 ml/L) sprayed during late afternoon.\n• Cold-pressed Neem Oil (10,000 ppm @ 4-5 ml/L).\n• Yellow sticky traps (25-30 traps per acre).",
        "prevention": "1. Plant exclusively TYLCV-resistant tomato hybrids (e.g., Tygress, Charger, Invicta).\n2. Install 50-mesh insect screening on high tunnels.\n3. Deploy silver reflective mulches to repel incoming whiteflies.",
        "fertilizer": "Maintain healthy organic soil conditions with balanced nutrition to support uninfected neighboring plants.",
        "pest_control": "Deploy yellow sticky cards (1 card per 20 sq meters) to monitor and mass-trap whitefly populations.",
        "farmer_tips": "Immediately pull, bag, and destroy symptomatic plants to prevent them from serving as viral reservoirs for whiteflies.",
        "ideal_climate": "Temperature: > 26°C | Abundant whitefly vector populations | Arid to sub-humid seasons",
        "economic_threshold": "Zero-tolerance for whitefly vectors in early seedling and vegetative stages.",
    },
    "Tomato Mosaic Virus": {
        "overview": "Tomato Mosaic Virus (ToMV) is a highly contagious Tobamovirus. It causes leaf mottling, distortion, and internal browning of tomato fruit. It is mechanically stable and can persist in dried plant residue and seed coats for years.",
        "crop": "Tomato",
        "scientific_crop": "Solanum lycopersicum",
        "category": "Viral Disease",
        "pathogen": "Tomato Mosaic Tobamovirus (ToMV)",
        "severity": "High",
        "status": "Disease Detected",
        "badge": "status-danger",
        "etiology": "Rigid rod-shaped positive-sense ssRNA virus. Transmitted mechanically via hands, pruning tools, trellising twine, contaminated seed coats, and tobacco products. Not transmitted by insect vectors like aphids or whiteflies.",
        "symptoms": "• Foliar: Mottled light and dark green mosaic patterns on leaves, blistering, leaf distortion, 'fern-leaf' or 'shoestring' malformation, and stunted plant vigor.\n• Fruit: Uneven ripening, yellow blotches, and internal brown necrosis (vascular browning) in fruit walls.",
        "causes": "Mechanical handling with contaminated hands/tools, planting uncertified seed lots, and workers handling tobacco products prior to touching plants.",
        "chemical_treatment": "• No chemical viricide exists. Infected plants cannot be cured and must be carefully removed and destroyed.",
        "organic_treatment": "• Milk Spray Decontamination: Spray 20% non-fat dry milk solution on hands and pruning tools during trellising to neutralize virus particles before they transfer to healthy plants.",
        "prevention": "1. Plant certified mosaic-resistant seed varieties (labeled ToMV / TMV resistant).\n2. Prohibit smoking and handling tobacco products anywhere near the greenhouse or garden.\n3. Wash hands with warm soapy water before working with plants.",
        "fertilizer": "Provide balanced organic feed to support neighboring healthy crops.",
        "pest_control": "Disinfect tools, stakes, and cages with 10% household bleach or 20% non-fat dry milk solution.",
        "farmer_tips": "Always work in younger, healthy blocks first before entering older or suspect blocks.",
        "ideal_climate": "Any temperature regime | Mechanical transmission through touch and tools",
        "economic_threshold": "Zero-tolerance. Rogue out symptomatic plants immediately.",
    },
}

# Standard 35 Vegetable Crops Catalog
DEFAULT_35_CROPS = {
    "Solanaceae": [
        {"id": "sol_tomato", "name": "Tomato", "scientific_name": "Solanum lycopersicum", "status": "AI Detection Available", "ai_supported": True, "description": "High-value commercial crop. Full AI diagnosis available for 10 conditions including Early Blight, Late Blight, Bacterial Spot, and Mites."},
        {"id": "sol_potato", "name": "Potato", "scientific_name": "Solanum tuberosum", "status": "AI Detection Available", "ai_supported": True, "description": "Staple tuber crop. Full AI diagnosis available for Early Blight, Late Blight, and Healthy Foliage."},
        {"id": "sol_capsicum", "name": "Capsicum / Bell Pepper", "scientific_name": "Capsicum annuum", "status": "AI Detection Available", "ai_supported": True, "description": "Sweet pepper crop. Full AI diagnosis available for Bacterial Spot and Healthy Foliage."},
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
# DATA ACCESS & PERSISTENCE LAYER
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
    return DEEP_DISEASE_KNOWLEDGE

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

def get_model_classes():
    db = get_diseases_database()
    if db and len(db.keys()) >= 15 and all(cls in db for cls in DEFAULT_MODEL_CLASSES):
        return list(db.keys())
    return DEFAULT_MODEL_CLASSES

def get_disease_detail(condition_name: str):
    db = get_diseases_database()
    if db and condition_name in db:
        return db[condition_name]
    
    # Fallback to rich built-in deep knowledge
    if condition_name in DEEP_DISEASE_KNOWLEDGE:
        return DEEP_DISEASE_KNOWLEDGE[condition_name]

    return {
        "crop": "Solanaceous Crop",
        "scientific_crop": "",
        "category": "General Condition",
        "severity": "Moderate",
        "status": "Analyzed",
        "badge": "status-warning",
        "overview": "Detailed information for this prediction is being added to the PlantCare AI knowledge base.",
        "symptoms": "Visible foliar lesions, chlorosis, or morphological changes as classified by the visual model.",
        "causes": "Environmental moisture, pathogen inoculums, or physiological stress factors.",
        "treatment": "Use locally approved products according to label directions and consult an agriculture professional when needed.",
        "prevention": "Maintain clean seed stock, adequate row aeration, and balanced crop nutrition.",
        "fertilizer": "Maintain balanced N-P-K nutrition according to soil requirements.",
        "pest_control": "Monitor insect vectors regularly.",
        "farmer_tips": "Observe foliage during early morning scouting."
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
# CSS DESIGN SYSTEM
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #059669;
        --primary-dark: #064e3b;
        --primary-deep: #022c22;
        --primary-light: #ecfdf5;
        --primary-border: #a7f3d0;
        --text-main: #0d1f17;
        --text-muted: #52665a;
        --card-bg: rgba(255, 255, 255, 0.95);
        --card-border: #e2ece6;
        --shadow-sm: 0 2px 6px rgba(6, 78, 59, 0.04);
        --shadow-md: 0 10px 25px rgba(6, 78, 59, 0.07);
        --shadow-lg: 0 18px 45px rgba(6, 78, 59, 0.11);
    }

    .stApp {
        background-color: #f8faf9;
        background-image: 
            radial-gradient(at 10% 12%, rgba(16, 185, 129, 0.09) 0px, transparent 52%),
            radial-gradient(at 90% 16%, rgba(5, 150, 105, 0.08) 0px, transparent 48%),
            radial-gradient(at 50% 85%, rgba(52, 211, 153, 0.07) 0px, transparent 55%),
            radial-gradient(at 88% 88%, rgba(6, 78, 59, 0.06) 0px, transparent 50%);
        background-attachment: fixed;
        color: var(--text-main);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #042f24 0%, #064e3b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    [data-testid="stSidebar"] * {
        color: #f4fff9 !important;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }

    .hero-banner {
        padding: 3.2rem 3rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #043628 0%, #064e3b 45%, #065f46 100%);
        color: white;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 1.8rem;
    }
    .hero-banner:after {
        content: "";
        position: absolute;
        width: 380px;
        height: 380px;
        right: -100px;
        top: -120px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.22) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.95rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #d1fae5;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .hero-title {
        font-size: clamp(2.2rem, 4.2vw, 3.6rem);
        line-height: 1.15;
        margin: 0.4rem 0 1rem;
        font-weight: 800;
        color: #ffffff;
    }
    .hero-desc {
        max-width: 680px;
        font-size: 1.1rem;
        line-height: 1.7;
        color: #ecfdf5;
        margin-bottom: 1.25rem;
    }
    .hero-pill {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-size: 0.85rem;
        font-weight: 700;
        color: #a7f3d0;
    }

    .product-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 1.6rem 1.75rem;
        margin: 0.85rem 0;
        box-shadow: var(--shadow-sm);
        backdrop-filter: blur(10px);
        transition: all 0.2s ease-in-out;
    }
    .product-card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--primary-border);
    }
    .product-card h3 {
        margin: 0 0 0.5rem;
        color: var(--primary-dark);
        font-size: 1.2rem;
        font-weight: 800;
    }
    .card-muted {
        color: var(--text-muted);
        line-height: 1.65;
        font-size: 0.95rem;
    }

    .metric-container {
        background: #ffffff;
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 1.2rem 1.25rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 850;
        color: var(--primary);
    }
    .metric-label {
        color: var(--text-muted);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    .result-panel {
        border-radius: 24px;
        padding: 2rem;
        background: #ffffff;
        border: 1px solid var(--card-border);
        box-shadow: var(--shadow-md);
        margin: 1rem 0;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 0.02em;
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
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: #f1f5f9;
        color: #475569;
        margin-left: 0.5rem;
    }

    .prob-grid-row {
        display: grid;
        grid-template-columns: 240px 1fr 85px;
        gap: 14px;
        align-items: center;
        margin: 0.75rem 0;
    }
    .prob-label {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--text-main);
    }
    .prob-track {
        height: 10px;
        background: #e8f0ec;
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
        font-weight: 800;
        color: var(--primary-dark);
        font-size: 0.95rem;
    }

    .info-layout-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.25rem;
        margin-top: 1rem;
    }
    .info-box {
        background: #ffffff;
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.35rem;
    }
    .info-box-title {
        font-size: 0.98rem;
        font-weight: 800;
        color: var(--primary-dark);
        margin-bottom: 0.45rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .info-box-text {
        font-size: 0.9rem;
        color: var(--text-main);
        line-height: 1.6;
        margin: 0;
    }

    .ad-card {
        background: #ffffff;
        border: 1px solid #c7eed8;
        border-radius: 22px;
        padding: 1.6rem;
        margin: 1.2rem 0;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    .ad-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .ad-badge {
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: #ecfdf5;
        color: #065f46;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        border: 1px solid #a7f3d0;
    }
    .ad-title {
        font-size: 1.28rem;
        font-weight: 850;
        color: #064e3b;
        margin: 0.35rem 0;
    }

    .disclaimer-card {
        font-size: 0.84rem;
        color: #64748b;
        background: #f8faf9;
        border-left: 3px solid #cbd5e1;
        padding: 0.85rem 1.15rem;
        border-radius: 0 10px 10px 0;
        margin-top: 1.5rem;
        line-height: 1.6;
    }

    .app-footer-bar {
        margin-top: 4rem;
        padding: 2rem 0.5rem 1.5rem 0.5rem;
        border-top: 1px solid var(--card-border);
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: var(--text-muted);
        font-size: 0.9rem;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .footer-brand {
        font-weight: 800;
        color: var(--primary-dark);
        font-size: 1rem;
    }

    @media (max-width: 850px) {
        .prob-grid-row {
            grid-template-columns: 1fr;
            gap: 6px;
        }
        .prob-pct {
            text-align: left;
        }
        .info-layout-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 2.1rem;
        }
        .hero-banner {
            padding: 2.25rem 1.75rem;
        }
        .app-footer-bar {
            flex-direction: column;
            text-align: center;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# MODEL LOADING & INFERENCE ENGINE
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
        warnings.append("The image appears dark. Ensure adequate lighting for reliable confidence.")
    elif brightness > 225:
        warnings.append("The image appears overexposed. Ensure leaf texture is clearly visible.")
    var = stat.var
    avg_var = sum(var[:3]) / 3.0
    if avg_var < 100:
        warnings.append("The image appears soft in focus. A sharper photo is recommended.")
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

    raise RuntimeError("Unable to complete screening for this image. Please upload a clear leaf photo and retry.")

# ============================================================
# WEATHER ADVISORY ENGINE
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
            "source": "Open-Meteo Meteorological API"
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
        reasons.append(f"High rain chance ({rain_prob:.0f}% > {spray_limits['max_rain_probability_pct']}%) risks pesticide wash-off before plant uptake.")
    if temp > spray_limits["max_temperature_c"]:
        suitable = False
        reasons.append(f"Elevated temperature ({temp:.1f}°C) may cause foliar scorch and fast droplet evaporation.")
    elif temp < spray_limits["min_temperature_c"]:
        suitable = False
        reasons.append(f"Low temperature ({temp:.1f}°C) slows systemic chemical absorption.")
    if humidity > spray_limits["max_relative_humidity_pct"]:
        reasons.append(f"High humidity ({humidity:.0f}%) extends drying time and may trigger fungal spore germination.")

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
# GEOSPATIAL MAP QUERY
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
    headers = {"User-Agent": "PlantCareAI/7.0 (Commercial Agritech AI)"}
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
# REPORT GENERATOR
# ============================================================
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
2. VISUAL SYMPTOMS & PATHOLOGICAL CAUSES
----------------------------------------------------------------------
Symptoms:
{info.get("symptoms", "N/A")}

Possible Causes:
{info.get("causes", "N/A")}

----------------------------------------------------------------------
3. TREATMENT & MANAGEMENT REGIMES
----------------------------------------------------------------------
{info.get("chemical_treatment", info.get("treatment", "N/A"))}

----------------------------------------------------------------------
4. PREVENTATIVE AGRONOMIC SCHEDULE
----------------------------------------------------------------------
{info.get("prevention", "N/A")}

----------------------------------------------------------------------
5. SMART FERTILIZER & PEST CONTROL GUIDANCE
----------------------------------------------------------------------
Fertilizer Guidance:
{info.get("fertilizer", "N/A")}

Pest Management:
{info.get("pest_control", "N/A")}

Farmer / Grower Tips:
{info.get("farmer_tips", "N/A")}

----------------------------------------------------------------------
6. AI PROBABILITY DISTRIBUTION (TOP 5)
----------------------------------------------------------------------
{top5_formatted}

======================================================================
Disclaimer: AI-assisted visual screening is intended as an initial 
assessment. Use locally approved products according to label directions 
and consult an agriculture professional when needed.
© 2026 PlantCare AI. All rights reserved.
======================================================================
"""

# ============================================================
# SESSION STATE & MODEL INITIALIZATION
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
            <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">{top_ad.get('company', 'SEA AUTO Ecosystem')}</span>
        </div>
        <div class="ad-title">{html.escape(top_ad.get('title', ''))}</div>
        <div style="font-size: 0.95rem; color: #334155; line-height: 1.6; margin-bottom: 0.75rem;">
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
            <div style="font-size: 2.2rem;">🌿</div>
            <div style="font-size: 1.55rem; font-weight: 850; letter-spacing: -0.02em;">PlantCare AI</div>
            <div style="opacity: 0.8; font-size: 0.82rem; margin-top: 0.3rem;">
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
        <div style="padding: 1.1rem; border-radius: 16px; background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.12);">
            <div style="font-size: 0.8rem; color: #a7f3d0; font-weight: 700;">✦ Powered by SEA AUTO</div>
        </div>
        """, unsafe_allow_html=True)

        return selected_page

# ============================================================
# PAGE 1: HOME
# ============================================================
def render_home_page():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-kicker">✦ SMART PLANT HEALTH</div>
        <div class="hero-title">PlantCare AI</div>
        <div style="font-size: 1.35rem; font-weight: 700; color: #a7f3d0; margin-bottom: 0.5rem;">
            AI-Powered Plant Health Screening
        </div>
        <div class="hero-desc">
            Upload a plant leaf image to receive an AI-assisted health assessment and practical plant-care guidance.
        </div>
        <span class="hero-pill">🌿 Powered by SEA AUTO</span>
    </div>
    """, unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1.4, 2.2])
    with btn_col1:
        if st.button("Analyze Leaf", type="primary", use_container_width=True):
            st.session_state["navigation_page_selector"] = "🔬 Disease Detection"
            st.rerun()
    with btn_col2:
        if st.button("Explore Crops", use_container_width=True):
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
            <h3>🔬 AI Leaf Screening</h3>
            <div class="card-muted">
                Fast, objective visual assessment from leaf imagery with verified neural network confidence distributions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="product-card">
            <h3>📊 Plant Health Analysis</h3>
            <div class="card-muted">
                Clear descriptions of symptoms, etiology, pathogen biology, and preventative cultural schedules.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="product-card">
            <h3>🌱 Practical Care Guidance</h3>
            <div class="card-muted">
                Responsible fertilizer schedules, spray windows, and verified local plant-care discovery.
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
    st.caption("Upload a clear photo of a plant leaf to receive an AI-assisted health assessment and practical care guidance.")

    model_classes = get_model_classes()

    uploaded_file = st.file_uploader(
        "Upload plant leaf image",
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
            st.image(image, caption="Uploaded Leaf Preview", use_container_width=True)

        with col_action:
            st.markdown("""
            <div class="product-card">
                <h3>Ready for Screening</h3>
                <div class="card-muted">
                    Click below to run model inference against verified pathological datasets.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔬 Analyze Leaf", type="primary", use_container_width=True):
                with st.spinner("Analyzing your leaf..."):
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
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: gap; gap: 0.5rem;">
                <div style="font-size: 0.82rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b;">
                    Screening Result
                </div>
                <div>
                    <span class="status-badge {info.get('badge', 'status-warning')}">● {info.get('status', 'Analyzed')}</span>
                    {conf_tag}
                </div>
            </div>
            <div style="font-size: 1.85rem; font-weight: 850; color: #0d1f17; margin-bottom: 0.5rem;">
                {p['condition']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if p["confidence"] < 60.0:
            st.info("💡 **Low-confidence result:** Try uploading a clearer leaf image with good lighting and minimal background.")

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
        <div class="product-card" style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📋</div>
            <h3>No Active Screening Record</h3>
            <div class="card-muted">
                Please analyze a plant leaf in the 'Disease Detection' section first to view and download your health report.
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
        st.image(p["image"], caption="Screened Leaf Specimen", use_container_width=True)
    with col_rep_meta:
        st.markdown(f"""
        <div class="hero-banner" style="padding: 1.8rem 2rem; margin-bottom: 1rem;">
            <div class="hero-kicker">HEALTH SCREENING DOSSIER</div>
            <div class="hero-title" style="font-size: 1.8rem;">{html.escape(p['condition'])}</div>
            <div>Confidence: <strong>{p['confidence']:.2f}%</strong> | Status: <strong>{info.get('status', 'Analyzed')}</strong></div>
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
    st.caption("Complete directory of 35 vegetable crops. AI model screening is actively calibrated for Solanaceae staples.")

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
                    <div style="background:#ffffff; border:1px solid #e2ece6; border-radius:14px; padding:1.1rem; margin-bottom:0.85rem; box-shadow:0 2px 5px rgba(0,0,0,0.02);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                            <strong style="color:#064e3b; font-size:1rem;">{crop['name']}</strong>
                            <span class="status-badge {badge_style}" style="font-size:0.68rem; padding:0.2rem 0.5rem;">{crop['status']}</span>
                        </div>
                        <div style="font-size:0.75rem; color:#64748b; font-style:italic; margin-bottom:0.4rem;">{crop.get('scientific_name', '')}</div>
                        <p style="font-size:0.85rem; color:#52665a; margin:0; line-height:1.5;">{crop['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================
# PAGE 5: ADVANCED & DEEP DISEASE KNOWLEDGE HUB
# ============================================================
def render_knowledge_hub():
    st.markdown("## 📚 Disease Knowledge Hub")
    st.caption("Advanced Clinical Pathology Dossiers & Integrated Disease Management (IDM) Compendium.")

    model_classes = get_model_classes()

    col_crop_sel, col_disease_sel = st.columns([1, 1.5])
    with col_crop_sel:
        crop_filter = st.selectbox(
            "Filter by Crop",
            ["All Crops", "Pepper (Bell)", "Potato", "Tomato"],
            key="crop_knowledge_filter"
        )
    
    filtered_classes = model_classes
    if crop_filter != "All Crops":
        filtered_classes = [c for c in model_classes if get_disease_detail(c).get("crop") == crop_filter]

    with col_disease_sel:
        selected_condition = st.selectbox(
            "Select Pathological Condition to Inspect",
            filtered_classes,
            key="condition_knowledge_selector"
        )

    info = get_disease_detail(selected_condition)

    st.markdown(f"""
    <div class="result-panel">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <span style="font-size: 0.85rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b;">
                    Botanical Specimen: <em>{info.get('scientific_crop', info.get('crop', 'Crop'))}</em>
                </span>
                <div style="font-size: 0.85rem; color: #047857; font-weight: 700; margin-top: 0.2rem;">
                    Pathogen Taxon: {info.get('pathogen', 'N/A')}
                </div>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="status-badge {info.get('badge', 'status-warning')}">● {info.get('category', 'Category')}</span>
                <span class="status-badge status-warning" style="background:#f1f5f9; color:#334155; border:1px solid #cbd5e1;">Severity: {info.get('severity', 'Moderate')}</span>
            </div>
        </div>
        <div style="font-size: 2rem; font-weight: 850; color: #0d1f17; margin-bottom: 0.6rem;">
            {selected_condition}
        </div>
        <div style="font-size: 0.98rem; color: #334155; line-height: 1.7;">
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
                    <span style="color:#064e3b; font-weight:700;">{info.get('ideal_climate', 'Warm, humid weather with prolonged foliar moisture')}</span>
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
                <div style="font-size:0.8rem; color:#b91c1c; margin-top:0.8rem;">
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
                        <div style="font-size: 0.85rem; color: #047857; font-weight: 700;">
                            📍 {html.escape(st_item.get('location', ''))}, {html.escape(st_item.get('state', ''))}
                        </div>
                    </div>
                    <span class="status-badge status-healthy" style="font-size: 0.75rem;">
                        {html.escape(st_item.get('crop', 'Crop'))}
                    </span>
                </div>
                <div style="font-size: 0.92rem; color: #334155; line-height: 1.6; margin-bottom: 0.75rem;">
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
            <div style="font-size: 1.2rem; font-weight: 800; color: {'#064e3b' if 'Favorable' in spray_status else '#991b1b'}; margin-bottom: 0.5rem;">
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
                            <h3 style="font-size: 1.1rem; margin-bottom: 0.3rem;">🏪 {html.escape(s['name'])}</h3>
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
            <div class="product-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🗺️</div>
                <h3>Locator Ready</h3>
                <div class="card-muted">
                    Enable location access on the left to locate certified suppliers.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE 9: CONTENT MANAGER
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
        <h1 style="color: #064e3b; font-size: 2.2rem; margin: 0.5rem 0 0.75rem;">About PlantCare AI</h1>
        <p class="card-muted" style="font-size: 1.05rem;">
            PlantCare AI is an AI-powered plant health screening application developed under SEA AUTO to help users 
            understand visible plant-health conditions from leaf images and receive practical plant-care guidance.
        </p>
        <p class="card-muted">
            The platform provides AI-assisted visual screening, comprehensive disease profiles, preventative agronomic schedules, 
            and live weather-informed spray advisories.
        </p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.25rem; margin-top: 1rem;">
        <div class="product-card">
            <h3>Product Purpose</h3>
            <div class="card-muted">
                To make early visual plant disease assessment accessible and straightforward for growers, gardeners, and agronomists.
            </div>
        </div>
        <div class="product-card">
            <h3>AI-Assisted Screening</h3>
            <div class="card-muted">
                High-precision deep learning image screening delivering transparent probability metrics and immediate guidance.
            </div>
        </div>
        <div class="product-card">
            <h3>SEA AUTO Ecosystem</h3>
            <div class="card-muted">
                PlantCare AI is part of SEA AUTO's technology ecosystem focused on building scalable, practical real-world solutions.
            </div>
        </div>
    </div>

    <div class="product-card" style="margin-top: 1.25rem; text-align: center;">
        <div style="font-size: 1.2rem; font-weight: 800; color: #064e3b; margin-bottom: 0.3rem;">
            Powered by SEA AUTO
        </div>
        <div style="font-size: 0.85rem; color: #64748b;">
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
        <div>© 2026 PlantCare AI. All rights reserved.</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()