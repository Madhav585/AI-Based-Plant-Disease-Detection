import json
from pathlib import Path

# Base Paths Setup
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
FARMER_IMAGES_DIR = IMAGES_DIR / "farmers"
AD_IMAGES_DIR = IMAGES_DIR / "advertisements"
CROP_IMAGES_DIR = IMAGES_DIR / "crops"

for directory in [DATA_DIR, IMAGES_DIR, FARMER_IMAGES_DIR, AD_IMAGES_DIR, CROP_IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 35 CROPS DATABASE (plant_database.json)
# ============================================================
PLANT_DATABASE = {
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
# 2. DEEP PATHOLOGY DATABASE (diseases.json)
# ============================================================
DISEASE_DATABASE = {
  "Pepper Bell Bacterial Spot": {
    "crop": "Pepper (Bell)",
    "scientific_crop": "Capsicum annuum",
    "category": "Bacterial Infection",
    "pathogen": "Xanthomonas campestris pv. vesicatoria",
    "severity": "High",
    "status": "AI Detection Available",
    "badge": "status-danger",
    "overview": "Bacterial spot causes water-soaked angular leaf spots, premature defoliation, and rough scabby fruit lesions.",
    "symptoms": "Small circular-to-angular dark spots with yellow halos; leaf drop; sunken scabs on fruit.",
    "causes": "Splashing rain, overhead irrigation, contaminated seed, and high humidity (> 80%).",
    "treatment": "Copper Hydroxide (2.0 g/L) + Mancozeb (2.0 g/L) tank mix applied early morning.",
    "prevention": "Use certified seeds, install drip irrigation, and maintain 2-year non-host rotation.",
    "fertilizer": "Balanced NPK (1:1:2); avoid high nitrogen which forces soft, vulnerable growth.",
    "pest_control": "Control thrips and flea beetles to prevent entry wounds.",
    "farmer_tips": "Prune lower foliage up to 15 cm above ground during a dry afternoon."
  },
  "Pepper Bell Healthy": {
    "crop": "Pepper (Bell)",
    "scientific_crop": "Capsicum annuum",
    "category": "Healthy Crop",
    "pathogen": "None",
    "severity": "None",
    "status": "AI Detection Available",
    "badge": "status-healthy",
    "overview": "Foliage exhibits vigorous chlorophyll pigmentation, balanced cellular turgor, and zero pathogen marks.",
    "symptoms": "Uniform deep green color, intact margins, and robust apical shoot growth.",
    "causes": "Optimal soil moisture, balanced fertilization, and adequate sun exposure.",
    "treatment": "No corrective chemical intervention needed.",
    "prevention": "Weekly scouting, organic straw mulching, and routine bed hygiene.",
    "fertilizer": "Apply balanced organic compost and maintain adequate calcium.",
    "pest_control": "Deploy yellow sticky traps for aphids and whiteflies.",
    "farmer_tips": "Ensure soil stays moist but well-drained to avoid flower drop."
  },
  "Potato Early Blight": {
    "crop": "Potato",
    "scientific_crop": "Solanum tuberosum",
    "category": "Fungal Infection",
    "pathogen": "Alternaria solani",
    "severity": "Medium",
    "status": "AI Detection Available",
    "badge": "status-warning",
    "overview": "Causes concentric target-board lesions on older leaves, reducing photosynthesis and tuber yield.",
    "symptoms": "Dark brown circular spots with concentric rings and yellow chlorotic halos on lower leaves.",
    "causes": "Alternating wet and dry cycles, plant stress, and overwintering spores in soil debris.",
    "treatment": "Chlorothalonil (2.0 g/L) or Azoxystrobin + Difenoconazole (1.0 ml/L).",
    "prevention": "3-year crop rotation, hill up soil over tubers, and avoid late-afternoon irrigation.",
    "fertilizer": "Ensure sufficient potassium and balanced micronutrients.",
    "pest_control": "Control potato beetles and flea beetles.",
    "farmer_tips": "Strip infected lower foliage as soon as spots appear."
  },
  "Potato Healthy": {
    "crop": "Potato",
    "scientific_crop": "Solanum tuberosum",
    "category": "Healthy Crop",
    "pathogen": "None",
    "severity": "None",
    "status": "AI Detection Available",
    "badge": "status-healthy",
    "overview": "Potato foliage demonstrates active vegetative growth, strong vascular stems, and clean stolons.",
    "symptoms": "Lush green foliage, uniform leaf margins, and absence of lesions.",
    "causes": "Certified seed tubers, loose sandy loam, and balanced moisture.",
    "treatment": "No chemical treatment indicated.",
    "prevention": "Hill up soil regularly and avoid prolonged foliar wetness.",
    "fertilizer": "Apply phosphorus and potassium-rich fertilizers during tuber bulking.",
    "pest_control": "Monitor leaf undersides for green peach aphids.",
    "farmer_tips": "Stop nitrogen feeding 3 weeks prior to harvest for skin set."
  },
  "Potato Late Blight": {
    "crop": "Potato",
    "scientific_crop": "Solanum tuberosum",
    "category": "Water Mold / Blight",
    "pathogen": "Phytophthora infestans",
    "severity": "Critical",
    "status": "AI Detection Available",
    "badge": "status-danger",
    "overview": "Destructive water-mold disease that rapidly rots leaves, stems, and stored tubers in cool, wet weather.",
    "symptoms": "Water-soaked dark lesions with white downy mold on leaf undersides; greasy brown stem cankers.",
    "causes": "Temperatures 10-21°C, relative humidity > 90%, and prolonged rain or heavy fog.",
    "treatment": "Dimethomorph (1.0 g/L), Cymoxanil + Mancozeb (2.5 g/L), or Mandipropamid (0.8 ml/L).",
    "prevention": "Plant certified seed tubers, eliminate cull piles, and monitor regional blight alerts.",
    "fertilizer": "Avoid excess nitrogen which promotes dense canopies.",
    "pest_control": "Eradicate wild nightshade weeds around field boundaries.",
    "farmer_tips": "Kill vines (desiccate) 2 weeks before digging if blight is present."
  },
  "Tomato Bacterial Spot": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Bacterial Infection",
    "pathogen": "Xanthomonas perforans",
    "severity": "High",
    "status": "AI Detection Available",
    "badge": "status-danger",
    "overview": "Causes foliar blighting, blossom drop, and rough scab-like spots on tomato fruit.",
    "symptoms": "Small angular dark greasy spots with yellow halos; leaves turn brown and ragged.",
    "causes": "Rain splash, warm temperatures (25-32°C), and infected seed lots.",
    "treatment": "Copper Hydroxide (2.0 g/L) + Mancozeb (2.0 g/L).",
    "prevention": "Use pathogen-free certified seed, drip irrigation, and tool disinfection.",
    "fertilizer": "Maintain balanced NPK with soil pH between 6.2 and 6.8.",
    "pest_control": "Control stink bugs and leaf-footed bugs.",
    "farmer_tips": "Prune and stake plants to keep foliage off wet soil."
  },
  "Tomato Early Blight": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Fungal Infection",
    "pathogen": "Alternaria linariae",
    "severity": "Medium",
    "status": "AI Detection Available",
    "badge": "status-warning",
    "overview": "Causes collar rot, target-like concentric leaf spots, and stem-end fruit rot.",
    "symptoms": "Concentric target rings surrounded by yellow halos starting on lowest leaves.",
    "causes": "Fungal spores splashing from soil debris during rain or overhead watering.",
    "treatment": "Mancozeb (2.5 g/L), Chlorothalonil (2.0 g/L), or Azoxystrobin (1.0 ml/L).",
    "prevention": "Mulch soil with organic straw, stake vines, and maintain 3-year rotation.",
    "fertilizer": "Maintain steady potassium and organic compost feeding.",
    "pest_control": "Control flea beetles and hornworms.",
    "farmer_tips": "Trim lower 30 cm of foliage once plants reach 1 meter height."
  },
  "Tomato Healthy": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Healthy Crop",
    "pathogen": "None",
    "severity": "None",
    "status": "AI Detection Available",
    "badge": "status-healthy",
    "overview": "Foliage exhibits rich green color, vigorous growth tips, and clean blossom clusters.",
    "symptoms": "Uniform deep green leaves, robust stems, and no necrotic lesions.",
    "causes": "Balanced watering, fertile loamy soil, and 6-8 hours of direct sunlight.",
    "treatment": "No corrective chemical treatment needed.",
    "prevention": "Prune non-fruiting suckers and maintain regular scouting.",
    "fertilizer": "Feed with low-nitrogen, high-phosphorus/potassium formula during fruiting.",
    "pest_control": "Scout weekly for hornworms, whiteflies, and spider mites.",
    "farmer_tips": "Water consistently at root level to prevent blossom end rot."
  },
  "Tomato Late Blight": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Water Mold / Blight",
    "pathogen": "Phytophthora infestans",
    "severity": "Critical",
    "status": "AI Detection Available",
    "badge": "status-danger",
    "overview": "Rapidly rots green and ripe fruit, turning vines and foliage brown and collapsed.",
    "symptoms": "Irregular water-soaked greasy patches with white fuzzy mold underneath in humidity.",
    "causes": "Cool, wet, overcast weather (12-22°C) with prolonged relative humidity (> 90%).",
    "treatment": "Mandipropamid (0.8 ml/L), Dimethomorph (1.0 g/L), or Cymoxanil + Mancozeb (2.5 g/L).",
    "prevention": "Plant resistant cultivars, maximize row spacing, and water at ground level.",
    "fertilizer": "Avoid high-nitrogen fertilizers that create dense foliar canopies.",
    "pest_control": "Clear nightshade weeds and volunteer potatoes.",
    "farmer_tips": "Remove and seal infected vines immediately in trash bags."
  },
  "Tomato Leaf Mold": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Fungal Infection",
    "pathogen": "Passalora fulva",
    "severity": "Medium",
    "status": "AI Detection Available",
    "badge": "status-warning",
    "overview": "Greenhouse and polyhouse disease favored by stagnant air and humidity above 85%.",
    "symptoms": "Yellow patches on upper leaf surface with olive-brown velvety mold underneath.",
    "causes": "Relative humidity > 85% and temperatures 20-25°C in enclosed structures.",
    "treatment": "Potassium Bicarbonate (3.0 g/L) or Copper Hydroxide (2.0 g/L).",
    "prevention": "Run greenhouse ventilation fans overnight and space plants generously.",
    "fertilizer": "Ensure adequate calcium and potash to toughen leaf cuticles.",
    "pest_control": "Scout for whiteflies and aphids.",
    "farmer_tips": "Prune dense interior foliage to allow sunlight penetration."
  },
  "Tomato Septoria Leaf Spot": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Fungal Infection",
    "pathogen": "Septoria lycopersici",
    "severity": "Medium",
    "status": "AI Detection Available",
    "badge": "status-warning",
    "overview": "Strips plants of lower leaves, exposing fruit to sunscald and reducing harvest.",
    "symptoms": "Small circular spots (2-3 mm) with dark margins and gray centers containing tiny black dots.",
    "causes": "Overwintering spores in soil debris splashing onto lower leaves during rain.",
    "treatment": "Chlorothalonil (2.0 g/L), Mancozeb (2.5 g/L), or Copper Octanoate (2.5 ml/L).",
    "prevention": "Mulch soil beds thoroughly, practice 3-year rotation, and use drip irrigation.",
    "fertilizer": "Apply balanced compost tea to reduce defoliation stress.",
    "pest_control": "Eradicate horsenettle and groundcherry weeds.",
    "farmer_tips": "Disinfect tomato cages and stakes between growing seasons."
  },
  "Tomato Spider Mites": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Pest Infestation",
    "pathogen": "Tetranychus urticae",
    "severity": "Medium",
    "status": "AI Detection Available",
    "badge": "status-warning",
    "overview": "Sap-sucking microscopic mites that cause stippling, chlorosis, and fine webbing.",
    "symptoms": "Yellow/white stippling dots across leaves, bronzed foliage, and silky webs.",
    "causes": "Hot, dry, dusty weather (> 28°C) and drought stress.",
    "treatment": "Abamectin (0.5 ml/L), Spiromesifen (0.8 ml/L), or Neem Oil (4.0 ml/L).",
    "prevention": "Keep plants hydrated, dampen dusty farm paths, and preserve predatory mites.",
    "fertilizer": "Avoid high-nitrogen synthetic feeds which boost mite reproduction.",
    "pest_control": "Release predatory mites (Phytoseiulus persimilis).",
    "farmer_tips": "Spray leaf undersides thoroughly with high-pressure nozzles."
  },
  "Tomato Target Spot": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Fungal Infection",
    "pathogen": "Corynespora cassiicola",
    "severity": "Medium",
    "status": "AI Detection Available",
    "badge": "status-warning",
    "overview": "Causes brown concentric leaf spots and sunken crater-like pits on mature tomato fruit.",
    "symptoms": "Circular brown spots with light centers and concentric rings; sunken fruit pits.",
    "causes": "Warm, humid climates (25-32°C) and poor canopy airflow.",
    "treatment": "Azoxystrobin (1.0 ml/L) or Chlorothalonil (2.0 g/L).",
    "prevention": "Increase row spacing, stake vines upright, and use drip lines.",
    "fertilizer": "Provide consistent calcium and potassium for fruit wall firmness.",
    "pest_control": "Control alternate weed hosts in field borders.",
    "farmer_tips": "Harvest mature green or blushing fruit promptly."
  },
  "Tomato Yellow Leaf Curl Virus": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Viral Disease",
    "pathogen": "Tomato Yellow Leaf Curl Geminivirus (TYLCV)",
    "severity": "Critical",
    "status": "AI Detection Available",
    "badge": "status-danger",
    "overview": "Whitefly-transmitted virus causing severe stunting, upward leaf curling, and flower drop.",
    "symptoms": "Upward cupping of leaves, interveinal yellowing, stunted growth, and zero fruit set.",
    "causes": "Vectored exclusively by silverleaf whiteflies (Bemisia tabaci).",
    "treatment": "No viricide exists. Control whitefly vectors using Imidacloprid (0.5 ml/L) or Pymetrozine (0.6 g/L).",
    "prevention": "Plant resistant hybrids (Tygress, Charger), use 50-mesh netting, and silver mulch.",
    "fertilizer": "Maintain healthy soil organic matter to support unaffected plants.",
    "pest_control": "Deploy yellow sticky traps (25-30 traps/acre).",
    "farmer_tips": "Immediately rogue out and bag infected plants."
  },
  "Tomato Mosaic Virus": {
    "crop": "Tomato",
    "scientific_crop": "Solanum lycopersicum",
    "category": "Viral Disease",
    "pathogen": "Tomato Mosaic Tobamovirus (ToMV)",
    "severity": "High",
    "status": "AI Detection Available",
    "badge": "status-danger",
    "overview": "Mechanically transmitted virus causing mottled foliage, shoestring leaves, and fruit browning.",
    "symptoms": "Light and dark green mosaic pattern on leaves, distorted foliage, and internal fruit browning.",
    "causes": "Physical handling, contaminated pruning shears, infected seed, and tobacco products.",
    "treatment": "No antiviral cure. Remove and burn symptomatic plants.",
    "prevention": "Plant certified mosaic-resistant seed, wash hands with soap, and prohibit tobacco near crops.",
    "fertilizer": "Provide balanced organic feed to strengthen healthy neighboring plants.",
    "pest_control": "Disinfect tools with 20% non-fat dry milk solution or 10% bleach.",
    "farmer_tips": "Work in young, healthy plant blocks before visiting older blocks."
  },
  "Brinjal Phomopsis Blight": {
    "crop": "Brinjal / Eggplant",
    "scientific_crop": "Solanum melongena",
    "category": "Fungal Infection",
    "pathogen": "Phomopsis vexans",
    "severity": "High",
    "status": "Knowledge Available / Training Planned",
    "badge": "status-warning",
    "overview": "Major disease of brinjal causing collar rot in seedlings, leaf spotting, and fruit rot.",
    "symptoms": "Circular brown leaf spots with pycnidia; sunken soft rotting spots on fruit.",
    "causes": "Overwinters in crop debris and infected seed; high humidity and warm rains.",
    "treatment": "Mancozeb (2.5 g/L) or Carbendazim (1.0 g/L) at 10-day intervals.",
    "prevention": "Hot-water seed treatment at 50°C for 30 mins; 3-year crop rotation.",
    "fertilizer": "Balanced NPK (100:50:50 kg/ha); avoid excess nitrogen.",
    "pest_control": "Control shoot and fruit borers (Leucinodes orbonalis).",
    "farmer_tips": "Collect and destroy all fallen diseased fruits immediately."
  },
  "Chilli Anthracnose / Dieback": {
    "crop": "Chilli",
    "scientific_crop": "Capsicum frutescens",
    "category": "Fungal Infection",
    "pathogen": "Colletotrichum capsici",
    "severity": "High",
    "status": "Knowledge Available / Training Planned",
    "badge": "status-warning",
    "overview": "Causes drying of twigs from top downwards (dieback) and sunken circular lesions on ripe chillies.",
    "symptoms": "Necrotic twigs turning straw-colored; dark concentric sunken spots with pinkish spore masses on fruits.",
    "causes": "Warm rainy weather (28-30°C), high humidity (> 80%), and infected seeds.",
    "treatment": "Azoxystrobin (1.0 ml/L) or Copper Oxychloride (3.0 g/L) at flowering stage.",
    "prevention": "Seed treatment with Thiram (2 g/kg); remove infected dead branches.",
    "fertilizer": "Apply potassium sulfate to improve fruit skin thickness.",
    "pest_control": "Manage chilli thrips (Scirtothrips dorsalis) with Spinosad.",
    "farmer_tips": "Harvest mature ripe chillies promptly to prevent fruit rot spread."
  },
  "Okra Yellow Vein Mosaic Virus": {
    "crop": "Okra / Lady Finger / Bhindi",
    "scientific_crop": "Abelmoschus esculentus",
    "category": "Viral Disease",
    "pathogen": "Bhendi Yellow Vein Mosaic Virus (BYVMV)",
    "severity": "Critical",
    "status": "Knowledge Available / Training Planned",
    "badge": "status-danger",
    "overview": "Most damaging viral disease of bhindi, turning leaves yellow and fruits small, hard, and unmarketable.",
    "symptoms": "Network of yellow veins on green background; leaves turn completely yellow; stunted growth.",
    "causes": "Transmitted by whitefly (Bemisia tabaci) in warm humid weather.",
    "treatment": "Control whitefly vector: Acetamiprid (0.4 g/L) or Thiamethoxam (0.3 g/L).",
    "prevention": "Grow resistant varieties (Arka Anamika, Parbhani Kranti); rogue out early infected plants.",
    "fertilizer": "Ensure balanced basal fertilization with farmyard manure.",
    "pest_control": "Install yellow sticky traps (20-25 traps/acre).",
    "farmer_tips": "Sow barrier crops like maize or sorghum around the okra plot."
  },
  "Cabbage Black Rot": {
    "crop": "Cabbage",
    "scientific_crop": "Brassica oleracea var. capitata",
    "category": "Bacterial Infection",
    "pathogen": "Xanthomonas campestris pv. campestris",
    "severity": "High",
    "status": "Knowledge Available / Training Planned",
    "badge": "status-danger",
    "overview": "Bacterial vascular disease causing characteristic V-shaped yellow lesions starting from leaf edges.",
    "symptoms": "V-shaped chlorotic lesions with black vein network at leaf margins; vascular blackening in stem.",
    "causes": "Infected seed, warm rains (25-30°C), and water splash through hydathodes.",
    "treatment": "Copper Hydroxide (2.0 g/L) + Streptocycline (0.1 g/L) preventative foliar spray.",
    "prevention": "Hot-water seed soak at 50°C for 30 minutes; 3-year non-crucifer rotation.",
    "fertilizer": "Avoid excessive nitrogen which promotes soft heading tissue.",
    "pest_control": "Control diamondback moth larvae.",
    "farmer_tips": "Do not cultivate or weed field while leaves are damp."
  },
  "Cucumber Downy Mildew": {
    "crop": "Cucumber",
    "scientific_crop": "Cucumis sativus",
    "category": "Oomycete / Water Mold",
    "pathogen": "Pseudoperonospora cubensis",
    "severity": "Critical",
    "status": "Knowledge Available / Training Planned",
    "badge": "status-danger",
    "overview": "Fast-spreading foliar disease causing angular yellow spots bounded by leaf veins.",
    "symptoms": "Angular yellow patches on upper leaf surface; purple-gray fuzzy sporulation on lower surface.",
    "causes": "Cool to warm humid conditions, high dew, and sprinkler irrigation.",
    "treatment": "Cymoxanil + Mancozeb (2.5 g/L) or Dimethomorph (1.0 g/L).",
    "prevention": "Trellis vines for aeration, use drip lines, and select tolerant hybrid seeds.",
    "fertilizer": "Maintain balanced potassium and calcium feeding.",
    "pest_control": "Scout for cucumber beetles and aphids.",
    "farmer_tips": "Spray undersides of leaves thoroughly before rain fronts."
  },
  "Onion Purple Blotch": {
    "crop": "Onion",
    "scientific_crop": "Allium cepa",
    "category": "Fungal Infection",
    "pathogen": "Alternaria porri",
    "severity": "High",
    "status": "Knowledge Available / Training Planned",
    "badge": "status-warning",
    "overview": "Causes purple sunken lesions on onion leaves and flower stalks, leading to premature leaf collapse.",
    "symptoms": "Small water-soaked spots developing purple-brown centers with yellow halos; leaf tips die back.",
    "causes": "Warm humid weather (25-30°C), heavy dew, and thrips injury.",
    "treatment": "Mancozeb (2.5 g/L) or Tebuconazole (1.0 ml/L) with sticker/spreader.",
    "prevention": "Good soil drainage, certified seed bulbs, and 3-year crop rotation.",
    "fertilizer": "Apply balanced sulfur and potassium for bulb skin durability.",
    "pest_control": "Control onion thrips (Thrips tabaci) with Fipronil (1.5 ml/L).",
    "farmer_tips": "Always add a surfactant/spreader to sprays because onion leaves have waxy cuticles."
  },
  "Spinach Downy Mildew": {
    "crop": "Spinach / Palak",
    "scientific_crop": "Spinacia oleracea",
    "category": "Oomycete / Water Mold",
    "pathogen": "Peronospora effusa",
    "severity": "High",
    "status": "Knowledge Available / Training Planned",
    "badge": "status-danger",
    "overview": "Causes yellow blotches on tender spinach leaves with dense purple-gray mold underneath.",
    "symptoms": "Pale yellow chlorotic patches on upper foliage; violet-gray fungal fuzz underneath; leaves curl and rot.",
    "causes": "Cool, wet, humid weather (10-18°C) and dense plant stands.",
    "treatment": "Copper Octanoate (2.5 ml/L) or Potassium Phosphite (2.0 ml/L).",
    "prevention": "Use resistant spinach varieties, thin plants for airflow, and irrigate in early morning.",
    "fertilizer": "Apply balanced organic compost; avoid high foliar nitrogen.",
    "pest_control": "Manage leaf miners with neem formulations.",
    "farmer_tips": "Harvest uninfected leaves early if disease pressure is rising."
  },
  "French Bean Rust": {
    "crop": "French Bean",
    "scientific_crop": "Phaseolus vulgaris",
    "category": "Fungal Infection",
    "pathogen": "Uromyces appendiculatus",
    "severity": "Medium",
    "status": "Knowledge Available / Training Planned",
    "badge": "status-warning",
    "overview": "Fungal rust producing reddish-brown powdery pustules on leaves, causing premature defoliation.",
    "symptoms": "Small raised reddish-brown powdery pustules surrounded by yellow halos on leaf undersides.",
    "causes": "Moderate temperatures (18-25°C), high humidity (> 95%), and extended leaf moisture.",
    "treatment": "Wettable Sulfur (3.0 g/L) or Hexaconazole (1.0 ml/L) at first pustule appearance.",
    "prevention": "Plant resistant cultivars, wide row spacing, and clean crop residue post-harvest.",
    "fertilizer": "Ensure adequate phosphorus and potassium; avoid excess nitrogen.",
    "pest_control": "Control bean aphids.",
    "farmer_tips": "Avoid touching wet bean foliage during harvesting."
  },
  "Cauliflower Downy Mildew": {
    "crop": "Cauliflower",
    "scientific_crop": "Brassica oleracea var. botrytis",
    "category": "Oomycete / Water Mold",
    "pathogen": "Hyaloperonospora parasitica",
    "severity": "High",
    "status": "Knowledge Available / Training Planned",
    "badge": "status-warning",
    "overview": "Attacks seedlings and mature curd foliage, causing yellow spots and gray downy mold.",
    "symptoms": "Yellow angular leaf spots on upper surface; white downy growth underneath; curd browning.",
    "causes": "Cool foggy weather (10-18°C) and heavy morning dew.",
    "treatment": "Metalaxyl + Mancozeb (2.5 g/L) or Dimethomorph (1.0 g/L).",
    "prevention": "Seed treatment with Thiram; nursery raised beds with good drainage.",
    "fertilizer": "Supplement with Boron (Borax @ 10 kg/ha) to prevent curd browning.",
    "pest_control": "Scout for aphids and cabbage loopers.",
    "farmer_tips": "Ensure nursery beds are not overcrowded."
  }
}

# ============================================================
# 3. WEATHER & ADVISORY THRESHOLDS (advisory_rules.json)
# ============================================================
ADVISORY_RULES = {
  "spray_rules": {
    "max_wind_speed_kmh": 15.0,
    "max_rain_probability_pct": 30.0,
    "min_temperature_c": 10.0,
    "max_temperature_c": 35.0,
    "max_relative_humidity_pct": 88.0
  },
  "disease_risk_thresholds": {
    "late_blight": {
      "min_temp": 10.0,
      "max_temp": 23.0,
      "min_humidity": 85.0,
      "risk_label": "Critical Risk (Late Blight / Oomycete Pressure)"
    },
    "early_blight_leaf_mold": {
      "min_temp": 22.0,
      "max_temp": 30.0,
      "min_humidity": 70.0,
      "risk_label": "High Risk (Fungal Blight & Leaf Mold Spore Germination)"
    },
    "bacterial_spot": {
      "min_temp": 24.0,
      "max_temp": 34.0,
      "min_humidity": 78.0,
      "risk_label": "Elevated Risk (Bacterial Foliar Invasion Pressure)"
    },
    "spider_mites": {
      "min_temp": 27.0,
      "max_temp": 45.0,
      "max_humidity": 50.0,
      "risk_label": "Elevated Risk (Hot, Dry Microclimate Encouraging Spider Mites)"
    }
  }
}

# ============================================================
# 4. FARMER STORIES (farmer_stories.json)
# ============================================================
FARMER_STORIES = [
  {
    "id": "story_001",
    "farmer_name": "Sanjay Singh",
    "location": "Kaithma",
    "state": "Bihar, India",
    "crop": "Tomato (Solanaceae)",
    "image": "",
    "short_description": "Controlled early-season bacterial spot in cherry tomatoes using early diagnostic screening and copper-bactericide tank mixes.",
    "story": "During continuous monsoon drizzle, early water-soaked lesions appeared on lower foliage. Using PlantCare AI, bacterial spot was identified before it could ascend to flowering clusters. Drip irrigation sanitation and preventative copper hydroxide sprays saved over 85% of harvest volume.",
    "contact_cta": "Community Grower Network",
    "status": "active",
    "priority": 1,
    "date": "2026-08-26"
  },
  {
    "id": "story_002",
    "farmer_name": "Suresh Kumar",
    "location": "Jalandhar",
    "state": "Punjab",
    "crop": "Potato (Kufri Jyoti)",
    "image": "",
    "short_description": "Prevented catastrophic tuber loss during late blight pressure with microclimate tracking and systemic anti-oomycete sprays.",
    "story": "Following three consecutive days of cool, overcast weather with 90% relative humidity, leaf tips developed dark greasy water-soaked lesions. Instant AI detection and spray advisory prompted targeted dimethomorph application before canopy collapse.",
    "contact_cta": "Contact via Farmer Advisory Portal",
    "status": "active",
    "priority": 2,
    "date": "2026-08-18"
  }
]

# ============================================================
# 5. SPONSORED PARTNERS (advertisements.json)
# ============================================================
ADVERTISEMENTS = [
  {
    "id": "ad_001",
    "title": "Smart Solar Automated Drip Irrigation",
    "company": "SEA AUTO",
    "image": "",
    "description": "Affordable IoT-enabled automatic drip irrigation system designed to reduce water usage by 60% and keep foliar canopies dry.",
    "category": "Agriculture Technology",
    "button_text": "Explore Product",
    "button_url": "https://example.com/sea-auto-irrigation",
    "status": "active",
    "priority": 1,
    "start_date": "2026-01-01",
    "end_date": "2026-12-31"
  },
  {
    "id": "ad_002",
    "title": "Certified Bio-Fungicide & Organic Plant Shield",
    "company": "AgriBio CropCare",
    "image": "",
    "description": "Broad-spectrum Bacillus subtilis & Trichoderma biological formulation for preventative root and foliar disease control.",
    "category": "Biological Crop Protection",
    "button_text": "View Organic Catalog",
    "button_url": "https://example.com/agribio-shield",
    "status": "active",
    "priority": 2,
    "start_date": "2026-01-01",
    "end_date": "2026-12-31"
  }
]

# Write all JSON databases safely
def build_all_files():
    files_map = {
        DATA_DIR / "plant_database.json": PLANT_DATABASE,
        DATA_DIR / "diseases.json": DISEASE_DATABASE,
        DATA_DIR / "advisory_rules.json": ADVISORY_RULES,
        DATA_DIR / "farmer_stories.json": FARMER_STORIES,
        DATA_DIR / "advertisements.json": ADVERTISEMENTS
    }
    for file_path, data in files_map.items():
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Generated & Updated: {file_path}")

    print("\n🚀 All databases, crop profiles, farmer stories, and folder layouts are 100% READY!")

if __name__ == "__main__":
    build_all_files()