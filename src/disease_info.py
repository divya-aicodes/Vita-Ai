"""Readable disease labels and conservative educational guidance."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiseaseInfo:
    crop: str
    condition: str
    status: str
    description: str
    symptoms: str
    prevention: str
    expert_warning: str
    category: str = "Other"
    model_supported: bool = True

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


EXPERT_WARNING = (
    "This is an educational screening result, not a confirmed diagnosis. "
    "Consult a qualified agricultural extension officer or plant pathologist "
    "before removing crops or applying chemical treatment."
)

_DETAILS = {
    "Apple Scab": (
        "A fungal disease that affects apple leaves and fruit.",
        "Olive-green to dark velvety spots; curled or prematurely falling leaves.",
        "Remove fallen infected material, improve airflow, avoid prolonged leaf wetness, and use locally approved resistant varieties.",
    ),
    "Black Rot": (
        "A fungal disease that can produce leaf spots, fruit rot, and branch cankers.",
        "Purple-bordered leaf spots, dark fruit lesions, or shriveled fruit.",
        "Prune dead wood with clean tools, remove infected debris, and maintain an open canopy.",
    ),
    "Cedar Apple Rust": (
        "A fungal disease requiring apple and juniper-family hosts to complete its cycle.",
        "Bright yellow-orange spots on upper leaf surfaces and tube-like growth below.",
        "Separate susceptible hosts where practical, remove nearby galls, and choose resistant cultivars.",
    ),
    "Powdery Mildew": (
        "A fungal disease that forms powder-like growth on leaf surfaces.",
        "White or gray powdery patches, distorted young leaves, and reduced vigor.",
        "Improve spacing and airflow, avoid excess nitrogen, remove heavily affected tissue, and keep foliage monitored.",
    ),
    "Cercospora Leaf Spot / Gray Leaf Spot": (
        "A fungal corn disease favored by warm, humid conditions and infected residue.",
        "Long rectangular gray-to-tan lesions that follow leaf veins.",
        "Rotate crops, manage infected residue, use resistant hybrids, and monitor humid fields.",
    ),
    "Common Rust": (
        "A fungal corn disease producing characteristic rust-colored pustules.",
        "Small cinnamon-brown raised pustules on both leaf surfaces.",
        "Use resistant hybrids, scout early, and follow local integrated disease-management advice.",
    ),
    "Northern Leaf Blight": (
        "A fungal corn disease that can reduce photosynthetic leaf area.",
        "Long cigar-shaped gray-green or tan lesions.",
        "Rotate crops, manage residue, improve field airflow, and select resistant hybrids.",
    ),
    "Esca (Black Measles)": (
        "A grapevine trunk-disease complex that can affect leaves, wood, and berries.",
        "Interveinal tiger-stripe discoloration, dark berry spots, or internal wood decay.",
        "Sanitize pruning tools, remove severely affected wood under expert guidance, and reduce vine stress.",
    ),
    "Leaf Blight (Isariopsis Leaf Spot)": (
        "A fungal grape leaf disease associated with warm and humid conditions.",
        "Angular brown lesions, yellow margins, and premature leaf drop.",
        "Remove infected debris, improve canopy airflow, and avoid unnecessary overhead irrigation.",
    ),
    "Huanglongbing (Citrus Greening)": (
        "A serious bacterial citrus disease spread mainly by psyllid insects.",
        "Blotchy asymmetric yellowing, small misshapen fruit, poor flavor, and branch decline.",
        "Contact local plant-health authorities promptly, use certified planting stock, and manage vectors under official guidance.",
    ),
    "Bacterial Spot": (
        "A bacterial disease that can affect leaves, stems, and fruit.",
        "Small water-soaked spots that become dark and may develop yellow halos.",
        "Use clean seed or transplants, avoid handling wet plants, sanitize tools, and reduce leaf wetness.",
    ),
    "Early Blight": (
        "A fungal disease commonly affecting older leaves first.",
        "Brown target-like spots with concentric rings and surrounding yellowing.",
        "Rotate crops, remove infected debris, mulch soil, improve airflow, and avoid wetting foliage.",
    ),
    "Late Blight": (
        "A fast-spreading water-mold disease favored by cool, wet weather.",
        "Water-soaked dark lesions, pale margins, and white growth beneath leaves in humidity.",
        "Isolate suspicious plants, avoid moving wet plant material, and contact a local expert quickly.",
    ),
    "Leaf Scorch": (
        "A fungal strawberry leaf disease that reduces healthy leaf area.",
        "Small purple spots that enlarge and merge, giving leaves a scorched appearance.",
        "Use clean plants, remove infected leaves, improve spacing, and irrigate near the soil surface.",
    ),
    "Leaf Mold": (
        "A fungal tomato disease favored by high humidity and limited airflow.",
        "Pale yellow upper-leaf patches with olive or gray mold underneath.",
        "Ventilate protected crops, space plants, water at soil level, and remove affected leaves carefully.",
    ),
    "Septoria Leaf Spot": (
        "A fungal tomato disease that usually begins on lower leaves.",
        "Many small circular spots with gray centers, dark edges, and tiny black dots.",
        "Remove lower infected foliage, mulch, rotate crops, and avoid overhead watering.",
    ),
    "Spider Mites / Two-Spotted Spider Mite": (
        "A mite infestation that damages leaves by feeding on plant cells.",
        "Fine yellow stippling, bronzing, webbing, and tiny mites beneath leaves.",
        "Inspect leaf undersides, reduce plant stress, conserve beneficial insects, and seek local advice for severe outbreaks.",
    ),
    "Target Spot": (
        "A fungal tomato disease that produces expanding ringed lesions.",
        "Brown lesions with concentric rings on leaves and sometimes fruit.",
        "Remove infected debris, improve airflow, rotate crops, and keep leaves dry when possible.",
    ),
    "Tomato Yellow Leaf Curl Virus": (
        "A viral tomato disease transmitted primarily by whiteflies.",
        "Upward leaf curling, yellow margins, stunting, and poor fruit set.",
        "Use clean transplants, manage whiteflies with integrated methods, remove confirmed infected plants, and control weeds.",
    ),
    "Tomato Mosaic Virus": (
        "A contagious viral disease spread through sap, tools, hands, and infected material.",
        "Mottled light and dark green leaves, distortion, and reduced growth.",
        "Use certified seed, sanitize hands and tools, avoid tobacco contamination, and remove confirmed infected plants.",
    ),
}

_CATEGORY_BY_CONDITION = {
    "Bacterial Spot": "Bacterial",
    "Huanglongbing (Citrus Greening)": "Bacterial",
    "Late Blight": "Oomycete",
    "Spider Mites / Two-Spotted Spider Mite": "Pest",
    "Tomato Yellow Leaf Curl Virus": "Viral",
    "Tomato Mosaic Virus": "Viral",
}


def _reference(
    crop: str,
    condition: str,
    category: str,
    description: str,
    symptoms: str,
    prevention: str,
) -> DiseaseInfo:
    return DiseaseInfo(
        crop=crop,
        condition=condition,
        status="Diseased",
        description=description,
        symptoms=symptoms,
        prevention=prevention,
        expert_warning=EXPERT_WARNING,
        category=category,
        model_supported=False,
    )


_REFERENCE_CONDITIONS = (
    _reference(
        "Banana",
        "Black Sigatoka",
        "Fungal",
        "A serious banana leaf-spot disease favored by warm, humid conditions.",
        "Narrow reddish-brown streaks expand into dark lesions with gray centers; heavily affected leaves may collapse.",
        "Use clean planting material, improve spacing and drainage, remove badly affected leaves responsibly, and follow local disease-management guidance.",
    ),
    _reference(
        "Banana",
        "Fusarium Wilt (Panama Disease)",
        "Fungal",
        "A persistent soilborne vascular wilt of banana caused by Fusarium strains.",
        "Older leaves yellow and wilt, pseudostems may split, and cut vascular tissue can show reddish-brown discoloration.",
        "Use certified clean plants, prevent movement of contaminated soil and tools, improve farm biosecurity, and report unusual severe wilt locally.",
    ),
    _reference(
        "Banana",
        "Banana Bunchy Top Virus",
        "Viral",
        "A systemic viral banana disease spread mainly by banana aphids and infected planting material.",
        "Short upright leaves form a bunched crown, with dark green streaks on veins and severe plant stunting.",
        "Use certified virus-free plants, control volunteer hosts, monitor aphids, and contact local plant-health authorities before removing suspected plants.",
    ),
    _reference(
        "Rice",
        "Rice Blast",
        "Fungal",
        "A destructive rice disease that can affect leaves, nodes, collars, and panicles.",
        "Diamond-shaped lesions with gray centers and brown margins; neck infection may cause blank or broken panicles.",
        "Use resistant varieties, balanced nitrogen, clean seed, suitable spacing, and locally recommended water and residue management.",
    ),
    _reference(
        "Rice",
        "Bacterial Leaf Blight",
        "Bacterial",
        "A bacterial rice disease that spreads through water, wounds, and infected material.",
        "Water-soaked streaks begin near leaf tips or margins, enlarge, turn yellow, and later become straw-colored.",
        "Use clean seed and resistant varieties, avoid injuring plants, manage irrigation movement, and sanitize field equipment.",
    ),
    _reference(
        "Rice",
        "Brown Spot",
        "Fungal",
        "A fungal rice disease often more severe when plants are nutritionally or environmentally stressed.",
        "Round to oval brown lesions with gray centers occur on leaves, glumes, and sometimes seedlings.",
        "Use healthy seed, support balanced soil fertility, manage crop residue, and reduce avoidable plant stress.",
    ),
    _reference(
        "Rice",
        "Sheath Blight",
        "Fungal",
        "A fungal disease that begins on leaf sheaths and can spread upward through dense rice canopies.",
        "Oval green-gray lesions with brown margins appear near the waterline and merge into larger blighted areas.",
        "Avoid excessive nitrogen and overly dense stands, manage residue and weeds, and use locally adapted integrated practices.",
    ),
    _reference(
        "Rice",
        "Rice Tungro",
        "Viral",
        "A viral disease complex transmitted by green leafhoppers.",
        "Plants become stunted with yellow-to-orange leaves, reduced tillering, and poorly developed panicles.",
        "Use resistant varieties and healthy seedlings, synchronize planting where advised, manage volunteer rice, and monitor leafhopper pressure.",
    ),
    _reference(
        "Wheat",
        "Leaf Rust",
        "Fungal",
        "A wind-dispersed rust disease that primarily affects wheat leaf blades.",
        "Small orange-brown powdery pustules are scattered across leaves and may darken late in the season.",
        "Choose resistant varieties, remove volunteer cereal hosts where appropriate, scout regularly, and follow regional rust alerts.",
    ),
    _reference(
        "Wheat",
        "Stripe Rust",
        "Fungal",
        "A cool-weather wheat rust that can spread rapidly over long distances.",
        "Yellow-orange pustules form narrow parallel stripes along leaf veins; seedlings may show more scattered pustules.",
        "Plant resistant varieties, monitor regional forecasts, control volunteer cereals, and seek timely extension advice when stripes appear.",
    ),
    _reference(
        "Wheat",
        "Fusarium Head Blight",
        "Fungal",
        "A flowering-stage wheat disease that can damage grain quality and may be associated with mycotoxins.",
        "One or more spikelets bleach early; salmon-pink growth and shriveled lightweight kernels may develop.",
        "Rotate away from susceptible cereal residue, select resistant varieties, manage residue, and obtain expert grain-safety guidance for suspected crops.",
    ),
    _reference(
        "Wheat",
        "Powdery Mildew",
        "Fungal",
        "A fungal wheat disease favored by dense, humid canopies.",
        "White cottony colonies develop on leaves and stems, later becoming gray with small dark fruiting bodies.",
        "Use resistant varieties, avoid excessive nitrogen and crowding, manage volunteer cereals, and maintain balanced crop growth.",
    ),
    _reference(
        "Cotton",
        "Bacterial Blight",
        "Bacterial",
        "A seedborne and residue-associated bacterial disease of cotton.",
        "Angular water-soaked leaf spots, dark stem lesions, and sunken boll spots may appear.",
        "Use clean seed and resistant cultivars, rotate crops, manage infected residue, and avoid field work while foliage is wet.",
    ),
    _reference(
        "Cotton",
        "Fusarium Wilt",
        "Fungal",
        "A soilborne vascular wilt of cotton that may be more severe where root-knot nematodes occur.",
        "Plants yellow, wilt, and become stunted; split stems can show brown vascular discoloration.",
        "Use resistant cultivars, rotate with non-host crops where locally effective, manage nematodes, and prevent contaminated soil movement.",
    ),
    _reference(
        "Cotton",
        "Verticillium Wilt",
        "Fungal",
        "A soilborne vascular disease that restricts water movement in cotton.",
        "Interveinal leaf yellowing progresses to browning, defoliation, stunting, and internal vascular discoloration.",
        "Select tolerant varieties, rotate crops, reduce plant stress, manage residue, and keep field equipment clean.",
    ),
    _reference(
        "Cotton",
        "Cotton Leaf Curl Virus",
        "Viral",
        "A whitefly-transmitted viral disease complex affecting cotton growth and yield.",
        "Leaves curl upward or downward, veins thicken, small leaf-like outgrowths may form, and plants become stunted.",
        "Use resistant varieties and clean planting material, manage whiteflies and alternate hosts with integrated methods, and monitor regional guidance.",
    ),
    _reference(
        "Mango",
        "Anthracnose",
        "Fungal",
        "A common mango disease affecting leaves, flowers, twigs, and fruit, especially in wet weather.",
        "Dark irregular leaf spots, blossom blight, twig dieback, and sunken black fruit lesions may occur.",
        "Prune for airflow, remove infected debris, avoid prolonged surface wetness, and handle fruit carefully after harvest.",
    ),
    _reference(
        "Mango",
        "Powdery Mildew",
        "Fungal",
        "A fungal mango disease that can damage flowers, young fruit, and new growth.",
        "White powdery growth covers flower clusters or young leaves, followed by browning and flower or fruit drop.",
        "Improve canopy airflow, remove heavily affected material, avoid excessive nitrogen, and monitor flowering growth closely.",
    ),
    _reference(
        "Mango",
        "Bacterial Black Spot",
        "Bacterial",
        "A bacterial mango disease that affects leaves, twigs, and fruit and can spread in wind-driven rain.",
        "Angular black leaf lesions with yellow halos, twig cankers, and raised cracked fruit spots may develop.",
        "Use clean nursery stock, sanitize pruning tools, reduce canopy injury and splash, and remove severely infected material appropriately.",
    ),
    _reference(
        "Onion",
        "Purple Blotch",
        "Fungal",
        "A fungal onion disease favored by warm, humid weather and prolonged leaf wetness.",
        "Small water-soaked spots enlarge into purple-brown oval lesions with concentric zones and yellow margins.",
        "Rotate crops, remove infected residue, improve spacing and drainage, and irrigate to limit leaf wetness.",
    ),
    _reference(
        "Onion",
        "Downy Mildew",
        "Oomycete",
        "A moisture-loving disease of onion foliage that can spread quickly in cool, humid conditions.",
        "Pale elongated patches develop gray-violet fuzzy growth, followed by yellowing, collapse, and secondary infection.",
        "Use clean sets or seed, improve airflow, rotate crops, remove volunteers, and avoid irrigation that leaves foliage wet overnight.",
    ),
    _reference(
        "Cucumber",
        "Downy Mildew",
        "Oomycete",
        "A rapidly spreading cucurbit leaf disease favored by humid conditions.",
        "Angular yellow patches bounded by veins develop dark purplish growth beneath leaves and later turn brown.",
        "Use resistant varieties where available, improve airflow, avoid prolonged leaf wetness, and follow regional outbreak alerts.",
    ),
    _reference(
        "Cucumber",
        "Angular Leaf Spot",
        "Bacterial",
        "A bacterial cucurbit disease spread by infected seed, splashing water, and handling wet plants.",
        "Small water-soaked angular lesions become tan, may tear away, and can release cloudy droplets in humidity.",
        "Use clean seed, rotate crops, avoid overhead irrigation and wet-field work, and sanitize tools and equipment.",
    ),
    _reference(
        "Cucumber",
        "Cucumber Mosaic Virus",
        "Viral",
        "A widely hosted virus transmitted mainly by aphids and infected plant material.",
        "Leaves show mosaic mottling, distortion, shoestring growth, and stunting; fruit may be misshapen or mottled.",
        "Use clean transplants, remove infected plants and nearby weed hosts, monitor aphids, and sanitize hands and tools.",
    ),
    _reference(
        "Chili Pepper",
        "Anthracnose",
        "Fungal",
        "A fungal pepper disease that commonly produces fruit rot in warm, wet conditions.",
        "Circular sunken fruit lesions develop tan-to-dark centers, concentric rings, and sometimes salmon-colored spore masses.",
        "Use clean seed and transplants, rotate crops, remove infected fruit, reduce splash, and avoid harvesting wet plants.",
    ),
    _reference(
        "Chili Pepper",
        "Leaf Curl Virus",
        "Viral",
        "A viral disease complex commonly associated with whitefly transmission.",
        "Young leaves curl, crinkle, and become smaller; plants may be stunted with shortened internodes and poor fruit set.",
        "Use healthy transplants, manage whiteflies and weed hosts with integrated methods, rogue confirmed plants, and monitor nearby crops.",
    ),
    _reference(
        "Peanut",
        "Early Leaf Spot",
        "Fungal",
        "A fungal peanut leaf disease that can cause premature defoliation.",
        "Brown circular spots often have yellow halos and are more prominent on upper leaf surfaces.",
        "Rotate crops, bury or manage infected residue, use resistant varieties, and maintain suitable spacing and field scouting.",
    ),
    _reference(
        "Peanut",
        "Late Leaf Spot",
        "Fungal",
        "A fungal peanut disease that often develops later in the season and can severely defoliate plants.",
        "Dark brown-to-black spots usually lack a strong yellow halo and produce spores mainly on lower leaf surfaces.",
        "Use resistant varieties, rotate crops, manage residue and volunteer peanuts, and scout lower foliage regularly.",
    ),
    _reference(
        "Papaya",
        "Papaya Ringspot Virus",
        "Viral",
        "An aphid-transmitted viral disease that affects papaya foliage, stems, and fruit.",
        "Leaves become mottled, narrow, and distorted; oily stem streaks and ring-shaped fruit markings may appear.",
        "Use clean plants and tolerant varieties where available, remove confirmed infected plants, manage weed hosts, and monitor aphids.",
    ),
    _reference(
        "Papaya",
        "Anthracnose",
        "Fungal",
        "A fungal disease that often appears as papaya fruit ripens.",
        "Small water-soaked fruit spots enlarge into sunken dark lesions, sometimes with salmon-colored spore masses.",
        "Remove infected fruit and debris, improve airflow, avoid fruit injury, reduce splash, and keep harvest containers clean.",
    ),
    _reference(
        "Coffee",
        "Coffee Leaf Rust",
        "Fungal",
        "A major fungal coffee disease that reduces functional leaf area and plant vigor.",
        "Pale yellow upper-leaf spots correspond to orange powdery patches below; severe infection causes leaf drop.",
        "Use resistant varieties, maintain balanced shade and nutrition, improve airflow, remove volunteer hosts, and monitor seasonal risk.",
    ),
    _reference(
        "Citrus",
        "Citrus Canker",
        "Bacterial",
        "A regulated bacterial citrus disease that affects leaves, stems, and fruit.",
        "Raised corky lesions develop water-soaked margins and yellow halos; severe infection may cause leaf or fruit drop.",
        "Do not move suspicious plant material, sanitize tools, use certified stock, and report suspected regulated disease to local plant-health authorities.",
    ),
)


def parse_class_label(label: str) -> tuple[str, str, str]:
    """Convert a PlantVillage label into crop, condition, and health status."""

    if "___" not in label:
        cleaned = re.sub(r"[_]+", " ", label).strip()
        return "Unknown", cleaned.title(), "Unknown"
    crop_raw, condition_raw = label.split("___", 1)
    crop = crop_raw.replace(",", "").replace("_", " ").strip()
    aliases = {
        "Corn (maize)": "Corn",
        "Pepper bell": "Bell Pepper",
    }
    crop = aliases.get(crop, crop)
    condition = re.sub(r"[_]+", " ", condition_raw)
    condition = condition.replace("Haunglongbing", "Huanglongbing")
    condition = condition.replace("Two-spotted spider mite", "Two-Spotted Spider Mite")
    condition = condition.strip().title()
    replacements = {
        "Cercospora Leaf Spot Gray Leaf Spot": "Cercospora Leaf Spot / Gray Leaf Spot",
        "Esca (Black Measles)": "Esca (Black Measles)",
        "Leaf Blight (Isariopsis Leaf Spot)": "Leaf Blight (Isariopsis Leaf Spot)",
        "Spider Mites Two-Spotted Spider Mite": "Spider Mites / Two-Spotted Spider Mite",
        "Tomato Yellow Leaf Curl Virus": "Tomato Yellow Leaf Curl Virus",
        "Tomato Mosaic Virus": "Tomato Mosaic Virus",
        "Northern Leaf Blight": "Northern Leaf Blight",
    }
    condition = replacements.get(condition, condition)
    status = "Healthy" if condition.casefold() == "healthy" else "Diseased"
    return crop, condition, status


def get_disease_info(label: str) -> DiseaseInfo:
    crop, condition, status = parse_class_label(label)
    if status == "Healthy":
        return DiseaseInfo(
            crop=crop,
            condition="Healthy",
            status="Healthy",
            description=f"The model found no supported disease pattern for this {crop.lower()} leaf.",
            symptoms="No strong visual pattern matching the supported disease classes was detected.",
            prevention="Continue routine scouting, balanced nutrition, clean tools, suitable watering, and good airflow.",
            expert_warning=EXPERT_WARNING,
            category="Healthy",
            model_supported=True,
        )
    details = _DETAILS.get(
        condition,
        (
            f"A possible {condition.lower()} pattern was detected on this {crop.lower()} leaf.",
            "Visible symptoms can overlap with nutrient stress, pest injury, weather damage, or other diseases.",
            "Keep tools clean, reduce prolonged leaf wetness, remove heavily affected debris where appropriate, and monitor nearby plants.",
        ),
    )
    return DiseaseInfo(
        crop,
        condition,
        status,
        details[0],
        details[1],
        details[2],
        EXPERT_WARNING,
        _CATEGORY_BY_CONDITION.get(condition, "Fungal"),
        True,
    )


def supported_conditions(labels: list[str]) -> list[DiseaseInfo]:
    return [get_disease_info(label) for label in labels]


def reference_conditions() -> list[DiseaseInfo]:
    """Return advisory entries that are not prediction classes for the installed CNN."""

    return list(_REFERENCE_CONDITIONS)


def library_conditions(labels: list[str]) -> list[DiseaseInfo]:
    """Return the complete, de-duplicated knowledge library."""

    combined = [*supported_conditions(labels), *reference_conditions()]
    unique = {(item.crop.casefold(), item.condition.casefold()): item for item in combined}
    return sorted(
        unique.values(), key=lambda item: (item.crop.casefold(), item.condition.casefold())
    )
