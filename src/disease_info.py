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
        )
    details = _DETAILS.get(
        condition,
        (
            f"A possible {condition.lower()} pattern was detected on this {crop.lower()} leaf.",
            "Visible symptoms can overlap with nutrient stress, pest injury, weather damage, or other diseases.",
            "Keep tools clean, reduce prolonged leaf wetness, remove heavily affected debris where appropriate, and monitor nearby plants.",
        ),
    )
    return DiseaseInfo(crop, condition, status, details[0], details[1], details[2], EXPERT_WARNING)


def supported_conditions(labels: list[str]) -> list[DiseaseInfo]:
    return [get_disease_info(label) for label in labels]

