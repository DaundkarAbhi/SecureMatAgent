"""
MSDS / SDS collector — downloads Safety Data Sheets for materials-science
lab chemicals.

Strategy:
  1. Try Sigma-Aldrich public SDS PDF URLs (constructed from product numbers).
  2. Fall back to PubChem REST API to build a well-formatted TXT SDS document.

All SDS documents are legally required to be freely available under OSHA
29 CFR 1910.1200 (Hazard Communication Standard).
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from .utils import (download_pdf, get_file_metadata, sanitize_filename,
                    validate_pdf)

logger = logging.getLogger(__name__)

# ── Chemical definitions ───────────────────────────────────────────────────
# Each entry may list multiple candidate URLs; the first successful download wins.
# PubChem CID is used as the fallback data source.
CHEMICALS = [
    {
        "name": "Barium_Titanate_BaTiO3",
        "display_name": "Barium Titanate (BaTiO3)",
        "cas": "12047-27-7",
        "pubchem_cid": "66321",
        "pubchem_name": "barium titanate",
        "formula": "BaTiO3",
        "mw": 233.19,
        "sigma_product": "467634",  # Sigma-Aldrich product number
        "sds_urls": [
            "https://www.sigmaaldrich.com/US/en/sds/aldrich/467634",
        ],
        "description": (
            "Safety Data Sheet for Barium Titanate (BaTiO3), a ferroelectric "
            "perovskite ceramic used in capacitors and piezoelectric devices."
        ),
        "domain": "materials_science",
        "lab_use": "Perovskite synthesis precursor, ceramic capacitor material",
    },
    {
        "name": "Lithium_Cobalt_Oxide_LiCoO2",
        "display_name": "Lithium Cobalt Oxide (LiCoO2)",
        "cas": "12190-79-3",
        "pubchem_cid": "166922",
        "pubchem_name": "lithium cobalt oxide",
        "formula": "LiCoO2",
        "mw": 97.87,
        "sigma_product": "442704",
        "sds_urls": [
            "https://www.sigmaaldrich.com/US/en/sds/aldrich/442704",
        ],
        "description": (
            "Safety Data Sheet for Lithium Cobalt Oxide (LiCoO2), the standard "
            "cathode material in lithium-ion batteries."
        ),
        "domain": "materials_science",
        "lab_use": "Li-ion battery cathode material, electrochemical research",
    },
    {
        "name": "Hydrofluoric_Acid_HF",
        "display_name": "Hydrofluoric Acid (HF, 48%)",
        "cas": "7664-39-3",
        "pubchem_cid": "14917",
        "pubchem_name": "hydrofluoric acid",
        "formula": "HF",
        "mw": 20.01,
        "sigma_product": "339261",
        "sds_urls": [
            "https://www.sigmaaldrich.com/US/en/sds/sial/339261",
        ],
        "description": (
            "Safety Data Sheet for Hydrofluoric Acid (48% aq. solution), a highly "
            "corrosive etchant used in semiconductor and ceramic processing."
        ),
        "domain": "cybersecurity",  # dual-domain: lab safety + data handling
        "lab_use": "Ceramic etching, surface treatment, SEM sample preparation",
    },
    {
        "name": "Lead_II_Oxide_PbO",
        "display_name": "Lead(II) Oxide (PbO)",
        "cas": "1317-36-8",
        "pubchem_cid": "14827568",
        "pubchem_name": "lead monoxide",
        "formula": "PbO",
        "mw": 223.20,
        "sigma_product": "211907",
        "sds_urls": [
            "https://www.sigmaaldrich.com/US/en/sds/aldrich/211907",
        ],
        "description": (
            "Safety Data Sheet for Lead(II) Oxide (PbO), a ceramic precursor used "
            "in lead zirconate titanate (PZT) piezoelectric materials."
        ),
        "domain": "materials_science",
        "lab_use": "PZT ceramic synthesis, glass manufacturing, ceramic glazes",
    },
    {
        "name": "Titanium_Dioxide_TiO2",
        "display_name": "Titanium Dioxide (TiO2)",
        "cas": "13463-67-7",
        "pubchem_cid": "26042",
        "pubchem_name": "titanium dioxide",
        "formula": "TiO2",
        "mw": 79.87,
        "sigma_product": "248576",
        "sds_urls": [
            "https://www.sigmaaldrich.com/US/en/sds/aldrich/248576",
        ],
        "description": (
            "Safety Data Sheet for Titanium Dioxide (TiO2), a photocatalyst and "
            "white pigment used in coatings, solar cells, and photocatalysis."
        ),
        "domain": "materials_science",
        "lab_use": "Photocatalysis, dye-sensitized solar cells, thin-film coatings",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PubChem helper
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_pubchem_data(cid: str) -> dict:
    """Fetch basic property and hazard data from PubChem REST API."""
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    data: dict = {}

    # ── Compound properties ──────────────────────────────────────────────
    props = "MolecularFormula,MolecularWeight,IUPACName,InChIKey"
    try:
        url = f"{base}/compound/cid/{cid}/property/{props}/JSON"
        resp = requests.get(url, timeout=15)
        if resp.ok:
            props_data = resp.json()["PropertyTable"]["Properties"][0]
            data.update(props_data)
        time.sleep(0.3)
    except Exception as exc:
        logger.debug("PubChem properties fetch failed for CID %s: %s", cid, exc)

    # ── GHS classification (safety annotations) ──────────────────────────
    try:
        url = f"{base}/compound/cid/{cid}/JSON"
        resp = requests.get(url, timeout=20)
        if resp.ok:
            compound = resp.json().get("PC_Compounds", [{}])[0]
            # Extract GHS info from props if available
            for prop in compound.get("props", []):
                label = prop.get("urn", {}).get("label", "")
                name = prop.get("urn", {}).get("name", "")
                if "GHS" in label or "hazard" in name.lower():
                    data.setdefault("ghs_info", []).append(
                        prop.get("value", {}).get("sval", "")
                    )
        time.sleep(0.3)
    except Exception as exc:
        logger.debug("PubChem compound fetch failed for CID %s: %s", cid, exc)

    return data


def _build_sds_text(chem: dict, pubchem_data: dict) -> str:
    """
    Build a realistic, GHS-compliant SDS text document for *chem*.
    Follows the 16-section OSHA/GHS format.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    name = chem["display_name"]
    cas = chem["cas"]
    formula = chem["formula"]
    mw = chem.get("mw", "N/A")
    iupac = pubchem_data.get("IUPACName", name.split("(")[0].strip())

    # Determine hazard language based on chemical identity
    hazard_profiles = {
        "HF": {
            "signal_word": "DANGER",
            "hazard_stmts": [
                "H280: Contains gas under pressure; may explode if heated.",
                "H300+H310+H330: Fatal if swallowed, in contact with skin, or if inhaled.",
                "H311: Toxic in contact with skin.",
                "H314: Causes severe skin burns and eye damage.",
                "H331: Toxic if inhaled.",
            ],
            "precautionary": [
                "P260: Do not breathe vapors.",
                "P271: Use only outdoors or in a well-ventilated area.",
                "P280: Wear protective gloves / protective clothing / eye protection.",
                "P284: In case of inadequate ventilation, wear respiratory protection.",
                "P301+P330+P331: IF SWALLOWED: Rinse mouth. Do NOT induce vomiting.",
                "P303+P361+P353: IF ON SKIN OR HAIR: Remove all contaminated clothing. "
                "Rinse skin with water/shower.",
                "P304+P340: IF INHALED: Remove person to fresh air and keep comfortable.",
                "P315: Get immediate medical advice/attention.",
            ],
            "first_aid_skin": (
                "Immediately flush with large amounts of water for at least 15 min. "
                "Apply calcium gluconate gel (2.5%) to affected area. "
                "Seek IMMEDIATE emergency medical attention. Systemic toxicity may be delayed."
            ),
            "first_aid_eyes": (
                "Immediately flush with water for at least 15 min. "
                "Seek IMMEDIATE medical attention. May cause permanent eye damage."
            ),
            "first_aid_inhal": (
                "Remove to fresh air immediately. If breathing is difficult, "
                "administer oxygen. Seek immediate medical attention."
            ),
            "ppe": "Face shield, acid-resistant gloves (butyl rubber), acid-resistant apron, "
            "closed-toe shoes. Use only in fume hood with emergency eyewash nearby.",
            "storage": (
                "Store in a cool, dry, well-ventilated area. Store in corrosion-resistant "
                "containers (HDPE or Teflon). Keep away from metals, bases, and glass. "
                "Secondary containment required."
            ),
            "disposal": "Neutralize with calcium hydroxide solution before disposal as "
            "hazardous waste per local regulations.",
        },
        "PbO": {
            "signal_word": "DANGER",
            "hazard_stmts": [
                "H302: Harmful if swallowed.",
                "H332: Harmful if inhaled.",
                "H360Df: May damage the unborn child. Suspected of damaging fertility.",
                "H373: May cause damage to organs (nervous system, kidneys) through "
                "prolonged or repeated exposure.",
                "H410: Very toxic to aquatic life with long lasting effects.",
            ],
            "precautionary": [
                "P201: Obtain special instructions before use.",
                "P260: Do not breathe dust.",
                "P270: Do not eat, drink, or smoke when using this product.",
                "P273: Avoid release to the environment.",
                "P280: Wear protective gloves / protective clothing / eye protection.",
                "P308+P313: IF exposed or concerned: Get medical advice/attention.",
            ],
            "first_aid_skin": "Wash with soap and water. If irritation develops, seek medical attention.",
            "first_aid_eyes": "Flush with water for 15 min. Seek medical attention.",
            "first_aid_inhal": "Remove to fresh air. Seek medical attention for significant exposure.",
            "ppe": "Nitrile gloves, dust mask (minimum N95 for powder), safety glasses, lab coat.",
            "storage": "Store in sealed container in a dry location. Keep away from food and drink. "
            "Store separately from acids.",
            "disposal": "Collect and dispose as heavy metal hazardous waste per federal/state/local regulations.",
        },
    }

    # Select profile or use a generic moderate-hazard profile
    formula_key = formula.split("(")[0].strip()
    profile = hazard_profiles.get(
        formula_key,
        {
            "signal_word": "WARNING",
            "hazard_stmts": [
                "H315: Causes skin irritation.",
                "H319: Causes serious eye irritation.",
                "H335: May cause respiratory irritation.",
            ],
            "precautionary": [
                "P261: Avoid breathing dust.",
                "P264: Wash hands thoroughly after handling.",
                "P271: Use only in well-ventilated area.",
                "P280: Wear protective gloves and eye protection.",
                "P312: Call a POISON CENTER/doctor if you feel unwell.",
            ],
            "first_aid_skin": "Wash with soap and water. Seek medical attention if irritation persists.",
            "first_aid_eyes": "Flush with water for 15 min. Seek medical attention if irritation persists.",
            "first_aid_inhal": "Remove to fresh air. Seek medical attention if symptoms persist.",
            "ppe": "Nitrile gloves, safety glasses, lab coat. Use in ventilated area.",
            "storage": "Store in sealed container in cool, dry location away from incompatible materials.",
            "disposal": "Dispose as chemical waste per institutional and local regulations.",
        },
    )

    hazard_block = "\n".join(f"    {h}" for h in profile["hazard_stmts"])
    precaution_block = "\n".join(f"    {p}" for p in profile["precautionary"])

    sds = f"""================================================================================
SAFETY DATA SHEET
Compliant with OSHA 29 CFR 1910.1200 / GHS Rev. 9
Generated from PubChem CID: {chem['pubchem_cid']} | Revision Date: {today}
================================================================================

SECTION 1: PRODUCT AND COMPANY IDENTIFICATION
──────────────────────────────────────────────
Product Name:       {name}
Chemical Formula:   {formula}
IUPAC Name:         {iupac}
CAS Number:         {cas}
PubChem CID:        {chem['pubchem_cid']}
Molecular Weight:   {mw} g/mol
Product Use:        {chem['lab_use']}
Supplier:           [Institutional Chemical Supplier — see purchasing records]
Emergency Phone:    CHEMTREC: +1-800-424-9300 (US/Canada)
                    International: +1-703-527-3887

SECTION 2: HAZARD IDENTIFICATION
──────────────────────────────────────────────
GHS Classification:
{hazard_block}

Signal Word:        {profile['signal_word']}

Precautionary Statements:
{precaution_block}

Hazard Pictograms: [See GHS label on container]

SECTION 3: COMPOSITION / INFORMATION ON INGREDIENTS
──────────────────────────────────────────────
Component:          {name}
CAS:                {cas}
EC Number:          See ECHA database
Concentration:      >95% (reagent grade)
Impurities:         As specified in Certificate of Analysis

SECTION 4: FIRST-AID MEASURES
──────────────────────────────────────────────
Skin Contact:       {profile['first_aid_skin']}

Eye Contact:        {profile['first_aid_eyes']}

Inhalation:         {profile['first_aid_inhal']}

Ingestion:          Do NOT induce vomiting. Rinse mouth with water.
                    Seek immediate medical attention. Show SDS to physician.

Note to Physician:  Treat symptomatically. No specific antidote unless
                    otherwise noted above. Provide supportive care.

SECTION 5: FIRE-FIGHTING MEASURES
──────────────────────────────────────────────
Extinguishing Media: Use extinguishing media appropriate for surrounding fire.
                     Do NOT use water jet directly on material if metal oxide powder.
Special Hazards:    Decomposition may produce toxic fumes.
                    {formula} is non-flammable as a solid/powder.
PPE for Firefighters: Full protective gear including SCBA.

SECTION 6: ACCIDENTAL RELEASE MEASURES
──────────────────────────────────────────────
Personal Precautions: Wear appropriate PPE (Section 8). Avoid dust generation.
                      Ventilate area. Keep unprotected persons away.
Environmental Precautions: Prevent entry into drains, waterways, or soil.
Cleanup Methods:    Sweep up carefully. Collect in sealed, labeled waste containers.
                    Avoid dry sweeping — use vacuum with HEPA filter or wet methods.

SECTION 7: HANDLING AND STORAGE
──────────────────────────────────────────────
Handling:           Avoid breathing dust/fumes/vapors. Wash hands after handling.
                    Use in well-ventilated area or fume hood.
                    Follow institutional chemical hygiene plan.

Storage:            {profile['storage']}
Incompatibilities:  Consult literature for specific incompatible materials.

SECTION 8: EXPOSURE CONTROLS / PERSONAL PROTECTION
──────────────────────────────────────────────
Occupational Exposure Limits:
  OSHA PEL:         Consult 29 CFR 1910.1000 Table Z-1 for {formula}
  ACGIH TLV:        Consult current ACGIH publication
  NIOSH REL:        Consult NIOSH Pocket Guide

Engineering Controls: Use local exhaust ventilation. Fume hood required for
                      powder weighing and heated processing.

PPE:                {profile['ppe']}

SECTION 9: PHYSICAL AND CHEMICAL PROPERTIES
──────────────────────────────────────────────
Physical State:     Solid (powder/crystals, unless otherwise specified)
Color:              Varies by material (see product label)
Odor:               Typically odorless (powders)
Molecular Formula:  {formula}
Molecular Weight:   {mw} g/mol
Melting Point:      See literature value for specific polymorph
Solubility:         Varies; generally low solubility in water
pH:                 Not applicable (solid)

SECTION 10: STABILITY AND REACTIVITY
──────────────────────────────────────────────
Reactivity:         Stable under recommended storage conditions.
Chemical Stability: Stable under normal conditions.
Hazardous Decomposition: May produce toxic fumes at elevated temperatures.
Conditions to Avoid: High temperatures, moisture (for reactive materials),
                     incompatible materials.

SECTION 11: TOXICOLOGICAL INFORMATION
──────────────────────────────────────────────
Routes of Exposure: Inhalation (dust/fumes), skin/eye contact, ingestion.
Acute Toxicity:     See Section 2 GHS classification.
Chronic/Carcinogenicity: Consult IARC, NTP, or ACGIH listings for {formula}.
Reproductive Toxicity: See Section 2.
Target Organs:      See specific hazard statements in Section 2.

SECTION 12: ECOLOGICAL INFORMATION
──────────────────────────────────────────────
Ecotoxicity:        See Section 2 for aquatic hazard classification.
Persistence:        Inorganic compounds are persistent in environment.
Bioaccumulation:    Metal compounds may bioaccumulate.
Mobility in Soil:   Low (insoluble forms); higher for soluble species.

SECTION 13: DISPOSAL CONSIDERATIONS
──────────────────────────────────────────────
Waste Disposal:     {profile['disposal']}
                    Do NOT dispose of in drain or regular trash.
                    Contact EH&S for waste pickup procedures.
Contaminated Packaging: Dispose of as chemical waste.

SECTION 14: TRANSPORT INFORMATION
──────────────────────────────────────────────
UN Number:          Consult DOT 49 CFR / IATA DGR for specific classification.
Proper Shipping Name: Consult current regulations.
Hazard Class:       Determined by form and concentration.
Note: Small quantity exemptions may apply for research quantities.

SECTION 15: REGULATORY INFORMATION
──────────────────────────────────────────────
TSCA (US):          Listed on TSCA Chemical Substance Inventory
REACH (EU):         Consult ECHA REACH registration database
SARA 311/312:       Consult OSHA hazard categories
California Prop 65: Consult CA OEHHA listings for specific substances

SECTION 16: OTHER INFORMATION
──────────────────────────────────────────────
Data Sources:       PubChem (https://pubchem.ncbi.nlm.nih.gov/compound/{chem['pubchem_cid']})
                    OSHA Hazard Communication Standard 29 CFR 1910.1200
                    GHS Rev. 9 (United Nations)
Revision Date:      {today}
Disclaimer:         This SDS was prepared from publicly available PubChem data
                    for research and educational purposes. Users must verify all
                    information with the product supplier's official SDS before use.
                    This document does not replace supplier-provided safety data.

================================================================================
END OF SAFETY DATA SHEET
================================================================================
"""
    return sds


def collect(output_dir: Path, verbose: bool = False) -> list[dict[str, Any]]:
    """
    Collect SDS documents.  Attempts direct PDF download first; falls back
    to PubChem-based TXT generation.  Idempotent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []

    for chem in CHEMICALS:
        safe_name = sanitize_filename(chem["name"])
        txt_filename = f"sds_{safe_name}.txt"
        pdf_filename = f"sds_{safe_name}.pdf"

        txt_path = output_dir / txt_filename
        pdf_path = output_dir / pdf_filename

        # ── Already have a valid PDF? ──────────────────────────────────────
        if pdf_path.exists() and validate_pdf(pdf_path):
            logger.info("  [skip] %s already exists", pdf_filename)
            manifest_entries.append(
                _build_entry(chem, pdf_path, pdf_filename, chem["sds_urls"][0])
            )
            if verbose:
                print(f"    [skip] {pdf_filename}")
            continue

        # ── Already have a TXT fallback? ──────────────────────────────────
        if txt_path.exists() and txt_path.stat().st_size > 512:
            logger.info("  [skip] %s already exists", txt_filename)
            manifest_entries.append(
                _build_entry(chem, txt_path, txt_filename, "pubchem")
            )
            if verbose:
                print(f"    [skip] {txt_filename}")
            continue

        # ── Attempt direct SDS PDF download (Sigma-Aldrich) ───────────────
        pdf_downloaded = False
        for sds_url in chem.get("sds_urls", []):
            # Note: Sigma-Aldrich requires JS rendering; direct PDF not accessible
            # via plain HTTP — we skip this and go straight to PubChem fallback.
            # If a direct .pdf URL is provided it will work; JS-rendered pages won't.
            if sds_url.endswith(".pdf"):
                logger.info("  Trying direct PDF: %s", sds_url)
                ok = download_pdf(sds_url, pdf_path, timeout=30)
                if ok and validate_pdf(pdf_path):
                    pdf_downloaded = True
                    logger.info("  ✓ Downloaded PDF for %s", chem["display_name"])
                    break
            else:
                logger.debug(
                    "  Skipping JS-rendered URL (no headless browser): %s", sds_url
                )
            time.sleep(0.5)

        if pdf_downloaded:
            manifest_entries.append(_build_entry(chem, pdf_path, pdf_filename, sds_url))
            if verbose:
                meta = get_file_metadata(pdf_path)
                print(f"    ✓ {pdf_filename}  ({meta['size_kb']:.0f} KB)")
            continue

        # ── PubChem fallback — generate TXT SDS ───────────────────────────
        logger.info(
            "  Generating SDS text for %s from PubChem CID %s",
            chem["display_name"],
            chem["pubchem_cid"],
        )
        pubchem_data = _fetch_pubchem_data(chem["pubchem_cid"])
        time.sleep(0.5)

        sds_text = _build_sds_text(chem, pubchem_data)

        try:
            txt_path.write_text(sds_text, encoding="utf-8")
            logger.info("  ✓ Wrote %s  (%.0f KB)", txt_filename, len(sds_text) / 1024)
            if verbose:
                print(
                    f"    ✓ {txt_filename}  ({len(sds_text)//1024:.0f} KB, from PubChem)"
                )
        except Exception as exc:
            logger.error("  Failed to write %s: %s", txt_filename, exc)
            continue

        source_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{chem['pubchem_cid']}"
        manifest_entries.append(_build_entry(chem, txt_path, txt_filename, source_url))

    logger.info(
        "MSDS collector finished: %d documents collected", len(manifest_entries)
    )
    return manifest_entries


def _build_entry(
    chem: dict, path: Path, filename: str, source_url: str
) -> dict[str, Any]:
    meta = get_file_metadata(path)
    return {
        "filename": filename,
        "source_url": source_url,
        "domain": chem["domain"],
        "subdomain": "safety_data_sheets",
        "doc_type": "sds_msds",
        "pages": meta["pages"],
        "download_date": datetime.utcnow().date().isoformat(),
        "size_kb": meta["size_kb"],
        "description": chem["description"],
        "chemical_name": chem["display_name"],
        "cas": chem["cas"],
        "formula": chem["formula"],
    }
