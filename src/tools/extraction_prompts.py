"""
Prompt templates for DataExtractorTool.

Each function returns a fully-formatted prompt string that instructs
Mistral 7B to extract a specific data type and return ONLY valid JSON.
"""

from __future__ import annotations


def material_properties_prompt(query: str, context: str) -> str:
    return f"""You are a materials science data extraction assistant.
CRITICAL: Respond with ONLY a valid JSON object. No markdown, no code fences, no prose, no explanation. Start your response with {{ and end with }}.

TASK: Extract material properties from the document chunks below.
USER QUERY: {query}

DOCUMENT CHUNKS:
{context}

INSTRUCTIONS:
- Extract ALL numerical material properties you can find.
- If a value is not present, omit that key entirely.
- "sources" must list the source filenames or document titles mentioned in the chunks.
- Lattice parameters should include units (e.g., "3.905 Å").

REQUIRED JSON SCHEMA:
{{
  "material": "<name of the material or compound>",
  "properties": {{
    "crystal_system": "<cubic|tetragonal|hexagonal|orthorhombic|monoclinic|triclinic|rhombohedral>",
    "space_group": "<e.g. Fm-3m or 225>",
    "lattice_a": "<value with unit>",
    "lattice_b": "<value with unit>",
    "lattice_c": "<value with unit>",
    "lattice_alpha": "<value in degrees>",
    "lattice_beta": "<value in degrees>",
    "lattice_gamma": "<value in degrees>",
    "density": "<value with unit>",
    "band_gap": "<value with unit>",
    "melting_point": "<value with unit>",
    "additional": {{
      "<property_name>": "<value with unit>"
    }}
  }},
  "sources": ["<source1>", "<source2>"]
}}

Return ONLY the JSON object. Start with {{ and end with }}:"""


def safety_data_prompt(query: str, context: str) -> str:
    return f"""You are a chemical safety data extraction assistant.
CRITICAL: Respond with ONLY a valid JSON object. No markdown, no code fences, no prose, no explanation. Start your response with {{ and end with }}.

TASK: Extract safety and hazard information from the document chunks below.
USER QUERY: {query}

DOCUMENT CHUNKS:
{context}

INSTRUCTIONS:
- Extract all GHS classifications, hazard statements, first aid measures, and PPE requirements.
- "first_aid" maps exposure route to recommended action (e.g. "inhalation": "Move to fresh air").
- "sources" must list the source filenames or document titles mentioned in the chunks.

REQUIRED JSON SCHEMA:
{{
  "chemical": "<chemical name or CAS number>",
  "ghs_classification": "<e.g. Flammable Liquid Cat. 2, Acute Toxicity Cat. 3>",
  "hazard_statements": ["<H-code: description>", ...],
  "first_aid": {{
    "inhalation": "<action>",
    "skin_contact": "<action>",
    "eye_contact": "<action>",
    "ingestion": "<action>"
  }},
  "ppe_required": ["<gloves>", "<goggles>", ...],
  "storage_disposal": "<storage and disposal instructions>",
  "sources": ["<source1>", "<source2>"]
}}

Return ONLY the JSON object. Start with {{ and end with }}:"""


def compliance_prompt(query: str, context: str) -> str:
    return f"""You are a compliance and regulatory requirements extraction assistant.
CRITICAL: Respond with ONLY a valid JSON object. No markdown, no code fences, no prose, no explanation. Start your response with {{ and end with }}.

TASK: Extract compliance controls and requirements from the document chunks below.
USER QUERY: {query}

DOCUMENT CHUNKS:
{context}

INSTRUCTIONS:
- Extract all control IDs, titles, and descriptions found in the text.
- "requirements" is a list of control items; include as many as found in the text.
- "sources" must list the source filenames or document titles mentioned in the chunks.

REQUIRED JSON SCHEMA:
{{
  "framework": "<e.g. NIST SP 800-53, ISO 27001, SOC 2>",
  "control_family": "<e.g. Access Control, Incident Response>",
  "requirements": [
    {{
      "id": "<control ID, e.g. AC-1>",
      "title": "<short title>",
      "description": "<full requirement text>"
    }}
  ],
  "sources": ["<source1>", "<source2>"]
}}

Return ONLY the JSON object. Start with {{ and end with }}:"""


def table_extraction_prompt(query: str, context: str) -> str:
    return f"""You are a document table extraction assistant.
CRITICAL: Respond with ONLY a valid JSON object. No markdown, no code fences, no prose, no explanation. Start your response with {{ and end with }}.

TASK: Find and extract tabular data from the document chunks below.
USER QUERY: {query}

DOCUMENT CHUNKS:
{context}

INSTRUCTIONS:
- Identify the most relevant table in the text and extract it completely.
- "headers" is a list of column header strings.
- "rows" is a list of rows; each row is a list of cell values (strings).
- Preserve all numeric values exactly as they appear.
- "sources" must list the source filenames or document titles mentioned in the chunks.

REQUIRED JSON SCHEMA:
{{
  "table_title": "<descriptive title for the extracted table>",
  "headers": ["<col1>", "<col2>", "<col3>"],
  "rows": [
    ["<r1c1>", "<r1c2>", "<r1c3>"],
    ["<r2c1>", "<r2c2>", "<r2c3>"]
  ],
  "sources": ["<source1>", "<source2>"]
}}

Return ONLY the JSON object. Start with {{ and end with }}:"""


def general_extraction_prompt(query: str, context: str) -> str:
    return f"""You are a document data extraction assistant.
CRITICAL: Respond with ONLY a valid JSON object. No markdown, no code fences, no prose, no explanation. Start your response with {{ and end with }}.

TASK: Extract key facts and data points from the document chunks below.
USER QUERY: {query}

DOCUMENT CHUNKS:
{context}

INSTRUCTIONS:
- Extract the most important facts, values, and findings relevant to the query.
- Each finding is a key-value pair: {{"key": "<label>", "value": "<data>"}}.
- Include units, dates, version numbers, and identifiers where present.
- "sources" must list the source filenames or document titles mentioned in the chunks.

REQUIRED JSON SCHEMA:
{{
  "document": "<document name or topic>",
  "key_findings": [
    {{"key": "<label>", "value": "<extracted value>"}},
    {{"key": "<label>", "value": "<extracted value>"}}
  ],
  "sources": ["<source1>", "<source2>"]
}}

Return ONLY the JSON object. Start with {{ and end with }}:"""


def retry_suffix(error_message: str) -> str:
    """Appended to a prompt on the retry attempt."""
    return (
        f"\n\nNOTE: Your previous response failed JSON parsing with this error: {error_message}\n"
        "Please return ONLY a valid JSON object with no extra text, no markdown fences, "
        "no explanation. Start your response with {{ and end with }}."
    )
