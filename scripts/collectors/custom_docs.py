"""
Custom documents generator — creates realistic synthetic lab documents for the
SecureMatAgent corpus.

These documents are realistic enough to serve as training/retrieval material
but are original works authored for this project (not scraped content).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import get_file_metadata

logger = logging.getLogger(__name__)

TODAY = datetime.utcnow().strftime("%Y-%m-%d")


# ── Document content definitions ──────────────────────────────────────────

XRD_PROTOCOL = """\
# XRD Sample Preparation Protocol
**Document ID:** LAB-PROC-XRD-001
**Version:** 2.3
**Effective Date:** {date}
**Author:** Materials Characterization Facility
**Review Cycle:** Annual

---

## 1. Purpose and Scope

This protocol describes the standard procedure for preparing powder and bulk samples
for X-ray diffraction (XRD) analysis using Bragg-Brentano geometry on the facility's
Bruker D8 Advance diffractometer (Cu Kα radiation, λ = 1.54056 Å).

Applies to: powder samples (ceramic, pharmaceutical, geological), thin-film specimens,
and oriented single-crystal mounts. Excludes: high-pressure cells and capillary geometry
(see LAB-PROC-XRD-002).

---

## 2. Equipment and Materials

### 2.1 Diffractometer Settings (Bruker D8 Advance)
| Parameter | Standard Scan | High-Resolution |
|-----------|--------------|-----------------|
| Voltage   | 40 kV        | 40 kV           |
| Current   | 40 mA        | 40 mA           |
| Divergence slit | 0.6 mm | 0.2 mm         |
| Anti-scatter slit | 8 mm | 4 mm           |
| Detector slit | 0.1 mm  | 0.05 mm         |
| Step size | 0.02 °2θ     | 0.01 °2θ        |
| Count time | 0.5 s/step  | 2 s/step        |
| Scan range | 10–80 °2θ   | 20–60 °2θ       |

### 2.2 Sample Holders
- **Zero-background Si holder** — preferred for small amounts (<50 mg) and amorphous content quantification
- **Aluminum holder (25 mm cavity)** — standard; do not use with Al-containing samples (fluorescence)
- **PMMA spinner holder** — for preferred-orientation minimization; use with granular powders
- **Glass slide** — for reference materials and quick checks only (introduces background)

### 2.3 Sample Preparation Materials
- Agate mortar and pestle (dedicated per material type — avoid cross-contamination)
- Analytical balance (±0.1 mg)
- Spatulas (stainless steel; Teflon for reactive materials)
- Isopropanol (99.5% for wet grinding)
- Backfill press assembly (for front-loading holders)
- Reference standard: NIST SRM 640f Silicon (a = 5.43123 Å at 25°C)

---

## 3. Step-by-Step Procedure

### 3.1 Sample Grinding
1. Weigh approximately 200–500 mg of sample into the labeled agate mortar.
2. Grind with pestle using circular motion for **5 minutes minimum**. Target particle size:
   1–10 μm (powder should feel silky between fingers, not gritty).
3. For hard materials (ceramics, minerals): grind dry for 3 min, then add 2–3 drops of
   isopropanol and grind wet for additional 2 min. Allow to dry completely before loading.
4. **CAUTION:** For toxic powders (PbO, BaTiO3, CoO compounds), perform all grinding in
   the designated fume hood wearing N95 respirator, nitrile gloves, and lab coat.
5. Transfer powder to a clean labeled vial. Record mass used.

### 3.2 Sample Mounting (Back-Loading, Standard Holder)
1. Invert the sample holder so the cavity faces down onto a flat glass plate.
2. Fill the cavity with powder using a spatula, working in layers.
3. Tamp gently with the backfill plate to compact (do NOT over-press — causes preferred orientation).
4. Scrape excess powder flush with the holder surface using a straight-edge spatula.
5. Flip the holder carefully. The sample surface should be level ± 0.1 mm with the holder face.
6. Inspect: surface should appear matte and uniform. Shiny patches indicate preferred orientation.

### 3.3 Mounting the Holder in the Diffractometer
1. Power on the X-ray tube (Bruker system must be logged in with DIFFRAC.SUITE software).
2. Allow tube to warm up for **10 minutes** if cold start (shutter closed).
3. Open the anti-scatter enclosure. **Verify the shutter indicator is CLOSED.**
4. Slide holder into the spinner stage bracket until the retention clip clicks.
5. Verify spinner rotation engages (green LED on controller).
6. Close enclosure. Log sample ID and date in the instrument logbook.

### 3.4 Alignment Check
1. Run the alignment routine (DIFFRAC.SUITE → Instrument → Auto-Align) if last alignment
   was >1 week ago or after any maintenance.
2. Check beam center: run a 5-scan at θ = 0°; peak should be symmetric ± 0.01°.
3. For high-resolution work: run Si SRM 640f reference (20–50 °2θ, step 0.01°) and
   verify a(Si) = 5.43123 ± 0.00010 Å after LeBail refinement in TOPAS.

### 3.5 Data Collection
1. In DIFFRAC.SUITE, create a new measurement file. Name format: `YYYYMMDD_SampleID_scantype.raw`
2. Enter scan parameters per Table in Section 2.1 (copy appropriate method template).
3. Confirm: X-ray shutter will open automatically when scan starts.
4. Start scan. Expected times: Standard = ~25 min; High-Resolution = ~80 min.
5. Do NOT leave the diffractometer unattended for scans >60 min without notifying the lab supervisor.

### 3.6 Data Export and Archiving
1. After scan completes, export as both `.raw` (native) and `.xy` (2-column ASCII).
2. Copy files immediately to the network share: `\\\\labserver\\XRD_Data\\[PI_Group]\\[Sample_ID]\\`
3. Back up raw data within 24 hours per the lab data management policy (LAB-POLICY-DM-001).
4. Do NOT store data only on the instrument computer — it is not backed up.

---

## 4. Troubleshooting

| Problem | Likely Cause | Corrective Action |
|---------|-------------|-------------------|
| Broad, shifted peaks | Sample surface below holder plane | Re-mount; check backfill |
| Missing peaks vs. reference | Wrong phase, wrong λ assumption | Verify compound and radiation source |
| High background between peaks | Fluorescence (Fe, Co, Ni with Cu Kα) | Switch to Co Kα source or use energy-discriminating detector |
| Peak splitting | Sample not centered; instrument misalignment | Re-align; run Si standard |
| Preferred orientation | Plate-like or needle-like crystallites | Use spinner; try side-drift loading; consider spray-drying |
| Amorphous hump 15–30 °2θ | Amorphous phase or substrate contribution | Use zero-background holder; subtract background |
| Software crash mid-scan | Known issue with DIFFRAC.SUITE v5.x | Save data, restart software, re-initialize goniometer |

---

## 5. Safety

### 5.1 Radiation Safety
- The diffractometer X-ray enclosure is **interlocked**. The shutter closes automatically if the
  door is opened. Never defeat interlocks.
- Annual radiation badge review is mandatory for all registered users.
- If a shutter failure alarm occurs: evacuate the area immediately and contact the Radiation
  Safety Officer (ext. 5678) before re-entering.
- Pregnant users must consult Radiation Safety before operating the instrument.

### 5.2 Chemical Handling
- Review the SDS for each sample material before grinding. See the lab SDS binder or
  query the LabChemSafe database.
- Use designated fume hoods for toxic powder preparation (PbO, BaTiO3, Li compounds,
  fluoride-containing samples).
- Dispose of sample powders as chemical waste — do NOT pour down the drain.

### 5.3 Electrical Safety
- The high-voltage generator (40 kV) is serviced only by Bruker-certified engineers.
- Report any unusual sounds (arcing), odors, or error codes to the lab supervisor immediately.

---

## 6. Related Documents
- LAB-PROC-XRD-002: Capillary and High-Pressure XRD
- LAB-PROC-XRD-003: Rietveld Refinement with TOPAS
- LAB-POLICY-DM-001: Research Data Management Policy
- NIST SRM 640f Certificate of Analysis

---

*This protocol is reviewed annually. Verify you are using the current version before use.*
*Questions: contact the Characterization Facility Manager (lab-xrd@example.edu)*
""".format(date=TODAY)


CRYSTAL_REFERENCE = """\
# Crystallographic Reference Data
**Document ID:** LAB-REF-CRYST-001
**Version:** 1.4
**Effective Date:** {date}
**Source:** Compiled from ICDD PDF-4+, ICSD, and primary literature

---

## 1. Purpose

This reference document provides crystallographic parameters for common materials science
compounds used in the group's research. It serves as the primary reference for:
- Phase identification during XRD analysis
- Lattice parameter validation in Rietveld refinement
- Setting up theoretical diffraction patterns (VESTA, TOPAS, FullProf)

Values are for room temperature (25°C) unless noted. Estimated standard deviations
in parentheses apply to last digit(s).

---

## 2. X-Ray Wavelengths (Kα₁)

| Anode | Kα₁ (Å)    | Kα₂ (Å)    | Kβ (Å)    | Common Use              |
|-------|------------|------------|------------|------------------------|
| Cu    | 1.540593   | 1.544427   | 1.392250   | Most labs; avoid Fe, Co, Ni samples |
| Mo    | 0.709319   | 0.713609   | 0.632305   | High-energy; penetrates bulk |
| Co    | 1.788965   | 1.792900   | 1.620840   | Fe-rich samples (reduces fluorescence) |
| Ag    | 0.559421   | 0.563813   | 0.497082   | High-pressure studies |
| Cr    | 2.289760   | 2.293663   | 2.084920   | Low-Z materials, polymer crystallography |

Weighted average Kα (used for d-spacing calculations):
- Cu Kα: **1.54178 Å**  (= 2/3 × Kα₁ + 1/3 × Kα₂)
- Mo Kα: **0.71073 Å**
- Co Kα: **1.79021 Å**

---

## 3. d-Spacing Formulae by Crystal System

**Bragg's Law:**  2d sinθ = nλ  →  d = λ / (2 sinθ)

| Crystal System | d-spacing Formula |
|----------------|-------------------|
| Cubic | 1/d² = (h²+k²+l²) / a² |
| Tetragonal | 1/d² = (h²+k²)/a² + l²/c² |
| Hexagonal | 1/d² = 4/3 · (h²+hk+k²)/a² + l²/c² |
| Orthorhombic | 1/d² = h²/a² + k²/b² + l²/c² |
| Rhombohedral | 1/d² = [(h²+k²+l²)sin²α + 2(hk+kl+hl)(cos²α−cosα)] / [a²(1−3cos²α+2cos³α)] |
| Monoclinic | 1/d² = (1/sin²β)[h²/a² + k²sin²β/b² + l²/c² − 2hl·cosβ/(ac)] |
| Triclinic | (Full tensor form — see VESTA documentation) |

---

## 4. Crystal Structure Reference Table

### 4.1 Perovskite-Structure Compounds (ABO₃)

| Material | Crystal System | Space Group | a (Å) | b (Å) | c (Å) | β (°) | Density (g/cm³) | ICSD # |
|----------|---------------|-------------|-------|-------|-------|-------|-----------------|--------|
| BaTiO₃ (cubic, >120°C) | Cubic | Pm-3m (221) | 4.0118 | = a | = a | 90 | 5.85 | 67520 |
| BaTiO₃ (tetragonal, RT) | Tetragonal | P4mm (99) | 3.9945 | = a | 4.0335 | 90 | 6.02 | 67518 |
| SrTiO₃ | Cubic | Pm-3m (221) | 3.9050 | = a | = a | 90 | 5.12 | 23076 |
| PbTiO₃ (tetragonal) | Tetragonal | P4mm (99) | 3.9040 | = a | 4.1522 | 90 | 7.96 | 24606 |
| CaTiO₃ (orthorhombic) | Orthorhombic | Pbnm (62) | 5.3796 | 5.4423 | 7.6401 | 90 | 4.04 | 31094 |
| LaFeO₃ | Orthorhombic | Pbnm (62) | 5.5560 | 5.5650 | 7.8560 | 90 | 6.65 | 28274 |

**Perovskite tolerance factor:**  t = (r_A + r_O) / [√2 (r_B + r_O)]
Stable perovskite: 0.78 < t < 1.05

### 4.2 Battery Electrode Materials

| Material | Crystal System | Space Group | a (Å) | b (Å) | c (Å) | Density (g/cm³) | Role |
|----------|---------------|-------------|-------|-------|-------|-----------------|------|
| LiCoO₂ | Hexagonal (layered) | R-3m (166) | 2.8160 | = a | 14.054 | 5.06 | Cathode |
| LiFePO₄ | Orthorhombic | Pnma (62) | 10.338 | 6.008 | 4.693 | 3.60 | Cathode (LFP) |
| LiMn₂O₄ | Cubic (spinel) | Fd-3m (227) | 8.2420 | = a | = a | 4.28 | Cathode |
| Li₄Ti₅O₁₂ | Cubic (spinel) | Fd-3m (227) | 8.3590 | = a | = a | 3.49 | Anode |
| Graphite (anode) | Hexagonal | P6₃/mmc (194) | 2.4612 | = a | 6.7079 | 2.27 | Anode |
| Li metal | Cubic (BCC) | Im-3m (229) | 3.5093 | = a | = a | 0.534 | Anode (Li metal) |

### 4.3 Binary Oxides and Common Reference Materials

| Material | Crystal System | Space Group | a (Å) | b (Å) | c (Å) | Density (g/cm³) | Notes |
|----------|---------------|-------------|-------|-------|-------|-----------------|-------|
| TiO₂ rutile | Tetragonal | P4₂/mnm (136) | 4.5937 | = a | 2.9587 | 4.25 | More stable polymorph |
| TiO₂ anatase | Tetragonal | I4₁/amd (141) | 3.7845 | = a | 9.5143 | 3.89 | Photocatalyst form |
| α-Al₂O₃ (corundum) | Rhombohedral | R-3c (167) | 4.7587 | = a | 12.991 | 3.99 | Refractory, substrate |
| ZrO₂ (monoclinic) | Monoclinic | P2₁/c (14) | 5.1454 | 5.2075 | 5.3107 | 5.68 | RT stable |
| ZrO₂ (tetragonal) | Tetragonal | P4₂/nmc (137) | 3.5900 | = a | 5.1700 | 6.10 | >1170°C; stabilized by Y₂O₃ |
| MgO | Cubic (rocksalt) | Fm-3m (225) | 4.2117 | = a | = a | 3.58 | Substrate, refractory |
| Si | Cubic (diamond) | Fd-3m (227) | 5.4310 | = a | = a | 2.33 | NIST SRM 640f standard |
| Fe (BCC, α) | Cubic | Im-3m (229) | 2.8665 | = a | = a | 7.87 | RT stable |
| Fe (FCC, γ) | Cubic | Fm-3m (225) | 3.5830 | = a | = a | 7.65 | >912°C |
| NiO | Cubic (rocksalt) | Fm-3m (225) | 4.1769 | = a | = a | 6.67 | SOFC cathode |
| CeO₂ | Cubic (fluorite) | Fm-3m (225) | 5.4110 | = a | = a | 7.22 | TWC catalyst support |

### 4.4 High-Entropy Ceramics (Representative)

| Composition | Crystal System | Space Group | a (Å) | Density est. | Reference |
|-------------|---------------|-------------|-------|-------------|-----------|
| (Hf₀.₂Zr₀.₂Ti₀.₂Nb₀.₂Ta₀.₂)C | Cubic (rocksalt) | Fm-3m | ~4.595 | ~10.5 | Harrington et al. 2019 |
| (Mg₀.₂Co₀.₂Ni₀.₂Cu₀.₂Zn₀.₂)O | Cubic (rocksalt) | Fm-3m | ~4.242 | ~6.2 | Rost et al. 2015 |
| (Ti₀.₂Zr₀.₂Hf₀.₂Nb₀.₂Ta₀.₂)N | Cubic (rocksalt) | Fm-3m | ~4.448 | ~9.8 | Braic et al. 2012 |

---

## 5. Common XRD Reference Standards

| Standard | Material | a (Å) | Purpose | NIST SRM # |
|----------|----------|-------|---------|-----------|
| Si powder | Silicon | 5.43123 | Instrument calibration | 640f |
| LaB₆ | Lanthanum hexaboride | 4.15695 | Line profile standard | 660c |
| Al₂O₃ | Corundum | a=4.7587, c=12.991 | d-spacing calibration | 676b |
| CeO₂ | Ceria | 5.41153 | Combined calibration | 674b |
| TiO₂ (rutile) | Rutile | a=4.5933, c=2.9592 | Crystallite size | 674c |

---

## 6. Systematic Absences Summary

| Bravais Lattice | Absent Reflections |
|----------------|-------------------|
| P (primitive) | None |
| I (body-centered) | h+k+l = odd |
| F (face-centered) | h,k,l not all same parity |
| A, B, C (base-centered) | Depends on centered face |
| R (rhombohedral) | -h+k+l ≠ 3n |

Common glide/screw systematic absences affect specific hkl families — consult
the International Tables for Crystallography, Volume A.

---

## 7. Scherrer Equation (Crystallite Size)

τ = Kλ / (β cosθ)

- τ: crystallite size (Å or nm)
- K: shape factor (typically 0.94 for spherical, 0.89 for cubic)
- λ: X-ray wavelength (Å)
- β: FWHM of the diffraction peak (radians) after instrumental broadening correction
- θ: Bragg angle

**Note:** Scherrer equation gives a lower bound on crystallite size. Microstrain and
stacking faults also broaden peaks. Use Williamson-Hall or Warren-Averbach analysis
for strain-broadening separation.

---

*Verify critical values against primary literature or ICDD PDF cards before publication.*
*ICSD accession numbers provided for database lookup.*
""".format(date=TODAY)


SECURITY_CHECKLIST = """\
# Laboratory Cybersecurity Checklist
**Document ID:** LAB-SEC-001
**Version:** 1.1
**Effective Date:** {date}
**Scope:** All research lab personnel with access to laboratory computing systems
**Review:** Semi-annual | Owner: Lab Safety & Compliance Officer

> **Purpose:** Practical, actionable security checklist. This is NOT a policy document —
> see INST-POLICY-SEC-2024 for full institutional policy. Use this checklist for
> onboarding new lab members and semi-annual compliance reviews.

---

## Section 1: Data Classification

Every file, dataset, or document generated in the lab falls into one of four levels.
**When in doubt, classify higher.**

| Level | Label | Examples | Who Can Access |
|-------|-------|---------|---------------|
| **1 — Public** | PUBLIC | Published papers, public protocols | Anyone |
| **2 — Internal** | INTERNAL | Draft manuscripts, group meeting slides, routine XRD data | Lab members only |
| **3 — Controlled** | CONTROLLED | Unpublished results with IP potential, export-controlled material data, CUI (if applicable), collaborator data under NDA | Named personnel only |
| **4 — Restricted** | RESTRICTED | Export-controlled technical data (EAR/ITAR), federal contract deliverables with data rights clauses | PI + explicitly authorized personnel + institutional approval |

**Required label:** Include the classification level in the filename or document header for
Level 2+. Example: `[CONTROLLED] BaTiO3_synthesis_protocol_v3.docx`

**Handling rules:**
- [ ] Never email Level 3/4 data without encryption or institutional secure file transfer
- [ ] Never store Level 3/4 data on personal cloud accounts (Google Drive, Dropbox personal)
- [ ] Controlled data on portable media requires encryption (BitLocker, VeraCrypt)

---

## Section 2: Access Control

### 2.1 User Accounts
- [ ] Each person has their own account on shared lab computers — **no shared logins**
- [ ] Default passwords changed within 24 hours of account creation
- [ ] Passwords: minimum 14 characters, include uppercase, lowercase, digit, symbol
- [ ] No passwords written on sticky notes or stored in plaintext files
- [ ] MFA enabled on all institutional accounts (email, VPN, research computing)

### 2.2 When People Leave
- [ ] Departing members: PI notifies IT and lab manager **on last day**
- [ ] Account disabled same day; no grace period for Level 3/4 systems
- [ ] All lab data copied to group storage before departure — personal laptop copies deleted
- [ ] Shared passwords (instrument accounts) rotated within 48 hours

### 2.3 Guest and Visitor Access
- [ ] Visitors use guest WiFi — never share lab WiFi credentials
- [ ] External collaborators get time-limited accounts; document expiry date
- [ ] No shoulder-surfing access to Level 3/4 screens without signed NDA on file

---

## Section 3: Instrument Computer Security

Many instruments (XRD, SEM, TEM, profilometers) run on embedded or legacy OS systems
connected to the lab network. These require special attention.

### 3.1 Baseline Requirements
- [ ] Instrument computers on isolated VLAN or instrument subnet (contact IT)
- [ ] No browsing, email, or USB drives on instrument computers (data-only use)
- [ ] Auto-login (if required by vendor) documented and restricted to instrument room
- [ ] Windows instruments: local firewall enabled; unnecessary services disabled
- [ ] Instrument OS patching: coordinate with vendor for approved updates — document deviations

### 3.2 Data Transfer from Instruments
- [ ] Transfer data via network share to lab server — **not** via personal USB drives
- [ ] If USB transfer is unavoidable: scan USB on lab-designated antivirus station before use
- [ ] Instrument computers: no internet access unless required for license validation
- [ ] Log who collects data and when (instrument logbook + digital log)

### 3.3 Legacy Systems (Windows XP/7)
- [ ] Legacy systems isolated from lab network if no vendor-supported security updates available
- [ ] Physical access restricted (locked instrument room)
- [ ] No sensitive data stored locally — transfer immediately after collection
- [ ] Document legacy system inventory and exception in Lab Security Assessment (annual)

---

## Section 4: Data Backup and Integrity Verification

**Rule of thumb: 3-2-1 backup** — 3 copies, 2 different media, 1 offsite.

### 4.1 Backup Schedule
| Data Type | Primary Storage | Backup Frequency | Offsite Backup |
|-----------|----------------|-----------------|----------------|
| Raw instrument data | Lab server | Daily automated | Institutional HPC/cloud (weekly) |
| Analysis results | Lab server | Daily automated | Same |
| Code/scripts | Institutional GitLab | On commit | GitLab remote (continuous) |
| Lab notebooks (electronic) | ELN system | Real-time sync | ELN vendor cloud |

- [ ] Verify automated backups are running monthly (check backup logs)
- [ ] Perform restore test quarterly — confirm you can actually recover files

### 4.2 Data Integrity
- [ ] Raw data files: read-only permissions after transfer from instrument
- [ ] Critical datasets: generate SHA-256 checksums and store separately
  ```
  # Linux/Mac:
  sha256sum rawdata.xy > rawdata.xy.sha256
  # Windows PowerShell:
  Get-FileHash rawdata.xy -Algorithm SHA256 | Out-File rawdata.xy.sha256
  ```
- [ ] Never overwrite raw data — keep original files; create processed copies
- [ ] Version control analysis scripts (Git); tag versions used for publications

---

## Section 5: Incident Response

**If you suspect a security incident (data breach, ransomware, unauthorized access, stolen device):**

### Immediate Actions (within 1 hour)
1. **Do not panic — do not try to fix it yourself**
2. Disconnect the affected device from the network (unplug Ethernet or disable WiFi)
3. Do NOT power off — preserves forensic evidence (unless instructed by IT)
4. Note the time and what you observed (screenshot if possible)
5. Contact: **IT Security Hotline: [extension on lab notice board]**
6. Notify the PI immediately

### Within 24 Hours
- [ ] File an incident report with IT Security
- [ ] Identify which data may have been affected and its classification level
- [ ] If Level 3/4 data involved: PI must notify Research Compliance Office
- [ ] If federally funded data: check sponsor agreement for breach notification requirements

### Suspected Data Tampering (XRD data, instrument logs)
- [ ] Compare files against backup copies using stored checksums
- [ ] Preserve original tampered files — do NOT delete
- [ ] Document discrepancies in writing and report to PI and Research Integrity Officer
- [ ] Do NOT publish or use data that cannot be verified as unaltered

---

## Section 6: Removable Media Policy

- [ ] Personal USB drives: **prohibited** on lab computers and instruments
- [ ] Lab-issued encrypted USB drives only (inventory in lab manager's office)
- [ ] All USB drives logged out/in; report lost drives immediately
- [ ] CD/DVD: for read-only reference material only; no writeable optical media
- [ ] Phones and tablets: camera use at instrument stations requires prior approval

---

## Section 7: Software and Licensing

- [ ] Only licensed or open-source software on lab computers
- [ ] Software download: institutional repositories or vendor sites only (no warez/torrent)
- [ ] Record software versions used in any publication (reproducibility + audit)
- [ ] License keys stored in password manager or lab manager's encrypted records
- [ ] Unused software uninstalled — reduces attack surface

---

## Section 8: Physical Security

- [ ] Lab door locked when unoccupied (even briefly)
- [ ] Visitor log maintained for Level 3/4 lab areas
- [ ] Screens locked when stepping away (Windows: Win+L; Mac: Ctrl+Cmd+Q)
- [ ] Tailgating policy: do not hold the door for badge-less visitors — direct them to front desk
- [ ] Report propped doors or bypassed badge readers to building security

---

## Onboarding Checklist (new lab members)

Complete within first week:
- [ ] Receive institutional computing account + MFA setup
- [ ] Review this checklist with lab manager — sign acknowledgment form
- [ ] Complete institutional cybersecurity training module (link in orientation packet)
- [ ] Receive lab network credentials and encrypted USB if needed
- [ ] Listed in access control register for any Level 3/4 data areas

---

*This document is reviewed semi-annually. Report security concerns to: lab-security@example.edu*
*For emergencies: IT Security: [institutional hotline] | PI cell: [see lab contact sheet]*
""".format(date=TODAY)


DOCUMENTS = [
    {
        "filename": "XRD_Sample_Preparation_Protocol.md",
        "content": XRD_PROTOCOL,
        "domain": "materials_science",
        "subdomain": "laboratory_protocols",
        "doc_type": "lab_protocol",
        "description": (
            "XRD Sample Preparation Protocol — step-by-step procedure for preparing "
            "powder and bulk samples for X-ray diffraction analysis. Covers grinding, "
            "mounting, diffractometer alignment, data collection, and safety."
        ),
    },
    {
        "filename": "Crystallographic_Reference_Data.md",
        "content": CRYSTAL_REFERENCE,
        "domain": "materials_science",
        "subdomain": "reference_data",
        "doc_type": "reference_table",
        "description": (
            "Crystallographic Reference Data — lattice parameters, space groups, and "
            "densities for common materials science compounds including perovskites, "
            "battery materials, and binary oxides. Includes XRD wavelengths and "
            "d-spacing formulae."
        ),
    },
    {
        "filename": "Lab_Cybersecurity_Checklist.md",
        "content": SECURITY_CHECKLIST,
        "domain": "cybersecurity",
        "subdomain": "lab_security",
        "doc_type": "security_checklist",
        "description": (
            "Laboratory Cybersecurity Checklist — practical security procedures for "
            "research labs covering data classification, access control, instrument "
            "computer security, backup/integrity verification, incident response, "
            "and removable media policies."
        ),
    },
]


def collect(output_dir: Path, verbose: bool = False) -> list[dict[str, Any]]:
    """Write all custom documents to *output_dir* and return manifest entries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []

    for doc in DOCUMENTS:
        out_path = output_dir / doc["filename"]

        if out_path.exists() and out_path.stat().st_size > 256:
            logger.info("  [skip] %s already exists", doc["filename"])
            if verbose:
                print(f"    [skip] {doc['filename']}")
        else:
            out_path.write_text(doc["content"], encoding="utf-8")
            logger.info(
                "  ✓ Wrote %s  (%.1f KB)", doc["filename"], len(doc["content"]) / 1024
            )
            if verbose:
                print(
                    f"    ✓ {doc['filename']}  ({len(doc['content'])//1024:.0f} KB, custom)"
                )

        meta = get_file_metadata(out_path)
        manifest_entries.append(
            {
                "filename": doc["filename"],
                "source_url": "generated",
                "domain": doc["domain"],
                "subdomain": doc["subdomain"],
                "doc_type": doc["doc_type"],
                "pages": meta["pages"],
                "download_date": TODAY,
                "size_kb": meta["size_kb"],
                "description": doc["description"],
            }
        )

    logger.info(
        "Custom docs collector finished: %d documents written", len(manifest_entries)
    )
    return manifest_entries
