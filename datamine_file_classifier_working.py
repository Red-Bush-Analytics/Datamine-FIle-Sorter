from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Rule model
# ---------------------------------------------------------------------

@dataclass
class Rule:
    type_name: str
    folder_name: str
    description: str
    anchors: Set[str] = field(default_factory=set)         # distinctive fields that should be present
    supporting: Set[str] = field(default_factory=set)      # additional expected fields
    min_anchor_matches: int = 0
    min_total_matches: int = 0
    forbidden_any: Set[str] = field(default_factory=set)
    confidence_base: float = 0.50
    notes: str = ""


@dataclass(frozen=True)
class ExtensionRule:
    type_name: str
    folder_name: str
    description: str
    suffixes: Set[str]
    confidence: float = 0.99
    notes: str = ""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

_TOKEN_SPLIT_RE = re.compile(r"[\s,;/]+")


def is_blank_cell(value: object) -> bool:
    """Return True for empty CSV cells without using pd.isna on object values.

    This avoids Pylance overload/type warnings while still handling normal
    pandas CSV blanks such as None, NaN, and empty strings.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True

    text = str(value).strip()
    return text == "" or text.upper() in {"NAN", "NA", "<NA>", "NONE"}


def first_nonblank_text(*values: object, default: str = "") -> str:
    """Return the first non-blank value as stripped text.

    Used to keep the final sorter-facing folder populated while preserving
    earlier summary-stage fields such as summary_suggested_folder.
    """
    for value in values:
        if not is_blank_cell(value):
            return str(value).strip()
    return default


def parse_columns(col_text: object) -> Set[str]:
    if is_blank_cell(col_text):
        return set()
    return {c.strip().upper() for c in str(col_text).split(",") if c.strip()}


def normalize_field_tokens(items: Iterable[str]) -> Set[str]:
    out = set()
    for item in items:
        text = str(item).strip().upper()
        if not text:
            continue
        # remove braces used in Datamine docs for parameterised field names
        text = text.replace("{", "").replace("}", "")
        # split grouped labels such as "XP YP ZP", "PID1 PID2 PID3", "VDIP, VAZI"
        parts = [p.strip() for p in _TOKEN_SPLIT_RE.split(text) if p.strip()]
        out.update(parts)
    return out


def get_suffix(file_value: object) -> str:
    """Return a lower-case file extension from a filename or path-like value."""
    if is_blank_cell(file_value):
        return ""
    text = str(file_value).strip().strip('"').strip("'")
    return Path(text).suffix.lower()


def base_classification(
    detected_type: str,
    suggested_folder: str,
    confidence: float,
    rule_notes: str,
    decision_basis: str = "",
    matched_anchors: str = "",
    missing_anchors: str = "",
    matched_supporting: str = "",
    forbidden_found: str = "",
) -> Dict[str, object]:
    return {
        "detected_type": detected_type,
        "suggested_folder": suggested_folder,
        "confidence": round(confidence, 3),
        "rule_notes": rule_notes,
        "matched_anchors": matched_anchors,
        "missing_anchors": missing_anchors,
        "matched_supporting": matched_supporting,
        "forbidden_found": forbidden_found,
        "decision_basis": decision_basis,
    }


def score_rule(columns: Set[str], rule: Rule) -> Tuple[bool, float, Dict[str, object]]:
    matched_anchors = sorted(rule.anchors.intersection(columns))
    matched_supporting = sorted(rule.supporting.intersection(columns))
    forbidden_found = sorted(rule.forbidden_any.intersection(columns))

    anchor_matches = len(matched_anchors)
    total_matches = anchor_matches + len(matched_supporting)

    is_match = (
        anchor_matches >= rule.min_anchor_matches
        and total_matches >= rule.min_total_matches
        and len(forbidden_found) == 0
    )

    confidence = rule.confidence_base
    if rule.anchors:
        confidence += min(0.20, 0.06 * anchor_matches)
    if rule.supporting:
        confidence += min(0.25, 0.03 * len(matched_supporting))
    if len(forbidden_found) > 0:
        confidence -= 0.15
    confidence = max(0.0, min(confidence, 0.99))

    details = {
        "matched_anchors": matched_anchors,
        "missing_anchors": sorted(rule.anchors.difference(columns)),
        "matched_supporting": matched_supporting,
        "forbidden_found": forbidden_found,
        "anchor_matches": anchor_matches,
        "total_matches": total_matches,
    }
    return is_match, confidence, details


# ---------------------------------------------------------------------
# Datamine summary-based rules
#
# These rules operate ONLY on the v3 batch-summary output.
# The summary contains logical explicit fields. Implicit/system fields
# may not be present, so some classifications remain heuristic.
# ---------------------------------------------------------------------

EXTENSION_RULES: List[ExtensionRule] = [
    ExtensionRule(
        type_name="Datamine_Macro_File",
        folder_name="34_Macros",
        description="Datamine macro file",
        suffixes={".mac"},
        confidence=1.00,
        notes="Extension-based rule. Datamine macro files are identified by the .mac extension.",
    ),
    ExtensionRule(
        type_name="Datamine_HTML_Script_File",
        folder_name="35_Scripts",
        description="Datamine recorded HTML script file",
        suffixes={".htm", ".html"},
        confidence=1.00,
        notes="Extension-based rule. Datamine recorded scripts are commonly stored as .htm files; .html is included as a safe alias.",
    ),
    ExtensionRule(
        type_name="Image_or_PDF_File",
        folder_name="36_Images",
        description="Image or PDF reference file",
        suffixes={".jpg", ".jpeg", ".png", ".bmp", ".pdf"},
        confidence=1.00,
        notes="Extension-based rule for image/reference folders requested for jpg, png, bmp and pdf. .jpeg is included as the common jpg alias.",
    ),
]


RULES: List[Rule] = [
    Rule(
        type_name="TONGRAD_TIV_File",
        folder_name="31_TIV",
        description="TONGRAD cumulative TIV output file",
        anchors=normalize_field_tokens(["COMP_KEY", "GRPH_KEY", "X1_AXIS", "Y1_LINE", "Y2_LINE"]),
        supporting=normalize_field_tokens(["CUTOFF", "CMGT", "CW", "VOLUME", "TONNES", "DENSITY", "CAT", "ZONET", "DESCRIP"]),
        forbidden_any=normalize_field_tokens(["COGSTEP"]),
        min_anchor_matches=3,
        min_total_matches=7,
        confidence_base=0.84,
        notes="Signature added from the supplied TIV example. TIV files commonly include graph/comparison keys and axis/line fields from TONGRAD cumulative outputs.",
    ),
    Rule(
        type_name="TONGRAD_File",
        folder_name="32_TONGRAD",
        description="TONGRAD tabular output file",
        anchors=normalize_field_tokens(["COGSTEP", "VOLUME", "TONNES", "DENSITY"]),
        supporting=normalize_field_tokens(["CMGT", "CW", "CAT", "ZONET", "DESCRIP", "MODEL", "MODEL_"]),
        forbidden_any=normalize_field_tokens(["COMP_KEY", "GRPH_KEY", "X1_AXIS", "Y1_LINE", "Y2_LINE"]),
        min_anchor_matches=3,
        min_total_matches=7,
        confidence_base=0.83,
        notes="Signature added from the supplied TONGRAD example. The COGSTEP field separates normal TONGRAD outputs from TIV outputs.",
    ),
    Rule(
        type_name="Estimation_Samples_Output_File",
        folder_name="33_Estimation_Samples_Output",
        description="Estimation samples output file",
        anchors=normalize_field_tokens(["X", "Y", "Z", "XC", "YC", "ZC", "ACTDIST", "TRANDIST"]),
        supporting=normalize_field_tokens(["ZONE", "FIELD", "GRADE", "WEIGHT", "AV-VGRAM", "CMGT", "CW", "AU", "AUGT"]),
        min_anchor_matches=6,
        min_total_matches=9,
        confidence_base=0.82,
        notes="Signature added from the supplied estimation samples output example. It uses both sample coordinates and block-centre coordinates plus distance/weight fields.",
    ),
    Rule(
        type_name="Attribute_Validation_File",
        folder_name="01_Attribute_Validation",
        description="Attribute validation file",
        anchors=normalize_field_tokens(["ATTTYPE", "ATTNAME"]),
        supporting=normalize_field_tokens(["VALUE", "MIN", "MAX", "DEFAULT"]),
        min_anchor_matches=2,
        min_total_matches=4,
        confidence_base=0.85,
        notes="Strong signature from Datamine validation fields.",
    ),
    Rule(
        type_name="Blast_Patterns_File",
        folder_name="02_Blast_Patterns",
        description="Blast patterns file",
        anchors=normalize_field_tokens(["PATTERN", "DESC", "SPACING", "BURDEN", "ROFFSET", "ROW"]),
        supporting=normalize_field_tokens(["DIP", "RNFIRST", "RNINC", "RNREPEAT", "HNFIRST", "HNINC", "SNFIRST", "SNINC", "HLENGTH", "SAMPLENG", "STEMMING"]),
        min_anchor_matches=4,
        min_total_matches=6,
        confidence_base=0.82,
        notes="Based on the Datamine blast-pattern field family.",
    ),
    Rule(
        type_name="Block_Model_File",
        folder_name="03_Block_Model",
        description="Block model file",
        anchors=normalize_field_tokens(["XC YC ZC", "IJK"]),
        supporting=normalize_field_tokens(["XMORIG YMORIG ZMORIG", "XINC YINC ZINC", "NX NY NZ", "BLKNUM", "ZONE", "DENSITY", "CMGT", "CW"]),
        min_anchor_matches=4,
        min_total_matches=5,
        confidence_base=0.80,
        notes="Summary-based heuristic. Some Datamine block-model fields may be implicit rather than explicit.",
    ),
    Rule(
        type_name="Dependency_File",
        folder_name="04_Dependency",
        description="Scheduling dependency file",
        anchors=normalize_field_tokens(["PNUM1", "PNUM2"]),
        min_anchor_matches=2,
        min_total_matches=2,
        confidence_base=0.98,
        notes="Distinctive Datamine dependency signature.",
    ),
    Rule(
        type_name="Desurveyed_Drillhole_File",
        folder_name="05_Desurveyed_Drillhole",
        description="Desurveyed drillhole file",
        anchors=normalize_field_tokens(["BHID", "FROM", "TO", "LENGTH"]),
        supporting=normalize_field_tokens(["X Y Z", "A0", "B0"]),
        min_anchor_matches=4,
        min_total_matches=7,
        confidence_base=0.88,
        notes="Strong drillhole signature with desurvey coordinates and orientation.",
    ),
    Rule(
        type_name="Downhole_Sample_File",
        folder_name="06_Downhole_Sample",
        description="Downhole sample file",
        anchors=normalize_field_tokens(["BHID", "FROM", "TO"]),
        forbidden_any=normalize_field_tokens(["X", "Y", "Z", "A0", "B0", "AT", "BRG", "DIP"]),
        min_anchor_matches=3,
        min_total_matches=3,
        confidence_base=0.62,
        notes="Conservative fallback for drillhole interval data when stronger drillhole signatures are absent.",
    ),
    Rule(
        type_name="Downhole_Survey_File",
        folder_name="07_Downhole_Survey",
        description="Downhole survey file",
        anchors=normalize_field_tokens(["BHID", "AT", "BRG", "DIP"]),
        min_anchor_matches=4,
        min_total_matches=4,
        confidence_base=0.92,
        notes="Strong Datamine survey signature.",
    ),
    Rule(
        type_name="Drillhole_Collars_File",
        folder_name="08_Drillhole_Collars",
        description="Drillhole collars file",
        anchors=normalize_field_tokens(["BHID", "XCOLLAR", "YCOLLAR", "ZCOLLAR"]),
        min_anchor_matches=4,
        min_total_matches=4,
        confidence_base=0.94,
        notes="Strong Datamine collars signature.",
    ),
    Rule(
        type_name="Ellipsoid_Nonwireframe_File",
        folder_name="09_Ellipsoid_Nonwireframe",
        description="Ellipsoid (non-wireframe) file",
        anchors=normalize_field_tokens(["AZI", "DIP", "ROLL", "RAD1", "RAD2", "RAD3"]),
        min_anchor_matches=6,
        min_total_matches=6,
        confidence_base=0.92,
        notes="Distinctive ellipsoid geometry signature.",
    ),
    Rule(
        type_name="Estimation_Parameters_File",
        folder_name="10_Estimation_Parameters",
        description="Estimation parameters file",
        anchors=normalize_field_tokens(["VALUE_IN", "VALUE_OU", "SREFNUM", "IMETHOD", "ANISO"]),
        supporting=normalize_field_tokens([
            "ZONE1_F", "ZONE2_F", "NUMSAM_F", "SVOL_F", "VAR_F", "MINDIS_F",
            "ANANGLE1", "ANANGLE2", "ANANGLE3",
            "ANDIST1", "ANDIST2", "ANDIST3",
            "POWER", "ADDCON", "VREFNUM", "LOG", "GENCASE",
            "DEPMEAN", "TOL", "MAXITER", "KRIGNEGW", "KRIGVARS",
            "LOCALMNP", "LOCALM_F"
        ]),
        min_anchor_matches=4,
        min_total_matches=9,
        confidence_base=0.78,
        notes="Parameter-table heuristic based on the Datamine estimation parameter fields.",
    ),
    Rule(
        type_name="Histogram_File",
        folder_name="11_Histogram",
        description="Histogram output file",
        anchors=normalize_field_tokens(["LOWER", "MIDDLE", "UPPER", "FREQENCY", "CUMFREQ"]),
        supporting=normalize_field_tokens(["AVIVAL", "FREQ-%", "CUMF-%", "TOTVA"]),
        min_anchor_matches=4,
        min_total_matches=6,
        confidence_base=0.84,
        notes="Histogram output signature. Keeps Datamine's documented field spelling FREQENCY.",
    ),
    Rule(
        type_name="Pictures_File",
        folder_name="12_Pictures",
        description="Pictures file",
        anchors=normalize_field_tokens(["XP", "YP", "ZP", "IMAGE"]),
        supporting=normalize_field_tokens(["HSIZE", "VSIZE", "DIPDIRN", "DIP", "ROLL"]),
        min_anchor_matches=4,
        min_total_matches=6,
        confidence_base=0.84,
        notes="Picture placement signature.",
    ),
    Rule(
        type_name="Planes_File",
        folder_name="13_Planes",
        description="Planes file",
        anchors=normalize_field_tokens(["XP", "YP", "ZP", "SDIP", "DIPDIRN"]),
        supporting=normalize_field_tokens(["HSIZE", "VSIZE", "SYMBOL", "COLOUR", "VARIANCE", "BLOCKID"]),
        min_anchor_matches=5,
        min_total_matches=7,
        confidence_base=0.83,
        notes="Plane geometry signature.",
    ),
    Rule(
        type_name="Plot_File",
        folder_name="14_Plot",
        description="Plot file",
        anchors=normalize_field_tokens(["X Y", "CODE", "CHARSIZE"]),
        supporting=normalize_field_tokens(["S1 S2", "XMIN XMAX YMIN YMAX", "XSCALE YSCALE", "XORIG YORIG", "ASPRATIO"]),
        min_anchor_matches=3,
        min_total_matches=6,
        confidence_base=0.72,
        notes="Plot-file heuristic. Documentation groups several fields in combined rows.",
    ),
    Rule(
        type_name="Plotter_Filter_File",
        folder_name="15_Plotter_Filter",
        description="Plotter filter file",
        anchors=normalize_field_tokens(["TEST", "IN", "OUT"]),
        min_anchor_matches=3,
        min_total_matches=3,
        confidence_base=0.90,
        notes="Small but distinctive signature.",
    ),
    Rule(
        type_name="Plotter_Pen_File",
        folder_name="16_Plotter_Pen",
        description="Plotter pen file",
        anchors=normalize_field_tokens(["COLOR", "PEN"]),
        min_anchor_matches=2,
        min_total_matches=2,
        confidence_base=0.92,
        notes="Small but distinctive signature.",
    ),
    Rule(
        type_name="Point_Data_File",
        folder_name="17_Point_Data",
        description="Point data file",
        anchors=normalize_field_tokens(["XPT", "YPT", "ZPT"]),
        supporting=normalize_field_tokens(["COLOUR", "SYMBOL", "CW", "CMGT", "SW"]),
        min_anchor_matches=3,
        min_total_matches=3,
        confidence_base=0.90,
        notes="Distinctive point-data coordinate signature.",
    ),
    Rule(
        type_name="Results_File",
        folder_name="18_Results",
        description="Results file",
        anchors=normalize_field_tokens(["MODEL", "BLOCKID", "DENSITY", "VOLUME", "TONNES"]),
        min_anchor_matches=5,
        min_total_matches=5,
        confidence_base=0.92,
        notes="Distinctive results-file signature.",
    ),
    Rule(
        type_name="Rosettes_File",
        folder_name="19_Rosettes",
        description="Rosettes file",
        anchors=normalize_field_tokens(["ROSNUM", "ROSXPOS", "ROSYPOS", "ROSZMIN", "ROSZMAX"]),
        supporting=normalize_field_tokens(["ROSAZIM", "ROSFANG", "ROSBWID"]),
        min_anchor_matches=5,
        min_total_matches=7,
        confidence_base=0.86,
        notes="Distinctive rosette geometry signature.",
    ),
    Rule(
        type_name="Sample_Pairs_File",
        folder_name="20_Sample_Pairs",
        description="Sample pairs file",
        anchors=normalize_field_tokens(["DISTANCE", "GRADE1", "GRADE2"]),
        supporting=normalize_field_tokens(["VLAGDIST", "VDIP VAZI", "WVDIP WVAZI", "DIP AZI", "WDIP WAZI", "X1 Y1 Z1", "X2 Y2 Z2", "VALUE11", "VALUE12", "VALUE21", "VALUE22"]),
        min_anchor_matches=3,
        min_total_matches=7,
        confidence_base=0.78,
        notes="Heuristic signature based on Datamine sample-pair outputs.",
    ),
    Rule(
        type_name="Schedule_File",
        folder_name="21_Schedule",
        description="Schedule file",
        anchors=normalize_field_tokens(["BLOCKID", "SLOT", "PERCENT", "DRAW"]),
        supporting=normalize_field_tokens(["VOLUME", "TONNES", "DENSITY", "START", "END", "LENGTH", "AREA"]),
        min_anchor_matches=4,
        min_total_matches=7,
        confidence_base=0.84,
        notes="Distinctive schedule signature.",
    ),
    Rule(
        type_name="Search_Volume_Parameters_Advanced_File",
        folder_name="22_Search_Volume_Advanced",
        description="Search volume parameters file (advanced estimation)",
        anchors=normalize_field_tokens(["SREFNUM", "SMETHOD", "SDIST1", "SDIST2", "SDIST3", "SANGLE1", "SANGLE2", "SANGLE3"]),
        supporting=normalize_field_tokens(["SAXIS1", "SAXIS2", "SAXIS3", "OPTKEY", "NSECTORS", "SPLITSEC", "MAXEMPSC", "MVSEARCH", "MINNUM1", "MAXNUM1", "SVOLFAC2", "MINNUM2", "MAXNUM2", "SVOLFAC3", "MINNUM3", "MAXNUM3", "MAXKEY"]),
        min_anchor_matches=6,
        min_total_matches=12,
        confidence_base=0.80,
        notes="Advanced-estimation search-volume signature.",
    ),
    Rule(
        type_name="Search_Volume_Parameters_ESTIMA_File",
        folder_name="23_Search_Volume_ESTIMA",
        description="Search volume parameters file (used by ESTIMA)",
        anchors=normalize_field_tokens(["SREFNUM", "SMETHOD", "SDIST1", "SDIST2", "SDIST3", "SANGLE1", "SANGLE2", "SANGLE3"]),
        supporting=normalize_field_tokens(["SAXIS1", "SAXIS2", "SAXIS3", "MINNUM1", "MAXNUM1", "SVOLFAC2", "MINNUM2", "MAXNUM2", "SVOLFAC3", "MINNUM3", "MAXNUM3", "OCTMETH", "MINOCT", "MINPEROC", "MAXPEROC", "MAXKEY"]),
        min_anchor_matches=6,
        min_total_matches=11,
        confidence_base=0.80,
        notes="ESTIMA search-volume signature.",
    ),
    Rule(
        type_name="Section_File",
        folder_name="24_Section",
        description="Section file",
        anchors=normalize_field_tokens(["XCENTRE", "YCENTRE", "ZCENTRE", "SDIP", "SAZI"]),
        supporting=normalize_field_tokens(["HSIZE", "VSIZE"]),
        min_anchor_matches=5,
        min_total_matches=7,
        confidence_base=0.90,
        notes="Distinctive section geometry signature.",
    ),
    Rule(
        type_name="STATS_Output_File",
        folder_name="25_STATS_Output",
        description="STATS output file",
        anchors=normalize_field_tokens(["NRECORDS", "NSAMPLES", "NMISVALS", "MINIMUM", "MAXIMUM", "RANGE", "TOTAL", "MEAN"]),
        min_anchor_matches=6,
        min_total_matches=8,
        confidence_base=0.88,
        notes="Distinctive STATS output signature.",
    ),
    Rule(
        type_name="String_File",
        folder_name="26_String",
        description="String file",
        anchors=normalize_field_tokens(["XP YP ZP", "PTN", "PVALUE"]),
        min_anchor_matches=5,
        min_total_matches=5,
        confidence_base=0.90,
        notes="Distinctive string-file signature.",
    ),
    Rule(
        type_name="Variogram_Experimental_File",
        folder_name="27_Variogram_Experimental",
        description="Variogram file - experimental",
        anchors=normalize_field_tokens(["GRADE", "AZI", "DIP", "LAG", "NO.PAIRS", "VGRAM"]),
        supporting=normalize_field_tokens(["AVE.DIST", "COVAR", "PWRVGRAM", "LOGVGRAM"]),
        min_anchor_matches=5,
        min_total_matches=7,
        confidence_base=0.84,
        notes="Experimental variogram signature.",
    ),
    Rule(
        type_name="Variogram_Model_File",
        folder_name="28_Variogram_Model",
        description="Variogram model file",
        anchors=normalize_field_tokens(["VREFNUM", "VANGLE1", "VANGLE2", "VANGLE3", "VAXIS1", "VAXIS2", "VAXIS3", "NUGGET"]),
        supporting=normalize_field_tokens(["ST1", "ST1PAR1", "ST1PAR2", "ST1PAR3", "ST1PAR4"]),
        min_anchor_matches=6,
        min_total_matches=9,
        confidence_base=0.86,
        notes="Variogram model parameter signature.",
    ),
    Rule(
        type_name="Wireframe_Points_File",
        folder_name="29_Wireframe_Points",
        description="Wireframe points file",
        anchors=normalize_field_tokens(["PID", "XP YP ZP"]),
        min_anchor_matches=4,
        min_total_matches=4,
        confidence_base=0.93,
        notes="Distinctive wireframe point signature.",
    ),
    Rule(
        type_name="Wireframe_Triangle_File",
        folder_name="30_Wireframe_Triangles",
        description="Wireframe triangle file",
        anchors=normalize_field_tokens(["TRIANGLE", "PID1 PID2 PID3"]),
        min_anchor_matches=4,
        min_total_matches=4,
        confidence_base=0.94,
        notes="Distinctive wireframe triangle signature.",
    ),
]


def classify_by_extension(file_value: object) -> Optional[Dict[str, object]]:
    suffix = get_suffix(file_value)
    if not suffix:
        return None

    for rule in EXTENSION_RULES:
        if suffix in rule.suffixes:
            return base_classification(
                detected_type=rule.type_name,
                suggested_folder=rule.folder_name,
                confidence=rule.confidence,
                rule_notes=rule.notes,
                decision_basis=f"extension={suffix}",
            )
    return None


def classify_columns(columns: Set[str]) -> Dict[str, object]:
    candidates = []
    for rule in RULES:
        matched, confidence, details = score_rule(columns, rule)
        if matched:
            candidates.append((rule, confidence, details))

    if not candidates:
        return base_classification(
            detected_type="Unknown",
            suggested_folder="99_Unknown",
            confidence=0.0,
            rule_notes="No rule met the minimum signature.",
        )

    def rank_key(item):
        rule, confidence, details = item
        richness = len(details["matched_anchors"]) + len(details["matched_supporting"])
        # prefer stronger matches when confidence ties
        return (confidence, details["anchor_matches"], details["total_matches"], richness)

    rule, confidence, details = sorted(candidates, key=rank_key, reverse=True)[0]

    basis_parts = []
    if details["matched_anchors"]:
        basis_parts.append(f"anchors={details['matched_anchors']}")
    if details["matched_supporting"]:
        basis_parts.append(f"supporting={details['matched_supporting']}")
    if details["forbidden_found"]:
        basis_parts.append(f"forbidden={details['forbidden_found']}")

    return base_classification(
        detected_type=rule.type_name,
        suggested_folder=rule.folder_name,
        confidence=confidence,
        rule_notes=rule.notes,
        matched_anchors=", ".join(details["matched_anchors"]),
        missing_anchors=", ".join(details["missing_anchors"]),
        matched_supporting=", ".join(details["matched_supporting"]),
        forbidden_found=", ".join(details["forbidden_found"]),
        decision_basis="; ".join(basis_parts),
    )


def classify_file(file_value: object, columns: Set[str]) -> Dict[str, object]:
    """Classify one summary row.

    Extension rules are deliberately evaluated first so that non-Datamine files
    such as .mac, .htm, .jpg, .png, .bmp and .pdf are still classified even
    when the Datamine binary summary parser did not extract columns.
    """
    extension_classification = classify_by_extension(file_value)
    if extension_classification is not None:
        return extension_classification

    return classify_columns(columns)


def classify_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)

    required_cols = {"file", "columns", "status"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Summary file is missing required columns: {sorted(missing)}. "
            f"Expected the v3 batch-summary format."
        )

    rows = []
    for _, row in df.iterrows():
        cols = parse_columns(row.get("columns", ""))
        classification = classify_file(row.get("file", ""), cols)

        out = row.to_dict()
        out["column_set_size"] = len(cols)
        out["column_set"] = ", ".join(sorted(cols))
        out.update(classification)

        # Final sorter-facing fields.
        # The summary-stage column summary_suggested_folder is intentionally
        # blank for most .dm/.csv/.xlsx files because the summary step only
        # pre-classifies simple extension-based files. The classifier's
        # suggested_folder is the actual classification result. These final_*
        # columns remove ambiguity for review and for the later sorter.
        out["final_detected_type"] = first_nonblank_text(
            out.get("detected_type"),
            out.get("summary_file_kind"),
            default="Unknown",
        )
        out["final_suggested_folder"] = first_nonblank_text(
            out.get("suggested_folder"),
            out.get("summary_suggested_folder"),
            default="99_Unknown",
        )

        rows.append(out)

    out_df = pd.DataFrame(rows)

    # Put the final review/sorter fields near the front of the CSV.
    front_cols = [
        "file",
        "file_name",
        "extension",
        "final_detected_type",
        "final_suggested_folder",
        "detected_type",
        "suggested_folder",
        "confidence",
        "summary_file_kind",
        "summary_suggested_folder",
    ]
    ordered_cols = [c for c in front_cols if c in out_df.columns]
    ordered_cols.extend([c for c in out_df.columns if c not in ordered_cols])
    out_df = out_df[ordered_cols]

    out_df = out_df.sort_values(
        by=["status", "detected_type", "confidence", "file"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    return out_df


def build_rule_catalog() -> pd.DataFrame:
    rows = []

    for rule in EXTENSION_RULES:
        rows.append({
            "matching_method": "extension",
            "type_name": rule.type_name,
            "folder_name": rule.folder_name,
            "description": rule.description,
            "extensions": ", ".join(sorted(rule.suffixes)),
            "anchors": "",
            "supporting": "",
            "min_anchor_matches": "",
            "min_total_matches": "",
            "notes": rule.notes,
        })

    for rule in RULES:
        rows.append({
            "matching_method": "columns",
            "type_name": rule.type_name,
            "folder_name": rule.folder_name,
            "description": rule.description,
            "extensions": "",
            "anchors": ", ".join(sorted(rule.anchors)),
            "supporting": ", ".join(sorted(rule.supporting)),
            "min_anchor_matches": rule.min_anchor_matches,
            "min_total_matches": rule.min_total_matches,
            "notes": rule.notes,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify files using the v3 Datamine batch-summary CSV plus extension-based rules."
    )
    parser.add_argument(
        "summary_csv",
        type=Path,
        nargs="?",
        help="Path to the v3 batch-summary CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to <summary stem>_classified.csv",
    )
    parser.add_argument(
        "--rules-out",
        type=Path,
        default=None,
        help="Optional CSV export of the rule catalog.",
    )

    args = parser.parse_args()

    if args.rules_out:
        rules_df = build_rule_catalog()
        args.rules_out.parent.mkdir(parents=True, exist_ok=True)
        rules_df.to_csv(args.rules_out, index=False)
        print(f"Rule catalog written to: {args.rules_out}")

    if args.summary_csv:
        summary_csv = args.summary_csv
        if not summary_csv.exists():
            raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

        classified = classify_summary(summary_csv)
        out_path = args.out or summary_csv.with_name(summary_csv.stem + "_classified.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        classified.to_csv(out_path, index=False)

        print(f"Classified file written to: {out_path}")
        print()
        print(classified[[
            "file",
            "final_detected_type",
            "final_suggested_folder",
            "confidence",
            "decision_basis",
        ]].to_string(index=False))
    elif not args.rules_out:
        parser.error("Provide a summary_csv path, or use --rules-out to export the rule catalog.")


if __name__ == "__main__":
    main()
