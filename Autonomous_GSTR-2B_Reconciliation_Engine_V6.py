import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
from datetime import datetime
from thefuzz import fuzz
import pdfplumber

# ============================================================
# AUTONOMOUS GSTR-2B RECONCILIATION ENGINE — V6.1
# Reconciliation Safety & Accuracy Hardened Edition
# V6.1 Fix: dual-key JSON items lookup ("itms" + "items"),
#           dual field-name aliases for tax components
#           (iamt/igst, camt/cgst, samt/sgst, csamt/cess),
#           dual Reverse Charge field ("rchrg" + "rev").
# ============================================================

st.set_page_config(
    page_title="GSTR-2B Auto-Reco V6.1",
    page_icon="📊",
    layout="wide"
)

st.title("⚡ Autonomous GSTR-2B Reconciliation Engine (V6.1)")
st.markdown("""
**Safety-hardened reconciliation assistance** with JSON/PDF parsing,
deterministic + fuzzy matching, document-type awareness, confidence gates,
duplicate detection, review queues and vendor follow-up.

**Privacy:** processing occurs in this application instance; this MVP does
not call external APIs.

**Disclaimer:** This is reconciliation assistance. It does not verify
official GST portal filing status, determine legal ITC eligibility, or replace
professional tax review.
""")

# ============================================================
# 1. SCHEMA / CONSTANTS
# ============================================================

COLUMN_ALIASES = {
    "GSTIN": ["GSTIN", "GSTIN/UIN", "Supplier GSTIN", "Vendor GSTIN"],
    "Vendor_Name": ["Vendor Name", "Supplier Name", "Party Name", "Trade Name"],
    "Invoice_Number": [
        "Invoice Number", "Invoice No", "Invoice No.", "Inv No",
        "Document Number", "Document No", "Note Number"
    ],
    "Invoice_Date": ["Invoice Date", "Inv Date", "Document Date", "Date", "Note Date"],
    "Taxable_Value": ["Taxable Value", "Taxable Value (₹)", "Taxable Amount", "Taxable Amt"],
    "IGST": ["IGST", "IGST Amount", "Integrated Tax"],
    "CGST": ["CGST", "CGST Amount", "Central Tax"],
    "SGST": ["SGST", "SGST Amount", "State Tax", "SGST/UTGST"],
    "CESS": ["CESS", "Cess Amount", "Cess"],
    "Total_Tax": ["Total Tax", "Tax Amount", "Total Tax Amount", "Total GST", "IGST+CGST+SGST"],
    "Document_Type": ["Document Type", "Doc Type", "Invoice Type", "Note Type", "Type"],
    "Original_Invoice_Number": [
        "Original Invoice Number", "Original Invoice No", "Original Document Number"
    ],
    "Original_Invoice_Date": [
        "Original Invoice Date", "Original Document Date"
    ],
    "Reverse_Charge": [
        "Reverse Charge", "RCM", "Is Reverse Charge", "Reverse Charge Applicable"
    ],
}

FINANCIAL_COLS = [
    "Taxable_Value", "IGST", "CGST", "SGST", "CESS", "Total_Tax"
]

MATCHED_CATEGORIES = {
    "EXACT", "NORMALIZED", "HIGH_CONFIDENCE", "PROBABLE", "CARRIED_FORWARD_MATCH"
}

MATCH_CATEGORIES = {
    "EXACT": "✅ Exact Match",
    "NORMALIZED": "🟢 Normalized Match",
    "HIGH_CONFIDENCE": "🟢 High Confidence Match",
    "PROBABLE": "🟡 Probable Match",
    "DATE_MISMATCH": "🔴 Date Mismatch",
    "TAX_MISMATCH": "🔴 Tax Mismatch",
    "REVIEW": "🟠 Review Required",
    "MULTIPLE_CANDIDATES": "🟠 Multiple Candidates (Review)",
    "MISSING": "🔴 Missing in GSTR-2B (ITC Review Required)",
    "CARRIED_FORWARD_MATCH": "🗓️ Resolved via Carry-Forward (Vendor Filed Late)",
    "BOOKS_DUPLICATE": "⚠️ Duplicate in Books",
    "GSTR2B_DUPLICATE": "⚠️ Duplicate in 2B",
    "DATA_QUALITY": "⚠️ Data Quality Issue",
}

DOCUMENT_TYPES = {
    "INVOICE": "INVOICE",
    "CREDIT_NOTE": "CREDIT_NOTE",
    "DEBIT_NOTE": "DEBIT_NOTE",
    "AMENDMENT": "AMENDMENT",
    "AMENDED_CREDIT_NOTE": "AMENDED_CREDIT_NOTE",
    "AMENDED_DEBIT_NOTE": "AMENDED_DEBIT_NOTE",
    "UNKNOWN": "UNKNOWN",
}

# ============================================================
# 2. SAFE PARSING / NORMALIZATION
# ============================================================

def _safe_float(d, key):
    if key not in d or d[key] is None or str(d[key]).strip() == "":
        return np.nan
    try:
        return float(d[key])
    except (TypeError, ValueError):
        return np.nan


def clean_invoice(val):
    if pd.isna(val) or str(val).strip() == "":
        return "UNKNOWN_INV"
    return re.sub(r"[^A-Z0-9]", "", str(val).upper())


def normalize_gstin(val):
    if pd.isna(val) or str(val).strip() == "":
        return "UNKNOWN_GSTIN"
    return re.sub(r"\s+", "", str(val).upper().strip())


def validate_gstin(val):
    if pd.isna(val) or not isinstance(val, str):
        return False
    pattern = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
    return bool(re.fullmatch(pattern, val.strip().upper()))


def infer_document_type(source_section="", explicit_type=""):
    text = f"{source_section} {explicit_type}".upper()
    explicit_upper = str(explicit_type).upper().strip()

    # GSTN explicit note-type shortcodes take priority over section-based
    # inference, since they're the most direct signal available.
    if explicit_upper in {"C", "CR", "CREDIT", "CREDIT NOTE"}:
        return "CREDIT_NOTE"
    if explicit_upper in {"D", "DR", "DEBIT", "DEBIT NOTE"}:
        return "DEBIT_NOTE"

    if "CDNRA" in text or "AMENDED CREDIT" in text:
        return "AMENDED_CREDIT_NOTE"
    if "CDNR" in text:
        # CDNR can contain credit/debit notes; retain UNKNOWN until the
        # actual note type is available rather than inventing a sign.
        if "DEBIT" in text:
            return "DEBIT_NOTE"
        if "CREDIT" in text:
            return "CREDIT_NOTE"
        return "UNKNOWN"
    if "B2BA" in text or "AMEND" in text:
        return "AMENDMENT"
    if "DEBIT" in text:
        return "DEBIT_NOTE"
    if "CREDIT" in text:
        return "CREDIT_NOTE"
    if "INVOICE" in text or "B2B" in text:
        return "INVOICE"
    return "UNKNOWN"


def parse_date_safe(value):
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    return pd.to_datetime(value, dayfirst=True, errors="coerce")


def within_tolerance(ref, cand, abs_tol, rel_tol_pct):
    if pd.isna(ref) or pd.isna(cand):
        return False
    diff = abs(float(ref) - float(cand))
    if diff <= abs_tol:
        return True
    base = max(abs(float(ref)), abs(float(cand)), 0.0001)
    return ((diff / base) * 100.0) <= rel_tol_pct


def calculate_financial_similarity(ref, cand, abs_tol, rel_tol_pct):
    # V5.1: missing != zero. Unknown financial values do not receive a
    # zero-score/zero-value substitution.
    if pd.isna(ref) or pd.isna(cand):
        return np.nan
    ref = float(ref)
    cand = float(cand)
    if ref == 0.0 and cand == 0.0:
        return 100.0
    diff = abs(ref - cand)
    if diff <= abs_tol:
        return 100.0
    base = max(abs(ref), abs(cand), 0.0001)
    rel_diff_pct = (diff / base) * 100.0
    if rel_diff_pct <= rel_tol_pct:
        return 100.0
    return max(0.0, 100.0 - ((rel_diff_pct - rel_tol_pct) * 2))


# ============================================================
# 3. JSON INGESTION
# ============================================================

def _find_all_sections(node, section_key):
    """V5.1: collect ALL matching section lists, not only the first one."""
    found_lists = []
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k).lower() == section_key.lower() and isinstance(v, list):
                found_lists.append(v)
            else:
                found_lists.extend(_find_all_sections(v, section_key))
    elif isinstance(node, list):
        for item in node:
            found_lists.extend(_find_all_sections(item, section_key))
    return found_lists


def _extract_invoice_rows(supplier_block, section_label):
    rows = []
    ctin = supplier_block.get("ctin", "")
    trade_name = supplier_block.get(
        "trdnm", supplier_block.get("trdname", "")
    )

    doc_list = supplier_block.get("inv") or supplier_block.get("nt") or []

    for doc in doc_list:
        inum = doc.get("inum", doc.get("ntnum", ""))
        idt = doc.get("idt", doc.get("ntdt", ""))

        # Preserve note/document metadata when present.
        explicit_type = (
            doc.get("typ") or doc.get("ntty") or doc.get("type") or ""
        )
        original_invoice_number = (
            doc.get("oinum") or doc.get("ontnum") or doc.get("original_inum") or ""
        )
        original_invoice_date = (
            doc.get("oidt") or doc.get("ontdt") or doc.get("original_idt") or ""
        )

        # V6.1 FIX: GST portal JSON uses either "itms" (older exports) or
        # "items" (newer exports, e.g. the format used in this test file).
        # Fall back through both keys before giving up.
        items = doc.get("itms") or doc.get("items") or []

        def _safe_float_multi(d, *keys):
            """Try multiple field-name variants; return first non-NaN found."""
            for key in keys:
                val = _safe_float(d, key)
                if not pd.isna(val):
                    return val
            return np.nan

        component_values = {"Taxable_Value": [], "IGST": [], "CGST": [], "SGST": [], "CESS": []}

        if items:
            for it in items:
                d = it.get("itm_det", it)
                # V6.1 FIX: accept both old-portal names (iamt/camt/samt/csamt)
                # AND new-portal names (igst/cgst/sgst/cess) for every component.
                component_values["Taxable_Value"].append(_safe_float(d, "txval"))
                component_values["IGST"].append(_safe_float_multi(d, "iamt", "igst"))
                component_values["CGST"].append(_safe_float_multi(d, "camt", "cgst"))
                component_values["SGST"].append(_safe_float_multi(d, "samt", "sgst"))
                component_values["CESS"].append(_safe_float_multi(d, "csamt", "cess"))

            vals = {}
            for col, arr in component_values.items():
                # If any item has a missing component, preserve unknown state.
                vals[col] = np.nan if any(pd.isna(x) for x in arr) else float(sum(arr))
        else:
            taxable = _safe_float(doc, "txval")
            if pd.isna(taxable):
                taxable = _safe_float(doc, "val")
            vals = {
                "Taxable_Value": taxable,
                # V6.1 FIX: same dual-alias lookup at doc level.
                "IGST": _safe_float_multi(doc, "iamt", "igst"),
                "CGST": _safe_float_multi(doc, "camt", "cgst"),
                "SGST": _safe_float_multi(doc, "samt", "sgst"),
                "CESS": _safe_float_multi(doc, "csamt", "cess"),
            }

        if all(not pd.isna(vals[c]) for c in ["IGST", "CGST", "SGST", "CESS"]):
            vals["Total_Tax"] = sum(vals[c] for c in ["IGST", "CGST", "SGST", "CESS"])
        else:
            vals["Total_Tax"] = np.nan

        document_type = infer_document_type(section_label, explicit_type)

        rows.append({
            "GSTIN": ctin,
            "Vendor_Name": trade_name,
            "Invoice_Number": inum,
            "Invoice_Date": idt,
            **vals,
            "Source_Section": section_label,
            "Document_Type": document_type,
            "Original_Invoice_Number": original_invoice_number,
            "Original_Invoice_Date": original_invoice_date,
            # V6.1 FIX: portal uses "rchrg" in some exports, "rev" in others.
            "Reverse_Charge": doc.get("rchrg") or doc.get("rev") or "",
        })
    return rows


def parse_gstr2b_json(uploaded_file):
    warnings = []
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"File is not valid JSON: {e}")

    sections = {
        "b2b": "B2B",
        "b2ba": "B2B_AMENDED",
        "cdnr": "CDNR",
        "cdnra": "CDNR_AMENDED",
    }

    all_rows = []
    for section_key, section_label in sections.items():
        section_lists = _find_all_sections(payload, section_key)
        for section_list in section_lists:
            for supplier_block in section_list:
                all_rows.extend(
                    _extract_invoice_rows(supplier_block, section_label)
                )

    if not all_rows:
        warnings.append(
            "No recognized B2B/B2BA/CDNR/CDNRA invoice sections found."
        )
        return pd.DataFrame(), warnings

    df = pd.DataFrame(all_rows)

    note_rows = df["Source_Section"].isin(["CDNR", "CDNR_AMENDED"])
    if note_rows.any():
        warnings.append(
            f"{int(note_rows.sum())} row(s) are note records. "
            "V5.1 does not silently infer credit/debit sign where GSTN metadata "
            "does not identify it; review Document_Type."
        )

    amendment_rows = df["Source_Section"].isin(["B2B_AMENDED", "CDNR_AMENDED"])
    if amendment_rows.any():
        warnings.append(
            f"{int(amendment_rows.sum())} amended row(s) detected. "
            "Original-document linkage must be reviewed where metadata is absent."
        )

    return df, warnings


# ============================================================
# 4. PDF INGESTION
# ============================================================

def parse_gstr2b_pdf(uploaded_file):
    warnings = []
    all_rows = []

    header_aliases = {
        "gstin": "GSTIN",
        "trade": "Vendor_Name",
        "invoice no": "Invoice_Number",
        "invoice number": "Invoice_Number",
        "invoice date": "Invoice_Date",
        "taxable value": "Taxable_Value",
        "integrated tax": "IGST",
        "central tax": "CGST",
        "state/ut tax": "SGST",
        "cess": "CESS",
    }

    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                header_row = [str(c or "").strip().lower() for c in table[0]]
                col_map = {}

                for i, h in enumerate(header_row):
                    for alias, standard in header_aliases.items():
                        if alias in h:
                            col_map[i] = standard
                            break

                if not col_map:
                    continue

                for row in table[1:]:
                    if not row or all(c is None or str(c).strip() == "" for c in row):
                        continue

                    record = {}
                    for i, val in enumerate(row):
                        if i in col_map:
                            record[col_map[i]] = str(val).strip() if val else ""

                    if record.get("GSTIN") or record.get("Invoice_Number"):
                        record["Source_Page"] = page_num
                        record["Source_Section"] = "PDF"
                        record["Document_Type"] = "UNKNOWN"
                        all_rows.append(record)

    if not all_rows:
        warnings.append(
            "No invoice tables detected in PDF. Prefer JSON/XLSX where available."
        )
        return pd.DataFrame(), warnings

    df = pd.DataFrame(all_rows)

    for col in ["Taxable_Value", "IGST", "CGST", "SGST", "CESS"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    tax_cols = [c for c in ["IGST", "CGST", "SGST", "CESS"] if c in df.columns]
    if tax_cols:
        df["Total_Tax"] = df[tax_cols].sum(axis=1, min_count=len(tax_cols))

    if "Invoice_Date" in df.columns:
        df["Invoice_Date"] = pd.to_datetime(
            df["Invoice_Date"], dayfirst=True, errors="coerce"
        )

    warnings.append(
        f"Extracted {len(df)} row(s) from PDF. PDF data is marked as "
        "best-effort and should be previewed before reconciliation."
    )
    return df, warnings


# ============================================================
# 5. COLUMN MAPPING / PREPROCESSING
# ============================================================

def map_columns(df):
    mapped_df = df.copy()
    col_map = {}
    ambiguous = []

    normalized_columns = {
        col: str(col).strip().lower() for col in mapped_df.columns
    }

    for standard, aliases in COLUMN_ALIASES.items():
        alias_set = {a.lower() for a in aliases}
        found = [
            col for col, normalized in normalized_columns.items()
            if normalized in alias_set
        ]

        if len(found) == 1:
            col_map[found[0]] = standard
        elif len(found) > 1:
            ambiguous.append(standard)

    mapped_df = mapped_df.rename(columns=col_map)

    missing = []
    for req in COLUMN_ALIASES:
        if req not in mapped_df.columns:
            # V5.1: preserve missingness as NaN. Never convert unknown
            # financial data to zero.
            mapped_df[req] = np.nan
            missing.append(req)

    return mapped_df, ambiguous, missing


def preprocess_data(df, source_name, tolerances):
    df, ambig, missing_cols = map_columns(df)

    for col in COLUMN_ALIASES:
        df[f"Raw_{col}"] = df[col]

    df["Parsed_GSTIN"] = df["GSTIN"].apply(normalize_gstin)
    df["Parsed_Inv"] = df["Invoice_Number"].apply(clean_invoice)
    df["Parsed_Date"] = df["Invoice_Date"].apply(parse_date_safe)

    df["Parsed_Document_Type"] = [
        infer_document_type(
            df.iloc[i].get("Source_Section", ""),
            df.iloc[i].get("Document_Type", "")
        )
        for i in range(len(df))
    ]

    df["Parsed_Original_Inv"] = df["Original_Invoice_Number"].apply(clean_invoice)

    # V6: Reverse Charge Mechanism flag, normalized to a clean tri-state.
    # RCM invoices are ITC-eligible only after the recipient self-pays tax,
    # so they must never be silently pooled with normal B2B invoices in
    # headline "claimable ITC" figures.
    def _parse_rcm(val):
        if pd.isna(val) or str(val).strip() == "":
            return "UNKNOWN"
        v = str(val).strip().upper()
        if v in {"Y", "YES", "TRUE", "1"}:
            return "YES"
        if v in {"N", "NO", "FALSE", "0"}:
            return "NO"
        return "UNKNOWN"

    df["Parsed_RCM"] = df["Reverse_Charge"].apply(_parse_rcm)

    df["Data_Quality_Issue"] = ""
    df["Match_Category"] = "PENDING"

    if ambig:
        df["Data_Quality_Issue"] += (
            f"Ambiguous mapping for {', '.join(ambig)}; "
        )

    # Numeric parsing: missing stays NaN.
    for col in FINANCIAL_COLS:
        parsed = pd.to_numeric(df[col], errors="coerce")
        raw = df[f"Raw_{col}"]
        nonblank = raw.notna() & (raw.astype(str).str.strip() != "")
        bad_parse = parsed.isna() & nonblank

        df.loc[bad_parse, "Data_Quality_Issue"] += f"Invalid {col}; "
        df[f"Parsed_{col}"] = parsed

        df.loc[parsed < 0, "Data_Quality_Issue"] += f"Negative {col}; "
        blank = parsed.isna() & ~bad_parse
        df.loc[blank, "Data_Quality_Issue"] += f"Missing {col}; "

    # If Total_Tax was not supplied, calculate it only when ALL components
    # are known. min_count prevents missing values becoming fake zeroes.
    if "Total_Tax" in missing_cols:
        components = ["Parsed_IGST", "Parsed_CGST", "Parsed_SGST", "Parsed_CESS"]
        df["Parsed_Total_Tax"] = df[components].sum(axis=1, min_count=4)

    invalid_gstin = ~df["Parsed_GSTIN"].apply(validate_gstin)
    df.loc[
        invalid_gstin & (df["Parsed_GSTIN"] != "UNKNOWN_GSTIN"),
        "Data_Quality_Issue"
    ] += "Invalid GSTIN format; "

    df.loc[
        df["Parsed_GSTIN"] == "UNKNOWN_GSTIN",
        "Data_Quality_Issue"
    ] += "Missing GSTIN; "

    df.loc[
        df["Parsed_Inv"] == "UNKNOWN_INV",
        "Data_Quality_Issue"
    ] += "Missing Invoice No; "

    if "Invoice_Date" not in missing_cols:
        df.loc[
            df["Parsed_Date"].isna() & df["Raw_Invoice_Date"].notna(),
            "Data_Quality_Issue"
        ] += "Invalid Date format; "

    # Composition check must be three-state:
    # valid, mismatch, or not verifiable.
    component_cols = [
        "Parsed_IGST", "Parsed_CGST", "Parsed_SGST", "Parsed_CESS"
    ]
    comp_known = df[component_cols].notna().all(axis=1)
    total_known = df["Parsed_Total_Tax"].notna()

    comp_calc = df[component_cols].sum(axis=1, min_count=4)
    tot = df["Parsed_Total_Tax"]
    comp_diff = abs(comp_calc - tot)

    comp_base = np.maximum(
        np.maximum(abs(comp_calc.fillna(0)), abs(tot.fillna(0))),
        0.0001
    )
    comp_rel_diff_pct = (comp_diff / comp_base) * 100.0

    comp_mismatch = (
        comp_known
        & total_known
        & (
            (comp_diff > tolerances["comp"]["abs"])
            & (comp_rel_diff_pct > tolerances["comp"]["rel"])
        )
    )

    df.loc[
        comp_mismatch,
        "Data_Quality_Issue"
    ] += "Tax Composition Mismatch; "

    valid_keys = (
        (df["Parsed_GSTIN"] != "UNKNOWN_GSTIN")
        & (df["Parsed_Inv"] != "UNKNOWN_INV")
    )

    dupes = (
        df.duplicated(
            subset=["Parsed_GSTIN", "Parsed_Inv"],
            keep=False
        )
        & valid_keys
    )

    df.loc[
        dupes,
        "Data_Quality_Issue"
    ] += f"Duplicate in {source_name}; "

    df.loc[
        df["Data_Quality_Issue"] != "",
        "Match_Category"
    ] = "DATA_QUALITY"

    df.loc[
        dupes,
        "Match_Category"
    ] = f"{source_name}_DUPLICATE"

    df["Row_ID"] = [
        f"{source_name}_{i}" for i in range(len(df))
    ]

    return df


# ============================================================
# 6. MATCHING SAFETY FUNCTIONS
# ============================================================

def comparable_financial_scores(b_row, c_row, tolerances):
    scores = []

    for col, tol_key in [
        ("Taxable_Value", "taxable"),
        ("Total_Tax", "total"),
        ("IGST", "comp"),
        ("CGST", "comp"),
        ("SGST", "comp"),
        ("CESS", "comp"),
    ]:
        score = calculate_financial_similarity(
            b_row[f"Parsed_{col}"],
            c_row[f"Parsed_{col}"],
            tolerances[tol_key]["abs"],
            tolerances[tol_key]["rel"],
        )
        if not pd.isna(score):
            scores.append(score)

    return scores


def financial_candidate_gate(
    b_row, c_row, tolerances, min_fin_similarity=70
):
    scores = comparable_financial_scores(b_row, c_row, tolerances)

    if not scores:
        return False, 0.0, "No comparable financial fields available."

    overall = float(np.mean(scores))

    # V5.1: require at least one KEY financial field to be within tolerance.
    key_results = []
    for col, tol_key in [
        ("Taxable_Value", "taxable"),
        ("Total_Tax", "total"),
    ]:
        ref = b_row[f"Parsed_{col}"]
        cand = c_row[f"Parsed_{col}"]
        if pd.notna(ref) and pd.notna(cand):
            key_results.append(
                within_tolerance(
                    ref,
                    cand,
                    tolerances[tol_key]["abs"],
                    tolerances[tol_key]["rel"],
                )
            )

    key_pass = any(key_results) if key_results else False

    if key_pass and overall >= min_fin_similarity:
        return True, round(overall, 2), "Financial proximity gate passed."

    return False, round(overall, 2), "Financial proximity gate failed."


def calculate_fuzzy_score(b_row, g2b_row, weights, tolerances):
    scores = {}
    total_score = 0.0
    active_w = 0.0

    if weights["inv"] > 0:
        sim = fuzz.ratio(
            b_row["Parsed_Inv"],
            g2b_row["Parsed_Inv"]
        )
        scores["inv"] = sim
        total_score += sim * weights["inv"]
        active_w += weights["inv"]

    if weights["date"] > 0:
        if (
            pd.notnull(b_row["Parsed_Date"])
            and pd.notnull(g2b_row["Parsed_Date"])
        ):
            diff = abs(
                (b_row["Parsed_Date"] - g2b_row["Parsed_Date"]).days
            )
            if diff == 0:
                sim = 100
            elif diff <= tolerances["date_days"]:
                sim = 80
            else:
                sim = max(0, 100 - diff * 2)

            scores["date"] = sim
            total_score += sim * weights["date"]
            active_w += weights["date"]

    for col, w_key, tol_key in [
        ("Taxable_Value", "taxable", "taxable"),
        ("IGST", "igst", "comp"),
        ("CGST", "cgst", "comp"),
        ("SGST", "sgst", "comp"),
        ("CESS", "cess", "comp"),
        ("Total_Tax", "total", "total"),
    ]:
        if weights[w_key] <= 0:
            continue

        sim = calculate_financial_similarity(
            b_row[f"Parsed_{col}"],
            g2b_row[f"Parsed_{col}"],
            tolerances[tol_key]["abs"],
            tolerances[tol_key]["rel"],
        )

        # V5.1: don't score unknown fields.
        if pd.notna(sim):
            scores[w_key] = sim
            total_score += sim * weights[w_key]
            active_w += weights[w_key]

    if active_w == 0:
        return 0.0, scores

    return total_score / active_w, scores


def document_type_compatible(b_type, g_type):
    if b_type == "UNKNOWN" or g_type == "UNKNOWN":
        return True

    if b_type == g_type:
        return True

    # Do not automatically match ordinary invoices to credit/debit notes
    # or amendments.
    incompatible = {
        ("INVOICE", "CREDIT_NOTE"),
        ("INVOICE", "DEBIT_NOTE"),
        ("INVOICE", "AMENDED_CREDIT_NOTE"),
        ("INVOICE", "AMENDED_DEBIT_NOTE"),
        ("INVOICE", "AMENDMENT"),
        ("CREDIT_NOTE", "INVOICE"),
        ("DEBIT_NOTE", "INVOICE"),
        ("AMENDED_CREDIT_NOTE", "INVOICE"),
        ("AMENDED_DEBIT_NOTE", "INVOICE"),
        ("AMENDMENT", "INVOICE"),
    }
    return (b_type, g_type) not in incompatible


# ============================================================
# 7. RECONCILIATION ENGINE
# ============================================================

def reconcile_engine(books_df, g2b_df, tolerances, weights, gates):
    if sum(weights.values()) <= 0:
        raise ValueError("Total active weights must be greater than 0.")

    books = preprocess_data(books_df, "BOOKS", tolerances)
    gstr2b = preprocess_data(g2b_df, "GSTR2B", tolerances)

    books["Matched_2B_Invoice"] = None
    books["Matched_2B_RowID"] = None
    books["Confidence_Score"] = np.nan
    books["Match_Reason"] = ""
    books["Score_Margin"] = np.nan

    matched_2b_ids = set()

    valid_b = books[
        ~books["Match_Category"].isin(
            ["DATA_QUALITY", "BOOKS_DUPLICATE"]
        )
    ]

    valid_2b = gstr2b[
        ~gstr2b["Match_Category"].isin(
            ["DATA_QUALITY", "GSTR2B_DUPLICATE"]
        )
    ]

    # --------------------------------------------------------
    # PASS 1: exact identity
    # --------------------------------------------------------
    for idx, b_row in valid_b.iterrows():
        b_gstin = b_row["Parsed_GSTIN"]
        b_inv = b_row["Parsed_Inv"]

        if b_gstin == "UNKNOWN_GSTIN" or b_inv == "UNKNOWN_INV":
            continue

        candidates = valid_2b[
            (valid_2b["Parsed_GSTIN"] == b_gstin)
            & (valid_2b["Parsed_Inv"] == b_inv)
            & (
                valid_2b["Row_ID"].map(
                    lambda x: x not in matched_2b_ids
                )
            )
        ]

        candidates = candidates[
            candidates.apply(
                lambda r: document_type_compatible(
                    b_row["Parsed_Document_Type"],
                    r["Parsed_Document_Type"]
                ),
                axis=1
            )
        ]

        if len(candidates) > 1:
            books.at[idx, "Match_Category"] = "MULTIPLE_CANDIDATES"
            books.at[idx, "Match_Reason"] = (
                f"{len(candidates)} exact identity candidates found; "
                "manual review required."
            )
            continue

        if len(candidates) != 1:
            continue

        match = candidates.iloc[0]

        # Exact identity does NOT mean exact financial match.
        tax_match = within_tolerance(
            b_row["Parsed_Total_Tax"],
            match["Parsed_Total_Tax"],
            tolerances["total"]["abs"],
            tolerances["total"]["rel"],
        )
        taxable_match = within_tolerance(
            b_row["Parsed_Taxable_Value"],
            match["Parsed_Taxable_Value"],
            tolerances["taxable"]["abs"],
            tolerances["taxable"]["rel"],
        )

        component_results = {}
        for comp in ["IGST", "CGST", "SGST", "CESS"]:
            b_val = b_row[f"Parsed_{comp}"]
            g_val = match[f"Parsed_{comp}"]
            component_results[comp] = (
                within_tolerance(
                    b_val,
                    g_val,
                    tolerances["comp"]["abs"],
                    tolerances["comp"]["rel"],
                )
                if pd.notna(b_val) and pd.notna(g_val)
                else None
            )

        comparable_components = [
            v for v in component_results.values()
            if v is not None
        ]
        components_match = (
            bool(comparable_components)
            and all(comparable_components)
        )

        dates_present = (
            pd.notnull(b_row["Parsed_Date"])
            and pd.notnull(match["Parsed_Date"])
        )
        date_diff = (
            abs(
                (b_row["Parsed_Date"] - match["Parsed_Date"]).days
            )
            if dates_present else None
        )
        date_match = (
            dates_present
            and date_diff <= tolerances["date_days"]
        )

        if tax_match and taxable_match and components_match:
            if not dates_present:
                cat = "NORMALIZED"
                reason = (
                    "Identity and financials matched, but dates are "
                    "unavailable for verification."
                )
            elif not date_match:
                cat = "DATE_MISMATCH"
                reason = (
                    f"Identity and financials matched, but date differs "
                    f"by {date_diff} day(s)."
                )
            else:
                cat = (
                    "EXACT"
                    if b_row["Raw_Invoice_Number"]
                    == match["Raw_Invoice_Number"]
                    else "NORMALIZED"
                )
                reason = (
                    "GSTIN and invoice identity matched; comparable "
                    "financial values are within configured tolerances."
                )
        else:
            cat = "TAX_MISMATCH"
            reason = (
                "Identity matched, but one or more comparable financial "
                "values exceed tolerance or cannot be verified."
            )

        books.at[idx, "Match_Category"] = cat
        books.at[idx, "Match_Reason"] = reason
        books.at[idx, "Matched_2B_Invoice"] = match["Raw_Invoice_Number"]
        books.at[idx, "Matched_2B_RowID"] = match["Row_ID"]
        books.at[idx, "Confidence_Score"] = 100.0
        matched_2b_ids.add(match["Row_ID"])

    # --------------------------------------------------------
    # PASS 2: fuzzy candidates
    # --------------------------------------------------------
    unmatched_b = books[
        (~books["Match_Category"].isin([
            "DATA_QUALITY",
            "BOOKS_DUPLICATE",
            "MULTIPLE_CANDIDATES"
        ]))
        & (books["Confidence_Score"].isna())
    ]

    for idx, b_row in unmatched_b.iterrows():

        pool = valid_2b[
            (valid_2b["Parsed_GSTIN"] == b_row["Parsed_GSTIN"])
            & (~valid_2b["Row_ID"].isin(matched_2b_ids))
        ].copy()

        if b_row["Parsed_GSTIN"] == "UNKNOWN_GSTIN":
            books.at[idx, "Match_Category"] = "MISSING"
            books.at[idx, "Match_Reason"] = "Missing/invalid GSTIN; fuzzy matching disabled."
            continue

        pool = pool[
            pool.apply(
                lambda r: document_type_compatible(
                    b_row["Parsed_Document_Type"],
                    r["Parsed_Document_Type"]
                ),
                axis=1
            )
        ]

        if pool.empty:
            books.at[idx, "Match_Category"] = "MISSING"
            books.at[idx, "Match_Reason"] = (
                "No unmatched GSTR-2B candidate with compatible document "
                "type and same GSTIN."
            )
            continue

        pool["Inv_Sim"] = pool["Parsed_Inv"].apply(
            lambda x: fuzz.ratio(
                b_row["Parsed_Inv"], x
            )
        )

        # V5.1 safer default: configurable but intentionally higher than V5.0.
        pool = pool[
            pool["Inv_Sim"] >= gates["min_inv_sim"]
        ]

        if pool.empty:
            books.at[idx, "Match_Category"] = "MISSING"
            books.at[idx, "Match_Reason"] = (
                f"No candidate passed invoice similarity gate "
                f"({gates['min_inv_sim']}%)."
            )
            continue

        if pd.notnull(b_row["Parsed_Date"]):
            pool["Date_Diff"] = (
                b_row["Parsed_Date"] - pool["Parsed_Date"]
            ).abs().dt.days
            primary_pool = pool[
                (pool["Date_Diff"] <= tolerances["date_days"])
                | pool["Date_Diff"].isna()
            ].copy()
            fallback_pool = pool[
                ~pool.index.isin(primary_pool.index)
            ].copy()
        else:
            primary_pool = pool.copy()
            fallback_pool = pool.iloc[0:0].copy()

        def apply_financial_gate(candidate_pool):
            if candidate_pool.empty:
                return candidate_pool.copy()

            passed = []
            for c_idx, c_row in candidate_pool.iterrows():
                ok, _, _ = financial_candidate_gate(
                    b_row,
                    c_row,
                    tolerances,
                    gates["min_fin_similarity"]
                )
                if ok:
                    passed.append(c_idx)

            return candidate_pool.loc[passed].copy()

        active_pool = apply_financial_gate(primary_pool)
        if active_pool.empty:
            active_pool = apply_financial_gate(fallback_pool)

        if active_pool.empty:
            books.at[idx, "Match_Category"] = "MISSING"
            books.at[idx, "Match_Reason"] = (
                "No candidate passed invoice, date and financial "
                "proximity gates."
            )
            continue

        scored = []
        for _, c_row in active_pool.iterrows():
            score, details = calculate_fuzzy_score(
                b_row, c_row, weights, tolerances
            )
            scored.append({
                "row": c_row,
                "score": score,
                "details": details
            })

        scored.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        if not scored or scored[0]["score"] < gates["min_score"]:
            books.at[idx, "Match_Category"] = "MISSING"
            books.at[idx, "Match_Reason"] = (
                f"Best candidate scored below minimum threshold "
                f"({gates['min_score']}%)."
            )
            continue

        best = scored[0]
        second = scored[1] if len(scored) > 1 else None
        margin = (
            best["score"] - second["score"]
            if second else best["score"]
        )

        books.at[idx, "Score_Margin"] = round(margin, 2)
        books.at[idx, "Confidence_Score"] = round(
            best["score"], 2
        )

        if second and margin < gates["min_margin"]:
            books.at[idx, "Match_Category"] = "MULTIPLE_CANDIDATES"
            books.at[idx, "Match_Reason"] = (
                f"Ambiguous fuzzy result. Best={best['score']:.1f}, "
                f"Second={second['score']:.1f}, margin={margin:.1f}."
            )
            # Critical: ambiguous candidates are NOT consumed.
            books.at[idx, "Matched_2B_Invoice"] = None
            books.at[idx, "Matched_2B_RowID"] = None
            continue

        cat = (
            "HIGH_CONFIDENCE"
            if best["score"] >= gates["high_confidence_score"]
            else "PROBABLE"
        )

        books.at[idx, "Match_Category"] = cat
        books.at[idx, "Matched_2B_Invoice"] = (
            best["row"]["Raw_Invoice_Number"]
        )
        books.at[idx, "Matched_2B_RowID"] = best["row"]["Row_ID"]
        books.at[idx, "Match_Reason"] = (
            f"Fuzzy score={best['score']:.1f}%; "
            f"invoice similarity={best['details'].get('inv', 0):.1f}%."
        )

        matched_2b_ids.add(best["row"]["Row_ID"])

    books["Status"] = books["Match_Category"].map(
        MATCH_CATEGORIES
    ).fillna("Unknown State")

    unmatched_2b = gstr2b[
        (~gstr2b["Row_ID"].isin(matched_2b_ids))
        & (gstr2b["Match_Category"] != "GSTR2B_DUPLICATE")
    ]

    return books, unmatched_2b, gstr2b


# ============================================================
# 7B. CREDIT/DEBIT NOTE PARENT LINKING
# ============================================================
# Roadmap item #2. Links each Credit/Debit Note (and their amended
# equivalents) to its original parent invoice within the SAME GSTIN, using
# the Original_Invoice_Number the note itself declares. This does not
# change any matching decision made above — it is a read-only annotation
# pass on top of the already-reconciled `books` DataFrame, so it can never
# make a previously-safe match unsafe.

NOTE_TYPES = {"CREDIT_NOTE", "DEBIT_NOTE", "AMENDED_CREDIT_NOTE", "AMENDED_DEBIT_NOTE"}
CREDIT_TYPES = {"CREDIT_NOTE", "AMENDED_CREDIT_NOTE"}
DEBIT_TYPES = {"DEBIT_NOTE", "AMENDED_DEBIT_NOTE"}


def link_credit_debit_notes(books_df):
    """
    Returns a COPY of books_df with four new columns:
      - Linked_Parent_RowID / Linked_Parent_Invoice: identity of the parent
        invoice found in the same Books upload, if any.
      - Net_ITC_Adjustment: signed tax impact of this note (negative for
        credit notes, positive for debit notes); NaN for ordinary invoices.
      - CN_DN_Link_Status: "LINKED", "PARENT_NOT_FOUND", or "" (not a note).
    Parent lookup key: (Parsed_GSTIN, cleaned Original_Invoice_Number)
    matched against ordinary INVOICE rows in the same DataFrame.
    """
    df = books_df.copy()
    df["Linked_Parent_RowID"] = None
    df["Linked_Parent_Invoice"] = None
    df["Net_ITC_Adjustment"] = np.nan
    df["CN_DN_Link_Status"] = ""

    invoice_rows = df[df["Parsed_Document_Type"] == "INVOICE"]
    # Index parent invoices by (GSTIN, cleaned invoice number) for O(1) lookup.
    parent_index = {}
    for _, r in invoice_rows.iterrows():
        key = (r["Parsed_GSTIN"], r["Parsed_Inv"])
        # If duplicates exist, keep the first — duplicates are already
        # flagged separately by the BOOKS_DUPLICATE data-quality check.
        parent_index.setdefault(key, r)

    note_mask = df["Parsed_Document_Type"].isin(NOTE_TYPES)
    for idx, row in df[note_mask].iterrows():
        key = (row["Parsed_GSTIN"], row["Parsed_Original_Inv"])
        parent = parent_index.get(key)

        tax = row["Parsed_Total_Tax"] if pd.notna(row["Parsed_Total_Tax"]) else 0.0
        if row["Parsed_Document_Type"] in CREDIT_TYPES:
            df.at[idx, "Net_ITC_Adjustment"] = -tax
        elif row["Parsed_Document_Type"] in DEBIT_TYPES:
            df.at[idx, "Net_ITC_Adjustment"] = tax

        if parent is not None and row["Parsed_Original_Inv"] != "UNKNOWN_INV":
            df.at[idx, "Linked_Parent_RowID"] = parent["Row_ID"]
            df.at[idx, "Linked_Parent_Invoice"] = parent["Raw_Invoice_Number"]
            df.at[idx, "CN_DN_Link_Status"] = "LINKED"
        else:
            df.at[idx, "CN_DN_Link_Status"] = "PARENT_NOT_FOUND"

    return df


def build_net_itc_summary(linked_books_df):
    """
    Vendor-wise Net ITC = sum(matched invoice tax) + sum(Net_ITC_Adjustment
    from linked notes). Unlinked notes are EXCLUDED from the net figure and
    surfaced separately, since their impact can't be safely netted without
    a confirmed parent.
    """
    df = linked_books_df

    invoice_tax = (
        df[df["Parsed_Document_Type"] == "INVOICE"]
        .groupby("Parsed_GSTIN")["Parsed_Total_Tax"].sum(min_count=1)
    )
    linked_note_adj = (
        df[(df["Parsed_Document_Type"].isin(NOTE_TYPES)) & (df["CN_DN_Link_Status"] == "LINKED")]
        .groupby("Parsed_GSTIN")["Net_ITC_Adjustment"].sum(min_count=1)
    )
    unlinked_notes = (
        df[(df["Parsed_Document_Type"].isin(NOTE_TYPES)) & (df["CN_DN_Link_Status"] == "PARENT_NOT_FOUND")]
        .groupby("Parsed_GSTIN")
        .agg(Unlinked_Note_Count=("Row_ID", "count"),
             Unlinked_Note_Value=("Net_ITC_Adjustment", lambda x: x.fillna(0).sum()))
    )

    summary = pd.concat(
        [invoice_tax.rename("Invoice_Tax"), linked_note_adj.rename("Linked_Note_Adjustment")],
        axis=1
    ).fillna(0)
    summary["Net_ITC"] = summary["Invoice_Tax"] + summary["Linked_Note_Adjustment"]
    summary = summary.join(unlinked_notes, how="left").fillna(
        {"Unlinked_Note_Count": 0, "Unlinked_Note_Value": 0}
    )
    return summary.reset_index().rename(columns={"Parsed_GSTIN": "GSTIN"})


# ============================================================
# 7C. CROSS-MONTH CARRY-FORWARD (LOCAL SQLITE PERSISTENCE)
# ============================================================
# Roadmap item #1 — the "vendor filed late" problem. Uses Python's builtin
# sqlite3 (no new infra/hosting dependency) to remember invoices that were
# MISSING this period, so that next period's run can auto-resolve them the
# moment the vendor's late filing shows up in a later GSTR-2B.
#
# HONEST LIMITATION: on ephemeral hosting (e.g. some free-tier redeploys),
# the local .db file can reset when the app container restarts. This is a
# real MVP-grade persistence layer, not a substitute for a hosted database —
# see the note printed in the UI below.

import sqlite3

DB_PATH = "gstr2b_ledger.db"


def init_ledger_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_itc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_label TEXT,
            gstin TEXT,
            parsed_invoice TEXT,
            raw_invoice_number TEXT,
            vendor_name TEXT,
            taxable_value REAL,
            total_tax REAL,
            saved_at TEXT,
            UNIQUE(period_label, gstin, parsed_invoice)
        )
    """)
    conn.commit()
    conn.close()


def save_missing_to_ledger(books_df, period_label, db_path=DB_PATH):
    """Persists every currently-MISSING books row so a future period can
    auto-resolve it. Safe to call multiple times for the same period —
    duplicates are ignored via the UNIQUE constraint."""
    init_ledger_db(db_path)
    missing = books_df[books_df["Match_Category"] == "MISSING"]
    conn = sqlite3.connect(db_path)
    saved = 0
    for _, r in missing.iterrows():
        if r["Parsed_GSTIN"] == "UNKNOWN_GSTIN" or r["Parsed_Inv"] == "UNKNOWN_INV":
            continue  # never persist unidentifiable rows — nothing to match on later
        try:
            conn.execute(
                """INSERT OR IGNORE INTO pending_itc
                   (period_label, gstin, parsed_invoice, raw_invoice_number,
                    vendor_name, taxable_value, total_tax, saved_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (period_label, r["Parsed_GSTIN"], r["Parsed_Inv"],
                 str(r["Raw_Invoice_Number"]), str(r["Raw_Vendor_Name"]),
                 float(r["Parsed_Taxable_Value"]) if pd.notna(r["Parsed_Taxable_Value"]) else None,
                 float(r["Parsed_Total_Tax"]) if pd.notna(r["Parsed_Total_Tax"]) else None,
                 datetime.now().isoformat())
            )
            saved += conn.total_changes
        except sqlite3.Error:
            continue
    conn.commit()
    conn.close()
    return saved


def get_pending_ledger(db_path=DB_PATH):
    init_ledger_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM pending_itc ORDER BY saved_at DESC", conn
    )
    conn.close()
    return df


def resolve_carry_forward(unmatched_2b_df, current_period_label, db_path=DB_PATH):
    """
    THE key carry-forward check: takes THIS period's leftover GSTR-2B rows
    (2B entries no current-period Books row consumed) and checks each one
    against invoices logged as MISSING in an EARLIER period's ledger.

    A hit means: the vendor filed late — an invoice your books recorded
    last period (and which was missing back then) has now shown up in this
    period's GSTR-2B. That resolves the OLD "missing" flag, even though
    the original books row itself lived in a prior period's upload and
    isn't part of this run's DataFrame.

    Returns a DataFrame of resolved carry-forward matches (for display /
    export) and removes each resolved entry from the pending ledger.
    """
    init_ledger_db(db_path)
    conn = sqlite3.connect(db_path)
    resolved_rows = []

    for _, row in unmatched_2b_df.iterrows():
        cur = conn.execute(
            """SELECT id, period_label, raw_invoice_number, vendor_name,
                      taxable_value, total_tax
               FROM pending_itc
               WHERE gstin = ? AND parsed_invoice = ? AND period_label != ?
               ORDER BY saved_at ASC LIMIT 1""",
            (row["Parsed_GSTIN"], row["Parsed_Inv"], current_period_label)
        )
        hit = cur.fetchone()
        if hit:
            pending_id, origin_period, orig_inv_no, vendor, taxable, tax = hit
            resolved_rows.append({
                "GSTIN": row["Parsed_GSTIN"],
                "Vendor_Name": vendor,
                "Invoice_Number": orig_inv_no,
                "Originally_Missing_In": origin_period,
                "Resolved_In": current_period_label,
                "Taxable_Value": taxable,
                "Total_Tax": tax,
            })
            conn.execute("DELETE FROM pending_itc WHERE id = ?", (pending_id,))

    conn.commit()
    conn.close()

    if not resolved_rows:
        return pd.DataFrame(columns=[
            "GSTIN", "Vendor_Name", "Invoice_Number", "Originally_Missing_In",
            "Resolved_In", "Taxable_Value", "Total_Tax"
        ])
    return pd.DataFrame(resolved_rows)


# ============================================================
# 8. VENDOR FOLLOW-UP
# ============================================================

def build_vendor_followup_table(res_df):
    actionable = [
        "MISSING",
        "TAX_MISMATCH",
        "DATE_MISMATCH",
    ]

    flagged = res_df[
        res_df["Match_Category"].isin(actionable)
        & (res_df["Parsed_GSTIN"] != "UNKNOWN_GSTIN")
    ].copy()

    if flagged.empty:
        return pd.DataFrame(columns=[
            "GSTIN", "Vendor_Name", "Invoice_Count",
            "ITC_At_Risk", "Invoice_List", "Category_Breakdown"
        ])

    def fmt_invoices(group):
        return "; ".join(
            group["Raw_Invoice_Number"]
            .astype(str)
            .fillna("(no number)")
        )

    def fmt_breakdown(group):
        return ", ".join(
            f"{cat}: {n}"
            for cat, n in group["Match_Category"].value_counts().items()
        )

    grouped = flagged.groupby("Parsed_GSTIN").apply(
        lambda g: pd.Series({
            "GSTIN": (
                g["Raw_GSTIN"].iloc[0]
                if pd.notna(g["Raw_GSTIN"].iloc[0])
                else g["Parsed_GSTIN"].iloc[0]
            ),
            "Vendor_Name": (
                g["Raw_Vendor_Name"].dropna().iloc[0]
                if g["Raw_Vendor_Name"].notna().any()
                else "(name not on file)"
            ),
            "Invoice_Count": len(g),
            "ITC_At_Risk": round(
                g["Parsed_Total_Tax"].fillna(0).sum(), 2
            ),
            "Invoice_List": fmt_invoices(g),
            "Category_Breakdown": fmt_breakdown(g),
        })
    ).reset_index(drop=True)

    return grouped.sort_values(
        "ITC_At_Risk",
        ascending=False
    )


def generate_email_template(
    vendor_name, gstin, invoice_list_str,
    itc_at_risk, period_label
):
    return f"""Subject: Action Needed — Invoices Missing/Mismatched in GSTR-2B ({period_label})

Dear {vendor_name or "Vendor"},

During our GST reconciliation for {period_label}, we found the following invoice(s) under GSTIN {gstin} showing as missing or mismatched in our GSTR-2B:

{invoice_list_str}

ITC associated with these exceptions is approximately ₹{itc_at_risk:,.2f}, subject to reconciliation and tax review.

Could you please check the filing details for the relevant period and confirm whether any correction or amendment is required?

Thank you,
{{Your Name / Company}}
"""


def generate_whatsapp_template(
    vendor_name, gstin, invoice_count,
    itc_at_risk, period_label
):
    return (
        f"Hi {vendor_name or 'there'}, regarding GST reconciliation for "
        f"{period_label}: {invoice_count} invoice(s) under GSTIN {gstin} "
        f"are showing as missing/mismatched in our GSTR-2B. Approx. "
        f"ITC involved is ₹{itc_at_risk:,.2f}. Could you please check "
        f"the filing details and let us know if any correction/amendment "
        f"is required? Thanks!"
    )


# ============================================================
# 9. TESTABLE GOLDEN-DATA HELPERS
# ============================================================

def run_golden_tests():
    """Small deterministic regression suite for the critical V5.1 fixes."""
    tol = {
        "taxable": {"abs": 1.0, "rel": 0.5},
        "comp": {"abs": 1.0, "rel": 0.5},
        "total": {"abs": 1.0, "rel": 0.5},
        "date_days": 2,
    }

    weights = {
        "inv": 30, "date": 20, "taxable": 10, "total": 15,
        "igst": 10, "cgst": 5, "sgst": 5, "cess": 5
    }

    gates = {
        "min_inv_sim": 70,
        "min_score": 75,
        "min_margin": 8,
        "min_fin_similarity": 70,
        "high_confidence_score": 90,
    }

    def make(gstin, inv, date, taxable, igst, cgst, sgst, cess=0, total=None):
        return {
            "GSTIN": gstin,
            "Vendor_Name": "Vendor",
            "Invoice_Number": inv,
            "Invoice_Date": date,
            "Taxable_Value": taxable,
            "IGST": igst,
            "CGST": cgst,
            "SGST": sgst,
            "CESS": cess,
            "Total_Tax": (
                total if total is not None
                else igst + cgst + sgst + cess
            ),
            "Document_Type": "INVOICE",
        }

    cases = [
        ("exact_match", make("06ABCDE1234F1Z5","INV001","01-04-2026",1000,180,0,0),
         make("06ABCDE1234F1Z5","INV001","01-04-2026",1000,180,0,0)),
        ("normalized_invoice", make("06ABCDE1234F1Z5","INV-001/26","01-04-2026",1000,180,0,0),
         make("06ABCDE1234F1Z5","INV00126","01-04-2026",1000,180,0,0)),
        ("amount_mismatch", make("06ABCDE1234F1Z5","INV002","01-04-2026",1000,180,0,0),
         make("06ABCDE1234F1Z5","INV002","01-04-2026",1100,198,0,0)),
        ("zero_tax", make("06ABCDE1234F1Z5","INV003","01-04-2026",1000,0,0,0),
         make("06ABCDE1234F1Z5","INV003","01-04-2026",1000,0,0,0)),
    ]

    rows = []
    for name, b, g in cases:
        bdf = pd.DataFrame([b])
        gdf = pd.DataFrame([g])
        try:
            res, _, _ = reconcile_engine(
                bdf, gdf, tol, weights, gates
            )
            category = res.iloc[0]["Match_Category"]
            rows.append({
                "Test": name,
                "Result": "PASS" if category in {
                    "EXACT", "NORMALIZED", "TAX_MISMATCH"
                } else "REVIEW",
                "Category": category,
            })
        except Exception as e:
            rows.append({
                "Test": name,
                "Result": "FAIL",
                "Category": str(e),
            })

    return pd.DataFrame(rows)


# ============================================================
# 10. STREAMLIT UI
# ============================================================

with st.sidebar:
    st.header("⚙️ Safety Settings")

    st.subheader("1. Financial tolerances")
    c1, c2 = st.columns(2)
    taxable_abs = c1.number_input("Taxable Abs (₹)", value=1.0, min_value=0.0)
    taxable_rel = c2.number_input("Taxable Rel (%)", value=0.5, min_value=0.0)
    comp_abs = c1.number_input("Component Abs (₹)", value=1.0, min_value=0.0)
    comp_rel = c2.number_input("Component Rel (%)", value=0.5, min_value=0.0)
    tot_abs = c1.number_input("Total Abs (₹)", value=1.0, min_value=0.0)
    tot_rel = c2.number_input("Total Rel (%)", value=0.5, min_value=0.0)
    date_tol = st.number_input("Date Tolerance (Days)", value=2, min_value=0)

    tols = {
        "taxable": {"abs": taxable_abs, "rel": taxable_rel},
        "comp": {"abs": comp_abs, "rel": comp_rel},
        "total": {"abs": tot_abs, "rel": tot_rel},
        "date_days": date_tol,
    }

    st.subheader("2. Matching gates")
    min_inv = st.slider("Min Invoice Similarity", 0, 100, 70)
    min_score = st.slider("Min Overall Score", 0, 100, 75)
    min_margin = st.slider("Min Score Margin", 0, 30, 8)
    min_fin_similarity = st.slider("Min Financial Similarity", 0, 100, 70)
    high_confidence_score = st.slider("High Confidence Score", 0, 100, 90)

    gates = {
        "min_inv_sim": min_inv,
        "min_score": min_score,
        "min_margin": min_margin,
        "min_fin_similarity": min_fin_similarity,
        "high_confidence_score": high_confidence_score,
    }

    st.subheader("3. Scoring weights")
    w_inv = st.slider("Invoice", 0, 100, 30)
    w_date = st.slider("Date", 0, 100, 20)
    w_taxval = st.slider("Taxable", 0, 100, 10)
    w_tot = st.slider("Total Tax", 0, 100, 15)
    w_igst = st.slider("IGST", 0, 100, 10)
    w_cgst = st.slider("CGST", 0, 100, 5)
    w_sgst = st.slider("SGST", 0, 100, 5)
    w_cess = st.slider("CESS", 0, 100, 5)

    weights = {
        "inv": w_inv,
        "date": w_date,
        "taxable": w_taxval,
        "total": w_tot,
        "igst": w_igst,
        "cgst": w_cgst,
        "sgst": w_sgst,
        "cess": w_cess,
    }

    st.divider()
    st.subheader("4. Return period & carry-forward")
    period_label = st.text_input(
        "This period's label (e.g. 'Aug-2026')",
        value=datetime.now().strftime("%b-%Y")
    )
    enable_carry_forward = st.checkbox(
        "Check this period's leftover GSTR-2B against invoices logged "
        "missing in earlier periods (late vendor filings)",
        value=True
    )
    st.caption(
        "⚠️ Uses a local SQLite file for continuity across runs. On some "
        "ephemeral free-tier hosts this file can reset on redeploy — for "
        "guaranteed continuity, point DB_PATH at a hosted database instead."
    )

    st.divider()
    file_books = st.file_uploader(
        "Upload Purchase Books (CSV/XLSX)",
        type=["csv", "xlsx"]
    )
    file_2b = st.file_uploader(
        "Upload GSTR-2B (JSON/PDF/CSV/XLSX)",
        type=["json", "pdf", "csv", "xlsx"]
    )

if file_books and file_2b and sum(weights.values()) > 0:
    try:
        b_df = (
            pd.read_csv(file_books)
            if file_books.name.lower().endswith(".csv")
            else pd.read_excel(file_books)
        )

        warns = []
        suffix = file_2b.name.lower()

        if suffix.endswith(".json"):
            g_df, warns = parse_gstr2b_json(file_2b)
        elif suffix.endswith(".pdf"):
            g_df, warns = parse_gstr2b_pdf(file_2b)
        elif suffix.endswith(".csv"):
            g_df = pd.read_csv(file_2b)
        else:
            g_df = pd.read_excel(file_2b)

        for w in warns:
            st.warning(w)

        if g_df.empty:
            st.error("No usable data could be extracted from GSTR-2B.")
        else:
            # PDF safety barrier.
            if suffix.endswith(".pdf"):
                st.info(
                    "PDF safety check: review the parsed preview before "
                    "reconciliation. PDF extraction is best-effort."
                )
                st.dataframe(g_df.head(20), use_container_width=True)

                confirm_pdf = st.checkbox(
                    "I reviewed the PDF-extracted rows and want to reconcile them."
                )
            else:
                confirm_pdf = True

            if confirm_pdf:
                with st.spinner("Running V6 reconciliation..."):
                    res, unmatched, full_2b = reconcile_engine(
                        b_df, g_df, tols, weights, gates
                    )

                    carried_forward_df = pd.DataFrame()
                    if enable_carry_forward:
                        carried_forward_df = resolve_carry_forward(
                            unmatched, period_label
                        )

                    res = link_credit_debit_notes(res)
                    net_itc_summary = build_net_itc_summary(res)

                if not carried_forward_df.empty:
                    st.success(
                        f"🗓️ {len(carried_forward_df)} invoice(s) previously "
                        "logged as missing were resolved this period — the "
                        "vendor filed late. See the Carry-Forward Ledger tab."
                    )

                if st.button("💾 Save this period's MISSING invoices to the carry-forward ledger"):
                    saved_n = save_missing_to_ledger(res, period_label)
                    st.info(
                        f"Logged missing invoices for period '{period_label}'. "
                        "They'll be auto-checked against future GSTR-2B uploads."
                    )

                st.subheader("📈 Reconciliation Dashboard")

                matched_count = len(
                    res[res["Match_Category"].isin(MATCHED_CATEGORIES)]
                )
                exception_count = len(res) - matched_count

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Books", len(res))
                c2.metric("2B Records", len(full_2b))
                c3.metric("Matched", matched_count)
                c4.metric("Exceptions", exception_count)
                c5.metric(
                    "ITC Exception Value",
                    f"₹{res.loc[~res['Match_Category'].isin(MATCHED_CATEGORIES), 'Parsed_Total_Tax'].fillna(0).sum():,.2f}"
                )

                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
                    "⚠️ Review Queue",
                    "📋 Full Audit",
                    "🏢 Vendor Summary",
                    "📤 Export",
                    "📣 Vendor Follow-up",
                    "🧪 Regression Tests",
                    "🔗 CN/DN Linking",
                    "🔁 RCM Invoices",
                    "🗓️ Carry-Forward Ledger",
                ])

                with tab1:
                    review = res[
                        ~res["Match_Category"].isin(MATCHED_CATEGORIES)
                    ][[
                        "Raw_GSTIN",
                        "Raw_Invoice_Number",
                        "Status",
                        "Confidence_Score",
                        "Match_Reason",
                    ]]
                    st.dataframe(review, use_container_width=True)

                with tab2:
                    st.dataframe(
                        res[[
                            "Raw_GSTIN",
                            "Raw_Invoice_Number",
                            "Parsed_Document_Type",
                            "Status",
                            "Matched_2B_Invoice",
                            "Confidence_Score",
                            "Score_Margin",
                            "Match_Reason",
                        ]],
                        use_container_width=True
                    )

                with tab3:
                    v_sum = res.groupby("Parsed_GSTIN").agg(
                        Count=("Row_ID", "count"),
                        Matched=("Match_Category", lambda x:
                            x.isin(MATCHED_CATEGORIES).sum()),
                        Exceptions=("Match_Category", lambda x:
                            (~x.isin(MATCHED_CATEGORIES)).sum()),
                        ITC=("Parsed_Total_Tax", lambda x:
                            x.fillna(0).sum()),
                    ).reset_index()
                    st.dataframe(v_sum, use_container_width=True)

                with tab4:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        res.to_excel(
                            writer, sheet_name="Reconciliation", index=False
                        )
                        res[
                            res["Match_Category"] == "MISSING"
                        ].to_excel(
                            writer, sheet_name="Missing_in_2B", index=False
                        )
                        res[
                            res["Match_Category"].isin([
                                "TAX_MISMATCH",
                                "DATE_MISMATCH",
                                "MULTIPLE_CANDIDATES",
                                "DATA_QUALITY",
                            ])
                        ].to_excel(
                            writer, sheet_name="Review_Queue", index=False
                        )
                        unmatched.to_excel(
                            writer, sheet_name="Unmatched_2B", index=False
                        )

                    st.download_button(
                        "📥 Download V5.1 Audit Report",
                        data=buf.getvalue(),
                        file_name=(
                            f"V5.1_Reconciliation_"
                            f"{datetime.now().strftime('%Y%m%d')}.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                        type="primary"
                    )

                with tab5:
                    followup_df = build_vendor_followup_table(res)

                    if followup_df.empty:
                        st.success(
                            "No vendors currently require follow-up."
                        )
                    else:
                        st.dataframe(
                            followup_df,
                            use_container_width=True
                        )

                        selected_gstin = st.selectbox(
                            "Generate message for GSTIN:",
                            followup_df["GSTIN"].tolist()
                        )
                        row = followup_df[
                            followup_df["GSTIN"] == selected_gstin
                        ].iloc[0]

                        period_label = st.text_input(
                            "Return period",
                            value="the current period"
                        )

                        msg_type = st.radio(
                            "Template",
                            ["Email", "WhatsApp"],
                            horizontal=True
                        )

                        if msg_type == "Email":
                            text = generate_email_template(
                                row["Vendor_Name"],
                                row["GSTIN"],
                                row["Invoice_List"],
                                row["ITC_At_Risk"],
                                period_label,
                            )
                        else:
                            text = generate_whatsapp_template(
                                row["Vendor_Name"],
                                row["GSTIN"],
                                row["Invoice_Count"],
                                row["ITC_At_Risk"],
                                period_label,
                            )

                        st.text_area(
                            "Copy this message:",
                            value=text,
                            height=280
                        )

                with tab6:
                    st.write(
                        "Run the deterministic regression suite against "
                        "the critical safety cases."
                    )
                    if st.button("Run Golden Tests"):
                        st.dataframe(
                            run_golden_tests(),
                            use_container_width=True
                        )

                with tab7:
                    st.caption(
                        "Credit/Debit Notes linked to their original invoice "
                        "(within this Books upload), with the net ITC impact "
                        "of each note. Unlinked notes need manual review — "
                        "their impact is NOT netted automatically."
                    )
                    notes_view = res[res["Parsed_Document_Type"].isin(NOTE_TYPES)][[
                        "Raw_GSTIN", "Raw_Vendor_Name", "Raw_Invoice_Number",
                        "Parsed_Document_Type", "Raw_Original_Invoice_Number",
                        "Linked_Parent_Invoice", "CN_DN_Link_Status",
                        "Net_ITC_Adjustment",
                    ]]
                    if notes_view.empty:
                        st.success("No credit/debit notes in this upload.")
                    else:
                        st.dataframe(notes_view, use_container_width=True)

                        unlinked_n = (notes_view["CN_DN_Link_Status"] == "PARENT_NOT_FOUND").sum()
                        if unlinked_n:
                            st.warning(
                                f"{unlinked_n} note(s) could not be linked to a "
                                "parent invoice — check Original_Invoice_Number "
                                "accuracy on these notes."
                            )

                    st.subheader("Net ITC by Vendor (Invoices − Linked Notes)")
                    st.dataframe(net_itc_summary, use_container_width=True)

                with tab8:
                    st.caption(
                        "Reverse Charge Mechanism invoices are isolated here "
                        "because ITC on them is only claimable after the "
                        "recipient self-pays the tax — they should never be "
                        "silently pooled into normal B2B claimable-ITC totals."
                    )
                    rcm_view = res[res["Parsed_RCM"] == "YES"][[
                        "Raw_GSTIN", "Raw_Vendor_Name", "Raw_Invoice_Number",
                        "Status", "Parsed_Total_Tax", "Match_Reason",
                    ]]
                    if rcm_view.empty:
                        st.info("No invoices flagged Reverse Charge = Y in this upload.")
                    else:
                        st.dataframe(rcm_view, use_container_width=True)
                        st.metric(
                            "Total RCM Tax (self-payable, not standard ITC pool)",
                            f"₹{rcm_view['Parsed_Total_Tax'].fillna(0).sum():,.2f}"
                        )

                with tab9:
                    st.caption(
                        "Invoices logged as MISSING in a prior period, and "
                        "whether this period's GSTR-2B finally resolved them."
                    )
                    if not carried_forward_df.empty:
                        st.subheader("✅ Resolved this run (vendor filed late)")
                        st.dataframe(carried_forward_df, use_container_width=True)
                    else:
                        st.write("No carry-forward resolutions this run.")

                    st.subheader("⏳ Still pending across all periods")
                    pending_now = get_pending_ledger()
                    if pending_now.empty:
                        st.success("Ledger is empty — nothing carried forward.")
                    else:
                        st.dataframe(pending_now, use_container_width=True)

    except Exception as e:
        st.error(f"Execution error: {e}")
else:
    st.info(
        "Upload both Purchase Books and GSTR-2B to start. "
        "For highest reliability, prefer structured JSON/XLSX over PDF."
    )
