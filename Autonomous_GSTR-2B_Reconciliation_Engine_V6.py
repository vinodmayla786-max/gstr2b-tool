import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
from datetime import datetime
from thefuzz import fuzz
import pdfplumber
import sqlite3
import uuid

# ============================================================
# AUTONOMOUS GSTR-2B RECONCILIATION ENGINE — V6.1 (Bugfix)
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

FINANCIAL_COLS = ["Taxable_Value", "IGST", "CGST", "SGST", "CESS", "Total_Tax"]

MATCHED_CATEGORIES = {"EXACT", "NORMALIZED", "HIGH_CONFIDENCE", "PROBABLE", "CARRIED_FORWARD_MATCH"}

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

NOTE_TYPES = {"CREDIT_NOTE", "DEBIT_NOTE", "AMENDED_CREDIT_NOTE", "AMENDED_DEBIT_NOTE"}
CREDIT_TYPES = {"CREDIT_NOTE", "AMENDED_CREDIT_NOTE"}
DEBIT_TYPES = {"DEBIT_NOTE", "AMENDED_DEBIT_NOTE"}

DB_PATH = "gstr2b_ledger.db"

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
    if pd.isna(val) or str(val).strip() == "": return "UNKNOWN_INV"
    return re.sub(r"[^A-Z0-9]", "", str(val).upper())

def normalize_gstin(val):
    if pd.isna(val) or str(val).strip() == "": return "UNKNOWN_GSTIN"
    return re.sub(r"\s+", "", str(val).upper().strip())

def validate_gstin(val):
    if pd.isna(val) or not isinstance(val, str): return False
    pattern = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
    return bool(re.fullmatch(pattern, val.strip().upper()))

def infer_document_type(source_section="", explicit_type=""):
    text = f"{source_section} {explicit_type}".upper()
    explicit_upper = str(explicit_type).upper().strip()

    if explicit_upper in {"C", "CR", "CREDIT", "CREDIT NOTE"}: return "CREDIT_NOTE"
    if explicit_upper in {"D", "DR", "DEBIT", "DEBIT NOTE"}: return "DEBIT_NOTE"
    if "CDNRA" in text or "AMENDED CREDIT" in text: return "AMENDED_CREDIT_NOTE"
    if "CDNR" in text:
        if "DEBIT" in text: return "DEBIT_NOTE"
        if "CREDIT" in text: return "CREDIT_NOTE"
        return "UNKNOWN"
    if "B2BA" in text or "AMEND" in text: return "AMENDMENT"
    if "DEBIT" in text: return "DEBIT_NOTE"
    if "CREDIT" in text: return "CREDIT_NOTE"
    if "INVOICE" in text or "B2B" in text: return "INVOICE"
    return "UNKNOWN"

def parse_date_safe(value):
    if pd.isna(value) or str(value).strip() == "": return pd.NaT
    return pd.to_datetime(value, dayfirst=True, errors="coerce")

def within_tolerance(ref, cand, abs_tol, rel_tol_pct):
    if pd.isna(ref) or pd.isna(cand): return False
    diff = abs(float(ref) - float(cand))
    if diff <= abs_tol: return True
    base = max(abs(float(ref)), abs(float(cand)), 0.0001)
    return ((diff / base) * 100.0) <= rel_tol_pct

def calculate_financial_similarity(ref, cand, abs_tol, rel_tol_pct):
    if pd.isna(ref) or pd.isna(cand): return np.nan
    ref, cand = float(ref), float(cand)
    if ref == 0.0 and cand == 0.0: return 100.0
    diff = abs(ref - cand)
    if diff <= abs_tol: return 100.0
    base = max(abs(ref), abs(cand), 0.0001)
    rel_diff_pct = (diff / base) * 100.0
    if rel_diff_pct <= rel_tol_pct: return 100.0
    return max(0.0, 100.0 - ((rel_diff_pct - rel_tol_pct) * 2))

# ============================================================
# 3. JSON INGESTION (BUGFIXED FOR V6.1)
# ============================================================

def _find_all_sections(node, section_key):
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
    trade_name = supplier_block.get("trdnm", supplier_block.get("trdname", ""))

    doc_list = supplier_block.get("inv") or supplier_block.get("nt") or []

    for doc in doc_list:
        inum = doc.get("inum", doc.get("ntnum", ""))
        idt = doc.get("idt", doc.get("ntdt", ""))

        explicit_type = doc.get("typ") or doc.get("ntty") or doc.get("type") or ""
        original_invoice_number = doc.get("oinum") or doc.get("ontnum") or doc.get("original_inum") or ""
        original_invoice_date = doc.get("oidt") or doc.get("ontdt") or doc.get("original_idt") or ""

        # FIX: Handle both "itms" and "items" keys from JSON
        items_list = doc.get("itms") or doc.get("items") or []

        component_values = {"Taxable_Value": [], "IGST": [], "CGST": [], "SGST": [], "CESS": []}

        if items_list:
            for it in items_list:
                # FIX: Handle nested "itm_det" OR flattened properties
                d = it.get("itm_det", it)
                
                component_values["Taxable_Value"].append(_safe_float(d, "txval"))
                
                igst_val = _safe_float(d, "iamt") if not pd.isna(_safe_float(d, "iamt")) else _safe_float(d, "igst")
                component_values["IGST"].append(igst_val)
                
                cgst_val = _safe_float(d, "camt") if not pd.isna(_safe_float(d, "camt")) else _safe_float(d, "cgst")
                component_values["CGST"].append(cgst_val)
                
                sgst_val = _safe_float(d, "samt") if not pd.isna(_safe_float(d, "samt")) else _safe_float(d, "sgst")
                component_values["SGST"].append(sgst_val)
                
                cess_val = _safe_float(d, "csamt") if not pd.isna(_safe_float(d, "csamt")) else _safe_float(d, "cess")
                component_values["CESS"].append(cess_val)

            vals = {}
            for col, arr in component_values.items():
                vals[col] = np.nan if any(pd.isna(x) for x in arr) else float(sum(arr))
        else:
            taxable = _safe_float(doc, "txval")
            if pd.isna(taxable): taxable = _safe_float(doc, "val")
            vals = {
                "Taxable_Value": taxable,
                "IGST": _safe_float(doc, "iamt") if not pd.isna(_safe_float(doc, "iamt")) else _safe_float(doc, "igst"),
                "CGST": _safe_float(doc, "camt") if not pd.isna(_safe_float(doc, "camt")) else _safe_float(doc, "cgst"),
                "SGST": _safe_float(doc, "samt") if not pd.isna(_safe_float(doc, "samt")) else _safe_float(doc, "sgst"),
                "CESS": _safe_float(doc, "csamt") if not pd.isna(_safe_float(doc, "csamt")) else _safe_float(doc, "cess"),
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
            "Reverse_Charge": doc.get("rchrg") or doc.get("rev") or "", # FIX: handle 'rev'
        })
    return rows

def parse_gstr2b_json(uploaded_file):
    warnings = []
    raw = uploaded_file.read()
    if isinstance(raw, bytes): raw = raw.decode("utf-8", errors="replace")

    try: payload = json.loads(raw)
    except json.JSONDecodeError as e: raise ValueError(f"File is not valid JSON: {e}")

    sections = {"b2b": "B2B", "b2ba": "B2B_AMENDED", "cdnr": "CDNR", "cdnra": "CDNR_AMENDED"}
    all_rows = []
    
    for section_key, section_label in sections.items():
        section_lists = _find_all_sections(payload, section_key)
        for section_list in section_lists:
            for supplier_block in section_list:
                all_rows.extend(_extract_invoice_rows(supplier_block, section_label))

    if not all_rows:
        warnings.append("No recognized B2B/B2BA/CDNR/CDNRA invoice sections found.")
        return pd.DataFrame(), warnings

    df = pd.DataFrame(all_rows)
    return df, warnings

# ============================================================
# 4. PDF INGESTION
# ============================================================
def parse_gstr2b_pdf(uploaded_file):
    warnings = []
    all_rows = []
    header_aliases = {
        "gstin": "GSTIN", "trade": "Vendor_Name", "invoice no": "Invoice_Number",
        "invoice number": "Invoice_Number", "invoice date": "Invoice_Date",
        "taxable value": "Taxable_Value", "integrated tax": "IGST",
        "central tax": "CGST", "state/ut tax": "SGST", "cess": "CESS",
    }

    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2: continue
                header_row = [str(c or "").strip().lower() for c in table[0]]
                col_map = {}
                for i, h in enumerate(header_row):
                    for alias, standard in header_aliases.items():
                        if alias in h:
                            col_map[i] = standard
                            break
                if not col_map: continue
                for row in table[1:]:
                    if not row or all(c is None or str(c).strip() == "" for c in row): continue
                    record = {}
                    for i, val in enumerate(row):
                        if i in col_map: record[col_map[i]] = str(val).strip() if val else ""
                    if record.get("GSTIN") or record.get("Invoice_Number"):
                        record["Source_Page"] = page_num
                        record["Source_Section"] = "PDF"
                        record["Document_Type"] = "UNKNOWN"
                        all_rows.append(record)

    if not all_rows:
        warnings.append("No invoice tables detected in PDF. Prefer JSON/XLSX where available.")
        return pd.DataFrame(), warnings

    df = pd.DataFrame(all_rows)
    for col in ["Taxable_Value", "IGST", "CGST", "SGST", "CESS"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")

    tax_cols = [c for c in ["IGST", "CGST", "SGST", "CESS"] if c in df.columns]
    if tax_cols: df["Total_Tax"] = df[tax_cols].sum(axis=1, min_count=len(tax_cols))
    if "Invoice_Date" in df.columns: df["Invoice_Date"] = pd.to_datetime(df["Invoice_Date"], dayfirst=True, errors="coerce")
    warnings.append("Extracted rows from PDF. PDF data is marked as best-effort.")
    return df, warnings

# ============================================================
# 5. COLUMN MAPPING / PREPROCESSING
# ============================================================

def map_columns(df):
    mapped_df = df.copy()
    col_map = {}
    ambiguous = []
    normalized_columns = {col: str(col).strip().lower() for col in mapped_df.columns}

    for standard, aliases in COLUMN_ALIASES.items():
        alias_set = {a.lower() for a in aliases}
        found = [col for col, normalized in normalized_columns.items() if normalized in alias_set]
        if len(found) == 1: col_map[found[0]] = standard
        elif len(found) > 1: ambiguous.append(standard)

    mapped_df = mapped_df.rename(columns=col_map)
    missing = []
    for req in COLUMN_ALIASES:
        if req not in mapped_df.columns:
            mapped_df[req] = np.nan
            missing.append(req)
    return mapped_df, ambiguous, missing

def preprocess_data(df, source_name, tolerances):
    df, ambig, missing_cols = map_columns(df)
    for col in COLUMN_ALIASES: df[f"Raw_{col}"] = df[col]

    df["Parsed_GSTIN"] = df["GSTIN"].apply(normalize_gstin)
    df["Parsed_Inv"] = df["Invoice_Number"].apply(clean_invoice)
    df["Parsed_Date"] = df["Invoice_Date"].apply(parse_date_safe)

    df["Parsed_Document_Type"] = [infer_document_type(df.iloc[i].get("Source_Section", ""), df.iloc[i].get("Document_Type", "")) for i in range(len(df))]
    df["Parsed_Original_Inv"] = df["Original_Invoice_Number"].apply(clean_invoice)

    def _parse_rcm(val):
        if pd.isna(val) or str(val).strip() == "": return "UNKNOWN"
        v = str(val).strip().upper()
        if v in {"Y", "YES", "TRUE", "1"}: return "YES"
        if v in {"N", "NO", "FALSE", "0"}: return "NO"
        return "UNKNOWN"

    df["Parsed_RCM"] = df["Reverse_Charge"].apply(_parse_rcm)
    df["Data_Quality_Issue"] = ""
    df["Match_Category"] = "PENDING"

    if ambig: df["Data_Quality_Issue"] += f"Ambiguous mapping for {', '.join(ambig)}; "

    for col in FINANCIAL_COLS:
        parsed = pd.to_numeric(df[col], errors="coerce")
        raw = df[f"Raw_{col}"]
        nonblank = raw.notna() & (raw.astype(str).str.strip() != "")
        bad_parse = parsed.isna() & nonblank
        df.loc[bad_parse, "Data_Quality_Issue"] += f"Invalid {col}; "
        df[f"Parsed_{col}"] = parsed
        df.loc[parsed < 0, "Data_Quality_Issue"] += f"Negative {col}; "
        blank = parsed.isna() & ~bad_parse
        # Important: Missing financial info is logged, but doesn't necessarily block matches entirely if Total_Tax is there
        if blank.any(): df.loc[blank, "Data_Quality_Issue"] += f"Missing {col}; "

    if "Total_Tax" in missing_cols:
        components = ["Parsed_IGST", "Parsed_CGST", "Parsed_SGST", "Parsed_CESS"]
        df["Parsed_Total_Tax"] = df[components].sum(axis=1, min_count=4)

    invalid_gstin = ~df["Parsed_GSTIN"].apply(validate_gstin)
    df.loc[invalid_gstin & (df["Parsed_GSTIN"] != "UNKNOWN_GSTIN"), "Data_Quality_Issue"] += "Invalid GSTIN format; "
    df.loc[df["Parsed_GSTIN"] == "UNKNOWN_GSTIN", "Data_Quality_Issue"] += "Missing GSTIN; "
    df.loc[df["Parsed_Inv"] == "UNKNOWN_INV", "Data_Quality_Issue"] += "Missing Invoice No; "
    if "Invoice_Date" not in missing_cols:
        df.loc[df["Parsed_Date"].isna() & df["Raw_Invoice_Date"].notna(), "Data_Quality_Issue"] += "Invalid Date format; "

    component_cols = ["Parsed_IGST", "Parsed_CGST", "Parsed_SGST", "Parsed_CESS"]
    comp_known = df[component_cols].notna().all(axis=1)
    total_known = df["Parsed_Total_Tax"].notna()
    comp_calc = df[component_cols].sum(axis=1, min_count=4)
    tot = df["Parsed_Total_Tax"]
    comp_diff = abs(comp_calc - tot)
    comp_base = np.maximum(np.maximum(abs(comp_calc.fillna(0)), abs(tot.fillna(0))), 0.0001)
    comp_rel_diff_pct = (comp_diff / comp_base) * 100.0

    comp_mismatch = comp_known & total_known & ((comp_diff > tolerances["comp"]["abs"]) & (comp_rel_diff_pct > tolerances["comp"]["rel"]))
    df.loc[comp_mismatch, "Data_Quality_Issue"] += "Tax Composition Mismatch; "

    valid_keys = (df["Parsed_GSTIN"] != "UNKNOWN_GSTIN") & (df["Parsed_Inv"] != "UNKNOWN_INV")
    dupes = df.duplicated(subset=["Parsed_GSTIN", "Parsed_Inv"], keep=False) & valid_keys
    df.loc[dupes, "Data_Quality_Issue"] += f"Duplicate in {source_name}; "

    # Set Match_Category based on Data Quality
    # NOTE: We only want to hard-block matching if GSTIN or Inv is completely missing, 
    # or if it's a Duplicate. Minor issues like "Missing CESS" should still attempt to match.
    critical_errors = df["Data_Quality_Issue"].str.contains("Missing GSTIN|Missing Invoice No|Duplicate|Invalid GSTIN")
    df.loc[critical_errors, "Match_Category"] = "DATA_QUALITY"
    df.loc[dupes, "Match_Category"] = f"{source_name}_DUPLICATE"

    df["Row_ID"] = [f"{source_name}_{i}" for i in range(len(df))]
    return df

# ============================================================
# 6. MATCHING SAFETY FUNCTIONS
# ============================================================

def comparable_financial_scores(b_row, c_row, tolerances):
    scores = []
    for col, tol_key in [("Taxable_Value", "taxable"), ("Total_Tax", "total"), ("IGST", "comp"), ("CGST", "comp"), ("SGST", "comp"), ("CESS", "comp")]:
        score = calculate_financial_similarity(b_row[f"Parsed_{col}"], c_row[f"Parsed_{col}"], tolerances[tol_key]["abs"], tolerances[tol_key]["rel"])
        if not pd.isna(score): scores.append(score)
    return scores

def financial_candidate_gate(b_row, c_row, tolerances, min_fin_similarity=70):
    scores = comparable_financial_scores(b_row, c_row, tolerances)
    if not scores: return False, 0.0, "No comparable financial fields available."
    overall = float(np.mean(scores))
    key_results = []
    for col, tol_key in [("Taxable_Value", "taxable"), ("Total_Tax", "total")]:
        ref, cand = b_row[f"Parsed_{col}"], c_row[f"Parsed_{col}"]
        if pd.notna(ref) and pd.notna(cand):
            key_results.append(within_tolerance(ref, cand, tolerances[tol_key]["abs"], tolerances[tol_key]["rel"]))
    key_pass = any(key_results) if key_results else False
    if key_pass and overall >= min_fin_similarity: return True, round(overall, 2), "Financial proximity gate passed."
    return False, round(overall, 2), "Financial proximity gate failed."

def calculate_fuzzy_score(b_row, g2b_row, weights, tolerances):
    scores = {}
    total_score, active_w = 0.0, 0.0
    if weights["inv"] > 0:
        sim = fuzz.ratio(b_row["Parsed_Inv"], g2b_row["Parsed_Inv"])
        scores["inv"] = sim
        total_score += sim * weights["inv"]
        active_w += weights["inv"]
    if weights["date"] > 0:
        if pd.notnull(b_row["Parsed_Date"]) and pd.notnull(g2b_row["Parsed_Date"]):
            diff = abs((b_row["Parsed_Date"] - g2b_row["Parsed_Date"]).days)
            if diff == 0: sim = 100
            elif diff <= tolerances["date_days"]: sim = 80
            else: sim = max(0, 100 - diff * 2)
            scores["date"] = sim
            total_score += sim * weights["date"]
            active_w += weights["date"]
    for col, w_key, tol_key in [("Taxable_Value", "taxable", "taxable"), ("IGST", "igst", "comp"), ("CGST", "cgst", "comp"), ("SGST", "sgst", "comp"), ("CESS", "cess", "comp"), ("Total_Tax", "total", "total")]:
        if weights[w_key] <= 0: continue
        sim = calculate_financial_similarity(b_row[f"Parsed_{col}"], g2b_row[f"Parsed_{col}"], tolerances[tol_key]["abs"], tolerances[tol_key]["rel"])
        if pd.notna(sim):
            scores[w_key] = sim
            total_score += sim * weights[w_key]
            active_w += weights[w_key]
    if active_w == 0: return 0.0, scores
    return total_score / active_w, scores

def document_type_compatible(b_type, g_type):
    if b_type == "UNKNOWN" or g_type == "UNKNOWN": return True
    if b_type == g_type: return True
    incompatible = {
        ("INVOICE", "CREDIT_NOTE"), ("INVOICE", "DEBIT_NOTE"),
        ("INVOICE", "AMENDED_CREDIT_NOTE"), ("INVOICE", "AMENDED_DEBIT_NOTE"), ("INVOICE", "AMENDMENT"),
        ("CREDIT_NOTE", "INVOICE"), ("DEBIT_NOTE", "INVOICE"),
        ("AMENDED_CREDIT_NOTE", "INVOICE"), ("AMENDED_DEBIT_NOTE", "INVOICE"), ("AMENDMENT", "INVOICE"),
    }
    return (b_type, g_type) not in incompatible

# ============================================================
# 7. RECONCILIATION ENGINE
# ============================================================

def reconcile_engine(books_df, g2b_df, tolerances, weights, gates):
    if sum(weights.values()) <= 0: raise ValueError("Total active weights must be greater than 0.")

    books = preprocess_data(books_df, "BOOKS", tolerances)
    gstr2b = preprocess_data(g2b_df, "GSTR2B", tolerances)

    books["Matched_2B_Invoice"] = None
    books["Matched_2B_RowID"] = None
    books["Confidence_Score"] = np.nan
    books["Match_Reason"] = ""
    books["Score_Margin"] = np.nan

    matched_2b_ids = set()

    # NOTE: Allow rows with minor Data Quality warnings to proceed, 
    # but block rows flagged with critical DATA_QUALITY or BOOKS_DUPLICATE
    valid_b = books[~books["Match_Category"].isin(["DATA_QUALITY", "BOOKS_DUPLICATE"])]
    valid_2b = gstr2b[~gstr2b["Match_Category"].isin(["DATA_QUALITY", "GSTR2B_DUPLICATE"])]

    # --- PASS 1: EXACT IDENTITY ---
    for idx, b_row in valid_b.iterrows():
        b_gstin, b_inv = b_row["Parsed_GSTIN"], b_row["Parsed_Inv"]
        if b_gstin == "UNKNOWN_GSTIN" or b_inv == "UNKNOWN_INV": continue

        candidates = valid_2b[(valid_2b["Parsed_GSTIN"] == b_gstin) & (valid_2b["Parsed_Inv"] == b_inv) & (~valid_2b["Row_ID"].isin(matched_2b_ids))]
        candidates = candidates[candidates.apply(lambda r: document_type_compatible(b_row["Parsed_Document_Type"], r["Parsed_Document_Type"]), axis=1)]

        if len(candidates) > 1:
            books.at[idx, "Match_Category"] = "MULTIPLE_CANDIDATES"
            books.at[idx, "Match_Reason"] = f"{len(candidates)} exact identity candidates found; manual review required."
            continue
        if len(candidates) != 1: continue

        match = candidates.iloc[0]
        tax_match = within_tolerance(b_row["Parsed_Total_Tax"], match["Parsed_Total_Tax"], tolerances["total"]["abs"], tolerances["total"]["rel"])
        taxable_match = within_tolerance(b_row["Parsed_Taxable_Value"], match["Parsed_Taxable_Value"], tolerances["taxable"]["abs"], tolerances["taxable"]["rel"])

        component_results = {}
        for comp in ["IGST", "CGST", "SGST", "CESS"]:
            b_val, g_val = b_row[f"Parsed_{comp}"], match[f"Parsed_{comp}"]
            component_results[comp] = within_tolerance(b_val, g_val, tolerances["comp"]["abs"], tolerances["comp"]["rel"]) if pd.notna(b_val) and pd.notna(g_val) else None
        
        comparable_components = [v for v in component_results.values() if v is not None]
        components_match = bool(comparable_components) and all(comparable_components)
        dates_present = pd.notnull(b_row["Parsed_Date"]) and pd.notnull(match["Parsed_Date"])
        date_diff = abs((b_row["Parsed_Date"] - match["Parsed_Date"]).days) if dates_present else None
        date_match = dates_present and date_diff <= tolerances["date_days"]

        if tax_match and taxable_match and components_match:
            if not dates_present:
                cat, reason = "NORMALIZED", "Identity and financials matched, but dates are unavailable for verification."
            elif not date_match:
                cat, reason = "DATE_MISMATCH", f"Identity and financials matched, but date differs by {date_diff} day(s)."
            else:
                cat = "EXACT" if b_row["Raw_Invoice_Number"] == match["Raw_Invoice_Number"] else "NORMALIZED"
                reason = "GSTIN and invoice identity matched; comparable financial values are within configured tolerances."
        else:
            cat, reason = "TAX_MISMATCH", "Identity matched, but one or more comparable financial values exceed tolerance or cannot be verified."

        books.at[idx, "Match_Category"] = cat
        books.at[idx, "Match_Reason"] = reason
        books.at[idx, "Matched_2B_Invoice"] = match["Raw_Invoice_Number"]
        books.at[idx, "Matched_2B_RowID"] = match["Row_ID"]
        books.at[idx, "Confidence_Score"] = 100.0
        matched_2b_ids.add(match["Row_ID"])

    # --- PASS 2: FUZZY CANDIDATES ---
    unmatched_b = books[(~books["Match_Category"].isin(["DATA_QUALITY", "BOOKS_DUPLICATE", "MULTIPLE_CANDIDATES"])) & (books["Confidence_Score"].isna())]

    for idx, b_row in unmatched_b.iterrows():
        pool = valid_2b[(valid_2b["Parsed_GSTIN"] == b_row["Parsed_GSTIN"]) & (~valid_2b["Row_ID"].isin(matched_2b_ids))].copy()
        if b_row["Parsed_GSTIN"] == "UNKNOWN_GSTIN":
            books.at[idx, "Match_Category"], books.at[idx, "Match_Reason"] = "MISSING", "Missing/invalid GSTIN; fuzzy matching disabled."
            continue

        pool = pool[pool.apply(lambda r: document_type_compatible(b_row["Parsed_Document_Type"], r["Parsed_Document_Type"]), axis=1)]
        if pool.empty:
            books.at[idx, "Match_Category"], books.at[idx, "Match_Reason"] = "MISSING", "No unmatched GSTR-2B candidate with compatible document type and same GSTIN."
            continue

        pool["Inv_Sim"] = pool["Parsed_Inv"].apply(lambda x: fuzz.ratio(b_row["Parsed_Inv"], x))
        pool = pool[pool["Inv_Sim"] >= gates["min_inv_sim"]]
        if pool.empty:
            books.at[idx, "Match_Category"], books.at[idx, "Match_Reason"] = "MISSING", f"No candidate passed invoice similarity gate ({gates['min_inv_sim']}%)."
            continue

        if pd.notnull(b_row["Parsed_Date"]):
            pool["Date_Diff"] = (b_row["Parsed_Date"] - pool["Parsed_Date"]).abs().dt.days
            primary_pool = pool[(pool["Date_Diff"] <= tolerances["date_days"]) | pool["Date_Diff"].isna()].copy()
            fallback_pool = pool[~pool.index.isin(primary_pool.index)].copy()
        else:
            primary_pool, fallback_pool = pool.copy(), pool.iloc[0:0].copy()

        def apply_financial_gate(c_pool):
            passed = []
            for c_idx, c_row in c_pool.iterrows():
                ok, _, _ = financial_candidate_gate(b_row, c_row, tolerances, gates["min_fin_similarity"])
                if ok: passed.append(c_idx)
            return c_pool.loc[passed].copy()

        active_pool = apply_financial_gate(primary_pool)
        if active_pool.empty: active_pool = apply_financial_gate(fallback_pool)
        if active_pool.empty:
            books.at[idx, "Match_Category"], books.at[idx, "Match_Reason"] = "MISSING", "No candidate passed invoice, date and financial proximity gates."
            continue

        scored = []
        for _, c_row in active_pool.iterrows():
            score, details = calculate_fuzzy_score(b_row, c_row, weights, tolerances)
            scored.append({"row": c_row, "score": score, "details": details})
        scored.sort(key=lambda x: x["score"], reverse=True)

        if not scored or scored[0]["score"] < gates["min_score"]:
            books.at[idx, "Match_Category"], books.at[idx, "Match_Reason"] = "MISSING", f"Best candidate scored below minimum threshold ({gates['min_score']}%)."
            continue

        best = scored[0]
        second = scored[1] if len(scored) > 1 else None
        margin = best["score"] - second["score"] if second else best["score"]

        books.at[idx, "Score_Margin"] = round(margin, 2)
        books.at[idx, "Confidence_Score"] = round(best["score"], 2)

        if second and margin < gates["min_margin"]:
            books.at[idx, "Match_Category"], books.at[idx, "Match_Reason"] = "MULTIPLE_CANDIDATES", f"Ambiguous fuzzy result. Best={best['score']:.1f}, Second={second['score']:.1f}, margin={margin:.1f}."
            continue

        cat = "HIGH_CONFIDENCE" if best["score"] >= gates["high_confidence_score"] else "PROBABLE"
        books.at[idx, "Match_Category"] = cat
        books.at[idx, "Matched_2B_Invoice"] = best["row"]["Raw_Invoice_Number"]
        books.at[idx, "Matched_2B_RowID"] = best["row"]["Row_ID"]
        books.at[idx, "Match_Reason"] = f"Fuzzy score={best['score']:.1f}%; invoice similarity={best['details'].get('inv', 0):.1f}%."
        matched_2b_ids.add(best["row"]["Row_ID"])

    books["Status"] = books["Match_Category"].map(MATCH_CATEGORIES).fillna("Unknown State")
    unmatched_2b = gstr2b[(~gstr2b["Row_ID"].isin(matched_2b_ids)) & (gstr2b["Match_Category"] != "GSTR2B_DUPLICATE")]

    return books, unmatched_2b, gstr2b

# ============================================================
# 7B & 7C. CN/DN LINKING & CARRY FORWARD
# ============================================================

def link_credit_debit_notes(books_df):
    df = books_df.copy()
    df["Linked_Parent_RowID"] = None
    df["Linked_Parent_Invoice"] = None
    df["Net_ITC_Adjustment"] = np.nan
    df["CN_DN_Link_Status"] = ""

    invoice_rows = df[df["Parsed_Document_Type"] == "INVOICE"]
    parent_index = {(r["Parsed_GSTIN"], r["Parsed_Inv"]): r for _, r in invoice_rows.iterrows()}

    note_mask = df["Parsed_Document_Type"].isin(NOTE_TYPES)
    for idx, row in df[note_mask].iterrows():
        key = (row["Parsed_GSTIN"], row["Parsed_Original_Inv"])
        parent = parent_index.get(key)
        tax = row["Parsed_Total_Tax"] if pd.notna(row["Parsed_Total_Tax"]) else 0.0
        
        if row["Parsed_Document_Type"] in CREDIT_TYPES: df.at[idx, "Net_ITC_Adjustment"] = -tax
        elif row["Parsed_Document_Type"] in DEBIT_TYPES: df.at[idx, "Net_ITC_Adjustment"] = tax

        if parent is not None and row["Parsed_Original_Inv"] != "UNKNOWN_INV":
            df.at[idx, "Linked_Parent_RowID"], df.at[idx, "Linked_Parent_Invoice"], df.at[idx, "CN_DN_Link_Status"] = parent["Row_ID"], parent["Raw_Invoice_Number"], "LINKED"
        else:
            df.at[idx, "CN_DN_Link_Status"] = "PARENT_NOT_FOUND"

    return df

def build_net_itc_summary(linked_books_df):
    df = linked_books_df
    invoice_tax = df[df["Parsed_Document_Type"] == "INVOICE"].groupby("Parsed_GSTIN")["Parsed_Total_Tax"].sum(min_count=1)
    linked_note_adj = df[(df["Parsed_Document_Type"].isin(NOTE_TYPES)) & (df["CN_DN_Link_Status"] == "LINKED")].groupby("Parsed_GSTIN")["Net_ITC_Adjustment"].sum(min_count=1)
    unlinked_notes = df[(df["Parsed_Document_Type"].isin(NOTE_TYPES)) & (df["CN_DN_Link_Status"] == "PARENT_NOT_FOUND")].groupby("Parsed_GSTIN").agg(Unlinked_Note_Count=("Row_ID", "count"), Unlinked_Note_Value=("Net_ITC_Adjustment", lambda x: x.fillna(0).sum()))

    summary = pd.concat([invoice_tax.rename("Invoice_Tax"), linked_note_adj.rename("Linked_Note_Adjustment")], axis=1).fillna(0)
    summary["Net_ITC"] = summary["Invoice_Tax"] + summary["Linked_Note_Adjustment"]
    summary = summary.join(unlinked_notes, how="left").fillna({"Unlinked_Note_Count": 0, "Unlinked_Note_Value": 0})
    return summary.reset_index().rename(columns={"Parsed_GSTIN": "GSTIN"})

def init_ledger_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_itc (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_label TEXT, gstin TEXT, parsed_invoice TEXT,
            raw_invoice_number TEXT, vendor_name TEXT,
            taxable_value REAL, total_tax REAL, saved_at TEXT,
            UNIQUE(period_label, gstin, parsed_invoice)
        )
    """)
    conn.commit()
    conn.close()

def save_missing_to_ledger(books_df, period_label, db_path=DB_PATH):
    init_ledger_db(db_path)
    missing = books_df[books_df["Match_Category"] == "MISSING"]
    conn = sqlite3.connect(db_path)
    saved = 0
    for _, r in missing.iterrows():
        if r["Parsed_GSTIN"] == "UNKNOWN_GSTIN" or r["Parsed_Inv"] == "UNKNOWN_INV": continue
        try:
            conn.execute(
                """INSERT OR IGNORE INTO pending_itc
                   (period_label, gstin, parsed_invoice, raw_invoice_number, vendor_name, taxable_value, total_tax, saved_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (period_label, r["Parsed_GSTIN"], r["Parsed_Inv"], str(r["Raw_Invoice_Number"]), str(r["Raw_Vendor_Name"]), float(r["Parsed_Taxable_Value"]) if pd.notna(r["Parsed_Taxable_Value"]) else None, float(r["Parsed_Total_Tax"]) if pd.notna(r["Parsed_Total_Tax"]) else None, datetime.now().isoformat())
            )
            saved += conn.total_changes
        except sqlite3.Error: continue
    conn.commit()
    conn.close()
    return saved

def get_pending_ledger(db_path=DB_PATH):
    init_ledger_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM pending_itc ORDER BY saved_at DESC", conn)
    conn.close()
    return df

def resolve_carry_forward(unmatched_2b_df, current_period_label, db_path=DB_PATH):
    init_ledger_db(db_path)
    conn = sqlite3.connect(db_path)
    resolved_rows = []

    for _, row in unmatched_2b_df.iterrows():
        cur = conn.execute("""SELECT id, period_label, raw_invoice_number, vendor_name, taxable_value, total_tax FROM pending_itc WHERE gstin = ? AND parsed_invoice = ? AND period_label != ? ORDER BY saved_at ASC LIMIT 1""", (row["Parsed_GSTIN"], row["Parsed_Inv"], current_period_label))
        hit = cur.fetchone()
        if hit:
            pending_id, origin_period, orig_inv_no, vendor, taxable, tax = hit
            resolved_rows.append({"GSTIN": row["Parsed_GSTIN"], "Vendor_Name": vendor, "Invoice_Number": orig_inv_no, "Originally_Missing_In": origin_period, "Resolved_In": current_period_label, "Taxable_Value": taxable, "Total_Tax": tax})
            conn.execute("DELETE FROM pending_itc WHERE id = ?", (pending_id,))
    conn.commit()
    conn.close()
    return pd.DataFrame(resolved_rows) if resolved_rows else pd.DataFrame(columns=["GSTIN", "Vendor_Name", "Invoice_Number", "Originally_Missing_In", "Resolved_In", "Taxable_Value", "Total_Tax"])

# ============================================================
# 8. VENDOR FOLLOW-UP
# ============================================================

def build_vendor_followup_table(res_df):
    flagged = res_df[res_df["Match_Category"].isin(["MISSING", "TAX_MISMATCH", "DATE_MISMATCH"]) & (res_df["Parsed_GSTIN"] != "UNKNOWN_GSTIN")].copy()
    if flagged.empty: return pd.DataFrame(columns=["GSTIN", "Vendor_Name", "Invoice_Count", "ITC_At_Risk", "Invoice_List", "Category_Breakdown"])
    return flagged.groupby("Parsed_GSTIN").apply(
        lambda g: pd.Series({
            "GSTIN": g["Raw_GSTIN"].iloc[0] if pd.notna(g["Raw_GSTIN"].iloc[0]) else g["Parsed_GSTIN"].iloc[0],
            "Vendor_Name": g["Raw_Vendor_Name"].dropna().iloc[0] if g["Raw_Vendor_Name"].notna().any() else "(name not on file)",
            "Invoice_Count": len(g),
            "ITC_At_Risk": round(g["Parsed_Total_Tax"].fillna(0).sum(), 2),
            "Invoice_List": "; ".join(g["Raw_Invoice_Number"].astype(str).fillna("(no number)")),
            "Category_Breakdown": ", ".join(f"{cat}: {n}" for cat, n in g["Match_Category"].value_counts().items()),
        })
    ).reset_index(drop=True).sort_values("ITC_At_Risk", ascending=False)

def generate_email_template(vendor_name, gstin, invoice_list_str, itc_at_risk, period_label):
    return f"""Subject: Action Needed — Invoices Missing/Mismatched in GSTR-2B ({period_label})\n\nDear {vendor_name or "Vendor"},\n\nDuring our GST reconciliation for {period_label}, we found the following invoice(s) under GSTIN {gstin} showing as missing or mismatched in our GSTR-2B:\n\n{invoice_list_str}\n\nITC associated with these exceptions is approximately ₹{itc_at_risk:,.2f}, subject to reconciliation and tax review.\n\nCould you please check the filing details for the relevant period and confirm whether any correction or amendment is required?\n\nThank you,\n{{Your Name / Company}}"""

def generate_whatsapp_template(vendor_name, gstin, invoice_count, itc_at_risk, period_label):
    return f"Hi {vendor_name or 'there'}, regarding GST reconciliation for {period_label}: {invoice_count} invoice(s) under GSTIN {gstin} are showing as missing/mismatched in our GSTR-2B. Approx. ITC involved is ₹{itc_at_risk:,.2f}. Could you please check the filing details and let us know if any correction/amendment is required? Thanks!"

# ============================================================
# 10. STREAMLIT UI
# ============================================================

with st.sidebar:
    st.header("⚙️ Safety Settings")
    c1, c2 = st.columns(2)
    taxable_abs = c1.number_input("Taxable Abs (₹)", value=1.0, min_value=0.0)
    taxable_rel = c2.number_input("Taxable Rel (%)", value=0.5, min_value=0.0)
    comp_abs = c1.number_input("Component Abs (₹)", value=1.0, min_value=0.0)
    comp_rel = c2.number_input("Component Rel (%)", value=0.5, min_value=0.0)
    tot_abs = c1.number_input("Total Abs (₹)", value=1.0, min_value=0.0)
    tot_rel = c2.number_input("Total Rel (%)", value=0.5, min_value=0.0)
    date_tol = st.number_input("Date Tolerance (Days)", value=2, min_value=0)

    tols = {"taxable": {"abs": taxable_abs, "rel": taxable_rel}, "comp": {"abs": comp_abs, "rel": comp_rel}, "total": {"abs": tot_abs, "rel": tot_rel}, "date_days": date_tol}

    st.subheader("2. Matching gates")
    min_inv = st.slider("Min Invoice Similarity", 0, 100, 70)
    min_score = st.slider("Min Overall Score", 0, 100, 75)
    min_margin = st.slider("Min Score Margin", 0, 30, 8)
    min_fin_similarity = st.slider("Min Financial Similarity", 0, 100, 70)
    high_confidence_score = st.slider("High Confidence Score", 0, 100, 90)
    gates = {"min_inv_sim": min_inv, "min_score": min_score, "min_margin": min_margin, "min_fin_similarity": min_fin_similarity, "high_confidence_score": high_confidence_score}

    st.subheader("3. Scoring weights")
    w_inv = st.slider("Invoice", 0, 100, 30)
    w_date = st.slider("Date", 0, 100, 20)
    w_taxval = st.slider("Taxable", 0, 100, 10)
    w_tot = st.slider("Total Tax", 0, 100, 15)
    w_igst = st.slider("IGST", 0, 100, 10)
    w_cgst = st.slider("CGST", 0, 100, 5)
    w_sgst = st.slider("SGST", 0, 100, 5)
    w_cess = st.slider("CESS", 0, 100, 5)
    weights = {"inv": w_inv, "date": w_date, "taxable": w_taxval, "total": w_tot, "igst": w_igst, "cgst": w_cgst, "sgst": w_sgst, "cess": w_cess}

    st.divider()
    period_label = st.text_input("This period's label", value=datetime.now().strftime("%b-%Y"))
    enable_carry_forward = st.checkbox("Check missing against late vendor filings", value=True)

    st.divider()
    file_books = st.file_uploader("Upload Purchase Books (CSV/XLSX)", type=["csv", "xlsx"])
    file_2b = st.file_uploader("Upload GSTR-2B (JSON/PDF/CSV/XLSX)", type=["json", "pdf", "csv", "xlsx"])

if file_books and file_2b and sum(weights.values()) > 0:
    try:
        b_df = pd.read_csv(file_books) if file_books.name.lower().endswith(".csv") else pd.read_excel(file_books)

        warns, suffix = [], file_2b.name.lower()
        if suffix.endswith(".json"): g_df, warns = parse_gstr2b_json(file_2b)
        elif suffix.endswith(".pdf"): g_df, warns = parse_gstr2b_pdf(file_2b)
        elif suffix.endswith(".csv"): g_df = pd.read_csv(file_2b)
        else: g_df = pd.read_excel(file_2b)

        for w in warns: st.warning(w)

        if g_df.empty:
            st.error("No usable data could be extracted from GSTR-2B.")
        else:
            confirm_pdf = True
            if suffix.endswith(".pdf"):
                st.info("PDF safety check: review the parsed preview before reconciliation.")
                st.dataframe(g_df.head(20), use_container_width=True)
                confirm_pdf = st.checkbox("I reviewed the PDF-extracted rows and want to reconcile them.")

            if confirm_pdf:
                with st.spinner("Running V6.1 reconciliation..."):
                    res, unmatched, full_2b = reconcile_engine(b_df, g_df, tols, weights, gates)
                    carried_forward_df = resolve_carry_forward(unmatched, period_label) if enable_carry_forward else pd.DataFrame()
                    res = link_credit_debit_notes(res)
                    net_itc_summary = build_net_itc_summary(res)

                if not carried_forward_df.empty:
                    st.success(f"🗓️ {len(carried_forward_df)} invoice(s) previously logged as missing were resolved this period.")

                if st.button("💾 Save this period's MISSING invoices to the carry-forward ledger"):
                    save_missing_to_ledger(res, period_label)
                    st.info("Logged missing invoices. They'll be checked against future GSTR-2B uploads.")

                st.subheader("📈 Reconciliation Dashboard")
                matched_count = len(res[res["Match_Category"].isin(MATCHED_CATEGORIES)])
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Books", len(res))
                c2.metric("2B Records", len(full_2b))
                c3.metric("Matched", matched_count)
                c4.metric("Exceptions", len(res) - matched_count)
                c5.metric("ITC Exception Value", f"₹{res.loc[~res['Match_Category'].isin(MATCHED_CATEGORIES), 'Parsed_Total_Tax'].fillna(0).sum():,.2f}")

                tab1, tab2, tab3, tab4, tab5, tab7, tab8, tab9 = st.tabs([
                    "⚠️ Review Queue", "📋 Full Audit", "🏢 Vendor Summary", "📤 Export",
                    "📣 Vendor Follow-up", "🔗 CN/DN Linking", "🔁 RCM Invoices", "🗓️ Carry-Forward Ledger"
                ])

                with tab1:
                    st.dataframe(res[~res["Match_Category"].isin(MATCHED_CATEGORIES)][["Raw_GSTIN", "Raw_Invoice_Number", "Status", "Confidence_Score", "Match_Reason"]], use_container_width=True)

                with tab2:
                    st.dataframe(res[["Raw_GSTIN", "Raw_Invoice_Number", "Parsed_Document_Type", "Status", "Matched_2B_Invoice", "Confidence_Score", "Score_Margin", "Match_Reason"]], use_container_width=True)

                with tab3:
                    st.dataframe(res.groupby("Parsed_GSTIN").agg(Count=("Row_ID", "count"), Matched=("Match_Category", lambda x: x.isin(MATCHED_CATEGORIES).sum()), Exceptions=("Match_Category", lambda x: (~x.isin(MATCHED_CATEGORIES)).sum()), ITC=("Parsed_Total_Tax", lambda x: x.fillna(0).sum())).reset_index(), use_container_width=True)

                with tab4:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                        res.to_excel(writer, sheet_name="Reconciliation", index=False)
                        res[res["Match_Category"] == "MISSING"].to_excel(writer, sheet_name="Missing_in_2B", index=False)
                        res[res["Match_Category"].isin(["TAX_MISMATCH", "DATE_MISMATCH", "MULTIPLE_CANDIDATES", "DATA_QUALITY"])].to_excel(writer, sheet_name="Review_Queue", index=False)
                        unmatched.to_excel(writer, sheet_name="Unmatched_2B", index=False)
                    st.download_button("📥 Download Report", data=buf.getvalue(), file_name=f"V6.1_Reconciliation_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

                with tab5:
                    followup_df = build_vendor_followup_table(res)
                    if followup_df.empty: st.success("No vendors currently require follow-up.")
                    else:
                        st.dataframe(followup_df, use_container_width=True)
                        row = followup_df[followup_df["GSTIN"] == st.selectbox("Generate message for GSTIN:", followup_df["GSTIN"].tolist())].iloc[0]
                        msg_type = st.radio("Template", ["Email", "WhatsApp"], horizontal=True)
                        st.text_area("Copy this message:", value=generate_email_template(row["Vendor_Name"], row["GSTIN"], row["Invoice_List"], row["ITC_At_Risk"], period_label) if msg_type == "Email" else generate_whatsapp_template(row["Vendor_Name"], row["GSTIN"], row["Invoice_Count"], row["ITC_At_Risk"], period_label), height=280)

                with tab7:
                    notes_view = res[res["Parsed_Document_Type"].isin(NOTE_TYPES)][["Raw_GSTIN", "Raw_Vendor_Name", "Raw_Invoice_Number", "Parsed_Document_Type", "Raw_Original_Invoice_Number", "Linked_Parent_Invoice", "CN_DN_Link_Status", "Net_ITC_Adjustment"]]
                    if notes_view.empty: st.success("No credit/debit notes in this upload.")
                    else:
                        st.dataframe(notes_view, use_container_width=True)
                        if (unlinked_n := (notes_view["CN_DN_Link_Status"] == "PARENT_NOT_FOUND").sum()): st.warning(f"{unlinked_n} note(s) could not be linked to a parent invoice.")
                    st.subheader("Net ITC by Vendor (Invoices − Linked Notes)")
                    st.dataframe(net_itc_summary, use_container_width=True)

                with tab8:
                    rcm_view = res[res["Parsed_RCM"] == "YES"][["Raw_GSTIN", "Raw_Vendor_Name", "Raw_Invoice_Number", "Status", "Parsed_Total_Tax", "Match_Reason"]]
                    if rcm_view.empty: st.info("No Reverse Charge invoices in this upload.")
                    else:
                        st.dataframe(rcm_view, use_container_width=True)
                        st.metric("Total RCM Tax (self-payable)", f"₹{rcm_view['Parsed_Total_Tax'].fillna(0).sum():,.2f}")

                with tab9:
                    if not carried_forward_df.empty:
                        st.subheader("✅ Resolved this run (vendor filed late)")
                        st.dataframe(carried_forward_df, use_container_width=True)
                    st.subheader("⏳ Still pending across all periods")
                    if (pending_now := get_pending_ledger()).empty: st.success("Ledger is empty — nothing carried forward.")
                    else: st.dataframe(pending_now, use_container_width=True)

    except Exception as e:
        st.error(f"Execution error: {e}")
