import streamlit as st
import numpy as np
import pandas as pd
import json
import urllib.request
import io
from datetime import datetime
from openai import OpenAI

# Optional libraries for file parsing
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Price Reasonableness Assessment (IQR)",
    page_icon="⚖️",
    layout="wide"
)

# --- CURRENCY CONVERSION HELPER ---
@st.cache_data(ttl=3600)
def get_exchange_rates_to_sgd():
    """
    Fetches live exchange rates (1 Foreign Currency = X SGD).
    Includes static fallback exchange rates for CNY, USD, JPY, EUR, GBP, etc.
    """
    fallback_rates = {
        "SGD": 1.0,
        "USD": 1.35,   # 1 USD ≈ 1.35 SGD
        "CNY": 0.19,   # 1 CNY ≈ 0.19 SGD
        "RMB": 0.19,   # Alias for CNY
        "JPY": 0.0088, # 1 JPY ≈ 0.0088 SGD
        "EUR": 1.46,   # 1 EUR ≈ 1.46 SGD
        "GBP": 1.72,   # 1 GBP ≈ 1.72 SGD
        "AUD": 0.88,   # 1 AUD ≈ 0.88 SGD
        "MYR": 0.30,   # 1 MYR ≈ 0.30 SGD
        "HKD": 0.17,   # 1 HKD ≈ 0.17 SGD
    }
    
    try:
        url = "https://open.er-api.com/v6/latest/SGD"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as res:
            if res.status == 200:
                data = json.loads(res.read().decode('utf-8'))
                rates = data.get("rates", {})
                if rates:
                    converted_rates = {"SGD": 1.0}
                    for curr, rate in rates.items():
                        if rate > 0:
                            converted_rates[curr.upper()] = 1.0 / rate
                    return converted_rates
    except Exception:
        pass # Fallback to static rates if API call fails
        
    return fallback_rates

def convert_to_sgd(amount, currency_code, rates):
    """Converts a given amount and currency code into SGD using the rate dictionary."""
    curr = str(currency_code).upper().strip()
    rate = rates.get(curr)
    
    if rate is None:
        rate = rates.get("USD", 1.35) if curr != "SGD" else 1.0
        
    return round(amount * rate, 2)

# --- QUOTATION FILE PARSER HELPER ---
def extract_text_from_file(uploaded_file):
    """Extracts raw text content from PDF, Word, or text files."""
    file_ext = uploaded_file.name.split(".")[-1].lower()
    extracted_text = ""
    
    try:
        if file_ext == "pdf" and pypdf:
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        elif file_ext in ["docx", "doc"] and docx:
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    extracted_text += " | ".join([cell.text.strip() for cell in row.cells]) + "\n"
    except Exception as e:
        st.warning(f"Note: Error extracting raw text from file: {e}")
        
    return extracted_text

def parse_quotation_with_llm(file_text, api_key):
    """Uses GPT-4o to extract and structure unstructured document text into line items."""
    if not file_text.strip():
        return []
        
    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Extract line items from this quotation document text into a JSON list.\n"
            "Identify each item's description, quantity, and unit rate.\n"
            "Return raw JSON format only:\n"
            "{\n"
            '  "items": [\n'
            '    {"Item Description": "Item name/desc", "Quantity": 1, "Unit Rate (SGD, incl. GST)": 100.00}\n'
            "  ]\n"
            "}\n\n"
            f"DOCUMENT TEXT:\n{file_text[:4000]}"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        raw_out = response.choices[0].message.content.strip()
        if raw_out.startswith("```json"):
            raw_out = raw_out[7:]
        if raw_out.startswith("```"):
            raw_out = raw_out[3:]
        if raw_out.endswith("```"):
            raw_out = raw_out[:-3]
            
        data = json.loads(raw_out.strip())
        return data.get("items", [])
    except Exception:
        return []

def scan_and_extract_file(uploaded_file, api_key):
    """
    Scans the uploaded quotation document and extracts line items to fill Step 2.
    """
    file_ext = uploaded_file.name.split(".")[-1].lower()
    parsed_items = []
    
    # 1. Parse Excel files directly
    if file_ext in ["xlsx", "xls"]:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            df_uploaded.columns = df_uploaded.columns.str.strip().str.lower()
            
            for _, row in df_uploaded.iterrows():
                desc = str(row.get("item description", row.get("description", row.get("item", "")))).strip()
                qty = float(row.get("quantity", row.get("qty", 1)))
                rate = float(row.get("unit rate (sgd, incl. gst)", row.get("unit rate", row.get("unit price", row.get("price", 0.0)))))
                
                if desc and rate > 0:
                    parsed_items.append({
                        "Item Description": desc,
                        "Quantity": int(qty),
                        "Unit Rate (SGD, incl. GST)": round(rate, 2)
                    })
        except Exception as e:
            st.error(f"Error parsing Excel file: {e}")

    # 2. Extract PDF / Word doc text and parse via OpenAI model
    elif file_ext in ["pdf", "docx", "doc"]:
        if not api_key:
            st.error("🔑 Please enter your OpenAI API key in the sidebar to scan PDF or Word documents.")
            return []
            
        raw_text = extract_text_from_file(uploaded_file)
        if raw_text:
            parsed_items = parse_quotation_with_llm(raw_text, api_key)
        else:
            st.error("Could not extract readable text from the document. Please ensure it is not an image-only scan.")

    return parsed_items

# --- INITIALIZE SESSION STATE FOR SEARCH HISTORY ---
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.title("⚙️ Configuration")
user_api_key = st.sidebar.text_input(
    label="OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Enter your OpenAI API key to run live market price assessments."
)

# Fetch exchange rates
exchange_rates = get_exchange_rates_to_sgd()

# Sidebar Search History Quick Log
st.sidebar.divider()
st.sidebar.title("📜 Search History")
if st.session_state.search_history:
    st.sidebar.caption(f"Total Assessments Run: **{len(st.session_state.search_history)}**")
    for idx, hist in enumerate(reversed(st.session_state.search_history)):
        sup = hist.get("supplier_name") or "Unknown Supplier"
        ref = hist.get("quotation_ref") or "No Ref"
        ts = hist.get("timestamp", "")
        cost = hist.get("total_cost", 0.0)
        
        with st.sidebar.expander(f"🕒 {ts} - {sup[:15]}...", expanded=False):
            st.markdown(f"**Supplier:** {sup}")
            st.markdown(f"**Quote Ref:** {ref}")
            st.markdown(f"**Total Quote Cost:** S${cost:,.2f}")
            st.markdown(f"**Items Assessed:** {len(hist.get('results', []))}")
            
    if st.sidebar.button("🗑️ Clear History", key="clear_hist_btn"):
        st.session_state.search_history = []
        st.rerun()
else:
    st.sidebar.caption("No assessment history yet.")

# --- HEADER ---
st.title("⚖️ Price Reasonableness Assessment")
st.subheader("Evaluate Small Value Purchases (SVP) via Singapore Sourcing & IQR Analysis")
st.divider()

# --- FILE UPLOAD & SUPPLIER INFO ---
st.markdown("### 📄 Step 1: Upload Quotation & Supplier Details")

col_sup1, col_sup2 = st.columns([2, 2])

with col_sup1:
    supplier_name = st.text_input(
        label="Supplier Name",
        placeholder="e.g., Tech Supplies Pte Ltd"
    )

with col_sup2:
    quotation_ref = st.text_input(
        label="Quotation Reference / No.",
        placeholder="e.g., QUO-2026-0891"
    )

uploaded_quote = st.file_uploader(
    label="Upload Supplier Quotation (PDF, Excel, Word)",
    type=["pdf", "xlsx", "xls", "docx", "doc"],
    help="Upload a vendor quote document, then click the scan button below to populate Step 2."
)

# Session state initialization for line items
if "line_items_data" not in st.session_state:
    st.session_state.line_items_data = pd.DataFrame([
        {"Item Description": "Logitech MX Master 3S Wireless Mouse", "Quantity": 1, "Unit Rate (SGD, incl. GST)": 139.00}
    ])

# Action button to trigger file scanning
scan_col1, scan_col2 = st.columns([1, 3])
with scan_col1:
    scan_button = st.button("🔍 Scan & Extract Quotation Items", use_container_width=True)

if scan_button:
    if uploaded_quote is None:
        st.warning("⚠️ Please upload a quotation file first before clicking scan.")
    else:
        with st.spinner(f"Scanning `{uploaded_quote.name}` and extracting line item details..."):
            extracted = scan_and_extract_file(uploaded_quote, user_api_key)
            if extracted:
                st.session_state.line_items_data = pd.DataFrame(extracted)
                st.success(f"✅ Successfully extracted {len(extracted)} line item(s) into Step 2 below!")
            else:
                st.warning("Could not automatically extract line items. Please enter details manually in Step 2.")

st.divider()

# --- MULTI-LINE ITEM INPUT TABLE ---
st.markdown("### 📝 Step 2: Line Items & Pricing Breakdown")
st.caption("Populated from scanned quote file or entered manually below. Total costs compute automatically.")

edited_df = st.data_editor(
    st.session_state.line_items_data,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Item Description": st.column_config.TextColumn("Item Description", width="large", required=True),
        "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, step=1, required=True),
        "Unit Rate (SGD, incl. GST)": st.column_config.NumberColumn("Unit Rate (S$, incl. GST)", min_value=0.01, format="S$%.2f", required=True)
    }
)

# Calculate line totals and total quote cost automatically
if not edited_df.empty:
    edited_df["Quantity"] = pd.to_numeric(edited_df["Quantity"], errors="coerce").fillna(1)
    edited_df["Unit Rate (SGD, incl. GST)"] = pd.to_numeric(edited_df["Unit Rate (SGD, incl. GST)"], errors="coerce").fillna(0.0)
    edited_df["Total Cost (SGD)"] = edited_df["Quantity"] * edited_df["Unit Rate (SGD, incl. GST)"]
    
    total_quote_cost = float(edited_df["Total Cost (SGD)"].sum())
    
    col_tot1, col_tot2, col_tot3 = st.columns(3)
    col_tot1.metric("Total Line Items", f"{len(edited_df)}")
    col_tot2.metric("Total Calculated Quote Cost", f"S${total_quote_cost:,.2f}")
    col_tot3.caption(f"Supplier: **{supplier_name if supplier_name else 'N/A'}**\n\nRef: **{quotation_ref if quotation_ref else 'N/A'}**")
else:
    total_quote_cost = 0.0

submit_button = st.button("Assess Market Price Reasonableness", type="primary")

# --- INITIALIZE ASSESSMENT STATE ---
if "assessment_results" not in st.session_state:
    st.session_state.assessment_results = []
if "assessed" not in st.session_state:
    st.session_state.assessed = False

# --- CALLBACK FUNCTIONS FOR INTERACTIVE ITEM EDITS ---
def remove_item(item_idx, market_idx):
    st.session_state.assessment_results[item_idx]["market_items"].pop(market_idx)

def update_item_data(item_idx, market_idx):
    price_key = f"price_input_{item_idx}_{market_idx}"
    curr_key = f"curr_input_{item_idx}_{market_idx}"
    
    if price_key in st.session_state and curr_key in st.session_state:
        new_price = float(st.session_state[price_key])
        new_curr = str(st.session_state[curr_key]).upper().strip()
        
        target = st.session_state.assessment_results[item_idx]["market_items"][market_idx]
        target["original_price"] = new_price
        target["currency"] = new_curr
        target["price_sgd"] = convert_to_sgd(new_price, new_curr, exchange_rates)

# --- PROCESSING & ASSESSMENT ---
if submit_button:
    if not user_api_key:
        st.error("🔑 Please enter your OpenAI API Key in the sidebar to proceed.")
    elif edited_df.empty or total_quote_cost <= 0:
        st.error("⚠️ Please enter at least one line item with a valid unit rate and quantity.")
    else:
        client = OpenAI(api_key=user_api_key)
        assessment_results = []
        
        with st.spinner("Searching Singapore market benchmarks & evaluating quote line items..."):
            try:
                for idx, row in edited_df.iterrows():
                    desc = str(row.get("Item Description", "")).strip()
                    qty = int(row.get("Quantity", 1))
                    unit_rate = float(row.get("Unit Rate (SGD, incl. GST)", 0.0))
                    
                    if not desc or unit_rate <= 0:
                        continue

                    system_instructions = (
                        "You are a procurement analysis bot specializing in Singapore market benchmarking.\n"
                        "Your task:\n"
                        "1. Search live online sources for the specified product, giving strong priority to Singapore retailers, suppliers, and e-commerce platforms (e.g., Shopee SG, Lazada SG, Amazon SG, Challenger, Gain City, Courts, local SG vendors). Assume prices for SG vendors are in SGD unless specified.\n"
                        "2. Aim to collect 10 distinct pricing data points from Singapore sources first. Supplement with overseas sources (e.g., Taobao, JD.com, Amazon US) only if fewer than 10 local Singapore price points are available.\n"
                        "3. Obtain final nett unit prices (inclusive of GST/taxes where applicable).\n"
                        "4. Explicitly specify the original unit price and 3-letter currency code (e.g., SGD, USD, CNY, JPY) for each item.\n"
                        "5. Include direct URLs or domain links for source verification.\n"
                        "6. Return your result strictly in raw JSON format without markdown blocks, adhering to this structure:\n"
                        "{\n"
                        '  "prices_found": [\n'
                        '    {"source_name": "Lazada SG", "original_price": 129.00, "currency": "SGD", "region": "Singapore", "url": "[https://www.lazada.sg/](https://www.lazada.sg/)..."},\n'
                        '    {"source_name": "Shopee SG", "original_price": 125.00, "currency": "SGD", "region": "Singapore", "url": "[https://shopee.sg/](https://shopee.sg/)..."}\n'
                        "  ],\n"
                        '  "notes": "Brief comment on market availability."\n'
                        "}"
                    )
                    
                    user_prompt = f"Find current live nett market unit prices (targeting 10 sources, Singapore first) for: {desc}"
                    
                    response = client.responses.create(
                        model="gpt-4o",
                        tools=[{"type": "web_search"}],
                        input=[
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2
                    )
                    
                    raw_text = response.output_text
                    clean_text = raw_text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    elif clean_text.startswith("```"):
                        clean_text = clean_text[3:]

                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]

                    data = json.loads(clean_text.strip())
                    web_items = data.get("prices_found", [])
                    
                    for item in web_items:
                        orig_p = float(item.get("original_price", 0.0))
                        curr = str(item.get("currency", "SGD")).upper()
                        item["price_sgd"] = convert_to_sgd(orig_p, curr, exchange_rates)

                    assessment_results.append({
                        "item_description": desc,
                        "quantity": qty,
                        "quoted_unit_rate": unit_rate,
                        "quoted_line_total": unit_rate * qty,
                        "market_items": web_items,
                        "notes": data.get("notes", "")
                    })

                st.session_state.assessment_results = assessment_results
                st.session_state.assessed = True

                # Save assessment to search history log
                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "supplier_name": supplier_name,
                    "quotation_ref": quotation_ref,
                    "total_cost": total_quote_cost,
                    "results": assessment_results
                }
                st.session_state.search_history.append(history_entry)

            except json.JSONDecodeError:
                st.error("Failed to parse pricing payload from search engine response. Please try again.")
            except Exception as e:
                st.error(f"An error occurred while connecting to OpenAI API: {e}")

# --- DISPLAY ASSESSMENT RESULTS ---
if st.session_state.assessed and st.session_state.assessment_results:
    st.divider()
    st.markdown("### 📊 Assessment Outcome Summary")
    
    if supplier_name or quotation_ref:
        st.info(f"**Supplier:** {supplier_name if supplier_name else 'N/A'} | **Quote Ref:** {quotation_ref if quotation_ref else 'N/A'}")

    for item_idx, res in enumerate(st.session_state.assessment_results):
        desc = res["item_description"]
        qty = res["quantity"]
        quoted_unit_rate = res["quoted_unit_rate"]
        market_items = res["market_items"]
        
        st.markdown(f"#### 📦 Item {item_idx + 1}: {desc} (Qty: {qty})")
        
        if not market_items or len(market_items) < 3:
            st.warning("⚠️ Fewer than 3 market price sources available for this item.")
        
        if market_items:
            # Re-sync prices
            for m in market_items:
                orig_p = float(m.get("original_price", 0.0))
                curr = str(m.get("currency", "SGD")).upper()
                m["price_sgd"] = convert_to_sgd(orig_p, curr, exchange_rates)

            df_m = pd.DataFrame(market_items)
            df_m["price_sgd"] = df_m["price_sgd"].astype(float)
            prices = df_m["price_sgd"].values
            
            # --- IQR CALCULATIONS ---
            q1 = float(np.percentile(prices, 25))
            median = float(np.median(prices))
            q3 = float(np.percentile(prices, 75))
            iqr = q3 - q1
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Quoted Unit Rate", f"S${quoted_unit_rate:,.2f}")
            m2.metric("25th Percentile (Q1)", f"S${q1:,.2f}")
            m3.metric("Median (Q2)", f"S${median:,.2f}")
            m4.metric("IQR Range", f"S${iqr:,.2f}")
            
            # Evaluation Banner Logic
            if quoted_unit_rate <= q1:
                st.success(
                    f"✅ **Reasonable Price**: Quoted unit rate (**S${quoted_unit_rate:,.2f}**) is **at or below** "
                    f"the 25th percentile ($Q_1 = \\text{{S\\$}}{q1:,.2f}$) benchmark target."
                )
            elif quoted_unit_rate <= median:
                st.warning(
                    f"⚠️ **Acceptable Price**: Quoted unit rate (**S${quoted_unit_rate:,.2f}**) is above $Q_1$ "
                    f"(\\text{{S\\$}}{q1:,.2f}) but within the median market rate (\\text{{S\\$}}{median:,.2f})."
                )
            else:
                st.error(
                    f"❌ **Unreasonable Price**: Quoted unit rate (**S${quoted_unit_rate:,.2f}**) exceeds "
                    f"the target $Q_1$ baseline (\\text{{S\\$}}{q1:,.2f}) and median market benchmark."
                )
            
            # Breakdown Table
            with st.expander(f"🔍 View Market Data Points ({len(market_items)} sources)", expanded=False):
                h_col1, h_col2, h_col3, h_col4, h_col5, h_col6, h_col7 = st.columns([2.2, 1.8, 1.3, 1.8, 1.2, 2.5, 1.0])
                h_col1.markdown("**Source / Retailer**")
                h_col2.markdown("**Original Price**")
                h_col3.markdown("**Currency**")
                h_col4.markdown("**Nett Price (SGD)**")
                h_col5.markdown("**Region**")
                h_col6.markdown("**Verify Source URL**")
                h_col7.markdown("**Action**")

                for m_idx, m_item in enumerate(market_items):
                    r_col1, r_col2, r_col3, r_col4, r_col5, r_col6, r_col7 = st.columns([2.2, 1.8, 1.3, 1.8, 1.2, 2.5, 1.0])
                    
                    orig_p = float(m_item.get("original_price", 0.0))
                    curr = str(m_item.get("currency", "SGD")).upper()
                    sgd_p = float(m_item.get("price_sgd", 0.0))
                    
                    r_col1.write(m_item.get("source_name", "N/A"))
                    
                    r_col2.number_input(
                        label="Price",
                        value=orig_p,
                        min_value=0.0,
                        step=1.0,
                        format="%.2f",
                        key=f"price_input_{item_idx}_{m_idx}",
                        on_change=update_item_data,
                        args=(item_idx, m_idx),
                        label_visibility="collapsed"
                    )
                    
                    r_col3.text_input(
                        label="Currency",
                        value=curr,
                        key=f"curr_input_{item_idx}_{m_idx}",
                        on_change=update_item_data,
                        args=(item_idx, m_idx),
                        label_visibility="collapsed"
                    )
                    
                    r_col4.write(f"**S${sgd_p:,.2f}**")
                    r_col5.write(m_item.get("region", "N/A"))
                    
                    url = m_item.get("url", "")
                    if url and url.startswith("http"):
                        r_col6.markdown(f"[🔗 Visit Source]({url})")
                    elif url:
                        r_col6.write(url)
                    else:
                        r_col6.write("N/A")
                        
                    r_col7.button(
                        "🗑️",
                        key=f"del_{item_idx}_{m_idx}",
                        on_click=remove_item,
                        args=(item_idx, m_idx),
                        help="Remove this price point"
                    )

        st.divider()

# --- HISTORICAL SEARCH RESULTS LOG SECTION ---
if st.session_state.search_history:
    st.divider()
    with st.expander("📜 Historical Assessment Audit Trail", expanded=False):
        st.markdown("Below is a record of all price reasonableness assessments conducted during this session.")
        
        for h_idx, record in enumerate(reversed(st.session_state.search_history)):
            st.markdown(f"#### 📅 {record['timestamp']} — Supplier: `{record['supplier_name'] or 'N/A'}` (Ref: `{record['quotation_ref'] or 'N/A'}`)")
            st.markdown(f"**Total Quote Value:** S${record['total_cost']:,.2f}")
            
            for item in record["results"]:
                st.markdown(f"- **Item:** {item['item_description']} | **Qty:** {item['quantity']} | **Quoted Unit Rate:** S${item['quoted_unit_rate']:,.2f} | **Line Total:** S${item['quoted_line_total']:,.2f}")
                
            st.divider()

# --- FOOTER ---
st.divider()
st.caption("GenAI Procurement Assistant • Singapore First Market Search & IQR Analytics")