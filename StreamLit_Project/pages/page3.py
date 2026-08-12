import streamlit as st
import numpy as np
import pandas as pd
import json
import urllib.request
import re
import base64
from datetime import datetime
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional libraries for document parsing
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

# --- EMBEDDED KNOWLEDGE BASE (FILES 1 & 2) ---
EMAIL_TEMPLATE_DOC = """
Subject: Seeking Director Approval: Small Value Purchase (SVP) – [Brief Description of Item/Service] – [Vendor Name]
Dear [Director's Name],

I am writing to seek your approval for a Small Value Purchase (SVP) for [Item/Service Description] from [Selected Vendor Name], for a total amount of S$ [Amount] (excluding GST).

Purchase Summary
Item/Service: [Brief description of what is being purchased]
Vendor Selected: [Selected Vendor Name]
Total Cost: S$ [Amount] (excluding GST)
Purpose/Project: [Brief explanation of why this purchase is needed and how it supports the project/team]

Price Reasonableness & Justification
In accordance with the SVP process guidelines, I have assessed the price reasonableness via [online pricing / formal quotations / past purchase records]:
[Price Breakdown / Vendor Comparison]

Justification for Selection:
[Justification Text]

Compliance Checks
Order Splitting: Confirmed that this is a single, complete requirement and not split from a larger order.
Full Cost Included: The quoted price includes all applicable delivery, handling, and incidental charges.
Supporting Documents: Screenshots of price comparisons, vendor quotes, and past records are attached for your reference.

Please let me know if you approve this purchase or if you need any additional information.

Thank you!
"""

SVP_PROCESS_DOC = """
Small Value Purchase (SVP) Process
Overview & Threshold:
The Small Value Purchase process applies only to purchases up to S$6,000 (excluding GST).

Determining Price Reasonableness:
Before proceeding with a purchase, the Buyer/Requestor must ensure the cost is reasonable.
Note: "Reasonable" does not necessarily mean the cheapest or lowest option available. As long as the price is justified and fair, the item or service can be procured.
To assess price reasonableness, reference online pricing, formal quotations, or past purchase records.

Next Steps Based on Assessment:
- If price is reasonable: Proceed to seek approval from designated Approving Authority.
- If price is NOT reasonable: Explore alternative suppliers or vendors.

Evaluating Higher-Priced Vendors:
If an intended vendor's price appears higher than expected, check if the premium is justified by:
1. Terms and Conditions (e.g., better warranty or payment terms)
2. Quality (e.g., superior materials or higher specifications)
3. Delivery Timeline (e.g., faster lead times or urgent delivery)
4. Macroeconomic Factors (e.g., supply chain disruptions, geopolitical events, rising fuel costs)
5. Availability of goods and/or services

Tips and Compliance:
1. Document price comparisons (keep evidence with date/timestamp).
2. Avoid Order Splitting (Strict non-compliance: never split large purchases under S$6,000 threshold to avoid formal tendering).
3. Check Preferred/Approved Vendor Lists first.
4. Provide clear justification text if selecting non-lowest cost vendor.
5. Factor in full costs (delivery, handling, taxes).
"""

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Price Reasonableness & SVP Assistant",
    page_icon="⚖️",
    layout="wide"
)

# --- CURRENCY CONVERSION HELPER ---
@st.cache_data(ttl=3600)
def get_exchange_rates_to_sgd():
    """Fetches live exchange rates or uses static fallbacks."""
    fallback_rates = {
        "SGD": 1.0, "USD": 1.35, "CNY": 0.19, "RMB": 0.19,
        "JPY": 0.0088, "EUR": 1.46, "GBP": 1.72, "AUD": 0.88,
        "MYR": 0.30, "HKD": 0.17
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
        pass
    return fallback_rates

def convert_to_sgd(amount, currency_code, rates):
    curr = str(currency_code).upper().strip()
    rate = rates.get(curr)
    if rate is None:
        rate = rates.get("USD", 1.35) if curr != "SGD" else 1.0
    return round(amount * rate, 2)

# --- RAG RETRIEVAL ENGINE ---
def chunk_text(text, chunk_size=120, overlap=25):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

@st.cache_resource
def get_rag_chunks():
    return chunk_text(SVP_PROCESS_DOC)

def retrieve_svp_guidelines(query, text_chunks, top_k=2):
    if not text_chunks:
        return ""
    vectorizer = TfidfVectorizer()
    corpus = text_chunks + [query]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return "\n".join([text_chunks[i] for i in top_indices if scores[i] > 0.02])

# --- FILE & MULTIMODAL VISION OCR HELPERS ---
def extract_text_from_file(uploaded_file):
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
        st.warning(f"Note: Error extracting text: {e}")
    return extracted_text

def parse_quotation_with_llm(file_text, api_key):
    if not file_text.strip():
        return []
    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "Extract line items from this quotation text into JSON format:\n"
            '{"items": [{"Item Description": "desc", "Quantity": 1, "Unit Rate (SGD, incl. GST)": 100.00}]}\n'
            f"DOCUMENT TEXT:\n{file_text[:4000]}"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content.strip())
        return data.get("items", [])
    except Exception:
        return []

def parse_image_with_vision(uploaded_file, api_key):
    try:
        client = OpenAI(api_key=api_key)
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode("utf-8")
        
        prompt = (
            "Analyze this quotation or e-commerce screenshot and extract product line items into JSON.\n"
            'Format: {"items": [{"Item Description": "Name", "Quantity": 1, "Unit Rate (SGD, incl. GST)": 99.00}]}'
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content.strip())
        return data.get("items", [])
    except Exception as e:
        st.error(f"Error reading screenshot: {e}")
        return []

def scan_and_extract_file(uploaded_file, api_key):
    file_ext = uploaded_file.name.split(".")[-1].lower()
    parsed_items = []
    
    if file_ext in ["xlsx", "xls"]:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            df_uploaded.columns = df_uploaded.columns.str.strip().str.lower()
            for _, row in df_uploaded.iterrows():
                desc = str(row.get("item description", row.get("description", row.get("item", "")))).strip()
                qty = float(row.get("quantity", row.get("qty", 1)))
                rate = float(row.get("unit rate (sgd, incl. gst)", row.get("unit rate", row.get("unit price", row.get("price", 0.0)))))
                if desc and rate > 0:
                    parsed_items.append({"Item Description": desc, "Quantity": int(qty), "Unit Rate (SGD, incl. GST)": round(rate, 2)})
        except Exception as e:
            st.error(f"Error parsing Excel: {e}")
    elif file_ext in ["png", "jpg", "jpeg"]:
        if not api_key:
            st.error("🔑 OpenAI API Key is required for Vision Screenshot parsing.")
            return []
        parsed_items = parse_image_with_vision(uploaded_file, api_key)
    elif file_ext in ["pdf", "docx", "doc"]:
        if not api_key:
            st.error("🔑 OpenAI API Key required for document scanning.")
            return []
        raw_text = extract_text_from_file(uploaded_file)
        if raw_text:
            parsed_items = parse_quotation_with_llm(raw_text, api_key)
        else:
            st.error("Could not extract readable text from document.")
    return parsed_items

# --- INITIALIZE SESSION STATE ---
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "assessment_results" not in st.session_state:
    st.session_state.assessment_results = []
if "assessed" not in st.session_state:
    st.session_state.assessed = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I am your SVP Policy Guide. Ask me anything about procurement thresholds, non-lowest cost justifications, or order splitting policy!"}
    ]

exchange_rates = get_exchange_rates_to_sgd()
rag_chunks = get_rag_chunks()

# --- SIDEBAR: CONFIGURATION & HISTORY ---
st.sidebar.title("⚙️ Configuration")
user_api_key = st.sidebar.text_input(
    label="OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Enter your OpenAI API key to run live web benchmarking & policy search."
)

st.sidebar.divider()
st.sidebar.title("📚 RAG Policy Engine")
st.sidebar.success("✅ Connected to SVP Policy Knowledge Base")

with st.sidebar.expander("SVP Rules Summary"):
    st.markdown("- **Threshold**: Up to S$6,000 (excl. GST).\n- **Reasonableness**: Non-cheapest option permitted if justified.\n- **Justifications**: Warranty, quality, delivery timeline, macroeconomic factors.\n- **Compliance**: Order splitting strictly prohibited.")

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
            st.markdown(f"**Total Cost:** S${cost:,.2f}")
            st.markdown(f"**Items:** {len(hist.get('results', []))}")
            
    if st.sidebar.button("🗑️ Clear History", key="clear_hist_btn"):
        st.session_state.search_history = []
        st.rerun()
else:
    st.sidebar.caption("No assessment history yet.")

# --- NAVIGATION TABS ---
tab_assessment, tab_chat = st.tabs(["📊 Price Assessment & Approval", "💬 SVP Policy Q&A Chatbot"])

# ==============================================================================
# TAB 1: PRICE REASONABLENESS ASSESSMENT
# ==============================================================================
with tab_assessment:
    st.title("⚖️ Price Reasonableness Assessment & Approval Helper")
    st.subheader("Evaluate Small Value Purchases (SVP) via Dynamic Market Sourcing, RAG Rules & IQR Analysis")
    st.divider()

    st.markdown("### 📄 Step 1: Upload Quotation & Supplier Details")
    col_sup1, col_sup2 = st.columns([2, 2])
    with col_sup1:
        supplier_name = st.text_input(label="Supplier Name", placeholder="e.g., Tech Supplies Pte Ltd")
    with col_sup2:
        quotation_ref = st.text_input(label="Quotation Reference / No.", placeholder="e.g., QUO-2026-0891")

    uploaded_quote = st.file_uploader(
        label="Upload Supplier Quotation or Screenshot (PDF, Excel, Word, PNG, JPG)",
        type=["pdf", "xlsx", "xls", "docx", "doc", "png", "jpg", "jpeg"],
        help="Upload a quote document or screenshot to scan items into Step 2."
    )

    if "line_items_data" not in st.session_state:
        st.session_state.line_items_data = pd.DataFrame([
            {"Item Description": "Logitech MX Master 3S Wireless Mouse", "Quantity": 1, "Unit Rate (SGD, incl. GST)": 139.00}
        ])

    scan_col1, _ = st.columns([1, 3])
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

    if not edited_df.empty:
        edited_df["Quantity"] = pd.to_numeric(edited_df["Quantity"], errors="coerce").fillna(1)
        edited_df["Unit Rate (SGD, incl. GST)"] = pd.to_numeric(edited_df["Unit Rate (SGD, incl. GST)"], errors="coerce").fillna(0.0)
        edited_df["Total Cost (SGD)"] = edited_df["Quantity"] * edited_df["Unit Rate (SGD, incl. GST)"]
        
        total_quote_cost = float(edited_df["Total Cost (SGD)"].sum())
        
        col_tot1, col_tot2, col_tot3 = st.columns(3)
        col_tot1.metric("Total Line Items", f"{len(edited_df)}")
        col_tot2.metric("Total Calculated Quote Cost", f"S${total_quote_cost:,.2f}")
        col_tot3.caption(f"Supplier: **{supplier_name if supplier_name else 'N/A'}**\n\nRef: **{quotation_ref if quotation_ref else 'N/A'}**")
        
        if total_quote_cost > 6000:
            st.error("🚨 **SVP Policy Alert:** Total quote cost exceeds S$6,000 threshold (excl. GST). SVP process may not apply.")
    else:
        total_quote_cost = 0.0

    submit_button = st.button("Assess Market Price Reasonableness & SVP Policy Compliance", type="primary")

    # Callbacks for interactive table edits
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

    if submit_button:
        if not user_api_key:
            st.error("🔑 Please enter your OpenAI API Key in the sidebar to proceed.")
        elif edited_df.empty or total_quote_cost <= 0:
            st.error("⚠️ Please enter at least one line item with a valid unit rate and quantity.")
        else:
            client = OpenAI(api_key=user_api_key)
            assessment_results = []
            
            with st.spinner("Searching available web sources across the market & checking SVP policy compliance..."):
                try:
                    for idx, row in edited_df.iterrows():
                        desc = str(row.get("Item Description", "")).strip()
                        qty = int(row.get("Quantity", 1))
                        unit_rate = float(row.get("Unit Rate (SGD, incl. GST)", 0.0))
                        
                        if not desc or unit_rate <= 0:
                            continue

                        retrieved_rules = retrieve_svp_guidelines(desc + " price reasonableness justification threshold", rag_chunks)

                        # Updated instructions strictly demanding direct deep-link product paths in url field
                        system_instructions = (
                            "You are a procurement analysis bot specializing in dynamic market benchmarking & SVP Policy Compliance.\n"
                            f"--- RAG SVP RULES ---\n{retrieved_rules}\n---------------------\n"
                            "Your task:\n"
                            "1. Search live online sources across any available public websites, e-commerce stores, vendor pages, or distributors globally or locally.\n"
                            "2. Collect 3-10 diverse market price sources.\n"
                            "3. CRITICAL REQUIREMENT FOR DEEP LINKS: In the 'url' field, you MUST supply the full, direct product permalink URL path (e.g. 'https://www.retailer.com/product/item-name-12345' or 'https://store.com/dp/B09HM94VDS'). DO NOT return just the base website domain (e.g., 'https://www.retailer.com').\n"
                            "4. Evaluate price reasonableness according to SVP guidelines.\n"
                            "5. Provide a 'suggestion_action' advising if the price is fair or requires specific justifications.\n"
                            "6. Return strictly raw JSON:\n"
                            '{"prices_found": [{"source_name": "Name of Retailer", "original_price": 129.00, "currency": "SGD", "region": "Country/Region", "url": "https://www.actualwebsite.com/full/product/path/or/id"}], "suggestion_action": "Suggested steps..."}'
                        )
                        
                        user_prompt = f"Find current live nett market unit prices and full direct product page URLs for: {desc}"
                        
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            response_format={"type": "json_object"},
                            messages=[
                                {"role": "system", "content": system_instructions},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.2
                        )
                        
                        data = json.loads(response.choices[0].message.content.strip())
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
                            "suggestion": data.get("suggestion_action", "")
                        })

                    st.session_state.assessment_results = assessment_results
                    st.session_state.assessed = True

                    history_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "supplier_name": supplier_name,
                        "quotation_ref": quotation_ref,
                        "total_cost": total_quote_cost,
                        "results": assessment_results
                    }
                    st.session_state.search_history.append(history_entry)

                except Exception as e:
                    st.error(f"An error occurred while connecting to OpenAI API: {e}")

    # Display Assessment Results
    if st.session_state.assessed and st.session_state.assessment_results:
        st.divider()
        st.markdown("### 📊 Assessment Outcome & SVP Guidance Summary")
        
        if supplier_name or quotation_ref:
            st.info(f"**Supplier:** {supplier_name if supplier_name else 'N/A'} | **Quote Ref:** {quotation_ref if quotation_ref else 'N/A'}")

        all_quoted_items = []

        for item_idx, res in enumerate(st.session_state.assessment_results):
            desc = res["item_description"]
            qty = res["quantity"]
            quoted_unit_rate = res["quoted_unit_rate"]
            market_items = res["market_items"]
            
            st.markdown(f"#### 📦 Item {item_idx + 1}: {desc} (Qty: {qty})")
            
            if not market_items or len(market_items) < 3:
                st.warning("⚠️ Fewer than 3 market price sources available for this item.")
            
            if market_items:
                for m in market_items:
                    orig_p = float(m.get("original_price", 0.0))
                    curr = str(m.get("currency", "SGD")).upper()
                    m["price_sgd"] = convert_to_sgd(orig_p, curr, exchange_rates)

                df_m = pd.DataFrame(market_items)
                df_m["price_sgd"] = df_m["price_sgd"].astype(float)
                prices = df_m["price_sgd"].values
                
                q1 = float(np.percentile(prices, 25))
                median = float(np.median(prices))
                q3 = float(np.percentile(prices, 75))
                iqr = q3 - q1
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Quoted Unit Rate", f"S${quoted_unit_rate:,.2f}")
                m2.metric("25th Percentile (Q1)", f"S${q1:,.2f}")
                m3.metric("Median (Q2)", f"S${median:,.2f}")
                m4.metric("IQR Range", f"S${iqr:,.2f}")
                
                if quoted_unit_rate <= q1:
                    st.success(
                        f"✅ **Reasonable Price**: Quoted unit rate (**S${quoted_unit_rate:,.2f}**) is **at or below** "
                        f"the 25th percentile target (**S${q1:,.2f}**)."
                    )
                    outcome_status = "Reasonable (At/Below Q1)"
                elif quoted_unit_rate <= median:
                    st.warning(
                        f"⚠️ **Acceptable Price**: Quoted unit rate (**S${quoted_unit_rate:,.2f}**) is above Q1 "
                        f"(**S${q1:,.2f}**) but within the median market rate (**S${median:,.2f}**)."
                    )
                    outcome_status = "Acceptable (Within Median)"
                else:
                    st.error(
                        f"❌ **Higher Price**: Quoted unit rate (**S${quoted_unit_rate:,.2f}**) exceeds "
                        f"the target Q1 baseline (**S${q1:,.2f}**) and median market benchmark (**S${median:,.2f}**)."
                    )
                    outcome_status = "Higher than Median (Justification Required)"
                
                if res.get("suggestion"):
                    st.info(f"💡 **SVP Suggestion**: {res['suggestion']}")

                all_quoted_items.append({"desc": desc, "quoted": quoted_unit_rate, "q1": q1, "median": median, "status": outcome_status})

                with st.expander(f"🔍 View & Edit Market Data Points ({len(market_items)} sources)", expanded=False):
                    h_col1, h_col2, h_col3, h_col4, h_col5, h_col6, h_col7 = st.columns([2.2, 1.8, 1.3, 1.8, 1.2, 2.5, 1.0])
                    h_col1.markdown("**Source / Retailer**")
                    h_col2.markdown("**Original Price**")
                    h_col3.markdown("**Currency**")
                    h_col4.markdown("**Nett Price (SGD)**")
                    h_col5.markdown("**Region**")
                    h_col6.markdown("**Verify Source Link**")
                    h_col7.markdown("**Action**")

                    for m_idx, m_item in enumerate(market_items):
                        r_col1, r_col2, r_col3, r_col4, r_col5, r_col6, r_col7 = st.columns([2.2, 1.8, 1.3, 1.8, 1.2, 2.5, 1.0])
                        orig_p = float(m_item.get("original_price", 0.0))
                        curr = str(m_item.get("currency", "SGD")).upper()
                        sgd_p = float(m_item.get("price_sgd", 0.0))
                        
                        r_col1.write(m_item.get("source_name", "N/A"))
                        r_col2.number_input(label="Price", value=orig_p, min_value=0.0, step=1.0, format="%.2f", key=f"price_input_{item_idx}_{m_idx}", on_change=update_item_data, args=(item_idx, m_idx), label_visibility="collapsed")
                        r_col3.text_input(label="Currency", value=curr, key=f"curr_input_{item_idx}_{m_idx}", on_change=update_item_data, args=(item_idx, m_idx), label_visibility="collapsed")
                        r_col4.write(f"**S${sgd_p:,.2f}**")
                        r_col5.write(m_item.get("region", "N/A"))
                        
                        # Enhanced URL renderer detecting root vs deep links
                        url = m_item.get("url", "").strip()
                        if url.startswith("http"):
                            clean_path = url.replace("https://", "").replace("http://", "").strip("/")
                            domain_parts = clean_path.split("/")
                            if len(domain_parts) == 1:
                                r_col6.markdown(f"[🌐 Homepage]({url}) *(Base Domain)*")
                            else:
                                r_col6.markdown(f"[🔗 Direct Item Link]({url})")
                        elif url:
                            r_col6.write(url)
                        else:
                            r_col6.write("N/A")
                            
                        r_col7.button("🗑️", key=f"del_{item_idx}_{m_idx}", on_click=remove_item, args=(item_idx, m_idx), help="Remove price point")

            st.divider()

        # Generate Director Email Draft
        st.markdown("### ✉️ Generate Director Approval Email (Doc 1 Template)")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            director_name = st.text_input("Director's Name", value="Dr. Tan")
            project_purpose = st.text_input("Purpose / Project", value="Replacement equipment for team deployment")
        with e_col2:
            selected_justification = st.selectbox(
                "Primary Selection Justification",
                [
                    "Lowest total cost available on market",
                    "Faster delivery timeline / urgent deployment critical for project (e.g., 3 days vs 3 weeks)",
                    "Superior warranty & support terms",
                    "Higher quality specifications / preferred onboarded vendor",
                    "Sole distributor / specific compatibility requirements"
                ]
            )
            alt_vendor_info = st.text_input("Alternative Vendor Comparison (Optional)", value="Option B (Alternative Vendor): S$ higher – 3 weeks lead time")

        if st.button("📝 Draft Approval Email", type="primary"):
            comparison_text = ""
            for itm in all_quoted_items:
                comparison_text += f"Option A (Selected Vendor - {supplier_name if supplier_name else 'Vendor'}): S$ {itm['quoted']:,.2f} (Assessment: {itm['status']})\n"
            if alt_vendor_info:
                comparison_text += f"{alt_vendor_info}\n"

            first_item = all_quoted_items[0]['desc'] if all_quoted_items else "Purchased Items"
            sup_name = supplier_name if supplier_name else "[Selected Vendor Name]"

            email_output = EMAIL_TEMPLATE_DOC \
                .replace("[Director's Name]", director_name) \
                .replace("[Brief Description of Item/Service]", first_item) \
                .replace("[Vendor Name]", sup_name) \
                .replace("[Selected Vendor Name]", sup_name) \
                .replace("[Item/Service Description]", first_item) \
                .replace("[Amount]", f"{total_quote_cost:,.2f}") \
                .replace("[Brief explanation of why this purchase is needed and how it supports the project/team]", project_purpose) \
                .replace("[online pricing / formal quotations / past purchase records]", "online pricing benchmarks and market IQR assessment") \
                .replace("[Price Breakdown / Vendor Comparison]", comparison_text) \
                .replace("[Justification Text]", f"{sup_name} was chosen because of {selected_justification.lower()}.")

            st.markdown("#### 📋 Copy-Ready Draft Email")
            st.code(email_output, language="markdown")

            st.download_button(
                label="📥 Download Audit Memo (.txt)",
                data=email_output,
                file_name=f"SVP_Approval_Memo_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

# ==============================================================================
# TAB 2: CONVERSATIONAL SVP POLICY CHATBOT
# ==============================================================================
with tab_chat:
    st.title("💬 SVP Policy Assistant")
    st.caption("Ask questions about Small Value Purchase rules, justification criteria, or threshold compliance.")
    
    # Render historical chat messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Process new user input
    if prompt := st.chat_input("Ask a question about SVP policy..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not user_api_key:
                response_text = "🔑 **API Key Missing**: Please enter your OpenAI API key in the sidebar to use the Q&A Assistant."
                st.markdown(response_text)
                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
            else:
                context_chunks = retrieve_svp_guidelines(prompt, rag_chunks, top_k=3)
                
                chat_system_prompt = (
                    "You are an expert Procurement Policy Assistant specializing in Small Value Purchase (SVP) guidelines.\n"
                    "Use the following official guidelines context to answer the user's questions clearly, accurately, and concisely.\n"
                    "If the answer is not explicitly contained in the guidelines, answer using standard corporate procurement logic while noting policy boundaries.\n\n"
                    f"--- POLICY CONTEXT ---\n{context_chunks if context_chunks else SVP_PROCESS_DOC}\n----------------------"
                )

                client = OpenAI(api_key=user_api_key)
                
                # Build chat history context
                messages_for_llm = [{"role": "system", "content": chat_system_prompt}]
                for m in st.session_state.chat_messages:
                    messages_for_llm.append({"role": m["role"], "content": m["content"]})

                try:
                    stream = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages_for_llm,
                        temperature=0.2,
                        stream=True
                    )
                    
                    full_response = st.write_stream(stream)
                    st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    err_msg = f"Error generating response: {e}"
                    st.error(err_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})
