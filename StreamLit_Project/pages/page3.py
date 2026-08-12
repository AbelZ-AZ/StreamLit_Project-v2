import streamlit as st
import pandas as pd
import urllib.parse

# Set page configuration
st.set_page_config(
    page_title="Market Price & Retailer Search Verification",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce & Retailer Search Verification App")
st.write("Scans market items and routes search links directly to target e-commerce platforms.")

# -----------------------------------------------------------------------------
# SAMPLE DATA INITIALIZATION
# -----------------------------------------------------------------------------
if "market_items" not in st.session_state:
    st.session_state.market_items = [
        {
            "desc": "Logitech MX Master 3S Wireless Mouse",
            "source_name": "Shopee",
            "price": "$129.00",
            "url": "https://shopee.sg/"
        },
        {
            "desc": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
            "source_name": "Lazada",
            "price": "$499.00",
            "url": "https://www.lazada.sg/"
        },
        {
            "desc": "Apple iPad Air 11-inch M2",
            "source_name": "Amazon",
            "price": "$899.00",
            "url": "https://www.amazon.sg/dp/B0D3HV1234"  # Direct item URL example
        },
        {
            "desc": "Anker PowerCore 20000mAh Power Bank",
            "source_name": "Challenger",
            "price": "$59.90",
            "url": ""
        },
        {
            "desc": "Nespresso Vertuo Pop Coffee Machine",
            "source_name": "FairPrice",
            "price": "$198.00",
            "url": ""
        },
        {
            "desc": "Keychron K2 Wireless Mechanical Keyboard",
            "source_name": "Courts",
            "price": "$139.00",
            "url": ""
        }
    ]

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_platform_search_url(source_name: str, desc: str) -> str:
    """
    Constructs a direct search URL targeting the platform's internal search engine.
    If the platform is unknown, defaults to a site-restricted Google search.
    """
    encoded_item = urllib.parse.quote(desc.strip())
    s_lower = source_name.strip().lower()

    if "shopee" in s_lower:
        return f"https://shopee.sg/search?keyword={encoded_item}"
    elif "lazada" in s_lower:
        return f"https://www.lazada.sg/catalog/?q={encoded_item}"
    elif "amazon" in s_lower:
        return f"https://www.amazon.sg/s?k={encoded_item}"
    elif "challenger" in s_lower or "hachi" in s_lower:
        return f"https://www.challenger.sg/search?query={encoded_item}"
    elif "fairprice" in s_lower:
        return f"https://www.fairprice.com.sg/search?query={encoded_item}"
    elif "courts" in s_lower:
        return f"https://www.courts.com.sg/catalogsearch/result/?q={encoded_item}"
    elif "harvey" in s_lower or "norman" in s_lower:
        return f"https://www.harveynorman.com.sg/catalogsearch/result/?q={encoded_item}"
    else:
        # Fallback: Google search strictly restricted to the platform domain
        clean_domain = s_lower.replace(" ", "") + ".com.sg"
        return f"https://www.google.com/search?q=site%3A{clean_domain}+{encoded_item}"

# -----------------------------------------------------------------------------
# MAIN APP INTERFACE - TABS
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Market Data & Direct Links", "➕ Add Item"])

with tab1:
    st.subheader("Scanned Market Items")
    
    # Table Header
    h_col1, h_col2, h_col3, h_col4 = st.columns([3, 1.5, 1.5, 3])
    h_col1.markdown("**Item Description**")
    h_col2.markdown("**Retailer / Platform**")
    h_col3.markdown("**Price**")
    h_col4.markdown("**Direct Action Links**")
    st.divider()

    # Iterate through market data items
    for idx, m_item in enumerate(st.session_state.market_items):
        desc = m_item.get("desc", "").strip()
        source_name = m_item.get("source_name", "Retailer").strip()
        price = m_item.get("price", "N/A")
        url = m_item.get("url", "").strip()

        # Generate targeted platform search link
        platform_search_url = get_platform_search_url(source_name, desc)

        r_col1, r_col2, r_col3, r_col4 = st.columns([3, 1.5, 1.5, 3])
        r_col1.write(desc)
        r_col2.write(f"**{source_name}**")
        r_col3.write(price)

        # Rendering link logic:
        # Check if url looks like a specific item page (has product indicators)
        has_direct_item_link = url.startswith("http") and any(
            path in url for path in ["/p/", "/dp/", "/product/", "item", "catalog"]
        )

        if has_direct_item_link:
            r_col4.markdown(f"[🔗 Direct Item]({url}) · [🔍 Search {source_name}]({platform_search_url})")
        else:
            r_col4.markdown(f"[🔍 Search on {source_name}]({platform_search_url})")

        st.markdown("---")

with tab2:
    st.subheader("Add New Item for Market Verification")
    with st.form("add_item_form"):
        new_desc = st.text_input("Item Description", placeholder="e.g. Sony WH-1000XM5 Headphones")
        new_source = st.selectbox("Retailer / Platform", ["Shopee", "Lazada", "Amazon", "Challenger", "FairPrice", "Courts", "Harvey Norman", "Other"])
        if new_source == "Other":
            new_source = st.text_input("Specify Platform Name", placeholder="e.g. Audio-Technica")
        
        new_price = st.text_input("Price (Optional)", placeholder="e.g. $299.00")
        new_url = st.text_input("Direct Product Link (Optional)", placeholder="https://...")

        submitted = st.form_submit_button("Add Item")
        if submitted:
            if new_desc.strip() and new_source.strip():
                st.session_state.market_items.append({
                    "desc": new_desc.strip(),
                    "source_name": new_source.strip(),
                    "price": new_price.strip() if new_price.strip() else "N/A",
                    "url": new_url.strip()
                })
                st.success(f"Added '{new_desc}' successfully!")
                st.rerun()
            else:
                st.error("Please fill in both the item description and retailer name.")
