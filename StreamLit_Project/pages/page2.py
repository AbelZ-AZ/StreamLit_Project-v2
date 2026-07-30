import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Methodology",
    page_icon="📊",
    layout="centered"
)

# 2. Header Elements
st.title("📊 Methodology")
st.subheader("How we evaluate price reasonableness.")
st.divider()

# 3. Introductory Text
st.markdown(
    "Price reasonableness is evaluated using a **multi-layered approach** "
    "to ensure accurate, data-driven procurement decisions."
)
st.write("") 

# 4. Interactive Visual Tabs
tab1, tab2, tab3 = st.tabs([
    "🌐 Market Benchmarking", 
    "📜 Historical Data", 
    "💱 Currency Adjustment"
])

with tab1:
    st.markdown("### Market Benchmarking")
    st.info("📊 **Core Action:** Comparison against live online prices.")
    st.write(
        "The bot scans verified external e-commerce and vendor websites in real-time. "
        "This ensures that your quotes are aligned with current market realities and "
        "prevents overpaying for readily available commercial items."
    )

with tab2:
    st.markdown("### Historical Data")
    st.success("🗄️ **Core Action:** Evaluation against our database of past purchases.")
    st.write(
        "By looking inward, the bot cross-references the item description and specifications "
        "with previous organisational procurement records. This helps maintain internal pricing "
        "consistency and flags inflation or sudden vendor price hikes."
    )

with tab3:
    st.markdown("### Currency Adjustment")
    st.warning("🔄 **Core Action:** Standardised using current FX rates.")
    st.write(
        "To ensure a true 'apples-to-apples' comparison, all cross-border pricing data "
        "and foreign vendor quotes are automatically converted and standardised using up-to-date "
        "Foreign Exchange (FX) rates."
    )

# 5. Footer
st.divider()
st.caption("GenAI Procurement Assistant • Methodology Overview")