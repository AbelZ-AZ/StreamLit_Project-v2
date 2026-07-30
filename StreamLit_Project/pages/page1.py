import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="About Us | GenAI Procurement Bot",
    page_icon="🤖",
    layout="centered"
)

# --- HEADER SECTION ---
st.title("🤖 GenAI Procurement Assistant")
st.subheader("Smart tools for smarter sourcing.")
st.divider()

# --- MAIN CONTENT CARD ---
# Using a container to group the main "About Us" message
with st.container(border=True):
    st.markdown("## 📋 About Us")
    st.write(
        """
        This GenAI Bot is designed to help **buyers** and **procurement officers** quickly and accurately assess price reasonableness for Small Value Purchases (SVP).
        
        By leveraging artificial intelligence, the tool reduces manual verification time, 
        ensures compliance, and provides data-driven insights at a glance.
        """
    )

st.write("") # Add some spacing

# --- KEY METRIC SECTION ---
# Using columns and a metric widget to highlight the SVP threshold beautifully
st.markdown("### 🔍 Procurement Scope")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.metric(
        label="SVP Threshold Limit", 
        value="< S$6,000", 
        help="Excluding GST"
    )

with col2:
    st.markdown("#### What is an SVP?")
    st.info(
        "**Small Value Purchases (SVP)** refer to procurement items valued below "
        "**S$6,000 (excluding GST)**. This bot is optimized specifically to handle "
        "the unique benchmarking requirements of these fast-tracked purchases."
    )

# --- FOOTER ---
st.divider()
st.caption("© 2026 GenAI Procurement Bot | Internal Use Only")