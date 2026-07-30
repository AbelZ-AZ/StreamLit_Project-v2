import streamlit as st

# --- 1. USER DATABASE & AUTHENTICATION ---
USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "user1": {"password": "user123", "role": "User"},
}


def authenticate(username, password):
    user = USERS.get(username)
    if user and user["password"] == password:
        return user["role"]
    return None


# --- 2. INITIALIZE SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# --- 3. LOGIN PAGE FUNCTION ---
def login_page():
    st.title("🔐 Login Page")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            role = authenticate(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.role = role
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")


# --- 4. LOGOUT FUNCTION ---
def logout():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.rerun()


# --- 5. MULTI-PAGE NAVIGATION LOGIC ---
# Updated paths to point to the "pages" subdirectory
login_screen = st.Page(login_page, title="Login", icon="🔒")
page_1 = st.Page("pages/page1.py", title="About Us", icon="🤖")
page_2 = st.Page("pages/page2.py", title="Methodology", icon="📊")
page_3 = st.Page("pages/page3.py", title="Cost Reasonableness Assessment", icon="📊")

# Route based on login status
if not st.session_state.logged_in:
    # Hide the sidebar navigation while logged out
    pg = st.navigation([login_screen], position="hidden")
else:
    # Expose the sub-pages after login
    pg = st.navigation([page_1, page_2, page_3])
    st.sidebar.button("Log out", on_click=logout)

# Run the selected page
pg.run()