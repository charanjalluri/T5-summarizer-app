"""
AI Summarizer - Working Version
"""

import streamlit as st
import hashlib

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except:
    GENAI_AVAILABLE = False

# ================================
# CONFIG
# ================================

st.set_page_config(
    page_title="🚀 AI Summarizer",
    page_icon="🤖",
    layout="wide"
)

# ================================
# AUTHENTICATION
# ================================

USERS = {
    "demo": hashlib.sha256("demo123".encode()).hexdigest(),
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
}

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if not st.session_state["authenticated"]:
        st.markdown("## 🔐 AI Summarizer Login")
        st.markdown("---")
        
        with st.form("login"):
            st.subheader("Please Login")
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                submitted = st.form_submit_button("🔓 Login", use_container_width=True)
            
            if submitted:
                pwd_hash = hashlib.sha256(pwd.encode()).hexdigest()
                if user in USERS and USERS[user] == pwd_hash:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = user
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        st.info("**Demo Account:**\n- Username: `demo`\n- Password: `demo123`")
        return False
    return True

# ================================
# GEMINI
# ================================

def get_client():
    if not GENAI_AVAILABLE:
        st.error("google-genai package not installed")
        return None
    
    key = None
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        key = st.secrets['GEMINI_API_KEY']
    
    if not key:
        with st.sidebar:
            st.markdown("### 🔑 API Configuration")
            key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Get your free key at https://aistudio.google.com/apikey"
            )
    
    if key:
        try:
            return genai.Client(api_key=key)
        except Exception as e:
            st.error(f"Failed to connect: {e}")
    return None

def summarize_text(client, text, style="concise", max_length=500):
    if not client or not text.strip():
        return ""
    
    style_prompts = {
        "concise": "Create a brief, concise summary in 3-5 sentences.",
        "detailed": "Create a comprehensive, detailed summary covering all key points.",
        "bullets": "Create a bullet-point summary with key takeaways and main ideas.",
        "academic": "Create a formal academic summary with technical detail."
    }
    
    prompt = f"""
{style_prompts.get(style, style_prompts['concise'])}

Keep the summary under {max_length} words.

TEXT TO SUMMARIZE:
{text[:10000]}
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text or "No summary generated."
    except Exception as e:
        return f"Error generating summary: {e}"

# ================================
# MAIN APP
# ================================

def main():
    # Check authentication
    if not check_password():
        return
    
    # Header
    st.title("🚀 AI Summarizer Pro")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        st.caption(f"👤 Logged in as: **{st.session_state.get('username', 'User')}**")
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            st.rerun()
    
    st.markdown("---")
    
    # Initialize Gemini client
    client = get_client()
    
    if not client:
        st.warning("⚠️ Please configure your Gemini API key in the sidebar")
        st.info("""
        **To get started:**
        1. Get a free API key: https://aistudio.google.com/apikey
        2. Enter it in the sidebar
        3. Start summarizing!
        """)
        return
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        style = st.selectbox(
            "📝 Summary Style",
            ["concise", "detailed", "bullets", "academic"],
            help="Choose how you want the summary formatted"
        )
        
        max_words = st.slider(
            "📏 Max Summary Length (words)",
            min_value=50,
            max_value=1000,
            value=200,
            step=50
        )
        
        st.markdown("---")
        
        # Stats
        if "summary_count" not in st.session_state:
            st.session_state.summary_count = 0
        
        st.markdown("### 📊 Statistics")
        st.metric("Total Summaries", st.session_state.summary_count)
    
    # Main content area
    st.subheader("📝 Text Summarization")
    
    # Text input
    text_input = st.text_area(
        "Enter or paste your text below:",
        height=300,
        placeholder="Paste your article, document, or any text you want to summarize...",
        help="Maximum 10,000 characters"
    )
    
    # Character count
    if text_input:
        char_count = len(text_input)
        st.caption(f"Characters: {char_count:,} / 10,000")
    
    # Summarize button
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        summarize_btn = st.button("✨ Summarize", type="primary", use_container_width=True)
    with col2:
        if text_input:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
            if clear_btn:
                st.rerun()
    
    # Process summarization
    if summarize_btn:
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to summarize")
        elif len(text_input) < 50:
            st.warning("⚠️ Text is too short. Please enter at least 50 characters.")
        else:
            with st.spinner("🤖 Generating summary..."):
                summary = summarize_text(client, text_input, style, max_words)
                
                if summary and not summary.startswith("Error"):
                    st.session_state.summary_count += 1
                    st.success("✅ Summary generated successfully!")
                    
                    # Display summary
                    st.markdown("### 📄 Summary")
                    st.markdown("---")
                    st.write(summary)
                    st.markdown("---")
                    
                    # Download options
                    st.markdown("### 💾 Download Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="📥 Download as TXT",
                            data=summary,
                            file_name="summary.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Create markdown version
                        md_content = f"# Summary\n\n**Style:** {style}\n\n**Generated:** {st.session_state.get('username', 'User')}\n\n---\n\n{summary}"
                        st.download_button(
                            label="📥 Download as MD",
                            data=md_content,
                            file_name="summary.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    
                    # Store in session for history
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "input": text_input[:200] + "..." if len(text_input) > 200 else text_input,
                        "summary": summary,
                        "style": style
                    })
                else:
                    st.error(summary)
    
    # History section
    if "history" in st.session_state and st.session_state.history:
        with st.expander("📜 Recent Summaries", expanded=False):
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                st.markdown(f"**Summary {len(st.session_state.history) - i}** ({item['style']})")
                st.caption(f"Input: {item['input']}")
                st.info(item['summary'])
                st.markdown("---")

if __name__ == "__main__":
    main()
