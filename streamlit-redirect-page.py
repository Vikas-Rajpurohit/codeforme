import streamlit as st
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="Website Update Notice",
    page_icon="🚀",
    layout="centered"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .subheader {
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        color: #3B82F6;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1.5rem;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .progress-container {
        margin: 2rem 0;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Website Migration Notice</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">We\'ve upgraded our platform for a better experience</p>', unsafe_allow_html=True)

# Information Box
st.markdown("""
<div class="info-box">
    <p>Our website has moved to a new location with an improved user interface and enhanced features. 
    You'll be automatically redirected to the new site shortly, or you can click the button below to go there now.</p>
</div>
""", unsafe_allow_html=True)

# URL Information
old_url = "http://eun043653-8082/"
new_url = "http://int-intranet.nomurnow.com/nucleus/dqinsights/"
user_guide = "https://confluence.nomura.com/ETCB/confluence/display/DataOffice/DQ+Insights"

# Display URLs
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Current location:**  \n{old_url}")
with col2:
    st.success(f"**New location:**  \n{new_url}")

# New Features Section
st.markdown('<p class="subheader">What\'s New in the Updated UI</p>', unsafe_allow_html=True)

with st.expander("Explore New Features and Improvements", expanded=True):
    features = [
        {
            "title": "Modernized Dashboard Interface",
            "description": "Cleaner layout with improved data visualization capabilities and intuitive navigation."
        },
        {
            "title": "Enhanced Search Functionality",
            "description": "Advanced filtering options and faster search results across all data insights."
        },
        {
            "title": "Personalized User Experience",
            "description": "Custom dashboard configurations that remember your preferences and most-used features."
        },
        {
            "title": "Improved Performance",
            "description": "Faster loading times and more responsive interface, even when handling large datasets."
        },
        {
            "title": "New Analysis Tools",
            "description": "Additional analytical features to help you derive more value from your data."
        }
    ]
    
    for feature in features:
        st.markdown(f"""
        <p class="feature-title">✨ {feature['title']}</p>
        <p>{feature['description']}</p>
        <hr>
        """, unsafe_allow_html=True)

# User Guide Link
st.markdown('<p class="subheader">Resources to Get Started</p>', unsafe_allow_html=True)
st.markdown(f"""
<div class="info-box">
    <p><strong>Need help navigating the new interface?</strong></p>
    <p>Our comprehensive user guide provides step-by-step instructions on using all the new features:</p>
    <p><a href="{user_guide}" target="_blank">{user_guide}</a></p>
</div>
""", unsafe_allow_html=True)

# Redirect button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("Go to New Website Now", key="redirect_button", use_container_width=True):
    st.markdown(f'<meta http-equiv="refresh" content="0;url={new_url}">', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Automatic redirect progress bar
st.markdown('<div class="progress-container">', unsafe_allow_html=True)
st.write("You will be automatically redirected in 15 seconds...")
progress_bar = st.progress(0)
for i in range(100):
    # Update progress bar
    progress_bar.progress(i + 1)
    # Sleep for a short time to simulate progress
    time.sleep(0.15)
    # If completed, redirect
    if i == 99:
        st.markdown(f'<meta http-equiv="refresh" content="0;url={new_url}">', unsafe_allow_html=True)
        break
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>If you encounter any issues during the transition, please contact the support team.</p>
    <p>Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)
