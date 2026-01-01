"""
================================================================================
SENTIMENT ANALYSIS WEB APPLICATION
================================================================================
Professional Streamlit application for real-time sentiment analysis

Author: [Your Name]
Date:  December 2024
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    APP_TITLE, APP_ICON, PAGE_LAYOUT, INITIAL_SIDEBAR_STATE,
    MIN_TEXT_LENGTH, EXAMPLES
)
from src.model_utils import load_model
from src. preprocessing import preprocess_text, TextPreprocessor
from src.ui_components import (
    render_custom_css, render_header, render_sentiment_result,
    render_confidence_scores, render_preprocessing_comparison,
    render_sidebar, render_examples, render_footer, render_batch_results
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource(show_spinner=False)
def initialize_app():
    """Initialize application resources"""
    model = load_model()
    preprocessor = TextPreprocessor()
    return model, preprocessor

with st.spinner("🔄 Loading models..."):
    sentiment_model, text_preprocessor = initialize_app()
    metadata = sentiment_model.get_model_info()

# ============================================================================
# RENDER UI
# ============================================================================

# Custom CSS
render_custom_css()

# Sidebar
show_preprocessing, show_probabilities, show_history = render_sidebar(metadata)

# Header
render_header()

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Analysis",
    "📊 Batch Analysis",
    "📈 Statistics",
    "ℹ️ About"
])

# ============================================================================
# TAB 1: SINGLE ANALYSIS
# ============================================================================

with tab1:
    st.markdown("### Enter Text to Analyze")
    
    # Text input
    text_input = st.text_area(
        "",
        height=150,
        placeholder="Type or paste your text here...\n\nExamples:\n• Product reviews\n• Customer feedback\n• Social media posts",
        key="main_text_input"
    )
    
    # Buttons
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Analysis
    if analyze_button:
        if text_input and len(text_input.strip()) >= MIN_TEXT_LENGTH:
            # Get prediction
            sentiment, confidence_scores, inference_time = sentiment_model. predict(text_input)
            
            # Update counter
            st.session_state. analysis_count += 1
            
            # Store in history
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({
                'text':  text_input[: 100] + '...' if len(text_input) > 100 else text_input,
                'sentiment': sentiment,
                'confidence': confidence_scores. get(sentiment, 0) if confidence_scores else 0,
                'timestamp': st.session_state.get('timestamp', 'N/A')
            })
            
            # Keep only last 10
            if len(st.session_state.history) > 10:
                st.session_state. history = st.session_state. history[-10:]
            
            if sentiment and confidence_scores:
                st.success("✅ Analysis Complete!")
                
                col1, col2 = st. columns([1, 1])
                
                with col1:
                    # Main result
                    confidence = confidence_scores.get(sentiment, 0)
                    render_sentiment_result(sentiment, confidence, inference_time)
                
                with col2:
                    # Confidence scores
                    if show_probabilities:
                        render_confidence_scores(confidence_scores, show_chart=True)
                
                # Preprocessing comparison
                if show_preprocessing:
                    st.markdown("---")
                    processed = preprocess_text(text_input)
                    stats = text_preprocessor.get_preprocessing_stats(text_input, processed)
                    render_preprocessing_comparison(text_input, processed, stats)
            
            else:
                st.warning("⚠️ Unable to analyze.  Text might be too short or contain only stopwords.")
        else:
            st.warning(f"⚠️ Please enter at least {MIN_TEXT_LENGTH} characters to analyze.")
    
    # Examples
    st.markdown("---")
    render_examples(EXAMPLES)
    
    # Handle example selection
    if 'example_text' in st. session_state:
        text_input = st.session_state.example_text
        del st.session_state.example_text

# ============================================================================
# TAB 2: BATCH ANALYSIS
# ============================================================================

with tab2:
    st.markdown("### 📊 Batch Text Analysis")
    st.info("Analyze multiple texts at once.  Enter one text per line.")
    
    batch_text = st.text_area(
        "Enter multiple texts (one per line):",
        height=200,
        placeholder="Enter texts here, one per line...\n\nExample:\nI love this product!\nThis is terrible.\nIt's okay."
    )
    
    if st.button("🔍 Analyze Batch", type="primary"):
        if batch_text: 
            lines = [line.strip() for line in batch_text.split('\n') 
                    if line.strip() and len(line.strip()) >= MIN_TEXT_LENGTH]
            
            if lines:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, line in enumerate(lines):
                    status_text.text(f"Processing {idx+1}/{len(lines)}...")
                    sentiment, confidence, _ = sentiment_model.predict(line)
                    
                    if sentiment and confidence:
                        results.append({
                            'Text': line[: 50] + '...' if len(line) > 50 else line,
                            'Sentiment':  sentiment,
                            'Confidence': f"{confidence. get(sentiment, 0)*100:.1f}%"
                        })
                    
                    progress_bar.progress((idx + 1) / len(lines))
                
                status_text.text("✅ Analysis complete!")
                
                if results:
                    render_batch_results(results)
            else:
                st.warning("No valid texts found.")

# ============================================================================
# TAB 3: STATISTICS
# ============================================================================

with tab3:
    st.markdown("### 📈 Model Performance Statistics")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metadata.get('test_accuracy', 0)*100:.2f}%")
    with col2:
        st.metric("Precision", f"{metadata.get('test_precision', 0)*100:.2f}%")
    with col3:
        st. metric("Recall", f"{metadata.get('test_recall', 0)*100:.2f}%")
    with col4:
        st.metric("F1-Score", f"{metadata.get('test_f1', 0)*100:.2f}%")
    
    st.markdown("---")
    
    # Model info
    st.markdown("### 🤖 Model Information")
    col1, col2 = st. columns(2)
    
    with col1:
        st. info(f"""
        **Model:** {metadata. get('model_name', 'N/A')}  
        **Training Date:** {metadata.get('training_date', 'N/A')}  
        **Training Samples:** {metadata.get('training_samples', 'N/A'):,}  
        **Test Samples:** {metadata.get('test_samples', 'N/A'):,}
        """)
    
    with col2:
        st.info(f"""
        **Vocabulary Size:** {metadata.get('vocabulary_size', 'N/A'):,}  
        **Max Features:** {metadata.get('max_features', 'N/A'):,}  
        **N-gram Range:** {metadata.get('ngram_range', 'N/A')}  
        **Inference Time:** {metadata.get('inference_time_ms', 'N/A'):.2f} ms
        """)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================

with tab4:
    from src.config import AUTHOR
    
    st.markdown("### ℹ️ About This Project")
    st.markdown(f"""
    ## 🎭 Sentiment Analysis System
    
    A professional web application for real-time sentiment analysis using Machine Learning. 
    
    ### 🎯 Purpose
    
    Built as part of the **{AUTHOR['project']}** for {AUTHOR['university']}.
    
    ### 🛠️ Technical Stack
    
    - **Backend:** Python, scikit-learn, NLTK
    - **Frontend:** Streamlit, Plotly
    - **ML:** TF-IDF + Logistic Regression
    - **Accuracy:** 85%+
    
    ### 👨‍💻 Author
    
    **{AUTHOR['name']}**  
    {AUTHOR['university']}  
    {AUTHOR['date']}
    
    ### 📞 Contact
    
    - 📧 Email: {AUTHOR['email']}
    - 🔗 GitHub: [Profile]({AUTHOR['github']})
    - 💼 LinkedIn: [Profile]({AUTHOR['linkedin']})
    
    ---
    
    Made with ❤️ using Streamlit
    """)

# Footer
render_footer()

# Initialize session state
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0

if 'history' not in st. session_state:
    st. session_state.history = []