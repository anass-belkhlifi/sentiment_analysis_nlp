"""
================================================================================
UI COMPONENTS MODULE
================================================================================
Reusable Streamlit UI components
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from .config import SENTIMENT_CONFIG, AUTHOR


def render_custom_css():
    """Render custom CSS styles"""
    st.markdown("""
        <style>
        /* Main header */
        .main-header {
            font-size: 3. 5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 0.5rem;
        }
        
        /* Sub header */
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* Result card */
        .result-card {
            padding: 2rem;
            border-radius:  15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin:  1rem 0;
            text-align: center;
            color: white;
        }
        
        /* Text area styling */
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            font-size: 1.1rem;
        }
        
        . stTextArea textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 0.2rem rgba(102,126,234,0.25);
        }
        
        /* Button styling */
        .stButton > button {
            border-radius:  25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #999;
            border-top: 1px solid #e0e0e0;
            margin-top: 3rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render application header"""
    st.markdown('<h1 class="main-header">🎭 Sentiment Analysis System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze sentiment in text using advanced NLP and Machine Learning</p>', unsafe_allow_html=True)


def render_sentiment_result(sentiment: str, confidence: float, inference_time: float):
    """
    Render sentiment analysis result
    
    Args:
        sentiment:  Predicted sentiment
        confidence:  Confidence score (0-1)
        inference_time: Processing time in ms
    """
    emoji = SENTIMENT_CONFIG.get(sentiment, {}).get('emoji', '🤔')
    gradient = SENTIMENT_CONFIG.get(sentiment, {}).get('gradient', 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)')
    
    st.markdown(f"""
        <div class="result-card" style="background: {gradient};">
            <div style="font-size: 5rem;">{emoji}</div>
            <div style="font-size: 2.5rem; font-weight: bold; margin:  1rem 0;">{sentiment}</div>
            <div style="font-size: 1.5rem; opacity: 0.9;">Confidence: {confidence*100:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.info(f"⚡ Processing time: {inference_time:. 2f} ms")


def render_confidence_scores(confidence_dict: Dict[str, float], show_chart: bool = True):
    """
    Render confidence scores
    
    Args:
        confidence_dict: Dictionary of sentiment -> confidence
        show_chart: Whether to show plotly chart
    """
    st.markdown("### 📊 Confidence Scores")
    
    if show_chart:
        import plotly.graph_objects as go
        
        colors = [SENTIMENT_CONFIG.get(k, {}).get('color', '#999') for k in confidence_dict.keys()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(confidence_dict.keys()),
                y=[v*100 for v in confidence_dict.values()],
                marker=dict(color=colors, line=dict(color='white', width=2)),
                text=[f'{v*100:.1f}%' for v in confidence_dict. values()],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Confidence: %{y:. 1f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            xaxis_title='Sentiment',
            yaxis_title='Confidence (%)',
            yaxis=dict(range=[0, 100]),
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Progress bars
    for label, prob in sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True):
        st.write(f"**{label}**")
        st.progress(float(prob))
        st.write(f"{prob*100:.2f}%")
        st.write("")


def render_preprocessing_comparison(original: str, processed: str, stats: dict):
    """
    Render preprocessing comparison
    
    Args:
        original: Original text
        processed: Processed text
        stats: Preprocessing statistics
    """
    st.markdown("### 🔧 Preprocessing Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Text:**")
        st.text_area("", original, height=100, key="orig_comp", disabled=True)
    
    with col2:
        st.markdown("**Processed Text:**")
        st.text_area("", processed, height=100, key="proc_comp", disabled=True)
    
    st.info(f"""
    **Statistics:**  
    • Original:  {stats['original_length']} chars, {stats['original_words']} words  
    • Processed: {stats['processed_length']} chars, {stats['processed_words']} words  
    • Reduction: {stats['reduction_percent']:.1f}%
    """)


def render_sidebar(metadata: dict):
    """
    Render sidebar content
    
    Args:
        metadata: Model metadata
    """
    with st.sidebar:
        st. markdown("# 🎭 Sentiment Analysis")
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Info")
        st.info(f"""
        **Model:** {metadata. get('model_name', 'N/A')}  
        **Accuracy:** {metadata.get('test_accuracy', 0)*100:.2f}%  
        **F1-Score:** {metadata.get('test_f1', 0)*100:.2f}%
        """)
        
        # How it works
        st.markdown("### 🔍 How It Works")
        st.markdown("""
        1. **Enter** your text
        2. **Click** Analyze button
        3. **View** sentiment result
        4. **See** confidence scores
        """)
        
        # Features
        st.markdown("### ✨ Features")
        st.markdown("""
        - ✅ Real-time analysis
        - ✅ Multi-domain support
        - ✅ Confidence scores
        - ✅ Batch processing
        - ✅ History tracking
        """)
        
        # Statistics
        st.markdown("### 📈 Session Stats")
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        
        col1, col2 = st. columns(2)
        with col1:
            st.metric("Analyses", st.session_state.analysis_count)
        with col2:
            st.metric("Accuracy", f"{metadata.get('test_accuracy', 0)*100:.1f}%")
        
        # Settings
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_preprocessing = st.checkbox("Show preprocessing", value=False)
        show_probabilities = st.checkbox("Show probabilities", value=True)
        show_history = st.checkbox("Show history", value=False)
        
        # About
        st.markdown("---")
        st.markdown("### 👨‍💻 About")
        st.markdown(f"""
        **{AUTHOR['project']}**  
        {AUTHOR['university']}  
        {AUTHOR['date']}
        
        [📁 GitHub]({AUTHOR['github']})  
        [📧 Contact](mailto:{AUTHOR['email']})
        """)
        
        return show_preprocessing, show_probabilities, show_history


def render_examples(examples: dict):
    """
    Render example buttons
    
    Args:
        examples: Dictionary of label -> example text
    """
    st.markdown("### 💡 Try These Examples")
    
    cols = st.columns(3)
    for idx, (label, example) in enumerate(examples.items()):
        with cols[idx % 3]:
            if st.button(f"📝 {label}", key=f"example_{idx}", use_container_width=True):
                st.session_state.example_text = example
                st.rerun()


def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown(f"""
        <div class="footer">
            <p><strong>Sentiment Analysis System</strong> | Built with Streamlit & scikit-learn</p>
            <p>© 2024 {AUTHOR['name']} | {AUTHOR['project']}</p>
            <p>
                <a href="{AUTHOR['github']}" target="_blank">GitHub</a> • 
                <a href="mailto:{AUTHOR['email']}">Contact</a>
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_batch_results(results: List[dict]):
    """
    Render batch analysis results
    
    Args:
        results: List of result dictionaries
    """
    if not results:
        st.warning("No results to display")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Display table
    st.success(f"✅ Analyzed {len(results)} texts successfully!")
    st.dataframe(df, use_container_width=True)
    
    # Statistics
    sentiment_counts = df['Sentiment'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive_count = sentiment_counts.get('Positive', 0)
        st.metric("😊 Positive", positive_count, f"{positive_count/len(results)*100:.1f}%")
    
    with col2:
        negative_count = sentiment_counts.get('Negative', 0)
        st.metric("😢 Negative", negative_count, f"{negative_count/len(results)*100:.1f}%")
    
    with col3:
        neutral_count = sentiment_counts.get('Neutral', 0)
        st.metric("😐 Neutral", neutral_count, f"{neutral_count/len(results)*100:.1f}%")
    
    # Visualization
    import plotly.express as px
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts. index,
        title='Sentiment Distribution',
        color=sentiment_counts.index,
        color_discrete_map={k: v['color'] for k, v in SENTIMENT_CONFIG.items()}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )