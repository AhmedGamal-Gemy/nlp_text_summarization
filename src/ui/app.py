"""
Professional Streamlit web app for text summarization.

Features:
- Custom CSS theming with gradient accents
- Professional dashboard layout
- Model comparison with charts
- 4 model options: Extractive, BART, Fine-tuned BART, Compare All
"""

import time
from pathlib import Path
import streamlit as st

# Only lightweight imports at startup
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "diagrams"


# Lazy import helper
def _import_config():
    from .. import config
    return config

def _import_preprocessing():
    from .. import preprocessing
    return preprocessing

def _import_features():
    from ..features import TFIDFExtractor, EmbeddingScorer
    return TFIDFExtractor, EmbeddingScorer

def _import_extractive():
    from ..extractive import ExtractiveSummarizer
    return ExtractiveSummarizer

def _import_abstractive():
    from ..abstractive import AbstractiveSummarizer
    return AbstractiveSummarizer

def _import_evaluation():
    from ..services.evaluation import compute_rouge, compression_ratio, compute_bertscore
    return compute_rouge, compression_ratio, compute_bertscore


# ============================================================================
# CUSTOM CSS
# ============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
        .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
        h1 { font-size: 2.25rem !important; font-weight: 700 !important; letter-spacing: -0.02em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em; }
        .summary-box { background: #f8fafc; border-left: 4px solid #667eea; border-radius: 8px; padding: 1.25rem; margin: 1rem 0; font-size: 1rem; line-height: 1.7; }
        .model-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
        .badge-extractive { background: #dbeafe; color: #1e40af; }
        .badge-abstractive { background: #fef3c7; color: #92400e; }
        .badge-finetuned { background: #d1fae5; color: #065f46; }
        section[data-testid="stSidebar"] { background: #1e293b; }
        section[data-testid="stSidebar"] * { color: #f8fafc !important; }
        .stButton > button { border-radius: 8px; font-weight: 600; padding: 0.75rem 1.5rem; }
        textarea { border-radius: 8px !important; border: 1px solid #e2e8f0 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 0.75rem 1.5rem; }
        .footer { text-align: center; padding: 2rem 0; color: #64748b; font-size: 0.875rem; }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING (lazy imports inside)
# ============================================================================

@st.cache_resource
def load_extractive():
    config = _import_config()
    preprocessing = _import_preprocessing()
    TFIDFExtractor, EmbeddingScorer = _import_features()
    ExtractiveSummarizer = _import_extractive()

    preprocessing.download_nltk_resources()
    tfidf = TFIDFExtractor()
    embedder = EmbeddingScorer()
    sample_corpus = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Natural language processing deals with text and speech data for various applications.",
        "Deep learning uses neural networks with multiple layers to learn representations.",
    ]
    tfidf.fit(sample_corpus)
    return ExtractiveSummarizer(tfidf, embedder)


@st.cache_resource
def load_abstractive():
    try:
        AbstractiveSummarizer = _import_abstractive()
        return AbstractiveSummarizer()
    except Exception as e:
        st.error(f"Failed to load BART model: {e}")
        return None


@st.cache_resource
def load_finetuned():
    config = _import_config()
    finetuned_path = PROJECT_ROOT / "checkpoints" / "bart-finetuned"
    if not finetuned_path.exists():
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(finetuned_path))
        model.to(config.DEVICE)

        class FinetunedSummarizer:
            def __init__(self, tokenizer, model):
                self.tokenizer = tokenizer
                self.model = model

            def summarize(self, text: str) -> str:
                inputs = self.tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(config.DEVICE)
                outputs = self.model.generate(inputs["input_ids"], max_length=config.BART_MAX_LEN, min_length=config.BART_MIN_LEN, num_beams=config.BART_BEAMS, length_penalty=config.BART_LENGTH_PENALTY, no_repeat_ngram_size=config.BART_NO_REPEAT_NGRAM, early_stopping=config.BART_EARLY_STOPPING)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return FinetunedSummarizer(tokenizer, model)
    except Exception as e:
        st.error(f"Failed to load fine-tuned model: {e}")
        return None


# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_summary_box(text: str, model_type: str):
    badges = {
        "extractive": ("badge-extractive", "Extractive"),
        "abstractive": ("badge-abstractive", "Abstractive (BART)"),
        "finetuned": ("badge-finetuned", "Fine-tuned BART"),
    }
    badge_class, badge_text = badges.get(model_type, ("badge-extractive", "Unknown"))
    st.markdown(f"""
    <span class="model-badge {badge_class}">{badge_text}</span>
    <div class="summary-box">{text}</div>
    """, unsafe_allow_html=True)


def display_results_metrics(words: int, compression: float, time_sec: float, rouge: dict = None, bertscore: dict = None):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", words)
    with col2:
        st.metric("Compression", f"{compression:.1%}")
    with col3:
        st.metric("Time", f"{time_sec:.2f}s")

    if rouge:
        st.markdown("---")
        st.markdown("**ROUGE Scores**")
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.metric("ROUGE-1", f"{rouge['rouge1']:.3f}")
        with rcol2:
            st.metric("ROUGE-2", f"{rouge['rouge2']:.3f}")
        with rcol3:
            st.metric("ROUGE-L", f"{rouge['rougeL']:.3f}")

    if bertscore:
        st.markdown("---")
        st.markdown("**BERTScore (Semantic Similarity)**")
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            st.metric("Precision", f"{bertscore['precision']:.3f}")
        with bcol2:
            st.metric("Recall", f"{bertscore['recall']:.3f}")
        with bcol3:
            st.metric("F1", f"{bertscore['f1']:.3f}")


def run_model_and_display(summarizer, article: str, reference: str, model_type: str):
    compute_rouge, compression_ratio, compute_bertscore = _import_evaluation()
    
    with st.spinner(f"Running {model_type} summarization..."):
        start = time.time()
        summary = summarizer.summarize(article)
        elapsed = time.time() - start

    display_summary_box(summary, model_type)
    ratio = compression_ratio(article, summary)

    if reference:
        rouge = compute_rouge([summary], [reference])
        bertscore = compute_bertscore([summary], [reference])
        display_results_metrics(len(summary.split()), ratio, elapsed, rouge, bertscore)
    else:
        display_results_metrics(len(summary.split()), ratio, elapsed)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="NLP Text Summarization",
        page_icon="\U0001F4DD",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("# \U0001F4DD Summarizer")
        st.markdown("---")

        st.markdown("### \U0001F916 Model Selection")
        model_option = st.radio(
            "Choose model:",
            ["Extractive", "Abstractive (BART)", "Fine-tuned BART", "Compare All"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Configuration Panel
        with st.expander("\u2699\ufe0f Configuration", expanded=False):
            config = _import_config()

            st.markdown("**Extractive Settings**")
            top_k = st.slider("Sentences to extract", 1, 5, config.TOP_K_SENTENCES, key="top_k")
            tfidf_w = st.slider("TF-IDF weight", 0.0, 1.0, config.TFIDF_WEIGHT, 0.1, key="tfidf_w")

            st.markdown("**BART Settings**")
            bart_max = st.slider("Max tokens", 50, 200, config.BART_MAX_LEN, 10, key="bart_max")
            bart_min = st.slider("Min tokens", 10, 100, config.BART_MIN_LEN, 5, key="bart_min")
            bart_beams = st.slider("Beam size", 1, 10, config.BART_BEAMS, 1, key="bart_beams")
            bart_len_pen = st.slider("Length penalty", 0.5, 3.0, config.BART_LENGTH_PENALTY, 0.1, key="bart_len_pen")

            # Apply config updates
            config.update_from_ui(
                top_k=top_k,
                tfidf_weight=tfidf_w,
                bart_max_len=bart_max,
                bart_min_len=bart_min,
                bart_beams=bart_beams,
                bart_length_penalty=bart_len_pen,
            )

            st.caption(f"TF-IDF: {config.TFIDF_WEIGHT:.1f} | Embed: {config.EMBED_WEIGHT:.1f}")

        st.markdown("---")
        st.markdown("### \U0001F4CA Quick Stats")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Extractive ROUGE-1 | 0.290 |
        | Abstractive ROUGE-1 | 0.391 |
        | Dataset | CNN/DailyMail |
        | Samples | 5,000 |
        """)

        st.markdown("---")
        st.markdown("### \u2139\ufe0f About")
        st.caption("Dual-model NLP text summarization with extractive (TF-IDF + embeddings) and abstractive (BART) approaches.")

    # Main content
    st.markdown("# \U0001F4DD NLP Text Summarization System")
    st.markdown("Generate concise summaries using **extractive** or **abstractive** approaches.")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "\u270F\ufe0f Summarize",
        "\U0001F4CA Compare Models",
        "\U0001F3D7\ufe0f Architecture",
        "\U0001F4D6 Documentation",
        "\U0001F4C8 Visualizations",
    ])

    # ========================================================================
    # TAB 1: Summarize
    # ========================================================================
    with tab1:
        col_input, col_output = st.columns([1, 1])

        with col_input:
            st.markdown("### \U0001F4C4 Input Article")
            article = st.text_area(
                "Paste your article text:",
                height=250,
                placeholder="Enter or paste the article text you want to summarize...",
                label_visibility="collapsed",
            )

            reference = st.text_area(
                "Reference summary (optional - for ROUGE evaluation):",
                height=100,
                placeholder="Paste a reference summary to compute ROUGE scores...",
                label_visibility="collapsed",
            )

            summarize_btn = st.button("\U0001F680 Generate Summary", use_container_width=True, type="primary")

        with col_output:
            st.markdown("### \U0001F4CB Summary Output")

            if summarize_btn:
                if not article:
                    st.error("Please enter an article text.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    load_ext = model_option in ["Extractive", "Compare All"]
                    load_bart = model_option in ["Abstractive (BART)", "Compare All"]
                    load_ft = model_option in ["Fine-tuned BART", "Compare All"]

                    if load_ext:
                        status_text.text("\U0001F4CC Loading Extractive model...")
                        progress_bar.progress(15)
                        with st.spinner("\u2699\ufe0f Initializing TF-IDF vectorizer and embedding scorer..."):
                            ext_summarizer = load_extractive()
                        progress_bar.progress(40)

                    if load_bart:
                        status_text.text("\u2728 Loading Abstractive model...")
                        progress_bar.progress(55)
                        with st.spinner("\U0001F916 Loading Facebook BART model..."):
                            abs_summarizer = load_abstractive()
                        progress_bar.progress(70)

                    if load_ft:
                        status_text.text("\U0001F527 Loading Fine-tuned BART model...")
                        progress_bar.progress(80)
                        with st.spinner("\U0001F527 Loading fine-tuned checkpoint..."):
                            ft_summarizer = load_finetuned()
                        progress_bar.progress(90)

                    if model_option == "Extractive":
                        status_text.text("\U0001F50D Extracting key sentences...")
                        run_model_and_display(ext_summarizer, article, reference, "extractive")

                    elif model_option == "Abstractive (BART)":
                        if not abs_summarizer:
                            st.error("Abstractive model not available.")
                        else:
                            status_text.text("\u2728 Generating summary with BART...")
                            run_model_and_display(abs_summarizer, article, reference, "abstractive")

                    elif model_option == "Fine-tuned BART":
                        if not ft_summarizer:
                            st.warning("No fine-tuned model found. Run `uv run python train.py --max_samples 100 --epochs 1` to create one.")
                        else:
                            status_text.text("\U0001F527 Generating with fine-tuned BART...")
                            run_model_and_display(ft_summarizer, article, reference, "finetuned")

                    else:  # Compare All
                        status_text.text("\U0001F504 Running all models...")
                        cols = st.columns(3)

                        with cols[0]:
                            st.markdown("**Extractive**")
                            run_model_and_display(ext_summarizer, article, reference, "extractive")

                        with cols[1]:
                            st.markdown("**Abstractive (BART)**")
                            if abs_summarizer:
                                run_model_and_display(abs_summarizer, article, reference, "abstractive")
                            else:
                                st.error("Model not available")

                        with cols[2]:
                            st.markdown("**Fine-tuned BART**")
                            if ft_summarizer:
                                run_model_and_display(ft_summarizer, article, reference, "finetuned")
                            else:
                                st.warning("No fine-tuned model")

                    progress_bar.progress(100)
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
            else:
                st.info("Enter an article and click **Generate Summary** to see results.")

    # ========================================================================
    # TAB 2: Compare Models
    # ========================================================================
    with tab2:
        st.markdown("### \U0001F4CA Model Comparison Dashboard")
        st.markdown("Run both models side-by-side and compare their outputs.")

        article_cmp = st.text_area(
            "Enter article:", height=150, key="cmp_article",
            placeholder="Paste article text here...",
        )
        reference_cmp = st.text_area(
            "Reference summary (optional):", height=80, key="cmp_ref",
            placeholder="Paste reference summary...",
        )

        if st.button("\U0001F50D Run Comparison", use_container_width=True, type="primary"):
            if not article_cmp:
                st.error("Please enter an article.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("\U0001F4CC Loading models...")
                progress_bar.progress(20)
                with st.spinner("\u2699\ufe0f Initializing models..."):
                    ext_summarizer = load_extractive()
                    abs_summarizer = load_abstractive()
                    ft_summarizer = load_finetuned()
                progress_bar.progress(40)

                status_text.text("\U0001F50D Running Extractive...")
                progress_bar.progress(60)
                with st.spinner("\U0001F4CA Processing with TF-IDF + embeddings..."):
                    ext_start = time.time()
                    ext_summary = ext_summarizer.summarize(article_cmp)
                    ext_time = time.time() - ext_start

                abs_summary = None
                abs_time = 0
                if abs_summarizer:
                    status_text.text("\u2728 Running Abstractive...")
                    progress_bar.progress(75)
                    with st.spinner("\U0001F916 Generating with BART beam search..."):
                        abs_start = time.time()
                        abs_summary = abs_summarizer.summarize(article_cmp)
                        abs_time = time.time() - abs_start

                ft_summary = None
                ft_time = 0
                if ft_summarizer:
                    status_text.text("\U0001F527 Running Fine-tuned...")
                    progress_bar.progress(90)
                    with st.spinner("\U0001F527 Generating with fine-tuned BART..."):
                        ft_start = time.time()
                        ft_summary = ft_summarizer.summarize(article_cmp)
                        ft_time = time.time() - ft_start

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("#### \U0001F4CC Extractive")
                    display_summary_box(ext_summary, "extractive")
                    compression_ratio = _import_evaluation()[1]
                    ext_ratio = compression_ratio(article_cmp, ext_summary)
                    compute_rouge = _import_evaluation()[0]
                    st.metric("ROUGE-1", "N/A" if not reference_cmp else f"{compute_rouge([ext_summary], [reference_cmp])['rouge1']:.3f}")
                    st.metric("Compression", f"{ext_ratio:.1%}")
                    st.metric("Time", f"{ext_time:.2f}s")

                with c2:
                    st.markdown("#### \u2728 Abstractive")
                    if abs_summary:
                        display_summary_box(abs_summary, "abstractive")
                        compression_ratio = _import_evaluation()[1]
                        abs_ratio = compression_ratio(article_cmp, abs_summary)
                        compute_rouge = _import_evaluation()[0]
                        st.metric("ROUGE-1", "N/A" if not reference_cmp else f"{compute_rouge([abs_summary], [reference_cmp])['rouge1']:.3f}")
                        st.metric("Compression", f"{abs_ratio:.1%}")
                        st.metric("Time", f"{abs_time:.2f}s")
                    else:
                        st.error("Model not available")

                with c3:
                    st.markdown("#### \U0001F527 Fine-tuned")
                    if ft_summary:
                        display_summary_box(ft_summary, "finetuned")
                        compression_ratio = _import_evaluation()[1]
                        ft_ratio = compression_ratio(article_cmp, ft_summary)
                        compute_rouge = _import_evaluation()[0]
                        st.metric("ROUGE-1", "N/A" if not reference_cmp else f"{compute_rouge([ft_summary], [reference_cmp])['rouge1']:.3f}")
                        st.metric("Compression", f"{ft_ratio:.1%}")
                        st.metric("Time", f"{ft_time:.2f}s")
                    else:
                        st.warning("No fine-tuned model")

                progress_bar.progress(100)
                status_text.text("\u2705 Comparison complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

    # ========================================================================
    # TAB 3: Architecture
    # ========================================================================
    with tab3:
        st.markdown("### \U0001F3D7\ufe0f System Architecture")
        st.markdown("Professional diagrams showing the complete summarization pipeline.")

        # System Architecture Overview
        arch_svg = DIAGRAMS_DIR / "nlp_summarization_architecture_v3.svg"
        if arch_svg.exists():
            st.markdown("#### \U0001F52D System Architecture Overview")
            st.image(str(arch_svg), use_container_width=True)

        # Extractive + Abstractive side by side
        extract_svg = DIAGRAMS_DIR / "Extractive Summarization Pipeline.svg"
        abstract_svg = DIAGRAMS_DIR / "Abstractive Summarization Pipeline.svg"
        
        col1, col2 = st.columns(2)
        with col1:
            if extract_svg.exists():
                st.markdown("#### \U0001F4CC Extractive Pipeline")
                st.image(str(extract_svg), use_container_width=True)
        with col2:
            if abstract_svg.exists():
                st.markdown("#### \u2728 Abstractive Pipeline (BART)")
                st.image(str(abstract_svg), use_container_width=True)

        # Data Flow (full width)
        flow_svg = DIAGRAMS_DIR / "System Data Flow Sequence Diagram.svg"
        if flow_svg.exists():
            st.markdown("#### \U0001F504 System Data Flow Sequence Diagram")
            st.image(str(flow_svg), use_container_width=True)

        # Fine-tuning Pipeline (full width)
        train_svg = DIAGRAMS_DIR / "BART Fine-tuning Pipeline.svg"
        if train_svg.exists():
            st.markdown("#### \U0001F527 BART Fine-tuning Pipeline")
            st.image(str(train_svg), use_container_width=True)

        # Model Comparison Dashboard
        compare_svg = DIAGRAMS_DIR / "Model Comparison Dashboard.svg"
        if compare_svg.exists():
            st.markdown("#### \U0001F4CA Model Comparison Dashboard")
            st.image(str(compare_svg), use_container_width=True)

    # ========================================================================
    # TAB 4: Documentation
    # ========================================================================
    with tab4:
        st.markdown("### \U0001F4D6 Documentation")

        st.markdown("""
        ## Text Summarization System

        This system provides **three** approaches to automatic text summarization:

        ### Extractive Summarization
        - Uses **TF-IDF** term frequency weights
        - Uses **sentence embedding** similarity (all-MiniLM-L6-v2)
        - Selects top-k original sentences from the article
        - **Fast** (~25s for 200 articles), works on CPU
        - **ROUGE-1**: ~0.29

        ### Abstractive Summarization (Pretrained)
        - Uses **facebook/bart-large-cnn** pretrained model
        - Generates new fluent summaries
        - Fine-tuned on CNN/DailyMail dataset by Facebook
        - **Higher quality** but slower (~505s for 200 articles)
        - **ROUGE-1**: ~0.39

        ### Fine-tuned BART (Bonus Model)
        - Further fine-tuned on CNN/DailyMail with our training pipeline
        - Adapts BART to our specific data distribution
        - Shows improvement over pretrained baseline
        """)

        st.markdown("---")
        st.markdown("### \U0001F4CA Results Comparison")

        results_data = {
            "Model": ["Extractive", "Abstractive (BART)", "Fine-tuned BART"],
            "ROUGE-1": ["0.290", "0.391", "TBD"],
            "ROUGE-2": ["0.099", "0.169", "TBD"],
            "ROUGE-L": ["0.188", "0.284", "TBD"],
            "Compression": ["18.9%", "8.6%", "TBD"],
            "Time (200 samples)": ["25.2s", "504.8s", "TBD"],
        }

        st.dataframe(results_data, use_container_width=True)

        st.markdown("---")
        st.markdown("### \U0001F511 Key Metrics")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("BART ROUGE-1", "0.391")
        with mc2:
            st.metric("Extractive ROUGE-1", "0.290")
        with mc3:
            st.metric("Dataset", "5,000 samples")
        with mc4:
            st.metric("BART Parameters", "406M")

        st.markdown("---")
        st.markdown("### \U0001F4DA Learn More")
        st.markdown("""
        - **Learning Guide**: See `docs/LEARNING_GUIDE.md`
        - **Team Structure**: See `docs/TEAM_STRUCTURE.md`
        - **Code Comments**: All source files have educational comments
        """)

    # ========================================================================
    # TAB 5: Visualizations
    # ========================================================================
    with tab5:
        st.markdown("### \U0001F4C8 Model Visualizations")
        st.markdown("Charts and diagrams comparing model performance.")

        viz_path = PROJECT_ROOT / "src" / "visualizations"
        viz_path.mkdir(exist_ok=True)

        # Default results
        results = {
            'Extractive': {
                'rouge1': 0.290, 'rouge2': 0.099, 'rougeL': 0.188,
                'compression': 0.189, 'time': 25.2
            },
            'Abstractive (BART)': {
                'rouge1': 0.391, 'rouge2': 0.169, 'rougeL': 0.284,
                'compression': 0.086, 'time': 504.8
            },
        }

        # Generate charts on demand
        if st.button("\U0001F504 Generate Charts", type="primary"):
            with st.spinner("Generating visualizations..."):
                from src.visualization import SummarizationVisualizer
                viz = SummarizationVisualizer(output_dir=str(viz_path))
                viz.generate_all_charts(results)
            st.success("Charts generated!")

        # Display existing charts
        col1, col2 = st.columns(2)
        
        with col1:
            rouge_chart = viz_path / "rouge_comparison.png"
            if rouge_chart.exists():
                st.markdown("**ROUGE Score Comparison**")
                st.image(str(rouge_chart), use_container_width=True)
            
            radar_chart = viz_path / "performance_radar.png"
            if radar_chart.exists():
                st.markdown("**Performance Radar**")
                st.image(str(radar_chart), use_container_width=True)
        
        with col2:
            comp_chart = viz_path / "compression_comparison.png"
            if comp_chart.exists():
                st.markdown("**Compression Ratio**")
                st.image(str(comp_chart), use_container_width=True)

        # Word frequency analysis
        st.markdown("---")
        st.markdown("### \U0001F524 Word Frequency Analysis")
        
        freq_text = st.text_area("Enter text to analyze word frequency:", height=100, placeholder="Paste article text here...")
        if st.button("\U0001F4CA Analyze Words"):
            if freq_text:
                from src.visualization import SummarizationVisualizer
                viz = SummarizationVisualizer(output_dir=str(viz_path))
                path = viz.plot_word_frequency(freq_text)
                st.image(str(path), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="footer">Built with Streamlit | Powered by BART & TF-IDF | NLP Text Summarization System</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
