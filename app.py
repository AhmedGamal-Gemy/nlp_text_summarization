"""
Streamlit web app for text summarization.

Provides a web interface for both extractive and abstractive
summarization with ROUGE metrics and comparison.
"""

import time
import os
from pathlib import Path
import streamlit as st

import config
import preprocessing
from features import TFIDFExtractor, EmbeddingScorer
from extractive import ExtractiveSummarizer
from abstractive import AbstractiveSummarizer
from evaluate import compute_rouge, compression_ratio


# Get the diagrams directory path
DIAGRAMS_DIR = Path(__file__).parent / "diagrams"


@st.cache_resource
def load_extractive():
    """Load extractive summarizer models.

    Returns:
        ExtractiveSummarizer instance.
    """
    # Download NLTK resources
    preprocessing.download_nltk_resources()

    # Initialize models
    tfidf = TFIDFExtractor()
    embedder = EmbeddingScorer()

    # Fit on sample data for demo
    sample_corpus = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Natural language processing deals with text and speech data for various applications.",
        "Deep learning uses neural networks with multiple layers to learn representations.",
    ]
    tfidf.fit(sample_corpus)

    return ExtractiveSummarizer(tfidf, embedder)


@st.cache_resource
def load_abstractive():
    """Load abstractive summarizer model.

    Returns:
        AbstractiveSummarizer instance.
    """
    try:
        return AbstractiveSummarizer()
    except Exception as e:
        st.error(f"Failed to load BART model: {e}")
        return None


def main():
    """Main Streamlit app."""
    # Set page config
    st.set_page_config(page_title="Text Summarization", page_icon="📝", layout="wide")

    # Title
    st.title("Text Summarization")
    st.sidebar.title("Settings")

    # Model selector
    model_option = st.sidebar.radio(
        "Select Model", ["Extractive", "Abstractive", "Compare Both"]
    )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summarize", "Compare Models", "Architecture", "About"]
    )

    # TAB 1: Summarize
    with tab1:
        st.subheader("Summarize Article")

        # Input
        article = st.text_area(
            "Enter article text:",
            height=200,
            placeholder="Paste your article text here...",
        )

        # Reference (optional)
        reference = st.text_area(
            "Reference summary (optional for ROUGE):",
            height=100,
            placeholder="Paste reference summary here...",
        )

        # Button
        if st.button("Summarize", use_container_width=True):
            if not article:
                st.error("Please enter an article")
                return

            # Load models
            if model_option in ["Extractive", "Compare Both"]:
                with st.spinner("Loading extractive model..."):
                    ext_summarizer = load_extractive()

            if model_option in ["Abstractive", "Compare Both"]:
                with st.spinner("Loading abstractive model..."):
                    abs_summarizer = load_abstractive()

            # Run based on option
            if model_option == "Extractive":
                start = time.time()
                summary = ext_summarizer.summarize(article)
                elapsed = time.time() - start

                # Display
                st.success("Extractive Summary:")
                st.write(summary)

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", len(summary.split()))
                with col2:
                    ratio = compression_ratio(article, summary)
                    st.metric("Compression", f"{ratio:.1%}")
                with col3:
                    st.metric("Time", f"{elapsed:.2f}s")

                # ROUGE if reference
                if reference:
                    rouge = compute_rouge([summary], [reference])
                    st.subheader("ROUGE Scores")
                    rcol1, rcol2, rcol3 = st.columns(3)
                    with rcol1:
                        st.metric("ROUGE-1", f"{rouge['rouge1']:.3f}")
                    with rcol2:
                        st.metric("ROUGE-2", f"{rouge['rouge2']:.3f}")
                    with rcol3:
                        st.metric("ROUGE-L", f"{rouge['rougeL']:.3f}")

            elif model_option == "Abstractive":
                if not abs_summarizer:
                    st.error("Abstractive model not available")
                    return

                start = time.time()
                summary = abs_summarizer.summarize(article)
                elapsed = time.time() - start

                # Display
                st.success("Abstractive Summary:")
                st.write(summary)

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", len(summary.split()))
                with col2:
                    ratio = compression_ratio(article, summary)
                    st.metric("Compression", f"{ratio:.1%}")
                with col3:
                    st.metric("Time", f"{elapsed:.2f}s")

                # ROUGE if reference
                if reference:
                    rouge = compute_rouge([summary], [reference])
                    st.subheader("ROUGE Scores")
                    rcol1, rcol2, rcol3 = st.columns(3)
                    with rcol1:
                        st.metric("ROUGE-1", f"{rouge['rouge1']:.3f}")
                    with rcol2:
                        st.metric("ROUGE-2", f"{rouge['rouge2']:.3f}")
                    with rcol3:
                        st.metric("ROUGE-L", f"{rouge['rougeL']:.3f}")

            else:  # Compare Both
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Extractive")
                    start = time.time()
                    ext_summary = ext_summarizer.summarize(article)
                    ext_time = time.time() - start
                    st.write(ext_summary)

                    ext_ratio = compression_ratio(article, ext_summary)
                    st.caption(f"Words: {len(ext_summary.split())}")
                    st.caption(f"Compression: {ext_ratio:.1%}")
                    st.caption(f"Time: {ext_time:.2f}s")

                    if reference:
                        rouge = compute_rouge([ext_summary], [reference])
                        st.caption(f"ROUGE-1: {rouge['rouge1']:.3f}")

                with col2:
                    st.subheader("Abstractive")
                    if not abs_summarizer:
                        st.error("Model not available")
                    else:
                        start = time.time()
                        abs_summary = abs_summarizer.summarize(article)
                        abs_time = time.time() - start
                        st.write(abs_summary)

                        abs_ratio = compression_ratio(article, abs_summary)
                        st.caption(f"Words: {len(abs_summary.split())}")
                        st.caption(f"Compression: {abs_ratio:.1%}")
                        st.caption(f"Time: {abs_time:.2f}s")

                        if reference:
                            rouge = compute_rouge([abs_summary], [reference])
                            st.caption(f"ROUGE-1: {rouge['rouge1']:.3f}")

    # TAB 2: Compare Models
    with tab2:
        st.subheader("Compare Models")

        article_cmp = st.text_area("Enter article:", height=150, key="cmp_article")
        reference_cmp = st.text_area("Reference:", height=80, key="cmp_ref")

        if st.button("Run Comparison", use_container_width=True):
            if not article_cmp:
                st.error("Please enter an article")
                return

            # Load models
            with st.spinner("Loading models..."):
                ext_summarizer = load_extractive()
                abs_summarizer = load_abstractive()

            # Run both
            ext_start = time.time()
            ext_summary = ext_summarizer.summarize(article_cmp)
            ext_time = time.time() - ext_start

            if abs_summarizer:
                abs_start = time.time()
                abs_summary = abs_summarizer.summarize(article_cmp)
                abs_time = time.time() - abs_start

            # Show side by side
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Extractive")
                st.write(ext_summary)
                ext_ratio = compression_ratio(article_cmp, ext_summary)
                st.metric(
                    "ROUGE-1",
                    "TBD"
                    if not reference_cmp
                    else f"{compute_rouge([ext_summary], [reference_cmp])['rouge1']:.3f}",
                )
                st.metric("Compression", f"{ext_ratio:.1%}")
                st.metric("Time", f"{ext_time:.2f}s")

            with c2:
                st.subheader("Abstractive")
                if abs_summarizer:
                    st.write(abs_summary)
                    abs_ratio = compression_ratio(article_cmp, abs_summary)
                    if reference_cmp:
                        rouge = compute_rouge([abs_summary], [reference_cmp])
                        st.metric("ROUGE-1", f"{rouge['rouge1']:.3f}")
                    else:
                        st.metric("ROUGE-1", "N/A")
                    st.metric("Compression", f"{abs_ratio:.1%}")
                    st.metric("Time", f"{abs_time:.2f}s")
                else:
                    st.error("Model not available")

    # TAB 3: Architecture - Show system diagrams
    with tab3:
        st.subheader("System Architecture")

        # Check if diagrams exist
        arch_svg = DIAGRAMS_DIR / "system_architecture.svg"
        extract_svg = DIAGRAMS_DIR / "extractive_pipeline.svg"
        abstract_svg = DIAGRAMS_DIR / "abstractive_pipeline.svg"
        flow_svg = DIAGRAMS_DIR / "data_flow.svg"

        # System Architecture Overview
        if arch_svg.exists():
            st.markdown("### 🔭 System Overview")
            st.image(str(arch_svg), use_container_width=True)
        else:
            st.info("System architecture diagram not found")

        # Extractive Pipeline
        if extract_svg.exists():
            st.markdown("### 📌 Extractive Pipeline")
            st.markdown("""
            **How it works:**
            1. Split article into sentences using NLTK
            2. Score each sentence with TF-IDF and sentence embeddings
            3. Combine scores (50% each) and select top-K sentences
            4. Restore original order
            """)
            st.image(str(extract_svg), use_container_width=True)
        else:
            st.info("Extractive pipeline diagram not found")

        # Abstractive Pipeline
        if abstract_svg.exists():
            st.markdown("### ✨ Abstractive Pipeline (BART)")
            st.markdown("""
            **How it works:**
            1. Tokenize article with BART tokenizer (max 1024 tokens)
            2. Encode with 12-layer transformer encoder
            3. Generate with beam search (beams=4, length penalty=2.0)
            4. Decode tokens back to text
            """)
            st.image(str(abstract_svg), use_container_width=True)
        else:
            st.info("Abstractive pipeline diagram not found")

        # Data Flow Sequence
        if flow_svg.exists():
            st.markdown("### 🔄 Data Flow Sequence")
            st.markdown("Complete workflow from input to output:")
            st.image(str(flow_svg), use_container_width=True)
        else:
            st.info("Data flow diagram not found")

        # Legend
        st.markdown("---")
        st.caption(
            "💡 Diagrams show the complete flow from article input to summary output"
        )

    # TAB 4: About
    with tab4:
        st.subheader("About")

        st.markdown("""
        ## Text Summarization System
        
        This system provides two approaches to automatic text summarization:
        
        ### Extractive Summarization
        - Uses TF-IDF term frequency weights
        - Uses sentence embedding similarity
        - Selects top-k original sentences from the article
        
        ### Abstractive Summarization  
        - Uses facebook/bart-large-cnn pretrained model
        - Generates new fluent summaries
        - Trained on CNN/DailyMail dataset
        """)

        # Results table
        st.subheader("Expected Results")

        results_data = {
            "Model": ["Extractive", "Abstractive (BART)"],
            "ROUGE-1": ["~0.30", "~0.42"],
            "ROUGE-2": ["~0.15", "~0.20"],
            "ROUGE-L": ["~0.25", "~0.35"],
            "Compression": ["~0.15", "~0.20"],
        }

        st.dataframe(results_data)

        # Metrics cards
        st.subheader("Key Metrics")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("BART ROUGE-1", "~0.42")
        with mc2:
            st.metric("Extractive ROUGE-1", "~0.30")
        with mc3:
            st.metric("Dataset", "5000 samples")
        with mc4:
            st.metric("BART Parameters", "406M")


if __name__ == "__main__":
    main()
