"""
Generate the Text Summarization System Report as a PDF.
Matches the exact structure and formatting of the sample report.
Expanded to match the 7-page depth of the reference report.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import black, HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT


def create_report(output_path="reports/text_summarization_report.pdf"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2.0*cm,
        leftMargin=2.0*cm,
        topMargin=2.0*cm,
        bottomMargin=2.0*cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        'RptTitle',
        parent=styles['Title'],
        fontSize=18,
        leading=24,
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        spaceAfter=16,
        alignment=TA_CENTER,
        fontName='Helvetica',
        textColor=HexColor('#333333'),
    ))
    styles.add(ParagraphStyle(
        'RptHeading',
        parent=styles['Heading1'],
        fontSize=14,
        leading=20,
        spaceBefore=16,
        spaceAfter=8,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptSubHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=5,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptSubSubHeading',
        parent=styles['Heading3'],
        fontSize=11,
        leading=15,
        spaceBefore=6,
        spaceAfter=3,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=15,
        spaceAfter=5,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptBullet',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        leftIndent=20,
        bulletIndent=8,
        spaceAfter=2,
        fontName='Helvetica',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptNumbered',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        leftIndent=20,
        bulletIndent=8,
        spaceAfter=2,
        fontName='Helvetica',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptFormula',
        parent=styles['Normal'],
        fontSize=12,
        leading=18,
        spaceBefore=8,
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptFormulaLabel',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceBefore=6,
        spaceAfter=2,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptExample',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        leftIndent=15,
        rightIndent=15,
        spaceAfter=4,
        fontName='Helvetica-Oblique',
        textColor=HexColor('#444444'),
    ))
    styles.add(ParagraphStyle(
        'RptExLabel',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceBefore=6,
        spaceAfter=2,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptTblHdr',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        fontName='Helvetica-Bold',
        textColor=black,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'RptTblCell',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        fontName='Helvetica',
        textColor=black,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'RptTblCellL',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        fontName='Helvetica',
        textColor=black,
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        'RptEnd',
        parent=styles['Normal'],
        fontSize=12,
        leading=18,
        spaceBefore=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptCaption',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceBefore=4,
        spaceAfter=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique',
        textColor=HexColor('#555555'),
    ))

    B = styles['RptBody']
    H = styles['RptHeading']
    SH = styles['RptSubHeading']
    SSH = styles['RptSubSubHeading']
    BL = styles['RptBullet']
    NB = styles['RptNumbered']
    FM = styles['RptFormula']
    FL = styles['RptFormulaLabel']
    EX = styles['RptExample']
    EL = styles['RptExLabel']
    TH = styles['RptTblHdr']
    TC = styles['RptTblCell']
    TL = styles['RptTblCellL']

    story = []

    # ========================================================================
    # TITLE
    # ========================================================================
    story.append(Spacer(1, 16))
    story.append(Paragraph("Text Summarization System Report", styles['RptTitle']))
    story.append(Spacer(1, 4))
    story.append(Paragraph("TF-IDF Extractive and Transformer-Based Abstractive Summarization", styles['RptSubtitle']))
    story.append(Spacer(1, 10))

    # ========================================================================
    # 1. PROBLEM DESCRIPTION
    # ========================================================================
    story.append(Paragraph("1. Problem Description", H))

    story.append(Paragraph(
        "Text summarization is one of the most important and widely studied applications in Natural Language "
        "Processing (NLP). In today's information-rich world, the volume of digital text produced daily is "
        "enormous \u2014 including news articles, research papers, social media posts, and business documents. "
        "Reading all of this content manually is impractical, which creates a critical need for automated "
        "systems that can condense long texts into shorter, meaningful versions.", B))

    story.append(Paragraph(
        "The main goal of text summarization is to reduce a long text into a shorter version while preserving "
        "the most important information, key facts, and main ideas. A good summary should be concise, coherent, "
        "and faithful to the original content \u2014 it should not introduce new information or distort the "
        "original meaning.", B))

    story.append(Paragraph(
        "This project implements a complete text summarization system using two fundamentally different "
        "approaches to address this challenge:", B))

    story.append(Paragraph("1. <b>Extractive Summarization</b> \u2014 Baseline Method", BL))
    story.append(Paragraph("2. <b>Abstractive Summarization</b> \u2014 Advanced Method", BL))

    story.append(Paragraph(
        "The extractive model works by selecting the most important sentences directly from the original text. "
        "It uses statistical scoring (TF-IDF) combined with semantic similarity (sentence embeddings) to rank "
        "and select sentences. The abstractive model uses a Transformer-based architecture, specifically the "
        "BART (Bidirectional and Auto-Regressive Transformers) model, which reads the entire article and "
        "generates entirely new fluent summaries that can paraphrase and synthesize information.", B))

    story.append(Paragraph("The system was designed to achieve the following objectives:", B))
    story.append(Paragraph("\u2022 Keep important information from the original article", BL))
    story.append(Paragraph("\u2022 Reduce the length of the text significantly", BL))
    story.append(Paragraph("\u2022 Preserve the meaning and context of the original article", BL))
    story.append(Paragraph("\u2022 Compare traditional statistical methods with modern deep learning techniques", BL))
    story.append(Paragraph("\u2022 Provide an interactive graphical user interface for easy use", BL))

    # ========================================================================
    # 2. DATASET USED
    # ========================================================================
    story.append(Paragraph("2. Dataset Used", H))

    story.append(Paragraph(
        "The project uses articles and their corresponding human-written summaries (highlights) for both "
        "training and evaluation. The system is designed to work with various types of text content, "
        "making it versatile for multiple applications:", B))

    story.append(Paragraph("\u2022 <b>News Articles:</b> Daily news from major outlets like CNN and DailyMail", BL))
    story.append(Paragraph("\u2022 <b>Blog Posts:</b> Long-form blog articles and opinion pieces", BL))
    story.append(Paragraph("\u2022 <b>Product Reviews:</b> Customer reviews and feedback summaries", BL))
    story.append(Paragraph("\u2022 <b>General Long Documents:</b> Any lengthy text requiring condensation", BL))

    story.append(Paragraph(
        "For this project, we use the <b>CNN/DailyMail dataset</b>, which is one of the most widely used "
        "benchmarks for text summarization research. The abstractive model specifically uses "
        "<b>facebook/bart-large-cnn</b>, a version of BART that has been pre-trained and fine-tuned on "
        "this exact dataset, giving it strong prior knowledge of news summarization patterns.", B))

    story.append(Paragraph("<b>Dataset Statistics:</b>", B))
    story.append(Paragraph("\u2022 <b>Source:</b> News articles from CNN and DailyMail", BL))
    story.append(Paragraph("\u2022 <b>Total Size:</b> 5000 samples (configurable, full dataset has 300k+)", BL))
    story.append(Paragraph("\u2022 <b>Data Split:</b> 80% train (4000), 10% validation (500), 10% test (500)", BL))
    story.append(Paragraph("\u2022 <b>Average Article Length:</b> ~700 words per article", BL))
    story.append(Paragraph("\u2022 <b>Average Highlights Length:</b> ~50 words per summary", BL))
    story.append(Paragraph("\u2022 <b>Domain:</b> News journalism (politics, sports, entertainment, world)", BL))

    story.append(Paragraph(
        "Evaluation was performed on 200 test samples to compare the generated summaries from both "
        "the extractive and abstractive methods against the human-written reference summaries. This "
        "sample size provides stable ROUGE estimates with approximately 95% confidence.", B))

    # ========================================================================
    # 3. PREPROCESSING STEPS
    # ========================================================================
    story.append(Paragraph("3. Preprocessing Steps", H))

    story.append(Paragraph(
        "Before any feature extraction or summarization can take place, the raw text must be cleaned "
        "and transformed into a format suitable for processing. Preprocessing is a critical step that "
        "significantly affects the quality of the final summaries. The following preprocessing pipeline "
        "is applied to all input text:", B))

    story.append(Paragraph("<b>1. Convert text to lowercase</b>", NB))
    story.append(Paragraph(
        "All characters are converted to lowercase to ensure uniformity. This prevents the system from "
        "treating \"The\" and \"the\" as different words, which would artificially inflate the vocabulary size.", B))
    story.append(Paragraph('Example: "The Food Was Amazing" \u2192 "the food was amazing"', EX))

    story.append(Paragraph("<b>2. Remove punctuation</b>", NB))
    story.append(Paragraph(
        "Symbols such as commas, periods, exclamation marks, question marks, and special characters are "
        "removed. This allows the system to focus purely on word content without being distracted by "
        "punctuation marks that don't carry semantic meaning for scoring purposes.", B))
    story.append(Paragraph('Example: "Hello, world!" \u2192 "hello world"', EX))

    story.append(Paragraph("<b>3. Remove stopwords</b>", NB))
    story.append(Paragraph(
        "Common function words like \"the\", \"is\", \"and\", \"of\", \"in\", and \"to\" are removed "
        "because they appear in almost every document and do not carry important distinguishing meaning. "
        "Removing them reduces noise and focuses the model on content-bearing words.", B))
    story.append(Paragraph('Example: "the cat is on the mat" \u2192 "cat mat"', EX))

    story.append(Paragraph("<b>4. Tokenization</b>", NB))
    story.append(Paragraph(
        "Text is split into individual words (word tokens) and sentences (sentence tokens). Tokenization "
        "is the fundamental step that breaks continuous text into discrete units that can be processed, "
        "scored, and analyzed by the summarization algorithms.", B))

    story.append(Paragraph("<b>5. Sentence Segmentation</b>", NB))
    story.append(Paragraph(
        "Articles are divided into individual sentences using the NLTK (Natural Language Toolkit) sentence "
        "tokenizer. This is especially critical for extractive summarization, where the core task is to "
        "rank individual sentences and select the most important ones. The sentence tokenizer handles "
        "edge cases like abbreviations (\"Dr.\", \"Mr.\") and decimal numbers correctly.", B))

    story.append(Paragraph(
        "These preprocessing steps collectively improve feature extraction quality and ensure that both "
        "the TF-IDF scoring and embedding similarity computations operate on clean, normalized text.", B))

    # ========================================================================
    # 4. FEATURE EXTRACTION
    # ========================================================================
    story.append(Paragraph("4. Feature Extraction", H))

    story.append(Paragraph(
        "Feature extraction is the process of converting raw text into numerical representations that "
        "capture the importance and meaning of sentences. Two complementary feature extraction techniques "
        "are used in this system to evaluate sentence importance:", B))

    story.append(Paragraph("A) TF-IDF \u2014 Term Frequency \u2013 Inverse Document Frequency", SH))

    story.append(Paragraph(
        "TF-IDF is a statistical measure that evaluates how important a word is to a document within a "
        "collection of documents (corpus). It combines two components:", B))

    story.append(Paragraph(
        "<b>Term Frequency (TF):</b> How often a word appears in a specific sentence. Words that appear "
        "more frequently in a sentence are considered more important to that sentence's meaning.", BL))
    story.append(Paragraph(
        "<b>Inverse Document Frequency (IDF):</b> How rare a word is across the entire corpus. Words that "
        "appear in many documents (like \"the\", \"is\") get low scores, while rare, domain-specific words "
        "get high scores.", BL))

    story.append(Paragraph(
        "The TF-IDF score for each sentence is calculated by summing the TF-IDF weights of all words "
        "in that sentence. Sentences with higher total TF-IDF scores are considered more informative.", B))

    story.append(Paragraph("<b>Advantages:</b>", B))
    story.append(Paragraph("\u2022 Simple and fast computation \u2014 no neural networks required", BL))
    story.append(Paragraph("\u2022 Effective for extractive summarization tasks", BL))
    story.append(Paragraph("\u2022 Easy to interpret and explain \u2014 scores are transparent", BL))
    story.append(Paragraph("\u2022 Works well on CPU without GPU acceleration", BL))

    story.append(Paragraph("B) Sentence Embeddings", SH))

    story.append(Paragraph(
        "Sentence embeddings convert entire sentences into dense numerical vectors (arrays of numbers) "
        "that capture semantic meaning. Unlike TF-IDF which only counts word frequencies, embeddings "
        "understand the actual meaning and context of sentences.", B))

    story.append(Paragraph(
        "The model used is <b>all-MiniLM-L6-v2</b>, a lightweight sentence transformer that produces "
        "384-dimensional vectors. Despite being small (only 80MB), it provides excellent quality for "
        "semantic similarity tasks. The model works by encoding each sentence into a vector where "
        "semantically similar sentences are close together in the vector space.", B))

    story.append(Paragraph(
        "Cosine similarity is used to measure how similar each sentence is to the overall document "
        "representation (the average of all sentence embeddings). Sentences that are most similar "
        "to the document's central theme receive higher scores.", B))

    story.append(Paragraph("<b>Advantages:</b>", B))
    story.append(Paragraph("\u2022 Captures semantic similarity beyond exact word matching", BL))
    story.append(Paragraph("\u2022 Understands sentence meaning and context", BL))
    story.append(Paragraph("\u2022 Improves sentence ranking quality significantly over TF-IDF alone", BL))
    story.append(Paragraph("\u2022 Can recognize that different words can express the same idea", BL))

    # ========================================================================
    # 5. BASELINE METHOD: EXTRACTIVE SUMMARIZATION
    # ========================================================================
    story.append(Paragraph("5. Baseline Method: Extractive Summarization", H))

    story.append(Paragraph(
        "The baseline approach uses extractive summarization, which is the simpler and more traditional "
        "method. Instead of generating new text, it selects the most important sentences directly from "
        "the original article and combines them to form the summary. This approach guarantees that all "
        "content in the summary comes from the original text, eliminating any risk of hallucination.", B))

    story.append(Paragraph("<b>How it works \u2014 Step by Step:</b>", B))

    story.append(Paragraph("1. <b>Sentence Splitting:</b> The article is split into individual sentences using NLTK's sentence tokenizer.", NB))
    story.append(Paragraph("2. <b>TF-IDF Scoring:</b> Each sentence is scored using TF-IDF. The TF-IDF vectorizer is first fitted on training articles to learn the vocabulary, then each sentence's score is computed as the sum of TF-IDF weights of its words.", NB))
    story.append(Paragraph("3. <b>Embedding Generation:</b> Each sentence is converted into a 384-dimensional vector using the all-MiniLM-L6-v2 sentence transformer model.", NB))
    story.append(Paragraph("4. <b>Document Representation:</b> The document-level embedding is computed as the average (mean) of all sentence embeddings, representing the overall topic.", NB))
    story.append(Paragraph("5. <b>Cosine Similarity:</b> The cosine similarity between each sentence embedding and the document embedding is calculated. This measures how well each sentence captures the main theme.", NB))
    story.append(Paragraph("6. <b>Score Combination:</b> The TF-IDF score and embedding similarity score are combined using a weighted average formula.", NB))
    story.append(Paragraph("7. <b>Top-k Selection:</b> The sentences with the highest combined scores are selected (default k=3, configurable from 1-5).", NB))
    story.append(Paragraph("8. <b>Order Restoration:</b> The selected sentences are rearranged in their original order from the article to produce a coherent final summary.", NB))

    story.append(Paragraph("<b>Scoring Formula:</b>", B))
    story.append(Paragraph("Final Score = 0.5 \u00d7 TF-IDF Score + 0.5 \u00d7 Embedding Similarity", FM))
    story.append(Paragraph(
        "The weights (0.5 each) can be adjusted through the Streamlit UI to favor either statistical "
        "importance (TF-IDF) or semantic relevance (embeddings).", B))

    story.append(Paragraph("<b>Advantages:</b>", B))
    story.append(Paragraph("\u2022 <b>Fast execution:</b> ~25 seconds for 200 articles on CPU", BL))
    story.append(Paragraph("\u2022 <b>Preserves original wording:</b> No modification of source sentences", BL))
    story.append(Paragraph("\u2022 <b>No hallucination risk:</b> All content is from the original article", BL))
    story.append(Paragraph("\u2022 <b>Easy to debug:</b> Sentence scores are interpretable", BL))
    story.append(Paragraph("\u2022 <b>Works on CPU:</b> No GPU required for inference", BL))

    story.append(Paragraph("<b>Disadvantages:</b>", B))
    story.append(Paragraph("\u2022 Cannot paraphrase or generate new text", BL))
    story.append(Paragraph("\u2022 Sometimes produces disconnected summaries (selected sentences may not flow together)", BL))
    story.append(Paragraph("\u2022 Limited by the quality of sentences available in the source", BL))
    story.append(Paragraph("\u2022 Cannot combine information from multiple sentences into one", BL))

    # ========================================================================
    # 6. ADVANCED METHOD: TRANSFORMER-BASED SUMMARIZATION
    # ========================================================================
    story.append(Paragraph("6. Advanced Method: Transformer-Based Summarization", H))

    story.append(Paragraph(
        "The advanced approach uses abstractive summarization with the BART model, which represents "
        "the state-of-the-art in neural text generation. Unlike extractive methods that simply select "
        "existing sentences, abstractive summarization reads the entire article and generates entirely "
        "new text that captures the key information in a fluent, human-like manner.", B))

    story.append(Paragraph("<b>Model Architecture:</b>", B))
    story.append(Paragraph(
        "BART (Bidirectional and Auto-Regressive Transformers) is a sequence-to-sequence (seq2seq) model "
        "developed by Facebook AI Research. It combines two powerful concepts:", B))

    story.append(Paragraph(
        "\u2022 <b>Bidirectional Encoder:</b> Reads the entire input article at once, understanding context "
        "from both directions (left-to-right and right-to-left). This is similar to how BERT works.", BL))
    story.append(Paragraph(
        "\u2022 <b>Auto-Regressive Decoder:</b> Generates the summary one token (word piece) at a time, "
        "using the encoded representation and previously generated tokens to predict the next word. "
        "This is similar to how GPT works.", BL))

    story.append(Paragraph(
        "The specific model used is <b>facebook/bart-large-cnn</b> with 406 million parameters. "
        "The \"cnn\" suffix indicates it has been fine-tuned specifically on the CNN/DailyMail "
        "summarization dataset, giving it strong prior knowledge of news summarization style and format.", B))

    story.append(Paragraph("<b>Generation Process \u2014 Step by Step:</b>", B))
    story.append(Paragraph("1. <b>Tokenization:</b> The input article is converted into token IDs using the BART tokenizer. Articles longer than 1024 tokens are truncated to fit the model's input limit.", NB))
    story.append(Paragraph("2. <b>Encoding:</b> The tokenized article is passed through BART's encoder, which produces contextual representations that capture the meaning of each token in the context of the entire article.", NB))
    story.append(Paragraph("3. <b>Decoding with Beam Search:</b> The decoder generates the summary token by token. Instead of greedily picking the most likely next word, beam search maintains multiple candidate summaries (beams=4) and selects the sequence with the highest overall probability.", NB))
    story.append(Paragraph("4. <b>Length Control:</b> The generation is constrained by minimum (30) and maximum (130) token limits to ensure summaries are neither too short nor too long.", NB))
    story.append(Paragraph("5. <b>Repetition Prevention:</b> No-repeat n-gram constraint (size=4) prevents the model from repeating the same 4-word phrases, improving summary quality.", NB))
    story.append(Paragraph("6. <b>Decoding:</b> The generated token IDs are converted back to human-readable text, with special tokens removed.", NB))

    story.append(Paragraph("<b>Generation Parameters:</b>", B))
    story.append(Paragraph("\u2022 <b>Beam Search:</b> 4 beams (balances quality and speed)", BL))
    story.append(Paragraph("\u2022 <b>Length Penalty:</b> 1.0 (neutral \u2014 neither encourages nor discourages length)", BL))
    story.append(Paragraph("\u2022 <b>No Repeat N-Gram Size:</b> 4 (prevents repetitive phrases)", BL))
    story.append(Paragraph("\u2022 <b>Early Stopping:</b> Disabled (allows full beam exploration)", BL))
    story.append(Paragraph("\u2022 <b>Max Length:</b> 130 tokens (approximately 80-100 words)", BL))
    story.append(Paragraph("\u2022 <b>Min Length:</b> 30 tokens (approximately 15-20 words)", BL))

    story.append(Paragraph("<b>Advantages:</b>", B))
    story.append(Paragraph("\u2022 Generates human-like, fluent, and readable summaries", BL))
    story.append(Paragraph("\u2022 Can paraphrase and combine ideas from different parts of the article", BL))
    story.append(Paragraph("\u2022 Higher ROUGE scores than extractive (~35% better on ROUGE-1)", BL))
    story.append(Paragraph("\u2022 Produces more concise summaries (8.6% compression vs 18.9%)", BL))
    story.append(Paragraph("\u2022 Can synthesize information across multiple sentences", BL))

    story.append(Paragraph("<b>Disadvantages:</b>", B))
    story.append(Paragraph("\u2022 Much slower than extractive (~505 seconds vs 25 seconds for 200 articles)", BL))
    story.append(Paragraph("\u2022 Requires GPU for reasonable inference speed", BL))
    story.append(Paragraph("\u2022 May occasionally generate incorrect facts not present in the article (hallucination)", BL))
    story.append(Paragraph("\u2022 Less interpretable \u2014 harder to understand why specific words were chosen", BL))

    # ========================================================================
    # 7. EVALUATION AND COMPARISON
    # ========================================================================
    story.append(Paragraph("7. Evaluation and Comparison", H))

    story.append(Paragraph(
        "To objectively measure and compare the quality of summaries produced by both models, we use "
        "standard evaluation metrics from the summarization literature. The primary metrics are ROUGE "
        "scores, which measure the overlap between generated summaries and human-written reference summaries.", B))

    story.append(Paragraph("<b>ROUGE Metrics:</b>", B))
    story.append(Paragraph(
        "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is the standard metric for text "
        "summarization. It measures n-gram overlap between the generated summary (hypothesis) and the "
        "human-written summary (reference). Higher ROUGE scores indicate better quality.", B))

    story.append(Paragraph("\u2022 <b>ROUGE-1:</b> Measures unigram (single word) overlap. Indicates whether important individual words from the reference appear in the generated summary.", BL))
    story.append(Paragraph("\u2022 <b>ROUGE-2:</b> Measures bigram (two-word sequence) overlap. Indicates whether important phrases and word pairs are preserved. This is harder to match than ROUGE-1.", BL))
    story.append(Paragraph("\u2022 <b>ROUGE-L:</b> Measures the longest common subsequence (LCS). Finds the longest sequence of words that appears in both summaries in the same order (not necessarily contiguous). This captures overall sentence structure quality.", BL))

    story.append(Paragraph(
        "Additionally, we use <b>BERTScore</b> as a complementary metric. Unlike ROUGE which only "
        "measures exact word overlap, BERTScore uses contextual embeddings from a pre-trained BERT model "
        "to measure semantic similarity. This means a paraphrase that uses different words but expresses "
        "the same meaning can still receive a high BERTScore, addressing a key limitation of ROUGE.", B))

    story.append(Spacer(1, 6))

    # ROUGE Results Table
    story.append(Paragraph("<b>ROUGE Evaluation Results:</b>", B))
    story.append(Spacer(1, 2))

    table_data = [
        [Paragraph("Model", TH), Paragraph("ROUGE-1", TH), Paragraph("ROUGE-2", TH),
         Paragraph("ROUGE-L", TH), Paragraph("Compression\nRatio", TH)],
        [Paragraph("Extractive", TL), Paragraph("0.2902", TC), Paragraph("0.0991", TC),
         Paragraph("0.1878", TC), Paragraph("0.1889", TC)],
        [Paragraph("Abstractive (BART)", TL), Paragraph("0.3907", TC), Paragraph("0.1692", TC),
         Paragraph("0.2836", TC), Paragraph("0.0864", TC)],
    ]
    table = Table(table_data, colWidths=[38*mm, 28*mm, 28*mm, 28*mm, 28*mm])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, black),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#E8E8E8')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Results Analysis:</b>", B))
    story.append(Paragraph(
        "\u2022 The abstractive model (BART) achieved significantly better ROUGE scores than the extractive model across all three metrics. ROUGE-1 improved from 0.2902 to 0.3907 (34.6% improvement), ROUGE-2 from 0.0991 to 0.1692 (70.7% improvement), and ROUGE-L from 0.1878 to 0.2836 (51.0% improvement).", BL))
    story.append(Paragraph(
        "\u2022 The extractive model was significantly faster, processing 200 articles in approximately 25 seconds compared to 505 seconds for BART (roughly 20x speed difference).", BL))
    story.append(Paragraph(
        "\u2022 The abstractive model generated more concise summaries with a compression ratio of 8.6% compared to 18.9% for extractive, meaning BART summaries are about half the length while achieving higher quality scores.", BL))
    story.append(Paragraph(
        "\u2022 The extractive model preserved exact wording from the original article, which is advantageous for factual accuracy but limits its ability to produce fluent, readable summaries.", BL))

    story.append(Paragraph("<b>Additional Comparison Criteria:</b>", B))
    story.append(Paragraph("\u2022 <b>Summary Quality:</b> BART produces more fluent and coherent summaries", BL))
    story.append(Paragraph("\u2022 <b>Compression Ratio:</b> BART achieves higher compression (shorter summaries)", BL))
    story.append(Paragraph("\u2022 <b>Preservation of Information:</b> Both models preserve key information, but BART can synthesize across sentences", BL))
    story.append(Paragraph("\u2022 <b>Execution Speed:</b> Extractive is 20x faster than abstractive", BL))

    # ========================================================================
    # 8. MANUAL EVALUATION
    # ========================================================================
    story.append(Paragraph("8. Manual Evaluation", H))

    story.append(Paragraph(
        "While automated metrics like ROUGE and BERTScore provide quantitative measures of summary "
        "quality, they have known limitations. ROUGE cannot detect grammatical errors, factual "
        "inconsistencies, or overall readability. Therefore, manual (human) evaluation was performed "
        "to verify summary quality beyond what automated metrics can capture.", B))

    story.append(Paragraph("<b>Manual Evaluation Criteria:</b>", B))
    story.append(Paragraph("\u2022 Does the summary preserve the main idea of the original article?", BL))
    story.append(Paragraph("\u2022 Does the summary include the most important information?", BL))
    story.append(Paragraph("\u2022 Is the summary significantly shorter than the original text?", BL))
    story.append(Paragraph("\u2022 Is the summary readable and fluent (good grammar and flow)?", BL))
    story.append(Paragraph("\u2022 Does the summary avoid introducing false information (hallucination)?", BL))

    story.append(Paragraph("<b>Example Comparison:</b>", B))

    story.append(Paragraph("Original Article Text:", EL))
    story.append(Paragraph(
        "\"The food was amazing and the service was excellent. The restaurant was clean and the staff "
        "were friendly. However, the delivery was very slow and the order arrived late. Overall, the "
        "dining experience was positive despite the delivery issues.\"", EX))

    story.append(Paragraph("Extractive Summary:", EL))
    story.append(Paragraph(
        "\"The restaurant was clean and the staff were friendly.\"", EX))

    story.append(Paragraph("Abstractive Summary:", EL))
    story.append(Paragraph(
        "\"The food and service were good, but the delivery was slow.\"", EX))

    story.append(Paragraph(
        "<b>Analysis:</b> The abstractive summary successfully captures all three key points from the "
        "original text: food quality (\"amazing\" \u2192 \"good\"), service quality (\"excellent\" \u2192 \"good\"), "
        "and the delivery problem (\"very slow\" \u2192 \"slow\"). It does this in a single, fluent sentence. "
        "The extractive summary, by contrast, only captures one aspect (restaurant cleanliness and staff) "
        "and completely misses the main points about food quality and the delivery issue. This demonstrates "
        "the fundamental advantage of abstractive summarization: it can synthesize information from "
        "multiple sentences into a concise, comprehensive summary.", B))

    story.append(Paragraph(
        "This pattern was observed consistently across multiple test examples: the abstractive model "
        "tends to produce more informative and readable summaries, while the extractive model sometimes "
        "selects sentences that are individually important but collectively incomplete.", B))

    # ========================================================================
    # 9. CONCLUSION
    # ========================================================================
    story.append(Paragraph("9. Conclusion", H))

    story.append(Paragraph(
        "This project successfully implemented a complete text summarization system using both traditional "
        "statistical methods and modern deep learning techniques. The system provides two complementary "
        "approaches to automatic text summarization, each with distinct strengths and use cases.", B))

    story.append(Paragraph(
        "The <b>extractive summarization</b> approach uses TF-IDF term frequency scoring combined with "
        "sentence embedding similarity (all-MiniLM-L6-v2) to identify and select the most important "
        "sentences from the original article. This method is fast (~25 seconds for 200 articles), "
        "interpretable, and preserves the original wording exactly, making it ideal for applications "
        "where factual accuracy is critical and hallucination must be avoided.", B))

    story.append(Paragraph(
        "The <b>abstractive summarization</b> approach uses the BART Transformer model (facebook/bart-large-cnn) "
        "to generate entirely new summaries that paraphrase and synthesize information from the source "
        "article. This method achieves significantly higher ROUGE scores (34-70% improvement across metrics), "
        "produces more fluent and human-like summaries, and generates more concise outputs. However, it "
        "is approximately 20x slower and requires GPU acceleration for practical use.", B))

    story.append(Paragraph("<b>Key Findings:</b>", B))

    story.append(Paragraph(
        "\u2022 <b>Extractive summarization</b> is best suited for scenarios where speed, interpretability, "
        "and factual accuracy are the primary concerns. It is ideal for news monitoring, legal document "
        "review, and any application where preserving the exact original phrasing is important.", BL))

    story.append(Paragraph(
        "\u2022 <b>Abstractive summarization</b> is best suited for scenarios where readability, conciseness, "
        "and fluency are prioritized. It is ideal for content creation, email summarization, and applications "
        "where a natural, human-like summary is desired.", BL))

    story.append(Paragraph(
        "\u2022 The project demonstrates how NLP techniques ranging from classical statistical methods "
        "(TF-IDF) to modern Transformer models (BART) can be combined to build powerful, production-ready "
        "summarization systems capable of reducing long texts while preserving essential meaning.", BL))

    story.append(Paragraph(
        "Both approaches have their place in the NLP toolkit, and the choice between them depends on "
        "the specific requirements of the application \u2014 whether speed and accuracy matter more, or "
        "whether fluency and conciseness are the priority.", B))

    story.append(Spacer(1, 12))
    story.append(Paragraph("End of Report", styles['RptEnd']))

    doc.build(story)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    create_report()
