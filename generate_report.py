"""
Generate the Text Summarization System Report as a PDF.
Matches the exact structure and formatting of the sample report.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import black, HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


def create_report(output_path="reports/text_summarization_report.pdf"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2.5*cm,
        leftMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles (unique names to avoid conflicts)
    styles.add(ParagraphStyle(
        'RptTitle',
        parent=styles['Title'],
        fontSize=18,
        leading=24,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica',
        textColor=HexColor('#333333'),
    ))
    styles.add(ParagraphStyle(
        'RptHeading',
        parent=styles['Heading1'],
        fontSize=14,
        leading=20,
        spaceBefore=18,
        spaceAfter=10,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptSubHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptBullet',
        parent=styles['Normal'],
        fontSize=11,
        leading=15,
        leftIndent=20,
        bulletIndent=8,
        spaceAfter=3,
        fontName='Helvetica',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptNumbered',
        parent=styles['Normal'],
        fontSize=11,
        leading=15,
        leftIndent=20,
        bulletIndent=8,
        spaceAfter=3,
        fontName='Helvetica',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptFormula',
        parent=styles['Normal'],
        fontSize=12,
        leading=20,
        spaceBefore=10,
        spaceAfter=10,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        textColor=black,
    ))
    styles.add(ParagraphStyle(
        'RptExample',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
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
        leading=15,
        spaceBefore=8,
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

    B = styles['RptBody']
    H = styles['RptHeading']
    SH = styles['RptSubHeading']
    BL = styles['RptBullet']
    NB = styles['RptNumbered']
    FM = styles['RptFormula']
    EX = styles['RptExample']
    EL = styles['RptExLabel']
    TH = styles['RptTblHdr']
    TC = styles['RptTblCell']
    TL = styles['RptTblCellL']

    story = []

    # TITLE
    story.append(Paragraph("Text Summarization System Report", styles['RptTitle']))
    story.append(Paragraph("TF-IDF Extractive and Transformer-Based Abstractive Summarization", styles['RptSubtitle']))
    story.append(Spacer(1, 10))

    # 1. PROBLEM DESCRIPTION
    story.append(Paragraph("1. Problem Description", H))
    story.append(Paragraph(
        "Text summarization is one of the most important applications in Natural Language Processing (NLP). "
        "The main goal is to reduce a long text into a shorter version while preserving the most important "
        "information and main ideas. This project implements a complete text summarization system using two "
        "different approaches:", B))
    story.append(Paragraph("1. <b>Extractive Summarization</b> \u2014 Baseline Method", BL))
    story.append(Paragraph("2. <b>Abstractive Summarization</b> \u2014 Advanced Method", BL))
    story.append(Paragraph(
        "The extractive model selects the most important sentences directly from the original text using "
        "TF-IDF scoring and sentence embeddings. The abstractive model uses a Transformer-based architecture, "
        "specifically the BART model, which generates new fluent summaries and can paraphrase the original text.", B))
    story.append(Paragraph("The system was designed to:", B))
    story.append(Paragraph("\u2022 Keep important information from the original article", BL))
    story.append(Paragraph("\u2022 Reduce the length of the text significantly", BL))
    story.append(Paragraph("\u2022 Preserve the meaning of the original article", BL))
    story.append(Paragraph("\u2022 Compare traditional and advanced summarization techniques", BL))

    # 2. DATASET USED
    story.append(Paragraph("2. Dataset Used", H))
    story.append(Paragraph(
        "The project uses articles and their corresponding summaries for training and evaluation. "
        "The system is suitable for various types of text including:", B))
    story.append(Paragraph("\u2022 News Articles", BL))
    story.append(Paragraph("\u2022 Blog Posts", BL))
    story.append(Paragraph("\u2022 Product Reviews", BL))
    story.append(Paragraph("\u2022 General Long Documents", BL))
    story.append(Paragraph(
        "The abstractive model uses <b>facebook/bart-large-cnn</b>, which was originally trained on "
        "the <b>CNN/DailyMail summarization dataset</b>. The dataset details are:", B))
    story.append(Paragraph("\u2022 <b>Source:</b> News articles from CNN and DailyMail", BL))
    story.append(Paragraph("\u2022 <b>Size:</b> 5000 samples (configurable)", BL))
    story.append(Paragraph("\u2022 <b>Split:</b> 80% train, 10% validation, 10% test", BL))
    story.append(Paragraph("\u2022 <b>Article Length:</b> ~700 words on average", BL))
    story.append(Paragraph("\u2022 <b>Highlights Length:</b> ~50 words on average", BL))
    story.append(Paragraph(
        "Evaluation was performed on multiple text samples to compare the generated summaries "
        "from both the extractive and abstractive methods.", B))

    # 3. PREPROCESSING STEPS
    story.append(Paragraph("3. Preprocessing Steps", H))
    story.append(Paragraph(
        "Before training and summarization, the text goes through several preprocessing steps:", B))
    story.append(Paragraph("<b>1. Convert text to lowercase</b>", NB))
    story.append(Paragraph('Example: "The Food Was Amazing" \u2192 "the food was amazing"', EX))
    story.append(Paragraph("<b>2. Remove punctuation</b>", NB))
    story.append(Paragraph(
        "Symbols such as commas, periods, and special characters are removed to focus on word content.", B))
    story.append(Paragraph("<b>3. Remove stopwords</b>", NB))
    story.append(Paragraph(
        'Common words like "the", "is", and "and" are removed because they do not carry important meaning.', B))
    story.append(Paragraph("<b>4. Tokenization</b>", NB))
    story.append(Paragraph(
        "Text is split into individual words and sentences for processing.", B))
    story.append(Paragraph("<b>5. Sentence Segmentation</b>", NB))
    story.append(Paragraph(
        "Articles are divided into individual sentences using the NLTK sentence tokenizer. "
        "This is critical for extractive summarization where we rank and select sentences.", B))
    story.append(Paragraph(
        "These preprocessing steps improve feature extraction and summarization quality.", B))

    # 4. FEATURE EXTRACTION
    story.append(Paragraph("4. Feature Extraction", H))
    story.append(Paragraph(
        "Two feature extraction techniques are used to evaluate sentence importance:", B))
    story.append(Paragraph("A) TF-IDF \u2014 Term Frequency \u2013 Inverse Document Frequency", SH))
    story.append(Paragraph(
        "TF-IDF measures how important a word is in a sentence compared to the whole document. "
        "Words with higher TF-IDF scores are considered more informative and relevant to the topic.", B))
    story.append(Paragraph("<b>Advantages:</b>", B))
    story.append(Paragraph("\u2022 Simple and fast computation", BL))
    story.append(Paragraph("\u2022 Effective for extractive summarization", BL))
    story.append(Paragraph("\u2022 Easy to interpret and explain", BL))
    story.append(Paragraph("B) Sentence Embeddings", SH))
    story.append(Paragraph(
        "Sentence embeddings convert sentences into dense numerical vectors that represent "
        "semantic meaning. The model used is <b>all-MiniLM-L6-v2</b>, which produces 384-dimensional vectors.", B))
    story.append(Paragraph("<b>Advantages:</b>", B))
    story.append(Paragraph("\u2022 Captures semantic similarity between sentences", BL))
    story.append(Paragraph("\u2022 Understands sentence meaning beyond keyword matching", BL))
    story.append(Paragraph("\u2022 Improves sentence ranking quality significantly", BL))

    # 5. BASELINE METHOD
    story.append(Paragraph("5. Baseline Method: Extractive Summarization", H))
    story.append(Paragraph(
        "The baseline approach uses extractive summarization, which selects the most important "
        "sentences directly from the original article.", B))
    story.append(Paragraph("<b>How it works:</b>", B))
    story.append(Paragraph("1. Split the article into individual sentences.", NB))
    story.append(Paragraph("2. Calculate TF-IDF scores for each sentence.", NB))
    story.append(Paragraph("3. Generate sentence embeddings using all-MiniLM-L6-v2.", NB))
    story.append(Paragraph("4. Compute cosine similarity between each sentence and the document representation.", NB))
    story.append(Paragraph("5. Combine TF-IDF score and embedding score using weighted average.", NB))
    story.append(Paragraph("6. Select the top-k most important sentences.", NB))
    story.append(Paragraph("7. Restore the original sentence order to produce the final summary.", NB))
    story.append(Paragraph("<b>Formula Used:</b>", B))
    story.append(Paragraph("Final Score = 0.5 \u00d7 TF-IDF Score + 0.5 \u00d7 Embedding Similarity", FM))
    story.append(Paragraph("<b>Advantages:</b>", B))
    story.append(Paragraph("\u2022 Fast execution (~25 seconds for 200 articles)", BL))
    story.append(Paragraph("\u2022 Preserves original wording exactly", BL))
    story.append(Paragraph("\u2022 No hallucination risk", BL))
    story.append(Paragraph("\u2022 Easy to debug and interpret", BL))
    story.append(Paragraph("<b>Disadvantages:</b>", B))
    story.append(Paragraph("\u2022 Cannot paraphrase or generate new text", BL))
    story.append(Paragraph("\u2022 Sometimes produces disconnected summaries", BL))

    # 6. ADVANCED METHOD
    story.append(Paragraph("6. Advanced Method: Transformer-Based Summarization", H))
    story.append(Paragraph(
        "The advanced approach uses abstractive summarization with the BART model, "
        "which generates entirely new summaries rather than extracting sentences.", B))
    story.append(Paragraph("<b>Model Used:</b>", B))
    story.append(Paragraph("<b>facebook/bart-large-cnn</b> (406M parameters)", B))
    story.append(Paragraph("<b>How it works:</b>", B))
    story.append(Paragraph("1. The input article is tokenized using the BART tokenizer (max 1024 tokens).", NB))
    story.append(Paragraph("2. The article is encoded into contextual representations by the encoder.", NB))
    story.append(Paragraph("3. The decoder generates a new summary word-by-word.", NB))
    story.append(Paragraph("4. Beam search is used to improve summary quality.", NB))
    story.append(Paragraph("<b>Generation Parameters:</b>", B))
    story.append(Paragraph("\u2022 <b>Beam Search:</b> 4", BL))
    story.append(Paragraph("\u2022 <b>Length Penalty:</b> 1.0 (reduced from 2.0 for more natural summaries)", BL))
    story.append(Paragraph("\u2022 <b>No Repeat N-Gram Size:</b> 4 (increased from 3)", BL))
    story.append(Paragraph("\u2022 <b>Early Stopping:</b> Disabled (allows full generation)", BL))
    story.append(Paragraph("\u2022 <b>Max Length:</b> 130 tokens (~80-100 words)", BL))
    story.append(Paragraph("\u2022 <b>Min Length:</b> 30 tokens (~15-20 words)", BL))
    story.append(Paragraph("<b>Advantages:</b>", B))
    story.append(Paragraph("\u2022 Generates human-like, fluent summaries", BL))
    story.append(Paragraph("\u2022 Can paraphrase and combine ideas from different parts", BL))
    story.append(Paragraph("\u2022 Higher ROUGE scores than extractive (~35% better on ROUGE-1)", BL))
    story.append(Paragraph("\u2022 Produces more concise summaries (8.6% compression vs 18.9%)", BL))
    story.append(Paragraph("<b>Disadvantages:</b>", B))
    story.append(Paragraph("\u2022 Slower than extractive methods (~505 seconds for 200 articles)", BL))
    story.append(Paragraph("\u2022 Requires GPU for reasonable speed", BL))
    story.append(Paragraph("\u2022 May occasionally generate incorrect facts (hallucination)", BL))

    # 7. EVALUATION AND COMPARISON
    story.append(Paragraph("7. Evaluation and Comparison", H))
    story.append(Paragraph(
        "Models were evaluated using ROUGE metrics, which measure n-gram overlap between "
        "generated summaries and human-written reference summaries.", B))
    story.append(Paragraph("\u2022 <b>ROUGE-1:</b> Measures unigram (single word) overlap", BL))
    story.append(Paragraph("\u2022 <b>ROUGE-2:</b> Measures bigram (two-word sequence) overlap", BL))
    story.append(Paragraph("\u2022 <b>ROUGE-L:</b> Measures longest common subsequence similarity", BL))
    story.append(Paragraph(
        "Additionally, <b>BERTScore</b> was used to measure semantic similarity using contextual embeddings, "
        "capturing meaning beyond exact word overlap.", B))
    story.append(Spacer(1, 8))

    # Table
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
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Results Analysis:</b>", B))
    story.append(Paragraph("\u2022 The abstractive model achieved better ROUGE scores than the extractive model across all metrics.", BL))
    story.append(Paragraph("\u2022 The extractive model was significantly faster (25s vs 505s for 200 samples).", BL))
    story.append(Paragraph("\u2022 The abstractive model generated more fluent and natural summaries.", BL))
    story.append(Paragraph("\u2022 The extractive model preserved exact wording from the original article.", BL))

    # 8. MANUAL EVALUATION
    story.append(Paragraph("8. Manual Evaluation", H))
    story.append(Paragraph(
        "Manual evaluation was performed to verify summary quality beyond automated metrics.", B))
    story.append(Paragraph("<b>Manual evaluation criteria:</b>", B))
    story.append(Paragraph("\u2022 Whether the summary preserves the main idea", BL))
    story.append(Paragraph("\u2022 Whether important information exists in the summary", BL))
    story.append(Paragraph("\u2022 Whether the summary is shorter than the original text", BL))
    story.append(Paragraph("\u2022 Readability and fluency of the summary", BL))
    story.append(Paragraph("<b>Example:</b>", B))
    story.append(Paragraph("Original Text:", EL))
    story.append(Paragraph(
        '"The food was amazing and the service was excellent. The restaurant was clean and the staff '
        'were friendly. However, the delivery was very slow and the order arrived late."', EX))
    story.append(Paragraph("Extractive Summary:", EL))
    story.append(Paragraph('"The restaurant was clean and the staff were friendly."', EX))
    story.append(Paragraph("Abstractive Summary:", EL))
    story.append(Paragraph('"The food and service were good, but the delivery was slow."', EX))
    story.append(Paragraph(
        "The abstractive summary preserves the important information (food quality, service, delivery issue) "
        "while remaining short and readable. The extractive summary selected a single sentence that misses "
        "the key point about food quality and the delivery problem.", B))

    # 9. CONCLUSION
    story.append(Paragraph("9. Conclusion", H))
    story.append(Paragraph(
        "This project implemented a complete text summarization system using both traditional "
        "and deep learning techniques. The extractive summarization approach uses TF-IDF and "
        "sentence embeddings to select the most important sentences, providing fast and interpretable "
        "results. The Transformer-based abstractive summarization using BART generates more fluent, "
        "human-like summaries with higher ROUGE scores.", B))
    story.append(Paragraph("Key findings from this project:", B))
    story.append(Paragraph(
        "\u2022 <b>Extractive summarization</b> is suitable when speed and factual accuracy are critical. "
        "It preserves the original wording and has no hallucination risk.", BL))
    story.append(Paragraph(
        "\u2022 <b>Abstractive summarization</b> is suitable for readability and advanced NLP applications. "
        "It produces fluent summaries that can paraphrase and synthesize information.", BL))
    story.append(Paragraph(
        "\u2022 The project demonstrates how NLP techniques and Transformer models can be combined "
        "to build powerful summarization systems capable of reducing long texts while preserving "
        "essential meaning.", BL))

    story.append(Spacer(1, 20))
    story.append(Paragraph("End of Report", styles['RptEnd']))

    doc.build(story)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    create_report()
