# NLP Text Summarization - Diagram Generation Prompts

Copy these prompts and pass them to Claude, Midjourney, or any AI diagram generator.

---

## 1. System Architecture Overview

**Prompt:**
```
Create a professional system architecture diagram for an NLP Text Summarization system with two parallel pipelines.

Style: Clean, modern, technical diagram with a light background. Use a horizontal flow layout.

Components (left to right):

LEFT - INPUT:
- A document icon labeled "Input Article (~700 words)"
- An arrow splitting into two paths

TOP PATH - EXTRACTIVE PIPELINE (blue theme):
1. "Sentence Tokenizer" (NLTK) - splits article into sentences
2. "TF-IDF Scorer" - scores sentences by keyword importance
3. "Embedding Scorer" - scores sentences by semantic similarity (all-MiniLM-L6-v2)
4. "Score Combiner" - merges scores (50/50 weighted)
5. "Top-K Selector" - picks highest scoring sentences
6. Output: "Extractive Summary" (exact sentences from source)

BOTTOM PATH - ABSTRACTIVE PIPELINE (orange theme):
1. "BART Tokenizer" - converts text to tokens with "summarize:" prefix
2. "BART Encoder" - 12-layer transformer, bidirectional attention
3. "Beam Search Decoder" - generates tokens autoregressively (4 beams)
4. "Detokenizer" - converts tokens back to text
5. Output: "Abstractive Summary" (new fluent text)

RIGHT - EVALUATION:
- Both outputs feed into "ROUGE Evaluator" (ROUGE-1, ROUGE-2, ROUGE-L)
- "Compression Ratio Calculator"
- Final output: "Metrics Dashboard"

Use distinct colors: blue for extractive, orange for abstractive, green for evaluation.
Include small icons for each component (document, brain, graph, etc.).
Add a legend at the bottom explaining the color coding.
```

---

## 2. Extractive Pipeline Deep Dive

**Prompt:**
```
Create a detailed technical flowchart showing the Extractive Summarization pipeline step by step.

Style: Vertical flowchart, professional technical style, white background.

Flow (top to bottom):

STEP 1: INPUT
- Box: "Article Text" with example text snippet
- Arrow down

STEP 2: SENTENCE TOKENIZATION
- Box: "NLTK sent_tokenize()"
- Show visual: One paragraph splitting into 6 separate sentence boxes
- Arrow down

STEP 3: DUAL SCORING (split into two parallel columns)

LEFT COLUMN - TF-IDF SCORING:
- "Fit TfidfVectorizer on sentences"
- "Calculate TF (term frequency per sentence)"
- "Calculate IDF (inverse document frequency)"
- "Sum TF×IDF per sentence"
- Output: "TF-IDF Scores: [0.94, 0.75, 0.85, 0.64, 1.00, 0.00]"

RIGHT COLUMN - EMBEDDING SCORING:
- "Load all-MiniLM-L6-v2 model"
- "Encode each sentence → 384-dim vector"
- "Compute average embedding (main topic)"
- "Cosine similarity: sentence vs topic"
- Output: "Embedding Scores: [0.83, 0.00, 0.74, 0.09, 1.00, 0.31]"

STEP 4: COMBINE
- Box: "Weighted Combination: 0.3×TF-IDF + 0.7×Embedding"
- Show math: Final scores calculated
- Arrow down

STEP 5: SELECT & SORT
- Box: "Rank by score → Select top 3 → Sort by original position"
- Visual: Highlight sentences [0], [2], [4] as selected
- Arrow down

STEP 6: OUTPUT
- Box: "Extractive Summary" with the 3 selected sentences joined
- Show: "Compression: 37.5% | Time: 0.84s"

Use color coding: blue for TF-IDF path, purple for embedding path, green for output.
Include small code snippets or formulas where relevant.
```

---

## 3. Abstractive Pipeline (BART) Deep Dive

**Prompt:**
```
Create a detailed technical diagram showing how BART generates summaries step by step.

Style: Technical architecture diagram, dark theme (like a code editor), with glowing elements.

Flow (left to right):

PHASE 1: PREPARATION
- Input: "Article Text (~200 words)"
- Box: "Add 'summarize:' prefix" (show: "summarize: Recent developments in AI...")
- Box: "BPE Tokenizer" → Show tokens: [72, 3987, 1012, 5, 890, ...]
- Box: "Truncate to 1024 tokens" (if article is too long)

PHASE 2: ENCODING (draw as a stack of 12 layers)
- Box: "BART Encoder (12 Transformer Layers)"
- Show internal: "Bidirectional Self-Attention"
- Show: "Each token attends to ALL other tokens"
- Output: "Contextual Representations (768-dim per token)"

PHASE 3: DECODING (show as a loop)
- Box: "Autoregressive Generation"
- Show step-by-step:
  Step 1: Input: <s> → Output: "The"
  Step 2: Input: <s> The → Output: "recent"
  Step 3: Input: <s> The recent → Output: "developments"
  ...
  Step N: Input: <s> The recent developments... → Output: </s> (stop)

PHASE 4: BEAM SEARCH (zoom into one step)
- Box: "Beam Search (4 beams)"
- Show 4 parallel paths with probabilities:
  Beam 1: "The" (0.92) → "recent" (0.87) → "developments" (0.81)
  Beam 2: "Recent" (0.05) → "AI" (0.03) → ...
  Beam 3: "AI" (0.02) → ...
  Beam 4: "In" (0.01) → ...
- Show: "Length Penalty = 2.0" and "No Repeat N-gram Size = 3"

PHASE 5: OUTPUT
- Box: "Detokenizer" → "Abstractive Summary (~50 words)"
- Show: "Compression: 8.6% | Time: 2.5s"

Use a dark background (#1a1a2e) with neon accents (cyan, magenta, yellow).
Make the beam search section visually prominent as it's the key innovation.
```

---

## 4. Data Flow Sequence Diagram

**Prompt:**
```
Create a professional sequence diagram showing the complete user interaction flow with the summarization system.

Style: Clean UML-style sequence diagram, light background, professional colors.

Actors (left to right):
- User (person icon)
- Streamlit App (browser icon)
- Preprocessing Module (gear icon)
- TF-IDF Module (chart icon)
- Embedding Module (brain icon)
- BART Model (neural network icon)
- ROUGE Evaluator (ruler icon)

Flow:

1. User → App: "Enter article text"
2. User → App: "Select model type (Extractive/Abstractive/Compare)"
3. App → App: "Display loading spinner"

4a. IF Extractive:
   - App → Preprocessing: "Tokenize sentences"
   - Preprocessing → App: "Return N sentences"
   - App → TF-IDF: "Score sentences"
   - TF-IDF → App: "TF-IDF scores"
   - App → Embedding: "Compute similarities"
   - Embedding → App: "Embedding scores"
   - App → App: "Combine scores, select top-K"
   - App → User: "Display extractive summary"

4b. IF Abstractive:
   - App → BART: "Tokenize + Encode"
   - BART → App: "Context embeddings"
   - App → BART: "Beam search generation"
   - BART → BART: "Loop: Expand 4 candidates"
   - BART → App: "Generated tokens"
   - App → BART: "Decode to text"
   - BART → App: "Summary text"
   - App → User: "Display abstractive summary"

5. IF ROUGE Evaluation:
   - User → App: "Enter reference summary"
   - App → ROUGE: "Compute ROUGE-1/2/L"
   - ROUGE → ROUGE: "Tokenize + Stem + Count n-grams"
   - ROUGE → App: "ROUGE scores"
   - App → User: "Display metrics dashboard"

Use different colored arrows for each path (blue=extractive, orange=abstractive, green=evaluation).
Add timing annotations where relevant (e.g., "~0.8s" for extractive, "~2.5s" for abstractive).
```

---

## 5. Model Comparison Dashboard

**Prompt:**
```
Create a professional dashboard-style visualization comparing three summarization models side by side.

Style: Modern dashboard with cards, charts, and metrics. Light theme with accent colors.

Layout: 3 columns (Extractive | Abstractive BART | Fine-tuned BART)

Each column contains:

CARD 1 - Model Info:
- Icon + Model Name
- Architecture type (Extractive vs Seq2Seq)
- Parameters (N/A vs 406M vs 406M+fine-tuned)
- Device (CPU vs GPU vs GPU)

CARD 2 - Performance Metrics:
- ROUGE-1: 0.290 | 0.391 | TBD
- ROUGE-2: 0.099 | 0.169 | TBD
- ROUGE-L: 0.188 | 0.284 | TBD
- Show as horizontal bar charts

CARD 3 - Efficiency:
- Compression: 18.9% | 8.6% | TBD
- Time (200 samples): 25.2s | 504.8s | TBD
- Show as gauge charts

CARD 4 - Summary Example:
- Same article input
- Different outputs for each model
- Highlight key differences

BOTTOM SECTION - Radar Chart:
- 5 axes: ROUGE-1, ROUGE-2, ROUGE-L, Speed, Compression
- 3 overlapping polygons (one per model)
- Legend explaining each model's color

Use color scheme: Blue (Extractive), Orange (BART), Green (Fine-tuned).
Include a "Key Insights" callout box at the bottom with bullet points.
```

---

## 6. Training Pipeline (Fine-tuning)

**Prompt:**
```
Create a technical diagram showing the BART fine-tuning pipeline for text summarization.

Style: Technical flowchart, dark theme, with data flow arrows.

Flow (top to bottom):

PHASE 1: DATA PREPARATION
- Box: "CNN/DailyMail Dataset"
- Show: 287,000 article-summary pairs
- Box: "Split: 80% Train / 10% Val / 10% Test"
- Box: "Preprocess: Add 'summarize:' prefix, tokenize"
- Show example: 
  Input: "summarize: Article text..."
  Target: "Reference summary..."

PHASE 2: MODEL INITIALIZATION
- Box: "Load Pretrained BART (facebook/bart-large-cnn)"
- Show: 406M parameters, 12 encoder + 12 decoder layers
- Box: "Freeze? No - full fine-tuning"

PHASE 3: TRAINING LOOP (show as a cycle)
- Box: "For each batch:"
  1. "Forward pass: article → BART → predicted summary"
  2. "Loss: Cross-entropy (predicted vs target)"
  3. "Backward pass: compute gradients"
  4. "Optimizer step: update weights"
- Show hyperparameters:
  - Learning rate: 3e-5
  - Batch size: 4
  - Epochs: 3
  - Warmup: 10%
  - Weight decay: 0.01

PHASE 4: EVALUATION
- Box: "Validate on held-out set"
- Box: "Compute ROUGE scores"
- Box: "Save best checkpoint"

PHASE 5: OUTPUT
- Box: "Fine-tuned Model" → "checkpoints/bart-finetuned/"
- Show: "Expected improvement: +2-5% ROUGE-1"

Use a dark background with neon accents.
Make the training loop visually circular to show iteration.
Include a small loss curve graph showing training progress.
```

---

## 7. ROUGE Metrics Explained

**Prompt:**
```
Create an educational infographic explaining ROUGE metrics for text summarization evaluation.

Style: Clean, educational infographic with examples and visual comparisons.

Sections (top to bottom):

HEADER: "Understanding ROUGE Metrics"
Subtitle: "Recall-Oriented Understudy for Gisting Evaluation"

SECTION 1: ROUGE-1 (Unigram Overlap)
- Definition: "Measures single word overlap between summary and reference"
- Example:
  Reference: "The cat sat on the mat"
  Generated: "A cat sat on a rug"
  Overlap: {"cat", "sat", "on"} → 3/5 = 0.60
- Visual: Highlight matching words in both texts
- Use case: "Does the summary contain the right keywords?"

SECTION 2: ROUGE-2 (Bigram Overlap)
- Definition: "Measures two-word sequence overlap"
- Example:
  Reference: "The cat sat on the mat"
  Generated: "A cat sat on a rug"
  Overlap: {"cat sat", "sat on"} → 2/4 = 0.50
- Visual: Highlight matching word pairs
- Use case: "Does the summary preserve phrase structure?"

SECTION 3: ROUGE-L (Longest Common Subsequence)
- Definition: "Measures longest sequence of words appearing in order"
- Example:
  Reference: "The cat sat on the mat"
  Generated: "A cat sat on a rug"
  LCS: "cat sat on" → length 3
- Visual: Show the common subsequence highlighted
- Use case: "Does the summary maintain overall coherence?"

SECTION 4: F-MEASURE
- Formula: F = 2 × (Precision × Recall) / (Precision + Recall)
- Explanation: "Balances precision (no hallucination) and recall (coverage)"
- Visual: Venn diagram showing precision vs recall

SECTION 5: INTERPRETATION GUIDE
- Table: ROUGE Score → Quality
  0.0-0.2: Poor
  0.2-0.3: Fair
  0.3-0.4: Good
  0.4+: Excellent
- Note: "ROUGE measures overlap, NOT semantic understanding"

Use a light background with distinct colors for each ROUGE type.
Include icons and visual highlights for matching words.
```

---

## 8. Project Structure & Team Roles

**Prompt:**
```
Create an organizational chart showing the project structure and team roles for an NLP Text Summarization project.

Style: Professional org chart with clear hierarchy and role descriptions.

Layout: Tree structure

ROOT: "NLP Text Summarization Project"
├── NLP Lead (Member #1)
│   ├── Extractive Pipeline (TF-IDF + Embeddings)
│   ├── config.py, features.py, extractive.py
│   └── Success: ROUGE-1 ~0.30, <50ms/article
│
├── Abstractive ML Engineer (Member #2)
│   ├── BART Model Integration
│   ├── abstractive.py, train.py
│   └── Success: ROUGE-1 ~0.40, GPU >80%
│
├── Data/NLP Engineer (Member #3)
│   ├── Preprocessing Pipeline
│   ├── preprocessing.py
│   └── Success: Tokenization >99% accuracy
│
├── Training Engineer (Member #4)
│   ├── Fine-tuning Pipeline
│   ├── train.py, hyperparameter tuning
│   └── Success: Fine-tuned ROUGE-1 >0.42
│
├── Evaluation Engineer (Member #5)
│   ├── ROUGE Metrics & Benchmarking
│   ├── evaluation.py
│   └── Success: Reproducible results, <1% variance
│
└── UI/Application Engineer (Member #6)
    ├── Streamlit App, FastAPI, CLI
    ├── app.py, api.py, demo.py
    └── Success: Clean UI, <100ms API response

Use color coding by role type:
- Blue: NLP/ML roles
- Green: Data/Training roles
- Orange: UI/Application roles

Include file icons next to each person's responsibilities.
Add a small legend explaining the color coding.
```

---

## Tips for Best Results

1. **For Claude**: Use the prompts as-is. Claude handles technical diagrams well.
2. **For Midjourney**: Add `--v 6 --ar 16:9 --style raw` to the end.
3. **For DALL-E**: Add "technical diagram, professional style, clean layout" to the prompt.
4. **For hand-drawn**: Print the prompts and sketch on paper, then digitize.

Each prompt is designed to generate a specific, informative diagram that complements the existing Mermaid diagrams in the Architecture tab.
