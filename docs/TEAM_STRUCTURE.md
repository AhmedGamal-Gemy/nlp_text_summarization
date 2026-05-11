# 👥 Team Structure - NLP Text Summarization Project

## Project Overview

This is an NLP project implementing dual-model text summarization (extractive + abstractive) with ~400M parameter BART model, TF-IDF pipelines, and production-ready deployment.

**Team Size**: 6 members
**Focus Areas**: ML/NLP (5 members) + UI/Application (1 member)

---

## Team Roles & Responsibilities

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PROJECT LEAD                                  │
│                    (Coordinates across all teams)                       │
└─────────────────────────────────────────────────────────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
    │   NLP LEAD    │    │  ML ENGINEER  │    │     UI/APP    │
    │  (Member #1)  │    │  (Member #2)  │    │  (Member #6)  │
    └───────────────┘    └───────────────┘    └───────────────┘
            │                    │
            ▼                    ▼
    ┌───────────────┐    ┌───────────────┐
    │    DATA/NLP   │    │   TRAINING    │
    │  (Member #3)  │    │  (Member #4)  │
    └───────────────┘    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │  EVALUATION   │
    │  (Member #5)  │
    └───────────────┘
```

---

## Role Breakdown

### Member #1: NLP Lead / Extractive Summarization Specialist
**Primary Focus**: Extractive pipeline (TF-IDF + Sentence Embeddings)

**Responsibilities**:
- Lead architect for extractive summarization system
- TF-IDF vectorizer design and optimization
- Sentence embedding strategy (all-MiniLM-L6-v2)
- Score combination algorithm (50/50 weighting)
- Sentence selection and ordering logic

**Owns Files**:
- `features.py` - TF-IDF and embedding extractors
- `extractive.py` - ExtractiveSummarizer class
- `config.py` - Hyperparameters (TOP_K_SENTENCES, TFIDF_WEIGHT, EMBED_WEIGHT)

**Key Decisions**:
- Vocabulary size and n-gram range
- TF-IDF vs CountVectorizer choice
- Embedding model selection
- Score weighting strategy

**Success Metrics**: ROUGE-1 ~0.30, processing speed <50ms per article

---

### Member #2: Abstractive ML Engineer / BART Specialist
**Primary Focus**: Abstractive pipeline (BART-large-cnn)

**Responsibilities**:
- Lead architect for abstractive summarization system
- BART model loading, inference, and optimization
- Beam search parameter tuning
- Tokenization strategy and prefix handling
- GPU memory optimization

**Owns Files**:
- `abstractive.py` - AbstractiveSummarizer class
- `train.py` - BART fine-tuning pipeline

**Key Decisions**:
- Beam size (default: 4)
- Length penalty (default: 2.0)
- Max/Min token limits (130/30)
- no_repeat_ngram_size (default: 3)

**Success Metrics**: ROUGE-1 ~0.40, GPU utilization >80%

---

### Member #3: Data/NLP Engineer / Preprocessing Specialist
**Primary Focus**: Text preprocessing and feature engineering

**Responsibilities**:
- Text cleaning pipeline (lowercase, punctuation, numbers)
- Sentence tokenization (NLTK sent_tokenize)
- Document/sentence boundary detection
- Special handling for edge cases (abbreviations, URLs, etc.)
- NLTK resource management

**Owns Files**:
- `preprocessing.py` - Text cleaning and sentence tokenization
- Data loading pipeline in `evaluate.py`, `train.py`

**Key Decisions**:
- Which punctuation to remove/keep
- How to handle numbers and dates
- Sentence splitting strategy
- Language detection (future enhancement)

**Success Metrics**: Tokenization accuracy >99%, handles 100% of edge cases

---

### Member #4: Training Engineer / Model Optimization
**Primary Focus**: BART fine-tuning and model optimization

**Responsibilities**:
- Fine-tuning pipeline design
- Learning rate scheduling and warmup
- Batch size and gradient accumulation
- Mixed precision training (FP16)
- Checkpoint management and model saving

**Owns Files**:
- `train.py` - Full fine-tuning pipeline
- `config.py` - Training hyperparameters

**Key Decisions**:
- Learning rate (default: 3e-5)
- Batch size per GPU (default: 4)
- Epochs (default: 3)
- Warmup ratio (default: 0.1)
- Weight decay (default: 0.01)

**Success Metrics**: Fine-tuned model ROUGE-1 >0.42, training time <6 hours

---

### Member #5: Evaluation/Research Engineer
**Primary Focus**: Metrics, benchmarking, and analysis

**Responsibilities**:
- ROUGE score implementation and verification
- Model comparison framework
- Benchmark dataset management (CNN/DailyMail)
- Compression ratio analysis
- Qualitative evaluation (human review)
- Statistical significance testing

**Owns Files**:
- `evaluate.py` - Full evaluation pipeline

**Key Decisions**:
- ROUGE variants to compute (1, 2, L)
- Stemming strategy
- Evaluation sample size (200 for speed, full for final)
- Test/val/train split ratios

**Success Metrics**: Reliable metrics, reproducible results, <1% variance

---

### Member #6: UI/Application Engineer
**Primary Focus**: User-facing applications and deployment

**Responsibilities**:
- Streamlit web application
- FastAPI REST server
- CLI demo interface
- Loading states and user feedback
- API endpoint design
- Documentation and examples

**Owns Files**:
- `app.py` - Streamlit web interface
- `api.py` - FastAPI REST server
- `demo.py` - CLI demonstration tool
- `docs/` - User-facing documentation

**Key Decisions**:
- UI/UX flow and layout
- Loading indicators and progress
- API request/response format
- Error handling and user messages

**Success Metrics**: Clean UI, <100ms API response, intuitive UX

---

## File Ownership Matrix

| File | Owner | Reviewers |
|------|-------|-----------|
| `config.py` | Lead (Member #1) | All |
| `preprocessing.py` | Data/NLP (Member #3) | Member #1, #5 |
| `features.py` | NLP Lead (Member #1) | Member #2, #3 |
| `extractive.py` | NLP Lead (Member #1) | Member #2, #5 |
| `abstractive.py` | Abstractive (Member #2) | Member #1, #5 |
| `evaluate.py` | Evaluation (Member #5) | Member #1, #2 |
| `train.py` | Training (Member #4) | Member #2, #5 |
| `app.py` | UI (Member #6) | Member #1 |
| `api.py` | UI (Member #6) | Member #2 |
| `demo.py` | UI (Member #6) | Member #1 |

---

## Collaboration Workflow

### Sprint Structure (2-week sprints)

```
Week 1: Planning + Development
├── Monday: Sprint planning, assign tasks
├── Tue-Thu: Development
└── Friday: Code review, merge to dev

Week 2: Integration + Testing
├── Monday: Integrate components
├── Tue-Thu: Testing and bug fixes
└── Friday: Sprint review, demo
```

### Code Review Requirements

| Change Type | Required Reviewers |
|-------------|-------------------|
| Config changes | 2 members |
| Model architecture | 2 members |
| New dependencies | Lead + 1 |
| UI changes | 1 member |
| Documentation | 1 member |

### Integration Points

1. **Preprocessing → Models**: Member #3's output feeds Members #1 and #2
2. **Models → Evaluation**: Member #5 tests both extractive and abstractive
3. **All → UI**: Member #6 wraps everything for users

---

## Technical Stack by Role

### ML/NLP Team (Members #1-5)
- **Python 3.12+**
- **PyTorch 2.4+** (CUDA for training/inference)
- **Transformers** (BART, tokenizers)
- **Sentence-Transformers** (embeddings)
- **scikit-learn** (TF-IDF)
- **NLTK** (text preprocessing)
- **ROUGE-Score** (evaluation)
- **HuggingFace Datasets** (data loading)

### UI/Application Team (Member #6)
- **Streamlit** (web UI)
- **FastAPI** (REST API)
- **Uvicorn** (ASGI server)
- **Pydantic** (data validation)

---

## Communication Plan

### Daily Standup (15 min, async)
```
Format: What I did yesterday / What I'm doing today / Blockers
Channel: Slack/Discord #nlp-summarization
```

### Weekly Sync (30 min, video)
- Progress update from each member
- Discuss blockers and dependencies
- Demo new features

### Architecture Decisions
- Propose in #nlp-architecture channel
- 48-hour comment period
- Final decision by Project Lead

---

## Onboarding Checklist for New Members

### Day 1
- [ ] Clone repository
- [ ] Set up Python environment (`uv venv --python 3.12`)
- [ ] Run `uv sync` to install dependencies
- [ ] Verify CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Run demo: `uv run python demo.py`

### Day 2
- [ ] Read `docs/LEARNING_GUIDE.md` (full project understanding)
- [ ] Read this `TEAM_STRUCTURE.md` (know your role)
- [ ] Explore codebase, focus on your area
- [ ] Run evaluation: `uv run python evaluate.py`

### Day 3+
- [ ] Review code in your ownership area
- [ ] Make first small contribution
- [ ] Meet with your buddy (assigned team member)
- [ ] Join daily standups

---

## Key Dependencies

```
Member #3 (Data)
        │
        ▼ (cleaned text)
Members #1, #2 (Models)
        │
        ▼ (summaries)
Member #5 (Evaluation)
        │
        ▼ (metrics)
Member #6 (UI/API)
```

**Critical Path**: Data preprocessing is the foundation. Member #3 must deliver reliable preprocessing before models can be properly developed.

---

*Last updated: May 2026*