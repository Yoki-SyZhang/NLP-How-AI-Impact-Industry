# NLP-How-AI-Impact-Industry

## Overview

Using ~200K news articles on AI/ML/DS, this project applies NLP methods to systematically understand how AI is impacting different industries based on real-world media discourse. Goals:

- Identify industries (and their companies) most likely to be impacted by AI
- Determine **how** they are impacted — through which mechanisms / technologies
- Surface what drives successful vs. unsuccessful AI adoption

Data source: `df_news_final_project = pd.read_parquet('https://storage.googleapis.com/msca-bdp-data-open/news_final_project/news_final_project.parquet', engine='pyarrow')`

## Pipeline

| Notebook | Stage |
|---|---|
| `01_Data_Exploration_and_Cleaning_New.ipynb` | Data cleaning |
| `02_Topic_Modeling.ipynb` | Topic modeling & industry mapping |
| `03`–`11_*_Specific_Topic_and_Entity_Sentiment.ipynb` | Per-industry sentiment & entity/mechanism analysis (Finance, Health, Physical Production, Content, Public, Consumer, Education, Infrastructure, Digital Services) |

### 1. Data Cleaning
Built an iterative, diagnosis-driven cleaning pipeline that refined ~200K raw articles down to ~149K high-quality documents. Combined standard normalization, near-duplicate detection, and heuristic rules targeting structural crawl noise (navigation blocks, HTML/entity residue, sign-in/cookie/footer boilerplate) to ensure clean input text.

### 2. Topic Modeling & Industry Mapping
Ran a two-stage BERTopic pipeline: a fine-grained pass (embeddings + UMAP + HDBSCAN) to isolate and filter out noise/boilerplate clusters, followed by a coarser pass to extract industry-level topics, cross-checked with multiple c-TF-IDF keyword representations. Used an LLM to classify each topic into one of 10 predefined industries, surfacing the core areas where AI is most discussed within each industry.

### 3. Sentiment Modeling
Used LLM weak supervision to label a stratified sample of text chunks (positive / negative / hard-to-say), augmented with synthetic data to balance the underrepresented negative class, then fine-tuned a DistilBERT classifier for chunk-level sentiment. Aggregated sentiment at the document level across industry, topic, and time dimensions to reveal how AI narratives differ across sectors and evolve over time.

### 4. Entity & Mechanism Analysis
Built a sentiment-conditioned NER pipeline that uses an LLM to extract four structured entity types — industry companies, AI companies, AI technologies, and impact mechanisms — separately from positively- and negatively-framed text. Estimated P(entity | sentiment) to identify which companies and mechanisms drive positive vs. negative AI narratives within each industry.

## Outputs
Cleaned corpus, topic-to-industry mappings, fine-tuned sentiment classifiers, and per-industry entity/mechanism rankings, summarized in `presentation.pdf`.
