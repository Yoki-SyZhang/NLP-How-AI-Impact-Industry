# NLP-How-AI-Impact-Industry  

With ~200K news articles on DS, ML and AI  
(`df_news_final_project = pd.read_parquet('https://storage.googleapis.com/msca-bdp-data-open/news_final_project/news_final_project.parquet', engine='pyarrow')`)  

The objective is to:

- Identify industries and their companies that are most likely to be impacted by AI over the next several years  
- Determine how those industries and their companies will be impacted and by what means / technologies  
- Provide insights into what can make AI adoption successful or unsuccessful  


---

## Break Down the Tasks

  
**Q1: Identify industries and their companies that are most likely to be impacted by AI over the next several years.？**

> Suggest measurable indicators to operationalize "most likely to be impacted"

- Topic Detection (industry-level structure)
- Entity Extraction（industries, company, technology）
- Frequency + Co-occurrence
- Time Analysis


**Q2: Determine how those industries and their companies will be impacted and by what means / technologies？**

> Define sentiment granularity

   | Level | Meaning |
   |--------|--------|
   | Document-level | Overall tone of article |
   | Topic-level | Average sentiment of articles within topic |
   | Entity-level | Sentiment toward specific companies |
   | Aspect-level (might skip for time issue) | Sentiment toward automation vs augmentation |

- Train Sentiment model (positively, negatively, hard-to-say, etc.)  
- Topic-level sentiment  
- Entity-level sentiment  
- Keyword/phrase mining（impact mechanism）


**Q3: Insights into what can make AI adoption successful or unsuccessful？**

- Topic modeling within positive vs negative clusters  
- Extractive summarization / LLM-assisted structured insight extraction  
  - Mechanism extraction  
  - Comparative analysis across industries  
  - Structured insight generation  

## Project Pipeline:

**0. Raw News Articles**

---

**1. Clean + Filter**  
(01_Data_Exploration_and_Cleaning.ipynb -> df_clean.parquet)

1. Data profiling
    * HTML artifacts
    * Language filtering
    * irrelevant crawl artifacts such as Keyword pre-filter
2. Cleaning
    * regex
    * unicode normalization
    * min length filter
3. Relevance filtering: not all ariticles are relevant。
    * Keyword pre-filter Then LLM confirm
    * zero-shot classification API

---

**2. Topic Discovery**  
(02_Topic_Modeling.ipynb -> df_topic_assigned.parquet & topic_keywords.json)

1. Vectorization
2. Topic modeling: BERTopic and Evaluate topic coherence
3. Define Topic → Industry mapping logic (LLM API)

---

**3. Entity Extraction**  
(03_Entity_Extraction.ipynb -> df_entities.parquet)

* How to Identify industries
* Find pretrained NER model
    * ORG（company）
    * PRODUCT（technology）
    * GPE（country）
    * Possibly JOB_ROLE (optional)
* Industry Mapping
    * Associate industry based on topic label
    * use API help mapping

---

**4. Sentiment Modeling**  
(04_Sentiment_Model_Training.ipynb -> model file)

1. study Aspect-Based Sentiment Analysis
2. Find labeled data -> sentiment_model.pt
    * Financial PhraseBank
    * Kaggle news sentiment dataset
3. Train model
    * DistilBERT
    * fine-tune

---

**5. Topic-level + Entity-level Aggregation**  
(05_Topic_and_Entity_Sentiment.ipynb)

* Predict sentiment
* Topic-level aggregation ()
* Entity-level sentiment (ORG: company + industry, TECH)
* Visualization over time
    - Industry sentiment trend
    - Company sentiment volatility
    - Emerging industries (growth curve)

---

**6. Business Insight Layer**  
(06_Impact_Mechanism_Analysis.ipynb)

* Answer
    * how those industries and their companies will be impacted by what means / technologies
    * Provide insights into what can make AI adoption successful or unsuccessful.
* Methods
    * phrase mining + keyword clustering.
    * Topic modeling within positive vs negative clusters
    * Comparative analysis
    * TextRank
    * LLM API

---
## Expected Folder Structure
```
NLP-How-AI-Impact-Industry/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── news_final_project.parquet
│   │
│   ├── interim/
│   │   ├── df_clean.parquet
│   │   ├── df_topic_assigned.parquet
│   │   ├── df_entities.parquet
│   │   └── sentiment_predictions.parquet
│   │
│   └── external/
│       └── labeled_sentiment_data.csv
│
├── notebooks/
│   ├── 01_Data_Exploration_and_Cleaning.ipynb
│   ├── 02_Topic_Modeling.ipynb
│   ├── 03_Entity_Extraction.ipynb
│   ├── 04_Sentiment_Model_Training.ipynb
│   ├── 05_Topic_and_Entity_Sentiment.ipynb
│   └── 06_Impact_Mechanism_Analysis.ipynb
│
├── models/
│   └── sentiment_model/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer/
│
├── artifacts/
│   ├── topic_keywords.json
│   ├── topic_labels.json
│   ├── llm_outputs/
│   │   ├── topic_labeling.json
│   │   └── insight_summaries.json
│   └── plots/
│       ├── topic_distribution.png
│       ├── sentiment_over_time.png
│       └── entity_sentiment_trend.png
│
└── presentation/
    └── NLP_AI_Impact_Final_Project.pdf
```

