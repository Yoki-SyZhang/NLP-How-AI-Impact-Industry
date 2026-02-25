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
**Raw News Articles**

↓

**Clean + Filter**   
(01_Data_Exploration_and_Cleaning.ipynb ->  df_clean.parquet)
1. Data profiling
    * HTML artifacts
    * Language filtering
    * irrelevant crawl artifacts such as Keyword pre-filter
2. Cleaning
    * regex
    * unicode normalization
    * min length filter
3. Relevance filtering: 不是所有文章都 relevant。
    * zero-shot classification API
    * 让 LLM 判断是否关于 “AI impact on industries”
  
↓
      
**Topic Discovery**
(02_Topic_Modeling.ipynb  ->  df_topic_assigned.parquet & topic_keywords.json)
1. Vectorization
2. Topic modeling: BERTopic and Evaluate topic coherence
3. Define Topic → Industry mapping logic (LLM API)

↓

**Entity Extraction** 
(03_Entity_Extraction.ipynb -> df_entities.parquet)
* How to Identify industries
* Find pretrained NER model
    * ORG（company）
    * PRODUCT（technology）
    * GPE（country）
    * Possibly JOB_ROLE (optional)
* Industry Mapping
    * 基于 topic label 关联 industry
    * 或用 API 辅助分类公司所属行业
  
↓
      
**Sentiment Modeling**
(04_Sentiment_Model_Training.ipynb  ->  model file)
1. study Aspect-Based Sentiment Analysis
2. Find labeled data  ->  sentiment_model.pt
    * Financial PhraseBank
    * Kaggle news sentiment dataset
3. Train model
    * DistilBERT
    * fine-tune
      
↓
      
**Topic-level + Entity-level Aggregation**
(05_Topic_and_Entity_Sentiment.ipynb)
* Predict sentiment
* Topic-level aggregation ()
* Entity-level sentiment (ORG: company + industry, TECH)
* Visualization over time
   - Industry sentiment trend
	- Company sentiment volatility
	- Emerging industries (growth curve)
      
↓
      
**Business Insight Layer**
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
