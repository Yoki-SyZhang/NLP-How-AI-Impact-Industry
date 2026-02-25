# NLP-How-AI-Impact-Industry
With ~200K news articles on DS, ML and AI, I aim to Identify industries and their companies that are most likely to be impacted by AI over the next several years. Determine how those industries and their companies will be impacted and by what means / technologies. Provide insights into what can make AI adoption successful or unsuccessful.

## Break down the tasks:

**Q1: Identify industries and their companies that are most likely to be impacted by AI over the next several years.？**
* Topic Detection
* Entity Extraction（industries and their companies）
* Frequency + Co-occurrence
* Time Analysis

**Q2: Determine how those industries and their companies will be impacted and by what means / technologies？**
* Sentiment analysis on how those are impacted (positively, negatively, hard-to-say, etc.) 
* Topic-level sentiment
* Entity-level sentiment
* Keyword/phrase mining（impact mechanism）

**Q3: Insights into what can make AI adoption successful or unsuccessful？**
* Topic modeling + clustering
* Extractive summarization
* Phrase mining
* LLM API for synthesis

## Project Pipeline:
**Raw News Articles**

↓

**Clean + Filter**   
(01_Data_Exploration_and_Cleaning.ipynb ->  df_clean.parquet)
1. Data profiling
    * HTML残留
    * 空文本
    * 重复
    * irrelevant crawl artifacts
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
2. Topic modeling: BERTopic 
3. Label topics using LLM API

↓

**Entity Extraction** 
(03_Entity_Extraction.ipynb -> df_entities.parquet)
* How to Identify industries
* Find pretrained NER model
    * ORG（company）
    * PRODUCT（technology）
    * GPE（country）
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
    * df_entities.groupby(["topic","year"])["sentiment"].mean()
      
↓
      
**Business Insight Layer**
(06_Impact_Mechanism_Analysis.ipynb)
* Extractive summarization
    * TextRank
    * LLM API 
* Answer
    * how those industries and their companies will be impacted by what means / technologies
    * Provide insights into what can make AI adoption successful or unsuccessful.
