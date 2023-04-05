# Investigate Prompt Learning

In this project, we investigate hard prompt and soft prompt on different NLP tasks, including:
- Text Classification
  - Datasets: Yahoo_answer_topic
- Sentiment Analysis
  - Datasets: Amazon Review, IMDB
- Semantic Inference
  - Datasets: SNLI
- Named Entity Recognition (NER)
  - Datasets: CoNLL 2003

> Hard Prompt 

A pre-defined hard template is added into each raw input sample, and ask the model to predict the **_mask_** token as the masked language model. 
Use sentiment analysis as an example:
  
          raw input: "I like this book."
          prompt: "Overall, it was <mask>."
          final input: "Overall, it was <mask>. I like this book."
          supervised training label: "Overall, it was good. I like this book."
          
This is able to eliminate the gap between pre-trained language model task and downstream tasks.
On the other hand, we don't need to fine-tune the additional parameters adding by the downstream tasks (e.g., 768 x 2 for RoBERTa on sentiment analysis).

> Soft Prompt 


