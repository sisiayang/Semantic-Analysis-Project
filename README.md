# Semantic-Analysis-Project
分析社群網路使用者的留言內容，又可細分為兩部分：
* 情緒分析
* 主題分析

### 情緒分析
利用 BERT 架一個 classifier，將情緒分為 0分(negative) ~ 10分(positive)
### 主題分析
根據留言內容分為13大不同的主題  
* 步驟
  1. 利用 keybert 提取留言關鍵字
  2. train Word2Vec model
  3. 分類最常出現的關鍵字（人工）
  4. 對所有留言，將關鍵字和第三步分好的關鍵字做相似度比對，一旦超過 threshold 即列為主題候選，取最相似的作為該留言主題  
  注：若是都不符合，歸為無法分類
