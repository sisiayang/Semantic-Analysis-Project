import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import word2vec
import config

w2v = word2vec.Word2Vec.load('../Word2Vec/word2vec.model')

def get_more_keywords(df, topic):
    keywords = config.use_to_find_keywords[topic]
    remove_words = config.remove_keywords[topic]
    append_words = config.append_keywords[topic]
    threshold = config.extra_threshold_list[topic]
    more = []

    # 節日為特例
    if(topic == '節日'):
        for i in range(len(df)):
            for kw_idx in ['kw1', 'kw2', 'kw3']:
                word = str(df.iloc[i][kw_idx])
                if(word.endswith(keywords[0]) and (w2v.wv.n_similarity(keywords[0], word) > threshold)):
                    more.append(word)
                    break
    else:
        for i in tqdm(range(len(df))):
            for kw_idx in ['kw1', 'kw2', 'kw3']:
                word = str(df.iloc[i][kw_idx])

                for w in keywords:
                    if((w in word) and (w2v.wv.n_similarity(w, word) > threshold)):
                        more.append(w)
                        break

    more = list(set(more))
    for i in remove_words:
        if(i in more):
            more.remove(i)
    for i in append_words:
        if(i not in more):
            more.append(i)

    config.more_keywords[topic] = more


def extra_processing(df, topic):
    keywords = config.more_keywords[topic]
    for idx in tqdm(range(len(df))):
        if(df.iloc[idx]['topic'] == 0):
            b = 0
            for key_label_num in ['kw1', 'kw2', 'kw3']:
                for key in keywords:
                    if(str(df.iloc[idx][key_label_num]) == key):
                        df.loc[idx, 'topic'] = topic
                        b = 1
                        break
                if(b == 1):
                    break
    return df


def get_similarity(word, topic):
    keywords = config.topic_keywords[topic]
    threshold = config.threshold_list[topic]
    for key in keywords:
        similarity = w2v.wv.similarity(key, word)
        if(similarity > threshold):
            return similarity
    return 0

def get_topic(df, topic):
    for idx in tqdm(range(len(df))):
        if(df.iloc[idx]['topic'] == 0):
            for key_label_num in ['kw1', 'kw2', 'kw3']:
                word = str(df.iloc[idx][key_label_num])
                if(word not in w2v.wv.index_to_key):
                    continue

                if(get_similarity(word, topic) != 0): # 表示 > threshold
                    df.loc[idx, 'topic'] = topic
                    break
                
    return df

def run(df):
    topic_list = [['旅遊', '飲食', '工作'], ['節日'], ['生日'], ['金錢'], ['交通', '課業', '氣象'], ['政治', '休閒娛樂'], ['生活', '年齡']]
    key_label = ['kw1', 'kw2', 'kw3']
    
    '''
    對於每個topic, 遍歷整個df, 若是已有topic則跳過該留言, 若沒有topic則進行以下比較: 
    1. 是否為分組檢查 -> if(len(topic) > 1)
    2. 是否需要額外找keyword? -> if(config.use_to_find_keywords[topic[t]] != [])
    3. 是否需要額外處理 -> if(config.more_keywords[topic[t]] != [])
    '''
    for topic in tqdm(topic_list):
        print('\n\n', topic)
        
        if(len(topic) == 1):
            print('first processing...')
            df = get_topic(df, topic[0])
            # 是否需要額外處理
            print('second processing...')
            if(config.use_to_find_keywords[topic[0]] != []):
                get_more_keywords(df, topic[0])  # 從df中取得關鍵字，會存進more_keywords中
            if(config.more_keywords[topic[0]] != []):
                df = extra_processing(df, topic[0])   # 用前面取得的關鍵字分類

        else:   # 多個同時進行
            print('first processing...')
            for idx in tqdm(range(len(df))):
                if(df.iloc[idx]['topic'] != 0):
                    continue
                label_list = [0] * len(topic)
                for kw_num in key_label:
                    word = str(df.iloc[idx][kw_num])
                    if((word not in w2v.wv.index_to_key) or word == 'nan'):
                        break
                    for t_idx in range(len(topic)):
                        label_list[t_idx] = get_similarity(word, topic[t_idx])

                    # 若已經有符合的topic，歸類為該主題並不用繼續看後續的kw
                    if(max(label_list) != 0):
                        df.loc[idx, 'topic'] = topic[np.argmax(label_list)]
                        break

            print('second processing...')
            # 處理額外處理
            for t in topic:
                # 是否需要額外處理
                if(config.use_to_find_keywords[t] != []):
                    get_more_keywords(df, t)  # 從df中取得關鍵字，會存進more_keywords中
                if(config.more_keywords[t] != []):
                    df = extra_processing(df, t)   # 用前面取得的關鍵字分類

        print('---------------------------------------------')
    df.loc[df['topic'] == 0, 'topic'] = '無法分類'
    return df