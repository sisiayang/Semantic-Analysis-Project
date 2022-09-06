import pandas as pd
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

def tokenizer(text):
    tmp = text.split(' ')
    return tmp

def getKetwords(dataframe, keybert_model, vectorizer):
    messages = dataframe['message'].tolist()
    keywords_list = []
    for i in tqdm(messages):
        try:
            keywords = keybert_model.extract_keywords(str(i), vectorizer=vectorizer, top_n=3)
        except AssertionError:
            keywords = [('', 0), ('', 0), ('', 0)]

        if((keywords == []) or (keywords[0][0] == 'nan')):
            keywords = [('', 0), ('', 0), ('', 0)]
        
        # 補齊關鍵字數量
        if(len(keywords) < 3):
            for j in range(3-len(keywords)):
                keywords.append(('', 0))

        # 處理異常狀況 (關鍵字長度過長)
        k = 0
        while(k != 3):
            if(len(str(keywords[k][0])) > 5):
                keywords.remove(keywords[k])
                keywords.append(('', 0))
            else:
                k += 1

        keywords_list.append(keywords)

    kw1 = []
    kw2 = []
    kw3 = []

    for i in keywords_list:
        kw1.append(list(i[0])[0])
        kw2.append(list(i[1])[0])
        kw3.append(list(i[2])[0])

    message_and_keywords_df = pd.DataFrame.from_dict({'message': messages, 'kw1': kw1, 'kw2': kw2, 'kw3': kw3})
    return message_and_keywords_df


def run(df, i):
    vectorizer = CountVectorizer(tokenizer=tokenizer, token_pattern=None)
    keybert_model = KeyBERT()
    output_path = ['../keywords_df/comments.csv', '../keywords_df/comments_comments.csv', '../keywords_df/allfeed.csv']

    keywords_df = getKetwords(df, keybert_model, vectorizer)
    keywords_df.to_csv(output_path[i], index=False)

    return keywords_df
    