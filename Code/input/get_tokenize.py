import pandas as pd
import numpy as np
from tqdm import tqdm
from  ckiptagger import data_utils, WS, POS, NER

ws = WS('../data')   # 斷詞
pos = POS('../data') # 詞性標記
ner = NER('../data') # 命名實體識別
# get stopwords
with open('../stopwords.txt', encoding = 'UTF-8') as f:
    stop_words = f.readlines()
    stop_words = [w.replace('\n', '') for w in stop_words]  #s.replace(old, new[, max])
    stop_words = [w.replace(' ', '') for w in stop_words]

def tokenizer(text):
    remove_list = ['。', '.', '，', ',', '！', '!', '？', '?', '~', ' ']
    ws_result = ws([text])  # 一次僅能一個句子(？
    pos_result = pos(ws_result)
    ner_result = ner(ws_result, pos_result)
    for i in ner_result[0]:
        if(('PERSON' in i) or ('TIME' in i)):  # 去除人名或時間
            t = list(i)
            if(t[3] in ws_result[0]):
                ws_result[0].remove(t[3])
                
    for i in range(len(ws_result[0])):
        # 去掉空格或換行字符
        ws_result[0][i] = ws_result[0][i].strip()
        # 去掉原本留著的標點符號
        for j in remove_list:
            ws_result[0][i] = ws_result[0][i].replace(j, '')

    tmp = list(filter(lambda x: x not in stop_words, ws_result[0]))
    return tmp

def get_data_token(df, output_path):

    df.fillna("",inplace=True)
    speech_list = list(df["message"])
    
    for i in tqdm(range(len(speech_list))):
        speech_list[i] = tokenizer(speech_list[i])
    for idx, speech in enumerate(speech_list):
        speech_list[idx] = ' '.join([word for word in speech if word.strip() not in stop_words])

    f = open(output_path, 'w', encoding="utf-8")
    for i in speech_list:
        f.write(i)
        f.write('\n')
    f.close()
    
    
    

    