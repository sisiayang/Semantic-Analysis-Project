import pandas as pd
import numpy as np
from tqdm import tqdm
from  ckiptagger import data_utils, WS, POS, NER

def get_ownerFbid(input_path, output_path):
    df = pd.read_csv(input_path)
    print('get OwnerFbid...\n')
    OwnerFbid = df['api_id'].dropna().astype(np.longlong).tolist()
    OwnerFbid_df = pd.DataFrame.from_dict({'OwnerFbid': OwnerFbid})

    OwnerFbid_df.to_csv(output_path, encoding='utf-8', index=False)
    return OwnerFbid_df

def get_messages(input_path, output_path, OwnerFbid):

    comments_df = pd.read_csv(input_path)
    # 只需留下OwnerFbid、Fbid、message
    c = list(comments_df.columns)
    c.remove('OwnerFbid')
    c.remove('message')
    c.remove('Fbid')

    comments_df = comments_df.drop(columns=c, axis=1)
    print('包含NaN: ', len(comments_df))
    comments_df = comments_df.dropna()
    print('去除NaN: ', len(comments_df))
    comments_df.head()

    message_df = pd.DataFrame([], columns=['OwnerFbid', 'Fbid', 'message'])
        
    for j in tqdm(range(len(comments_df))):
        d = comments_df.iloc[j:j+1, :]
        if(d['OwnerFbid'].values in OwnerFbid):
            message_df = pd.concat([message_df, d])
    print('final dataframe size: ', len(message_df))

    message_df.to_csv(output_path, index=False)
    return message_df

# 將相同post留言合併為文章
# def get_combine_message(df, output_path):
#     df = df.sort_values(by=['Fbid'])

#     new_df = pd.DataFrame(columns=['Fbid', 'message'])
#     Fbid = df.iloc[0:1, :]['Fbid'].tolist()[0]
#     new_message = ''
#     for i in tqdm(range(len(df))):
#         d = df.iloc[i:i+1, :]
#         tmp_Fbid = d['Fbid'].tolist()[0]
#         message = d['message'].tolist()[0]
#         #print(new_message)
#         if(str(message) == 'nan'):
#             message = ''

#         if(tmp_Fbid == Fbid):
#             new_message = new_message + '。' + message
#         else:
#             new_df = pd.concat([new_df, pd.DataFrame([[Fbid, new_message]], columns=['Fbid', 'message'])])
#             Fbid = tmp_Fbid
#             new_message = message

#     if(tmp_Fbid == Fbid):
#         new_df = pd.concat([new_df, pd.DataFrame([[Fbid, new_message]], columns=['Fbid', 'message'])])
#     print('combine df size: ', len(new_df))
#     new_df.to_csv(output_path, index=False)
#     return new_df