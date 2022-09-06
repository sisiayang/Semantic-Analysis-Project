from operator import index
import get_keywords
import get_topic
import pandas as pd


if __name__ == '__main__':
    input_path = ['../../input/messages_tokenize/comments.txt', '../../input/messages_tokenize/comments_comments.txt', '../../input/messages_tokenize/allfeed.txt']
    output_path = ['../../output/comments.csv', '../../output/comments_comments.csv', '../../output/allfeed.csv']
    local_output_path = ['../output/comments.csv', '../output/comments_comments.csv', '../output/allfeed.csv']
    
    for i in range(3):
        with open(input_path[i], 'r', encoding='utf-8') as f:
            txt = f.readlines()
            txt = [w.replace('\n', '') for w in txt]
        
        df = pd.DataFrame.from_dict({'message': txt})
        print('get keyword...')
        df = get_keywords.run(df, i)
    
        print('get topic...')
        df['topic'] = 0
        df = get_topic.run(df)

        df.to_csv(local_output_path[i], index=False)

        output_df = pd.read_csv(output_path[i])
        output_df['topic'] = df['topic'].tolist()
        output_df.to_csv(output_path[i], index=False)

        print('Done！\n--------------------------------------------\n')

