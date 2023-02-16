import pandas as pd
import torch
from data_transform import tokenize, get_dataloader
from predict import bert_predict, get_predict_score, min_max_normolize, process_nan_data
from model import BertClassifier

if __name__ == '__main__':
    input_path = ['../../input/messages_to_be_analyzed/comments.csv', '../../input/messages_to_be_analyzed/comments_comments.csv', '../../input/messages_to_be_analyzed/allfeed.csv']
    output_path = ['../../output/comments.csv', '../../output/comments_comments.csv', '../../output/allfeed.csv']
    local_output_path = ['../output/comments.csv', '../output/comments_comments.csv', '../output/allfeed.csv']
    bert_classifier = torch.load('../bert_model')

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    for i in range(3):
        df = pd.read_csv(input_path[i])
        output_df = pd.read_csv(output_path[i])

        print('process data...')
        messages = df['message'].tolist()
        input_ids, attention_masks = tokenize(messages)
        dataloader = get_dataloader(input_ids, attention_masks)

        print('predict the result...')
        probs = bert_predict(bert_classifier, dataloader, device)
        score = get_predict_score(probs)
        new_score = min_max_normolize(score)

        print('process nan data...')
        new_score = process_nan_data(messages, new_score)

        new_score = [round(x) for x in new_score]
        df['orig_sentiment'] = score
        df['sentiment'] = new_score
        df.to_csv(local_output_path[i], index=False)

        output_df['sentiment'] = new_score
        output_df.to_csv(output_path[i], index=False)

        print('done\n--------------------------------------------------')