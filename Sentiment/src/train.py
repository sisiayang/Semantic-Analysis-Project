import config
import pandas as pd
import torch
from data_transform import tokenize, get_dataloader
from engine import set_seed, initialize_model, train
from predict import bert_predict, evaluate_accuracy_score
from model import BertClassifier

def process_data(df_path):
    df = pd.read_csv(df_path)
    X, y = df['review'].tolist(), df['label'].tolist()
    input_ids, attention_masks = tokenize(X)
    y = torch.tensor(y)
    dataloader = get_dataloader(input_ids, attention_masks, y)

    return dataloader

def run(device=None):
    print('get dataloader...')
    train_dataloader = process_data(config.TRAINING_FILE)
    valid_dataloader = process_data(config.VALID_FILE)
    
    set_seed(42)
    bert_classifier, optimizer, scheduler = initialize_model(device=device, total_num=len(train_dataloader))
    print('training...')
    train(
        model=bert_classifier, 
        train_dataloader=train_dataloader, 
        val_dataloader=valid_dataloader, 
        evaluation=True, 
        device=device, 
        optimizer=optimizer, 
        scheduler=scheduler
    )
    print('save the model!')
    torch.save(bert_classifier, config.MODEL_PATH)

if __name__== "__main__" :
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    run(device=device)

    # use model to evaluate
    print('evaluate accuracy...')
    bert_model = torch.load(config.MODEL_PATH)

    df = pd.read_csv(config.TESTING_FILE)
    X, y = df['review'].tolist(), df['label'].tolist()
    input_ids, attention_masks = tokenize(X)
    dataloader = get_dataloader(input_ids, attention_masks)

    probs = bert_predict(bert_model, dataloader, device)
    evaluate_accuracy_score(probs, y)