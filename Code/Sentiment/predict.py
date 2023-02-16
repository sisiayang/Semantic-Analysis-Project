import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def bert_predict(model, test_dataloader, device=None):
    model.eval()
    all_logits = []

    for batch in tqdm(test_dataloader):
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs

def get_predict_class(probs):
    # Get predictions from the probabilities
    threshold = 0.5
    preds = np.where(probs[:, 1] > threshold, 1, 0)
    return preds

def get_predict_score(probs):
    score = probs[:, 1]
    score = [x*10 for x in score]
    #score = min_max_normolize(score)
    #score = [round(x) for x in score]
    return score

def evaluate_accuracy_score(probs, y_true):
    y_pred = get_predict_class(probs)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'\nAccuracy: {accuracy*100:.2f}%')

def min_max_normolize(preds):
    new_preds = []
    norm_dict = {2: (2, 0, 3, 0), 8: (8, 2, 6, 4), 10: (10, 8, 10, 7)}
    for pred in preds:
        for key in norm_dict.keys():
            if(pred <= key):
                num_tuple = norm_dict[key]
                max, min, new_max, new_min = num_tuple[0], num_tuple[1], num_tuple[2], num_tuple[3]
                break
        
        new_pred = ((pred-min)/(max-min))*(new_max-new_min)+new_min
        new_preds.append(new_pred)

    return new_preds

def process_nan_data(message, score):
    for i in range(len(message)):
        if(str(message[i]) == 'nan'):
            score[i] = -1
            
    return score