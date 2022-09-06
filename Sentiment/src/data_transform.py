import config
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def tokenize(messages):
    tokenizer = config.TOKENIZER
    input_ids = []
    attention_masks = []
    for mes in tqdm(messages):
        if(str(mes) == 'nan'):
            mes = ' '
        # 轉成token
        tokens = tokenizer.encode_plus(mes, add_special_tokens=True, max_length=config.MAX_LEN, padding='max_length', truncation=True)
        input_ids.append(tokens.get('input_ids'))
        attention_masks.append(tokens.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def get_dataloader(input_ids, attention_masks, labels=None):
    # Create the DataLoader
    if(labels == None):
        dataset = TensorDataset(input_ids, attention_masks)

    else:   # for training、valid
        dataset = TensorDataset(input_ids, attention_masks, labels)

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=config.BATCH_SIZE)

    return dataloader