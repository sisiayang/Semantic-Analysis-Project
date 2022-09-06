import transformers
import torch.nn as nn

MAX_LEN = 150
BATCH_SIZE = 32
EPOCHS = 2
BERT_PATH = 'bert-base-chinese'
MODEL_PATH = '../bert_model'
TRAINING_FILE = '../training_data/train_df.csv'
TESTING_FILE = '../training_data/test_df.csv'
VALID_FILE = '../training_data/valid_df.csv'
LOSS_FUN = nn.CrossEntropyLoss()
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH, 
    do_lower_case = True
)