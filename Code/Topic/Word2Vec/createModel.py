import pandas as pd
from gensim.models import word2vec
import jieba
from opencc import OpenCC
from tqdm import tqdm
from gensim.corpora import WikiCorpus
import config

def get_wiki_corpus():
    wiki_corpus = WikiCorpus('wiki_articles/zhwiki-20220701-pages-articles-multistream.xml.bz2', dictionary={})
    text_num = 0

    with open('wiki_articles/wiki_text.txt', 'w', encoding='utf-8') as f:
        for text in wiki_corpus.get_texts():
            f.write(' '.join(text)+'\n')
            text_num += 1
            if text_num % 100000 == 0:
                print('{} articles processed.'.format(text_num))

        print('{} articles processed.'.format(text_num))

def get_wiki_text_seg():
    # Initial
    cc = OpenCC('s2t')
    # Tokenize
    with open('wiki_articles/wiki_text_seg.txt', 'w', encoding='utf-8') as new_f:
        with open('wiki_articles/wiki_text.txt', 'r', encoding='utf-8') as f:
            for times, data in tqdm(enumerate(f, 1)):
                # print('data num:', times)
                data = cc.convert(data)
                data = jieba.cut(data)
                data = [word for word in data if word != ' ']
                data = ' '.join(data)

                new_f.write(data)

def train():
    train_data = word2vec.LineSentence('wiki_articles/wiki_text_seg.txt')
    w2v = word2vec.Word2Vec(
        train_data,
        min_count=config.min_count,
        vector_size=config.vector_size,
        workers=config.workers,
        epochs=config.epochs,
        window=config.window_size,
        sg=config.sg,
        seed=config.seed,
        batch_words=config.batch_words
    )
    w2v.save(config.save_path)

def extended_train():
    w2v = word2vec.Word2Vec.load(config.save_path)
    data_path = ['../../input/messages_tokenize/comments.txt', '../../input/messages_tokenize/comments_comments.txt', '../../input/messages_tokenize/allfeed.txt']

    more_sentence = []

    for i in range(3):
        f = open(data_path[i], 'r', encoding='utf-8')
        for i in f.readlines():
            i = i.replace('\n', '').split(' ')
            more_sentence.append(i)
        f.close()
    print('more sentence num: ', len(more_sentence))
    w2v.build_vocab(more_sentence, update=True)
    w2v.train(more_sentence, total_examples = w2v.corpus_count, epochs=4)
    w2v.save(config.save_path)

if __name__ == '__main__':
    print('get corpus...\n')
    get_wiki_corpus()
    get_wiki_text_seg()
    print('start training...\n-------------------------------------\n')
    train()
    print('\nextended train...')
    extended_train()
    print('--------------------------------\ncomplete!')