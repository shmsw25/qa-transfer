import json
import os

def create_shared(config, shared, shared_path):

    if config.using_shared:
        print ("move shared")
        a = input()
        return

    word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
    word_counter = shared['lower_word_counter'] if config.lower_word else shared['word_counter']
    char_counter = shared['char_counter']
    if config.finetune:
        shared['word2idx'] = {word: idx + 2 for idx, word in
                                  enumerate(word for word, count in word_counter.items()
                                            if count > config.word_count_th or (config.known_if_glove and word in word2vec_dict))}
    else:
        assert config.known_if_glove
        assert config.use_glove_for_unk
        shared['word2idx'] = {word: idx + 2 for idx, word in
              enumerate(word for word, count in word_counter.items()
              if count > config.word_count_th and word not in word2vec_dict)}
        shared['char2idx'] = {char: idx + 2 for idx, char in
              enumerate(char for char, count in char_counter.items()
              if count > config.char_count_th)}
    NULL = "<NULL>"
    UNK = "<UNK>"
    shared['word2idx'][NULL] = 0
    shared['word2idx'][UNK] = 1
    shared['char2idx'][NULL] = 0
    shared['char2idx'][UNK] = 1
    json.dump({'word2idx': shared['word2idx'], 'char2idx': shared['char2idx']}, open(shared_path, 'w'))

def get_word2vec(word_counter):
    glove_corpus = '6B'
    glove_vec_size= 100
    
    glove_path = os.path.join("/home/sewon/data/glove", "glove.{}.{}d.txt".format(glove_corpus, glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


