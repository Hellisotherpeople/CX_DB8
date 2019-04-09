import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, BertEmbeddings, ELMoEmbeddings
import torch
# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
#import numpy as np
from docx import Document 
import sys
import numpy as np
from itertools import islice
from collections import deque



document = Document() ## Create a python-docx document 


class underline: ## weird stuff to make python show underlining and highlighting in it's printed output - kinda neat
    start_underline='\033[04m'
    end_underline = '\033[0m'
    start_highlight = '\x1b[4;37;45m'


"""
###### The Stacked Embeddings are the list of word embeddings which we choose for converting a word window into a "meaning" vector. #####

To enable a set of embeddings, remove the pound sign "#" from the beginning of the Embedding. 

FAST: To summarize extremely fast, enable on the "WordEmbeddings" selections (en is fasttext, extvec is a skip-gram model) 
MEDIUM: To summarize with much more semantic understanding but running still relatively quickly, enable the "FlairEmbeddings" with the word "fast" at the end (this is the default). This is a lot slower than "FAST" but has dramatically improved abilities to understand meaning. 
SLOW: To summarize with the state-of-the-art, but much more time consuming concatonation of BERT, Flair, and FastText, uncomment the "BertEmbeddings" layer, and remove the word "-fast" from the end of "news-forward-fast" and "news-backward-fast". This is about 2x as slow  as the medium settings 


"""
stacked_embeddings = DocumentPoolEmbeddings([
                                        WordEmbeddings('en'),
                                        #WordEmbeddings('glove'),
                                        WordEmbeddings('extvec'),#ELMoEmbeddings('original'),
                                        #BertEmbeddings('bert-base-cased'),
                                        #FlairEmbeddings('news-forward-fast'),
                                        #FlairEmbeddings('news-backward-fast'),
                                        ]) #, mode='max')

def set_card():
    print("Input the Card Text, press Ctrl-D to end text entry")
    card = sys.stdin.read()#input("Input the Card Text: ")
    card_tag = input("Input the card_tag, or a -1 to summarize in-terms of the card itself: ")
    card = str(card)
    if str(card_tag) == "-1": #This will not work with large documents when bert is enabled
        card_tag = Sentence(str(card))
        tag_str = ""
    else:    
        tag_str = str(card_tag)
        card_tag = Sentence(str(card_tag))
    return card, card_tag, tag_str

def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def create_ngram(num_context, card, card_tag):
    #card = set_card()
    card_words_org = card.split()
    ngram_list = []
    #print(len(card_words_org))
    
    for i in range(0, len(card_words_org)):
        #print(i)
        '''
        if i >= len(card_words_org) - (num_context):
            new_word = card_words_org[i-num_context:i]
        elif i >= num_context:
            new_word = card_words_org[i-num_context:i+num_context]
        else:
            new_word = card_words_org[i:i+num_context] #make it so that each word takes it's prior words as context as well 
        '''
        
        new_m = card_words_org[i:i]
        new_l = card_words_org[i-num_context:i]
        new_r = card_words_org[i:i+num_context]
        new_word = new_l + new_m + new_r
        print(new_word)
    #for new_word in window(card_words_org, num_context):
        #print(list(new_word))
        new_string = ''
        for word in list(new_word):
            new_string += word
            new_string += " "
        if new_string == "":
            new_string = " "
        ngram_list.append(Sentence(new_string))
        

    card_words = ngram_list
    return card_words, card_words_org




def embed(card_tag, card_as_sentence, card_words, card_words_org):
    stacked_embeddings.embed(card_tag)
    #stacked_embeddings.embed(card_as_sentence)
    #print(card_as_sentence.get_embedding().reshape(1,-1))
    word_list = []
    count = 0
    token_removed_ct = 0
    card_tag_emb = card_tag.get_embedding().reshape(1,-1)
    for word in card_words_org: #card_as_sentence:
        #word = word.reshape(1,-1)
        n_gram_word = card_words[count]
        #print(n_gram_word)
        '''
        if len(n_gram_word) == 0:
            th_tensor = torch.zeros(list(card_tag.get_embedding().size()))
            word_sim = cosine_similarity(card_tag.get_embedding().reshape(1,-1), th_tensor.reshape(1,-1))
        else:
        '''
        stacked_embeddings.embed(n_gram_word)
        word_sim = cosine_similarity(card_tag_emb, n_gram_word.get_embedding().reshape(1,-1))
        #word_sim = jaccard_similarity_score(card_tag_emb, n_gram_word.get_embedding().reshape(1,-1), normalize=True)
        word_tup = (card_words_org[count], word_sim) #card_words_org[count]
        count += 1
        word_list.append(word_tup)
    print(len(word_list))
    print(len(card_words))
    print(len(card_words_org))
    return word_list


def summarize(word_list):
    word_val_list = []
    for sum_word in word_list:
        word_val_list.append(float(sum_word[1]))
    return word_val_list

def run_loop(context, card, card_tag):
    card_as_sentence = Sentence(card)
    card_words, card_words_org = create_ngram(context, card, card_tag)
    word_list = embed(card_tag, card_as_sentence, card_words, card_words_org)
    return word_list

def parse_word_val_list(word_list, h, n):
    sum_str = ""
    removed_str = ""
    token_removed_ct = 0
    for sum_word in word_list:
        if float(sum_word[1]) > h:
            sum_str += underline.start_highlight
            runner = par.add_run(sum_word[0] + " ")
            runner.underline = True
            runner.bold = True
            sum_str += str(sum_word[0])
            sum_str += " "
            sum_str += underline.end_underline
        elif float(sum_word[1]) > n:
            sum_str += underline.start_underline
            runner = par.add_run(sum_word[0] + " ")
            runner.underline = True
            sum_str += str(sum_word[0])
            sum_str += " "
            sum_str += underline.end_underline
        else:
            token_removed_ct += 1
            sum_str += str(sum_word[0])
            runner = par.add_run(sum_word[0] + " ")
            sum_str += " "
            removed_str += str(sum_word[0])
            removed_str += " "
    return sum_str, removed_str, token_removed_ct

def new_sum(word_list):
    val_list = summarize(word_list)
    th_9 = np.percentile(val_list, 90)
    th_75 = np.percentile(val_list, 60)
    th_50 = np.percentile(val_list, 50)
    th_25 = np.percentile(val_list, 35)
    th_10 = np.percentile(val_list, 10)
    sum_str, removed_str, token_removed_ct = parse_word_val_list(word_list, th_75, th_25)


    print("CARD_TAG :")
    print(card_tag)
    print("CARD: ")
    print(card)
    print("GENERATED SUMMARY: ")
    print(sum_str)
    print("tokens removed:" + " " + str(token_removed_ct))
    print("NOT UNDERLINED")

    print(removed_str)


for i in range(0, 1000):
    cont = input("Do you want to continue? type y for yes else for stop!")
    if cont == "y":
        card, card_tag, tag_str = set_card()
        head = document.add_heading(tag_str, level=1)
        card_auth = input("what's the card author and date?")
        card_cite = input("what's the citation?")
        a_par = document.add_paragraph()
        a_par.add_run(card_auth).bold = True
        c_par = document.add_paragraph(card_cite)
        par = document.add_paragraph()
        word_list = run_loop(20, card, card_tag)
        new_sum(word_list)
    else:
        document.save('test_sum.docx')
        break


