import glob
import os
from bs4 import BeautifulSoup
import bs4
import string
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
import csv
from random import shuffle
stacked_embeddings = DocumentPoolEmbeddings([
                                        WordEmbeddings('en'),
                                        #WordEmbeddings('glove'),
                                        #WordEmbeddings('extvec'),#ELMoEmbeddings('original'),
                                        #BertEmbeddings('bert-base-cased'),
                                        #FlairEmbeddings('news-forward-fast'),
                                        #FlairEmbeddings('news-backward-fast'),
                                        ]) #, mode='max')


num_of_cards = 0
os.chdir("/home/lain/manjaro-windows-shared/card_data")

cards = dict()
with open('card_classification.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile)
    with open('card_data_pos.txt', 'a') as f:
        paths = glob.glob("*.html5")
        shuffle(paths)
        for file in paths:
            print("File parsed: " + file)
            with open(file) as fp:
                soup = BeautifulSoup(fp, "lxml")
                all_card_tags = soup.find_all('h4')
                for h4 in all_card_tags:
                    try:
                        card_str = h4.find_next("p").find_next("p").text
                        tag_str = h4.text
                        #cards.update({'tag': tag_str, 'cite': h4.find_next("strong").text, 'pocket': h4.find_previous("h1").text, 'hat': h4.find_previous("h2").text, 'block': h4.find_previous("h3").text, 'card' : card_str})
                        card_dict = {'cite': h4.find_next("strong").text, 'pocket': h4.find_previous("h1").text, 'hat': h4.find_previous("h2").text, 'block': h4.find_previous("h3").text, 'tag/query': tag_str,'card' : card_str}
                        num_of_cards += 1
                        #if num_of_cards % 10 == 0:
                        #    print(cards.items)
                        print(card_dict)
                        label = input("What is the ground truth label of this?")
                        if label == "":
                            pass
                        elif label == "f":
                            break
                        elif label == "stop":
                            csvfile.close()
                            sys.exit()
                        else:
                            spamwriter.writerow([label, card_dict['card']])
                            csvfile.flush()
                    except SystemExit:
                        print("Exiting!")
                        sys.exit()
                    except:
                        pass
                    

'''
            counter = 0
            num_of_cards += 1
            card_str = ""
            for content in h4.next_siblings:
                    if content.name == "h4":
                        print("------ TAG -------")
                        print(content.text)
                        print("---------- CARD --------")
                        print(card_str)
                        if num_of_cards == 2:
                            c_set = Sentence(card_str)
                            t_set = Sentence(content.text)
                            stacked_embeddings.embed(c_set)
                            stacked_embeddings.embed(t_set)
                            card_emb = c_set.get_embedding().reshape(1,-1)
                            tag_embedding = t_set.get_embedding().reshape(1,-1)
                                f_embe = np.concatenate((card_emb, tag_embedding), axis=None)
                                print(f_embe)

                            break
                        elif content.name == "p" and counter == 0:
                            # this is likely the tag
                            #tag = content.text
                            #print(tag)
                            counter += 1
                        elif content.name == "h1":
                            pass
                        elif content.name == "h2":
                            pass
                        elif content.name == "h3":
                            pass
                        elif content.name == "p":
                            card_str += content.text
                        else:
                            pass
'''
'''
    with open('card_data.txt', 'a') as f:
        for item in data:
            if item is '\n':
                f.write('\n')
            else:
                f.write(str(item['data']) + '\t' + str(item['tag']) + '\n')
'''
print("Number of cards processed: " + str(num_of_cards))

