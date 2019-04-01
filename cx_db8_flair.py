import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, BertEmbeddings, ELMoEmbeddings
import torch
# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from docx import Document 

document = Document()


class underline:
    start_underline='\033[04m'
    end_underline = '\033[0m'
    start_highlight = '\x1b[4;37;45m'


stacked_embeddings = DocumentPoolEmbeddings([
                                        WordEmbeddings('en'),#WordEmbeddings('glove'),
                                        WordEmbeddings('extvec'),#ELMoEmbeddings('original'),
                                        #BertEmbeddings('bert-base-cased'),
                                        #FlairEmbeddings('news-forward-fast'),
                                        #FlairEmbeddings('news-backward-fast')
                                       ])

def set_card():
    card = input("Input the Card Text: ")
    card_tag = input("Input the card_tag, or a -1 to summarize in-terms of the card itself: ")
    card = str(card)
    tag_str = str(card_tag)
    if str(card_tag) == "-1":
        print("YOU DIDN'T USE A CARD")
        card_tag = Sentence(str(card))
    else:
        card_tag = Sentence(str(card_tag))
    return card, card_tag, tag_str

#card_tag = Sentence('The Executive Order harms health, safety, and the environment')

"""
card = 
The Executive Order will block or force the repeal of regulations needed to protect health, safety, and the environment, across a broad range of topics—from automobile safety, to occupational health, to air pollution, to endangered species. 4. The Executive Order mandates that, when implementin g the command to repeal at least two rules for each new one, agencies must focus on costs while ignoring benefits. Indeed, the Executive Order directs agencies to disregard the benefits of new and existing rules— including benefits to consumers, to workers, to people exposed to pollution, and to the economy—even when the benefits far exceed costs. The Executive Order’s direction to federal agencies to zero out costs to regulated industries, while entirely ignoring benefits to the Americans whom Congress enacted these statutes to protect, will force agencies to take regulatory actions that harm the people of this nation. 5. To repeal two regulations for the purpose of adopting one new one, based solely on a directive to impose zero net costs and without any consideration of benefits, is arbitrary, capricious, an abuse of discretion, and not in accordance with law, for at least three reasons. First, no governing statute authorizes any agency to withhold a regulation intended to address identified harms to public safety, health, or other statutory objectives on the basis of an arbitrary upper limit on total costs (for fiscal year 2017, a limit of $0) that regulations may impose on regulated entities or the economy. Second, the Executive Order forces agencies to repeal regulations that they have already determined, through notice-and-comment rulemaking, advance the purposes of the underlying statutes, and forces the agencies to do so for the sole purpose of eliminating costs that the underlying statutes do not direct be eliminated. Third, no governing statute authorizes an agency to base its actions on a decisionmaking criterion of zero net cost across multiple regulations. Case 1:17-cv-00253 Document 1 Filed 02/08/17 Page 4 of 49 5 6. Rulemaking in compliance with the Executive Order’ s “1-in, 2-out” requirement cannot be undertaken without violating the statutes from which the agencies derive their rulemaking authority and the Administrative Procedure Act (APA). 7. The implementation of governing statutes, passed by Congress and signed into law by previous Presidents, will slow to a halt under the Executive Order. In addition to complying with the substantive requirements of thos e laws and the procedural requirements of the APA, agencies, to issue a new proposed or final rule, will be required to undertake new cost assessments both of the new proposed or final rule and at least two existing rules—although the new rule and the existing rules need not have any substantive relationship to one another and, with approval from OMB, need not even be issued by the same agency.
"""


def create_ngram(num_context, card, card_tag):
    #card = set_card()
    card_words_org = card.split()
    ngram_list = []
    for i in range(0, len(card_words_org)):
        if i > num_context:
            new_word = card_words_org[i-num_context:i+num_context]
        else:
            new_word = card_words_org[i:i+num_context] #make it so that each word takes it's prior words as context as well
        print(new_word)
        new_string = ''
        for word in new_word:
            new_string += word
            new_string += " "
        if new_string == "":
            new_string = " "
        ngram_list.append(Sentence(new_string))

    card_words = ngram_list
    return card_words, card_words_org

#print(card_words)
'''
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])
'''
# just embed a sentence using the StackedEmbedding as you would with any single embedding.




def embed(card_tag, card_as_sentence, card_words, card_words_org):
    stacked_embeddings.embed(card_tag)
    #stacked_embeddings.embed(card_as_sentence)
    print(card_as_sentence.get_embedding().reshape(1,-1))
    word_list = []
    count = 0
    token_removed_ct = 0
    card_tag_emb = card_tag.get_embedding().reshape(1,-1)
    for word in card_as_sentence:
        #word = word.reshape(1,-1)
        n_gram_word = card_words[count]
        '''
        if len(n_gram_word) == 0:
            th_tensor = torch.zeros(list(card_tag.get_embedding().size()))
            word_sim = cosine_similarity(card_tag.get_embedding().reshape(1,-1), th_tensor.reshape(1,-1))
        else:
        '''
        stacked_embeddings.embed(n_gram_word)
        word_sim = cosine_similarity(card_tag_emb, n_gram_word.get_embedding().reshape(1,-1))
        word_tup = (card_words_org[count], word_sim)
        count += 1
        word_list.append(word_tup)
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
    document.save('test_sum.docx')
    return sum_str, removed_str, token_removed_ct

def new_sum(word_list):
    val_list = summarize(word_list)
    th_9 = np.percentile(val_list, 90)
    th_75 = np.percentile(val_list, 75)
    th_50 = np.percentile(val_list, 50)
    th_25 = np.percentile(val_list, 25)
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

card, card_tag, tag_str = set_card()

head = document.add_heading(tag_str, level=1)

par = document.add_paragraph()
word_list = run_loop(10, card, card_tag)
new_sum(word_list)

