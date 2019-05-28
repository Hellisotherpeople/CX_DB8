import tensorflow as tf
import tensorflow_hub as hub
import os
import operator
import math
from sklearn.metrics.pairwise import cosine_similarity
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

embed = hub.Module(module_url)

card_tag = ["Jerry Ehman wrote the Wow Siginal"]


card = """
If the evidence gets out to the public while the scientists are still analyzing the signal, Forgan said they could manage the public's expectations by using something called the Rio Scale. It's essentially a numeric value that represents the degree of likelihood that an alien contact is "real." (Forgan added that the Rio Scale is also undergoing an update, and more should be coming out about it in May.)

If the aliens did arrive here, "first contact" protocols likely would be useless, because if they're smart enough to show up physically, they could probably do anything else they like, according to Shostak. "Personally, I would leave town," Shostak quipped. "I would get a rocket and get out of the way. I have no idea what they are here for."

But there's little need to worry. An "Independence Day" scenario of aliens blowing up important national buildings such as the White House is extremely unlikely, Forgan said, because interstellar travel is difficult. (This feeds into something called the Drake Equation, which considers where the aliens could be and helps show why we haven't heard anything from them yet.) [The Father of SETI: Q&A with Astronomer Frank Drake]

Early SETI work
To find a signal, first we have to be listening for it. SETI "listening" is going on all over the world, and in fact, this has been happening for many decades. The first modern SETI experiment took place in 1960. Under Project Ozma, Cornell University astronomer Frank Drake pointed a radio telescope (located at Green Bank, West Virginia) at two stars called Tau Ceti and Epsilon Eridani. He scanned at a frequency astronomers nickname "the water hole," which is close to the frequency of light that's given off by hydrogen and hydroxyl (one hydrogen atom bonded to one oxygen atom). [13 Ways to Find Intelligent Aliens]

In 1977, The Ohio State University SETI's program made international headlines after a project volunteer, Jerry Ehman, wrote, "Wow!" beside a strong signal a telescope there received. The Aug. 15, 1977, "Wow" signal was never repeated, however.
"""

card_words_org = card.split()

ngram_length = 10 # the 'n' in the ngrams


ngram_list = []
# generate ngrams
for i in range(0, len(card_words_org)):
    new_word = card_words_org[i - ngram_length:i + ngram_length] #make it so that each word takes it's prior words as context as well
    print(new_word)
    new_string = ''
    for word in new_word:
        new_string += word
        new_string += " "
    ngram_list.append(new_string)

card_words = ngram_list


with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    card_tag_embeddings = session.run(embed(card_tag)) # card_tag for user specified
    card_words_embeddings = session.run(embed(card_words)) # (card_words represents the array of card ngrams)
    #print(cosine_similarity(card_tag_embeddings, card_words_embeddings[0].reshape(1, -1)))
    word_list = []
    count = 0
    token_removed_ct = 0
    for word in card_words_embeddings:
        word = word.reshape(1,-1)
        # determine the similarity between the card tag and the current ngram
        word_sim = cosine_similarity(card_tag_embeddings, word)
        word_tup = (card_words_org[count], word_sim)
        count += 1
        word_list.append(word_tup)
    summary_string = ""
    removed_str = ""
    for sum_word in word_list:
        if float(sum_word[1]) > 0.25:
            summary_string += str(sum_word[0])
            summary_string += " "
        else:
            token_removed_ct += 1
            removed_str += str(sum_word[0])
            removed_str += " "
    print("CARD: ")
    print(card)
    print("GENERATED SUMMARY: ")
    print(summary_string)
    print("tokens removed:" + " " + str(token_removed_ct))
    print("NOT UNDERLINED")
    print(removed_str)
