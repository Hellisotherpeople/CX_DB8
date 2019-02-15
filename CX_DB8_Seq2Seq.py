import glob
import os
from bs4 import BeautifulSoup
import bs4
import string


num_of_cards = 0
os.chdir("/home/lain/manjaro-windows-shared/debate_cards-2014-2017")

with open('card_data.txt', 'a') as f:
    for file in glob.glob("*.html5"):
        print("File parsed: " + file)
        print(num_of_cards)
        with open(file) as fp:
            soup = BeautifulSoup(fp, "lxml")
            all_card_tags = soup.find_all('h4')
            data = []

            for h4 in all_card_tags:
                counter = 0
                for content in h4.next_siblings:
                    if content.name == "h4":
                        break
                    elif content.name == "p" and counter == 0:
                        # this is likely the tag
                        counter += 1
                    elif content.name == "h1":
                        pass
                    elif content.name == "h2":
                        pass
                    elif content.name == "h3":
                        pass
                    elif content.name == "p":
                        # print(content.text)
                        underlined = content.find_all(
                            'span', class_="underline")
                        real_underlined = ''
                        for a_word in underlined:
                            # use NLTK for tokenization instead ALSO, TODO: Directly write the strings to the end file
                            split = a_word.text.split()
                            for part in split:
                                real_underlined += part
                                real_underlined += ' '
                        # print('--------------------------------------------')
                        # print(real_underlined)
                        if real_underlined:
                            num_of_cards += 1
                            f.write(content.text + '\t' + real_underlined + '\n')

                    '''
                        if isinstance(stuff, bs4.element.Tag):
                            # This stuff is underlined
                            split_segment = stuff.text.split()
                            for word in split_segment:
                                # easy heuerestic to remove weblinks, a word should be less than 20 characters
                                if word and len(word) < 20:
                                    word = ''.join(
                                        ch for ch in word if ch.isalnum())
                                    if word.isspace() or not word:
                                        segment = '\n'
                                    else:
                                        segment = {'data': word.lower(), 'tag': "und"}
                                    data.append(segment)
                                else:
                                    data.append('\n')
                        else:
                            split_segment = stuff.split()
                            for word in split_segment:
                                if word and len(word) < 20:
                                    word = ''.join(
                                        ch for ch in word if ch.isalnum())
                                    if word.isspace() or not word:
                                        segment = '\n'
                                    else:
                                        segment = {'data': word.lower(), 'tag': "non"}
                                    data.append(segment)
                                else:
                                    data.append('\n')
                    # print(underlined)
                    # for the_text in underlined:
                    #    print(the_text)
                        '''
                else:
                    pass
'''
    with open('card_data.txt', 'a') as f:
        for item in data:
            if item is '\n':
                f.write('\n')
            else:
                f.write(str(item['data']) + '\t' + str(item['tag']) + '\n')
'''
print("Number of cards processed: " + str(num_of_cards))
'''
        card = {'tag': h4.text, 'cite': h4.find_next(
            "strong"), 'pocket': h4.find_previous("h1").text, 'hat': h4.find_previous("h2").text, 'block': h4.find_previous("h3").text}
    '''

# assert that all fields are filled in for each card
# from one h4 to another
# if strong and no underline, get rid of it
# if strong and underline, keep it
# get rid of stuff between <a> tags
# get all <p>

'''
    all_blocks = soup.find_all('span', class_="underline")
    for h3 in all_blocks:
        print(h3)
   '''
