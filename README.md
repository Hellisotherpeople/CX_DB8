# CX_DB8
A contextual, token extracting summarizer designed from the ground up by a former debater for debaters (but it'll also work on your reading) 
oh, and also, 
Parsing code for a novel dataset designed for usage in the field of NLP &

## UPDATE 4/9/2019

** Install Instructions ** 

1. Procure a Laptop with modern Linux install (say Ubuntu, but I developed this with Manjaro) either on baremetal or in a VM (Virtualbox is what I used ). Linux is free and has the same backend architecture (unix) as MacOS so if you're used to MacOS Linux isn't totally alien. 

2. Confirm that you have Python 3.6+ (Should come preinstalled in a modern Linux install). Open a terminal and Type "python" or "python3" (if python2 comes up), and take note of which version of Python 3 you have. 

3. Cofnrim that you have "pip" (The python package manager). usually a simple "pip" or "pip3" will confirm if you have it or not. If this is not present, it can be installed via apt-get (for ubuntu users) 

5. Install [PyTorch](https://pytorch.org/). The install instructions are quite easy - you'll want to install the "pip" package, correponding to your Python version. Advanced users who want to take advantage of their GPU and who have installed CUDA and CUDNN can select their corresponding CUDA version - but most users will want to select "None". Finally, PyTorch will generate a one line command. 

A user with Python 3.7 and not using cuda would run this command: 
```
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl
pip3 install torchvision
```
5. move to your favorite empty directory, and clone flair into that directory 
```
git clone https://github.com/zalandoresearch/flair.git
cd flair-master
pip3 install requirments.txt
```
6. finally, in the same directory as flair, download (from this repo) cx_db8_flair.py and cx_requirments.txt
```
pip3 install cx_requirments.txt
```
It's installed! You should be ready to cx_db8 this from the command line with a 
```
python3 -i cx_db8_flair.py
```

It will prompt you with a screen asking you to continue. Press "y" or it will end execution, and save any summaries found to "test_sum.docx". It will then prompt you to add card text into Ctrl-D is pressed. Then it will ask for a card tag, or allow the user to give a "generic" summary by entering -1. It will then prompty for a card author and date, and then the citation. After all of this are entered, the summary is generated and the prompt will ask you to either continue execution with a "y" or to end exeuction. 

Depending on the paramaters chosen, summarization may take a long time (more accurate) or be nearly instant. The default settings are a balance

Please submit issues for help with installation or running cx_db8. 


## UPDATE 3/30/2019

Hello all. I am actively working on this project. For now, I'm working on 3 things 

1. The speed of inference. I assume most debaters will not have access to a GPU or the inclination to get PyTorch to play nice with their own GPU. This means that by default, my summarizer will use embeddings that are fast, but maybe slightly less effective at understanding semantic meaning (like, 1-2 % worse). The default settings that this code is uploaded with should allow for extremely fast (~1 sec) summarization of a large piece of evidence. If you want the extra 1-2 %, you can experiment with big contextual models like BERT and ELMO, rather than use the default FastText + ExtVec vectors. Refer to the Flair documentation on the various different embeddings supported to see how to change them 

2. The word document creation - I'm imagining a program that runs in a loop, asking for the user to give it cards (or go through a word doc or folder), so that it can summarize N documents per run instead of one at a time. I'd also like it to support things like Verbatim formats (I want it to highlight, AND emphasize) instead of my crappy underline or bold underline method avalible at present. 

3. Exposing paramaters to the user: It's super easy to change what proportion of the document is underlined or highlighted, and what sets of vector embeddings are used. Right now, it defaults to underlining 75% of the document and highlighting 25% of the document - yet sometimes a team needs different proportions of each. 

3. Documentation - I'm bad at this and it's a low priority for me while I try to get the actual tool working. 

4. Windows support - Honestly I'm going to need some evidence that people are actually trying to use this before I focus on it. For now main functionality comes first. 


**Card tag given: "Humanity is an abstraction".**

![](https://github.com/Hellisotherpeople/CX_DB8/blob/master/Summarizer.JPG)


**And now, it also generates word docs!!**

![](https://github.com/Hellisotherpeople/CX_DB8/blob/master/Summarizer2.JPG)


I finally got fed up with trying to experiment with supervised summarization systems. Thanks to the magical folks at Google for creating the unsupervised ["Universal Sentence Encoder"](https://tfhub.dev/google/universal-sentence-encoder/2) which is more rightly called the "Universal Text Encoder" given how smoothly it works on Words, Sentences, or even whole Documents. 

Also, high school debators are really quite bad at properly summarizing debate documents. Garbage in - Garbage out when it came to Supervised Learning methods. 

So, I enlist the help of the wikipedia pre-trained Universal Sentence Encoder to create the (to my knowledge) **worlds first contextual token-level extractive summarizer.**


It works by first computing the "meaning" of the card tag, which can be anything. Then, it goes through every word in the actual card. Now, here, I could just compute the meaning of the individual word like "The" = 0.11 and "Person" = 0.12, but that doesn't include the context of the other words around it. 

So, instead I take N words before and after the string, and use those to compute the meaning of the N gram generated for each word. The first word only includes N words after the first word, and the last word only includes N words before the first word. This allows the CX_DB8 system to properly compute the meaning of each word in the text **in the context of the N words before and after it**. I then compute a similarity score between each word with context and the card tag, and decide to include it in the summarized text if the similarity score is over a user defined threshhold. If the user gives a 1, then the summarizer only includes the exact same string, if the user gives a 0 it includes everything. I've experimented and found a score of about 0.25 - 0.4 to be very effective

I am strapped for time but will provide a good setup and install script and instructions on how to install and use this. If you know how to code, you should be able to easily get this working on your own system. **it doesn't even require a GPU, since the Universal Sentence Encoder model is CPU friendly** (but you might need Mac or Linux - have not tested Windows support yet and that's a big deal since no one runs word on a Linux box) 



## What is this? 

So - I'm really passionate about someone using the [various](https://github.com/google-research/bert) [different](https://blog.openai.com/better-language-models/) [new](https://github.com/zalandoresearch/flair) [cool](http://jalammar.github.io/illustrated-transformer/) [innovations](https://github.com/abisee/pointer-generator) that currently exist in NLP to solve a surprisngly well defined problems that (should) have been solved by now: Token-Level Extractive Summarization.



## Why token level extractive summarization? 

In American competative cross-examination debate (also known as Policy Debate) - a unique form of evidence summarization is utilized - Summarization by highlighting - which this "card" will illustrate: 

![alt_tab](https://github.com/Hellisotherpeople/CX_DB8/blob/master/evidence_example.JPG)



In this example, everything after the citation is the verbatim supreme court dissenting opinion. Competative debators summarize this document by simply underlining the important parts of the document. They read outloud the highlighed portions of the document. 

There are a multitude of NLP tasks that we could apply to a dataset of thousands of pieces of debate evidence. I am most interested in the task of creating an automatic evidence underlining tool. 

## So.... where is that tool? 

I'm publishing the dataset parsing and creation tools now to prove that I am (to my knowledge) the first one to write a parsing script capable of converting competative debate evidence into CoNNEL 2003 / Seq2Seq friendly data types. In theory, any Seq2Seq or PoS tagging model that accepts one of those formats can utilize this dataset. I will write a quick tutorial on how to gather these dataset files yourself: 

*Step 1: Download all open evidence files from 2013-2017 ( https://openev.debatecoaches.org/ ) and unzip them into a directory

*Step 2: Convert all evidence from docx files to html5 files using pandoc with this command: 
```
for f in *.docx; do pandoc "$f" -s -o "${f%.docx}.html5"; done
```
*Step 3: run one of my .py files with python3 to have it parse all the html5 files and it will dump out a file titled "card_data.txt" in the output format specified by the .py file

I will document how to change the file locations at a future time. 



## What are your plans? 
So, it looks like the prefered framework for doing state of the art NLP work [Flair](https://github.com/zalandoresearch/flair/issues/563#issuecomment-470010988) is finally being updated to allow it to train with large datasets. This is going to allow me to create a sentence compression (highlighter) model as soon as the patch allowing large datasets makes it into Flair. I will be training this model using the latest in sentence and word pre-trained embeddings (like google BERT) to give the sentence compression model far more semantic understanding. 

After the sentence compression model is developed, I have 2 other models I want to create for the good of the debate community. 

1. A document classifier. (also possible using Flair). I will soon write some parsing code to extract the class of a debate card. For instance, an answer to the Capitalism Kritik by saying that there would be tranisition wars would be classified as "A2 Capitalism - Transition Wars". Given the way that debate documents are hierachially structured using verbatims "hats and pockets" features - it should be possible to automatically extract a reasonable class for each document. With some cleaning, I hope to get a document classifier that can classify an arbitrary piece of evidence into 1 of N (I assume N is around 200) buckets. 

2. A card tag generator - basically just a abstractive summarizer model. I'd explore Pointer-Generator networks for this, as they are mature and allow for abstractive summaries which utilize lots of words from the source text. 

Combing the 3 models, a document highlighter, a document classifier, and a card tag generator with a system to automate taking in new cards (RSS feed or something involving Kafka / REST Apis) - a fully end to end researching system can be created - which will automate away all tasks of debate. 

And when I am done, debate will no longer favor large schools or affluant kids with more free-time than their poorer peers. Imagine it - entire debate cases created nearly automatically
