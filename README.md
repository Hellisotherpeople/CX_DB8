# CX_DB8
Parsing code for a novel dataset designed for usage in the field of NLP &amp; a call to action 


## What is this? 

So - I'm really passionate about someone using the [various](https://github.com/google-research/bert) [different](https://blog.openai.com/better-language-models/) [new](https://github.com/zalandoresearch/flair) [cool](http://jalammar.github.io/illustrated-transformer/) [innovations](https://github.com/abisee/pointer-generator) that currently exist in NLP to solve a surprisngly well defined problems that (should) have been solved by now: Toke-Level Extractive Summarization.


Unfortunately, that does not seem to be the case :( 

## Why token level extractive summarization? 

In American competative cross-examination debate (also known as Policy Debate) - a unique form of evidence summarization is utilized - Summarization by highlighting - which this "card" will illustrate: 

![alt_tab](https://github.com/Hellisotherpeople/CX_DB8/blob/master/evidence_example.JPG)



In this example, everything after the citation is the verbatim supreme court dissenting opinion. Competative debators summarize this document by simply underlining the important parts of the document. They read outloud the highlighed portions of the document. 

There are a multitude of NLP tasks that we could apply to a dataset of thousands of pieces of debate evidence. I am most interested in the task of creating an automatic evidence underlining tool. 

## So.... where is that tool? 

That's been the issue! Everything I've found in the relm of sequence to sequence tools do things just a little bit differently than what I need to solve this task. This readme will be updated with specific repos and papers that I've explored in this task - annotated with my description of the reason that I couldn't get the tool working 

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
