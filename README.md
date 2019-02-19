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

This readme is unfinished but will be updated in the coming days
