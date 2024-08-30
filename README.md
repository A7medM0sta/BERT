# BERT
**BERT** is one the hottest model architecture which is widely used for solving different Natural Langugage Processing (NLP) tasks nowadays. So, it is very important to gain a in depth knowledge about this model. You will find several blogs, tutorials etc. on BERT. But it is very rare to find a complete package to understand the full model. In this notebook I try to merge several popular blogposts and tutorials to make a complete package.

Hope that, this notebook gives you a better undestanding on BERT. The links of those blogposts and tutorials are given in  the [References](#8) section. Check out those links for more details. 



<h2><center>Let's Start</center></h2>  

Today we are going to talk about bert.

<center><img src="https://i.gr-assets.com/images/S/compressed.photo.goodreads.com/books/1557559619l/5674402.jpg"/ width="300" height="300" ></center>


No, no, we are going to talk about Google BERT. 



<a class="anchor" id="0.1"></a>
#  Table of Contents
i.  [BERT, What is BERT !](#1)

ii.  [High Level Overview of BERT](#2)

iii.  [How BERT comes and why it becomes so popular ?](#3)

   - [Limitation of Transformer](#3.1)
   - [Transfer Learning of NLP](#3.2)
   - [Fully Replacement of LSTM](#3.3)


iv. [What is inside the BERT ?](#4)

   - [From Word to Vectors](#4.1)
   - [Encoder from the Transformer](#4.2) 
   - [Maksed Language Modeling](#4.3)
   - [Next Sentence Prediction ](#4.4)


v. [BERT as Transfer Learning in NLP](#5)


vi. [BERT for Different NLP tasks](#6)

   - [Sentence Pair Classification Task](#6.1)
   - [Single Sentence Classfication Task](#6.2)
   - [Question Answering Task](#6.3)
   - [Single Sentence Tagging Task](#6.4)


vii. [Applications](#7)


viii. [References](#8) 


# 1. BERT, What is BERT ! <a class="anchor" id="1"></a>

**BERT** (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) is awesome research in Natural Language Processing (NLP) published by researchers at Google AI Language in 2018. It has caused a stir in the Machine Learning community by presenting state-of-the-art results in a wide variety of NLP tasks, including Neural Machine Translation, Question Answering (SQuAD v1.1), Sentence Pair Classification task (MNLI), Sentiment Analysis, Text Summarization and others.
From the abbreviation of the BERT, we can figure out some kind of feature of it.

   1. It is Bi-directional 
   2. It uses an Encoder Representation 
   3. It has a Transformer based architecture 

So basically, BERT’s key technical innovation is applying the bidirectional training of Transformer, a popular attention model, to language modeling. This is in contrast to previous efforts which looked at a text sequence either from left to right or combined left-to-right and right-to-left training. The paper’s results show that a language model that is bi-directionally trained can have a deeper sense of language context and flow than single-direction language models. In the paper, the researchers detail a novel technique named Masked Language Modelling (MLM) which allows bidirectional training in models in which it was previously impossible. We will walk through all the details later in this material.


<center><img src = "data:image/jpeg;base64,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" width = 350 height = 450 /><\center>
[Back to Table of Contents](#0.1)


# 2. High Level Overview of BERT <a class="anchor" id="2"></a>

Now, let's look at a *high-level* overview of the **BERT architecture**. It is a **Transformer** based model architecture, which opens a new era to work with NLP tasks. In the last session, we learned about the Transformer. A transformer is an Encoder-Decoder model architecture that also uses positional encoding, self-attention, multi-head attention, and also with Residual connection.

$\;\;\;\;\;\;$ 

<center><img src = "https://glassboxmedicine.files.wordpress.com/2019/08/figure1modified.png?w=451&h=647" width = 400 height = 400 /> <\center>

$\;\;\;\;\;\;$ 

Now, BERT uses only the **Encoder** portion of the Transformer architecure canceling out the Decoder part.Like the Transformer BERT also uses postional encoding, self attention, multi head attention and Residual Connection. It uses exactly same architecture for encoders that is used in the Transformer. In short BERT is basically, **Stacking of Encoders** and the encoder is from the Transformer.  

$\;\;\;\;\;\;$ 

<center><img src = "http://jalammar.github.io/images/bert-encoders-input.png" width = 700 height = 400 /><\center>

$\;\;\;\;\;\;$ 

There are two types of BERT.
 
 1.$BERT$ $base$
    
 2.$BERT$ $large$


In bert_base, it is used tweleve (12) encoder stacking. On the other hand, in bert_large twenty four (24) encoder stacking is used. The types are also different from feedforward-networks (no. of hidden neurons). For the bert_base , 768 neurons are used in the feedforward network where the bert_large uses 1024 hidden units.  



<center> <img src = "http://jalammar.github.io/images/bert-base-bert-large-encoders.png" width = 600 height = 350 /> <\center>

    
[Back to Table of Contents](#0.1)