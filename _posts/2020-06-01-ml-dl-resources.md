---
layout: post
title: Resources to study machine learning
categories: self-study machine-learning deep-learning
---

<!-- TOC -->

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [The starting point](#the-starting-point)
- [I want more machine learning](#i-want-more-machine-learning)
- [I want to learn deep learning](#i-want-to-learn-deep-learning)
- [... we need to go deeper](#-we-need-to-go-deeper)
- [Final thoughts](#final-thoughts)
- [References](#references)

<!-- /TOC -->

## Introduction

The internet is full of online resources that can be used to learn machine learning. From papers to blog posts and YouTube videos, the amount of material is so overwhelming that sometimes, we just don't know where to begin and what paths to take.

In my years studying AI and machine learning, either by myself or as a necessity for graduate school, I have come in contact with a lot of study material. Nowadays (and before I forget everything), I feel like I am ready to point others to the steps I took when learning all the concepts I learned on the topic of ML.

In this post, I organize all the material I came in contact with in my machine learning/deep learning academic journey into what I feel like are the optimal steps someone should take them. There were definitely some books I read and courses I took too early during my academic journey and later realized I hadn't actually learned as much as I could have if I had studied the prerequisites first, so I truly believe that starting from the basics and working one's way up is the right thing to do.

This post is supposed to guide people directly to (what I believe is) the best material to learn the theory behind most machine learning techniques, both the traditional ones as well as the new buzzwordy stuff.

Just a warning. These are courses and text books that present the theory behind the area and do not go into the cutting edge research topics.

A good portion of the material mentioned in this post can be found for free on the internet. The things that involve money are certificates for the Coursera courses[^coursera-ml] [^coursera-dl] and the text books[^bishop] [^hastie] [^haykin] (if you have morals, that is). Nielsen's[^nielsen] and Goodfellow's[^goodfellow] books are freely and legally available online in the links I provide. The others are not.

## Prerequisites

The prerequisites to understand machine learning on more than the intuitive level are calculus, linear algebra, probability and statistics. If you already studied these topics in college, then it might be easier to pick them up again while studying machine learning. If not, then I'd suggest looking into specialized content for each area. I prepared another post regarding how I approached these prerequisites [here]({% post_url 2020-07-01-mathematics-self-study %}).

## The starting point

**Start with [Machine Learning by Andrew Yang on Coursera](https://www.coursera.org/learn/machine-learning)**. The course itself does not presume much familiarity with linear algebra and calculus and begins with the most basic algorithms, so it not only will give students a good ML foundation, but will also help diagnose gaps in the basics without hindering progress too much.

## I want more machine learning

If you're still interested in other traditional machine learning techniques, I believe that the books by Bishop[^bishop] and Hastie[^hastie] are the way to go. They describe techniques such as principal component analysis, support vector machines, clustering, boosting etc., as well their own takes on regression, classification and neural networks. Personally, I always felt that the explanations in these books, especially Bishop's[^bishop], are excessively convoluted, even for the simplest techniques. I only really understood some things after a few years in the field. However, if you feel inclined, they are great text books with a lot of information, specially for those interested in the mathematical details of each method.

If for some reason there is a particular method that you feel like you cannot completely grasp only by reading text books, I suggest looking for a YouTube video on the topic. Some techniques, such as clustering and PCA, can be better understood visually first. One example of a good channel is [StatQuest](https://www.youtube.com/user/joshstarmer).

If you want to test some of these algorithms or just see usage examples without having to implement them, the scikit-learn library[^sklearn] has implementations of most of them, with many Python examples and visualizations.

## I want to learn deep learning

For people specially interested in neural networks, my suggestion is to thestart by **reading Nielsen's book[^nielsen] and use Haykin's[^haykin] for consultation**. Nielsen's book feels kind of short, having only 5 chapters, book it goes right to the point and can actually be quite dense when you really want to understand everything as best as you can. Haykin's book can be taken as a stable source of information for everything related to the foundations of neural networks (everything not related to deep learning).

Next, **Goodfellow's book[^goodfellow] tied together with either the course by deeplearning.ai[^coursera-dl] or the lectures by DeepMind + UCL[^deepmind-pl]** shall familiarize students with deep learning as a whole. At this point, Goodfellow's book may have some content that was already seem in previous material, so it should be okay to skip directly to [part 2 of the book](https://www.deeplearningbook.org/contents/part_practical.html). This material should cover deep/convolutional/recurrent neural networks, as well as more exotic topics such as optimization techniques, attention mechanisms and generative models.

## ... we need to go deeper

![I'd like to know more](../images/know-more-meme.jpg)

If, after going through everything I mentioned here, you'd still like to know more about some topic in particular, I suggest looking into the references used in the books for that particular topic. The books themselves usually mention the authors of each technique by name and reference the seminal paper of each method. As a rebel student, I dare say that seminal papers, especially the older ones, usually do a bad job at teaching the method they are presenting. There's always some new material that uses more contemporary language, better graphics or more understandable math. But I also believe that it is important to have contact with the original material. Never limit yourself! [^1]

## Final thoughts

In this article, I basically described the steps I took to learn machine learning and the material that helped me. I came in contact with some of this material in graduate school, but after so many years without taking formal classes, I had to learn how to fend for myself whenever I needed to learn something new or go deeper into something I felt I didn't know enough about.

I admit that this post is a little dry, but I think it's best to upload it as it is and review it as time goes by instead of sitting on it and wait until it is a polished syllabus. I just hope it may be of some help to whoever finds it, in any state it is found.

## References

[^bishop]: Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer-Verlag New York Inc. https://doi.org/10.1198/tech.2007.s518
[^hastie]: Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer New York. https://doi.org/10.1007/978-0-387-84858-7
[^haykin]: Haykin, S. S., & Haykin, S. S. (2009). Neural Networks and Learning Machines (3rd ed.). Prentice Hall.
[^goodfellow]: Goodfellow, I., Bengio, Y., & Courville, A. (2017). Deep Learning. The MIT Press. https://www.deeplearningbook.org/
[^nielsen]: Nielsen, M. A. (2015). Neural Networks and Deep Learning. Determination Press. http://neuralnetworksanddeeplearning.com/
[^coursera-dl]: [Deep Learning Specialization by deeplearning.ai](https://www.coursera.org/specializations/deep-learning)
[^deepmind-pl]: [DeepMind x UCL \| Deep Learning Lecture Series 2020](https://www.youtube.com/playlist?list=PLqYmG7hTraZCDxZ44o4p3N5Anz3lLRVZF)
[^sklearn]: https://scikit-learn.org/stable/
[^my-algos]: [My own machine learning algorithm implementations in C++](http://douglasrizzo.com.br/machine_learning/)
[^1]: I also believe that newer articles suffer less from this.
