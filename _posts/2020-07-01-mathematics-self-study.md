---
layout: post
title: Resources to self-study mathematics for machine learning
categories: mathematics self-study machine-learning
---

> Two plus two is four, minus one, that's three. Quick maths.
>
> [Michael Dapaah](https://youtu.be/3M_5oYU-IsU?t=63)

<!-- TOC -->

- [Introduction](#introduction)
- [Study areas](#study-areas)
  - [Algebra and pre-calculus](#algebra-and-pre-calculus)
  - [Calculus](#calculus)
  - [Linear Algebra](#linear-algebra)
- [Software and apps](#software-and-apps)
- [Online communities](#online-communities)
- [My failures](#my-failures)
- [Final thoughts](#final-thoughts)

<!-- /TOC -->

## Introduction

When I started my PhD, I didn't know at first if my research would take me in a path whose foundations I was comfortable with, such as algorithms, logic and other discrete areas, or if fate would take me towards the more mathematical path. As I started to get really interested in the applications of neural networks for reinforcement learning, I realized little by little that I would not only need to remember all the mathematics I had learned before in my life, but also learn and get comfortable with some new stuff.

In this article, I go over:

- what topics I studied;
- what material worked for me;
- in which order I studied them;
- where to find them online (the free ones, at least);
- which software, online services and apps I used;
- where to find online communities to post questions and answers;
- where I failed in my studies;
- some final tips.

## Study areas

### Algebra and pre-calculus

I recommend the [free algebra book by James Brennan](https://jamesbrennan.org/algebra/) and the [free precalculus book by Stitz and Zaeger](https://www.stitz-zeager.com/), which I found out about in the very helpful [/r/learnmath](https://reddit.com/r/learnmath) subreddit. My recommendation would be to evaluate which topics you remember and just skim over them, while giving yourself more time over the stuff you don't remember or have never learned.

Personally, I made a lot of progress by just following both books until I felt I was comfortable with my knowledge. I tried not to skip anything, but I also felt like I could work fast as I had already learned most of this stuff before.

### Calculus

To study calculus, I used the [books by Stewart](https://stewartcalculus.com/media/16_home.php), 8th edition. I knew I was prepared to start calculus when I had studied enough algebra to fulfill the prerequisites at the start of the book. I also worked through the two review lists available in Stewart's website [[1]](https://stewartcalculus.com/data/CALCULUS_8E/upfiles/6e_reviewofalgebra.pdf) [[2]](https://stewartcalculus.com/data/CALCULUS_8E/upfiles/6e_reviewofanalgeom.pdf). Those lists helped me to prioritize what I needed to focus on my aforementioned precalculus quest.

### Linear Algebra

I decided to start studying linear algebra alongside calculus. I had an easier time with linear algebra as I already had some familiarity with matrix operations, not only because I was already working with neural networks for quite some time, but I also implemented [my own matrix class in C++](douglasrizzo.github.io/matrix) at one point.

I used a 1986 [Brazilian undergrad text book](http://mtm.ufsc.br/~muniz/mtm5512/refs/Algebra_Linear_Boldrini.pdf), but also searched many concepts on YouTube, since I did not get the intuition behind some of them. After I had started using the old Brazilian book, I got acquainted with [Larson's book](https://www.amazon.com/gp/product/B019EB9S4O/), which is a lot newer. From a quick skim of the book, I'd definitely recommend it instead of the one I used.

## Software and apps

Two apps that really helped me, especially with calculus, were:

- Desmos: a very famous graphical calculator, which is freely available [online](https://www.desmos.com/calculator) or as an app;
- [Wolfram Alpha](https://www.wolframalpha.com/): which math student doesn't know this one? It simplifies, solves for x, finds derivatives, minimum and maximum values of functions. You name it, Wolfram Alpha does it. Just be careful that, sometimes, its answers are either more convoluted than they need to be, or simpler than you'd want them to be. Instead of paying for a subscription, I recommend buying the smartphone app, which is a one-time purchase that gives you the coveted "step-by-step solution" functionality.

As for calculator apps, I recommend looking into [SpeedCrunch](http://speedcrunch.org/) for [quick maths](https://youtu.be/3M_5oYU-IsU?t=63) or [Qalculate!](https://qalculate.github.io/) for more symbolic stuff (with fractions, units of measurement etc).

## Online communities

It came a point in which I started having the kind of questions that I couldn't quite google my way to the answer. When I started to look for a place to post my questions, I found the extremely active and welcoming [/r/learnmath](https://reddit.com/r/learnmath) subreddit. The people there are really nice and welcoming of noobs and they are a lot less strict than other sites, such as the Math Stack Exchange.

Another place I suggest people to look are Discord servers, where people gather to talk about homework. An example is [Homework Help](https://discord.gg/USVc7XX).

Overall, my advice with regard to communities is to both **seek and provide help**. Seeing people ask questions to which I knew the answer to made me realize how much I had actually learned in my journey, which really motivated me. Also, I could help a random stranger online, which is nice.

## My failures

Humble time. Since I didn't know where to start, I decided to enroll in online university courses and let the lectures guide me. I had had a great experience with Coursera and the machine learning course by Andrew Ng, so I thought I'd have an equally positive experience with Coursera's math courses[^calc-courses].

However, unlike the previous courses I took, I couldn't watch the lectures or complete the assignments before or after the correct dates, which really frustrated me.

Another problem I faced was that I just couldn't follow the professors in some of the lectures. Sometimes, the exercises after a lesson seemed to have nothing to do with the lesson itself. Since I had no other material to follow, I felt stuck in some lessons and couldn't meet the deadlines.

## Final thoughts

- **Don't force it:** Studying mathematics ended up becoming a hobby of mine, one that I could spend an entire day doing, if I felt motivated. There were days that I would eat up over 20 pages of a book, a pretty decent number if you ask me, since I'd be able to end a 400 page book in 20 days (not that I did that). My main problem was, and still is, consistency. I can't really do the whole "study everyday for 30 minutes" thing. I usually studied for the whole day, twice a week.

  I'll admit I did not study any of the books from cover to cover and I am still working through them, but I never felt hindered or delayed by the method I chose to study mathematics. My only limitations are time, since I have other responsibilities, and motivation.

- **Find the format that is best for you:** since I did very well in online AI courses [[1]](https://www.coursera.org/verify/BJWU9J3F9J4X) [[2]](https://www.coursera.org/verify/5EET54PYUU9T), I thought the same would be true for online mathematics courses. However, when it came to mathematics, I felt like having complete free reign of what I needed to study instead of being guided worked much better. Text books were great for that, as I could choose which topics to study, hop from book to book and just take my time overall in what I realized would be a long-term project.

---

[^calc-courses]: I enrolled in the UC Irvine precalculus course (which is not available on Coursera anymore, but can be found [here](https://www.math.uci.edu/node/22608)) and later, in the University of Ohio calculus course (which also is not available anymore, but a few pointers to where the material is currently available are [[link]](https://mooculus.osu.edu/lectures) [[link]](https://www.youtube.com/user/kisonecat/playlists) [[link]](https://www.coursera.org/instructor/jimfowler)).
