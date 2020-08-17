---
layout: post
title: Using task-spooler to queue experiments on Linux
categories: linux deep-learning tutorial
---

In this post, I'll introduce you to [task-spooler](http://viric.name/soft/ts/) (“ts” for short), a Linux program that lets you queue tasks to be executed either sequentially or in parallel, according to a user-defined number of slots.

<!-- TOC -->

- [Problem setting](#problem-setting)
- [A crude solution](#a-crude-solution)
- [The downsides](#the-downsides)
- [Enter ts](#enter-ts)
- [Installing ts](#installing-ts)
- [Closing remarks](#closing-remarks)

<!-- /TOC -->

## Problem setting

Let's say you have a set of experiments to execute on your Linux machine. Each experiment is called via the python command and a script is passed, with some accompanying command-line args. For example, experiment 1 can be called like so:

```sh
python train.py --network 1 --dataset 1
```

Now let's say there are 5 networks you want to test in 4 different datasets, for a total of 20 experiments. You also know that you can run 3 of these experiments at the same time in a single machine with little to no loss of performance. The question is: what's an efficient way to automatically execute all of these experiments?

## A crude solution

A crude solution may involve creating 3 shell scripts, or 3 shell functions, which encapsulate equal-sized batches of experiments. For example, one way to go about organizing our experiments is the following.

Script 1:

```sh
ts python train.py --network 1 --dataset 1
ts python train.py --network 1 --dataset 2
[...]
ts python train.py --network 2 --dataset 3
```

Script 2:

```sh
ts python train.py --network 2 --dataset 4
ts python train.py --network 3 --dataset 1
[...]
ts python train.py --network 4 --dataset 2
```

Script 3:

```sh
ts python train.py --network 4 --dataset 3
ts python train.py --network 1 --dataset 2
[...]
ts python train.py --network 5 --dataset 4
```

You then run each script/function in a dedicated terminal, maybe using tmux/byobu, and log out of the machine, leaving the experiments to finish.

## The downsides

One downside of this method is pretty obvious: the load has been manually balanced from the start and, if one of our manual queues finish before the others, it will not be able to get tasks that are queued in our other scripts, in case those exist. This may result in idle computation resources.

Another downside of this method is that there is no way to prioritize which experiments should be executed first. We can take an educated guess and say that the first experiment of each manual queue will finish first, then the second experiment of each manual queue and so on. But this method is unpredictable and, with time, this estimate may diverge.

The solution to these issues would be to have an easy way to queue all experiments, tell the computer how many of them can be executed in parallel and let it consume the queue of experiments automatically.

## Enter ts

`ts` works with the concept of _slots_. You tell it the maximum number of slots it has to run processes and, when queuing a process, how many slots this process requires to run. By default, `ts` works with one slot and each new process requires one slot, so a single task will be executed every time.

In our problem, it seems suitable to tell `ts` that it can execute 3 tasks simultaneously, so we go ahead and do that through the `-S` flag. Please note that, in some systems, the actual program may be called `tsp` due to a naming conflict with some other famous Linux program.

```sh
ts -S 3
```

Then we can go ahead and queue all our experiments. They will all receive a default slot requirement of 1:

```sh
ts python train.py --network 1 --dataset 1
ts python train.py --network 1 --dataset 2
ts python train.py --network 1 --dataset 3
[...]
ts python train.py --network 5 --dataset 4
```

Or, if you want to get fancy:

```sh
for i in $(seq 1 5); do
  for j in $(seq 1 4); do
    ts python train.py --network $i --dataset $j
  done
done
```

This way, `ts` will call the first three python processes and call the fourth one when the first one is finished etc.

In order to check on the list of processes and their statuses, just call `ts` with no arguments. In the example output below, I have called `ts sleep 60` a bunch of times with a number of 4 slots:

```
ID   State      Output               E-Level  Times(r/u/s)   Command [run=4/4]
12   running    /tmp/ts-out.AEnsRv                           sleep 60
13   running    /tmp/ts-out.uDzxH8                           sleep 60
14   running    /tmp/ts-out.pGUSbK                           sleep 60
15   running    /tmp/ts-out.MWfo2l                           sleep 60
16   queued     (file)                                       sleep 60
17   queued     (file)                                       sleep 60
18   queued     (file)                                       sleep 60
19   queued     (file)                                       sleep 60
20   queued     (file)                                       sleep 60
0    finished   /tmp/ts-out.UPIohv   0        60.00/0.00/0.00 sleep 60
1    finished   /tmp/ts-out.l2bBJq   0        60.00/0.00/0.00 sleep 60
6    finished   /tmp/ts-out.4SZ7O2   0        60.00/0.00/0.00 sleep 60
7    finished   /tmp/ts-out.3qkpyK   0        60.00/0.00/0.00 sleep 60
8    finished   /tmp/ts-out.ngun7M   0        60.00/0.00/0.00 sleep 60
9    finished   /tmp/ts-out.Fb5QLU   0        60.00/0.00/0.00 sleep 60
10   finished   /tmp/ts-out.lLGHom   0        60.00/0.00/0.00 sleep 60
11   finished   /tmp/ts-out.0B8fyB   0        60.00/0.00/0.00 sleep 60
```

You can see there are 4 processes running, 5 queued and 8 have finished after having been running for (unsurprisingly) 60 seconds.

## Installing ts

Now that I have got your attention, it is time I teach you how to install `ts`. In some Linux distributions, ts is available either via official or unofficial repositories. For example, in Arch-based systems, you can find it in the AUR:

```sh
yay -S task-spooler
```

In Debian:

```sh
sudo apt-get install task-spooler
```

However, it is not available in all distros, but it is quite easy to download source from [the developer's website](http://viric.name/soft/ts/) and do the good old untar/`make`/`make install` combo.

## Closing remarks

`ts` is an extremely simple Linux program that follows the Unix philosophy to the letter: it does a single job and does it well. There are many other functionalities I haven't covered here, but you can cancel queued tasks, consult the output of running and queued tasks and so on.

I understand that there may be a fancier solution to this problem, probably using enterprise-grade software that I an unfamiliar with. However, since most small-time researchers, grad students and hobbyists will really only be using the Linux command-line to ssh on a remote server or run local experiments, I believe having `ts` in the toolbelt is very beneficial.
