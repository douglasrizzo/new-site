---
layout: post
title: Sieve of Eratosthenes
categories: python programming mathematics
---

In this notebook, I implement a few versions of the [sieve of Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) for finding prime numbers.

Starting from $n=2$ and given the set $P$ of prime numbers up until $n-1$, we check if $n$ is divisible by any $p \in P$. If not, $P \leftarrow P + \{n\}$.


```python
from math import sqrt

from math import sqrt, pi
import numpy as np

cached_primes = [2]
evaluated = 2
```

The first version of the sieve works just like explained at the top.


```python
def eratosthenes(n):
    primes = []
    for i in range(2, n):
        if all([i % p != 0 for p in primes]):
            primes.append(i)

    return primes
```

The second version starts with 2 already in the set of primes and only checks the primality of odd numbers. It should be twice as fast.This second version starts with 2 already in the list of primes and only checks odd numbers. It should be twice as fast.


```python
def eratosthenes_no_evens(n):
    primes = [2]
    for i in range(3, n, 2):
        if all([i % p != 0 for p in primes]):
            primes.append(i)

    return primes
```

The last version uses all the gimmicks from the previous ones, but also caches the primes found in previous runs and reuses them in subsequent calls of the function.


```python
def eratosthenes_cached(n):
    global evaluated, cached_primes

    if evaluated < n:
        start = evaluated if evaluated % 2 != 0 else evaluated + 1
        for i in range(start, n, 2):
            if all([i % p != 0 for p in cached_primes]):
                cached_primes.append(i)

    evaluated = n

    for i in range(len(cached_primes)):
        if cached_primes[i] >= n:
            return cached_primes[:i]
    return cached_primes
```

At this point, I suspected that calls to functions such as all and any were not optimal. They could implement some kind of early stopping, but I wanted to try a different aspproach in case they didn't. These versions use a `for/else` loop to add primes to the set, instead of a call to the `all()` function.


```python
def eratosthenes_no_any(n):
    primes = []
    for i in range(2, n):
        for p in primes:
            if i % p == 0:
                break
        else:
            primes.append(i)

    return primes

def eratosthenes_no_evens_no_any(n):
    primes = [2]
    for i in range(3, n, 2):
        for p in primes:
            if i % p == 0:
                break
        else:
            primes.append(i)

    return primes

def eratosthenes_cached_no_any(n):
    global evaluated, cached_primes

    if evaluated < n:
        start = evaluated if evaluated % 2 != 0 else evaluated + 1
        for i in range(start, n, 2):
            for p in cached_primes:
                if i % p == 0:
                    break
            else:
                cached_primes.append(i)

    evaluated = n

    for i in range(len(cached_primes)):
        if cached_primes[i] >= n:
            return cached_primes[:i]
    return cached_primes
```

Check if the output of all functions is equal.


```python
a = eratosthenes(1000)
b = eratosthenes_no_evens(1000)
c = eratosthenes_cached(1000)
d = eratosthenes_no_any(1000)
e = eratosthenes_no_evens_no_any(1000)
f = eratosthenes_cached_no_any(1000)

a == b == c == d == e == f
```




    True



Here we can see that result 2 is roughly twice as fast as result 1.
Result 3 is the fastest, since the primes up until 1000 have been cached in the previous cell.
After replacing the call to `all()` (which is not as smart as I thought it was) with the `for-break` loop, results 4 and 5 are virtually equal in terms of speed.
Result 6 was not changed, since its results were already cached.


```python
%timeit eratosthenes(1000)
%timeit eratosthenes_no_evens(1000)
%timeit eratosthenes_cached(1000)
%timeit eratosthenes_no_any(1000)
%timeit eratosthenes_no_evens_no_any(1000)
%timeit eratosthenes_cached_no_any(1000)
```

    6.92 ms ± 41.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    3.6 ms ± 43.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    12 µs ± 738 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    907 µs ± 24.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    920 µs ± 63.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    12.2 µs ± 225 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


Here, I tried to evaluate if caching was indeed efficient. For some reason, it seems it isn't.


```python
%timeit -r 3 -n 10 eratosthenes(10); eratosthenes(100); eratosthenes(1000); eratosthenes(10000)
%timeit -r 3 -n 10 eratosthenes_no_evens(10); eratosthenes_no_evens(100); eratosthenes_no_evens(1000); eratosthenes_no_evens(10000)
%timeit -r 3 -n 10 eratosthenes_cached(10); eratosthenes_cached(100); eratosthenes_cached(1000); eratosthenes_cached(10000)
%timeit -r 3 -n 10 eratosthenes_no_any(10); eratosthenes_no_any(100); eratosthenes_no_any(1000); eratosthenes_no_any(10000)
%timeit -r 3 -n 10 eratosthenes_no_evens_no_any(10); eratosthenes_no_evens_no_any(100); eratosthenes_no_evens_no_any(1000); eratosthenes_no_evens_no_any(10000)
%timeit -r 3 -n 10 eratosthenes_cached_no_any(10); eratosthenes_cached_no_any(100); eratosthenes_cached_no_any(1000); eratosthenes_cached_no_any(10000)
```

    540 ms ± 10.8 ms per loop (mean ± std. dev. of 3 runs, 10 loops each)
    274 ms ± 5.47 ms per loop (mean ± std. dev. of 3 runs, 10 loops each)
    500 ms ± 24.2 ms per loop (mean ± std. dev. of 3 runs, 10 loops each)
    52.4 ms ± 121 µs per loop (mean ± std. dev. of 3 runs, 10 loops each)
    51.9 ms ± 1.1 ms per loop (mean ± std. dev. of 3 runs, 10 loops each)
    49.9 ms ± 631 µs per loop (mean ± std. dev. of 3 runs, 10 loops each)

