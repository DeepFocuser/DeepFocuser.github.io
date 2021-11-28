---
title: Hungarian algorithm
author: JONGGON KIM
date: 2021-11-22 20:00:00 +0900
categories: [알고리즘, 코드 조각]
tags: [Hungarian algorithm, Optimal Bipartite Matching]
math: true
mermaid: true
# pin: true
---

[DETR](https://arxiv.org/abs/2005.12872) 논문을 읽기에 앞서 Hungarian algorithm 알고리즘에 대한 이해가 필요하여 정리해본다.

Hungarian algorithm 알고리즘 코드 구현은 "from scipy.optimize import linear_sum_assignment" 즉  scipy.optimize 에 linear_sum_assignment 함수명으로 구현되어 있다.

## Hungarian algorithm
---

Hungarian method는 다항 시간에 할당 문제를 해결하는 방법인 동시에  이후의 primal–dual methods 예측하는 조합 최적화 알고리즘이다.(예전에 배웠는데, 기억이...) 이 알고리즘은 두 헝가리 수학자 Dénes Kőnig와 Jenő Egerváry의 초기 연구에 크게 기반을 두고 있기에 'Hungarian method"라는 이름을 부여한 Harold Kuhn이 1955년에 개발하고 알렸다.

## python 코드 구현 
---
![Desktop View](https://github.com/DeepFocuser/DeepFocuser.github.io/blob/gh-pages/post/BipartiteGraph/BG.JPG?raw=true){: width="490" height="310" }

```python
```
