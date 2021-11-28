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

Hungarian algorithm 알고리즘 코드 구현은 파이썬에서 "from scipy.optimize import linear_sum_assignment" 로 사용할 수 있다. 자세한 사용 방법은 [여기](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html)를 보면 된다.

## Hungarian algorithm
---

Hungarian method는 다항 시간에 할당 문제를 해결하는 방법인 동시에 later primal–dual methods 예측(???)하는 조합 최적화 알고리즘이다.(예전에 배웠던것 같은데 기억이...) 이 알고리즘은 Harold Kuhn이 1955년에 개발했다. 
Hungarian method 라는 이름이 붙은 이유는 이 알고리즘이 두명의 헝가리 수학자 Dénes Kőnig와 Jenő Egerváry의 초기 연구에 기반을 두고 있기 때문이라고 한다.

## python 코드 구현 
---
![Desktop View](){: width="490" height="310" }

```python
```

## 참고
---
> [scipy Hungarian algorithm 문서 참고](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html)


