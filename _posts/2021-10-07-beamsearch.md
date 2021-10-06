---
title: Beam Search Decoder
author: JONGGON KIM
date: 2021-10-07 00:00:00 +0900
categories: [AI, 자연어 처리, 논문리뷰]
tags: [Beam Search]
math: true
mermaid: true
# pin: true
---

> <https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/> 를 참고

## Beam Search Decoder
---

![Desktop View](https://d2l.ai/_images/beam-search.svg){: width="1000" height="600" }

Beam Search는 Greedy Search 알고리즘(k=1)을 확장한 것이며, output sequences 의 리스트를 반환한다. beam search는 가능한 모든 단계(경로 를 탐색하고, k개를 유지한다. k는 사용자가 설정하는 hyper-parameter이며 beam의 수 또는 sequence 확률을 통한 병렬 탐색을 제어한다. 기계번역 작업에서는 보통 k=5 or k=10을 사용 한다. k가 크면 여러 candidate sequence들이 target sequence와 매칭될 가능성이 더 높아지기 때문에 좋은 성능을 보이나, 이는 결과적으로는 decoding 속도를 감소시킨다.(성능 vs 속도 trade-off 관계)

## Beam Search Algorithm
---
주어진 확률 sequence 와 beam width parameter k에 대해 beam search를 수행하는 함수를 정의할 수 있다.

1. 각 단계에서, 각 candidate sequence들은 가능한 다음 경로들로 확장된다.
2. 각 candidate step 은 확률들을 곱하여 점수를 계산한다.
3. 확률이 높은 k개의 후보만 선택되고 나머지는 제거된다.
4. 이 과정은 sequence의 끝에 다다를때까지 반복된다.
    - search process는 아래의 경우에 멈춘다.
        - end-of-sequence token에 도달할 때
        - a maximum length에 도달할 때
        - threshold likelihood에 도달할 때

```python
# beam search
import numpy as np

def beam_search_decoder(data, k):

	sequences_index = [[list(), 0.0]]

	for row in data:
		all_candidates = list()

		# 가능한 다음 경로로 확장하기
		for i in range(len(sequences_index)):
			seq, score = sequences_index[i]

			for j in range(len(row)):
				'''
					np.log에 -를 붙여서 최소화 문제로 바꿈.
					가장 이상적인 경우는 score = 0
					가장 안좋은 경우 score = 무한대
				'''
				candidate = [seq + [j], score - np.log(row[j]+1e-7)]
				all_candidates.append(candidate)

		# score에 따라 오름차순 정렬
		ordered = sorted(all_candidates, key=lambda tup:tup[1])

		# best k개 뽑기
		sequences_index = ordered[:k]
	return sequences_index

# 5개 단어의 어휘에 대해 10개 단어의 시퀀스를 정의
probability_sequence = [[0.1, 0.2, 0.3, 0.4, 0.5],
						[0.5, 0.4, 0.3, 0.2, 0.1],
						[0.1, 0.2, 0.3, 0.4, 0.5],
						[0.5, 0.4, 0.3, 0.2, 0.1],
						[0.1, 0.2, 0.3, 0.4, 0.5],
						[0.5, 0.4, 0.3, 0.2, 0.1],
						[0.1, 0.2, 0.3, 0.4, 0.5],
						[0.5, 0.4, 0.3, 0.2, 0.1],
						[0.1, 0.2, 0.3, 0.4, 0.5],
						[0.5, 0.4, 0.3, 0.2, 0.1]]

result = np.array(probability_sequence)
result = beam_search_decoder(result, 3)

# 결과 출력
for sequence_index in result:
	print(sequence_index)

'''
결과
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 0], 6.9314698055996535]
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 1], 7.154613306913874]
[[4, 0, 4, 0, 4, 0, 4, 0, 3, 0], 7.154613306913874]
'''
```

이제 출력으로 나온 sequence_index 를 단어 사전을 이용해 단어로 바꿔주면 decoding이 완성된다.

