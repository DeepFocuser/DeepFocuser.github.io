---
title: Attention Is ALL You Need
author: JONGGON KIM
date: 2021-10-06 20:00:00 +0900
categories: [AI, 자연어 처리]
tags: [Transformer]
math: true
mermaid: true
---
> [Attention Is ALL You Need 논문 바로가기](https://arxiv.org/abs/1706.03762)

대학원에서 공부할 당시 자연어 처리 맛만 보고, 회사 다니고 나서는 자연어 처리쪽 업무 경험은 전무하다. 앞으로도 특별한 일이 없는 한 딱히 있을 것 같진 않지만... transformer가 자연어처리 외에 다른 분야에도 사용되는 경우도 많다고 하니( ex) DETR ), 리뷰를 해보자. 본 논문은 transformer Model 구조 파악이 목적이다. 본인은 컴퓨터 비전 관련한 일을 하기 때문에...

자세히 리뷰하든, 간단히 리뷰하든 나중에 결국은 잊어먹더라... 최대한 핵심만 남겨놓자. 

## Abstract
---
Rnn, Convolution Layer을 사용하지 않은 Encoder, Decoder 구조 즉, transformer를 제안한다. 
영어-독어 번역, 영어-프랑스어 번역에서 가장 좋은 결과를 냈고, 다른 작업(English Constituency parsing)에서도 성공적으로 적용이 가능했다고 한다. 

병렬화 학습 가능하고, 학습 시간 적게 걸리고, BLUE 지표에서 엄청 좋았다는등 자랑도 한다.

## Introduction
---
Recurrent neural network 기반의 모델이 최신을 이끌어왔고, 적용범위를 넓히기 위한 노력들을 계속해왔다.

Recurrent model은 시간을 포함하는 특성때문에 병렬화를 어렵게 한다. 최근에는 factorization tricks 와 conditional computation 같은 기법으로 계산 효율성을 증가시켰다고 한다. conditional computation는 모델성능까지 증가시켰다고 한다. 그러나 여전히 시간을 계산한다는 근본적인 제약이 남아 있다.

Attention 메커니즘은 다양한 작업에서 강력한 시퀀스 모델링이나 변환 모델에서 필수적인 부분이고, 입력 또는 출력의 시퀀스의 거리에 관계 없이 종속성 모델링이 가능하게 한다. 모든 경우는 아니지만, 몇몇 경우에는 Attention 메커니즘이 Recurrent network와 함께 사용된다고 한다.(seq2seq with attention 같은 것을 말하는 듯 하다.)

## Background
---
ByteNet 나 ConvS2S는 두 임의의 입력 또는 출력 위치의 신호를 연관시키는데 필요한 operation의 수가 위치간의  거리에 따라 증가한다. 이러한 점은 먼 위치사이 에서의 의존도를 학습하기 어렵게 만든다고 한다. 그런데 transformer는 operation의 수가 상수라고 한다. (Muiti-Head Attention 때문이라고 한다.)
 
 > 정확히 어떤의미인지는 와닿지 않으나, transformer가  입,출력간의 의존도를 학습하는데 있어서 ByteNet, ConvS2S와 같은 모델들보다 계산이 적게 들어간다는 의미인듯 하다.

Self-attention은 시퀀스의 표현을 계산하기 위해 단일 시퀀스의 다른 위치들을 연관시키는 Attention 메커니즘이다.

End-to-End memory 네트워크는 recurrent 어텐션 메커니즘에 기반한다.

Transformer는 오로지 Self-attention에 의존한 첫번째 변환 모델이다. 

## Model Architecture
---
![Desktop View](https://github.com/DeepFocuser/DeepFocuser.github.io/blob/gh-pages/post/transformer/transformer.PNG?raw=true){: width="1000" height="600" }

- Transformer 구조 
  - stacked self-attention
  - point-wise
  - fully connected layers

- ### Encoder and Decoder Stacks

  - #### Encoder
    - encoder는 6개의 동일한 층을 쌓은 구조
    - 각 층은 2개의 sub-layer를 가지고 있음.
      - 첫번째 층 : multi-head self-attention mechanism

      - 두번째 층 : position-wise fully connected feed-forward network
    - residual connection 사용 / layer normalization 사용
    - 각 sub-layer의 output : LayerNorm(x+Sublayer(x))
    - residual connection을 사용하기 위해 모든 sub-layer, embedding layers는 `512 차원의 output` 을 생성
  - #### Decoder
    - decoder도 encoder와 같이 6개의 동일한 층을 쌓은 구조 
    - encoder의 sub-layer 2개에 + 1개의 sub-layer를 더 넣음(multi-head attention)
      - encoder stack의 출력에 대해  multi-head-attention을 수행한다. 
    - residual connection 사용 / layer normalization 사용
    - `modified self-attention sub-layer`
      - 현재 위치가 다음 위치에 주목하는 것을 방지 하기 위한 장치
      - 이 masking은 `output embeddings들이 한 위치씩 offset되어 있다는 사실과 결합되어` i위치에 대한 예측이 반드시 위치 i보다 작은 위치에서 알려진 출력에만 의존할 수 있도록 한다.
          - `무슨 말인지 와닿지 않는다.(1)` - 해결
            - Transformer는 문장 행렬로 입력을 한꺼번에 받으므로 현재 시점의 단어를 예측하고자 할 때, 입력 문장 행렬로부터 미래 시점의 단어 까지도 참고할 수 있는 현상이 발생한다. 이 문제를 해결하기 위해 Transformer의 decoder는 현재 시점의 예측에서 현재 시점보다 미래에 있는 단어를 참고하지 못하도록 마스크를 씌워준다는 얘기이다. 자세한 설명은 [여기](https://wikidocs.net/31379)를 보면 될 것 같다. 
            - [코드와 설명 주석이 있는 링크](https://github.com/DeepFocuser/Pytorch-Transformer/blob/main/core/model/InputLayer.py)
            - [위 코드에서 생성한 mask 그림](https://github.com/DeepFocuser/Pytorch-Transformer/blob/main/core/model/decoder_mask.png) 
              - encoder와 decoder의 입력 문장에 <PAD> 토큰이 있는 경우 attention에서 제외해주는 mask도 같이 적용한 결과이다.
              - <PAD> 토큰의 경우에는 실질적인 의미를 가진 단어가 아니므로, Transformer에서는 Key의 경우에 <PAD> 토큰이 존재한다면 이에 대해서는 계산을 제외하도록 마스킹(Masking)을 해준다.
  
  - ### Attention
    ![Desktop View](https://github.com/DeepFocuser/DeepFocuser.github.io/blob/gh-pages/post/transformer/attention.PNG?raw=true){: width="1000" height="600" }

    attention 함수는 query와 key-value 쌍을 output으로 맵핑 하는 것으로 설명될 수 있다. 여기서 query, key, values 그리고 output은 모두 다 vector이다. 출력은 values의 가중치 합으로 계산되며. 여기서 각 value에 할당된 가중치들은 해당 key를 가진 query의 호환성 함수에 의해 계산된다.
    ->(weight는 query, key로 만들고, 그 만들어진 weight를 value와 계산한다.)
    - #### Scaled Dot-Product Attention
      
      - 입력은 $queries,keys : d_{k} 차원, values : d_{v} 차원으로 구성 된다.$ 모든 키를 사용하여 쿼리의 내적을 계산하고, 
      $\sqrt{d_k}$ 로 나눈다. 그리고 values에 대한 가중치를 얻기 위해 softmax 함수를 적용한다.

      - 실제로는, queries 세트에 대한 attention 함수를 동시에 계산하여 행렬 Q로 묶는다. 키와 값도 행렬 K와 V로 묶입니다. 출력 행렬을 다음과 같이 계산한다.
        $$ Attention(Q, K, V) = softmax({QK^T \over \sqrt{d_k}}) $$
      
      가장 많이 사용하는 attention 함수들로는 additive attention 함수, dot-product 함수가 있다. dot-product 함수는 scaling factor $1 \over \sqrt{d_k}$가 있다는 점 외에는 본 논문에서 사용한 알고리즘과 동일하다.

      두 알고리즘은 비슷하지만, dot-product attention 함수가 실전에서 더 빠르고 공간 효율적이라고 한다.

    - #### Multi-Head Attention
      $d_{model}-차원$의 keys, values, queries를 사용하여 single attention function 을 수행하는 대신, $d_{k}, d_{k}, d_{v}$차원에 대해 학습된 서로 다른 선형 프로젝션을 사용하여 queries, keys, values 를 h번 선형으로 투영하는 것이 좋다는 것을 발견했다. 이렇게 하면 $d_{v}-dimensional$ output values를 산출하는 attention function을 병렬적으로 수행할 수 있다. 이 값들은 concat 되고 한번더 투영되어, 최종적인 값을 결과로 뽑아낸다.(Figure 2 보라)
      Multi-head attention을 통해 모델이 서로 다른 위치에서 서로 다른 표현 하위 공간의 정보에 공동으로 주의를 기울일 수 있다. 하나의 attention head로, 평균을 내는 것은 공동으로 주의를 기울이는 것을 억제한다.

      $$ MultiHead(Q, K, V) = Concat(Head_{1}, ..., head_{h})W^O $$
      $$ where Head_{i} = Attention(QW_{i}^Q, KW_{i}^K, VW_{i}^V) $$

      여기서 projections 들은 파라미터 행렬들이다. 
      $$ W_{i}^Q \varepsilon R^{d_{model} X d_{k}}, W_{i}^K \varepsilon R^{d_{model} X d_{k}}, W_{i}^V \varepsilon R^{d_{model} X d_{v}} and W^O \varepsilon R^{hd_{v} X d_{model}} $$

      h = 8 인 병렬 attention layer, heads를 사용했으며, 각각 $ d_{k} = d_{v} = {d_{model} \over h} = 64$ 이다. 
      
    - #### Applications of Attention in out Model
      - encoder-decoder attention layer안에서, queries 는 이전 decoder layer 에서, memory keys, values 는 encoder의 출력에서 온다. 이는 decoder의 모든 위치가 입력 sequence의 모든 위치들에 주의를 기울일 수 있게 하는 것을 가능하게 한다.
      seq2seq 모델안의 encoder-decoder attention 메커니즘을 흉내냈다.
      - encoder는 self-attention layers 들을 포함한다. self-attention layer에서 모든 keys, values, queries 는 같은 장소에서 오고, 이 경우에 encoder의 이전 layer의 output이다.
      encoder의 각 위치들은 encoder의 이전 layer의 모든 위치들에 주의를 기울일수 있다.  
      - 비슷하게, decoder의 self-attention layer는 decoder의 각 위치가 decoder의 해당 위치까지 그리고 그 위치를 포함하는 모든 위치에 주의를 기울일 수 있도록 한다. 자동 회귀 속성을 유지하려면 decoder에서 왼쪽으로의 정보 흐름을 방지해야 한다. 잘못된 연결에 해당하는 softmax 입력의 모든 값을 마스킹(-∞로 설정)하여 scaled dot-product Attention 내부에서 이것을 구현한다. (Figure 2를 보라.)
        - `무슨 말인지 와닿지 않는다.(2)`-해결
          - `무슨 말인지 와닿지 않는다.(1)` 의 해결 내용과 같다. 

  - ### Position-wise Feed-Forward Networks
    attention sub-layers 외'에도 encoder 및 decoder의 각 계층에는 각 위치에 개별적이고 동일하게 적용되는 fully connected feed-forward network 가 포함된다. 이것은 사이에 ReLU 활성화가 있는 두 개의 선형 변환으로 구성된다.
    $$ FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2} $$
    선형 변환은 다른 위치들에서 동일하지만, layer마다 다른 파라미터들을 사용한다. 
  - ### Embeddings and Softmax
    다른 sequence 변환 모델과 비슷하게, 우리는 input tokens 과 output tokens를 $d_{model}$ 차원 벡터로 바꾸기 위해 학습된 embeddings 을 사용한다. 우리는 또한 decoder 출력을 예측된 next-token 확률로 바꾸기 위해 학습된 선형 변환과 softmax함수를 사용한다. 우리 모델에서는 2개의 embedding layers 와 pre-softmax(예측 softmax) 선형 변환 간에 같은 가중치 행렬을 공유한다. embedding layer에서는 $\sqrt{d_{model}}$을 가중치에 곱한다.
    - `코드를 봐야 알 것 같다.(3)` - 해결
      - [코드와 설명 주석이 있는 링크](https://github.com/DeepFocuser/Pytorch-Transformer/blob/main/core/model/InputLayer.py)
      - https://nlp.seas.harvard.edu/2018/04/03/attention.html 에 Shared Embeddings 이란 제목으로 설명되어 있다.
  - ### Positional Encoding
    transformer는 recurrence 와 convolution을 포함하고 있지 않기 때문에 모델이 시간 순서정보를 사용하게 하기 위해서, sequence의 토큰의 상대적 위치 또는 절대적 위치에 어떤 정보를 넣어줘야만 한다. 이를 위해서 encoder와 decoder stacks의 아랫부분의 input embeddings에 `positional encoding`이라는 것을 추가한다. embeddings과 `positional encoding`이 덧셈이 가능하게 하기 위해 `positional encoding` 은  embeddings과 같이 $d_{model}$ 차원을 가진다. 

    본 논문에서는 다른 주기를 가지는 sine, cosine 함수를 사용한다.

    $$ PE_{pos, 2i} = sin({position \over 10000^{2i \over d_{model}}}) $$

    $$ PE_{pos, 2i+1} = cos({position \over 10000^{2i \over d_{model}}}) $$

    - `여기 내용을 이해가기가 쉽지 않다.(4)`
      - seq2seq 모델같은 경우는 RNN을 사용하므로, 입력 자체에 시간 속성이 부여되어 있다. 그런데 transformer같은 경우는 그런게 없다. 그래서 transformer의 encoder, decoder에 시간 속성을 부여하기 위해서 위의 sin, cos 함수를 더해주는 것이다.
      - [코드와 설명 주석이 있는 링크](https://github.com/DeepFocuser/Pytorch-Transformer/blob/main/core/model/InputLayer.py)
      - [위 코드에서 생성한 Positional Encoding 그림](https://github.com/DeepFocuser/Pytorch-Transformer/blob/main/core/model/pe.png) 

아래의 내용부터는 그렇게 중요하다고 생각되지 않는다. 
따라서 무슨내용을 다뤘는지만 간단히 설명하고 넘어간다.

## Why Self-Attention 
---

왜 self-Attention을 사용했는지에 대해 말하고 있고, 
몇가지 장점을 설명하고 있다.(계산량, 병렬화, long-range dependencies 학습)

부수적인 이익으로 self-attention은 해석가능한 모델을 산출해낸다고 한다.

## Training
---

  - ### Training Data and Batching
    데이터셋에 대한 설명
  - ### Hardward and Schedule
    어떤 GPU룰 썼고, 어떻게 학습을 했고, 어떤 hyperparameter를 사용했는지에 대한 설명
  - ### Optimizer
    어떤 Optimizer를 사용했고, 학습률은 어떻게 설정했는지에 대한 설명
  - ### Regularization
    학습할때 사용한 3가지 type의 regularization 설명

## Results
---
  - ### Machine Translation
  - ### Model Variations
  - ### English Constituency Parsing

## Conclusion
---
번역 작업에서 Transformer은 RNN or CONV layer 기반의 아키텍쳐보다 훨씬 더 빠르게 학습이 가능했음. WMT2014 데이터셋 기반 영어-독어 번역, 영어-프랑스어 번역 작업에서 가장 좋은 결과를 성취할 수 있었다.

Transformer 모델을 input-output 구조를 가지고 있는 문제들(images, audio, video)에 확장할 계획이다. 

## Code
---
논문을 리뷰하며 잘 이해가 되지 않는 부분들이 있었다.(표시해 놓음) 
이제 코드를 구현하면서 내가 제대로 이해하지 못한 부분을 채워나가는 
시간이 필요할 것 같다.
  * 며칠에 걸쳐서 독일어-영어 번역기 Transformer 모델 구현을 완료했다. 역시 논문을 읽는 것과 구현 사이에는 엄청난 괴리가 있다. 수많은 사이트들을 참고했고, 하나하나 직접 구현했다. 그 결과는 [여기 내 깃허브 저장소](https://github.com/DeepFocuser/Pytorch-Transformer)에 있다. 상세한 설명과 참고 자료등을 주석으로 달아놨으니 누군가에겐 도움이 되길바란다.

<!-- https://ghdic.github.io/math/default/mathjax-%EB%AC%B8%EB%B2%95/ -->
