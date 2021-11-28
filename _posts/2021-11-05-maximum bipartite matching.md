---
title: Maximum Bipartite Matching
author: JONGGON KIM
date: 2021-11-05 20:00:00 +0900
categories: [알고리즘, 코드 조각]
tags: [Maximum Bipartite Matching]
math: true
mermaid: true
# pin: true
---

[DETR](https://arxiv.org/abs/2005.12872) 논문을 읽기에 앞서 ~~Maximum Bipartite Matching 알고리즘에 대한 이해가 필요하여 정리해본다.~~ 사실 [`Optimal Bipartite Matching-Hungarian algorithm`](https://deepfocuser.github.io/posts/hungarian-algorithm/)에 대한 내용이 필요하다.
DETR을 이해하는데 있어 Maximum Bipartite Matching이 필요한건 아니지만 알고리즘 공부겸 남겨놓자

Maximum Bipartite Matching 알고리즘 코드 구현은 [DFS 코드](https://deepfocuser.github.io/posts/bfsdfs/)를 기반으로 한다.

## Maximum Bipartite Matching(이분 매칭)
---
두 개의 정점 그룹이 존재할 때 모든 간선(경로)의 용량이 1이면서 양쪽 정점이 서로 다른 그룹에 속하는 그래프를 이분 그래프(Bipartite Graph)라고 한다. 예를 들어, 한쪽 그룹은 X 그룹, 다른 한쪽 그룹은 Y 그룹이라고 할 때 모든 경로의 방향이 X->Y인 그래프의 최대 유량을 구하는 것이 Bipartite Matching(이분 매칭)이다.`이분 매칭을 통해 구하고자 하는 것은 최대 매칭 수이다.` 매칭을 한다는 것은 어떤 정점이 그것이 가리키는 위치의 다른 정점을 점유한 상태를 말하며
각 정점은 한 개씩만 점유 가능하고 여러개의 정점을 점유할 수 없다.

## python 코드 구현 
---
![Desktop View](https://github.com/DeepFocuser/DeepFocuser.github.io/blob/gh-pages/post/BipartiteGraph/BG.JPG?raw=true){: width="490" height="310" }

```python
graph = {
    'A': [1, 2], # A
    'B': [1],    # B
    'C': [2, 3], # C
    'D': [4, 5], # D
    'E': [3]     # E
}

# for문 이용한 dfs 코드로도 시도했으나 실패
def dfs_recursive(departure_node):

    # dfs_recursive(result[node])에 or 앞에 오는 경우
    # if start_node == "NONE":
    #     return False

    for destination_node in graph[departure_node]:

        # 이미 처리한 정점은 고려하지 않음 - 재귀시에만 값 유지
        if visited[destination_node]:
            continue

        visited[destination_node] = True
        '''
        # dfs_recursive(result[node])에 or 앞에 오는 경우에는 아래의 코드가 필요
        if start_node == "NONE": #
            return False
        '''
        #if dfs_recursive(result[node]) or result[node] == "NONE":
        # dfs_recursive(result[destination_node] 는 이전 노드를 다시 매칭하기 위함
        if result[destination_node] == "NONE" or dfs_recursive(result[destination_node]): # 앞에 것이 True면 바로 if문 안으로 들어간다.
            result[destination_node] = departure_node # 매칭
            return True

    return False

if __name__ == "__main__":

    '''
    구현하기전 생각해볼 것? 필요한 변수
    
    1. 시작점에서 목적지점까지 한 사이클을 돌 때, 방문했는지 안했는지 판단할 변수가 필요하다. --> visited
    2. 목점지점 입장에서 시작점을 기록할 변수가 필요하다 --> result
    
    좀더 빠른 속도를 위해 모든 변수 전부다 dictionary로 !!!
    '''

    graph_length = len(graph)
    result = { vertex_destination_node : vertex_departure_node for vertex_destination_node, vertex_departure_node in
               zip(range(1, graph_length+1), ["NONE"]*graph_length) }

    '''
    result print
    {1: 'NONE', 2: 'NONE', 3: 'NONE', 4: 'NONE', 5: 'NONE'}
    '''

    for _, departure_node in enumerate(graph.keys(), start=1): # ['A', 'B', 'C', 'D', 'E']
        # visited는 시작 노드마다 각각 적용되어야 함 - for문 한 사이클 돌고 초기화 -> 시작노드가 n개 이므로
        visited = {vertex_destination_node: judgment for vertex_destination_node, judgment in
                   zip(range(1, graph_length+1), [False]*graph_length)}
        '''
        visitied print
        {1: False, 2: False, 3: False, 4: False, 5: False}
        '''
        dfs_recursive(departure_node)

    all_length=len(result.values())
    matching_number = all_length - list(result.values()).count("NONE")
    print(f"최대 매칭 : {matching_number}")
    for dest, dep in sorted(result.items(), key=lambda x: x[-1]):
        if dep != "NONE":
            print(f"{dep} : {dest}")
    '''
    print 
    최대매칭 : 4
    A : 2
    B : 1
    C : 3
    D : 4
    '''

```
