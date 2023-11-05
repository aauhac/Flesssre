[Jupyter nbviewer](https://nbviewer.org/)

https://wikidocs.net/50739
https://wikidocs.net/50739
https://github.com/Kyubyong/wordvectors
gensim을 사용하여 간단하게 만들어보려 했으나 알 수 없는 오류가 계속 떠서 word2vec모델을 학습시킨 후 문장을 벡터로 변환한 후 코사인 유사도를 비교해 rank가 높은 문장들 몇개를 추출하여 요약하는 방식 사용

인공지능 기사 링크 크롤링
``` python
from selenium import webdriver
from selenium.webdriver.common.by import By
import re
from konlpy.tag import Komoran
import numpy as np
import gensim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

driver = webdriver.Chrome()

driver.get("https://www.aitimes.kr/news/articleList.html?sc_section_code=S1N5&view_type=sm")
 
type2 = driver.find_element(By.CLASS_NAME, "type2")
links = type2.find_elements(By.CLASS_NAME, 'thumb')

list = []
for l in links:
    list.append(l.get_attribute("href"))

driver.close()
print("done")
```

모여진 링크마다 크롤링 후 전처리
```python
train = [] 
from bs4 import BeautifulSoup
import requests

for i in list:
    response = requests.get(i)
    soup = BeautifulSoup(response.text, "html.parser")
    s = soup.select('#article-view-content-div > p')
    for i in s:
        train.append(i.text)
print("done")

for i, e in enumerate(train):          #\xa0, \n, \r, 괄호와 괄호 안의 내용 제거 후 a-z, A-Z, 한국어, 공백, .을 제외하고 모든것을 삭제
    cle1 = re.sub("[\n\r\']", '', re.sub(r'\xa0', ' ', e))
    cle2 = re.sub("[^0-9A-Za-z가-힣\s.]", '', re.sub("\([^)]*\)", " ", cle1))
    train[i] = cle2
print("done")

with open("stopword.txt", 'r', encoding='UTF8') as f:  #불용어 처리
    stopwords = f.read().split()
print("done")

train_s = []
for i in train:
    for s in i.split("."):
        if s.strip() != '':
            train_s.append(str(s).strip())  #양 끝의 공백 제거 후 비어있는 리스트 제거
print("done")

komoran = Komoran()
train_w = []
for i in train_s:
    train_w.append(komoran.nouns(i))  #명사 추출
print("done")

train_dsw = []
for s in train_w:
    l = []
    for w in s:
        if w not in stopwords: #불용어 제거
            l.append(w)
    train_dsw.append(l)
print("done")
```

word2vec 모델 생성

```python

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = Word2Vec(sentences=train_dsw, vector_size=100, window=5, min_count=5, workers=4, sg=0)
print("done")

model.wv.save_word2vec_format('ko_w2v')
print("done")

loaded_model = KeyedVectors.load_word2vec_format("ko_w2v")
print("done")

embedding_dim = 100
zero_vector = np.zeros(embedding_dim)
print("done")

def calculate_sentence_vector(sentence, model):
    word_vectors = [model[word] for word in sentence if word in model]   #문장 벡터 생성 함수. 없는 단어면 0으로 대체

    if word_vectors:
        sentence_vector = sum(word_vectors) / len(word_vectors)
    else:
        sentence_vector = np.zeros(model.vector_size)
    
    return sentence_vector

print("done") 

```

테스트 해볼 기사 크롤링

```python

from bs4 import BeautifulSoup
import requests

links = []
response = requests.get('https://www.aitimes.kr/news/articleList.html?sc_sub_section_code=S2N3&view_type=sm')
soup = BeautifulSoup(response.text, "html.parser")
s = soup.select("#section-list > ul > li")
for i in s:
    links.append("https://www.aitimes.kr" + i.select_one("a").get("href"))
print("done")

test = []
for i in links:
    response = requests.get(i)
    soup = BeautifulSoup(response.text, "html.parser")
    s = soup.select('#article-view-content-div > p')
    te = ''
    for t in s:
        te += t.text + " "
    test.append(te)
print("done")

for i, e in enumerate(test):
    tcl1 = re.sub("[\n\r\']", '', re.sub(r'\xa0', ' ', e))
    tcl2 = re.sub("[^0-9A-Za-z가-힣\s.]", '', re.sub("\([^)]*\)", " ", tcl1))
    test[i] = tcl2
    
test_s = []
for i in test:
    test_sl = []
    for s in i.split("."):
        if s.strip() != '':
            test_sl.append(str(s).strip())
    test_s.append(test_sl)

komoran = Komoran()
test_w = []
for i in test_s:
    test_wl = []
    for j in i:
        test_wl.append(komoran.nouns(j))
    test_w.append(test_wl)

test_dsw = []
for s in test_w:
    new_sentences = []
    for sentence in s:
        new_words = []
        for word in sentence:
            if word not in stopwords:
                new_words.append(word)
        new_sentences.append(new_words)
    test_dsw.append(new_sentences)
    
print("done")

```



```python

p_list = []
for i in test_dsw:
    p = [calculate_sentence_vector(t, loaded_model) for t in i] #문장 벡터 생성
    p_list.append(p)
print("done") 

```

코사인 유사도 행렬 생성 후 리스트에 정리

```python

sim_mat_list = []

for p in p_list:
    sim_mat = np.zeros([len(p), len(p)])
    for i in range(len(p)):
        for j in range(len(p)):
            sim_mat[i][j] = cosine_similarity(p[i].reshape(1, embedding_dim),
                                              p[j].reshape(1, embedding_dim))[0,0]
    sim_mat_list.append(sim_mat)
print("done")

```

코사인 유사도 그래프 그려보기

```python

def draw_graphs(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(nx_graph,pos,font_color='red')
    plt.show()

for i in sim_mat_list:
    draw_graphs(i)

```

랭크매긴 후 요약

```python

def calculate_score(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix) #유사도 행렬을 비교해 rank 매기는 함수
    scores = nx.pagerank(nx_graph)
    return scores


for i, e in enumerate(sim_mat_list):
    s = ''
    score = calculate_score(e)
    d = sorted(score.items(), key=lambda x: x[1], reverse=True) #rank 매긴 후 상위 3개 추출
    wc = [j[0] for j in d]
    c = wc[:3]
    for n in c:
        s += test_s[i][n] + ". " #rank가 높은 문장의 index를 가져와 문장 요약
    print(s, '\n')

```


