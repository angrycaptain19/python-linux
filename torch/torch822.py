# 03. 토치텍스트 튜토리얼(Torchtext tutorial) - 한국어
# 1. 형태소 분석기 Mecab 설치
# Colab에 Mecab 설치

# 2. 훈련 데이터와 테스트 데이터로 다운로드 하기
import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_df = pd.read_table('ratings_train.txt')
test_df = pd.read_table('ratings_test.txt')

print(train_df.head())
print('훈련 데이터 샘플의 개수 : {}'.format(len(train_df)))
print('테스트 데이터 샘플의 개수 : {}'.format(len(test_df)))

# 3. 필드 정의하기(torchtext.data)
from torchtext import data # torchtext.data 임포트
from konlpy.tag import Mecab

# Mecab을 토크나이저로 사용
tokenizer = Mecab()
# 네이버 영화 리뷰 데이터는 3개의 열로 구성되어져 있으므로 3개의 필드를 정의합니다.

# 필드 정의
ID = data.Field(sequential = False,
                use_vocab = False) # 실제 사용은 하지 않을 예정

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer.morphs, # 토크나이저로는 Mecab 사용.
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True)

# 4. 데이터셋 만들기
from torchtext.data import TabularDataset

train_data, test_data = TabularDataset.splits(
        path='.', train='ratings_train.txt', test='ratings_test.txt', format='tsv',
        fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True)

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))

print(vars(train_data[0]))

# 5. 단어 집합(Vocabulary) 만들기
# 정의한 필드에 .build_vocab() 도구를 사용하면 단어 집합을 생성합니다.
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))

print(TEXT.vocab.stoi)

# 6. 토치텍스트의 데이터로더 만들기
from torchtext.data import Iterator
batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)
print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))
print('테스트 데이터의 미니 배치 수 : {}'.format(len(test_loader)))

batch = next(iter(train_loader)) # 첫번째 미니배치
print(batch.text)





