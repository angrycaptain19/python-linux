'''
토치텍스트가 제공하는 기능

파일 로드하기(File Loading) : 다양한 포맷의 코퍼스를 로드합니다.
토큰화(Tokenization) : 문장을 단어 단위로 분리해줍니다.
단어 집합(Vocab) : 단어 집합을 만듭니다.
정수 인코딩(Integer encoding) : 전체 코퍼스의 단어들을 각각의 고유한 정수로 맵핑합니다.
단어 벡터(Word Vector) : 단어 집합의 단어들에 고유한 임베딩 벡터를 만들어줍니다. 랜덤값으로 초기화한 값일 수도 있고, 사전 훈련된 임베딩 벡터들을 로드할 수도 있습니다.
배치화(Batching) : 훈련 샘플들의 배치를 만들어줍니다. 이 과정에서 패딩 작업(Padding)도 이루어집니다.

'''

# 1. 훈련 데이터와 테스트 데이터로 분리하기
import urllib.request
import pandas as pd

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
print(df.head())
print('전체 샘플의 개수 : {}'.format(len(df)))

train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

# 2. 필드 정의하기(torchtext.data)
from torchtext import data # torchtext.data 임포트

# 필드 정의
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)
'''
sequential : 시퀀스 데이터 여부. (True가 기본값)
use_vocab : 단어 집합을 만들 것인지 여부. (True가 기본값)
tokenize : 어떤 토큰화 함수를 사용할 것인지 지정. (string.split이 기본값)
lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
is_target : 레이블 데이터 여부. (False가 기본값)
fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행된다.
'''

# 3. 데이터셋 만들기
from torchtext.data import TabularDataset
train_data, test_data = TabularDataset.splits(
        path='.', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))
print(vars(train_data[0]))

# 필드 구성 확인.
print(train_data.fields.items())

# 4. 단어 집합(Vocabulary) 만들기
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))

# 5. 토치텍스트의 데이터로더 만들기
from torchtext.data import Iterator
batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

batch = next(iter(train_loader)) # 첫번째 미니배치
print(type(batch))
print(batch.text)

# 6. <pad>토큰이 사용되는 경우
batch = next(iter(train_loader)) # 첫번째 미니배치
print(batch.text[0]) # 첫번째 미니배치 중 첫번째 샘플




