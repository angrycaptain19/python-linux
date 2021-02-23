# 1. spaCy 사용하기
import spacy
# spacy_en = spacy.load('en')
spacy_en = spacy.load('en_core_web_sm')

def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]

en_text = "A Dog Run back corner near spare bedrooms"
print(tokenize(en_text))

# 2. NLTK 사용하기
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

print(word_tokenize(en_text))

# 3. 띄어쓰기로 토큰화
print(en_text.split())

# 4. 한국어 띄어쓰기 토큰화
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"
print(kor_text.split())

# 5. 형태소 토큰화 : 형태소 분석기 Mecab 사용하기
from konlpy.tag import Mecab
tokenizer = Mecab()
print(tokenizer.morphs(kor_text))

# 6. 문자 토큰화
print(list(en_text))








