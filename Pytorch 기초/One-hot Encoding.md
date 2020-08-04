# One-hot Encoding



### Categorical   vs Continuous Value

- Categorical Value
  - 보통 discrete value
  - 단어나 클래스



- Continuous Value
  - 키, 몸무게
  - 주로 연속적인 값들



- 가장 결정적인 차이
  - Continuous Value 는 비슷한 값은 비슷한 의미를 가짐
  - Categorical Value는 비슷한 값일지라도 상관없음



### Categorical Vaue: Text

- 단어를 사전 순으로 index에 mapping 해보자

  |  0   |  1   |   2    |  3   |  4   |  5   |   6    |  7   |   8    |  9   |  10  |   11   |  12  |  13  |   14   |  15  |
  | :--: | :--: | :----: | :--: | :--: | :--: | :----: | :--: | :----: | :--: | :--: | :----: | :--: | :--: | :----: | :--: |
  | 가위 | 공책 | 교과서 | 노트 | 딱풀 | 볼펜 | 색연필 | 샤프 | 싸인펜 | 연필 |  자  | 지우개 | 책상 |  칼  | 필기장 | 필통 |

  

- distance
  - distance(연필, 볼펜) < distance(연필, 자)
  - distance(공책, 필기장) < distance(공책, 딱풀)



- table

  ||연필 - 볼펜| = 4 > 1  =|연필 - 자|

  |공책 - 필기장| = 13 > 3  =|공책 - 딱풀|



### One- hot Encoding

- 크기가 의미를 갖는 integer 값 대신, 1개의 1과 n-1개의 0으로 이루어진 n차원의 벡터

  ![image-20200804225706107](C:\Users\multicampus\NLP\Pytorch 기초\image-20200804225706107.png)



### Sparse VS Dense Vector

- vector의 대부분의 element가 0인 경우, Sparse Vector라고 부름

  <=> Dense Vector



- One-hot vector는 Sparse Vector의 정점

  ![image-20200804225816019](C:\Users\multicampus\NLP\Pytorch 기초\image-20200804225816019.png)



### One-hot Vector의 문제점

- 서로 다른 두 벡터는 항상 직교한다.
  - Cosine similarity가 0
  - element wise 곱이 0
- 따라서 두 샘플 사이의 유사도를 구할 수 없다.



### Motivation of Embedding Vectors

- NLP에서 단어는 categorical and discrete value의 속성을 가짐
  - 따라서 one-hot representation으로 표현
  - 하지만, 실제 존재하는 단어 사이의 유사도를 표현하지는 못함



- Word Embedding Vectors

  - Word2Vect 또는 DNN을 통해 차원 축소 및 dense vector로 표현

    ![Create Word2Vec in Gensim using Gutenberg corpus – Fhadli's Board](https://fhadlisboard.com/wp-content/uploads/2018/09/1-vvtIsW1AblmgLkq1peKfOg.png)







































