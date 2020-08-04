# Feature

> 샘플을 잘 설명하는 특징



## Featurn in Machine Learning

- MNIST Classification

  >**MNIST 데이터베이스** (Modified [National Institute of Standards and Technology](https://ko.wikipedia.org/wiki/미국_국립표준기술연구소) database)는 손으로 쓴 숫자들로 이루어진 대형 [데이터베이스](https://ko.wikipedia.org/wiki/데이터베이스)이며, 다양한 [화상 처리](https://ko.wikipedia.org/wiki/디지털_화상_처리) 시스템을 [트레이닝](https://ko.wikipedia.org/w/index.php?title=트레이닝_세트&action=edit&redlink=1)하기 위해 일반적으로 사용된다.[[1\]](https://ko.wikipedia.org/wiki/MNIST_데이터베이스#cite_note-1)[[2\]](https://ko.wikipedia.org/wiki/MNIST_데이터베이스#cite_note-2) 이 데이터베이스는 또한 [기계 학습](https://ko.wikipedia.org/wiki/기계_학습) 분야의 트레이닝 및 테스트에 널리 사용된다.

  ![MNIST sample images.](https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/MnistExamples.png/220px-MnistExamples.png)

  > MNIST 데이터베이스는 60,000개의 트레이닝 이미지와 10,000개의 테스트 이미지를 포함한다. 트레이닝 세트의 절반과 테스트 세트의 절반은 NIST의 트레이닝 데이터셋에서 취합하였으며, 그 밖의 트레이닝 세트의 절반과 테스트 세트의 절반은 NIST의 테스트 데이터셋으로부터 취합되었다.

  - 특정 위치에 곧은(휘어진) 선이 얼마나 있는가?
  - 특정 위치에 선이 얼마나 굵은가?
  - 특정 위치에 선이 얼마나 기울어져 있는가?



### No need of Hand-crafted Feature in Deep Learning

- Traditional Machine Learning
  - 사람이 데이터를 면밀히 분석 후,  가정을 세움
  - 가정에 따라 전처리를 하여 feature를 추출
  - 추출된 feature를 model에 넣어 학습
  - 장점: 사람이 학습하기 쉬움
  - 단점: 사람이 미처 생각하지 못한 특징의 존재 가능성



- Current Deep Learning
  - Raw 데이터에 최소한의 전처리를 수행
  - 데이터를 model에 넣어 학습
  - 장점: 구현이 용이하고 미처 발견하지 못한 특징도 활용
  - 단점: 사람이 해석하기 어려움



### Feature Vector

- 각 특징들을 모아서 하나의 vector로 만든것
  - Tabular Dataset의 각 row도 이에 해당
- 각 차원(dimension)은 어떤 속성에 대한 level을 나타냄
  - 각 속성에 대한 level이 비슷할수록 비슷한 샘플이라고 볼 수 있음
- feature vector를 통해 샘플 사이의 거리(유사도)를 계산할 수 있음



























