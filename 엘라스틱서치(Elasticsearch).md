# 엘라스틱서치(Elasticsearch)

>  Apache Lucene( 아파치 루씬 ) 기반의 Java 오픈소스 분산 검색 엔진
>
> Elasticsearch를 통해 루씬 라이브러리를 단독으로 사용할 수 있게 되었으며, 방대한 양의 데이터를 신속하게, 거의 실시간( NRT, Near Real Time )으로 저장, 검색, 분석할 수 있습니다.



### 1. 개요

#### 1) 엘라스틱 서치 데이터 간략 소개

- 엘라스틱서치는 검색 엔진인 아파치 루씬(Apache Lucene)으로 구현한 RESTful API 기반의 검색 엔진입니다.
- 엘라스틱서치 아키텍쳐: 클러스터 기반
  - 수평확장: 클러스터를 무한으로 확장할 수 있음
  - 인덱스 샤딩: 엘라스틱서치는 인덱스를 조각내서 샤드(shard)라는 조각난 데이터로 만듭니다. 때문에 나누어진 데이터를 편하게 각 호스트에 분리해서 보관할 수 있습니다.



![img](https://postfiles.pstatic.net/MjAxOTAxMDlfMTY4/MDAxNTQ3MDMzOTk5NDMy.EtJwOxMBUT0ZFm3717uaXp4jT5yRiKH__4Cys_lPNXwg.lBqx0PEoiWNe3OeJLx1139OLXaYUb1kBN-l4Vu4nF2wg.PNG.takane7/%EC%97%98%EB%9D%BC%EC%8A%A4%ED%8B%B1_%EA%B0%84%EB%9E%B5_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EB%8B%A4%EC%9D%B4%EC%96%B4%EA%B7%B8%EB%9E%A8.png?type=w966)



#### 2) RDBMS와 용어 비교

![image-20201103135430179](C:\Users\42Maru\AppData\Roaming\Typora\typora-user-images\image-20201103135430179.png)

![img](https://t1.daumcdn.net/cfile/tistory/998444375C98CC021F)

![img](https://t1.daumcdn.net/cfile/tistory/99E0A9425C98CF7A0A)



### 2. 데이터 구조

#### 1) 데이터 다이어그램

![img](https://postfiles.pstatic.net/MjAxOTAxMDlfMjM0/MDAxNTQ3MDM0NDExMTY1.W9_9oPoPNtXXe5iDnbPDYz0GXR7EJSJn8nJovl3VL0Eg.ayrn5KknULLi4TM5VMYDFhaWUYyFQd3xPG05M9DADxUg.PNG.takane7/%EC%97%98%EB%9D%BC%EC%8A%A4%ED%8B%B1_%EC%83%81%EC%84%B8_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EB%8B%A4%EC%9D%B4%EC%96%B4%EA%B7%B8%EB%9E%A8.png?type=w966)

- 엘라스틱서치는 위와 같이 데이터(document)를 `엘라스틱 인덱스`로 만든 뒤, `샤드`로 분리하여 보관하고 있다. 샤드는 논리적/물리적으로 분할된 인덱스인데, 각각의 엘라스틱 서치 샤드는 `루씬 인덱스`이기도 하다.

- `루씬`은 새로운 데이터를 엘라스틱서치 인덱스에 저장할 때 `세그먼트`를 생성한다. 루씬의 인덱스 조각인 이 세그먼트를 조합해 저장한 데이터의 검색을 할 수 있다.

  --> 색인 처리량이 매우 중요할 때는 세그먼트를 더 생성하기도 한다.

- 루씬은 순차적으로 세그먼트를 검색하므로 세그먼트 수가 많아지면 검색속도도 따라서 느려지게 된다.



#### 2) 데이터 설명

![image-20201103135750797](C:\Users\42Maru\AppData\Roaming\Typora\typora-user-images\image-20201103135750797.png)



### 3. 클러스터 구조

#### 1) 클러스터 다이어그램

![img](https://postfiles.pstatic.net/MjAxOTAxMDlfMjg3/MDAxNTQ3MDM0NzI3Mjc1.-zcjUIUWYX4Zd9TTmAUL9oHGcaVjloX2SREzK6aYW2Eg._iD23khdCEeGti22bG_uUh5HY-IJXUDz-l9nOF7AxQUg.PNG.takane7/%EC%97%98%EB%9D%BC%EC%8A%A4%ED%8B%B1_%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0_%EB%8B%A4%EC%9D%B4%EC%96%B4%EA%B7%B8%EB%9E%A8.png?type=w966)

![image-20201103140012952](C:\Users\42Maru\AppData\Roaming\Typora\typora-user-images\image-20201103140012952.png)

- 위 다이어그램은 3개의 엘라스틱서치 인스턴스 환경에서, 4개의 샤드를 2개의 복제본으로 구성했을 때의 구조입니다.
- 엘라스틱 서치는 클러스터 구조로 구성되어 있으며 샤드와 복제본의 수를 설정해두면 스스로 각 노드에 샤드를 분배하여 장애발생 시 데이터 손실을 최소호합니다. 프라이머리 샤드가 손실되었을 경우에는 레플리카를 프라이머리로 승격시켜 데이터 손실을 방지합니다.





### 3. ELK 스택

> Elasticsearch는 검색을 위해 단독으로 사용되기도 하며, ELK( Elasticsearch / Logstatsh / Kibana )스택으로 사용되기도 합니다.



#### 1) Logstash

- 다양한 소스(DB, csv 파일 등)의 로그 또는 트랜잭션 데이터를 수집, 집계, 파싱하여 Elasticsearch로 전달

#### 2) Elasticsearch

- Logstash로부터 받은 데이터를 검색 및 집계를 하여 필요한 관심 있는 정보를 획득

#### 3) Kibana

- Elasticsearch의 빠른 검색을 통해 데이터를 시각화 및 모니터링



![img](https://t1.daumcdn.net/cfile/tistory/993B7E495C98CAA706)

### 4. ElasticSearch 특징

#### 1) Scale out

- 샤드를 통해 규모가 수평적으로 늘어날 수 있음

#### 2) 고가용성

- Replica를 통해 데이터의 안전성을 보장

#### 3) Schema Free

- Json 문서를 통해 데이터 검색을 수행하므로 스키마 개념이 없음

#### 4) Restful

- 데이터 CRUD 작업은 HTTP Restful API를 통해 수행하며, 각각 다음과 같이 대응

  ![image-20201103141355063](C:\Users\42Maru\Desktop\42MARU\lucy\딥러닝\image-20201103141355063.png)

#### 5) 역색인(Inverted Index)

Elasticsearch는 텍스트를 파싱해서 검색어 사전을 만든 다음에 inverted index 방식으로 텍스트를 저장합니다.

"Lorem Ipsum is simply dummy text of the printing and typesetting industry"

예를 들어, 이 문장을 모두 파싱해서 각 단어들( Lorem, Ipsum, is, simply .... )을 저장하고,

대문자는 소문자 처리하고, 유사어도 체크하고... 등의 작업을 통해 텍스트를 저장합니다.

때문에 RDBMS보다 전문검색( Full Text Search )에 빠른 성능을 보입니다.



---

### Reference

- https://victorydntmd.tistory.com/308

- 