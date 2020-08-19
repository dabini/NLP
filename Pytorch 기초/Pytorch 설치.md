# Pytorch 설치

### Framework

- 프레임워크란?
  - 규칙의 집합
  - 앞서 개발한 사람들의 지혜와 사례를 활용
  - 정형화된 형태의 규칙으로 만든 프로그램



- 제어반전

  - 프로그램이 가이드라인 대로 코딩해야 함

  - 라이브러리 => 코딩을 할 때 필요한 라이브러리 활용

    

## Deep Learning Modeling

### 개발환경 생성 및 활성화

- `Anaconda Prompt` 실행 후 명령어 기입

  ```powershell
  # 가상환경(NLPApps) 설치
  conda create -n NLPApps python=3.7 anaconda
  ```

  ```powershell
  # 가상환경 활성화
  conda activate NLPApps
  ```

  

### pytorch 외 필요한 것 설치

> visual code c++ 설치



- `pytorch`, `jupyter` 설치

  ```powershell
  conda install pytorch-cpu torchvision-cpu -c pytorch
  ```

  ```powershell
  # jupyter 설치
  pip install jupyter
  ```

  ```powershell
  # jupyter notebook 실행
  jupyter notebook
  ```


mmand Mode ( press Esc to enable) 



shift-Enter : run cell, select below



Ctrl-Enter : run cell



Alt-Enter : run cell, insert below 



Y : to code 



M : to markdown



B : insert cell below



X : cut selected cell



C : copy selected cell



Shift-V : paste cell above



V : paste cell below



Z : undo last cell deletion



D,D : delete selected cell



Shift-M : merge cell below



Edit Mode ( press Enter to enable)



Shift-Tab : 툴팁표시



Ctrl-] : indent



Ctrl-Shift- : split cell



