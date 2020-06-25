# Pytorch 설치

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

  