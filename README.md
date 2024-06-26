# DeepLearning_Classification
DeepLearning Course HW2 - MNIST Classfication

## Folder Structure
<<<<<<< HEAD
```bash
├── dataset.py: MNIST 데이터셋을 로드하고 전처리하는 코드가 포함되어 있습니다.
├── model.py: LeNet-5와 사용자 정의 MLP 모델을 정의하는 코드가 포함되어 있습니다.
├── main.py: 모델을 훈련하고 평가하는 메인 코드가 포함되어 있습니다.
```

## How to use
- 데이터셋 다운로드: MNIST 데이터셋을 다운로드하고 데이터 디렉토리에 압축을 해제합니다.
- 데이터 전처리: dataset.py 파일을 실행하여 데이터셋을 전처리합니다.
- 모델 선택: model.py 파일에서 사용할 모델을 선택합니다 (LeNet-5 또는 사용자 정의 MLP).
- 모델 훈련: main.py 파일을 실행하여 모델을 훈련합니다.
- 모델 평가: 훈련된 모델의 성능을 평가하고 결과를 확인합니다.

## Result (Report)
#### 0. Model Parameters
- LeNet-5 parameter count: 44,426
1) conv1 layer:
  입력 채널 수: 1
  출력 채널 수: 6
  커널 크기: 5x5
  파라미터 수: (입력 채널 수 * 출력 채널 수 * 커널 높이 * 커널 너비) + 출력 채널 수 (편향)
  = (1 * 6 * 5 * 5) + 6 = 156
2) conv2 layer:
  입력 채널 수: 6
  출력 채널 수: 16
  커널 크기: 5x5
  파라미터 수: (입력 채널 수 * 출력 채널 수 * 커널 높이 * 커널 너비) + 출력 채널 수 (편향)
  = (6 * 16 * 5 * 5) + 16 = 2416
3) fc1 layer:
  입력 뉴런 수: 16 * 4 * 4
  출력 뉴런 수: 120
  파라미터 수: (입력 뉴런 수 * 출력 뉴런 수) + 출력 뉴런 수 (편향)
  = (16 * 4 * 4 * 120) + 120 = 30840
4) fc2 layer:
  입력 뉴런 수: 120
  출력 뉴런 수: 84
  파라미터 수: (입력 뉴런 수 * 출력 뉴런 수) + 출력 뉴런 수 (편향)
  = (120 * 84) + 84 = 10164
5) fc3 레이어:
  입력 뉴런 수: 84
  출력 뉴런 수: 10
  파라미터 수: (입력 뉴런 수 * 출력 뉴런 수) + 출력 뉴런 수 (편향)
  = (84 * 10) + 10 = 850

- Custom MLP parameter count: 242,762
1) fc1 레이어
  입력 뉴런 수: 28 * 28
  출력 뉴런 수: 256
  파라미터 수: (입력 뉴런 수 * 출력 뉴런 수) + 출력 뉴런 수 (편향)
  = (28 * 28 * 256) + 256 = 200960
2) fc2 레이어:
  입력 뉴런 수: 256
  출력 뉴런 수: 128
  파라미터 수: (입력 뉴런 수 * 출력 뉴런 수) + 출력 뉴런 수 (편향)
  = (256 * 128) + 128 = 32896
3) fc3 레이어:
  입력 뉴런 수: 128
  출력 뉴런 수: 64
  파라미터 수: (입력 뉴런 수 * 출력 뉴런 수) + 출력 뉴런 수 (편향)
  = (128 * 64) + 64 = 8256
4) fc4 레이어:
  입력 뉴런 수: 64
  출력 뉴런 수: 10
  파라미터 수: (입력 뉴런 수 * 출력 뉴런 수) + 출력 뉴런 수 (편향)
  = (64 * 10) + 10 = 650

#### 1. Plot of Training and Testing Statistics
- CustomMLP (Loss & Accuracy)
![cmlp_curves](https://github.com/YewonMin/DeepLearning_Classification/assets/108216502/703bffeb-bd4b-48c7-9a33-d4f7492768c8)
- LeNet5 (Loss & Accuracy)
![lenet5_curves](https://github.com/YewonMin/DeepLearning_Classification/assets/108216502/e6e7b53e-8bc7-4746-93c5-5db74d5e4165)

#### 2. Comparison between LeNet-5 and Custom MLP
- LeNet-5
![image](https://github.com/YewonMin/DeepLearning_Classification/assets/108216502/1e8d7a1d-e4ce-4303-8ecf-8353716cade0)
- Custom MLP
![image](https://github.com/YewonMin/DeepLearning_Classification/assets/108216502/aaa67cf7-d61f-4214-a04e-d9ab12ebcbf5)
- 두 모델의 훈련 및 테스트 정확도를 살펴보면, LeNet-5의 train 및 test Acuury가 Custom MLP보다 거의 비슷하거나 항상 높음
- 따라서, 위의 결과를 통해 기본 basic LeNet-5에서는 MNIST 데이터셋에서 Custom MLP보다 더 좋은 예측 성능을 보인다는 것을 알 수 있다.

#### 3.Regularization Techniques for Improving LeNet-5 Model
- Dropout & Weight Decay 적용
- 성능 결과
![image](https://github.com/YewonMin/DeepLearning_Classification/assets/108216502/e6ae8028-31fe-4572-bc74-663d49c535ab)
![lenet5_curves_regularization](https://github.com/YewonMin/DeepLearning_Classification/assets/108216502/3785a5ea-0543-49b6-85d7-339909c629fc)

* 기본적인 LeNet-5 모델에서의 성능은, 학습 및 테스트 과정에서의 손실 값과 정확도가 점차적으로 개선되는 것을 확인할 수 있습니다. 특히, 테스트 정확도는 대략 98.92%에 달함
* Dropout & Weight Decay를 추가해서 정규화 기법을 적용한 모델의 경우, 학습 및 테스트 과정에서의 성능 역시 개선되었음. 특히, 손실 값은 약간 증가하였지만, 이는 모델이 더 일반화되었음을 나타냄. 또한, 테스트 정확도는 약 99.17%로 더 높은 값을 보여줌. 이는 정규화 기법이 모델의 과적합을 방지하고 일반화 성능을 향상시킨 결과로 해석됨
* 따라서, 정규화 기법을 적용함으로써 모델의 성능이 개선되었으며, 특히 테스트 정확도에서 뚜렷한 향상을 보임. 결과적으로, 모델의 안정성과 일반화 능력이 향상되었음을 알 수 있음.
