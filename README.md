# Machine-Learning-Project-ManuFact
제품제조 품질평가 점수 회귀예측

# 📈 비선형 회귀 분석 프로젝트


## 제품의 품질평가 점수를 예측하는 회귀분석 프로젝트
<img src='https://www.s-ge.com/sites/default/files/styles/sge_header_lg/public/static/images/advanced-manufacturing-industrial-robot-with-conveyor-in-manufacture-factory-concept.jpg?itok=rVis9kL_' width="800px">   

### 📌 데이터 세트 주제  
- 해당 데이터 세트는 다양한 프로세스 매개변수와 제품 품질 간의 관계를 탐색하도록 설계되었습니다.  
- 이러한 제품 제조에 필요한 특성을 나타내는 Feature들을 활용하여 제품의 품질 평가 점수를 예측합니다.

### 📌 Feature별 설명  
- Temperature (°C) : 제조 과정 중 온도를 섭씨 단위로 나타냅니다. 온도는 많은 제조 공정에서 중요한 역할을 하며 재료 특성과 제품 품질에 영향을 미칩니다. 

- Pressure (kPa) :  제조 과정에서 가해지는 압력으로 킬로파스칼(kPa) 단위로 측정됩니다. 압력은 재료 변형과 제조 공정의 전반적인 결과에 영향을 미칠 수 있습니다.  
 
- Temperature x Pressure : 이 기능은 온도와 압력의 상호작용 Feature로, 두 공정 매개변수의 결합 효과를 포착합니다.  

- Material Fusion Metric (재료 융합 지표) : 온도의 제곱과 압력의 세제곱의 합으로 계산되는 파생 지표입니다. 이는 제조 공정 중 재료 융합 관련 측정을 나타냅니다.  

- Material Transformation Metric (재료 변환 지표) : 온도의 세제곱에서 압력의 제곱을 뺀 값으로 계산되는 또 다른 파생 지표입니다. 이는 재료 변형 역학에 대한 통찰력을 제공합니다. **(Target Data)**

- Quality Rating : 독립변수(Target Data)인 '품질평가'는 생산된 품목의 전반적인 품질평가를 나타냅니다. 품질은 제조에 있어서 중요한 측면이며, 이 등급은 최종 제품의 품질을 측정하는 척도로 사용됩니다. **Target Data**

  ### ✏️ 제품 제조 품질 평가 점수 예측 회귀 프로젝트 진행 방향성
  - [데이터 전처리 (결측치, 중복된 데이터, 이상치 등 제거 및 일반화 작업 , 필요없는 Feature 제거)](#전처리-작업)
  - [독립변수와 종속변수들의 상관관계 확인](#correlation-종속변수와의-상관관계-분석)
  - [회귀 분석 실시](#📌-전처리-완료)
  - [1 Cycle - 회귀모델들의 성능 분석 & 선형 vs 비선형 데이터 판별](#1-Cycle)
  - [2 Cycle - OLS 회귀분석 & 다중공선성 확인](#2-Cycle)
  - [3 Cycle - 교차검증으로 과적합 해소](#3-Cycle)
  - [4 Cycle - 회귀모델 데이터 회귀 분석 및 기존 데이터세트와 성능 비교](#4-Cycle)  
  - [5 Cycle - Target Data 분포 조정 후 다시 회귀분석](#5-Cycle)  
  - [6 Cycle - 다중공선성 해소 없이 차원축소로 회귀분석](#6-Cycle)
  - [최종 결론](#Total-Result)

## 데이터세트(csv파일 PNG) <ManuFact Quality Rating Predict>
## chapter1
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/502153ea-8ec5-44cc-b0c6-70b640c5a7ab' width="600px">  

[최종결론으로 이동](#Final-Chapter)

## 전처리 작업
- ✏️ 결측치 및 중복된 데이터가 있는지 확인 후, 존재하지 않았기 때문에, 따로 작업하지 않았습니다.
- ✏️ 그러나 온도와 압력에 관련된 Feature가 존재하였는데, 온도와 압력의 상호작용을 나타내는 Feature가 따로 존재하였기 때문에, 온도와 압력 Feature의 역할을 대체할 수 있을 것이라 판단되어 해당 두 개의 Feature를 제거 하였습니다.
- ✏️ 이상치가 존재하였으므로, 표준화 작업 후 이상치가 존재하는 데이터를 제거해주었습니다.  

## 해당 데이터셋의 Feature's 분포 확인 (히스토그램)  
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/4566ec7e-cbea-44ba-b4f5-187476b10968' width="600px">    

### 히스토그램  

- 해당 데이터 특성상 품질평가 점수의 분포가 99~100 사이의 점수에 매우 많이 분포되어 있기 때문에, 이러한 분포가 나왔습니다.
- 로그치환 , PowerTransformer(yeo-johnson) 를 활용하여 분포 조정 시행  
- 5 Cycle에서 Target 데이터에 대한 분포를 조정하겠습니다. 
[5 Cycle로 이동](#5-Cycle)

## correlation 종속변수와의 상관관계 분석
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/faf4f4e8-e735-49e3-973f-cca650957ee5' width="600px">    

위 이미지와 같이 세개의 Feature 모두 타겟 데이터에 대해 음의 상관관계를 띄고 있으며 그 중에서도 재료 변환 지표가 가장 상관관계가 높았으며, 가장 상관관계가 낮은 Feature는 온도와 압력의 상호작용을 나타내는 Feature의 상관관계가 가장 낮았습니다.

## Heatmap (각 컬럼들의 상관관계 확인)
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/7b01dc7c-000e-46d9-93ec-e0297b14c0a1' width="600px">  

히트맵을 통해 컬럼들의 상관관계를 확인 해 본 결과 재료의 융합지표와 변환 지표 간의 상관관계가 매우 높은 것을 알 수 있으며, 이는 다중공선성에서 문제가 발생할 가능성이 매우 높을 수 있습니다.  
  
👉 [2 Cycle로 이동](#2-Cycle)

## 1 Cycle
> ### 회귀모델들을 사용하여 성능을 평가하고, 독립변수들과 종속변수간의 산점도를 분석하여 총합하여 선형/비선형 데이터 판별
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/b92e22ac-8ad1-4efe-902b-90017a7380fd" width="600px">  

- 독립변수들의 종속변수에 관한 분포를 산점도로 나타낸 이미지 입니다.   
- 위 이미지를 종합해서 보면, 첫번째 온도와압력의 상호작용 Feature의 경우 상단에 몰려있고, 넓게 분포되어 있어, 뚜렷한 선형관계를 보이지 않습니다.
- 나머지 두개의 산점도 역시 대부분 상단에 몰려있으며, 오른쪽으로 갈 수록 분포가 줄어드는 형태를 보입니다. 
- 위 세개의 산점도를 총합하여 분석해보면, 선형관계보다는 비선형쪽에 더 가까운 것으로 판단됩니다.
- 하지만 더 명확한 분석을 위해 각 선형회귀모델(LinearRegression)과 나머지 회귀모델들의 성능을 비교 후 확인 해 보겠습니다.     

> ### LinearRegression (선형회귀분석)  
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/ae8a388f-e35e-47da-a878-d91c2b20d71f" width="600px">  

- 선형회귀모델로 분석한 결과 R2 Score가 0.3450으로 매우 낮게 나온 것을 확인 할 수 있습니다.  
- 이를 미루어 보았을 때 선형 데이터보다는 비선형데이터에 가깝다고 판단 할 수 있습니다. 

> ### Polynomial Regression (다항회귀분석)
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/6735940f-8a10-4bbf-a0c6-af03a28f6da0' width="800px">  

- 데이터의 R² 점수가 다항식의 차수가 증가함에 따라 급격히 증가하고 있으며, 특히 1차항식에서 2차항식으로 넘어갈 때 큰 향상을 보입니다.
- 다항회귀와 같은 경우에도 선형모델인 1차항식에서 데이터의 패턴을 잘 설명하지 못하지만, 고차항으로 올라갈수록 데이터의 패턴을 훨씬 잘 설명할 수 있으므로, 해당 데이터는 비선형에 더 가깝다고 볼 수 있겠습니다.  

> ### 전체 회귀모델 성능 분석 시각화   
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/46d62ecd-231b-43b2-a1c8-bbaa146aefee" width="800px">


### 📃 1 Cycle Result
> Linear Regression 선형 회귀 모델을 제외한 나머지 회귀모델들의 성능이 매우 높은 것을 알 수 있으므로, 해당 데이터셋은 비선형 데이터로 판단 할 수 있습니다.

## 2 Cycle
> ### OLS 회귀분석 & (VIF)다중공선성을 확인 하여 Feature 제거 및 보류 작업 실시  
  
- OLS 이미지
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/e133a155-c4ed-44e4-85d6-6bea6682b91c' width="600px">   
---   

- #### VIF 점수 이미지
<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/aeaaa4e5-f219-42f1-a00b-67c2886a0983' width="600px">  
---    

- pairplot 이미지  

<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/ec626bf9-bd62-4e15-81f4-2ef657c6cdaa' width="600px">  

### 📃 2 Cycle Result  
> 앞서 히트맵에서 봤듯이, 재료 융합 지표와 변환 지표 간의 상관관계가 매우 높음을 알 수 있었고, 다중공선성 문제가 발생할 것으로 예상을 하였습니다.  
   
> pairplot 및 vif 점수를 봐서 알 듯이 재료 융합 지표 Feature의 다중공선성이 심하다는 것을 알 수 있었습니다.  
   
> correlation 결과 타겟데이터의 상관관계에 대해 재료 변환 지표가 융합지표 보다 더 높게 나타 났기 때문에, **재료 융합 지표 Feature를 제거** 후 회귀분석을 진행해보도록 하겠습니다.
 
👉 [독립변수와 종속변수들의 상관관계 확인](#correlation-종속변수와의-상관관계-분석)  

## 3 Cycle
> #### 다중공선성 해소 한 데이터에서 교차검증으로 과적합을 방지하겠습니다.

- ### 📌 재료 융합 지표 Feature 제거 후 OLS 및 VIF 점수 이미지  

- feature 제거 후 OLS 분석 결과  
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/10c3b2d8-5903-4b53-b567-7615579384fd" width="600px">  
  
- feature 제거 후 VIF 점수 결과 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/59e71d4a-c2a7-4081-88e0-3cf97c337c5e" width="600px">  

👉 [제거 전 VIF 이미지 확인](#VIF-점수-이미지)

- 재료 융합 지표 Feature를 제거 후 OLS 확인 결과 R2 Score가 0.954에서 0.818로 떨어졌음을 알 수 있지만, Feature들의 다중공선성 문제는 해결한 모습을 볼 수 있습니다.    
- 우선은 다중공선성 해소한 데이터로 교차검증을 통해 과적합을 방지해보도록 하겠습니다.   
 
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/825709c4-f26c-40e7-9c59-af9b54e813cf" width="600px">    

- 해당 코드에서는 가장 성능이 높게 나왔던 RandomForest회귀모델로 교차 검증을 실시 하였습니다. 
- 훈련 데이터 세트와 검증 데이터 세트를 분리하여 해당 데이터의 손실값(MSE)를 비교하여 과적합을 검증할 수 있습니다.
- `parameters = {'max_depth': [11,12,13], 'min_samples_split': [14,15,16], 'n_estimators': [10, 50, 100]}`   


<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/90e26ef4-1f0d-4994-bddd-99e8e4aaa04c" width="600px">   

하이퍼파라미터 튜닝을 통해서 과적합을 방지하였습니다.
해당 결과 훈련 데이터와 검증데이터 간의 손실값이 크게 발생하지 않았음을 알 수 있습니다. 
  

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/353460c4-c1f0-46f0-9229-a318657fb920" width="900px">    

### 📃 3 Cycle Result

> 과적합 여부 시각화 결과 또한 훈련 데이터에서 예측한 loss 값과 검증데이터에서 예측한 loss 값 차이가 거의 일치하였으므로 해당 데이터에서 과적합을 방지 할 수 있엇습니다.

## 4 Cycle
> #### 다중공선성 Feauture 제거 후 데이터와 원본 데이터 회귀모델들의 분석 성능을 비교   
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/14ecfa06-a0c3-44a4-bae1-2b3bce69799e" width="800px">  

### 📃 4 Cycle Result
> 다중 공선성 제거 후 결과 역시 기존 데이터와 마찬가지로 선형 회귀를 제외한 모든 회귀모델들의 성능이 매우 좋게 나오는 것을 알 수 있습니다.  

## 5 Cycle
> #### Target 데이터 분포 조정 (PowerTransformer , log)  

- 원본 데이터 Target 데이터 분포 히스토그램
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/c9eb842d-1d9c-480b-88ad-e8983f72252d" width="600px">  

- Target 데이터 PowerTransformer(yeo-johnson) 작업 후 분포 히스토그램
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/4f7fce61-54b9-48c5-956c-6882debeb776" width="600px">  

- Target 데이터 로그 치환 작업 후 분포 히스토그램
<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/f4e6b135-6ff8-4b0f-ab67-9aef19613e01" width="600px">  

- 로그 치환의 경우 기존 데이터의 타겟 데이터 분포가 큰 차이가 없기 때문에 따로 작업은 하지 않았습니다.   

<img src='https://github.com/dosel70/MachineLearning-Project/assets/143694489/2f07e586-9f0b-4d28-bf27-c848f25b693d' width="500px">  

> ### 📌 Target data 분포 작업 후 회귀분석(PowerTransformer)과 기존 데이터 회귀분석  

<img src="https://github.com/dosel70/MachineLearning-Project/assets/143694489/5932c145-8d89-4d9c-94e4-328ee07136af" width="600px">  


### 📃 5 Cycle Result
### 💡 Target 데이터의 타겟 데이터에 PowerTransformer 작업 후 회귀모델들로 분석 결과 성능점수(R2 Score)가 1.00 값을 나오는 것을 알 수 있으며, 매우 성능이 좋게 나오는 것을 확인 할 수 있지만, 오히려 일반화 성능을 가지지 못한다고 볼 수 있습니다.  


[처음으로 이동](#chapter1)
  
## Final Chapter
### 📃 Total Score
> 해당 제조 품질점수 예측 데이터에는 과적합이 존재하지 않습니다.

> 해당 데이터는 비선형데이터입니다.  

> Linear Regression 회귀모델을 제외한 모든 회귀모델들의 성능이 매우 좋은 것을 알 수 있으므로 매우 좋은 데이터임을 알 수 있습니다.  

> 가장 성능이 좋은 회귀모델은 RandomForestRegressor , GradientBoostingRegressor 입니다.  
