## Step47: Softmax와 Cross-Entropy (분류 문제)

### 회귀 vs 분류

step46까지는 **회귀(regression)** 문제만 다뤘다:

||회귀 (step43~46)|분류 (step47~)|
|---|---|---|
|**목표**|연속적인 값 예측|어떤 클래스에 속하는지 예측|
|**출력 예시**|`3.14` (실수 하나)|`[0.1, 0.7, 0.2]` (클래스별 확률)|
|**손실 함수**|MSE: $\frac{1}{N}\sum(y - \hat{y})^2$|Cross-Entropy: $-\frac{1}{N}\sum\log p_{t_i}$|
|**데이터 예시**|$y = \sin(2\pi x)$|이미지 → 고양이/개/새|

---

## Softmax

### 왜 필요한가?

모델의 원시 출력(logit)은 아무 범위의 실수값이다:

$$\text{logit} = [2.1,; -0.3,; 0.8]$$

이대로는 "확률"이 아니다 (합이 1이 아니고, 음수도 있음). **softmax**가 이것을 확률로 변환한다.

### 공식

$$p_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

### 변환 과정 예시

$$[2.1,; -0.3,; 0.8]$$

$$\downarrow \text{exp}$$

$$[e^{2.1},; e^{-0.3},; e^{0.8}] = [8.17,; 0.74,; 2.23]$$

$$\downarrow \div \text{sum}(= 11.14)$$

$$[0.73,; 0.07,; 0.20]$$

결과의 성질:

- 모든 값이 $(0, 1)$ 범위
- 합이 정확히 $1$
- 가장 큰 logit($2.1$)이 가장 높은 확률($0.73$)

### 구현: 기존 연산의 조합

```rust
pub fn softmax_simple(x: &Variable) -> Variable {
    let e = exp(x);                        // exp(x): (N, C)
    let s = sum_with(&e, Some(1), true);   // 행별 합: (N, 1)
    &e / &s                                // broadcast: (N, C) / (N, 1)
}
```

`exp`, `sum_with`, `div`는 모두 이미 backward가 구현되어 있으므로, softmax의 **backward는 자동으로 처리**된다.

---

## Cross-Entropy

### 왜 필요한가?

분류 문제에서 "모델이 얼마나 틀렸는지"를 측정해야 한다.

정답이 클래스 2일 때, 모델의 softmax 출력이:

- $p = [0.1, 0.2, \mathbf{0.7}]$ → 정답 클래스의 확률이 높음 → 좋음
- $p = [0.5, 0.4, \mathbf{0.1}]$ → 정답 클래스의 확률이 낮음 → 나쁨

### 공식

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log p_{t_i}$$

- $N$: 샘플 수
- $t_i$: $i$번째 샘플의 정답 클래스 인덱스
- $p_{t_i}$: softmax 출력에서 정답 클래스의 확률

### $-\log$의 직관

|$p_{t_i}$ (정답 확률)|$-\log(p_{t_i})$ (손실)|의미|
|---|---|---|
|$1.0$|$0$|완벽한 예측|
|$0.7$|$0.36$|괜찮은 예측|
|$0.1$|$2.30$|나쁜 예측|
|$0.01$|$4.61$|매우 나쁜 예측|

> 정답 확률이 낮을수록 손실이 **급격히** 증가한다.

### MSE와의 비교

분류에서 MSE 대신 cross-entropy를 쓰는 이유:

||MSE|Cross-Entropy|
|---|---|---|
|**확률이 0.01일 때**|$(1 - 0.01)^2 = 0.98$|$-\log(0.01) = 4.61$|
|**확률이 0.001일 때**|$(1 - 0.001)^2 = 0.998$|$-\log(0.001) = 6.91$|
|**기울기 크기**|거의 변화 없음|크게 변화|

MSE는 확률이 아주 나쁠 때도 기울기가 작아서 학습이 느리다. Cross-entropy는 $-\log$의 특성으로 나쁜 예측에 **강한 페널티**를 주어 빠르게 교정한다.

---

## Softmax Cross-Entropy (Fused)

### 왜 합치는가?

softmax와 cross-entropy를 **분리**해서 계산하면 수치 문제가 생긴다:

```
x = [100, 200, 300]
exp(300) = 1.9 × 10^130  → overflow!
```

**합쳐서** 계산하면 안전하다:

$$-\log\left(\frac{e^{x_{t_i}}}{\sum_j e^{x_j}}\right) = -x_{t_i} + \log\sum_j e^{x_j}$$

그리고 $\log\sum e^{x_j}$는 최댓값 $m$을 빼서 안정화:

$$\log\sum_j e^{x_j} = m + \log\sum_j e^{x_j - m}$$

> $e^{x_j - m}$은 최댓값이 $e^0 = 1$이므로 overflow 없음

### Forward 구현

```
1. 각 행에서 max를 뺀다 (overflow 방지)
2. exp → sum → 나누기 = softmax
3. 정답 클래스의 -log(확률)의 평균 = loss
```

### Backward

softmax cross-entropy의 기울기는 놀라울 정도로 단순하다:

$$\frac{\partial L}{\partial x_i} = \frac{p_i - \mathbb{1}_{i = t}}{N}$$

- $p_i$: softmax 확률
- $\mathbb{1}_{i=t}$: 정답 클래스이면 1, 아니면 0 (one-hot)
- $N$: 샘플 수

> **의미**: 기울기 = (예측 확률 - 정답). 정답 클래스의 확률이 1에 가까우면 기울기가 0에 가깝고 (이미 잘 맞추고 있으니 변할 필요 없음), 0에 가까우면 기울기가 크다 (많이 틀렸으니 크게 수정).

### 구체적 예시

정답 클래스 = 2, softmax 출력 = $[0.1, 0.2, 0.7]$:

$$\text{gradient} = \frac{[0.1 - 0,; 0.2 - 0,; 0.7 - 1]}{N} = \frac{[0.1,; 0.2,; -0.3]}{N}$$

- 클래스 0, 1: 양의 기울기 → 점수를 **낮추는** 방향
- 클래스 2 (정답): 음의 기울기 → 점수를 **높이는** 방향

---

## 새로 추가된 함수 정리

### exp

$$y = e^x, \quad \frac{dy}{dx} = e^x$$

> 미분해도 자기 자신. softmax 계산에 사용.

### log

$$y = \ln x, \quad \frac{dy}{dx} = \frac{1}{x}$$

> cross-entropy의 $-\log p$ 계산에 사용.

### softmax_simple

기존 연산 조합 → backward 자동:

$$p = \frac{\exp(x)}{\text{sum}(\exp(x), \text{axis}=1)}$$

### softmax_cross_entropy_simple

fused 구현 (수치 안정 + 효율적 backward):

$$L = -\frac{1}{N}\sum_i \log p_{t_i}, \quad \nabla_x L = \frac{p - \text{one_hot}(t)}{N}$$