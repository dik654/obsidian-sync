# Step 47: Softmax와 Cross-Entropy (회귀 → 분류)

## 회귀 vs 분류

step46까지는 **회귀(regression)**: 연속적인 실수값을 예측 ($y = \sin(2\pi x)$). step47부터는 **분류(classification)**: 입력이 어떤 클래스에 속하는지 예측.

||회귀|분류|
|---|---|---|
|출력|실수값 (예: 3.14)|각 클래스의 확률 (예: [0.1, 0.7, 0.2])|
|손실 함수|MSE|Cross-Entropy|
|출력 변환|없음|Softmax (확률로 변환)|

분류에는 두 가지 새 도구가 필요하다:

1. **Softmax**: 모델의 원시 출력을 확률로 변환
2. **Cross-Entropy**: "확률이 정답에 얼마나 가까운가"를 측정하는 손실 함수

---

## 1. Exp 함수

### Forward

$$y = e^x$$

지수 함수. 모든 실수를 양수로 변환한다 ($e^x > 0$). softmax에서 음수를 양수로 바꾸기 위해 필요.

### Backward 유도

$y = e^x$이므로 미분:

$$\frac{dy}{dx} = e^x = y$$

지수 함수는 **미분해도 자기 자신**이 되는 유일한 함수 계열이다.

> **왜?** $y = e^x$의 정의: $e = \lim_{n \to \infty}(1 + \frac{1}{n})^n$에서 출발. 미분의 정의로 계산하면: $$\frac{d}{dx}e^x = \lim_{h \to 0}\frac{e^{x+h} - e^x}{h} = e^x \cdot \lim_{h \to 0}\frac{e^h - 1}{h} = e^x \cdot 1 = e^x$$ ($\lim_{h \to 0}\frac{e^h - 1}{h} = 1$은 $e$의 정의에서 나오는 성질)

체인룰 적용:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{dy}{dx} = g_y \cdot e^x$$

### Rust 구현

```rust
struct ExpFn;
impl Function for ExpFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(f64::exp)]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        vec![&gys[0] * &exp(&xs[0])]  // gy * exp(x)
    }
}
```

---

## 2. Log 함수

### Forward

$$y = \ln(x)$$

자연로그. Cross-entropy에서 $-\log(p)$를 계산하기 위해 필요.

### Backward 유도

$y = \ln(x)$이므로 $e^y = x$. 양변을 $x$로 미분하면:

$$e^y \cdot \frac{dy}{dx} = 1$$

$$\frac{dy}{dx} = \frac{1}{e^y} = \frac{1}{x}$$

체인룰 적용:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{1}{x} = \frac{g_y}{x}$$

### Rust 구현

```rust
struct LogFn;
impl Function for LogFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(f64::ln)]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        vec![&gys[0] / &xs[0]]  // gy / x
    }
}
```

---

## 3. Softmax

### 왜 필요한가

모델의 원시 출력(logit)은 $[-\infty, +\infty]$ 범위의 임의의 실수: $$\mathrm{logit} = [2.1,\ -0.3,\ 0.8]$$

이것을 "확률"로 해석하려면 세 가지 조건이 필요하다:

1. 각 값이 $\geq 0$
2. 각 값이 $\leq 1$
3. 전체 합이 $= 1$

### 공식의 유도

**아이디어**: $e^x > 0$이므로 지수 함수로 양수를 만들고, 전체 합으로 나누면 합이 1.

$$p_k = \frac{e^{x_k}}{\sum_{j=1}^{C} e^{x_j}}$$

**세 조건 검증**:

- $e^{x_k} > 0$이므로 $p_k > 0$ ✓
- 분자 $e^{x_k}$는 분모 $\sum_j e^{x_j}$의 일부이므로 $p_k \leq 1$ ✓
- $\sum_k p_k = \frac{\sum_k e^{x_k}}{\sum_j e^{x_j}} = 1$ ✓

**예시**: $$[2.1,\ -0.3,\ 0.8] \xrightarrow{\mathrm{softmax}} [0.72,\ 0.07,\ 0.21]$$

큰 값(2.1)이 가장 높은 확률(0.72)을 받고, 작은 값(-0.3)이 가장 낮은 확률(0.07)을 받는다. 크기 **순서가 보존**되면서 확률 분포가 된다.

### 배치 Softmax

입력이 $(N, C)$ 행렬일 때, **각 행(샘플)마다 독립적**으로 softmax 적용:

$$p_{ik} = \frac{e^{x_{ik}}}{\sum_{j=1}^{C} e^{x_{ij}}}$$

### Rust 구현 (기존 연산 조합)

```rust
pub fn softmax_simple(x: &Variable) -> Variable {
    let e = exp(x);                          // (N, C) element-wise exp
    let s = sum_with(&e, Some(1), true);     // axis=1, keepdims → (N, 1)
    &e / &s                                  // broadcast: (N, C) / (N, 1)
}
```

`exp`, `sum_with`, `div`의 조합이므로 **역전파는 자동 처리**된다 — 각 연산이 이미 backward를 가지고 있고, 체인룰로 자동 전파.

---

## 4. Cross-Entropy 손실

### 왜 필요한가

분류에서 정답이 클래스 2일 때, $p_2$(클래스 2의 예측 확률)가 1에 가까울수록 좋다. 이걸 **손실**로 표현하려면:

- $p_2 = 1.0$ → loss = 0 (완벽한 예측)
- $p_2 = 0.01$ → loss 큼 (나쁜 예측)

### 공식의 유도

**정보 이론적 동기**: 확률이 $p$인 사건의 "놀라움(surprise)"은 $-\log(p)$:

|확률 $p$|$-\log(p)$|해석|
|---|---|---|
|1.0|0|확실한 사건, 놀랍지 않음|
|0.5|0.69|반반, 약간 놀라움|
|0.01|4.61|드문 사건, 매우 놀라움|

> **왜 $-\log$인가?**
> 
> 1. $p = 1$이면 놀라움 = 0 → $-\log(1) = 0$ ✓
> 2. $p$가 작을수록 놀라움이 커야 함 → $-\log$는 단조감소이므로 ✓
> 3. 독립 사건의 놀라움은 더해져야 함 → $-\log(p_1 \cdot p_2) = -\log(p_1) - \log(p_2)$ ✓
> 
> 이 세 가지 공리를 만족하는 유일한 함수가 $-\log(p)$이다 (Shannon, 1948).

정답 클래스 $t_i$에 대해 모델이 예측한 확률이 $p_{i,t_i}$이므로, 놀라움의 평균:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i,t_i}$$

- $N$: 배치 크기
- $t_i$: 샘플 $i$의 정답 클래스
- $p_{i,t_i}$: 샘플 $i$에서 정답 클래스의 softmax 확률

---

## 5. Fused Softmax Cross-Entropy

### 왜 합치는가 — 수치 안정성

별도로 계산하면 두 가지 문제가 발생한다:

**문제 1: Exp 오버플로우** $$e^{1000} = \infty \quad (\mathrm{f64\ overflow})$$

**해결: Max 빼기 트릭**

softmax에 상수 $m$을 빼도 결과가 같다는 성질을 이용:

$$\frac{e^{x_k - m}}{\sum_j e^{x_j - m}} = \frac{e^{x_k} \cdot e^{-m}}{\sum_j e^{x_j} \cdot e^{-m}} = \frac{e^{x_k}}{\sum_j e^{x_j}}$$

> **유도**: 분자와 분모에 같은 $e^{-m}$이 곱해지므로 약분된다.

$m = \max_j(x_j)$로 설정하면 지수의 최댓값이 0: $$e^{x_k - \max(x)} \leq e^0 = 1$$

어떤 큰 수가 와도 오버플로우가 발생하지 않는다.

**문제 2: log(0)**

softmax 출력이 0에 매우 가까울 때 $\log(0) = -\infty$. `max(p, 1e-15)`로 클리핑하여 방지.

### Forward 구현

```rust
fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
    // 1. 수치 안정 softmax: 각 행에서 max를 빼고 exp
    for i in 0..n {
        let max_val = x.row(i).max();           // overflow 방지
        softmax[i] = exp(x[i] - max_val);
        softmax[i] /= softmax[i].sum();
    }
    // 2. cross-entropy: 정답 클래스의 -log(확률)의 평균
    for i in 0..n {
        loss -= softmax[[i, t[i]]].max(1e-15).ln();  // log(0) 방지
    }
    loss /= n;
}
```

---

## 6. Softmax Cross-Entropy Backward 유도

step47에서 가장 핵심적인 수학. 전용 Function struct로 구현하는 이유가 바로 이 우아한 역전파 공식을 직접 사용하기 위함이다.

### 목표

logit $x_{ik}$에 대한 손실 $L$의 기울기 $\frac{\partial L}{\partial x_{ik}}$를 구한다.

### 단계 1: Softmax의 편미분

$p_k = \frac{e^{x_k}}{\sum_j e^{x_j}}$를 $x_m$으로 미분한다.

**경우 1: $k = m$ (같은 인덱스)**

몫의 미분법. $f = e^{x_k}$, $g = \sum_j e^{x_j}$:

$$\frac{\partial p_k}{\partial x_k} = \frac{f'g - fg'}{g^2} = \frac{e^{x_k} \cdot g - e^{x_k} \cdot e^{x_k}}{g^2}$$

$$= \frac{e^{x_k}}{g} - \frac{e^{x_k}}{g} \cdot \frac{e^{x_k}}{g} = p_k - p_k^2 = p_k(1 - p_k)$$

**경우 2: $k \neq m$ (다른 인덱스)**

분자 $e^{x_k}$는 $x_m$과 무관하므로 $f' = 0$:

$$\frac{\partial p_k}{\partial x_m} = \frac{0 - e^{x_k} \cdot e^{x_m}}{g^2} = -\frac{e^{x_k}}{g} \cdot \frac{e^{x_m}}{g} = -p_k \cdot p_m$$

**통합** (크로네커 델타 $\delta_{km}$):

$$\frac{\partial p_k}{\partial x_m} = p_k(\delta_{km} - p_m)$$

### 단계 2: Cross-Entropy와 결합

단일 샘플: $\ell = -\log p_t$ (정답 클래스 $t$). $x_m$으로 미분:

$$\frac{\partial \ell}{\partial x_m} = -\frac{1}{p_t} \cdot \frac{\partial p_t}{\partial x_m}$$

단계 1에서 $k = t$를 대입:

$$\frac{\partial p_t}{\partial x_m} = p_t(\delta_{tm} - p_m)$$

대입하면:

$$\frac{\partial \ell}{\partial x_m} = -\frac{1}{p_t} \cdot p_t(\delta_{tm} - p_m) = -(\delta_{tm} - p_m) = p_m - \delta_{tm}$$

### 단계 3: 결과

$$\frac{\partial \ell}{\partial x_m} = p_m - \delta_{tm}$$

벡터로 쓰면:

$$\nabla_x \ell = \mathbf{p} - \mathrm{onehot}(t)$$

**예시**: 3클래스, 정답이 클래스 1, softmax 출력이 $[0.3, 0.5, 0.2]$이면: $$\nabla_x \ell = [0.3,\ 0.5,\ 0.2] - [0,\ 1,\ 0] = [0.3,\ -0.5,\ 0.2]$$

- 정답 클래스(1): 기울기 $= p_1 - 1 = -0.5$ → 해당 logit을 **키우는** 방향
- 오답 클래스(0, 2): 기울기 $= p_k > 0$ → 해당 logit을 **줄이는** 방향

### 단계 4: 배치 평균

$$L = \frac{1}{N}\sum_{i} \ell_i$$이므로:

$$\nabla_x L = \frac{\mathbf{P} - \mathrm{onehot}(\mathbf{t})}{N}$$

### Backward 구현

```rust
fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
    // 1. softmax 재계산 (수치 안정 버전)
    // 2. (softmax - one_hot): 정답 위치에서만 1을 빼면 끝
    for i in 0..n {
        softmax[[i, self.t[i]]] -= 1.0;
    }
    // 3. N으로 나누기
    let gx = softmax.mapv(|v| v / n as f64);
    // 4. upstream gradient 곱하기
    let gx = gx.mapv(|v| v * gy_val);
}
```

one_hot 벡터를 명시적으로 만들 필요 없이, `softmax[[i, t[i]]] -= 1.0` 한 줄로 $\mathbf{P} - \mathrm{onehot}(\mathbf{t})$가 완성된다.

---

## 7. `softmax_simple` vs `softmax_cross_entropy_simple`

||`softmax_simple`|`softmax_cross_entropy_simple`|
|---|---|---|
|구현|기존 연산 조합 (exp, sum, div)|전용 Function struct|
|역전파|자동 (체인룰)|수동 (유도한 공식 직접 사용)|
|수치 안정|max 빼기 없음|max 빼기 + log(0) 방지|
|용도|확률값을 직접 볼 때|학습용 손실 계산|

---

# Step 48: 미니배치 학습과 Spiral 데이터셋

## step47 → step48의 전환

||step47|step48|
|---|---|---|
|목적|softmax, cross-entropy **함수 구현**|실제 데이터셋으로 **모델 훈련**|
|데이터|임의 입력으로 단일 forward/backward 검증|Spiral 데이터셋 300개로 300에폭 학습|
|학습 방식|없음 (검증만)|미니배치 + 에폭 + 셔플|

---

## 1. Spiral 데이터셋

### 구조

3개 클래스가 각각 원점에서 바깥으로 뻗어나가는 **나선 팔(spiral arm)** 을 형성하는 2D 데이터.

- 클래스 수: 3
- 클래스당 샘플: 100개
- 총 샘플: 300개
- 입력 차원: 2 (x, y 좌표)

### 생성 공식

클래스 $j$의 $i$번째 점:

$$r = \frac{i}{100}$$

$$\theta = 4j + 4 \cdot \frac{i}{100} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.2^2)$$

$$\mathbf{x} = (r\sin\theta,\ r\cos\theta)$$

> **유도/의도**:
> 
> - $r = i/100$: 원점에서 점점 멀어짐 (나선이 바깥으로 뻗어감)
> - $4j$: 클래스마다 시작 각도를 $4$ 라디안씩 오프셋 → 3개 나선이 $120°$씩 떨어져 배치
> - $4 \cdot \frac{i}{100}$: 바깥으로 갈수록 각도가 증가 → 나선 형태
> - $\epsilon$: 정규분포 노이즈로 약간의 흩어짐 추가 → 현실적인 데이터

### 왜 이 데이터셋인가

- 직선 하나로는 분리 불가 (**비선형 결정 경계** 필요)
- MLP + sigmoid 활성화가 이런 비선형 경계를 학습할 수 있음을 보여주는 대표적 예제

### Rust 구현 (`lib.rs`에 추가)

```rust
pub fn get_spiral(train: bool) -> (ArrayD<f64>, Vec<usize>) {
    let seed: u64 = if train { 1984 } else { 2020 };
    let mut state: u64 = seed;
    // ...
    for j in 0..num_class {
        for i in 0..num_data {
            let rate = i as f64 / num_data as f64;
            let radius = 1.0 * rate;
            // Box-Muller: 균일분포 → 정규분포
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let theta = j as f64 * 4.0 + 4.0 * rate + noise * 0.2;
            x[ix] = (radius * sin(theta), radius * cos(theta));
            t[ix] = j;
        }
    }
}
```

---

## 2. 미니배치 학습

### 왜 미니배치가 필요한가

**풀 배치 (전체 데이터를 한 번에)**:

- 300개 전부로 gradient 계산 → 1에폭에 1번 업데이트
- 정확한 gradient지만 업데이트 빈도가 낮음
- 데이터가 수백만 개면 메모리에 안 올라감

**미니배치 (작은 단위로 나눠서)**:

- 30개씩 10번 → 1에폭에 10번 업데이트
- gradient에 노이즈가 있지만, 이 노이즈가 오히려 로컬 미니마를 탈출하는 데 도움
- 메모리 사용량 제한 가능

### 용어 정리

|용어|의미|step48의 값|
|---|---|---|
|데이터 크기|전체 샘플 수|300|
|배치 크기|한 번에 처리하는 샘플 수|30|
|이터레이션|한 에폭 내 업데이트 횟수|$\lceil 300/30 \rceil = 10$|
|에폭|전체 데이터를 한 바퀴 도는 단위|300|
|총 업데이트|에폭 × 이터레이션|3000|

---

## 3. 데이터 셔플

### 왜 셔플하는가

셔플 없이 매 에폭 같은 순서로 배치를 구성하면:

- 배치 1은 항상 샘플 0~~29, 배치 2는 항상 30~~59, ...
- 특정 배치의 구성이 편향될 수 있음 (예: 한 배치에 클래스 0만 집중)
- 모델이 **배치 순서에 대한 패턴**을 학습할 위험

셔플하면:

- 매 에폭마다 배치 구성이 바뀜
- 모델이 다양한 조합을 경험 → **일반화** 성능 향상

### Fisher-Yates 셔플

배열을 균일하게 랜덤 섞는 알고리즘. Python의 `np.random.permutation`에 해당.

```
[0, 1, 2, 3, 4] 에서 시작
i=4: j=rand(0..4) → swap(4, j)
i=3: j=rand(0..3) → swap(3, j)
i=2: j=rand(0..2) → swap(2, j)
i=1: j=rand(0..1) → swap(1, j)
```

> **왜 이것이 균일 분포인가?** 위치 $n-1$에 올 수 있는 원소: $n$개 중 하나 (확률 $1/n$) 위치 $n-2$에 올 수 있는 원소: 남은 $n-1$개 중 하나 (확률 $1/(n-1)$) 특정 순열이 나올 확률: $\frac{1}{n} \cdot \frac{1}{n-1} \cdots \frac{1}{1} = \frac{1}{n!}$ 모든 $n!$개 순열이 동일한 확률 → **균일 분포** ✓

```rust
fn shuffle(indices: &mut [usize], rng: &mut SimpleRng) {
    let n = indices.len();
    for i in (1..n).rev() {
        let j = (rng.next_f64() * (i + 1) as f64) as usize;
        indices.swap(i, j);
    }
}
```

---

## 4. 학습 루프 구조

```
for epoch in 0..300 {
    인덱스 셔플: [0..299] → 무작위 순서

    for batch in 0..10 {
        ① 미니배치 추출:  x[batch_index], t[batch_index]
        ② 순전파:         y = model.forward(batch_x)
        ③ 손실 계산:      loss = softmax_cross_entropy(y, batch_t)
        ④ 기울기 초기화:  model.cleargrads()
        ⑤ 역전파:         loss.backward()
        ⑥ 파라미터 업데이트: optimizer.update()  // SGD: p ← p - lr × grad
    }

    에폭 평균 손실 출력
}
```

### 손실 추적 방식

각 배치의 손실을 배치 크기로 가중 합산하고, 에폭 끝에 전체 데이터 수로 나눔:

$$\mathrm{avg_loss} = \frac{\sum_{b} L_b \times |B_b|}{N}$$

> **왜 단순 평균이 아닌가?** 마지막 배치가 다른 배치보다 작을 수 있다 (300이 배치 크기로 나누어떨어지지 않을 때). 배치 크기로 가중하면 각 **샘플이 동등하게** 반영된다.

---

## 5. 하이퍼파라미터

|파라미터|값|의미|
|---|---|---|
|`max_epoch`|300|전체 데이터를 300바퀴|
|`batch_size`|30|한 번에 30개 처리|
|`hidden_size`|10|은닉층 뉴런 10개|
|`lr`|1.0|학습률 (보통보다 높지만 이 문제에서는 잘 동작)|

모델 구조: **입력 2 → 은닉 10 (sigmoid) → 출력 3**

---

## 6. 학습 결과

```
epoch 1,   loss 1.1576
epoch 31,  loss 0.7531
epoch 91,  loss 0.3749
epoch 151, loss 0.1853
epoch 300, loss 0.0918
Final loss: 0.0815
```

손실이 1.16 → 0.08로 수렴. 3클래스 랜덤 분류의 이론적 손실이 $-\log(1/3) \approx 1.10$이므로, 초기 손실 1.16은 거의 랜덤 수준에서 시작하여 잘 학습되었음을 확인.

---

## 7. 추상화의 흐름 (step43 → step48)

```
step43: 파라미터(W, b) 개별 관리        + MSE (회귀)
step44: Layer로 파라미터 묶기            + MSE (회귀)
step45: Model로 레이어 묶기             + MSE (회귀)
step46: Optimizer로 업데이트 분리        + MSE (회귀)
step47: 분류 도구 도입                  + Softmax + Cross-Entropy
step48: 실전 학습                       + Spiral 데이터셋 + 미니배치 + 셔플
```