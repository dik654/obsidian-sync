## MLP (Multi-Layer Perceptron)

### TwoLayerNet의 한계

step45에서 만든 `TwoLayerNet`은 **2층 고정** 구조다:

```rust
struct TwoLayerNet {
    l1: Linear,  // 항상 2개
    l2: Linear,
}
```

3층 네트워크가 필요하면? `ThreeLayerNet`을 새로 만들어야 한다. 10층이면? 비현실적이다.

### MLP: 임의의 층 수를 지원

MLP는 층 수를 **배열로 지정**한다:

```rust
MLP::new(&[10, 1])       // 2층: Linear(10) → σ → Linear(1)
MLP::new(&[20, 10, 1])   // 3층: Linear(20) → σ → Linear(10) → σ → Linear(1)
MLP::new(&[64, 32, 16, 1]) // 4층
```

> 배열의 각 원소 = 해당 층의 **출력 크기** 입력 크기는 lazy initialization으로 자동 결정

### 활성화 함수의 적용 위치

**마지막 층에는 활성화 함수를 적용하지 않는다.**

```
x → [Linear] → sigmoid → [Linear] → sigmoid → [Linear] → 출력
      1층                   2층                   3층(마지막)
      ✅ 활성화             ✅ 활성화             ❌ 활성화 없음
```

이유:

- **회귀 문제**: 출력이 임의의 실수값이어야 함. sigmoid를 적용하면 $(0, 1)$ 범위로 제한됨
- **분류 문제**: 마지막에 softmax 등 별도의 함수를 적용해야 하므로, 모델 자체는 원시 점수(logit)를 출력

### forward 동작 흐름

`MLP::new(&[10, 1])`에 `x` (100, 1)을 넣으면:

|단계|연산|shape|
|---|---|---|
|입력|`x`|(100, 1)|
|1층|`Linear(10).forward(x)` = $x W_1 + b_1$|(100, 10)|
|활성화|`sigmoid(h)`|(100, 10)|
|2층|`Linear(1).forward(h)` = $h W_2 + b_2$|(100, 1)|
|출력|활성화 없이 그대로 반환|(100, 1)|

---

## SGD (Stochastic Gradient Descent) 옵티마이저

### 왜 옵티마이저가 필요한가?

step45까지는 파라미터 업데이트를 **직접 작성**했다:

```rust
for p in model.params() {
    let grad = p.grad().unwrap();
    p.set_data(&p.data() - &grad.mapv(|v| v * lr));
}
```

이 코드에는 **업데이트 규칙**이 하드코딩되어 있다:

$$p \leftarrow p - \text{lr} \times \frac{\partial L}{\partial p}$$

다른 규칙을 쓰고 싶으면? 코드를 직접 수정해야 한다.

### 옵티마이저로 분리

업데이트 규칙을 **옵티마이저 객체**로 캡슐화하면:

```rust
// SGD를 쓸 때
let optimizer = SGD::new(0.2).setup(&model);

// Momentum으로 바꾸고 싶으면 이 한 줄만 변경
// let optimizer = MomentumSGD::new(0.2, 0.9).setup(&model);

// 훈련 루프는 동일
optimizer.update();
```

### SGD 업데이트 규칙

가장 단순한 최적화 알고리즘:

$$p \leftarrow p - \eta \cdot \nabla L$$

- $p$: 파라미터 (W 또는 b)
- $\eta$: 학습률 (learning rate)
- $\nabla L$: 손실에 대한 파라미터의 기울기

> 기울기의 **반대 방향**으로 이동하여 손실을 줄인다.

### 다른 옵티마이저들 (SGD와의 비교)

|옵티마이저|업데이트 규칙|특징|
|---|---|---|
|**SGD**|$p \leftarrow p - \eta \nabla L$|가장 단순. 진동이 심할 수 있음|
|**Momentum**|$v \leftarrow \alpha v - \eta \nabla L$, $p \leftarrow p + v$|관성을 추가. 진동 감소|
|**Adam**|1차/2차 모멘트를 모두 추적|학습률을 파라미터별로 적응적 조절|

SGD는 기울기만 보고 바로 이동하지만, Momentum은 **이전 이동 방향의 관성**을 유지하고, Adam은 **기울기의 크기에 따라 학습률을 자동 조절**한다.

---

## 추상화의 전체 흐름

```
step43: W1, b1, W2, b2  (파라미터 개별 관리)
          ↓
step44: l1(W,b), l2(W,b)  (Layer로 파라미터 묶기)
          ↓
step45: model(l1, l2)     (Model로 레이어 묶기 — TwoLayerNet)
          ↓
step46: MLP(&[10,1])      (범용 모델) + SGD (옵티마이저 분리)
```

|step|cleargrads|파라미터 업데이트|모델 정의|
|---|---|---|---|
|43|`w1.cleargrad(); b1.cleargrad(); ...` (4번)|수동 4번|없음 (변수 나열)|
|44|`l1.cleargrads(); l2.cleargrads();` (2번)|수동 (레이어 루프)|없음 (레이어 나열)|
|45|`model.cleargrads();` (1번)|수동 (모델 루프)|`TwoLayerNet` (직접 정의)|
|46|`model.cleargrads();` (1번)|`optimizer.update()` (1번)|`MLP::new(&[10,1])` (한 줄)|