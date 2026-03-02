## Step 59: RNN, Truncated BPTT, Adam 옵티마이저

### 한마디 직관: CNN vs RNN vs LSTM

- **CNN = 패턴 탐지기** — "이 데이터에 이 형태가 있는가?"
- **RNN = 단기 요약기** — "지금까지 본 것을 종합하면 다음은 무엇인가?"
- **LSTM = 선택적 장기 요약기** — 중요한 건 **더하고**, 불필요한 건 **잊고**, 필요한 것만 **꺼냄**

RNN/LSTM의 핵심은 매 시간 스텝마다 입력을 받아서 **지금까지의 모든 정보를 하나의 벡터 $h$로 압축**하는 것이다:

```
x₀ → [압축] → h₁  ("x₀을 요약한 벡터")
x₁ → [압축] → h₂  ("x₀, x₁을 요약한 벡터")
x₂ → [압축] → h₃  ("x₀, x₁, x₂를 요약한 벡터")
```

RNN은 $h$를 매번 통째로 덮어쓰므로 오래된 정보가 빠르게 사라진다. LSTM은 별도의 셀 상태 $c$에 선택적으로 기억/망각하여 오래된 정보도 보존할 수 있다. (→ [[step 60 LSTM]])

---

### CNN에서 RNN으로

Step 57까지는 **CNN** — 공간적 패턴(이미지의 특징)을 다뤘다. Step 59부터는 **RNN** — 시간적 패턴(시퀀스, 시계열)을 다룬다.

피드포워드 네트워크(MLP, CNN)는 입력과 출력이 독립적이다. 이전 입력이 다음 출력에 영향을 주지 않는다. 하지만 시계열 데이터(주가, 자연어, 음성)는 이전 정보가 다음 예측에 필수적이다.

---

### RNN (Recurrent Neural Network)

핵심 아이디어: **은닉 상태(hidden state)** $h$가 시간 스텝 간에 전달되어 "기억"을 유지한다.

$$h_t = \tanh(x_t W_x + b + h_{t-1} W_h)$$

- $x_t$: 현재 시점의 입력 (shape: batch $\times$ input\_dim)
- $h_{t-1}$: 이전 시점의 은닉 상태 (shape: batch $\times$ hidden\_size)
- $W_x$: 입력 → 은닉 가중치, $W_h$: 은닉 → 은닉 가중치
- $\tanh$: 출력을 $[-1, 1]$로 제한하여 값의 폭발을 방지

시간 전개(unfolding)하면 일반적인 피드포워드 네트워크와 동일한 구조가 된다:

```
x₀ → [RNN] → h₁ → [Linear] → y₁
       ↓
x₁ → [RNN] → h₂ → [Linear] → y₂
       ↓
x₂ → [RNN] → h₃ → [Linear] → y₃
```

$h$가 `Variable`이므로 계산 그래프가 시간 스텝을 넘어 자동 연결된다. 이 연결을 통해 역전파가 시간을 거슬러 전파된다.

#### tanh를 쓰는 이유

활성화 함수 없이 $h_t = x_t W_x + h_{t-1} W_h$이면, $h$의 크기가 시간 스텝마다 기하급수적으로 증가할 수 있다. $\tanh$는 출력을 $[-1, 1]$로 제한하여:

1. **값의 폭발 방지**: 수치적 안정성 확보
2. **비선형성 도입**: 선형만으로는 여러 층을 쌓아도 하나의 선형 변환과 동일
3. **0 중심 출력**: sigmoid($[0,1]$)과 달리 양수/음수 모두 출력 → 기울기 편향이 적음

단, $\tanh$의 도함수 $1 - \tanh^2(x)$는 최대 1이고 $|x|$가 크면 0에 가까워진다. 이것이 기울기 소실의 원인 중 하나이다 (→ [[step 60 LSTM]]).

#### RNN 구현

```rust
pub struct RNN {
    x2h: Linear,                          // x → h 변환 (bias 포함)
    w_h: RefCell<Option<Variable>>,       // h → h 가중치 (lazy init)
    h: RefCell<Option<Variable>>,         // 현재 은닉 상태
    hidden_size: usize,
}
```

순전파에서 핵심은 **첫 스텝과 이후 스텝의 분기**:

```rust
pub fn forward(&self, x: &Variable) -> Variable {
    let h_new = if self.h.borrow().is_none() {
        // 첫 스텝: h가 없으므로 x→h만 사용
        tanh(&self.x2h.forward(x))
    } else {
        // 이후 스텝: x2h(x) + h @ W_h
        let h = self.h.borrow().clone().unwrap();
        let w_h = self.w_h.borrow().as_ref().unwrap().clone();
        tanh(&(&self.x2h.forward(x) + &matmul(&h, &w_h)))
    };
    *self.h.borrow_mut() = Some(h_new.clone());
    h_new
}
```

**설계 포인트:**
- `w_h`는 `RefCell<Option<Variable>>`로 **lazy init** — 첫 forward 호출 시 Xavier 초기화
- `x2h`는 `Linear` 레이어(bias 포함)이지만, `w_h`는 별도의 `Variable` — `h→h` 변환에 bias를 추가하면 이중 bias가 됨
- `h`는 `Variable`이므로 **계산 그래프가 시간 스텝을 넘어 자동 연결**됨 — 별도의 "연결" 로직 불필요

---

### BPTT (Backpropagation Through Time)

RNN을 시간 축으로 펼치면 일반적인 피드포워드 네트워크와 동일하므로, 일반 역전파를 그대로 적용할 수 있다. 이를 **BPTT**라 한다.

#### 기울기 전파의 수학

$h_t = \tanh(x_t W_x + h_{t-1} W_h)$에서 $\frac{\partial h_t}{\partial h_{t-1}}$을 chain rule로 구하면:

$$\frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}(1 - h_t^2) \cdot W_h$$

여기서 $\mathrm{diag}(1 - h_t^2)$는 $\tanh$의 도함수를 대각 행렬로 만든 것이다.

시간 $T$에서의 손실이 시간 $0$의 은닉 상태에 미치는 영향:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \prod_{t=1}^{T} \mathrm{diag}(1 - h_t^2) \cdot W_h$$

$W_h$가 $T$번 곱해진다. $W_h$의 최대 특이값(singular value) $\sigma_{\max}$에 따라:

- $\sigma_{\max} < 1$: $\|W_h^T\| \to 0$ (기울기 소실)
- $\sigma_{\max} > 1$: $\|W_h^T\| \to \infty$ (기울기 폭발)

$\tanh$의 도함수 $1 - h_t^2 \in (0, 1]$이 추가로 곱해지므로, 소실 방향으로 더 편향된다.

문제: 시퀀스 길이 = 799 이면 799번의 forward가 하나의 계산 그래프에 연결된다. 역전파 시 이 전체를 역순회해야 하므로 **메모리와 계산 비용이 시퀀스 길이에 비례**한다.

### Truncated BPTT

**해결책**: 일정 스텝(`bptt_length=30`)마다 역전파를 **끊는다**.

```
[────── segment 1 ──────]  [────── segment 2 ──────]  [── ...
x₀  x₁  ...  x₂₉        x₃₀ x₃₁  ...  x₅₉
      ← backward          ← backward
         unchain!              unchain!
```

절차:

1. `bptt_length` 스텝 동안 forward + loss 누적
2. `loss.backward()` — 기울기 계산 (현재 세그먼트 내에서만)
3. `loss.unchain_backward()` — 그래프 절단
4. `optimizer.update()` — 파라미터 갱신
5. loss를 0으로 리셋하고 다음 세그먼트 시작

**핵심**: 은닉 상태 $h$의 **값은 유지**되지만 **그래프 연결이 끊김** → 다음 세그먼트에서 $h$는 **상수로 취급**됨. 따라서 순전파 시 장기 기억은 보존되지만, 역전파는 `bptt_length` 이전으로 거슬러 올라가지 않는다.

```rust
// 학습 루프 핵심
for i in 0..seqlen {
    let y = model.forward(&x);
    loss = &loss + &mean_squared_error(&y, &t);
    count += 1;

    // bptt_length마다 역전파 + 그래프 절단
    if count % bptt_length == 0 || count == seqlen {
        model.cleargrads();
        loss.backward(false, false);
        loss.unchain_backward();  // 그래프 절단
        optimizer.update(&model.params());
    }
}
```

#### `unchain_backward()` 구현

```rust
pub fn unchain_backward(&self) {
    if let Some(creator) = self.inner.borrow().creator.clone() {
        let mut funcs = vec![creator];
        while let Some(state_ref) = funcs.pop() {
            let state = state_ref.borrow();
            for input in &state.inputs {
                let mut inner = input.inner.borrow_mut();
                if let Some(c) = inner.creator.take() {
                    funcs.push(c);
                }
            }
        }
    }
}
```

`loss`에서 출발해 계산 그래프를 역순 탐색하며, 모든 중간 Variable의 `creator`를 `None`으로 교체한다. `creator`가 없으면 `backward()`가 그 지점에서 멈추므로, 이전 세그먼트로의 역전파가 차단된다.

**왜 이것으로 충분한가?** `loss.backward()`는 `creator`가 있는 Variable만 추적한다. `creator.take()`로 모든 중간 노드의 연결을 끊으면, 이후 backward에서는 해당 구간을 아예 "존재하지 않는 것"으로 취급한다.

---

### Adam 옵티마이저

SGD의 한계: 모든 파라미터에 동일한 학습률 → 기울기가 큰 파라미터와 작은 파라미터에 동일한 보폭으로 이동.

**Adam** (Adaptive Moment Estimation, Kingma & Ba 2014)은 각 파라미터별로 **적응적 학습률**을 사용한다:

$$m \leftarrow \beta_1 m + (1-\beta_1) g \quad \text{(1차 모멘트: 기울기의 이동평균)}$$
$$v \leftarrow \beta_2 v + (1-\beta_2) g^2 \quad \text{(2차 모멘트: 기울기 제곱의 이동평균)}$$
$$\hat{m} = \frac{m}{1-\beta_1^t}, \quad \hat{v} = \frac{v}{1-\beta_2^t} \quad \text{(바이어스 보정)}$$
$$p \leftarrow p - \mathrm{lr} \times \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

**직관**: $\frac{m}{\sqrt{v}}$는 기울기를 자신의 표준편차로 나누는 것. 이는 **신호 대 잡음 비(SNR)**와 같다. 기울기가 일관된 방향이면 $|m| \approx \sqrt{v}$이므로 큰 보폭, 기울기가 진동하면 $|m| \ll \sqrt{v}$이므로 작은 보폭을 취한다.

#### 바이어스 보정: 왜 $\frac{1}{1 - \beta^t}$인가?

$m$과 $v$는 0으로 초기화된다. EMA(지수이동평균)를 $t$번 펼치면:

$$m_t = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i$$

기울기 $g_i$가 모두 같은 분포에서 온다고 가정하면 ($\mathbb{E}[g_i] = \mu$):

$$\mathbb{E}[m_t] = (1-\beta_1) \mu \sum_{i=1}^{t} \beta_1^{t-i} = (1-\beta_1) \mu \cdot \frac{1-\beta_1^t}{1-\beta_1} = \mu (1-\beta_1^t)$$

$m_t$의 기댓값이 $\mu$가 아니라 $(1-\beta_1^t) \mu$이다. 즉 **편향(bias)**이 있다. 보정:

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \quad \Rightarrow \quad \mathbb{E}[\hat{m}_t] = \mu$$

$t=1$일 때 $\beta_1 = 0.9$이면 보정 계수 $\frac{1}{1-0.9} = 10$배. 즉 $m_1 = 0.1 g_1$이지만 $\hat{m}_1 = g_1$로 복원된다. $t$가 커지면 $\beta_1^t \to 0$이어서 보정이 불필요해진다.

$v$에 대해서도 동일한 논리로 $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$.

#### 구현에서의 보정 통합

실제 구현에서는 $\hat{m}$과 $\hat{v}$를 명시적으로 계산하지 않고, 학습률에 보정을 흡수한다:

$$\mathrm{lr}_t = \mathrm{lr} \times \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$$

$$p \leftarrow p - \mathrm{lr}_t \times \frac{m}{\sqrt{v} + \epsilon}$$

이것은 $\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon'} = \frac{m/(1-\beta_1^t)}{\sqrt{v/(1-\beta_2^t)} + \epsilon'} = \frac{m}{(1-\beta_1^t)} \cdot \frac{\sqrt{1-\beta_2^t}}{\sqrt{v} + \epsilon'\sqrt{1-\beta_2^t}}$와 근사적으로 같다 ($\epsilon$ 항의 미세한 차이는 무시).

```rust
pub struct Adam {
    lr: f64,        // 0.001 (기본값)
    beta1: f64,     // 0.9  — 1차 모멘트 감쇠율
    beta2: f64,     // 0.999 — 2차 모멘트 감쇠율
    eps: f64,       // 1e-8 — 0 나누기 방지
    ms: RefCell<Vec<ArrayD<f64>>>,   // 1차 모멘트 (파라미터별)
    vs: RefCell<Vec<ArrayD<f64>>>,   // 2차 모멘트 (파라미터별)
    t: Cell<u32>,                     // 타임스텝
}

pub fn update(&self, params: &[Variable]) {
    self.t.set(self.t.get() + 1);
    let t = self.t.get() as f64;
    let fix1 = 1.0 - self.beta1.powf(t);
    let fix2 = 1.0 - self.beta2.powf(t);
    let lr_t = self.lr * fix2.sqrt() / fix1;   // 바이어스 보정된 학습률

    for (i, p) in params.iter().enumerate() {
        if let Some(grad) = p.grad() {
            // m ← β₁·m + (1-β₁)·grad
            ms[i] = &ms[i] * self.beta1 + &grad * (1.0 - self.beta1);
            // v ← β₂·v + (1-β₂)·grad²
            vs[i] = &vs[i] * self.beta2 + &(&grad * &grad) * (1.0 - self.beta2);
            // p ← p - lr_t · m / (√v + ε)
            let update = ms[i].mapv(|m| m * lr_t)
                       / vs[i].mapv(|v| v.sqrt() + self.eps);
            p.set_data(&p.data() - &update);
        }
    }
}
```

**SGD vs Adam 비교:**

| | SGD | Adam |
|---|---|---|
| 학습률 | 모든 파라미터에 동일 | 파라미터별 적응적 |
| 모멘텀 | 없음 (SGD+Momentum은 별도) | 1차 모멘트로 내장 |
| 하이퍼파라미터 | lr | lr, β₁, β₂, ε |
| 수렴 속도 | 느림 | 빠름 (특히 RNN) |
| 안정성 | 진동 가능 | 안정적 |

---

### SinCurve 데이터셋

사인 곡선의 **다음 값을 예측**하는 시계열 데이터:

$$x_i = \sin\left(\frac{2\pi i}{25}\right), \quad t_i = \sin\left(\frac{2\pi (i+1)}{25}\right)$$

1000개 포인트, 80/20 train/test 분할 → train 799 샘플.

```rust
pub struct SinCurve {
    x: Vec<f64>,  // 입력: sin(t)
    t: Vec<f64>,  // 타겟: sin(t+1)
}

impl SinCurve {
    pub fn new(train: bool) -> Self {
        let y: Vec<f64> = (0..1000)
            .map(|i| (2.0 * PI * i as f64 / 25.0).sin())
            .collect();
        let split = 800;
        let (x, t) = if train {
            (y[..split - 1].to_vec(), y[1..split].to_vec())
        } else {
            (y[split - 1..999].to_vec(), y[split..1000].to_vec())
        };
        SinCurve { x, t }
    }
}
```

---

### SimpleRNN 모델

`RNN` + `Linear`을 조합한 시계열 예측 모델:

```rust
struct SimpleRNN {
    rnn: RNN,       // 입력 → 은닉 상태 (기억 유지)
    fc: Linear,     // 은닉 상태 → 출력 (예측값)
}

fn forward(&self, x: &Variable) -> Variable {
    let h = self.rnn.forward(x);   // (batch, hidden_size)
    self.fc.forward(&h)            // (batch, 1)
}
```

**Model 트레잇을 사용하지 않는 이유**: `RNN`은 `Linear`과 달리 `reset_state()`, 별도의 `w_h` 등 추가 상태가 있어 `Model` 트레잇의 `layers() -> Vec<&Linear>` 인터페이스에 맞지 않는다. 대신 `params() -> Vec<Variable>`로 직접 파라미터를 수집하고 `Adam.update()`에 전달한다.

---

### 학습 결과

```
SinCurve train set: 799 samples
epoch   1 | loss 0.015231
epoch   5 | loss 0.000246
epoch  10 | loss 0.000116
epoch  15 | loss 0.000068
epoch  20 | loss 0.000047
```

20 에폭, loss 0.015 → 0.00005 (300배 감소). `hidden_size=100`, `bptt_length=30`, `Adam(lr=0.001)`.

---

### 핵심 포인트 정리

1. **RNN의 "기억"은 Variable**: `h`가 Variable이므로 별도의 시간 연결 로직 없이 계산 그래프가 자동으로 시간을 넘어 연결됨
2. **Truncated BPTT = 두 가지 절단**: `backward()`로 기울기를 계산한 뒤 `unchain_backward()`로 그래프를 끊음. 값은 유지되지만 그래프 연결은 제거
3. **Adam의 핵심**: 기울기를 자신의 크기로 정규화 → 모든 파라미터가 비슷한 속도로 수렴
4. **lazy init 패턴**: 입력 크기를 미리 알 수 없을 때 첫 forward에서 가중치를 초기화하는 Rust 패턴 (`RefCell<Option<Variable>>`)
