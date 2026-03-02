## Step 60: LSTM과 SeqDataLoader (배치 시계열 학습)

### 한마디 직관

**LSTM = 선택적 장기 요약기**

RNN이 매번 $h$를 통째로 덮어써서 오래된 정보가 사라지는 반면, LSTM은 셀 상태 $c$에 대해:
- **forget 게이트**: 불필요한 기억을 지우고
- **input 게이트**: 새로운 정보를 선택적으로 더하고
- **output 게이트**: 필요한 것만 꺼내서 출력

$c$가 덧셈으로만 업데이트되므로 오래된 정보도 보존할 수 있다.

---

### RNN의 한계: 기울기 소실

Step 59의 RNN:

$$h_t = \tanh(x_t W_x + h_{t-1} W_h)$$

역전파 시 $\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - \tanh^2) \cdot W_h$가 시간 스텝마다 곱해진다. 이 곱이 반복되면:

- $\|W_h\| < 1$이면 기울기가 기하급수적으로 감소 → **기울기 소실 (vanishing gradient)**
- $\|W_h\| > 1$이면 기울기가 기하급수적으로 증가 → **기울기 폭발 (exploding gradient)**

결과: 시퀀스가 길어지면 **장기 의존성(long-term dependency)** 학습이 불가능하다. 30스텝 전의 정보가 현재 출력에 영향을 주지 못함.

---

### LSTM (Long Short-Term Memory)

**핵심 아이디어**: 은닉 상태 $h$와 별도로 **셀 상태(cell state)** $c$를 유지한다. $c$는 **덧셈**으로만 업데이트되므로 기울기가 곱셈 반복 없이 안정적으로 전파된다.

#### 4개 게이트

| 게이트 | 수식 | 활성화 | 역할 |
|--------|------|--------|------|
| **Forget** $f$ | $\sigma(x W_{xf} + h W_{hf} + b_f)$ | sigmoid (0~1) | 과거 기억 중 **잊을 비율** |
| **Input** $i$ | $\sigma(x W_{xi} + h W_{hi} + b_i)$ | sigmoid (0~1) | 새 정보 중 **기억할 비율** |
| **Candidate** $g$ | $\tanh(x W_{xg} + h W_{hg} + b_g)$ | tanh (-1~1) | 새로운 **후보 기억** |
| **Output** $o$ | $\sigma(x W_{xo} + h W_{ho} + b_o)$ | sigmoid (0~1) | 셀 상태 중 **출력할 비율** |

#### 왜 sigmoid와 tanh인가?

**게이트(f, i, o)에 sigmoid**: 게이트는 "얼마나 통과시킬지"의 비율이다. sigmoid의 출력 범위 $[0, 1]$이 자연스럽게 비율(percentage)을 표현한다. 0이면 완전 차단, 1이면 완전 통과.

**후보(g)에 tanh**: 후보 기억 $g$는 셀 상태에 **더해지는** 새로운 정보다. tanh의 출력 범위 $[-1, 1]$은:
1. **양수/음수 모두 가능** → 셀 상태를 증가시킬 수도, 감소시킬 수도 있음
2. **크기 제한** → 한 번에 너무 큰 값이 더해지는 것을 방지
3. **0 중심** → 평균적으로 셀 상태에 편향을 주지 않음

#### 상태 업데이트

$$c_t = f \odot c_{t-1} + i \odot g \quad \text{(선택적 망각 + 선택적 기억)}$$
$$h_t = o \odot \tanh(c_t) \quad \text{(셀 상태를 필터링한 출력)}$$

- $\odot$: 원소별 곱 (Hadamard product)
- $c$는 "장기 기억", $h$는 "단기 기억(출력)" 역할

#### 기울기 보존 원리: RNN vs LSTM 비교

**RNN의 기울기 전파** ([[step 59 RNN]]):

$$\frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}(1 - h_t^2) \cdot W_h$$

시간 $T$에서 시간 $0$까지:

$$\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T} \mathrm{diag}(1-h_t^2) \cdot W_h$$

**행렬 $W_h$가 $T$번 곱해진다.** 행렬곱의 반복은 고유값(eigenvalue)에 의해 지배된다:
- 최대 고유값 $|\lambda_{\max}| < 1$ → $W_h^T \to 0$ (기울기 소실)
- 최대 고유값 $|\lambda_{\max}| > 1$ → $W_h^T \to \infty$ (기울기 폭발)

행렬곱에서는 $|\lambda_{\max}| = 1$인 경우조차 고유벡터 방향이 아닌 성분은 소멸한다.

**LSTM의 기울기 전파**:

$$\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t)$$

시간 $T$에서 시간 $0$까지:

$$\frac{\partial c_T}{\partial c_0} = \prod_{t=1}^{T} \mathrm{diag}(f_t)$$

이것은 **대각 행렬의 곱**이므로 각 차원이 독립적으로:

$$\left(\frac{\partial c_T}{\partial c_0}\right)_j = \prod_{t=1}^{T} f_{t,j}$$

$f \in [0, 1]$이므로 폭발은 불가능하고, $f \approx 1$이면 기울기가 거의 보존된다. 핵심 차이:

| | RNN | LSTM |
|---|---|---|
| 반복 연산 | $W_h$ 행렬곱 | $f$ 원소별 곱 |
| 기울기 스케일 결정 | $W_h$의 고유값 (고정) | $f$의 값 (학습 가능!) |
| 차원 간 간섭 | 있음 (행렬이므로) | 없음 (대각이므로) |
| 폭발 가능성 | $|\lambda| > 1$이면 가능 | $f \le 1$이므로 불가 |

$f$는 **학습 가능**하다는 것이 결정적 차이이다. 네트워크가 장기 기억이 필요하다고 판단하면 $f \to 1$로 학습하여 기울기를 보존하고, 불필요하면 $f \to 0$으로 학습하여 잊는다.

#### LSTM Backward 상세

$c_t = f \odot c_{t-1} + i \odot g$, $h_t = o \odot \tanh(c_t)$에서 $\frac{\partial \mathcal{L}}{\partial h_t}$가 주어졌을 때:

$$\frac{\partial \mathcal{L}}{\partial o} = \frac{\partial \mathcal{L}}{\partial h_t} \odot \tanh(c_t) \quad (\because h_t = o \odot \tanh(c_t))$$

$$\frac{\partial \mathcal{L}}{\partial c_t} = \frac{\partial \mathcal{L}}{\partial h_t} \odot o \odot (1 - \tanh^2(c_t)) \quad (\tanh\text{의 도함수})$$

$c_t$에 대한 기울기가 구해지면, 각 게이트로 전파:

$$\frac{\partial \mathcal{L}}{\partial f} = \frac{\partial \mathcal{L}}{\partial c_t} \odot c_{t-1} \quad (\because c_t = f \odot c_{t-1} + \ldots)$$

$$\frac{\partial \mathcal{L}}{\partial i} = \frac{\partial \mathcal{L}}{\partial c_t} \odot g$$

$$\frac{\partial \mathcal{L}}{\partial g} = \frac{\partial \mathcal{L}}{\partial c_t} \odot i$$

$$\frac{\partial \mathcal{L}}{\partial c_{t-1}} = \frac{\partial \mathcal{L}}{\partial c_t} \odot f \quad \text{(이전 시점으로 전파)}$$

각 게이트의 활성화 함수를 통과:

$$\frac{\partial \mathcal{L}}{\partial z_f} = \frac{\partial \mathcal{L}}{\partial f} \odot f \odot (1-f) \quad (\sigma\text{의 도함수: } \sigma'(z) = \sigma(z)(1-\sigma(z)))$$

$$\frac{\partial \mathcal{L}}{\partial z_g} = \frac{\partial \mathcal{L}}{\partial g} \odot (1-g^2) \quad (\tanh\text{의 도함수})$$

마지막으로 $z_f = x W_{xf} + h W_{hf}$에서 가중치 기울기:

$$\frac{\partial \mathcal{L}}{\partial W_{xf}} = x^T \cdot \frac{\partial \mathcal{L}}{\partial z_f}, \quad \frac{\partial \mathcal{L}}{\partial h_{t-1}} += \frac{\partial \mathcal{L}}{\partial z_f} \cdot W_{hf}^T$$

dezero에서는 이 backward를 **자동 미분**으로 처리한다 — 각 게이트가 Variable 연산(`sigmoid`, `tanh`, `matmul`, `*`, `+`)이므로, 그래프를 따라 자동으로 역전파가 수행된다. 위의 수식은 자동 미분이 내부적으로 계산하는 것과 정확히 동일하다.

```
      ┌──── f ⊙ ────┐
c_{t-1} ─────────────+ ──→ c_t ──→ tanh ──→ ⊙ ──→ h_t
                     ↑                       ↑
         i ⊙ g ─────┘                       o
```

---

### LSTM 구현

```rust
pub struct LSTM {
    // 입력 → 게이트 (Linear 4개, bias 포함)
    x2f: Linear, x2i: Linear, x2o: Linear, x2g: Linear,
    // 은닉 → 게이트 (가중치 4개, bias 없음, lazy init)
    w_hf: RefCell<Option<Variable>>,
    w_hi: RefCell<Option<Variable>>,
    w_ho: RefCell<Option<Variable>>,
    w_hg: RefCell<Option<Variable>>,
    // 상태
    h: RefCell<Option<Variable>>,  // 은닉 상태 (단기 기억)
    c: RefCell<Option<Variable>>,  // 셀 상태 (장기 기억)
    hidden_size: usize,
}
```

**왜 4개의 Linear + 4개의 Variable로 분리하는가?**

일반적인 LSTM 구현에서는 $[W_{xf}; W_{xi}; W_{xo}; W_{xg}]$를 하나의 큰 행렬로 결합한 뒤 Slice로 분리한다. 하지만 dezero에는 Slice 연산의 backward가 구현되어 있지 않으므로, 각 게이트를 독립된 Linear으로 구현한다.

#### 순전파

```rust
pub fn forward(&self, x: &Variable) -> Variable {
    // 4개 게이트 계산
    let (f, i, o, g) = if self.h.borrow().is_none() {
        // 첫 스텝: h가 없으므로 x→gate만 사용
        (
            sigmoid(&self.x2f.forward(x)),
            sigmoid(&self.x2i.forward(x)),
            sigmoid(&self.x2o.forward(x)),
            tanh(&self.x2g.forward(x)),
        )
    } else {
        // 이후 스텝: x→gate + h@W_h
        let h = self.h.borrow().clone().unwrap();
        (
            sigmoid(&(&self.x2f.forward(x) + &matmul(&h, &w_hf))),
            sigmoid(&(&self.x2i.forward(x) + &matmul(&h, &w_hi))),
            sigmoid(&(&self.x2o.forward(x) + &matmul(&h, &w_ho))),
            tanh(&(&self.x2g.forward(x) + &matmul(&h, &w_hg))),
        )
    };

    // 셀 상태 업데이트
    let c_new = if self.c.borrow().is_none() {
        &i * &g  // 첫 스텝: forget할 것이 없음 → c₀ = i ⊙ g
    } else {
        let c = self.c.borrow().clone().unwrap();
        &(&f * &c) + &(&i * &g)  // c_new = f ⊙ c + i ⊙ g
    };

    // 은닉 상태 업데이트
    let h_new = &o * &tanh(&c_new);  // h_new = o ⊙ tanh(c_new)

    *self.h.borrow_mut() = Some(h_new.clone());
    *self.c.borrow_mut() = Some(c_new);
    h_new
}
```

**파라미터 수 비교**: RNN은 $W_x$, $W_h$, $b$ (3개). LSTM은 이것이 4세트 → 12개. `hidden_size=100`이면 RNN 파라미터 ~10K, LSTM ~40K.

---

### SeqDataLoader: 시계열 배치 학습

#### 왜 일반 DataLoader를 쓸 수 없는가?

일반 `DataLoader`는 데이터를 **랜덤 셔플**한다. 이미지 분류에서는 순서가 무관하므로 셔플이 유익하지만, RNN/LSTM은 은닉 상태가 **시간 순서대로** 전파되어야 한다. 셔플하면 $h_t$가 $x_t$가 아닌 임의의 $x_j$에서 온 것이 되어 의미가 없다.

#### SeqDataLoader 원리

시퀀스를 `batch_size`개의 **병렬 스트림**으로 분할한다:

```
전체 데이터: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
batch_size = 3, jump = 12/3 = 4

스트림 0: [0]  [1]  [2]  [3]     ← idx = 0*4+t
스트림 1: [4]  [5]  [6]  [7]     ← idx = 1*4+t
스트림 2: [8]  [9]  [10] [11]    ← idx = 2*4+t

t=0: batch = [data[0], data[4], data[8]]
t=1: batch = [data[1], data[5], data[9]]
t=2: batch = [data[2], data[6], data[10]]
t=3: batch = [data[3], data[7], data[11]]
```

각 스트림 내에서는 시간 순서가 유지된다. 배치의 각 요소는 서로 다른 시작점에서 출발하지만, 각각 연속적인 시퀀스를 따른다.

```rust
pub struct SeqDataLoader<'a> {
    dataset: &'a SinCurve,
    batch_size: usize,
    pub jump: usize,     // = dataset.len() / batch_size (스트림당 길이)
    current: usize,      // 현재 시간 스텝
}

impl<'a> Iterator for SeqDataLoader<'a> {
    type Item = (Variable, Variable);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.jump { return None; }

        let mut x_data = Vec::with_capacity(self.batch_size);
        let mut t_data = Vec::with_capacity(self.batch_size);

        for j in 0..self.batch_size {
            let idx = j * self.jump + self.current;  // 스트림 j의 t번째 샘플
            let (x, t) = self.dataset.get(idx);
            x_data.push(x);
            t_data.push(t);
        }
        self.current += 1;

        // shape: (batch_size, 1)
        Some((
            Variable::new(ArrayD::from_shape_vec(IxDyn(&[self.batch_size, 1]), x_data).unwrap()),
            Variable::new(ArrayD::from_shape_vec(IxDyn(&[self.batch_size, 1]), t_data).unwrap()),
        ))
    }
}
```

**에폭 간 리셋**: 매 에폭 시작 시 `dataloader.reset()`으로 `current=0`으로 돌림. 모델의 은닉 상태도 `model.reset_state()`로 초기화.

---

### BetterRNN 모델

Step 59의 `SimpleRNN`에서 `RNN`을 `LSTM`으로 교체하고, `SeqDataLoader`로 배치 학습:

```rust
struct BetterRNN {
    rnn: LSTM,      // RNN → LSTM으로 교체
    fc: Linear,
}

fn forward(&self, x: &Variable) -> Variable {
    let h = self.rnn.forward(x);   // (batch, hidden_size)
    self.fc.forward(&h)            // (batch, 1)
}
```

학습 루프에서의 차이:

```rust
// step 59: 단일 샘플 순차 처리
for i in 0..seqlen {
    let (x_val, t_val) = train_set.get(i);
    let x = Variable::new(/* shape: (1, 1) */);
    // ...
}

// step 60: 배치 병렬 처리
let mut dataloader = SeqDataLoader::new(&train_set, batch_size);
for (x, t) in &mut dataloader {
    // x: (30, 1), t: (30, 1) — 30개 스트림을 동시에 처리
    let y = model.forward(&x);
    // ...
}
```

---

### 학습 결과

```
SinCurve: 799 samples, batch_size=30, 26 steps/epoch
epoch   1 | loss 0.499015
epoch   5 | loss 0.286951
epoch  10 | loss 0.142047
epoch  15 | loss 0.066237
```

15 에폭, loss 0.499 → 0.066 (7.5배 감소), 9.96초.

Step 59 (RNN, 단일 샘플)은 20 에폭에 48.82초 → Step 60 (LSTM, batch=30)은 15 에폭에 9.96초. 에폭당 처리 속도가 크게 향상.

---

### RNN vs LSTM 비교

| | RNN (step 59) | LSTM (step 60) |
|---|---|---|
| 상태 | $h$ 1개 | $h$, $c$ 2개 |
| 게이트 | 없음 | forget, input, output, candidate |
| 기울기 흐름 | $W_h$ 행렬곱 반복 → 소실/폭발 | $c$를 통한 덧셈 → 보존 |
| 파라미터 | ~10K (hidden=100) | ~40K (hidden=100) |
| 장기 의존성 | 약함 (~10 스텝) | 강함 (~100+ 스텝) |
| 데이터 로딩 | 단일 샘플 순차 | SeqDataLoader 배치 병렬 |
| 학습 시간 | 48.82초 (20 에폭) | 9.96초 (15 에폭) |

---

### 핵심 포인트 정리

1. **기울기 소실의 원인**: RNN은 $W_h$ 행렬곱이 시간 스텝마다 반복. 고유값 < 1이면 소멸, > 1이면 폭발
2. **LSTM의 해결법**: 셀 상태 $c$를 별도로 두고 **덧셈**으로만 업데이트 → $\frac{\partial c_t}{\partial c_{t-1}} = f$ (행렬곱이 아닌 원소별 곱)
3. **게이트의 직관**: forget = "이전 기억을 얼마나 유지?", input = "새 정보를 얼마나 받아들일?", output = "무엇을 내보낼?"
4. **SeqDataLoader**: 시계열의 시간 순서를 유지하면서 배치 병렬화. `jump = len / batch_size`로 스트림 분할
5. **LSTM의 후속**: GRU (게이트 2개로 단순화), Transformer (어텐션으로 RNN 자체를 대체)
