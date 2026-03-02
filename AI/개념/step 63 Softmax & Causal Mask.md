## Step 63: Softmax(axis)와 Causal Mask — Attention의 확률 변환과 미래 차단

### 한마디 직관

- **Softmax = 임의의 숫자를 확률로** — 어떤 실수 벡터든 "합이 1인 양수 벡터"로 바꾼다
- **Causal Mask = 미래를 못 보게 가림** — 시험 중 뒷문제 답을 가리는 것

이 두 연산으로 "어떤 토큰에 얼마나 집중할지"를 **확률**로 표현하되, **미래 정보는 차단**할 수 있다.

---

### Attention에서의 역할

Step 62에서 구현한 $Q K^T$는 토큰 간 유사도 **점수(score)**를 만든다. 하지만 이 점수는 임의의 실수(양수, 음수, 크기 제한 없음)이므로 가중합의 "가중치"로 바로 쓸 수 없다. 가중치는 합이 1인 확률이어야 의미가 있다.

$$\mathrm{scores} = \frac{Q K^T}{\sqrt{D}} \quad \text{shape: } (B, H, T, T)$$

$\mathrm{scores}[b, h, i, j]$ = "토큰 $i$가 토큰 $j$를 얼마나 참조하는가"의 원시 점수.

이 점수를 **확률**로 바꾸고(Softmax), **미래 토큰을 차단**해야(Causal Mask) 비로소 Attention이 완성된다:

$$\mathrm{Attention}(Q, K, V) = \underbrace{\mathrm{softmax}\!\left(\mathrm{causal\_mask}\!\left(\frac{Q K^T}{\sqrt{D}}\right)\right)}_{\text{확률 행렬 } (B,H,T,T)} \cdot V$$

---

### Softmax(axis): 임의 축에서의 확률 변환

#### 정의

축 $a$를 따라:

$$\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

출력의 성질:
- 모든 값이 양수 ($e^x > 0$ for any $x$)
- 해당 축의 합이 1 ($\sum_i \mathrm{softmax}(x)_i = 1$)

즉 **확률 분포**가 된다.

#### 왜 하필 $e^x$인가?

"임의의 실수 → 양수"로 바꾸는 함수는 많다. $x^2$, $|x|$, $\mathrm{ReLU}(x)$ 등. 왜 지수함수 $e^x$를 쓸까?

**1. 순서 보존 + 차이 증폭**

$e^x$는 단조 증가이므로 입력의 순서가 보존된다. 그리고 입력 차이를 **지수적으로 증폭**한다:

$$x = [1, 2, 3] \quad \Rightarrow \quad e^x = [2.72, 7.39, 20.09]$$

차이가 1이었던 것이 출력에서는 $2.72 : 7.39 : 20.09$로 큰 값이 훨씬 더 부각된다. 이것이 "가장 관련 있는 토큰에 집중"하는 Attention의 핵심 메커니즘을 만든다.

**2. 미분 가능한 argmax**

argmax는 가장 큰 원소만 1, 나머지 0으로 만드는 연산이지만 미분 불가능하다:

$$\mathrm{argmax}([1, 3, 2]) = [0, 1, 0]$$

softmax는 이것의 **연속적(미분 가능) 근사**다:

$$\mathrm{softmax}([1, 3, 2]) \approx [0.09, 0.67, 0.24]$$

입력 차이가 클수록 argmax에 가까워지고, 차이가 작으면 균등 분포에 가까워진다. 이 "부드러움(softness)" 덕분에 기울기를 흘릴 수 있고, 학습이 가능하다.

**3. log-linear 모델과의 연결**

softmax는 지수족(exponential family) 분포와 연결된다. $e^{x_i} / \sum e^{x_j}$는 에너지 $x_i$에 대한 **볼츠만 분포(Gibbs distribution)**와 동일한 형태다. 통계역학에서 시스템이 에너지 $E_i$인 상태에 있을 확률이 $P_i \propto e^{-E_i / kT}$인 것과 같다.

#### √D 스케일링: 왜 필요한가

Attention의 score는 $Q$와 $K$의 내적이다:

$$\mathrm{score} = q \cdot k = \sum_{d=1}^{D} q_d \cdot k_d$$

$q_d$와 $k_d$가 평균 0, 분산 1인 독립 확률변수라고 가정하면:

$$\mathrm{E}[q_d \cdot k_d] = \mathrm{E}[q_d] \cdot \mathrm{E}[k_d] = 0$$

$$\mathrm{Var}(q_d \cdot k_d) = \mathrm{E}[q_d^2] \cdot \mathrm{E}[k_d^2] = 1 \cdot 1 = 1$$

$D$개의 독립적인 항을 더하므로:

$$\mathrm{Var}(q \cdot k) = D \cdot 1 = D$$

**문제**: $D=64$이면 내적의 표준편차가 $\sqrt{64} = 8$이다. 즉 score 값이 $[-20, +20]$ 같은 큰 범위로 퍼진다. 이런 큰 값에 softmax를 적용하면:

$$\mathrm{softmax}([0, 0, 20]) \approx [0.0, 0.0, 1.0]$$

거의 one-hot이 되어 기울기가 사라진다 (softmax 포화). $\frac{\partial y_i}{\partial x_i} = y_i(1-y_i)$이므로 $y_i \approx 1$이면 기울기 $\approx 0$.

**해결**: $\sqrt{D}$로 나누면 분산이 다시 1이 된다:

$$\mathrm{Var}\!\left(\frac{q \cdot k}{\sqrt{D}}\right) = \frac{\mathrm{Var}(q \cdot k)}{D} = \frac{D}{D} = 1$$

score가 적당한 범위에 있으므로 softmax가 포화되지 않고 의미 있는 기울기를 유지한다.

#### Temperature와의 관계

$\sqrt{D}$ 스케일링은 사실 **temperature** $T$의 특수한 경우다:

$$\mathrm{softmax}\!\left(\frac{x}{T}\right)$$

| Temperature | 효과 | softmax 출력 |
|---|---|---|
| $T \to 0$ | 극도로 날카로움 | $\approx$ argmax (one-hot) |
| $T = 1$ | 기본 softmax | 차이에 비례한 확률 |
| $T \to \infty$ | 극도로 부드러움 | 균등 분포 $[1/n, \ldots, 1/n]$ |

유도: $T$가 크면 $x/T \to 0$이므로 모든 $e^{x_i/T} \to 1$, 즉 균등 분포.

Attention에서 $T = \sqrt{D}$는 "내적의 크기를 정규화하여 softmax가 적절한 범위에서 동작하게" 하는 역할이다. LLM 추론 시 temperature를 조절하면 출력의 다양성을 제어할 수 있는 것도 같은 원리.

#### 수치 안정성: max 빼기

실제 구현에서는 $e^{1000}$ 같은 값이 오버플로를 일으킨다 ($e^{710} > \mathrm{f64::MAX}$). 해결법:

$$\mathrm{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

이것이 원래 softmax와 동일한 이유 — $c = \max(x)$로 놓으면:

$$\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i} \cdot e^{-c}}{\sum_j e^{x_j} \cdot e^{-c}} = \frac{\cancel{e^{-c}} \cdot e^{x_i}}{\cancel{e^{-c}} \cdot \sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

$e^{-c}$가 분자·분모에서 약분된다. max를 빼면 지수의 최대값이 0($e^0 = 1$)이 되어 오버플로가 불가능.

```rust
// 축을 따라 max → 빼기 → exp → 합 → 나누기
let max_vals = x.map_axis(Axis(axis), |lane| {
    lane.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
});
// max_shape에 axis 위치에 1을 삽입하여 broadcast 가능하게
let shifted = x - &max_broadcast;  // 최대값이 0이 됨
let exp_x = shifted.mapv(f64::exp);
let y = &exp_x / &sum_broadcast;   // 확률로 정규화
```

검증: `[1000.0, 1001.0, 1002.0]` → softmax 합 = 1.0000000000 (오버플로 없음)

#### 기존 softmax_simple과의 차이

| | softmax_simple | softmax(axis) |
|---|---|---|
| 지원 차원 | 2D only | N-D |
| 축 지정 | axis=1 고정 | 임의 축, 음수 지원 (-1 = 마지막) |
| 수치 안정성 | max 빼기 없음 | max 빼기 포함 |
| 구현 | 기존 연산 조합 (exp, sum_with, div) | 단일 Function |
| 역전파 | 자동 (조합이므로 긴 계산 그래프) | 직접 구현 (효율적) |
| 용도 | 분류 출력 | **Attention scores** |

#### 역전파 유도

softmax의 야코비안(Jacobian)을 구해보자. $y_i = \frac{e^{x_i}}{S}$에서 $S = \sum_j e^{x_j}$.

이것은 $f(x) = \frac{g(x)}{h(x)}$ 형태이므로 **몫의 미분법(quotient rule)**을 적용한다:

$$\frac{\partial}{\partial x_j}\!\left(\frac{g}{h}\right) = \frac{g'h - gh'}{h^2}$$

**$i = j$인 경우** (자기 자신에 대한 미분):

$g = e^{x_i}$이고 $\frac{\partial g}{\partial x_i} = e^{x_i}$, $h = S$이고 $\frac{\partial S}{\partial x_i} = e^{x_i}$

$$\frac{\partial y_i}{\partial x_i} = \frac{e^{x_i} \cdot S - e^{x_i} \cdot e^{x_i}}{S^2} = \frac{e^{x_i}}{S} \cdot \frac{S - e^{x_i}}{S} = y_i(1 - y_i)$$

**$i \neq j$인 경우** (다른 원소에 대한 미분):

$g = e^{x_i}$이고 $\frac{\partial e^{x_i}}{\partial x_j} = 0$ ($i \neq j$이므로), $\frac{\partial S}{\partial x_j} = e^{x_j}$

$$\frac{\partial y_i}{\partial x_j} = \frac{0 \cdot S - e^{x_i} \cdot e^{x_j}}{S^2} = -\frac{e^{x_i}}{S} \cdot \frac{e^{x_j}}{S} = -y_i y_j$$

**크로네커 델타(Kronecker delta)**로 통합: $\delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$

$$\frac{\partial y_i}{\partial x_j} = y_i(\delta_{ij} - y_j)$$

검증: $i=j$이면 $y_i(1 - y_i)$ ✓, $i \neq j$이면 $y_i(0 - y_j) = -y_iy_j$ ✓

이 야코비안은 $n \times n$ 행렬이다. 하지만 역전파에서는 이 행렬을 명시적으로 만들 필요 없이 chain rule의 **벡터-야코비안 곱(VJP)**을 효율적으로 계산할 수 있다:

$$\frac{\partial L}{\partial x_i} = \sum_j \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i} = \sum_j g_{y_j} \cdot y_j(\delta_{ji} - y_i)$$

$\delta_{ji}$가 있는 항과 없는 항을 분리:

$$= \sum_j g_{y_j} y_j \delta_{ji} - y_i \sum_j g_{y_j} y_j$$

$\delta_{ji}$는 $j=i$일 때만 1이므로 첫째 항에서 $j=i$만 남는다:

$$= g_{y_i} y_i - y_i \sum_j g_{y_j} y_j = y_i \!\left( g_{y_i} - \sum_j g_{y_j} y_j \right)$$

벡터 형태로:

$$g_x = y \odot \left( g_y - \underbrace{\sum_{\mathrm{axis}}(g_y \odot y)}_{\text{스칼라 (keepdims)}} \right)$$

$\odot$는 원소별 곱(Hadamard product). $\sum_{\mathrm{axis}}$는 해당 축을 따라 합산하되 keepdims=true로 broadcast 가능하게 유지.

```rust
fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
    let gy_y = gy * y;                          // gy ⊙ y
    let sum_gy_y = gy_y.sum_axis(Axis(axis));   // Σ(gy ⊙ y), 스칼라
    // keepdims를 위해 axis 위치에 1 삽입 → broadcast 가능
    let gx = y * (gy - sum_broadcast);          // y ⊙ (gy - Σ(gy⊙y))
}
```

**검증**: `sum(softmax(x))`의 기울기가 0인 이유:
- $g_y = [1, 1, \ldots, 1]$ (sum의 기울기)
- $\sum_j g_{y_j} y_j = \sum_j 1 \cdot y_j = 1$ (softmax의 합은 항상 1)
- $g_{x_i} = y_i(1 - 1) = 0$

softmax의 출력 합은 항상 1이라는 상수이므로, 그 합의 미분은 0. 테스트에서 확인됨.

---

### Causal Mask: 미래 토큰 차단

#### 왜 필요한가: 자기회귀와 Teacher Forcing

GPT 같은 **자기회귀(autoregressive)** 모델은 토큰을 하나씩 순차적으로 생성한다:

```
<start> → "I"를 예측
<start> I → "love"를 예측
<start> I love → "coding"을 예측
```

**추론(inference)** 시에는 실제로 하나씩 생성하므로 미래 정보를 볼 수 없다 — 아직 생성하지 않았으니까.

**훈련(training)** 시에는 정답 시퀀스 전체를 이미 알고 있다. 효율을 위해 모든 토큰을 **동시에** 입력하고 각 위치의 다음 토큰을 한번에 예측한다. 이를 **Teacher Forcing**이라 한다:

```
입력:  [<start>, I, love, coding]
정답:  [I, love, coding, <end>]
```

하지만 이때 "I"를 예측하는 위치에서 "love", "coding"이 보이면 **커닝**이다. 정답을 미리 보고 예측하는 것이므로 학습이 무의미해진다. Causal mask는 이 커닝을 방지한다:

| | 추론 시 | 훈련 시 (Teacher Forcing) |
|---|---|---|
| 입력 방식 | 토큰 하나씩 순차 | 전체 시퀀스 동시 |
| 미래 차단 | 자연스럽게 불가 (아직 없음) | **Causal Mask로 강제 차단** |
| 속도 | 느림 (순차) | 빠름 (병렬) |

Teacher Forcing + Causal Mask = "추론처럼 한 토큰씩 생성하는 것을 흉내내면서 병렬로 빠르게 학습".

#### 구현 원리

scores 행렬 $(T \times T)$에서 "미래" 위치를 $-\infty$로 설정:

$$\mathrm{mask}[i][j] = \begin{cases} 0 & \text{if } j \leq i \quad \text{(과거 + 현재: 참조 가능)} \\ -\infty & \text{if } j > i \quad \text{(미래: 참조 불가)} \end{cases}$$

$T=4$일 때 scores에 mask를 적용하면:

```
[  s₀₀,   -∞,   -∞,   -∞ ]    ← 토큰 0: 자기 자신만 볼 수 있음
[  s₁₀,  s₁₁,   -∞,   -∞ ]    ← 토큰 1: 0, 1만 볼 수 있음
[  s₂₀,  s₂₁,  s₂₂,   -∞ ]    ← 토큰 2: 0, 1, 2만 볼 수 있음
[  s₃₀,  s₃₁,  s₃₂,  s₃₃ ]    ← 토큰 3: 모든 토큰을 볼 수 있음
```

#### 왜 $-\infty$인가: softmax와의 상호작용

$-\infty$를 넣는 이유는 softmax의 성질 때문이다:

$$e^{-\infty} = \lim_{x \to -\infty} e^x = 0$$

따라서:

$$\mathrm{softmax}(\ldots, -\infty, \ldots)_k = \frac{e^{-\infty}}{\sum} = \frac{0}{\sum} = 0$$

미래 토큰의 Attention 가중치가 **정확히 0**이 된다. 0에 가까운 것이 아니라 수학적으로 정확히 0.

다른 방법 (예: 매우 작은 음수 $-10^{9}$)을 쓸 수도 있지만, $-\infty$가 가장 깔끔하다.

**왜 곱셈이 아닌 덧셈(+mask)인가?**: softmax 전에 score에 0을 곱하면 score가 0이 되지만, $\mathrm{softmax}(0) \neq 0$이다. 반면 $-\infty$를 더하면 $\mathrm{softmax}(-\infty) = 0$이 보장된다.

실제 출력 (모든 score가 동일할 때의 mask + softmax):

```
[1.0000, 0.0000, 0.0000, 0.0000]   ← 1개 토큰 → 확률 1/1
[0.5000, 0.5000, 0.0000, 0.0000]   ← 2개 토큰 → 확률 1/2씩
[0.3333, 0.3333, 0.3333, 0.0000]   ← 3개 토큰 → 확률 1/3씩
[0.2500, 0.2500, 0.2500, 0.2500]   ← 4개 토큰 → 확률 1/4씩
```

#### 역전파 유도

Causal mask의 순전파를 수식으로 쓰면:

$$y_{ij} = \begin{cases} x_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

마스크되지 않은 위치 ($j \leq i$): $y_{ij} = x_{ij}$이므로 $\frac{\partial y_{ij}}{\partial x_{ij}} = 1$.

마스크된 위치 ($j > i$): $y_{ij} = -\infty$ (상수). 입력 $x_{ij}$와 무관한 상수로 대체되었으므로 $\frac{\partial y_{ij}}{\partial x_{ij}} = 0$.

chain rule 적용:

$$\frac{\partial L}{\partial x_{ij}} = \frac{\partial L}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial x_{ij}} = \begin{cases} g_{y_{ij}} \cdot 1 = g_{y_{ij}} & \text{if } j \leq i \\ g_{y_{ij}} \cdot 0 = 0 & \text{if } j > i \end{cases}$$

직관: "상수로 대체된 곳은 입력에 대한 의존성이 끊겼으므로 기울기가 흐르지 않는다."

```rust
fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
    // 하삼각+대각선: 기울기 그대로 통과
    // 상삼각(미래): 기울기 0으로
    for b in 0..batch {
        for i in 0..t_row {
            for j in (i + 1)..t_col {  // j > i인 위치만
                gx_slice[b * stride + i * t_col + j] = 0.0;
            }
        }
    }
}
```

#### Causal vs Bidirectional

| | Causal (GPT) | Bidirectional (BERT) |
|---|---|---|
| 마스크 | 상삼각 $-\infty$ (하삼각만 보임) | 마스크 없음 (전체 보임) |
| 참조 범위 | 과거 + 현재만 | 모든 토큰 (과거 + 현재 + 미래) |
| 학습 목표 | 다음 토큰 예측 | 마스크된 토큰 복원 (MLM) |
| 용도 | 텍스트 **생성** | 텍스트 **이해** (분류, QA 등) |

BERT는 동일한 Attention에서 causal mask를 제거하면 된다 — 구조적으로 마스크 하나의 차이.

---

### 전체 Attention 파이프라인

Step 62의 `transpose_axes`, `batched_matmul`과 이번 step의 `softmax`, `causal_mask`를 조합하면 완전한 Attention이 된다:

```rust
// Q, K, V: (B=1, H=2, T=4, D=3)

// 1. K의 마지막 두 축 전치
let k_t = transpose_axes(&k, &[0, 1, 3, 2]);   // (1,2,4,3) → (1,2,3,4)

// 2. Q와 K^T의 배치 행렬곱 → 유사도 점수
let scores = batched_matmul(&q, &k_t);           // (1,2,4,3)@(1,2,3,4) → (1,2,4,4)

// 3. √D 스케일링 → softmax 포화 방지
let scaled = &scores / d_k.sqrt();                // (1,2,4,4)

// 4. 미래 토큰 차단
let masked = causal_mask(&scaled);                // 상삼각 → -∞

// 5. 확률 변환 (각 행의 합 = 1, 미래 = 0)
let probs = softmax(&masked, -1);                 // (1,2,4,4)

// 6. 가중합 → 최종 출력
let out = batched_matmul(&probs, &v);             // (1,2,4,4)@(1,2,4,3) → (1,2,4,3)
```

```
Q shape:      [1, 2, 4, 3]
scores shape: [1, 2, 4, 4]
probs shape:  [1, 2, 4, 4]    ← 각 행의 합 = 1, 미래 위치 = 0
output shape: [1, 2, 4, 3]
Q grad shape: [1, 2, 4, 3]    ← 모든 grad 유한 (NaN/Inf 없음)
```

역전파 경로: `out → probs → softmax → masked → causal_mask → scaled → scores → Q, K` — 6단계의 연산을 거쳐 모든 입력에 기울기가 전파된다.

---

### nanoGPT 로드맵

```
[x] Embedding, AdamW          ← step 61
[x] Transpose(axes)           ← step 62
[x] Batched Matmul            ← step 62
[x] Softmax(axis)             ← step 63 (이번)
[x] Causal Mask               ← step 63 (이번)
[ ] LayerNorm, GELU
[ ] Multi-Head Attention
[ ] GPT Block 통합
```

다음은 **LayerNorm과 GELU** — Transformer 블록의 정규화와 활성화 함수.
