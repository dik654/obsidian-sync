## Step 65: Multi-Head Attention — 모든 부품의 조립

### 한마디 직관

- **Self-Attention = 각 토큰이 다른 토큰을 얼마나 참조할지 결정** — "나에게 중요한 정보가 어디에 있는가?"
- **Multi-Head = 같은 질문을 여러 관점에서** — 한 헤드는 문법, 다른 헤드는 의미, 또 다른 헤드는 위치 관계를 담당

Step 61~64에서 만든 모든 부품(Embedding, Transpose, BatchedMatMul, Softmax, CausalMask, LayerNorm, GELU)을 조합하여 GPT의 핵심 모듈을 완성한다.

---

### Self-Attention: "누구를 참조할 것인가"

#### 검색 엔진 비유

Self-Attention은 본질적으로 **검색(retrieval)** 시스템이다:

| Attention | 검색 엔진 | 역할 |
|---|---|---|
| Query ($Q$) | 검색어 | "내가 찾고 있는 것" |
| Key ($K$) | 문서 제목/태그 | "나는 이런 정보를 가지고 있다" |
| Value ($V$) | 문서 본문 | "실제 전달할 내용" |

"I love **coding** because **it** is fun"에서 "it"이 무엇을 가리키는지 결정하는 과정:

1. "it"의 Query: "나는 대명사다. 내가 가리키는 대상이 뭐지?"
2. 각 토큰의 Key: "I"→"주어", "love"→"동사", "coding"→"명사/활동", ...
3. Score: Query와 Key의 내적 → "coding"의 Key와 가장 높은 유사도
4. "coding"의 Value(실제 의미 벡터)를 가져와서 "it"의 표현에 반영

#### Self-Attention의 전체 공식

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^T}{\sqrt{D_h}}\right) V$$

각 단계를 분해하면:

$$\underbrace{Q K^T}_{(T, T)} \xrightarrow{\div \sqrt{D_h}} \underbrace{\mathrm{scores}}_{(T, T)} \xrightarrow{\mathrm{mask}} \xrightarrow{\mathrm{softmax}} \underbrace{\mathrm{probs}}_{(T, T)} \xrightarrow{\times V} \underbrace{\mathrm{output}}_{(T, D_h)}$$

이미 step 62~63에서 각 단계를 개별 구현했다. 이번 step에서는 이것들을 **하나의 모듈로 조립**한다.

---

### 왜 Multi-Head인가: 단일 Attention의 한계

#### 단일 헤드의 문제

단일 Attention은 하나의 $Q$, $K$, $V$ 변환으로 토큰 간 관계를 포착한다. 그런데 자연어에서는 동시에 여러 종류의 관계가 존재한다:

"The cat sat on the mat because **it** was soft"

- **구문적 관계**: "it" → "mat" (주어-서술어 일치)
- **의미적 관계**: "it" → "mat" (soft의 대상)
- **위치적 관계**: "it" → 가까운 명사를 선호

하나의 $(Q, K)$ 쌍으로는 이 모든 관계를 동시에 포착하기 어렵다. 내적 $q \cdot k$는 결국 **하나의 유사도 점수**이므로, 여러 관점의 정보가 하나의 숫자로 압축된다.

#### Multi-Head의 해결책: 부분 공간 분할

차원 $D$를 $H$개의 헤드로 분할하여 각 헤드가 $D_h = D/H$ 차원의 부분 공간에서 독립적으로 Attention을 수행한다:

$$D = H \times D_h$$

각 헤드는 서로 다른 선형 변환 $(W_Q^{(h)}, W_K^{(h)}, W_V^{(h)})$을 학습하므로, **서로 다른 관계 패턴**을 포착할 수 있다.

실제로 학습된 Attention 헤드를 분석하면:
- 어떤 헤드는 **인접 토큰**에 집중 (로컬 패턴)
- 어떤 헤드는 **구문 구조**(주어-동사)에 집중
- 어떤 헤드는 **구분자 토큰**([SEP])에 집중
- 어떤 헤드는 **이전 토큰**에 고르게 집중 (n-gram 효과)

#### 앙상블 효과

Multi-Head Attention은 일종의 **앙상블(ensemble)**이다:

$$\mathrm{MultiHead}(x) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_H) \cdot W_O$$

각 헤드가 독립적으로 패턴을 찾고, 결과를 합쳐서 출력 프로젝션으로 통합한다. 단일 큰 Attention보다 표현력이 풍부한 이유:

| | Single Head ($D=256$) | Multi-Head ($H=8$, $D_h=32$) |
|---|---|---|
| Attention 행렬 | $1 \times (T, T)$ | $8 \times (T, T)$ |
| 관계 패턴 | 1가지 | 최대 8가지 |
| 파라미터 수 | 동일 | 동일 |
| 표현력 | 하나의 관점 | 다양한 관점의 앙상블 |

파라미터 수가 같은 이유: Single head는 $(D, D)$ 프로젝션 하나, Multi-head는 $(D, D_h)$ 프로젝션 $H$개. 총 파라미터 $= D \times D_h \times H = D \times D$로 동일하다.

---

### CausalSelfAttention 데이터 흐름

전체 데이터 흐름을 shape와 함께 추적한다. $B=2$, $T=4$, $D=8$, $H=2$, $D_h=4$를 예시로:

#### Phase 1: Q, K, V 프로젝션

```
x: (B, T, D) = (2, 4, 8)
    ↓ reshape → (B*T, D) = (8, 8)     ← Linear은 2D 입력만 처리
    ↓
    ├→ Q_proj: (8, 8) @ W_Q → (8, 8)
    ├→ K_proj: (8, 8) @ W_K → (8, 8)
    └→ V_proj: (8, 8) @ W_V → (8, 8)
```

왜 하나의 큰 행렬이 아닌 별도의 $W_Q$, $W_K$, $W_V$인가?

$Q$, $K$, $V$는 서로 다른 역할을 수행한다. $Q$는 "무엇을 찾을까", $K$는 "무엇을 제공할까", $V$는 "어떤 정보를 전달할까". 같은 입력 $x$에서 서로 다른 선형 변환으로 다른 표현을 만들어야 이 역할 분리가 가능하다.

만약 $Q = K = x$ (변환 없이)라면 자기 자신과의 내적이 항상 가장 크므로 ($\|x\|^2 \geq x \cdot y$, Cauchy-Schwarz) 모든 토큰이 자기 자신에만 집중하게 된다.

#### Phase 2: 헤드 분할 (Head Split)

```
Q: (8, 8) → reshape → (2, 4, 2, 4) → transpose [0,2,1,3] → (2, 2, 4, 4)
   (B*T, D)   (B, T, H, D_h)                                  (B, H, T, D_h)
```

이 과정의 의미:

1. **reshape** $(B \cdot T, D) \to (B, T, H, D_h)$: $D$ 차원을 $H$개의 $D_h$ 차원으로 쪼갬
2. **transpose** $[0, 2, 1, 3]$: $(B, T, H, D_h) \to (B, H, T, D_h)$ — 헤드 축을 배치 축 바로 뒤로

왜 transpose가 필요한가? 각 헤드가 독립적으로 Attention을 수행하려면, 헤드 축이 배치 축처럼 앞에 와야 한다. $(B, H, T, D_h)$에서 마지막 두 차원 $(T, D_h)$에 대해 행렬곱을 수행하면, $B \times H$개의 독립적인 $(T, D_h)$ Attention이 된다.

```
(B, H, T, D_h) = (2, 2, 4, 4)
 ↑  ↑  ↑    ↑
 |  |  |    └── 각 헤드의 feature 차원
 |  |  └─────── 시퀀스 위치 (토큰)
 |  └────────── 헤드 인덱스 (독립 Attention)
 └───────────── 배치 인덱스
```

#### Phase 3: Scaled Dot-Product Attention

```
K^T = transpose(K, [0,1,3,2])
    = (2, 2, 4, 4) → (2, 2, 4, 4)     ← D_h=T=4인 예시라서 shape 동일

scores = Q @ K^T / √D_h
       = (2,2,4,4) @ (2,2,4,4) → (2, 2, 4, 4)
       = scores / √4 = scores / 2

masked = causal_mask(scores)            ← 상삼각 → -∞

probs = softmax(masked, axis=-1)        ← 각 행의 합 = 1, 미래 = 0
      = (2, 2, 4, 4)

out = probs @ V
    = (2,2,4,4) @ (2,2,4,4) → (2, 2, 4, 4)
                                  (B, H, T, D_h)
```

#### Phase 4: 헤드 병합 (Head Merge) + 출력 프로젝션

```
out: (2, 2, 4, 4)   (B, H, T, D_h)
    ↓ transpose [0,2,1,3] → (2, 4, 2, 4)   (B, T, H, D_h)
    ↓ reshape → (2, 4, 8)                    (B, T, D)
    ↓ reshape → (8, 8)                       (B*T, D)
    ↓ out_proj: (8, 8) @ W_O → (8, 8)
    ↓ reshape → (2, 4, 8)                    (B, T, D)
```

헤드 분할의 정확한 역연산: `transpose [0,2,1,3]`은 자기 자신의 역순열이므로 ([0,2,1,3]을 두 번 적용하면 항등), Phase 2의 역과정으로 원래 차원 배치를 복원한다.

출력 프로젝션 $W_O$의 역할: 각 헤드의 출력을 단순 연결(concat)하면 헤드 간 상호작용이 없다. $W_O$가 헤드들의 정보를 **혼합**하여 최종 표현을 만든다.

---

### Q, K, V의 기하학적 의미

#### 내적 = 코사인 유사도 × 크기

$$q \cdot k = \|q\| \|k\| \cos\theta$$

두 벡터의 내적은 **방향이 비슷할수록 크다**. $W_Q$와 $W_K$는 입력 벡터를 "Query 공간"과 "Key 공간"으로 사영(projection)하여, 이 공간에서의 유사도가 원하는 관계를 반영하도록 학습한다.

학습 전: $W_Q$, $W_K$는 랜덤 → 내적도 랜덤 → Attention이 거의 균등 분포
학습 후: 관련 있는 토큰 쌍의 $q \cdot k$가 크게, 무관한 쌍은 작게 → 유의미한 가중합

#### $D_h$의 역할: 표현력 vs 효율

$D_h$가 클수록 각 헤드가 더 풍부한 관계를 포착할 수 있지만, $D = H \times D_h$이므로 트레이드오프가 있다:

| 설정 | 헤드 수 ($H$) | 헤드 차원 ($D_h$) | 특성 |
|---|---|---|---|
| $D=768, H=12$ | 12 | 64 | GPT-2 Small — 균형 |
| $D=768, H=1$ | 1 | 768 | 풍부하지만 단일 관점 |
| $D=768, H=768$ | 768 | 1 | 많은 관점이지만 각각 빈약 |

실용적으로 $D_h = 64$ 또는 $D_h = 128$이 주로 사용된다.

---

### 역전파: 조합된 연산의 자동 미분

CausalSelfAttention은 새로운 역전파 공식이 필요 없다. 이미 구현된 연산들의 조합이므로, 계산 그래프를 통해 자동으로 역전파된다:

```
x → reshape → Linear(Q) → reshape → transpose → ─┐
                                                    ├→ batched_matmul → /√D_h
x → reshape → Linear(K) → reshape → transpose → ─┘      ↓
                                                    causal_mask
                                                         ↓
                                                      softmax
                                                         ↓
x → reshape → Linear(V) → reshape → transpose → ─→ batched_matmul
                                                         ↓
                                                    transpose → reshape
                                                         ↓
                                                    Linear(out) → reshape → output
```

역전파 경로 (output에서 x까지):

1. **reshape**: shape 변환만, 기울기를 원래 shape으로 되돌림
2. **Linear(out)**: $g_x = g_y W_O^T$, $g_{W_O} = x^T g_y$
3. **transpose [0,2,1,3]**: 역순열 적용 (자기 역이므로 동일 연산)
4. **batched_matmul** (probs @ V):
   - $g_{\mathrm{probs}} = g_y V^T$ (마지막 두 축 전치)
   - $g_V = \mathrm{probs}^T g_y$
5. **softmax**: $g_x = y \odot (g_y - \sum(g_y \odot y))$ (step 63)
6. **causal_mask**: 마스크 위치의 기울기 → 0 (step 63)
7. **스케일링** $\div \sqrt{D_h}$: $g_x = g_y / \sqrt{D_h}$
8. **batched_matmul** (Q @ K^T):
   - $g_Q = g_{\mathrm{scores}} K$
   - $g_K = g_{\mathrm{scores}}^T Q$
9. **transpose → reshape → Linear(Q/K/V)**: 각 프로젝션의 가중치 기울기

총 8개의 학습 가능 파라미터: $W_Q$, $b_Q$, $W_K$, $b_K$, $W_V$, $b_V$, $W_O$, $b_O$ (4 Linear $\times$ (W + b)).

---

### Attention의 계산 복잡도

$$\mathrm{scores} = Q K^T \quad \Rightarrow \quad (T, D_h) \times (D_h, T) = O(T^2 D_h)$$

Attention 행렬 $(T, T)$의 크기가 시퀀스 길이 $T$의 **제곱**에 비례한다. 이것이 Transformer의 근본적 병목:

| 시퀀스 길이 | Attention 행렬 크기 | 메모리 (f64, 단일 헤드) |
|---|---|---|
| $T = 128$ | $16,384$ | 128 KB |
| $T = 1024$ | $1,048,576$ | 8 MB |
| $T = 8192$ | $67,108,864$ | 512 MB |

이를 해결하기 위한 후속 연구:
- **Flash Attention**: 정확한 Attention을 타일링(tiling)으로 메모리 효율적으로 계산
- **Sparse Attention** (GPT-3): 전체가 아닌 일부 토큰만 참조
- **Linear Attention**: 커널 트릭으로 $O(T)$로 근사
- **Mamba/SSM**: Attention을 아예 제거하고 상태 공간 모델 사용

---

### 코드 발췌

```rust
pub struct CausalSelfAttention {
    n_head: usize,
    n_embd: usize,
    q_proj: Linear,      // (D, D) — Q 프로젝션
    k_proj: Linear,      // (D, D) — K 프로젝션
    v_proj: Linear,      // (D, D) — V 프로젝션
    out_proj: Linear,    // (D, D) — 출력 프로젝션
    attn_dropout: f64,
}

/// x: (B, T, D) → (B, T, D)
pub fn forward(&self, x: &Variable) -> Variable {
    let (b, t, d) = (shape[0], shape[1], shape[2]);
    let d_head = d / self.n_head;

    // Phase 1: Q, K, V 프로젝션 (Linear은 2D만 처리)
    let x_2d = reshape(x, &[b * t, d]);
    let q = self.q_proj.forward(&x_2d);   // (B*T, D)
    let k = self.k_proj.forward(&x_2d);
    let v = self.v_proj.forward(&x_2d);

    // Phase 2: 헤드 분할 — (B*T, D) → (B, H, T, D_h)
    let q = transpose_axes(&reshape(&q, &[b, t, self.n_head, d_head]),
                           &[0, 2, 1, 3]);
    let k = transpose_axes(&reshape(&k, &[b, t, self.n_head, d_head]),
                           &[0, 2, 1, 3]);
    let v = transpose_axes(&reshape(&v, &[b, t, self.n_head, d_head]),
                           &[0, 2, 1, 3]);

    // Phase 3: Scaled Dot-Product Attention
    let k_t = transpose_axes(&k, &[0, 1, 3, 2]);           // K^T
    let scores = &batched_matmul(&q, &k_t) / (d_head as f64).sqrt();
    let attn = dropout(&softmax(&causal_mask(&scores), -1),
                       self.attn_dropout);
    let out = batched_matmul(&attn, &v);

    // Phase 4: 헤드 병합 + 출력 프로젝션
    let out = reshape(&transpose_axes(&out, &[0, 2, 1, 3]), &[b, t, d]);
    let out_2d = self.out_proj.forward(&reshape(&out, &[b * t, d]));
    reshape(&out_2d, &[b, t, d])
}
```

---

### 검증: 인과성(Causality) 테스트

Multi-Head Attention이 정말로 미래 정보를 차단하는지 검증하는 핵심 테스트:

```rust
// 동일한 입력에서 마지막 토큰만 변경
let x1 = make_input(1, 4, 8, 0.0);         // 원본
let x2 = x1.clone();
x2[3, :] = 999.0;                           // 마지막 토큰만 완전히 다른 값

let y1 = attn.forward(&x1);
let y2 = attn.forward(&x2);

// 위치 0, 1, 2의 출력은 동일해야 함 (causal mask 때문)
assert!(y1[0..3] == y2[0..3]);

// 위치 3의 출력은 달라야 함 (입력이 바뀌었으므로)
assert!(y1[3] != y2[3]);
```

이 테스트가 통과한다는 것은: 토큰 $i$의 출력이 토큰 $0, 1, \ldots, i$에만 의존하고, $i+1, \ldots, T-1$에는 의존하지 않는다는 것을 의미한다. 이것이 바로 **인과적(causal)** Self-Attention의 정의다.

---

### Self-Attention이 RNN을 대체한 이유

#### RNN의 한계 (step 59~60에서 경험)

RNN/LSTM은 시퀀스를 **순차적으로** 처리한다:

$$h_t = f(h_{t-1}, x_t) \quad \text{— } h_t \text{는 } h_{t-1}에 의존$$

| 문제 | 설명 |
|---|---|
| 병렬화 불가 | $h_t$를 계산하려면 $h_{t-1}$이 필요 → GPU 병렬화 어려움 |
| 장거리 의존성 | 100 토큰 전의 정보가 $h$를 100번 거쳐야 도달 → 기울기 소실 |
| 메모리 병목 | 모든 과거 정보를 고정 크기 $h$에 압축 |

#### Self-Attention의 해결

$$\mathrm{output}_t = \sum_{j \leq t} \alpha_{tj} \cdot V_j \quad \text{— } t \text{에서 모든 과거 토큰에 직접 접근}$$

| 장점 | 설명 |
|---|---|
| 완전 병렬 | 모든 토큰 쌍을 동시에 계산 → GPU에 최적 |
| $O(1)$ 거리 | 어떤 두 토큰이든 단 한 번의 Attention으로 연결 |
| 동적 메모리 | 필요한 토큰에 선택적으로 집중 (고정 크기 $h$ 불필요) |

대가: $O(T^2)$ 계산량. 하지만 GPU의 병렬성으로 인해 실질적으로 RNN보다 훨씬 빠르다.

---

### Transformer 블록에서의 위치

CausalSelfAttention은 Transformer 블록의 절반이다. 나머지 절반은 FFN (Feed-Forward Network):

```
입력 x
  ↓
LayerNorm ──→ CausalSelfAttention ──→ + x (residual)
                                       ↓
              LayerNorm ──→ FFN(GELU) ──→ + (residual)
                                       ↓
                                     출력
```

이 블록을 $N$번 쌓으면 GPT의 본체:

| 모델 | 블록 수 ($N$) | $D$ | $H$ | $D_h$ |
|---|---|---|---|---|
| GPT-2 Small | 12 | 768 | 12 | 64 |
| GPT-2 Medium | 24 | 1024 | 16 | 64 |
| GPT-2 Large | 36 | 1280 | 20 | 64 |
| GPT-3 175B | 96 | 12288 | 96 | 128 |

---

### nanoGPT 로드맵

```
[x] Embedding, AdamW          ← step 61
[x] Transpose(axes)           ← step 62
[x] Batched Matmul            ← step 62
[x] Softmax(axis)             ← step 63
[x] Causal Mask               ← step 63
[x] LayerNorm                 ← step 64
[x] GELU                      ← step 64
[x] Multi-Head Attention      ← step 65 (이번)
[ ] GPT Block 통합 (Attention + FFN + Residual)
[ ] GPT 모델 (Embedding + N × Block + LM Head)
[ ] 텍스트 생성
```

다음은 **GPT Block** — CausalSelfAttention과 FFN을 LayerNorm, Residual Connection으로 감싸 하나의 반복 단위를 완성.
