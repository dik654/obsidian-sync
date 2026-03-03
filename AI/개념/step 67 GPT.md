## Step 67: GPT — 완전한 Decoder-only Transformer 언어 모델

### 한마디 직관

- **GPT = "다음 단어를 맞히는 기계"** — 이전 토큰들을 보고 다음 토큰의 확률을 출력
- Step 61~66의 모든 부품이 하나의 모델로 조립되는 마지막 단계

"The cat sat on the" → GPT → "mat" (0.35), "floor" (0.20), "bed" (0.15), ...

---

### GPT의 전체 구조

#### 아키텍처 다이어그램

```
토큰 인덱스 [42, 17, 99, 5]    (B, T)
         │
    ┌────┴────┐
    ↓         ↓
Token Emb   Pos Emb         [step 61]
    ↓         ↓
    └────+────┘              (B, T, D)
         │
   TransformerBlock × N      [step 66]
         │
      LayerNorm              [step 64]
         │
   Linear (LM Head)          D → vocab_size
         │
      logits                 (B, T, vocab_size)
```

수식으로:

$$h_0 = E_{\mathrm{tok}}[x] + E_{\mathrm{pos}}[0, 1, \ldots, T-1]$$
$$h_l = \mathrm{TransformerBlock}_l(h_{l-1}) \quad l = 1, \ldots, N$$
$$\mathrm{logits} = \mathrm{LN}(h_N) \cdot W_{\mathrm{lm}} + b_{\mathrm{lm}}$$

$\mathrm{logits}[b, t, v]$ = 위치 $t$에서 다음 토큰이 $v$일 점수. 이 점수에 softmax를 적용하면 확률 분포가 된다.

---

### 입력: Token Embedding + Position Embedding

#### 왜 두 가지 임베딩이 필요한가

**Token Embedding**만으로는 **순서 정보가 없다**. "cat sat on" 과 "on sat cat"이 동일하게 처리된다. Attention은 집합(set) 연산이므로 — $\sum_j \alpha_{ij} V_j$에서 순서가 관여하지 않는다.

RNN은 순차적으로 처리하므로 순서가 자연스럽게 인코딩된다:

$$h_t = f(h_{t-1}, x_t) \quad \text{— }t\text{가 곧 순서}$$

하지만 Transformer는 모든 토큰을 **동시에** 처리하므로, 순서 정보를 **명시적으로** 주입해야 한다. 이것이 Position Embedding의 역할이다.

#### Position Embedding의 구현

GPT-2/3는 **학습 가능한** position embedding을 사용한다:

$$E_{\mathrm{pos}} \in \mathbb{R}^{T_{\max} \times D}$$

위치 $t$의 임베딩 = $E_{\mathrm{pos}}[t]$, 이것은 Token Embedding과 동일한 룩업 연산이다. 단지 인덱스가 토큰 ID 대신 **위치 인덱스** $(0, 1, 2, \ldots)$일 뿐.

```rust
// 위치 인덱스 [0, 1, ..., T-1]을 Variable로
let pos_idx = Variable::new(ArrayD::from_shape_vec(
    IxDyn(&[t]),
    (0..t).map(|i| i as f64).collect(),
).unwrap());

let tok_emb = self.token_emb.forward(idx);      // (B, T, D)
let pos_emb = self.pos_emb.forward(&pos_idx);   // (T, D)
let x = &tok_emb + &pos_emb;                     // broadcast: (B, T, D)
```

#### 원래 Transformer의 Sinusoidal Positional Encoding과의 비교

Vaswani et al. (2017)은 학습하지 않는 **고정 사인파**를 사용했다:

$$\mathrm{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/D}}\right), \quad \mathrm{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/D}}\right)$$

왜 이런 형태인가? 각 차원 $i$가 다른 주파수의 사인파를 사용한다. 낮은 차원은 빠른 주기 (인접 토큰 구분), 높은 차원은 느린 주기 (먼 토큰 구분). 이진법과 유사한 원리다:

```
pos=0: 0 0 0 0
pos=1: 1 0 0 0     ← 최하위 비트만 변화
pos=2: 0 1 0 0
pos=3: 1 1 0 0
pos=4: 0 0 1 0     ← 더 높은 비트가 변화
```

사인파를 쓰면 이 이진법적 인코딩을 연속적으로 만들 수 있다.

또한 $\mathrm{PE}(t+k)$를 $\mathrm{PE}(t)$의 선형 변환으로 표현할 수 있어, 모델이 상대 위치를 학습하기 쉽다:

$$\mathrm{PE}(t+k) = M_k \cdot \mathrm{PE}(t) \quad \text{(회전 행렬)}$$

**학습 가능한 임베딩이 더 나은 이유**: 실험적으로 둘의 성능 차이는 거의 없지만, 학습 가능한 임베딩은 데이터에 맞는 최적의 위치 표현을 찾을 수 있다. GPT-2/3는 학습 가능한 버전을 채택. 단점은 $T_{\max}$를 초과하는 시퀀스에 대해 일반화할 수 없다는 것 (학습하지 않은 위치이므로).

#### block_size의 의미

Position Embedding의 크기가 $T_{\max}$ (= `block_size`)이므로, 모델은 이 길이까지의 시퀀스만 처리할 수 있다. GPT-2는 1024, GPT-3는 2048, GPT-4는 128K.

생성 시 `block_size`를 초과하면 가장 최근 `block_size`개 토큰만 사용한다:

```rust
let start = if tokens.len() > self.block_size {
    tokens.len() - self.block_size
} else { 0 };
let ctx = &tokens[start..];  // 최대 block_size 토큰
```

이것은 **슬라이딩 윈도우(sliding window)** 방식이다. 이전 문맥은 잘린다.

---

### 학습: 다음 토큰 예측 (Next Token Prediction)

#### 학습 목표

입력 시퀀스 $[x_0, x_1, \ldots, x_{T-1}]$에서 각 위치의 다음 토큰을 예측:

| 입력 위치 | 보이는 토큰 | 예측 대상 |
|---|---|---|
| $t=0$ | $x_0$ | $x_1$ |
| $t=1$ | $x_0, x_1$ | $x_2$ |
| $t=2$ | $x_0, x_1, x_2$ | $x_3$ |
| $\vdots$ | | |
| $t=T-1$ | $x_0, \ldots, x_{T-1}$ | $x_T$ |

한 번의 forward pass로 $T$개의 예측을 **동시에** 수행한다 (causal mask 덕분). 이것이 RNN 대비 Transformer의 학습 효율이 높은 핵심 이유다.

#### Loss: Cross-Entropy

$$L = -\frac{1}{T}\sum_{t=0}^{T-1} \log P(x_{t+1} \mid x_0, \ldots, x_t)$$

$P(x_{t+1} = v \mid \cdot) = \mathrm{softmax}(\mathrm{logits}[t])_v$

이것은 이미 step 47에서 구현한 `softmax_cross_entropy_simple`과 동일하다. 3D logits을 2D로 펼쳐서 사용:

```rust
let logits = gpt.forward(&idx);                    // (B, T, V)
let logits_2d = reshape(&logits, &[B * T, V]);     // (B*T, V)
let targets: Vec<usize> = /* B*T개의 정답 토큰 */;
let loss = softmax_cross_entropy_simple(&logits_2d, &targets);
```

#### 왜 Cross-Entropy인가

Cross-entropy는 두 확률 분포 사이의 "거리"를 측정한다:

$$H(p, q) = -\sum_v p(v) \log q(v)$$

여기서 $p$는 정답 분포 (one-hot: 정답 토큰만 1, 나머지 0), $q$는 모델의 예측 확률.

$p$가 one-hot이면:

$$H(p, q) = -\log q(x_{\mathrm{correct}})$$

정답 토큰에 부여한 확률의 **음의 로그**. 확률이 높을수록 (→ 1) loss가 낮아지고 (→ 0), 확률이 낮을수록 (→ 0) loss가 커진다 (→ ∞).

Cross-entropy를 최소화하는 것은 **최대 우도 추정(MLE)**과 동일하다:

$$\arg\min_\theta H(p, q_\theta) = \arg\max_\theta \prod_{t} P_\theta(x_{t+1} \mid x_{\leq t})$$

#### Perplexity: 언어 모델의 표준 지표

$$\mathrm{PPL} = e^{L} = e^{-\frac{1}{T}\sum \log P(x_{t+1})}$$

직관: "모델이 각 위치에서 평균적으로 몇 개의 후보 중에서 고르는 느낌인가"

| PPL | 의미 |
|---|---|
| 1 | 완벽한 예측 (확률 1) |
| $V$ (vocab size) | 랜덤 추측 (균등 분포) |
| 20~30 | 좋은 언어 모델 (GPT-2 수준) |

초기 loss가 $\ln(V) \approx 2.3$ (vocab=10)인 것은 모델이 균등 분포에서 시작함을 의미. 학습이 진행되면 loss가 0에 가까워진다 (특정 토큰에 높은 확률 부여).

우리 실험에서: loss 2.0 → 0.005 = PPL 7.4 → 1.005. 거의 완벽한 예측.

---

### 생성: 자기회귀 디코딩 (Autoregressive Decoding)

#### 기본 원리

학습된 GPT는 $P(x_{t+1} \mid x_0, \ldots, x_t)$를 출력한다. 이 확률에서 토큰을 하나 **샘플링**하고, 그 토큰을 입력에 추가하여 다시 forward pass를 수행한다:

```
시작: [a]
  → GPT → P(next | a) → argmax → 'b'

[a, b]
  → GPT → P(next | a, b) → argmax → 'c'

[a, b, c]
  → GPT → P(next | a, b, c) → argmax → 'a'

...
```

각 스텝에서:
1. 현재까지의 토큰 시퀀스를 GPT에 입력
2. **마지막 위치**의 logit만 사용 (이전 위치는 이미 결정됨)
3. logit에서 다음 토큰 선택
4. 선택한 토큰을 시퀀스에 추가
5. 반복

#### Greedy Decoding vs Sampling

**Greedy (우리 구현)**: 항상 가장 높은 확률의 토큰을 선택.

$$x_{t+1} = \arg\max_v \mathrm{logits}[t, v]$$

장점: 결정적(deterministic), 구현 간단.
단점: 항상 같은 출력, 다양성 없음, 반복에 빠지기 쉬움.

**Temperature Sampling**: logits를 temperature $\tau$로 나눈 뒤 확률적으로 선택.

$$P(v) = \mathrm{softmax}\!\left(\frac{\mathrm{logits}}{\tau}\right)_v, \quad x_{t+1} \sim P$$

$\tau = 0$ → greedy, $\tau = 1$ → 원래 분포, $\tau > 1$ → 더 균등 (창의적).

Step 63에서 softmax의 temperature 해석을 다뤘다 — 그것이 바로 여기에 적용된다.

**Top-k Sampling**: 상위 $k$개 토큰만 후보로 두고 나머지는 확률 0.

$$P(v) \propto \begin{cases} \mathrm{softmax}(\mathrm{logits})_v & v \in \mathrm{top\text{-}k} \\ 0 & \text{otherwise} \end{cases}$$

**Nucleus (Top-p) Sampling**: 누적 확률이 $p$를 넘지 않는 최소 집합에서 샘플링.

실제 ChatGPT 등에서는 temperature + top-p를 조합하여 사용한다.

#### KV Cache: 생성 속도 최적화

현재 구현은 매 스텝마다 **전체 시퀀스를 다시 계산**한다. 시퀀스가 길어지면 $O(T^2)$ 연산이 반복되어 $O(T^3)$이 된다.

KV Cache는 이전 스텝에서 계산한 K, V를 **캐시**하여 재사용한다:

| | KV Cache 없이 | KV Cache 사용 |
|---|---|---|
| 스텝 1 | Q,K,V 전체 계산 (T=1) | Q,K,V 전체 계산 (T=1) |
| 스텝 2 | Q,K,V 전체 재계산 (T=2) | 새 토큰의 Q만 계산, 캐시된 K,V에 추가 |
| 스텝 $n$ | $O(n^2)$ | $O(n)$ |
| 총 | $O(T^3)$ | $O(T^2)$ |

구현 원리: Attention에서 $K$ = [이전 캐시 K ; 새 K], $V$ = [이전 캐시 V ; 새 V]로 연결(concat)하고, $Q$는 마지막 토큰 하나만 계산.

이것은 우리의 "밑바닥" 구현에서는 생략했지만, 실제 프로덕션 LLM에서는 필수적인 최적화다.

---

### Weight Tying: 파라미터 공유

GPT-2/3에서는 Token Embedding의 가중치 $E_{\mathrm{tok}}$와 LM Head의 가중치 $W_{\mathrm{lm}}$을 **공유(tie)**한다:

$$W_{\mathrm{lm}} = E_{\mathrm{tok}}^T$$

왜 가능한가?
- Token Embedding: 토큰 ID → 의미 벡터 ($V \times D$)
- LM Head: 의미 벡터 → 토큰 점수 ($D \times V$)

두 연산이 **역방향**이므로, 같은 가중치의 전치를 쓰는 것이 자연스럽다. "cat"의 임베딩 벡터와 "cat" 토큰의 예측 가중치가 같다면, 유사한 의미의 벡터는 "cat"에 높은 점수를 부여한다.

효과:
- 파라미터 수 $V \times D$만큼 절약 (vocab_size가 크면 상당함)
- 입력 공간과 출력 공간이 동일하게 정렬 → 학습 효율 향상
- 실험적으로 거의 항상 성능 향상

우리 구현에서는 단순성을 위해 분리된 가중치를 사용했지만, 실제 GPT 구현에서는 tying이 표준.

---

### 실험 결과: Char-level "abc" 학습

#### 학습 설정

```
텍스트: "abcabcabcabcabcabc"
vocab: {a, b, c} → vocab_size = 3
모델: D=16, H=2, N=2, block_size=32
옵티마이저: AdamW(lr=0.005, wd=0.0)
시퀀스 길이: 6
```

#### 학습 곡선

| epoch | loss | PPL | 의미 |
|---|---|---|---|
| 1 | 2.0 | 7.4 | 거의 랜덤 (3개 중 7.4개?  → loss가 ln(3)=1.1보다 높음, 초기 가중치 때문) |
| 20 | 0.04 | 1.04 | 패턴 거의 학습 |
| 60 | 0.007 | 1.007 | 거의 완벽 |

#### 생성 결과

```
시작: "a"
생성: "abcabcacacac"
```

"abc" 패턴을 성공적으로 학습. 완벽하지는 않지만 (ac 반복이 섞임), 3개 문자의 주기 패턴을 포착했다. 더 긴 학습이나 더 큰 모델로 개선 가능.

---

### GPT-1 → GPT-4: 같은 구조, 다른 스케일

놀라운 사실: GPT-1(2018)부터 GPT-4(2023)까지 **기본 아키텍처는 동일**하다. Token Embedding + Position Embedding + TransformerBlock × N + LM Head. 변한 것은 스케일과 학습 데이터:

| 모델 | 연도 | 파라미터 | 데이터 | block_size |
|---|---|---|---|---|
| GPT-1 | 2018 | 117M | BookCorpus (5GB) | 512 |
| GPT-2 | 2019 | 1.5B | WebText (40GB) | 1024 |
| GPT-3 | 2020 | 175B | 570GB 혼합 | 2048 |
| GPT-4 | 2023 | ~1.8T (추정) | ~13T 토큰 (추정) | 128K |

아키텍처 변화보다 **scaling이 성능을 결정**한다는 것이 Kaplan et al.의 Scaling Law의 핵심 발견이다. 우리가 구현한 코드와 GPT-4의 코드 사이에 근본적인 알고리즘 차이는 없다 — 스케일, 데이터, 학습 인프라의 차이일 뿐이다.

물론 세부 개선은 있다:
- **RoPE** (Rotary Position Embedding): 상대 위치 인코딩 → 긴 시퀀스 일반화
- **GQA** (Grouped Query Attention): KV 캐시 메모리 절약
- **Flash Attention**: 정확한 Attention을 메모리 효율적으로
- **MoE** (Mixture of Experts): 파라미터를 늘리되 연산은 적게
- **RLHF/DPO**: 인간 선호에 맞게 정렬

하지만 **뼈대**는 우리가 step 61~67에서 구현한 것과 동일하다.

---

### 코드 발췌

```rust
pub struct GPT {
    token_emb: Embedding,           // vocab_size → D
    pos_emb: Embedding,             // block_size → D
    blocks: Vec<TransformerBlock>,  // N개 블록
    ln_f: LayerNorm,                // 최종 정규화
    lm_head: Linear,                // D → vocab_size
    block_size: usize,
}

pub fn forward(&self, idx: &Variable) -> Variable {
    let (b, t) = (shape[0], shape[1]);

    // Token + Position Embedding
    let tok_emb = self.token_emb.forward(idx);
    let pos_idx = Variable::new(/* [0, 1, ..., T-1] */);
    let pos_emb = self.pos_emb.forward(&pos_idx);
    let mut x = &tok_emb + &pos_emb;

    // Transformer Blocks × N
    for block in &self.blocks {
        x = block.forward(&x);
    }

    // LN → LM Head
    x = self.ln_f.forward(&x);
    let x_2d = reshape(&x, &[b * t, self.n_embd]);
    let logits = self.lm_head.forward(&x_2d);
    reshape(&logits, &[b, t, self.vocab_size])
}

pub fn generate(&self, start_tokens: &[usize], max_new_tokens: usize) -> Vec<usize> {
    let _guard = no_grad();
    let _test = test_mode();
    let mut tokens = start_tokens.to_vec();

    for _ in 0..max_new_tokens {
        // 최근 block_size 토큰만 사용
        let ctx = &tokens[tokens.len().saturating_sub(self.block_size)..];
        let logits = self.forward(&/* ctx as Variable */);
        // 마지막 위치의 logit에서 argmax
        let next_token = argmax(&logits[0, -1, :]);
        tokens.push(next_token);
    }
    tokens
}
```

---

### nanoGPT 로드맵: Phase 1 완료!

```
[x] Embedding, AdamW          ← step 61
[x] Transpose(axes)           ← step 62
[x] Batched Matmul            ← step 62
[x] Softmax(axis)             ← step 63
[x] Causal Mask               ← step 63
[x] LayerNorm                 ← step 64
[x] GELU                      ← step 64
[x] Multi-Head Attention      ← step 65
[x] TransformerBlock           ← step 66
[x] GPT 모델                  ← step 67 (이번)
[x] char-level 학습 + 생성    ← step 67 (이번)
```

**밑바닥부터 GPT를 구현 완료.** Token Embedding부터 자기회귀 생성까지, 모든 연산의 순전파와 역전파를 직접 구현했다.
