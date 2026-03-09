## Step 70: Sentence Embedding — Contrastive Learning (NT-Xent)

### 한마디 직관

- **Sentence Embedding = "비슷한 의미의 문장은 비슷한 벡터"** — Word2Vec의 단어 수준 아이디어를 문장 수준으로 확장
- BERT 인코더 + Mean Pooling + NT-Xent Loss: 단어 벡터를 넘어 **문장 전체의 의미**를 하나의 벡터로 압축
- Word2Vec(2013) → SBERT(2019) → SimCLR(2020) → CLIP(2021)으로 이어지는 contrastive learning 계보의 핵심

"The meaning of a sentence is determined by how it relates to other sentences." — 분포 가설의 문장 수준 확장

---

### 왜 문장 임베딩이 필요한가

#### 단어 벡터의 한계

Word2Vec은 단어를 벡터로 잘 표현하지만, **문장**의 의미를 포착하지 못한다:

```
"강아지가 고양이를 쫓는다" vs "고양이가 강아지를 쫓는다"
→ 같은 단어 집합이지만 의미가 완전히 다름

"은행에 돈을 맡겼다" vs "강둑의 은행에 앉았다"
→ "은행"이 같은 벡터를 갖지만, 맥락에 따라 의미가 달라짐
```

단어 벡터의 단순 합산(BoW)이나 평균은 **어순, 구문 구조, 문맥 의존적 의미**를 모두 잃는다.

#### 문장 임베딩의 요구사항

좋은 문장 벡터는 다음을 만족해야 한다:

1. **의미 보존**: "The cat sat on the mat" ≈ "A feline was sitting on the rug"
2. **구분력**: "The dog bit the man" ≠ "The man bit the dog"
3. **효율성**: 추론 시 문장 하나에 대해 벡터 하나를 생성, 유사도는 단순 내적

이 조건을 만족하면 **검색, 분류, 클러스터링, RAG** 등에 직접 사용 가능하다.

---

### BERT를 문장 인코더로 활용하기

#### BERT의 출력 구조

BERT는 입력 토큰 시퀀스 $(x_1, x_2, \ldots, x_T)$에 대해 **토큰별** hidden state를 출력한다:

$$H = \mathrm{BERT}(x_1, \ldots, x_T) \in \mathbb{R}^{T \times D}$$

문장 하나에 대해 $T$개의 $D$차원 벡터가 나온다. 이것을 **하나의 벡터**로 줄여야 한다.

#### Pooling 전략: [CLS] vs Mean Pooling

**[CLS] 토큰 사용:**
$$\vec{s} = H[0] \quad (D\text{차원})$$

BERT는 원래 [CLS] 토큰에 문장 전체 정보를 집약하도록 사전학습되었다 (NSP task). 하지만 Reimers & Gurevych (2019)는 이 벡터가 **코사인 유사도에 최적화되지 않았다**는 것을 보였다. 실제로 [CLS]의 코사인 유사도는 GloVe 평균보다 더 나쁘다!

**Mean Pooling:**
$$\vec{s} = \frac{1}{T} \sum_{t=1}^{T} H[t] \quad (D\text{차원})$$

모든 토큰의 hidden state를 평균. 실험적으로 [CLS]보다 더 좋은 결과를 보인다 (SBERT).

왜 평균이 더 좋은가? BERT의 사전학습 목표(MLM)는 **각 토큰 위치에 정보를 분산 저장**하도록 유도한다. [CLS]에만 집중하면 대부분의 정보를 버리게 된다. 평균은 이 분산된 정보를 **모두 활용**한다.

우리 구현에서:

```rust
// SentenceEmbedding.encode()
let summed = sum_with(&x, Some(1), false);  // (B, T, D) → (B, D)
let t_inv = Variable::new(arr0(1.0 / t as f64).into_dyn());
&summed * &t_inv  // mean pooling
```

---

### Contrastive Learning의 핵심 원리

#### 직관: 정렬(Alignment)과 균등(Uniformity)

Wang & Isola (2020)은 좋은 representation이 만족해야 할 두 가지 성질을 정의했다:

1. **Alignment** $\mathcal{L}_\mathrm{align}$: 양의 쌍은 가까워야 한다

$$\mathcal{L}_\mathrm{align} = \mathbb{E}_{(x, x^+) \sim p_\mathrm{pos}} \left[\|f(x) - f(x^+)\|^2\right]$$

2. **Uniformity** $\mathcal{L}_\mathrm{uniform}$: 전체 분포는 초구면에 균등해야 한다

$$\mathcal{L}_\mathrm{uniform} = \log \mathbb{E}_{x, y \stackrel{\mathrm{i.i.d.}}{\sim} p_\mathrm{data}} \left[e^{-2\|f(x) - f(y)\|^2}\right]$$

두 번째 조건이 없으면 모든 벡터가 한 점으로 **붕괴(collapse)**한다 — "모든 문장이 같은 벡터" → alignment은 완벽하지만 무의미.

```
나쁜 해: alignment만 최적화              좋은 해: alignment + uniformity
┌─────────────────────────┐     ┌─────────────────────────┐
│                         │     │  ·       ·              │
│         ···· ←모든 점   │     │     · ·    · ·          │
│        (collapse)       │     │  ·              ·       │
│                         │     │     ·   ·   ·    ·      │
│                         │     │        ·       ·        │
└─────────────────────────┘     └─────────────────────────┘
```

Contrastive loss는 이 두 목표를 **하나의 손실 함수**로 자연스럽게 결합한다.

#### Collapse의 수학: 왜 alignment만으로는 안 되는가

Alignment만 최소화하는 trivial solution을 분석한다. 인코더 $f$가 상수 함수 $f(x) = c$ (모든 입력에 같은 벡터)이면:

$$\mathcal{L}_\mathrm{align} = \mathbb{E}\left[\|c - c\|^2\right] = 0 \quad \text{(완벽한 alignment)}$$

하지만 이 표현은 **입력에 대한 정보를 전혀 보존하지 않는다**. Contrastive loss는 분모의 $\sum_j \exp(s_j/\tau)$가 **uniformity를 암묵적으로 강제**한다:

$$\mathcal{L} = -\log\frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$$

만약 모든 $s_{ij}$가 동일하면 (collapse 상태): $\mathcal{L} = -\log\frac{1}{N} = \log N$. 이것은 **최대 loss**이다. Loss를 줄이려면 $s_{ii}$를 다른 $s_{ij}$보다 크게 만들어야 하는데, 이는 양의 쌍을 다른 쌍과 **구별**해야 함을 의미한다.

$$\frac{\partial \mathcal{L}}{\partial s_{ij}} = \frac{1}{N}\left(\mathrm{softmax}(S_i)_j - \delta_{ij}\right) \neq 0 \quad \text{unless perfect separation}$$

분모가 부정 샘플들의 유사도 합산이므로, 부정 쌍의 유사도를 낮추는 방향으로 gradient가 흐른다. 이것이 **uniformity를 자연스럽게 유도**하는 메커니즘이다.

---

### NT-Xent Loss: 수학적 유도

#### 배경: InfoNCE (Noise-Contrastive Estimation)

van den Oord et al. (2018, CPC 논문)이 제안한 InfoNCE loss:

$$\mathcal{L}_\mathrm{InfoNCE} = -\mathbb{E}\left[\log \frac{\exp(f(x)^\top f(x^+) / \tau)}{\sum_{j=0}^{N-1} \exp(f(x)^\top f(x_j) / \tau)}\right]$$

이것은 $(N-1)$-way 분류 문제이다: $N$개 후보 중 진짜 양의 쌍을 찾아내는 것.

**정보 이론적 해석**: InfoNCE의 최적값은 양의 쌍 사이의 **상호 정보(mutual information)**에 대한 하한이다:

$$\mathcal{L}_\mathrm{InfoNCE} \geq -I(x; x^+) + \log N$$

따라서 InfoNCE를 최소화하면 $I(x; x^+)$의 **하한**이 최대화된다. 배치 크기 $N$이 클수록 더 타이트한 하한을 제공한다.

#### NT-Xent: 대칭 InfoNCE

SimCLR (Chen et al., 2020)은 InfoNCE를 **양방향으로 대칭화**한 NT-Xent를 사용한다.

배치 내 $N$개의 양의 쌍 $(z_a^{(i)}, z_b^{(i)})$에 대해:

**1단계: L2 정규화**

$$\hat{z} = \frac{z}{\|z\|_2}$$

이것은 모든 벡터를 **단위 초구면** 위로 투사한다. 이후 내적이 곧 코사인 유사도가 된다.

**2단계: 유사도 행렬**

$$S_{ij} = \frac{\hat{z}_a^{(i)} \cdot \hat{z}_b^{(j)}}{\tau}$$

$S$는 $N \times N$ 행렬. 대각선 $S_{ii}$가 양의 쌍의 유사도.

**3단계: 대칭 Cross-Entropy**

$$\mathcal{L} = \frac{1}{2} \left[\mathcal{L}_{a \to b} + \mathcal{L}_{b \to a}\right]$$

여기서:
$$\mathcal{L}_{a \to b} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}$$

$$\mathcal{L}_{b \to a} = -\frac{1}{N} \sum_{j=1}^{N} \log \frac{\exp(S_{jj})}{\sum_{i=1}^{N} \exp(S_{ij})}$$

$\mathcal{L}_{a \to b}$는 "각 $z_a^{(i)}$에 대해 올바른 $z_b^{(i)}$를 찾아라"이고, $\mathcal{L}_{b \to a}$는 그 반대 방향이다. **CLIP**은 정확히 이 대칭 구조를 이미지-텍스트 쌍에 적용한 것이다.

#### 왜 대칭인가

비대칭 loss를 사용하면 한쪽 인코더만 업데이트가 잘 된다:
- $\mathcal{L}_{a \to b}$만 사용 → $z_a$의 gradient만 강하게 흐름
- $z_b$ 인코더는 상대적으로 학습이 느림

대칭 loss는 양쪽 인코더를 **균등하게** 학습시킨다. 우리 구현처럼 같은 인코더를 공유하는 경우에도, 대칭화는 **gradient 방향의 균형**을 맞추어 학습 안정성을 높인다.

---

### NT-Xent의 Backward 유도

#### Step 1: Cross-Entropy → dL/dS

$\mathcal{L}_{a \to b}$의 행 $i$에서 softmax를 $p_{ij} = \mathrm{softmax}(S_i)_j$로 쓰면:

$$\frac{\partial \mathcal{L}_{a \to b}}{\partial S_{ij}} = \frac{1}{N}(p_{ij} - \delta_{ij})$$

$\delta_{ij}$는 크로네커 델타. 이것은 표준 softmax cross-entropy gradient와 동일하다.

$\mathcal{L}_{b \to a}$는 $S^T$에 대한 cross-entropy이므로, $S$에 대한 gradient는 전치된다:

$$\frac{\partial \mathcal{L}_{b \to a}}{\partial S_{ij}} = \frac{1}{N}(q_{ij} - \delta_{ij})$$

여기서 $q_{ij} = \mathrm{softmax}(S^\top_j)_i$ (열 방향 softmax).

합산하면:

$$\frac{\partial \mathcal{L}}{\partial S_{ij}} = \frac{1}{2}\left[\frac{p_{ij} - \delta_{ij}}{N} + \frac{q_{ij} - \delta_{ij}}{N}\right]$$

#### Step 2: S → dL/dẑ (정규화된 벡터)

$S = \hat{Z}_a \hat{Z}_b^\top / \tau$이므로:

$$\frac{\partial \mathcal{L}}{\partial \hat{z}_a^{(i)}} = \frac{1}{\tau} \sum_j \frac{\partial \mathcal{L}}{\partial S_{ij}} \hat{z}_b^{(j)}$$

행렬 형태: $\nabla_{\hat{Z}_a} \mathcal{L} = \frac{1}{\tau} \frac{\partial \mathcal{L}}{\partial S} \hat{Z}_b$

대칭적으로: $\nabla_{\hat{Z}_b} \mathcal{L} = \frac{1}{\tau} \left(\frac{\partial \mathcal{L}}{\partial S}\right)^\top \hat{Z}_a$

#### Step 3: ẑ → dL/dz (L2 정규화 역전파)

$\hat{z} = z / \|z\|$의 야코비안:

$$\frac{\partial \hat{z}_k}{\partial z_l} = \frac{1}{\|z\|}\left(\delta_{kl} - \hat{z}_k \hat{z}_l\right)$$

이것은 **단위 벡터 $\hat{z}$에 직교하는 공간으로의 투사(projection)**이다!

$$\nabla_z \mathcal{L} = \frac{1}{\|z\|}\left(g - \hat{z}(\hat{z}^\top g)\right)$$

여기서 $g = \nabla_{\hat{z}} \mathcal{L}$. 직관적으로, L2 정규화의 backward는 gradient에서 **현재 방향 성분을 제거**한다. 단위 구면 위에서의 이동만 허용하는 것이다.

```rust
// L2 norm backward 구현
for k in 0..d {
    g_a[[i, k]] = (g_a_hat[[i, k]] - z_a_hat[[i, k]] * dot_a) / norms_a[i] * gy_val;
}
```

---

### 온도 파라미터 τ의 역할

온도 $\tau$는 유사도 행렬 $S$의 **엔트로피**를 제어한다:

$$p_{ij} = \frac{\exp(\cos(\hat{z}_a^{(i)}, \hat{z}_b^{(j)}) / \tau)}{\sum_k \exp(\cos(\hat{z}_a^{(i)}, \hat{z}_b^{(k)}) / \tau)}$$

| $\tau$ | softmax 분포 | 학습 특성 |
|---|---|---|
| $\tau \to 0$ | one-hot (가장 유사한 것만 1) | **hard negative**에 극도로 집중, gradient 폭발 위험 |
| $\tau = 0.07$ | CLIP 기본값 | 실무에서 가장 많이 사용 |
| $\tau = 0.5$ | 부드러운 분포 | 학습 안정적, 구분력은 낮음 |
| $\tau \to \infty$ | 균등 분포 | 모든 부정 샘플을 동등하게 취급, 학습 불가 |

**$\tau$의 수학적 효과**: $1/\tau$는 폰 미제스-피셔 분포(von Mises-Fisher distribution)의 **집중 모수(concentration parameter) $\kappa$**와 동일한 역할을 한다. 이 분포는 단위 구면 위의 가우시안에 해당하며, $\kappa$가 클수록 (= $\tau$가 작을수록) 평균 방향 주위에 **좁게 집중**된다.

실전 팁: $\tau$를 학습 가능한 파라미터로 두는 방법도 있다 (CLIP). 이 경우 $\tau$도 gradient descent로 최적화되며, 학습 초기에는 크게 시작하여 점차 줄어드는 경향이 있다.

#### τ와 Gradient 크기: Hard Negative에 대한 집중 효과

온도가 gradient의 **크기 분포**를 어떻게 바꾸는지 분석한다.

$\mathcal{L}_{a \to b}$의 부정 샘플 $j$ ($j \neq i$)에 대한 gradient 크기:

$$\left|\frac{\partial \mathcal{L}}{\partial S_{ij}}\right| = \frac{1}{N} \cdot p_{ij} = \frac{1}{N} \cdot \frac{\exp(\cos_{ij}/\tau)}{\sum_k \exp(\cos_{ik}/\tau)}$$

$\tau$가 작으면 softmax가 날카로워져서, **가장 유사한 부정 샘플**(hard negative)에 gradient가 집중된다:

| 부정 샘플 유사도 | $\tau = 1.0$ | $\tau = 0.1$ | $\tau = 0.01$ |
|---|---|---|---|
| $\cos = 0.9$ (hard) | 0.30 | 0.88 | $\approx 1.0$ |
| $\cos = 0.5$ | 0.20 | 0.12 | $\approx 0.0$ |
| $\cos = 0.1$ (easy) | 0.14 | 0.00 | $\approx 0.0$ |

$\tau \to 0$에서 gradient는 **가장 어려운 부정 샘플 하나**에만 집중된다 (hard negative mining의 자동화). 하지만 너무 작으면 gradient가 불안정해진다: $1/\tau$가 gradient 스케일에 곱해지므로 gradient 폭발 위험.

$$\frac{\partial \mathcal{L}}{\partial \hat{z}_a} = \frac{1}{\tau} \frac{\partial \mathcal{L}}{\partial S} \hat{Z}_b$$

$\tau = 0.01$이면 gradient가 100배 증폭된다. 이것이 작은 $\tau$에서 학습률을 줄여야 하는 이유다.

---

### Projection Head: 왜 필요하고 왜 버리는가

#### SimCLR의 발견

Chen et al. (2020)은 인코더 출력 $h$에 비선형 projection head $g(h)$를 추가하는 것이 **downstream 성능을 크게 향상**시킨다는 것을 발견했다:

| 설정 | Linear eval accuracy |
|---|---|
| $h$ 직접 사용 | 64.5% |
| Linear projection $W h$ | 66.3% |
| MLP projection $W_2 \sigma(W_1 h)$ | **69.3%** |

그런데 놀라운 것은: **$g(h)$가 아닌 $h$를 downstream에 사용**했을 때가 더 좋다!

#### 왜 이런 현상이 발생하는가

Projection head는 contrastive loss를 위한 **정보 병목**을 형성한다:

$$h \xrightarrow{g(\cdot)} z \quad \dim(h) > \dim(z)$$

$z$는 contrastive task에 **필요한 정보만** 보존하고, 나머지는 버린다. 예를 들어, 색상 변환(data augmentation)에 불변해야 하므로 $z$에서 색상 정보가 제거된다.

반면 $h$는 **풍부한 정보를 모두 보존**한다. Downstream task는 색상 정보를 필요로 할 수 있으므로, $h$가 더 범용적이다.

**수학적 관점**: contrastive loss는 $z$ 공간에서 데이터 증강에 불변한 representation을 학습한다. 이 불변성은 projection head에 의해 **분리**된다: 불변 정보는 $h$와 $z$ 모두에 포함되지만, 변형 정보는 $h$에만 남고 $z$에서는 제거된다.

우리 구현:

```rust
pub struct SentenceEmbedding {
    // ... encoder components ...
    projection: Linear,    // D → proj_dim (contrastive용, 학습 시만 사용)
}

// 학습 시: forward() → projection 포함 → NT-Xent loss
pub fn forward(&self, token_ids, segment_ids) -> Variable { /* ... projection 포함 */ }

// 추론 시: encode() → projection 전 → 문장 벡터로 사용
pub fn encode(&self, token_ids, segment_ids) -> Variable { /* ... mean pooling까지 */ }
```

---

### Contrastive Learning의 계보

Word2Vec에서 CLIP까지, contrastive learning의 핵심 아이디어는 동일하다: **양의 쌍은 가깝게, 부정 쌍은 멀리.**

| 모델 | 연도 | 양의 쌍 정의 | 부정 샘플 전략 | 핵심 기여 |
|---|---|---|---|---|
| **Word2Vec** | 2013 | (center, context) 동시 출현 | Unigram^0.75 분포 | 부정 샘플링으로 softmax 대체 |
| **SimCLR** | 2020 | 같은 이미지의 두 augmentation | 배치 내 다른 이미지 | projection head, 큰 배치 |
| **SBERT** | 2019 | (문장, 패러프레이즈) 쌍 | 배치 내 다른 문장 | BERT + mean pool + contrastive |
| **CLIP** | 2021 | (이미지, 설명 텍스트) 쌍 | 배치 내 다른 쌍 | 대칭 CE, zero-shot 전이 |
| **SimCSE** | 2021 | 같은 문장, dropout만 다르게 | 배치 내 다른 문장 | 데이터 없이 contrastive 학습 |

공통 패턴:
1. 양의 쌍의 **정의**가 다를 뿐, loss 구조는 동일 (InfoNCE/NT-Xent)
2. **배치 내 부정 샘플(in-batch negatives)**: 별도의 부정 샘플링 없이, 같은 미니배치의 다른 샘플을 부정으로 사용. 배치 크기가 곧 부정 샘플 수.
3. 배치가 클수록 더 많은 부정 샘플 → **더 타이트한 MI 하한** → 더 좋은 학습

#### Word2Vec vs NT-Xent: 구조적 비교

$$\underbrace{-\log\sigma(u_{\mathrm{pos}}^\top v) - \sum_k \log\sigma(-u_{\mathrm{neg}}^\top v)}_{\text{Word2Vec SGNS}} \quad \longleftrightarrow \quad \underbrace{-\log\frac{\exp(s_{\mathrm{pos}}/\tau)}{\sum_j \exp(s_j/\tau)}}_{\text{NT-Xent}}$$

왼쪽은 이진 분류(sigmoid), 오른쪽은 다중 분류(softmax). 본질은 같지만:
- SGNS: 각 부정 샘플을 **독립적**으로 처리 → 부정 샘플 간 상호작용 없음
- NT-Xent: 모든 부정 샘플을 **동시에 비교** (softmax의 분모) → 더 어려운 분류 문제 → 더 좋은 representation

---

### SBERT (Sentence-BERT)

Reimers & Gurevych (2019)의 핵심 아이디어: 사전학습된 BERT를 **문장 임베딩에 최적화**하여 fine-tuning.

#### 문제: BERT의 cross-encoder 병목

BERT로 두 문장의 유사도를 측정하려면:

```
[CLS] 문장A [SEP] 문장B [SEP] → BERT → 유사도 점수
```

$N$개 문장 쌍의 유사도를 구하려면 BERT를 $N^2$번 호출해야 한다! 10,000개 문장 → 1억 번의 BERT forward — 실용적이지 않다.

#### 해결: bi-encoder + contrastive learning

```
문장A → BERT → mean pool → 벡터 a     → cos(a, b)
문장B → BERT → mean pool → 벡터 b  ─┘
```

각 문장을 **독립적으로** 인코딩한 후 코사인 유사도를 계산. $N$개 문장 → $N$번의 BERT forward + $O(N^2)$의 내적 계산 (매우 빠름).

SBERT의 학습:
1. NLI 데이터셋의 (premise, entailment) 쌍을 양의 쌍으로 사용
2. Contrastive loss (또는 triplet loss)로 fine-tuning
3. 학습 후 mean-pooled 벡터를 문장 임베딩으로 사용

우리 구현의 `SentenceEmbedding`은 이 SBERT 아키텍처를 간소화한 것이다.

---

### BERT 임베딩의 비등방성(Anisotropy) 문제

#### 현상: 코사인 유사도가 0.99

Ethayarajh (2019)는 충격적인 발견을 보고했다: 사전학습된 BERT의 hidden state들은 **좁은 원뿔(cone)** 안에 몰려 있어, 무작위 두 문장의 코사인 유사도가 0.95~0.99에 달한다.

$$\mathbb{E}[\cos(\mathrm{BERT}(s_1), \mathrm{BERT}(s_2))] \approx 0.95 \quad \text{(무작위 문장 쌍)}$$

모든 문장 벡터가 거의 같은 방향을 가리키므로, 코사인 유사도로는 **의미 차이를 구분할 수 없다**. 유사도 범위가 [0.93, 0.99]처럼 극도로 좁아진다.

#### 원인: 단어 빈도의 지배

이 현상의 핵심 원인은 **고빈도 단어의 지배**이다. BERT의 hidden state는 입력 토큰의 임베딩에 크게 의존하는데, "the", "is", "a" 같은 고빈도 단어가 거의 모든 문장에 포함된다. 이 단어들의 임베딩이 hidden state를 **공통 방향으로 편향**시킨다.

수학적으로, 문장 벡터 $s = \frac{1}{T}\sum_t h_t$에서:

$$s \approx \underbrace{\frac{1}{T}\sum_{t \in \mathrm{freq}} h_t}_{\text{공통 성분 (크다)}} + \underbrace{\frac{1}{T}\sum_{t \in \mathrm{rare}} h_t}_{\text{의미 성분 (작다)}}$$

공통 성분이 지배적이어서 모든 $s$가 비슷한 방향을 갖는다.

#### Contrastive Learning이 해결하는 방법

NT-Xent loss의 uniformity 효과가 이 문제를 직접 해결한다:

1. **L2 정규화**: $\hat{z} = z/\|z\|$로 벡터를 단위 구면 위에 투사. 크기 차이 제거
2. **분모의 repulsion**: 부정 쌍을 밀어내는 gradient가 벡터를 **구면 위에 균등하게** 분산
3. **공통 성분 제거**: 모든 문장에 공통인 방향은 구분에 도움이 안 되므로, contrastive loss가 자연스럽게 그 방향의 분산을 줄인다

결과적으로 contrastive fine-tuning 후 코사인 유사도의 범위가 [-0.7, 0.99]처럼 넓어져, 의미 차이를 효과적으로 구분할 수 있게 된다. 우리 실험에서도:

```
학습 전: 모든 문장 벡터가 비슷 (BERT 초기화의 비등방성)
학습 후: cos(같은 그룹) = 0.9951, cos(다른 그룹) = -0.6978
         → 범위가 2.0 가까이 확대됨
```

---

### SimCSE: Dropout만으로 Contrastive Learning

Gao et al. (2021)의 놀라운 발견: **같은 문장을 두 번 인코딩**하되, dropout만 다르게 적용하면 양의 쌍이 된다!

```
"The cat sat on the mat"  ──→ BERT (dropout mask 1) ──→ z_a
"The cat sat on the mat"  ──→ BERT (dropout mask 2) ──→ z_b
                                   같은 문장, 다른 dropout
```

왜 이것이 작동하는가:

1. **Dropout은 암묵적 data augmentation**: 랜덤 뉴런을 끄는 것은 네트워크의 "관점"을 바꾸는 것. 같은 입력이지만 다른 feature subset으로 표현
2. **최소한의 변형**: 의미를 전혀 바꾸지 않으면서 표현만 미세하게 다르게 함
3. **비등방성 해소**: contrastive loss의 uniformity 효과가 동일하게 작동

SimCSE의 성능은 **레이블 없이도** SBERT에 근접하거나 능가한다. 이것은 contrastive learning에서 **양의 쌍의 정의**가 얼마나 유연할 수 있는지를 보여준다.

이 아이디어의 확장:
- **ESimCSE**: 문장 길이를 랜덤으로 조절 (단어 반복)하여 양의 쌍 생성
- **DiffCSE**: 차이를 만드는 변환과 보존하는 변환을 분리

---

### InfoNCE가 상호 정보의 하한인 이유: 증명

#### 설정

$X$와 $X^+$는 양의 쌍, $\{X_j^-\}_{j=1}^{N-1}$은 $p(x)$에서 i.i.d.로 뽑은 부정 샘플.

밀도 비 (density ratio):

$$r(x, x^+) = \frac{p(x, x^+)}{p(x) p(x^+)}$$

InfoNCE의 최적 critic $f^*(x, x^+) \propto r(x, x^+)$일 때:

#### 증명 스케치

InfoNCE의 기대값을 전개하면:

$$\mathcal{L}_\mathrm{InfoNCE} = \mathbb{E}\left[-\log\frac{e^{f(x, x^+)/\tau}}{\frac{1}{N}\sum_{j=0}^{N-1} e^{f(x, x_j)/\tau}}\right] + \log N$$

Jensen's inequality에 의해:

$$\mathcal{L}_\mathrm{InfoNCE} \geq -\mathbb{E}\left[\log\frac{p(x^+ \mid x)}{p(x^+)}\right] + \log N = -I(X; X^+) + \log N$$

따라서:

$$I(X; X^+) \geq \log N - \mathcal{L}_\mathrm{InfoNCE}$$

**해석**:
- $\mathcal{L}_\mathrm{InfoNCE}$를 최소화하면 MI의 **하한**이 최대화된다
- 하한의 상한은 $\log N$: 배치 크기가 곧 추정 가능한 MI의 한계
- $N = 65536$ (CLIP) → $\log N \approx 11.1$ nats까지 추정 가능
- 이것이 contrastive learning에서 **큰 배치가 중요한 이유**

| 배치 크기 $N$ | MI 추정 상한 ($\log N$) | 대표 모델 |
|---|---|---|
| 256 | 5.5 nats | SimCLR (TPU 1개) |
| 4096 | 8.3 nats | SimCLR (TPU 128개) |
| 32768 | 10.4 nats | CLIP |
| 65536 | 11.1 nats | CLIP (최대) |

배치를 키울수록 MI 추정이 정확해지지만, 수확 체감(diminishing returns)이 있다: $\log N$의 증가율은 $1/N$으로 감소한다.

---

### Cross-Encoder vs Bi-Encoder: 정확도-효율 트레이드오프

#### 수학적 비교

**Cross-Encoder** (BERT 원본):
$$\mathrm{score}(A, B) = f_\mathrm{BERT}([A; B])$$

두 문장을 concat하여 jointly 인코딩. Self-attention이 두 문장 사이의 **토큰 수준 상호작용**을 직접 포착한다.

**Bi-Encoder** (SBERT/우리 구현):
$$\mathrm{score}(A, B) = \cos(g(A), g(B))$$

각 문장을 독립 인코딩 후 유사도 계산. 두 문장 사이의 상호작용은 **벡터 내적**으로만 포착된다.

| | Cross-Encoder | Bi-Encoder |
|---|---|---|
| **표현력** | 토큰 수준 교차 attention | 벡터 수준 내적만 |
| **N쌍 유사도** | $O(N^2)$ BERT 호출 | $O(N)$ BERT + $O(N^2)$ 내적 |
| **10K 문장** | ~2.7일 | ~5초 + 내적 |
| **정확도** | STS-B: 90.1 | STS-B: 85.4 (SBERT) |
| **검색 사용** | 불가 (쿼리마다 전체 재인코딩) | 가능 (문서 벡터 사전 계산) |

Bi-encoder의 정확도 손실은 **정보 병목**에서 온다: $D$차원 벡터 하나로 문장의 모든 의미를 압축해야 하므로, 미묘한 의미 차이가 손실된다. 하지만 실무에서는 5%의 정확도 차이보다 **50,000배의 속도 차이**가 더 중요하다.

---

### 학습 결과 분석

```
epoch   1 | loss 1.1949    ← log(N) = log(4) ≈ 1.386에 가까운 초기 loss
epoch  25 | loss 0.2880    ← 빠르게 감소
epoch 100 | loss 0.1907    ← 수렴

cos(A, A') = 0.9951  ← 같은 그룹 (같은 토큰, 다른 순서): 거의 동일
cos(A, B)  = -0.6978  ← 다른 그룹 (다른 토큰): 반대 방향까지 밀려남
```

#### 초기 loss와 $\log N$의 관계

초기 loss가 $\log N \approx 1.386$에 가까운 것은 우연이 아니다. 랜덤 초기화 시 모든 유사도가 비슷하면:

$$\mathcal{L}_\mathrm{init} \approx -\log\frac{1/N}{N \cdot 1/N} = \log N$$

이것은 **uniform softmax**의 엔트로피와 동일하다. 학습이 진행되면 양의 쌍의 softmax 확률이 1에 가까워지고, loss는 0에 수렴한다.

실제 값이 정확히 $\log 4 = 1.386$이 아닌 이유: 초기화 시 벡터들이 완벽히 균등하지 않고, L2 정규화 후 코사인 유사도에 약간의 편향이 있다. 또한 대칭 loss이므로 $\frac{1}{2}(\mathcal{L}_{a \to b} + \mathcal{L}_{b \to a})$로 평균되면서 약간의 차이가 발생한다.

#### loss의 이론적 하한

loss의 최솟값은 0이 아니다. 유한한 배치에서 양의 쌍이 **완벽하게 동일**해도 부정 쌍과의 유사도가 0이 아닌 한 loss > 0이다. 4차원에서 3개 직교 벡터를 양의 쌍으로 사용한 경우:

```
perfect alignment loss: 0.2395  ← 이론적 최솟값에 가까움
shuffled alignment loss: 2.2395  ← 잘못된 매칭, loss ≈ 2 × log(N)에 가까움
```

---

### SumFn backward 버그 수정

이번 스텝 구현 중 발견한 `SumFn::backward`의 숨겨진 버그:

`sum_with(x, Some(1), false)`로 (B, T, D) → (B, D)를 만든 후 backward에서 gradient를 (B, D) → (B, T, D)로 broadcast해야 하는데, 기존 코드는 축을 다시 삽입하지 않았다:

```rust
// 기존 (버그): (B, D)를 직접 (B, T, D)로 broadcast — B≠T이면 실패
let broadcast = gy_data.broadcast(IxDyn(&self.x_shape)).unwrap();

// 수정: keepdims=false일 때 축을 다시 삽입 후 broadcast
// (B, D) → (B, 1, D) → broadcast → (B, T, D)
if let Some(axis) = self.axis {
    if !self.keepdims {
        let mut shape = gy_data.shape().to_vec();
        shape.insert(axis, 1);
        // reshape 후 broadcast
    }
}
```

ndarray의 broadcast는 NumPy와 동일하게 **오른쪽 정렬** 후 차원을 맞추므로, (B, D)를 (B, T, D)로 broadcast하면 D↔D, B↔T로 매칭되어 B≠T이면 실패한다. B==T인 경우에만 우연히 성공하지만, 의미적으로 틀린 broadcast가 된다.

---

### 코드 발췌

#### NT-Xent Forward

```rust
// L2 정규화 → 유사도 행렬 → 대칭 cross-entropy
for i in 0..n {
    for j in 0..n {
        let mut dot = 0.0;
        for k in 0..d { dot += z_a_norm[[i,k]] * z_b_norm[[j,k]]; }
        sim[[i, j]] = dot / self.temperature;
    }
}
// CE(S, diag) + CE(S^T, diag)
for i in 0..n {
    // 수치 안정: max 빼기
    let max_val = (0..n).map(|j| sim[[i,j]]).fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = (0..n).map(|j| (sim[[i,j]] - max_val).exp()).sum();
    loss -= sim[[i,i]] - max_val - sum_exp.ln();  // = -log_softmax(S[i,i])
}
```

#### L2 Normalization Backward

```rust
// g - ẑ(g·ẑ) : gradient에서 현재 방향 성분 제거 (접선 투사)
// / ||z||    : 스케일 복원
for k in 0..d {
    g_a[[i, k]] = (g_a_hat[[i, k]] - z_a_hat[[i, k]] * dot_a) / norms_a[i];
}
```

---

### Contrastive Loss 변형들: Triplet → InfoNCE → NT-Xent

#### Triplet Loss (FaceNet, 2015)

가장 초기의 contrastive 형태:

$$\mathcal{L} = \max(0,\ \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha)$$

$a$: anchor, $p$: positive, $n$: negative, $\alpha$: margin.

한계점:
- 한 번에 **부정 샘플 1개**만 비교 → 정보 효율 낮음
- **margin $\alpha$** 선택이 어려움 (너무 크면 학습 불가, 너무 작으면 불충분)
- Semi-hard negative mining 필요 (Hermans et al., 2017)

#### N-pair Loss (Sohn, 2016)

Triplet의 일반화: $N$개 부정 샘플을 동시에 비교

$$\mathcal{L} = -\log\frac{\exp(f(a)^\top f(p))}{\exp(f(a)^\top f(p)) + \sum_{k=1}^{N} \exp(f(a)^\top f(n_k))}$$

이것은 사실상 **InfoNCE**와 동일하다. Margin 파라미터가 불필요하고, softmax 분모가 자동으로 **적응적 margin** 역할을 한다.

#### 왜 softmax가 margin보다 좋은가

Triplet loss에서 margin $\alpha$는 고정된 "기준"이다: anchor-positive 거리가 anchor-negative 거리보다 $\alpha$만큼 작으면 loss = 0. 쉬운 샘플이든 어려운 샘플이든 같은 기준.

Softmax(InfoNCE)에서는 **상대적 비교**가 이루어진다:

$$p_i = \frac{\exp(s_{\mathrm{pos}})}{\exp(s_{\mathrm{pos}}) + \sum \exp(s_{\mathrm{neg}})}$$

$s_\mathrm{pos}$가 모든 $s_\mathrm{neg}$보다 **충분히** 클 때만 $p_i \to 1$. "충분히"의 기준은 부정 샘플들의 유사도에 **자동 적응**한다. 어려운 부정 샘플이 있으면 기준이 높아지고, 쉬운 샘플만 있으면 기준이 낮아진다.

---

### 연결: Word2Vec → Sentence Embedding → 벡터 검색

Phase 3의 두 스텝을 관통하는 핵심:

```
Step 69: 단어 → 벡터       (Word2Vec, SGNS)
Step 70: 문장 → 벡터       (SentenceEmbedding, NT-Xent)
         ↓
Phase 4: 벡터 → 검색       (Brute Force → IVF → PQ → HNSW)
         ↓
Phase 9: RAG = 임베딩 + 검색 + LLM
```

이 벡터들은 다음 단계(Phase 4)에서 **벡터 검색**의 입력이 된다:
1. 쿼리 문장 → `encode()` → 벡터
2. 데이터베이스의 모든 문장 → 사전에 `encode()` → 벡터 인덱스
3. 코사인 유사도 기반 **최근접 이웃 검색** → 가장 유사한 문장 반환

이것이 **RAG (Retrieval-Augmented Generation)**의 핵심 파이프라인이다. 좋은 문장 임베딩이 없으면 검색 품질이 낮아지고, 검색 품질이 낮으면 LLM에 잘못된 문맥이 제공된다.
