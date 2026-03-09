## Step 69: Word2Vec — Skip-gram + Negative Sampling

### 한마디 직관

- **Word2Vec = "비슷한 맥락의 단어는 비슷한 벡터"** — 분포 가설(distributional hypothesis)의 구현체
- 전체 어휘 softmax(수만 차원) 대신 **K개 부정 샘플**만으로 학습 → 실용적 속도
- 2013년 논문이지만, 이후 모든 임베딩 기법(BERT, Contrastive Learning)의 **설계 원리**가 여기서 시작

"You shall know a word by the company it keeps." — J.R. Firth (1957)

---

### 왜 단어를 벡터로 표현하는가

#### One-hot의 근본적 한계

단어를 정수 인덱스나 one-hot 벡터로 표현하면:
- "cat" = [0, 0, 1, 0, 0, ...], "dog" = [0, 0, 0, 1, 0, ...]
- **모든 단어 쌍의 거리가 동일**: $\|\mathrm{cat} - \mathrm{dog}\| = \|\mathrm{cat} - \mathrm{pizza}\| = \sqrt{2}$

수학적으로 보면, one-hot 벡터 $e_i$와 $e_j$ ($i \neq j$)는:
$$e_i^\top e_j = 0, \quad \|e_i - e_j\|^2 = 2$$

모든 쌍이 **직교**하고 **등거리**이다. $V$차원 공간에서 $V$개의 직교 벡터는 **가장 효율적으로 구분 가능한** 배치이지만, 단어 사이의 **관계를 전혀 인코딩하지 못한다**.

#### 밀집 벡터: 기하학적 의미

$D$차원 실수 벡터 ($D \ll V$, 예: $D = 100$, $V = 50000$)로 표현하면:

$$\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

코사인 유사도가 **의미적 유사성**을 반영하도록 학습된다:
- cos("cat", "dog") ≈ 0.95 (비슷한 동물)
- cos("cat", "pizza") ≈ 0.1 (관련 없음)
- cos("king", "queen") ≈ 0.8 (비슷한 역할)

차원 축소의 정보 이론적 관점: $V$차원 one-hot에서 $D$차원 밀집 벡터로의 변환은 **정보 병목(information bottleneck)**을 형성한다. 이 병목이 단어의 **본질적 의미 구조**만 보존하도록 학습된다.

---

### 분포 가설과 그 수학적 정당화

#### 분포 가설 (Distributional Hypothesis)

**"단어의 의미는 그 주변 문맥에 의해 결정된다."** (Harris, 1954; Firth, 1957)

```
"The _____ sat on the mat."
→ cat, dog, baby, ...  (비슷한 맥락 → 비슷한 의미)

"I want to _____ a beer."
→ drink, buy, order, ...  (비슷한 맥락 → 비슷한 의미)
```

#### 왜 이것이 수학적으로 작동하는가: PMI 행렬

두 단어 $w$와 $c$의 **점별 상호 정보(Pointwise Mutual Information)**:

$$\mathrm{PMI}(w, c) = \log \frac{P(w, c)}{P(w) \cdot P(c)}$$

- $\mathrm{PMI} > 0$: 기대보다 자주 같이 나타남 (의미적 연관)
- $\mathrm{PMI} = 0$: 독립 (관련 없음)
- $\mathrm{PMI} < 0$: 기대보다 드물게 같이 나타남 (상호 배타적)

$V \times V$ 크기의 PMI 행렬은 모든 단어 쌍의 관계를 완전히 기술한다. 그런데 이 행렬은 **저랭크(low-rank)**이다! 의미적으로 유사한 단어는 비슷한 PMI 패턴을 갖기 때문이다.

$$\mathrm{PMI} \approx W_{\mathrm{in}} \cdot W_{\mathrm{out}}^\top \quad (V \times D)(D \times V)$$

이것이 Word2Vec이 학습하는 것의 **본질**이다. $D$차원 벡터 2개의 내적으로 PMI를 근사한다.

---

### Skip-gram의 원래 모델과 그 한계

#### 전체 Softmax 모델

"The cat sat on the mat"에서 중심 단어 "sat"과 윈도우 내 문맥 단어의 쌍을 생성:
- (sat, cat), (sat, on) — 윈도우 크기 1
- (sat, The), (sat, cat), (sat, on), (sat, the) — 윈도우 크기 2

원래 확률 모델:

$$P(\mathrm{context} \mid \mathrm{center}) = \frac{\exp(u_{\mathrm{ctx}}^\top v_{\mathrm{ctr}})}{\sum_{w \in V} \exp(u_w^\top v_{\mathrm{ctr}})}$$

$v_{\mathrm{ctr}} = W_{\mathrm{in}}[\mathrm{center}]$ (중심 단어 임베딩), $u_w = W_{\mathrm{out}}[w]$ (문맥 단어 임베딩).

#### 계산량 분석: 왜 비현실적인가

분모 $Z = \sum_{w \in V} \exp(u_w^\top v_{\mathrm{ctr}})$의 계산에 $O(V \times D)$ 연산 필요. 이것을 **매 학습 샘플마다** 계산해야 한다.

| | Softmax | Negative Sampling |
|---|---|---|
| Forward 계산 | $O(V \times D)$ | $O(K \times D)$ |
| Backward 계산 | $O(V \times D)$ | $O(K \times D)$ |
| $V = 50000$, $K = 5$ | 50,000번 | 5번 |
| 속도 비율 | 1× | **10,000×** |

실제 학습에서 코퍼스 크기가 수십억 단어이므로, 이 차이는 **주 단위 학습을 시간 단위로** 줄인다.

---

### Negative Sampling: softmax → 이진 분류

#### 핵심 아이디어

전체 어휘에 대한 확률을 계산하는 대신, **이진 분류**로 대체:
- "이 (center, context) 쌍이 실제 데이터에서 온 것인가?" (positive = yes)
- 랜덤으로 뽑은 (center, random_word) 쌍은? (negative = no)

#### 손실 함수와 기하학적 의미

$$L = -\log \sigma(u_{\mathrm{pos}}^\top v) - \sum_{k=1}^{K} \log \sigma(-u_{\mathrm{neg}_k}^\top v)$$

$\sigma(x) = \frac{1}{1+e^{-x}}$는 시그모이드 함수.

**각 항의 기하학적 의미**:

첫째 항 $-\log\sigma(u_{\mathrm{pos}}^\top v)$: 정답 문맥 벡터 $u_{\mathrm{pos}}$를 중심 벡터 $v$와 **같은 방향**으로 정렬시키는 힘. 내적이 클수록 $\sigma \to 1$, loss $\to 0$.

둘째 항 $-\sum_k \log\sigma(-u_{\mathrm{neg}_k}^\top v)$: 부정 벡터를 $v$와 **반대 방향**으로 밀어내는 힘. 내적이 작을수록 (음수일수록) $\sigma(-\mathrm{dot}) \to 1$, loss $\to 0$.

```
벡터 공간에서 하나의 학습 스텝:

  u_neg2          v (center)           u_pos
    ←──── 밀어냄 ────○──── 끌어당김 ────→
  u_neg1
    ←──── 밀어냄 ────┘
```

이것은 **인력과 척력**의 물리 시스템과 동일하다. 정답 쌍은 인력, 부정 쌍은 척력. 평형 상태에서 유사한 단어끼리 가까이 모이게 된다.

---

### SGNS가 PMI를 학습한다는 증명

#### Levy & Goldberg (2014)의 핵심 정리

Word2Vec의 Skip-gram with Negative Sampling (SGNS)이 **실제로 무엇을 최적화하는지** 수학적으로 증명한 획기적 논문이다.

코퍼스의 모든 (center, context) 쌍에 대한 SGNS의 기대 손실:

$$J = \sum_{w,c} \left[ \#(w,c) \cdot \log\sigma(u_c^\top v_w) + K \cdot P_n(c) \cdot \#(w) \cdot \log\sigma(-u_c^\top v_w) \right]$$

$\#(w,c)$: 동시 출현 횟수, $\#(w)$: 단어 $w$의 총 출현 횟수, $P_n(c)$: 부정 샘플링 분포.

**고정된 $(w, c)$에 대해** $x = u_c^\top v_w$로 놓고, $x$에 대해 미분하여 최적점을 구하면:

$$\frac{\partial J}{\partial x} = \#(w,c) \cdot (1 - \sigma(x)) - K \cdot P_n(c) \cdot \#(w) \cdot \sigma(x) = 0$$

이 방정식을 풀면:

$$\sigma(x^*) = \frac{\#(w,c)}{\#(w,c) + K \cdot P_n(c) \cdot \#(w)}$$

$P_n(c) = \frac{\#(c)}{|D|}$ (unigram 분포, 0.75승 무시)이면:

$$x^* = \log\frac{\#(w,c) \cdot |D|}{K \cdot \#(w) \cdot \#(c)} = \log\frac{P(w,c)}{P(w) P(c)} - \log K$$

따라서:

$$\boxed{u_c^\top v_w = \mathrm{PMI}(w, c) - \log K}$$

**SGNS의 최적해는 shifted PMI 행렬의 분해**이다!

$$W_{\mathrm{in}} \cdot W_{\mathrm{out}}^\top \approx \mathrm{PMI} - \log K$$

$K$를 증가시키면 $\log K$가 커져서 부정 PMI 값이 더 많이 절삭된다. $K = 5$이면 $\log 5 \approx 1.6$ 이하의 PMI는 0에 가깝게 억제된다.

#### 왜 이것이 중요한가

1. **SGNS ≈ 행렬 분해**: Word2Vec은 단순한 신경망이 아니라, 동시 출현 통계의 **암묵적 행렬 분해**
2. **GloVe와의 동치성**: GloVe는 동일한 행렬을 **명시적으로** 분해. 결과적으로 두 방법은 같은 것을 학습
3. **차원 D의 역할**: PMI 행렬 (rank ≈ $V$)을 rank $D$로 근사. $D$가 PMI의 "유효 랭크"보다 크면 완벽 복원, 작으면 가장 중요한 $D$개의 특이값만 보존

---

### 부정 샘플링 분포: Unigram$^{0.75}$의 수학

#### 왜 0.75인가: Zipf 법칙과의 관계

자연어의 단어 빈도는 **Zipf 법칙**을 따른다:

$$f(r) \propto \frac{1}{r^{\alpha}}, \quad \alpha \approx 1$$

빈도 순위 $r$에 반비례. 가장 빈번한 단어("the")는 1000번째 단어보다 ~1000배 빈번하다.

균등 분포로 부정 샘플을 뽑으면: 빈도 높은 "the", "a"와의 구별이 너무 쉬워 학습 신호가 약하다.
원래 unigram $P(w)$로 뽑으면: Zipf 분포의 긴 꼬리 때문에 희귀 단어가 거의 안 뽑힌다.

$P_{\mathrm{neg}}(w) \propto f(w)^{0.75}$는 이 두 극단의 **절충**:

$$\text{균등}: f^{0} \quad \longleftrightarrow \quad f^{0.75} \quad \longleftrightarrow \quad \text{원래 분포}: f^{1}$$

| 단어 | 빈도 $f$ | $f^{0.75}$ | $f^{1}$ (원래) | 비율 변화 |
|---|---|---|---|---|
| "the" | 100 | 31.6 | 100 | ↓ 68% 억제 |
| "cat" | 10 | 5.6 | 10 | ↓ 44% 억제 |
| "quasar" | 1 | 1.0 | 1 | 불변 |

0.75는 Mikolov et al.이 **실험적으로** 찾은 값이다. 0.5~1.0 범위에서 비슷한 성능이지만, 0.75가 최적.

구현에서는 이 분포를 **테이블로 이산화**하여 $O(1)$ 샘플링한다:

```rust
fn build_negative_table(word_counts: &[usize], table_size: usize) -> Vec<usize> {
    let total: f64 = word_counts.iter()
        .map(|&c| (c as f64).powf(0.75)).sum();
    // 각 단어를 f^0.75 비율만큼 테이블에 채움
    // 이후 table[uniform_random_index]로 O(1) 샘플링
}
```

실험 결과 (`test_negative_table_distribution`):
```
빈도 100:10:1 → 테이블 비율: 82.7% : 14.7% : 2.6%
```
원래 비율 90.1:9.0:0.9에서 희귀 단어의 비율이 3배 증가.

---

### Gradient 유도: 상세 버전

#### Forward

$$L = -\frac{1}{N}\sum_{i=1}^{N}\left[\log\sigma(u_{\mathrm{pos}_i}^\top v_i) + \sum_{k=1}^{K} \log\sigma(-u_{\mathrm{neg}_{ik}}^\top v_i)\right]$$

모든 항을 통합하면, 라벨 $\ell \in \{+1, -1\}$:

$$L = -\frac{1}{N}\sum_{i,j} \log\sigma(\ell_{ij} \cdot s_{ij})$$

여기서 $s_{ij} = u_j^\top v_i$ (내적), $\ell = +1$ (정답), $\ell = -1$ (부정).

#### 단일 항의 미분

$f(s) = -\log\sigma(\ell \cdot s)$에 대해:

$$\frac{df}{ds} = -\frac{1}{\sigma(\ell s)} \cdot \sigma(\ell s)(1 - \sigma(\ell s)) \cdot \ell = -\ell(1 - \sigma(\ell s))$$

시그모이드의 대칭성 $1 - \sigma(x) = \sigma(-x)$을 사용하면:

- $\ell = +1$ (정답): $\frac{df}{ds} = -(1 - \sigma(s)) = \sigma(s) - 1$
- $\ell = -1$ (부정): $\frac{df}{ds} = +(1 - \sigma(-s)) = \sigma(s)$

**핵심 통찰**: $y = 1$ (정답) 또는 $y = 0$ (부정)으로 놓으면:

$$\boxed{\frac{\partial L}{\partial s_{ij}} = \frac{1}{N}(\sigma(s_{ij}) - y_{ij})}$$

이것은 **logistic regression의 gradient와 정확히 동일**하다. Cross-entropy (step 47) gradient $(\mathrm{softmax} - \mathrm{one\_hot})/N$의 이진 분류 버전이다.

#### 벡터에 대한 gradient

$s_{ij} = u_j^\top v_i = \sum_d u_{jd} \cdot v_{id}$이므로:

$$\frac{\partial L}{\partial v_i} = \frac{1}{N}\sum_{j \in \{pos, neg_1, \ldots, neg_K\}} (\sigma(s_{ij}) - y_{ij}) \cdot u_j$$

$$\frac{\partial L}{\partial u_j} = \frac{1}{N}(\sigma(s_{ij}) - y_{ij}) \cdot v_i$$

직관:
- 정답인데 내적이 작으면 ($\sigma \approx 0$): $(\sigma - 1) < 0$ → $v_i$를 $u_j$ 방향으로 끌어당김
- 부정인데 내적이 크면 ($\sigma \approx 1$): $(\sigma - 0) > 0$ → $v_i$를 $u_j$ 반대 방향으로 밀어냄

```rust
fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
    for i in 0..n {
        for j in 0..k1 {
            let sig = 1.0 / (1.0 + (-dot).exp());
            let label_01 = if self.labels[idx] > 0.0 { 1.0 } else { 0.0 };
            let grad_dot = (sig - label_01) / n as f64 * gy_val;
            for dd in 0..d {
                gv[[i, dd]] += grad_dot * u_data[[idx, dd]];  // ∂L/∂v
                gu[[idx, dd]] += grad_dot * v_data[[i, dd]];  // ∂L/∂u
            }
        }
    }
}
```

#### Embedding의 scatter-add backward와의 관계

`Word2Vec.forward`에서 `w_out.forward(all_ids)`가 호출되면, Embedding의 backward는 **scatter-add** (step 61):

$$\frac{\partial L}{\partial W_{\mathrm{out}}[w]} = \sum_{j: \mathrm{ids}[j] = w} \frac{\partial L}{\partial u_j}$$

같은 단어가 여러 번 나타나면 gradient가 **누적**된다. 이것이 빈도 높은 단어의 벡터가 더 빨리 학습되는 메커니즘이다. Word2Vec에서는 같은 부정 단어가 여러 샘플에 등장할 수 있으므로, scatter-add가 정확한 gradient 전파에 필수적이다.

---

### 수치 안정성: log-sigmoid의 구현

$\log\sigma(x) = -\log(1 + e^{-x})$를 직접 계산하면:

| $x$ | $e^{-x}$ | $\sigma(x)$ | $\log\sigma(x)$ | 문제 |
|---|---|---|---|---|
| 100 | $\approx 0$ | $\approx 1$ | $\approx 0$ | 안전 |
| 0 | 1 | 0.5 | -0.693 | 안전 |
| -100 | $\approx 10^{43}$ | $\approx 0$ | $\log(0)$ = **-∞** | **overflow + -∞** |

안정 버전:

$$\log\sigma(x) = \min(0, x) - \log(1 + e^{-|x|})$$

| $x$ | $\min(0,x)$ | $e^{-|x|}$ | $\log(1+e^{-|x|})$ | 합계 |
|---|---|---|---|---|
| 100 | 0 | $\approx 0$ | $\approx 0$ | $\approx 0$ ✓ |
| -100 | -100 | $\approx 0$ | $\approx 0$ | $\approx -100$ ✓ |

**두 경우 모두** $e^{-|x|}$는 0에 가까워 overflow가 없다.

```rust
let x = self.labels[idx] * dot;
let log_sig = x.min(0.0) - (1.0 + (-x.abs()).exp()).ln();
```

이 트릭은 `softplus(-x) = log(1 + exp(-x))`의 수치 안정 버전이기도 하다.

---

### 두 가지 Embedding: W_in과 W_out의 기하학

#### 왜 두 개인가: 자기 내적 편향 문제

만약 $u = v$ (같은 행렬)라면:

$$P(w \mid w) \propto \sigma(v_w^\top v_w) = \sigma(\|v_w\|^2)$$

$\|v_w\|^2 \geq 0$이므로 $\sigma(\|v_w\|^2) \geq 0.5$. 어떤 단어든 **자기 자신을 예측하는 확률이 0.5 이상**이 되어, 모든 단어가 자기 자신 방향으로 편향된다. 이것은 원래 softmax에는 없는 인위적 편향이다.

두 행렬을 분리하면 $u_w^\top v_w$가 양수/음수 모두 될 수 있어 이 문제가 해결된다.

#### 벡터 공간의 비대칭성

$W_{\mathrm{in}}$과 $W_{\mathrm{out}}$은 **같은 의미 구조를 다른 관점에서** 인코딩한다:

$$u_c^\top v_w = \mathrm{PMI}(w, c) - \log K$$

$v_w$: 단어 $w$가 **중심**일 때의 특성 ("이 단어가 어떤 문맥을 끌어들이는가")
$u_c$: 단어 $c$가 **문맥**일 때의 특성 ("이 단어가 어떤 중심 단어 주변에 나타나는가")

예: "the"는 중심 단어로는 거의 모든 명사/동사를 예측 (비특이적), 문맥 단어로는 명사 앞에서만 특이적으로 나타남. $v_{\mathrm{the}}$와 $u_{\mathrm{the}}$는 이 비대칭 역할을 반영하여 상당히 다를 수 있다.

#### 학습 후 어떤 벡터를 사용하는가

통상 $W_{\mathrm{in}}$만 사용한다. $W_{\mathrm{out}}$은 버리거나, 두 행렬의 평균을 쓰기도 한다 (GloVe는 평균 사용).

Levy et al. (2015)는 $W_{\mathrm{in}}$만 사용하는 것이 약간 더 좋은 경향이 있다고 보고했다.

---

### 단어 유추(Word Analogy)는 왜 작동하는가

#### 유명한 예

$$\vec{\mathrm{king}} - \vec{\mathrm{man}} + \vec{\mathrm{woman}} \approx \vec{\mathrm{queen}}$$

이것이 왜 가능한지 수학적으로 분석한다.

#### PMI의 선형성에서 유도

SGNS의 최적해에서 $v_w^\top u_c = \mathrm{PMI}(w, c) - \log K$. 이를 행렬 형태로 쓰면:

$$V \cdot U^\top = M \quad (M_{wc} = \mathrm{PMI}(w,c) - \log K)$$

"king"과 "queen"의 차이는:
$$v_{\mathrm{king}} - v_{\mathrm{queen}} \approx v_{\mathrm{man}} - v_{\mathrm{woman}}$$

이것이 성립하려면, 모든 문맥 단어 $c$에 대해:

$$(v_{\mathrm{king}} - v_{\mathrm{queen}})^\top u_c \approx (v_{\mathrm{man}} - v_{\mathrm{woman}})^\top u_c$$

즉:

$$\mathrm{PMI}(\mathrm{king}, c) - \mathrm{PMI}(\mathrm{queen}, c) \approx \mathrm{PMI}(\mathrm{man}, c) - \mathrm{PMI}(\mathrm{woman}, c)$$

이것은 **"왕족 여부에 따른 PMI 차이와 성별에 따른 PMI 차이가 독립적"**이라는 조건이다. 자연어에서 이 조건이 **대략적으로** 성립하기 때문에 유추가 작동한다.

$$\log\frac{P(c \mid \mathrm{king})}{P(c \mid \mathrm{queen})} \approx \log\frac{P(c \mid \mathrm{man})}{P(c \mid \mathrm{woman})}$$

"crown"이라는 문맥 단어를 생각하면:
- king/queen 모두 "crown"과 자주 동시 출현 (왕족 속성) → 비율 ≈ 1
- man/woman 모두 "crown"과 드물게 동시 출현 → 비율 ≈ 1

"he"라는 문맥 단어:
- king은 "he"와 자주, queen은 "she"와 자주 → 비율 > 1
- man도 "he"와 자주, woman은 "she"와 자주 → 비율 > 1 (비슷한 크기)

**성별과 지위가 독립적인 축**으로 인코딩되기 때문에, 벡터 산술이 의미 관계를 조작할 수 있다.

---

### Contrastive Learning과의 연결

#### Word2Vec은 최초의 Contrastive Learning

Word2Vec의 loss를 다시 보자:

$$L = -\log\sigma(u_{\mathrm{pos}}^\top v) - \sum_{k=1}^{K} \log\sigma(-u_{\mathrm{neg}_k}^\top v)$$

이것은 현대 Contrastive Learning의 원형이다:

| | Word2Vec (2013) | SimCLR (2020) | CLIP (2021) |
|---|---|---|---|
| Positive pair | (center, context) | (augmented view 1, view 2) | (image, text) |
| Negative pair | (center, random word) | (view 1, other images) | (image, other texts) |
| Similarity | $u^\top v$ (내적) | $\cos(z_i, z_j)$ | $\cos(I, T)$ |
| Loss | $-\log\sigma(s^+) - \sum\log\sigma(-s^-)$ | InfoNCE | InfoNCE |

InfoNCE loss (van den Oord et al., 2018):

$$L_{\mathrm{InfoNCE}} = -\log\frac{\exp(s^+ / \tau)}{\exp(s^+ / \tau) + \sum_k \exp(s_k^- / \tau)}$$

$K \to \infty$에서 InfoNCE는 **mutual information의 하한**을 최대화한다:

$$L_{\mathrm{InfoNCE}} \geq -I(X; Y) + \log K$$

Word2Vec도 동일한 원리: center와 context의 상호 정보를 최대화하면서, 부정 샘플로 제약을 건다.

---

### Skip-gram vs CBOW

Word2Vec에는 두 가지 변형이 있다:

| | Skip-gram | CBOW |
|---|---|---|
| 입력 | 중심 단어 1개 | 문맥 단어 여러 개 |
| 출력 | 문맥 단어 예측 | 중심 단어 예측 |
| 수식 | $P(c \mid w) = \sigma(u_c^\top v_w)$ | $P(w \mid C) = \sigma(v_w^\top \bar{u}_C)$ |
| 특징 | 희귀 단어에 강함 | 빈번한 단어에 강함 |
| 속도 | 느림 (쌍이 많음) | 빠름 |

CBOW에서 $\bar{u}_C = \frac{1}{|C|}\sum_{c \in C} u_c$는 문맥 벡터의 **평균**. 이 평균화가 문제다:

- "I went to the **bank** to deposit money" → 문맥 평균이 "money"의 정보를 포함 → "bank" = 은행
- "I sat on the river **bank** and fished" → 문맥 평균이 "river"의 정보를 포함 → "bank" = 강둑

CBOW는 문맥을 평균내므로, 개별 문맥 단어의 정보가 **희석**된다. Skip-gram은 각 (center, context) 쌍을 **개별적으로** 학습하므로 더 다양한 관계를 포착한다.

---

### Word2Vec → Transformer Embedding: 정적 vs 문맥적

#### 정적 임베딩의 근본 한계

Word2Vec은 단어당 **하나의 벡터**만 학습한다. "bank"가 100번 나타나면 모든 문맥의 평균적 의미가 하나의 벡터에 압축된다:

$$v_{\mathrm{bank}} = \frac{1}{N}\sum_{i=1}^{N} v_{\mathrm{bank}}^{(i)} \quad \text{(모든 문맥의 평균)}$$

금융 문맥의 "bank"와 강둑 문맥의 "bank"가 같은 벡터. 이것이 **정적 임베딩(static embedding)**의 한계다.

#### BERT/GPT의 문맥 임베딩

Transformer의 출력 벡터는 **문맥에 따라 변한다**:

$$h_{\mathrm{bank}}^{(\mathrm{financial})} \neq h_{\mathrm{bank}}^{(\mathrm{river})}$$

Self-Attention이 주변 토큰의 정보를 주입하므로, 같은 단어라도 다른 표현을 갖는다.

그러나 Transformer의 **첫 번째 레이어 입력**은 Token Embedding lookup이므로 여전히 **정적**이다. Word2Vec의 원리가 이 초기 임베딩에 암묵적으로 적용된다고 볼 수 있다.

| | Word2Vec | BERT 첫 레이어 | BERT 마지막 레이어 |
|---|---|---|---|
| 유형 | 정적 | 정적 | 문맥적 |
| 차원 | $D = 100\sim300$ | $D = 768$ | $D = 768$ |
| 학습 | NS loss | MLM loss (end-to-end) | MLM loss (end-to-end) |
| "bank" 표현 | 1개 | 1개 | 문맥별 상이 |
| 유사도 계산 | $\cos(v_w, v_c)$ | 불가 (문맥 없음) | $\cos(h_w^{(1)}, h_w^{(2)})$ |

---

### 실험 결과

#### Loss 수렴

코퍼스 "you say goodbye and i say hello", 6개 단어, 윈도우 1, $K=5$:

```
epoch   1 | loss 4.1123
epoch   3 | loss 3.7652
epoch  50 | loss 1.5175
epoch 200 | loss 1.4879
```

초기 loss의 이론값: 랜덤 초기화에서 내적 $\approx 0$이므로 $\sigma(0) = 0.5$. 각 샘플의 loss:

$$-\log(0.5) - 5 \times \log(0.5) = 6 \times 0.693 \approx 4.16$$

실측 4.11과 거의 일치. 학습이 진행되면 정답 내적은 커지고 부정 내적은 작아져 loss가 감소한다.

#### 유사 단어 벡터 검증

패턴 "a X b Y a X b Y ..." 반복 (a=0, X=2, b=1, Y=3). 윈도우 1에서:
- a의 문맥: {Y, X}, b의 문맥: {X, Y} → **a와 b는 같은 문맥 집합**
- X의 문맥: {a, b}, Y의 문맥: {b, a} → **X와 Y는 같은 문맥 집합**

학습 후:

```
cos(a, b) = 0.9768  ← 같은 역할 (주변에 X/Y)
cos(X, Y) = 0.9912  ← 같은 역할 (주변에 a/b)
cos(a, X) = -0.0373 ← 다른 역할 (직교에 가까움)
```

같은 문맥 패턴의 단어(a↔b, X↔Y)는 코사인 유사도 $> 0.97$, 다른 패턴(a↔X)은 $\approx 0$. **분포 가설이 벡터 공간에 정확히 인코딩**되었다.

PMI 행렬로 이해하면: a와 b는 동일한 PMI 행(X,Y에 대해 같은 값)을 갖으므로, 행렬 분해 시 같은 벡터에 매핑된다. a와 X는 PMI 행이 완전히 다르므로 직교한다.

---

### GloVe와의 비교: 암묵적 vs 명시적 행렬 분해

GloVe (Pennington et al., 2014)는 Word2Vec의 암묵적 행렬 분해를 **명시적으로** 수행:

$$\min_{W, \tilde{W}} \sum_{i,j} f(X_{ij})\left(W_i^\top \tilde{W}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

$X_{ij}$: 동시 출현 횟수. $f(X)$: 가중 함수 ($X$가 작으면 가중치 낮춤).

| | Word2Vec (SGNS) | GloVe |
|---|---|---|
| 데이터 | 개별 (center, context) 쌍 | 동시 출현 행렬 $X$ (사전 계산) |
| 최적화 대상 | PMI 행렬 (암묵적) | $\log X$ 행렬 (명시적) |
| 학습 방식 | SGD on 개별 샘플 | SGD on 행렬 원소 |
| 장점 | 온라인 학습 가능 | 글로벌 통계 직접 활용 |
| 메모리 | $O(V \times D)$ | $O(V^2)$ (동시 출현 행렬) |

실험적으로 두 방법의 성능은 **거의 동일**하다 (Levy et al., 2015). 이것은 둘이 같은 것을 학습한다는 Levy & Goldberg의 이론적 결과와 일치한다.

---

### 고빈도 단어 서브샘플링 (Subsampling)

Mikolov et al.은 학습 전에 **고빈도 단어를 확률적으로 제거**하는 기법도 제안했다:

$$P(\mathrm{discard} \mid w) = 1 - \sqrt{\frac{t}{f(w)}}$$

$t \approx 10^{-5}$가 임계값. "the" ($f \approx 0.05$)의 제거 확률은 약 98.6%.

왜 효과적인가:
1. **계산 절약**: 가장 빈번한 단어가 대부분 제거됨
2. **학습 품질 향상**: "the cat"에서 "the"는 의미 정보가 거의 없음. 제거하면 "cat"이 더 유의미한 문맥 단어와 쌍을 이룸
3. **Effective window 확대**: "the"가 제거되면 실제로는 떨어져 있던 단어가 윈도우 안에 들어옴

---

### 로드맵

- Step 68: BERT (Encoder Transformer, 양방향 마스크 복원)
- **Step 69: Word2Vec (Skip-gram + Negative Sampling)** ← 현재
- 다음: Sentence Embedding (BERT + Contrastive Learning) 또는 벡터 검색 (Phase 4)
