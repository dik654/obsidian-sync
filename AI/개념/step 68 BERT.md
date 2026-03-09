## Step 68: BERT — Bidirectional Encoder Transformer

### 한마디 직관

- **BERT = "빈칸 채우기 기계"** — 주변 문맥을 양방향으로 보고 마스크된 토큰을 복원
- GPT와 95% 동일한 구조에서 **causal mask 하나를 빼면** 양방향이 된다

"The [MASK] sat on the mat" → BERT → "[MASK] = cat (0.8), dog (0.1), ..."

---

### GPT → BERT: 코드 한 줄의 차이

#### 양방향의 비밀

GPT의 attention 연산:

$$\mathrm{scores} = \frac{QK^\top}{\sqrt{d_k}}$$

여기에 **causal mask**를 적용한다:

$$\mathrm{scores}_{ij} \leftarrow \begin{cases} \mathrm{scores}_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

$j > i$인 위치 (미래 토큰)를 $-\infty$로 만들면, softmax 후 attention weight가 0이 된다:

$$\alpha_{ij} = \frac{\exp(\mathrm{scores}_{ij})}{\sum_k \exp(\mathrm{scores}_{ik})} \xrightarrow{\mathrm{scores}_{ij} = -\infty} 0$$

BERT는 이 마스크를 **제거**한다. 즉 모든 $j$에 대해 $\alpha_{ij} > 0$:

```rust
let masked_scores = if self.use_causal_mask {
    causal_mask(&scores)  // GPT: 미래 차단
} else {
    scores                // BERT: 전부 참조
};
```

이것이 전부다. **코드 한 줄**의 차이가 단방향 모델과 양방향 모델을 가른다.

#### 수학적으로 보는 차이

GPT에서 위치 $t$의 출력은:

$$h_t^{\mathrm{GPT}} = \sum_{j=0}^{t} \alpha_{tj} V_j$$

BERT에서 위치 $t$의 출력은:

$$h_t^{\mathrm{BERT}} = \sum_{j=0}^{T-1} \alpha_{tj} V_j$$

합산 범위가 $[0, t]$에서 $[0, T-1]$로 확장된다. 이것은 각 토큰이 **모든 다른 토큰의 정보**를 참조할 수 있다는 뜻이다.

#### 실험적 검증: 인과성 vs 양방향성

GPT에서는 마지막 토큰을 변경해도 이전 위치의 출력이 **불변**이다 (causal mask가 미래를 차단하므로). BERT에서는 **모든 위치의 출력이 변한다**:

```
// GPT (step 67): 마지막 토큰 변경 → t=0,1,2 불변
ids1 = [1, 3, 5, 7]  →  logits1
ids2 = [1, 3, 5, 2]  →  logits2
assert!(logits1[0..3] == logits2[0..3])  // ✓ 동일

// BERT: 마지막 토큰 변경 → t=0,1,2도 변경
ids1 = [1, 3, 5, 7]  →  logits1
ids2 = [1, 3, 5, 2]  →  logits2
assert!(logits1[0..3] != logits2[0..3])  // ✓ 달라짐
```

이것이 **양방향 어텐션의 핵심**: 어떤 위치를 바꾸면 모든 위치에 영향이 전파된다.

---

### Attention 행렬의 형태 비교

#### GPT: 하삼각 행렬

$$A^{\mathrm{GPT}} = \begin{pmatrix}
\alpha_{00} & 0 & 0 & 0 \\
\alpha_{10} & \alpha_{11} & 0 & 0 \\
\alpha_{20} & \alpha_{21} & \alpha_{22} & 0 \\
\alpha_{30} & \alpha_{31} & \alpha_{32} & \alpha_{33}
\end{pmatrix}$$

각 행의 합 = 1 (softmax). 위치 $t$는 자기 자신과 과거만 참조한다.

#### BERT: 완전 행렬 (Full Attention)

$$A^{\mathrm{BERT}} = \begin{pmatrix}
\alpha_{00} & \alpha_{01} & \alpha_{02} & \alpha_{03} \\
\alpha_{10} & \alpha_{11} & \alpha_{12} & \alpha_{13} \\
\alpha_{20} & \alpha_{21} & \alpha_{22} & \alpha_{23} \\
\alpha_{30} & \alpha_{31} & \alpha_{32} & \alpha_{33}
\end{pmatrix}$$

모든 원소가 양수. 위치 $t$는 **모든 위치**를 참조한다.

#### 정보 흐름 그래프

```
GPT:                            BERT:
t=0 → t=0                      t=0 ↔ t=0, t=1, t=2, t=3
t=0,t=1 → t=1                  t=0, t=1, t=2, t=3 ↔ t=1
t=0,t=1,t=2 → t=2              t=0, t=1, t=2, t=3 ↔ t=2
t=0,t=1,t=2,t=3 → t=3          t=0, t=1, t=2, t=3 ↔ t=3
```

GPT는 **단방향 정보 흐름**, BERT는 **완전 연결 정보 흐름**.

---

### 왜 양방향이 더 나은가 (그리고 왜 GPT는 단방향인가)

#### 양방향의 장점: 문맥 완전 활용

"I went to the **bank** to deposit money"에서 "bank"의 의미를 결정하려면:
- 왼쪽만 보면: "I went to the" → 은행? 강둑?
- **오른쪽도 보면**: "to deposit money" → **은행**!

양방향 모델은 후속 문맥까지 활용하여 더 정확한 표현을 학습한다.

#### 그런데 왜 GPT는 단방향인가?

**생성을 할 수 없기 때문이다.** BERT는 양방향이라 다음 토큰을 순차적으로 생성하는 것이 불가능하다. 생성 시에는 아직 존재하지 않는 미래 토큰을 참조할 수 없으므로, causal mask가 필수적이다.

| | GPT | BERT |
|---|---|---|
| Attention | 단방향 (causal) | 양방향 (full) |
| 학습 목표 | 다음 토큰 예측 (NTP) | 마스크 토큰 복원 (MLM) |
| **생성 가능** | ✓ (자기회귀) | ✗ |
| **이해 능력** | 약함 | 강함 |
| 용도 | 텍스트 생성, 대화 | 분류, NER, QA |

#### Autoregressive vs Autoencoding

이 두 접근법은 NLP의 근본적인 학습 패러다임 차이다:

**Autoregressive (GPT)**: 확률 분해의 **연쇄 법칙(chain rule)** 사용

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

한 방향으로 순차적으로 조건부 확률을 곱한다. 자연스럽게 생성이 가능하다.

**Autoencoding (BERT)**: **노이즈 제거(denoising)** 방식

$$\hat{x} = f_\theta(\tilde{x}) \quad \text{where } \tilde{x} = \mathrm{mask}(x)$$

입력의 일부를 망가뜨리고 (마스킹) 원래를 복원한다. 양방향 문맥을 자연스럽게 활용한다.

---

### BERT의 아키텍처

#### GPT 대비 추가 요소: Segment Embedding

BERT는 **두 문장을 하나의 입력으로** 처리해야 하는 태스크가 있다 (NSP, QA 등):

```
[CLS] 문장A의 토큰들 [SEP] 문장B의 토큰들 [SEP]
  0    0   0   0   0   0    1   1   1   1   1     ← segment ID
```

Segment Embedding은 **어떤 문장에 속하는지**를 알려주는 추가 임베딩이다:

$$h_0 = E_{\mathrm{tok}}[x] + E_{\mathrm{pos}}[0, 1, \ldots, T-1] + E_{\mathrm{seg}}[\mathrm{seg\_ids}]$$

GPT에는 2개 (Token + Position), BERT에는 3개 (Token + Position + Segment).

```rust
// 3개 임베딩 합산
let tok_emb = self.token_emb.forward(token_ids);   // (B, T, D)
let pos_emb = self.pos_emb.forward(&pos_idx);      // (T, D) → broadcast
let seg_emb = self.segment_emb.forward(segment_ids); // (B, T, D)
let x = &(&tok_emb + &pos_emb) + &seg_emb;
```

$E_{\mathrm{seg}} \in \mathbb{R}^{2 \times D}$로, 단 2개의 임베딩 벡터만 있다 (문장 A=0, 문장 B=1).

#### 전체 구조 다이어그램

```
token_ids [42, 17, 99, 5]     segment_ids [0, 0, 1, 1]    (B, T)
      │                            │
      ↓                            ↓
  Token Emb                   Segment Emb
      │                            │
      └──────── + ────── + ────────┘
               │       │
          Pos Emb      │
               │       │
               └───+───┘           (B, T, D)
                   │
         TransformerBlock × N      is_causal=false (양방향!)
                   │
               LayerNorm
                   │
          Linear (MLM Head)        D → vocab_size
                   │
               logits              (B, T, vocab_size)
```

---

### MLM: Masked Language Modeling

#### 학습 절차

1. 입력 시퀀스에서 **15%의 토큰을 선택**
2. 선택된 토큰 중:
   - 80%는 `[MASK]` 토큰으로 교체
   - 10%는 랜덤 토큰으로 교체
   - 10%는 그대로 유지
3. 모델은 **선택된 위치의 원래 토큰을 예측**

#### 왜 15%만 마스킹하는가

100%를 마스킹하면 양방향 문맥이 아예 없어진다 (모든 것이 마스크). 너무 적으면 학습 효율이 떨어진다. 15%는 "충분한 문맥을 보면서도 예측할 것이 있는" 최적 지점이다.

실험적으로 10~20%가 비슷한 성능, 15%가 약간 최적.

#### 왜 100%를 [MASK]로 바꾸지 않는가

미세 조정(fine-tuning) 시에는 `[MASK]` 토큰이 **없다**. 사전 학습에서 항상 `[MASK]`만 보면, 실제 토큰에 대한 표현 학습이 부족해진다.

- 80% `[MASK]`: 복원 능력 학습
- 10% 랜덤: "이 토큰이 올바른가?" 판별 능력
- 10% 유지: 원래 토큰에 대한 표현 학습

#### Loss: masked_softmax_cross_entropy

GPT는 **모든 위치**에서 loss를 계산한다. BERT는 **마스크된 위치만**:

$$L = -\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \log P(x_i \mid \tilde{x})$$

$\mathcal{M}$은 마스크된 위치의 집합, $\tilde{x}$는 마스킹된 입력.

구현에서 주의할 점: `masked_softmax_cross_entropy`는 **계산 그래프를 유지**해야 한다. 단순히 마스크된 데이터를 추출하면 그래프가 끊어져 backward가 동작하지 않는다:

```rust
// ✗ 잘못된 구현: 데이터 추출 → 그래프 단절
let masked_data = logits.data();  // ArrayD (그래프 없음)
let new_var = Variable::new(masked_data);  // backward 불가

// ✓ 올바른 구현: Function trait으로 마스크 적용
struct MaskedSoftmaxCrossEntropyFn { t, mask }
impl Function for MaskedSoftmaxCrossEntropyFn {
    fn forward(..) { /* 마스크 위치만 loss 합산 */ }
    fn backward(..) {
        // 마스크 위치: (softmax - one_hot) / M
        // 비마스크 위치: 0 (gradient 없음)
    }
}
```

backward에서 gradient의 형태:

$$\frac{\partial L}{\partial \mathrm{logits}_{i,j}} = \begin{cases}
\frac{1}{|\mathcal{M}|}(\mathrm{softmax}(z_i)_j - \mathbf{1}_{j = t_i}) & \text{if } i \in \mathcal{M} \\
0 & \text{otherwise}
\end{cases}$$

마스크되지 않은 위치에는 gradient가 0이므로, 해당 위치의 예측은 학습에 영향을 주지 않는다.

---

### Pre-training과 Fine-tuning

#### BERT의 사전 학습 태스크

원래 BERT (Devlin et al., 2018)는 두 가지 태스크로 사전 학습한다:

**1. MLM (Masked Language Modeling)**: 위에서 설명한 마스크 복원

**2. NSP (Next Sentence Prediction)**: 두 문장이 연속인지 판별

```
입력: [CLS] 고양이가 앉았다 [SEP] 매트 위에 [SEP]
레이블: IsNext (연속)

입력: [CLS] 고양이가 앉았다 [SEP] 주가가 올랐다 [SEP]
레이블: NotNext (비연속)
```

`[CLS]` 토큰의 출력 벡터를 이진 분류에 사용한다. Segment embedding이 여기서 필요하다 (문장 A vs B).

> **후속 연구의 발견**: RoBERTa (Liu et al., 2019)는 NSP를 제거해도 성능이 **동일하거나 더 좋다**는 것을 보였다. MLM만으로 충분하다.

#### Fine-tuning 패러다임

BERT의 혁신은 **사전 학습 + 미세 조정** 패러다임을 확립한 것이다:

```
단계 1: 대규모 코퍼스로 MLM 사전 학습 (비지도 학습)
        → 범용 언어 표현 습득

단계 2: 태스크별 소량 데이터로 미세 조정 (지도 학습)
        → 분류, NER, QA 등 특정 태스크 해결
```

이것은 컴퓨터 비전의 ImageNet 사전 학습과 동일한 패러다임이다:

| | NLP (BERT) | Vision (ResNet) |
|---|---|---|
| 사전 학습 데이터 | Wikipedia + BookCorpus | ImageNet |
| 사전 학습 태스크 | MLM | 이미지 분류 |
| 미세 조정 | 분류 헤드 교체 | FC layer 교체 |

GPT도 미세 조정이 가능하지만, BERT가 양방향이므로 이해 태스크에서 일관되게 우수하다.

---

### 코드 설계: GPT와 BERT의 통합

#### 파라미터화로 코드 중복 제거

GPT와 BERT는 **attention mask 여부**만 다르다. 이를 기존 코드에 플래그로 추가하여 공유한다:

```rust
// SelfAttention: GPT/BERT 겸용
pub struct SelfAttention {
    use_causal_mask: bool,  // true=GPT, false=BERT
    // ... 나머지 필드 동일
}

// 하위 호환: 기존 GPT 코드는 변경 없이 동작
pub type CausalSelfAttention = SelfAttention;

impl SelfAttention {
    pub fn new(..) -> Self { Self::new_with_mask(.., true) }   // GPT (기본)
    pub fn new_with_mask(.., use_causal_mask) -> Self { .. }   // 명시적 선택
}
```

TransformerBlock도 동일하게:

```rust
impl TransformerBlock {
    pub fn new(..) -> Self { Self::new_with_causal(.., true) }   // GPT (기본)
    pub fn new_with_causal(.., is_causal) -> Self { .. }         // 명시적 선택
}
```

GPT는 `TransformerBlock::new()`를, BERT는 `TransformerBlock::new_with_causal(.., false)`를 사용. **기존 GPT 코드는 한 줄도 수정하지 않는다.**

---

### BERT vs GPT: 표현력의 차이

#### Information Bottleneck 관점

GPT에서 위치 $t$가 접근할 수 있는 정보량:

$$I_t^{\mathrm{GPT}} = H(x_0, x_1, \ldots, x_t)$$

BERT에서:

$$I_t^{\mathrm{BERT}} = H(x_0, x_1, \ldots, x_{T-1})$$

$I_t^{\mathrm{BERT}} \geq I_t^{\mathrm{GPT}}$ (등호는 $t = T-1$일 때만). 더 많은 정보에 접근할수록 더 정확한 표현을 만들 수 있다.

#### Layer 수에 따른 정보 전파

GPT에서 $L$개 layer를 거치면, 위치 $t$에서 정보가 도달할 수 있는 최대 거리는 $t$이다 (자기보다 왼쪽만). BERT에서는 **1개 layer만으로** 모든 위치의 정보가 도달한다.

그래서 실험적으로 BERT는 같은 layer 수에서 더 좋은 표현을 학습한다.

---

### 실험 결과

#### MLM 학습 (step68 테스트)

패턴 `0,1,2,0,1,2,...`에서 짝수 위치를 마스킹하고 복원:

```
epoch   1 | loss 1.6683
epoch   3 | loss 0.8430
epoch  25 | loss 0.0092
epoch 100 | loss 0.0010
```

Loss가 1.67 → 0.001로 **99.9% 감소**. GPT의 char-level 학습과 비슷한 수렴 속도.

초기 loss가 $\ln(V) = \ln(5) \approx 1.61$에 가까운 것은 마스크 위치에서 균등 분포로 시작함을 의미.

#### MLM 예측 정확도

학습 후 `[MASK],1,2,[MASK],1,2`를 입력하면:

```
position 0 prediction: 0 (expected 0)  ✓
position 3 prediction: 0 (expected 0)  ✓
```

양방향 문맥 `_,1,2,...`를 보고 패턴 `0,1,2`의 첫 번째가 0임을 정확히 예측한다.

---

### BERT의 후속 발전

| 모델 | 연도 | 핵심 차이 |
|---|---|---|
| BERT | 2018 | MLM + NSP, WordPiece |
| RoBERTa | 2019 | NSP 제거, 더 많은 데이터, 동적 마스킹 |
| ALBERT | 2019 | 파라미터 공유 (embedding factorization) |
| ELECTRA | 2020 | MLM 대신 Replaced Token Detection (GAN식) |
| DeBERTa | 2020 | 분리된 어텐션 (content + position) |

RoBERTa의 핵심 발견:
- NSP는 불필요 → MLM만으로 충분
- 정적 마스킹(같은 위치 반복) → **동적 마스킹**(매 에폭 다른 위치) = 더 좋음
- 더 큰 배치, 더 많은 데이터 = 더 좋음

---

### 로드맵

- Step 67: GPT (Decoder-only, 자기회귀 생성)
- **Step 68: BERT (Encoder-only, 양방향 마스크 복원)** ← 현재
- 다음 Phase: Word2Vec 등 임베딩 모델
