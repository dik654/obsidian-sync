## Step 66: TransformerBlock — GPT의 반복 단위

### 한마디 직관

- **TransformerBlock = Attention + FFN을 Residual로 감싼 것** — GPT의 "벽돌 한 장"
- 이 블록을 $N$번 쌓으면 GPT의 본체가 완성된다

Step 61~65에서 만든 모든 부품을 하나의 모듈로 조립하는 단계. 새로운 수학 연산은 없지만, **왜 이 구조인가**에 대한 깊은 이해가 핵심이다.

---

### 블록의 구조: Pre-LN Transformer

#### 전체 흐름

```
입력 x (B, T, D)
  │
  ├─────────────────────────┐
  ↓                         │
LayerNorm ──→ Attention ──→ Dropout ──→ + (residual)
                                        │
  ├─────────────────────────────────────┘
  ↓                         │
LayerNorm ──→ FFN(GELU) ──→ Dropout ──→ + (residual)
                                        │
                                      출력 (B, T, D)
```

수식으로:

$$h = x + \mathrm{Dropout}(\mathrm{Attention}(\mathrm{LN}(x)))$$
$$\mathrm{out} = h + \mathrm{Dropout}(\mathrm{FFN}(\mathrm{LN}(h)))$$

두 개의 서브블록이 있고, 각각 동일한 패턴 `정규화 → 변환 → 정규화 → 잔차`를 따른다. 이 패턴이 왜 필요한지를 하나씩 뜯어본다.

---

### Residual Connection: 깊은 네트워크의 생명선

#### 근본 문제: 기울기 소실

10개 층을 쌓은 네트워크에서 역전파를 생각하자. Chain rule에 의해:

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_{10}} \cdot \frac{\partial x_{10}}{\partial x_9} \cdot \frac{\partial x_9}{\partial x_8} \cdots \frac{\partial x_1}{\partial x_0}$$

각 야코비안 $\frac{\partial x_{i+1}}{\partial x_i}$의 특이값(singular value)이 1보다 약간 작으면 (예: 0.9), 10개를 곱하면 $0.9^{10} \approx 0.35$. 100개를 쌓으면 $0.9^{100} \approx 2.6 \times 10^{-5}$. 기울기가 사실상 사라진다.

반대로 특이값이 1보다 약간 크면 (예: 1.1), $1.1^{100} \approx 13{,}781$. 기울기가 폭발한다.

$$\text{소실}: \prod_{i=1}^{N} \sigma_i < 1 \quad \text{폭발}: \prod_{i=1}^{N} \sigma_i > 1$$

특이값이 **정확히 1**이어야 안정적인데, 이는 모든 층이 직교 변환(orthogonal transformation)이어야 한다는 비현실적 조건이다.

#### Residual의 해결: 곱셈을 덧셈으로

$$y = x + F(x) \quad \Rightarrow \quad \frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$$

항등 행렬 $I$가 더해져 있으므로, 야코비안의 특이값이 **최소 1**로 보장된다. $\frac{\partial F}{\partial x}$가 0에 가까워도 기울기가 $I$를 통해 그대로 흐른다.

$N$개 블록을 쌓으면:

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_N} \prod_{i=1}^{N} \left(I + \frac{\partial F_i}{\partial x_i}\right)$$

이 곱을 전개하면 **$2^N$개의 경로**가 나온다:

$$= \frac{\partial L}{\partial x_N} \left( I + \sum_{i} \frac{\partial F_i}{\partial x_i} + \sum_{i<j} \frac{\partial F_j}{\partial x_j}\frac{\partial F_i}{\partial x_i} + \cdots \right)$$

첫 번째 항 $I$가 **직통 경로(skip path)**: 모든 $F$를 건너뛰고 기울기가 입력까지 직접 전달된다. 나머지 $2^N - 1$개의 항은 다양한 조합의 $F$를 거치는 경로다.

핵심: 직통 경로 $I$ 덕분에, 다른 모든 경로의 기울기가 0이 되더라도 최소한의 기울기가 보장된다. 이것이 ResNet(He et al., 2015)의 핵심 발견이며, 이후 Transformer에 그대로 적용되었다.

#### LSTM과의 비교

LSTM (step 60)의 cell state 경로를 다시 보자:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

forget gate $f_t \in (0, 1)$가 **곱해진다**. $f_t = 0.95$가 100 스텝 동안 유지되면 $0.95^{100} \approx 0.006$으로 여전히 소실 가능. forget gate가 1에 가까워야 하는데, 이는 새 정보를 거의 안 받아들인다는 뜻이므로 학습이 제한된다.

Residual은 **덧셈**이므로 이런 문제가 없다:

| | LSTM cell state | Residual |
|---|---|---|
| 정보 전달 | $c_t = f \odot c_{t-1} + \ldots$ (곱셈) | $x_{l+1} = x_l + F(x_l)$ (덧셈) |
| 기울기 | $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ (0~1) | $\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F}{\partial x}$ (≥1) |
| 100 스텝 | $\prod f_t \approx 0$ (소실 가능) | 직통 경로 $I$ 보장 |
| 선택적 망각 | forget gate로 제어 | 불가능 (항상 보존) |

Residual의 단점은 "선택적 망각"이 불가능하다는 것이다. 모든 정보가 계속 누적된다. 이 문제는 나중에 Attention이 어떤 정보에 집중할지 선택함으로써 간접적으로 해결한다.

#### Residual Stream: 현대적 해석

Anthropic의 mechanistic interpretability 연구에서 제안한 **residual stream** 관점은 Transformer를 이해하는 강력한 프레임워크다.

Residual connection 덕분에 블록의 입출력이 **동일한 벡터 공간** $\mathbb{R}^D$에 살고 있다. 이를 하나의 "개울(stream)"로 보면:

```
x₀ ──→──→──→──→──→──→──→──→──→──→ xₙ
      ↑         ↑         ↑
    +F₁(x)   +F₂(x)   +F₃(x)
```

$x_0$(초기 임베딩)에서 시작하여, 각 블록이 **정보를 더하는(write)** 방식으로 표현을 점진적으로 구축한다. 각 Attention 헤드와 FFN은 이 stream에서 필요한 정보를 **읽고(read)**, 자신의 계산 결과를 **쓴다(write)**.

$$x_N = x_0 + \sum_{l=1}^{N} \left( \mathrm{Attn}_l(\mathrm{LN}(x_{l-1})) + \mathrm{FFN}_l(\mathrm{LN}(x_{l-1} + \mathrm{Attn}_l)) \right)$$

이 관점에서:
- **Embedding**은 stream의 초기값 설정
- **Attention**은 다른 위치에서 정보를 읽어와 stream에 추가
- **FFN**은 현재 위치의 정보를 변환하여 stream에 추가
- **LM Head (unembedding)**는 최종 stream에서 다음 토큰 확률을 읽어냄

각 블록은 stream을 **파괴하지 않고 보강**한다. 이것이 residual connection의 본질적 의미다.

#### 초기 상태: 항등 변환에 가까움

학습 초기에 Xavier 초기화로 가중치가 작은 값($\sim 1/\sqrt{D}$)이므로:

$$F(x) \approx 0 \quad \Rightarrow \quad \mathrm{out} \approx x + 0 = x$$

블록이 거의 항등 변환이 되어, 깊은 네트워크도 **처음부터 안정적**이다. 96개 블록(GPT-3)을 쌓아도 초기에는 입력이 거의 그대로 통과한다. 학습이 진행되면서 $F(x)$가 점점 유의미한 변환을 학습한다.

실제 테스트 결과도 이를 확인: 초기화 직후 출력이 입력과 유사하다 (residual ratio로 측정).

---

### Pre-LN vs Post-LN: 정규화 위치의 영향

#### Post-LN의 문제를 구체적으로

Post-LN에서 $N$개 블록의 출력:

$$x_N = \mathrm{LN}(\mathrm{LN}(\cdots \mathrm{LN}(x_0 + F_1(x_0)) + F_2(\cdot) \cdots) + F_N(\cdot))$$

LayerNorm이 residual 합산 **이후에** 적용된다. 문제를 분석하면:

**순전파의 문제**: $x + F(x)$에서 $F(x)$의 크기가 작으면 residual이 지배적이고, 크면 $F(x)$가 지배적이다. 이 **두 스케일의 혼합**을 LayerNorm이 정규화해야 하는데, 층이 깊어질수록 residual이 누적되어 스케일이 커진다.

**역전파의 문제**: 기울기 경로에 LayerNorm의 야코비안이 **직렬로** 삽입된다:

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_N} \prod_{i=l+1}^{N} \frac{\partial \mathrm{LN}_i}{\partial (\cdot)} \cdot \left(I + \frac{\partial F_i}{\partial x_i}\right)$$

LayerNorm의 야코비안은 step 64에서 유도한 바와 같이 복잡하다:

$$\frac{\partial \hat{x}_j}{\partial x_d} = \frac{1}{\sigma}\left(\delta_{jd} - \frac{1}{D} - \frac{\hat{x}_j \hat{x}_d}{D}\right)$$

이 행렬이 $N$번 곱해지면 기울기가 불안정해진다. 특히 $\sigma$(표준편차)가 층마다 달라지므로, 곱의 크기를 예측하기 어렵다.

**실험적 증거**: Xiong et al. (2020, "On Layer Normalization in the Transformer Architecture")에서 Post-LN은 learning rate warmup 없이 발산하지만, Pre-LN은 warmup 없이도 안정적으로 수렴함을 보였다.

#### Pre-LN이 해결하는 방식

$$\mathrm{out} = x + F(\mathrm{LN}(x))$$

기울기:

$$\frac{\partial \mathrm{out}}{\partial x} = I + \frac{\partial F}{\partial \mathrm{LN}} \cdot \frac{\partial \mathrm{LN}}{\partial x}$$

$I$ (직통 경로)가 LayerNorm을 **완전히 우회**한다. LayerNorm의 복잡한 야코비안은 $F$ 경로에만 영향을 미치고, 직통 경로는 순수한 항등 변환이다.

$N$개 블록에서:

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_N} + \text{(}F\text{를 거치는 항들)}$$

첫 번째 항이 **LayerNorm 없이** 직접 전달된다. 이것이 Pre-LN이 안정적인 핵심 이유다.

#### Post-LN의 장점은 없는가?

있다. Post-LN은 **최종 출력의 크기가 일정**하다 (마지막 LayerNorm이 정규화하므로). Pre-LN은 residual이 누적되어 출력 크기가 커질 수 있다. 최근 연구에서는 Pre-LN의 마지막에 추가 LayerNorm을 넣어 이 문제를 해결한다 (GPT-2에서도 마지막에 `ln_f`를 추가).

| | Post-LN | Pre-LN |
|---|---|---|
| 정규화 위치 | 서브레이어 **뒤** | 서브레이어 **앞** |
| 학습 안정성 | warmup 필수 | warmup 불필요 |
| 기울기 경로 | LN 야코비안이 직통 경로에 포함 | 직통 경로가 LN 우회 |
| 출력 크기 | 일정 (LN이 마지막) | 누적 (추가 LN 필요) |
| 성능 (수렴 후) | **약간 더 나은 경우 있음** | 비슷하거나 약간 낮음 |
| 사용처 | 원 Transformer (2017) | **GPT-2/3, LLaMA** (현재 표준) |

---

### Attention과 FFN의 역할 분담

#### Attention은 선형이다

Attention의 출력을 다시 보자:

$$\mathrm{output}_i = \sum_j \alpha_{ij} V_j$$

가중치 $\alpha_{ij} \geq 0$이고 $\sum_j \alpha_{ij} = 1$ (softmax 출력). 이것은 $V$ 벡터들의 **볼록 결합(convex combination)**이다.

볼록 결합의 기하학적 의미: 출력 벡터는 항상 $V$ 벡터들이 이루는 볼록 껍질(convex hull) 안에 있다. 새로운 방향이나 크기를 생성할 수 없다. 입력에 없는 정보는 만들어낼 수 없다는 뜻이다.

구체적 예시: 3개의 토큰 벡터가 $(1, 0)$, $(0, 1)$, $(-1, 0)$이라면, Attention의 출력은 이 세 점이 이루는 삼각형 안의 점만 가능하다. $(0, 2)$ 같은 점은 절대 만들 수 없다.

#### FFN이 비선형성을 추가

$$\mathrm{FFN}(x) = \mathrm{GELU}(x W_1 + b_1) W_2 + b_2$$

GELU는 원소별 비선형 함수이므로, 입력 공간의 볼록 껍질을 **벗어날 수 있다**. $D$ 차원을 $4D$ 차원으로 확장한 뒤 비선형 활성화를 적용하면, 입력에 없던 새로운 특징을 생성할 수 있다.

#### FFN을 Key-Value Memory로 해석하기

Geva et al. (2021, "Transformer Feed-Forward Layers Are Key-Value Memories")의 통찰:

$W_1$의 각 행을 "key" $k_i$, $W_2$의 각 열을 "value" $v_i$로 보면:

$$\mathrm{FFN}(x) = \sum_{i=1}^{4D} \underbrace{\mathrm{GELU}(x \cdot k_i + b_{1,i})}_{\text{매칭 점수}} \cdot v_i$$

이것은 **암묵적 검색(implicit retrieval)**이다:
1. 입력 $x$와 각 "key" $k_i$의 유사도를 계산
2. GELU로 thresholding (관련 없는 key는 비활성화)
3. 활성화된 key에 대응하는 "value" $v_i$를 가중합

$4D$개의 key-value 쌍이 **학습된 지식**을 저장한다. 실제로 GPT-2의 FFN을 분석하면, 특정 뉴런이 특정 패턴에 반응하는 것이 관찰된다:

- 어떤 뉴런은 "날짜 뒤에 연도가 올 때" 활성화
- 어떤 뉴런은 "인물 이름 뒤에 직업이 올 때" 활성화
- 어떤 뉴런은 "코드의 여는 괄호 뒤에" 활성화

이 관점에서 FFN은 **학습된 메모리 테이블**이고, 4배 확장은 메모리 슬롯의 수를 결정한다.

#### 왜 두 단계가 교대해야 하는가

Attention과 FFN이 **교대하는** 이유를 더 깊이 이해하기 위해, 각 블록에서 일어나는 일을 단계별로 추적하자:

**블록 1**:
1. Attention: 인접 토큰들의 정보를 수집 (주로 로컬 패턴, bigram 수준)
2. FFN: 수집한 정보를 기반으로 "주어+동사 패턴 감지" 같은 특징 생성

**블록 3~6** (중간):
1. Attention: 블록 1~2에서 생성된 고수준 특징을 기반으로 더 넓은 범위의 토큰 참조
2. FFN: "문장의 주제 파악", "감정 분석" 같은 더 추상적인 특징 생성

**블록 10~12** (후반):
1. Attention: 전체 문맥을 종합하여 다음 토큰 예측에 필요한 정보 수집
2. FFN: 최종 예측을 위한 구체적 토큰 후보 활성화

이처럼 **"수집(Attention) → 처리(FFN)"의 반복**이 점점 추상적인 표현을 구축한다. Attention만으로는 수집만 하고 처리를 못하며, FFN만으로는 각 토큰이 고립되어 문맥을 볼 수 없다.

---

### FFN의 확장 비율: 왜 4배인가

#### 수학적 관점: rank와 표현력

$\mathrm{FFN}(x) = \mathrm{GELU}(x W_1) W_2$에서 (bias 생략):

만약 GELU가 없다면 $x W_1 W_2$인데, $W_1 \in \mathbb{R}^{D \times 4D}$와 $W_2 \in \mathbb{R}^{4D \times D}$의 곱 $W_1 W_2 \in \mathbb{R}^{D \times D}$이므로 $D \times D$ 행렬 하나와 동일하다. 즉 중간 확장이 의미 없다.

GELU가 **원소별 비선형성**을 추가하므로 중간 차원이 의미를 갖는다:
- $4D$ 차원에서 GELU가 각 뉴런을 독립적으로 활성화/비활성화
- 이 활성화 패턴의 조합이 $2^{4D}$개의 선형 영역(linear region)을 만듦
- 더 많은 뉴런 = 더 많은 선형 영역 = 더 복잡한 함수 근사 가능

확장 비율에 따른 선형 영역 수:

| 확장 비율 | 중간 차원 ($D=768$) | 최대 선형 영역 | 파라미터 |
|---|---|---|---|
| 1× | 768 | $2^{768}$ | $2D^2$ |
| **4×** | 3072 | $2^{3072}$ | $8D^2$ |
| 8× | 6144 | $2^{6144}$ | $16D^2$ |

선형 영역 수는 지수적으로 증가하지만, 파라미터는 선형으로 증가한다. 4배가 좋은 균형점인 이유: 2배는 표현력이 부족하고, 8배는 파라미터가 2배 더 필요하지만 성능 향상은 미미하다.

#### 3D → 2D 변환 패턴

Linear 레이어는 2D 입력 $(N, D)$만 처리한다. Transformer의 텐서는 3D $(B, T, D)$이므로 변환이 필요하다. 이 패턴은 CausalSelfAttention (step 65)에서도 동일하게 사용되었다:

```rust
// 3D → 2D: 배치와 시퀀스를 합침
let normed_2d = reshape(&normed, &[b * t, d]);     // (B*T, D)

// Linear 연산 (2D): 각 토큰에 동일한 변환 적용
let ffn_out = mlp_proj.forward(&gelu(&mlp_fc.forward(&normed_2d)));

// 2D → 3D: 원래 shape으로 복원
let ffn_out = reshape(&ffn_out, &[b, t, d]);       // (B, T, D)
```

각 토큰 위치 $(b, t)$에서 **동일한 가중치로 독립적 변환**을 적용하므로, $B \times T$개의 벡터를 한번에 처리하는 것과 동일하다. "Position-wise" Feed-Forward Network라고 불리는 이유다.

---

### Dropout의 세 위치

TransformerBlock에서 dropout이 적용되는 위치가 세 곳이다:

1. **Attention dropout** (CausalSelfAttention 내부): attention 가중치 $\alpha_{ij}$의 일부를 0으로
2. **Residual dropout 1**: Attention 출력에, residual 합산 직전
3. **Residual dropout 2**: FFN 출력에, residual 합산 직전

각 위치의 역할:

| 위치 | 대상 | 효과 |
|---|---|---|
| Attention | 가중치 $\alpha_{ij}$ | 특정 토큰에 과도하게 의존하는 것 방지 |
| Residual 1 | Attention 출력 전체 | 특정 feature에 과적합 방지 |
| Residual 2 | FFN 출력 전체 | FFN의 특정 뉴런에 과적합 방지 |

왜 LayerNorm 뒤가 아니라 residual 합산 직전인가? LayerNorm 뒤에 dropout을 넣으면, 정규화된 값의 일부가 0이 되어 평균/분산이 왜곡된다. Residual 합산 직전에 넣으면, dropout이 "이 서브레이어의 기여를 일부 무시"하는 효과가 되어 더 자연스럽다.

---

### 블록 수와 성능: Scaling Law

블록 수 $N$, 모델 차원 $D$, 총 파라미터 수 $P$의 관계:

$$P \approx N \times (12 D^2) + V \times D$$

(블록당 $\approx 12D^2$ 파라미터: Attention $4D^2$ + FFN $8D^2$, $V$는 vocabulary size)

Kaplan et al. (2020, "Scaling Laws for Neural Language Models")의 발견:

$$L(P) = \left(\frac{P_c}{P}\right)^{\alpha_P}$$

손실 $L$은 파라미터 수 $P$의 **멱법칙(power law)**을 따른다. $\alpha_P \approx 0.076$. 이것은 파라미터를 10배 늘려도 손실은 $10^{0.076} \approx 1.19$배만 줄어든다는 뜻이다.

| 모델 | $N$ (블록) | $D$ | $H$ | 파라미터 |
|---|---|---|---|---|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| GPT-2 Medium | 24 | 1024 | 16 | 345M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-3 | 96 | 12288 | 96 | 175B |

블록 수와 $D$를 동시에 늘린다. $D$만 키우면 한 블록 안에서의 표현력은 올라가지만 깊이(depth)가 부족하고, $N$만 키우면 각 블록의 용량이 부족하다.

---

### 전체 파라미터 구성

$D=8$, $H=2$ 기준 (우리 구현):

| 컴포넌트 | Shape | 원소 수 |
|---|---|---|
| LN1 $\gamma$, $\beta$ | $(D,)$ × 2 | 16 |
| Attn $W_Q, W_K, W_V, W_O$ | $(D, D)$ × 4 | 256 |
| Attn $b_Q, b_K, b_V, b_O$ | $(D,)$ × 4 | 32 |
| LN2 $\gamma$, $\beta$ | $(D,)$ × 2 | 16 |
| FFN $W_1$ | $(D, 4D)$ | 256 |
| FFN $b_1$ | $(4D,)$ | 32 |
| FFN $W_2$ | $(4D, D)$ | 256 |
| FFN $b_2$ | $(D,)$ | 8 |
| **총 16개 텐서** | | **872** |

파라미터 분포: Attention이 약 33%, FFN이 약 63%, LayerNorm이 약 4%. **FFN이 블록 파라미터의 대부분**을 차지한다. 이것이 "FFN = 학습된 메모리"라는 해석과 일치한다 — 메모리를 많이 저장하려면 많은 파라미터가 필요하다.

---

### 코드 발췌

```rust
pub struct TransformerBlock {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp_fc: Linear,      // D → 4D
    mlp_proj: Linear,    // 4D → D
    resid_dropout: f64,
}

pub fn forward(&self, x: &Variable) -> Variable {
    let (b, t, d) = (shape[0], shape[1], shape[2]);

    // Sub-block 1: LN → Attention → Dropout → Residual
    let h = x + &dropout(
        &self.attn.forward(&self.ln1.forward(x)),
        self.resid_dropout,
    );

    // Sub-block 2: LN → FFN(GELU) → Dropout → Residual
    let normed = self.ln2.forward(&h);
    let normed_2d = reshape(&normed, &[b * t, d]);
    let ffn_out = self.mlp_proj.forward(
        &gelu(&self.mlp_fc.forward(&normed_2d))
    );
    let ffn_out = reshape(&ffn_out, &[b, t, d]);
    &h + &dropout(&ffn_out, self.resid_dropout)
}
```

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
[x] Multi-Head Attention      ← step 65
[x] TransformerBlock           ← step 66 (이번)
[ ] GPT 모델 (Embedding + N × Block + LM Head)
[ ] 텍스트 생성
```

다음은 **GPT 모델** — Embedding(토큰 + 위치) + TransformerBlock × N + LayerNorm + LM Head로 완전한 언어 모델을 완성.
