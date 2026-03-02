## Step 64: LayerNorm과 GELU — Transformer의 정규화와 활성화

### 한마디 직관

- **LayerNorm = 각 샘플을 평균 0, 분산 1로** — 값의 스케일을 맞춰 학습을 안정화
- **GELU = 부드러운 ReLU** — 음수를 완전히 죽이지 않고 확률적으로 부드럽게 전환

Transformer의 매 블록에서: `LayerNorm → Attention → LayerNorm → FFN(GELU)` — 이 두 연산이 반복된다.

---

### 왜 정규화가 필요한가

#### Internal Covariate Shift (원래 동기)

딥러닝에서 각 레이어의 출력 분포는 학습 중에 계속 변한다. 앞쪽 레이어의 가중치가 업데이트되면 뒤쪽 레이어의 입력 분포가 달라진다. 이를 **Internal Covariate Shift**라 한다.

구체적으로: 레이어 $l$의 출력 $z^{(l)} = f(W^{(l)} z^{(l-1)})$에서, $W^{(l-1)}$이 업데이트되면 $z^{(l-1)}$의 분포가 바뀌고, 레이어 $l$은 **새로운 입력 분포에 다시 적응**해야 한다. 깊은 네트워크에서 이 효과가 누적되면 학습이 극도로 불안정해진다.

#### 더 근본적인 이유: 손실 곡면의 평탄화

2018년 Santurkar et al.의 연구에 따르면, 정규화의 실질적 이점은 Internal Covariate Shift 해소보다 **손실 곡면(loss landscape)을 매끄럽게** 만드는 것이다.

정규화 없이 $y = Wx$이면 $W$의 스케일이 곧 $y$의 스케일을 결정한다. $W$의 작은 변화가 깊은 층에서 증폭되어 손실 함수가 급격하게 변하는 "뾰족한" 곡면을 만든다. 정규화를 적용하면:

$$\hat{y} = \frac{y - \mu}{\sigma}$$

$y$의 스케일이 항상 일정하게 유지되므로, $W$의 변화가 이후 층에 미치는 영향이 제한된다. 결과적으로 **손실 곡면의 Lipschitz 상수가 작아져** 큰 학습률을 사용할 수 있고, 수렴이 빨라진다.

---

### LayerNorm vs BatchNorm: 왜 Transformer는 LayerNorm인가

#### BatchNorm (Batch Normalization, 2015)

배치 방향으로 정규화: 같은 feature의 값들을 배치 내에서 평균 0, 분산 1로 맞춤.

$$\hat{x}_{i,d} = \frac{x_{i,d} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}} \quad \text{where } \mu_d = \frac{1}{N}\sum_{i=1}^{N} x_{i,d}$$

| 축 | 의미 |
|---|---|
| $i$ | 배치 내 샘플 인덱스 |
| $d$ | feature 인덱스 |
| $\mu_d$ | feature $d$의 **배치 평균** |

```
배치 = [샘플1, 샘플2, 샘플3]
       feature: [f1, f2, f3]

BatchNorm: f1끼리 묶어서 정규화, f2끼리, f3끼리 (↓ 세로 방향)
```

**문제점**: 배치 크기에 의존. 배치가 1이면 통계량을 구할 수 없고, 배치가 작으면 추정이 불안정. 시퀀스 길이가 가변적인 Transformer에서 특히 문제.

#### LayerNorm (Layer Normalization, 2016)

각 샘플 내에서 feature 방향으로 정규화:

$$\hat{x}_{i,d} = \frac{x_{i,d} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \quad \text{where } \mu_i = \frac{1}{D}\sum_{d=1}^{D} x_{i,d}$$

| 축 | 의미 |
|---|---|
| $i$ | 샘플 (배치 내 위치, 또는 시퀀스 내 토큰) |
| $d$ | feature 인덱스 |
| $\mu_i$ | 샘플 $i$의 **feature 평균** |

```
배치 = [샘플1, 샘플2, 샘플3]
       feature: [f1, f2, f3]

LayerNorm: 샘플1의 [f1,f2,f3]을 정규화, 샘플2의 [f1,f2,f3]을... (→ 가로 방향)
```

**장점**: 배치 크기와 무관. 각 샘플을 독립적으로 처리하므로 배치가 1이어도 동작. 추론 시 별도의 이동 평균이 필요 없음.

| | BatchNorm | LayerNorm |
|---|---|---|
| 정규화 축 | 배치 (↓) | feature (→) |
| 통계량 의존 | 배치 크기에 의존 | 배치 크기와 무관 |
| 추론 시 | 이동 평균 필요 | 동일한 공식 사용 |
| 가변 시퀀스 | 문제됨 | 문제 없음 |
| 주 사용처 | CNN (이미지) | **Transformer (NLP)** |

#### 학습 가능한 파라미터: $\gamma$와 $\beta$

정규화만 하면 표현력이 제한된다 (항상 평균 0, 분산 1). 네트워크가 **필요하면 원래 분포로 되돌릴 수 있도록** 스케일($\gamma$)과 시프트($\beta$)를 학습 가능하게 한다:

$$y = \gamma \odot \hat{x} + \beta$$

초기값: $\gamma = [1, 1, \ldots, 1]$, $\beta = [0, 0, \ldots, 0]$ — 처음에는 항등 변환.

만약 정규화가 해로운 경우, 학습을 통해 $\gamma$와 $\beta$가 원래 분포를 복원할 수 있다 ($\gamma = \sigma$, $\beta = \mu$로 학습하면 정규화가 취소됨).

---

### LayerNorm 역전파 유도

순전파를 단계별로 정리:

$$\mu = \frac{1}{D}\sum_{d} x_d, \quad \sigma = \sqrt{\frac{1}{D}\sum_{d}(x_d - \mu)^2 + \epsilon}, \quad \hat{x}_d = \frac{x_d - \mu}{\sigma}, \quad y_d = \gamma_d \hat{x}_d + \beta_d$$

#### $g_\beta$와 $g_\gamma$

$y_d = \gamma_d \hat{x}_d + \beta_d$에서:

$$\frac{\partial L}{\partial \beta_d} = \sum_{\text{batch}} \frac{\partial L}{\partial y_d} = \sum_{\text{batch}} g_{y,d}$$

$$\frac{\partial L}{\partial \gamma_d} = \sum_{\text{batch}} g_{y,d} \cdot \hat{x}_d$$

배치의 모든 샘플에서 기울기를 합산. $\gamma$, $\beta$는 모든 샘플에서 공유되므로.

#### $g_x$: 가장 복잡한 부분

$g_{\hat{x}} = g_y \odot \gamma$로 놓으면, $\hat{x}_d = \frac{x_d - \mu}{\sigma}$에 대한 $g_x$를 구해야 한다.

**핵심 관찰**: $\hat{x}_d$는 $x_d$에만 의존하는 것이 아니라, $\mu$와 $\sigma$를 통해 **모든 $x_j$에 의존**한다. 따라서 $\frac{\partial \hat{x}_j}{\partial x_d}$는 $j = d$인 경우만이 아니라 모든 $j$에 대해 계산해야 한다.

$\hat{x}_j = \frac{x_j - \mu}{\sigma}$를 $x_d$에 대해 미분:

$$\frac{\partial \hat{x}_j}{\partial x_d} = \frac{1}{\sigma}\!\left(\delta_{jd} - \frac{1}{D}\right) - \frac{\hat{x}_j}{\sigma} \cdot \frac{\partial \sigma}{\partial x_d}$$

$\sigma$의 $x_d$ 미분을 구하면:

$$\sigma^2 = \frac{1}{D}\sum_k (x_k - \mu)^2 + \epsilon$$

$$\frac{\partial \sigma^2}{\partial x_d} = \frac{2}{D}(x_d - \mu) - \frac{2}{D^2}\sum_k(x_k - \mu) = \frac{2}{D}(x_d - \mu)$$

(두 번째 항은 $\sum_k(x_k - \mu) = 0$이므로 사라짐)

$$\frac{\partial \sigma}{\partial x_d} = \frac{1}{2\sigma} \cdot \frac{2}{D}(x_d - \mu) = \frac{x_d - \mu}{D\sigma} = \frac{\hat{x}_d}{D}$$

대입하면:

$$\frac{\partial \hat{x}_j}{\partial x_d} = \frac{1}{\sigma}\!\left(\delta_{jd} - \frac{1}{D} - \frac{\hat{x}_j \hat{x}_d}{D}\right)$$

chain rule 적용:

$$g_{x,d} = \sum_j g_{\hat{x},j} \cdot \frac{\partial \hat{x}_j}{\partial x_d} = \frac{1}{\sigma}\!\left(g_{\hat{x},d} - \frac{1}{D}\sum_j g_{\hat{x},j} - \frac{\hat{x}_d}{D}\sum_j g_{\hat{x},j}\hat{x}_j\right)$$

벡터 형태:

$$g_x = \frac{1}{\sigma}\!\left(g_{\hat{x}} - \mathrm{mean}(g_{\hat{x}}) - \hat{x} \cdot \mathrm{mean}(g_{\hat{x}} \odot \hat{x})\right)$$

여기서 $\mathrm{mean}$은 마지막 축(feature)을 따라 계산, keepdims=true.

```rust
fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
    let g_xhat = &gy * &gamma;                           // gy ⊙ γ
    let g_xhat_mean = g_xhat.mean_axis(last);            // mean(g_xhat)
    let g_xhat_xhat = &g_xhat * &x_hat;                 // g_xhat ⊙ x_hat
    let g_xhat_xhat_mean = g_xhat_xhat.mean_axis(last); // mean(g_xhat ⊙ x_hat)

    // gx = (1/σ) * (g_xhat - mean(g_xhat) - x_hat * mean(g_xhat ⊙ x_hat))
    let gx = std_inv * (g_xhat - g_xhat_mean - x_hat * g_xhat_xhat_mean);
}
```

**검증**: `sum(layer_norm(x))`의 $g_x$는 0이어야 한다. LayerNorm의 출력 합은 $\sum_d (\gamma_d \hat{x}_d + \beta_d)$이고, $\sum_d \hat{x}_d = 0$(평균 0이므로), 따라서 합은 $\sum_d \beta_d$로 $x$에 무관한 상수. 수치 미분과 일치 확인됨 (오차 < $10^{-3}$).

**$\sum_k(x_k - \mu) = 0$의 증명**: $\mu = \frac{1}{D}\sum_k x_k$이므로 $\sum_k(x_k - \mu) = \sum_k x_k - D\mu = D\mu - D\mu = 0$.

---

### GELU: Gaussian Error Linear Unit

#### ReLU의 한계

ReLU($x$) = $\max(0, x)$는 단순하고 효과적이지만:

- $x = 0$에서 미분 불가 (kink)
- $x < 0$이면 기울기가 0 → 뉴런이 "죽을" 수 있음 (dead neuron)
- 전환이 갑작스러움: 0 바로 아래와 위에서 행동이 완전히 다름

#### GELU의 아이디어: Dropout에서 출발

GELU를 이해하려면 **Dropout**에서 시작한다. Dropout은 각 뉴런의 출력 $x$에 Bernoulli 마스크 $m \in \{0, 1\}$을 곱한다:

$$\mathrm{Dropout}(x) = x \cdot m, \quad m \sim \mathrm{Bernoulli}(p)$$

여기서 $p$는 고정된 확률이다. GELU의 핵심 아이디어: **통과 확률 $p$를 입력값 $x$에 의존하게** 만들면 어떨까?

$$\mathrm{StochasticGELU}(x) = x \cdot m, \quad m \sim \mathrm{Bernoulli}(\Phi(x))$$

큰 양수 $x$는 높은 확률로 통과하고, 큰 음수 $x$는 높은 확률로 차단된다. 이것은 직관적으로 합리적이다 — 활성화가 큰 뉴런은 유지하고, 작은 뉴런은 제거하는 "adaptive dropout"이다.

**GELU는 이 확률적 연산의 기댓값(deterministic version)**이다:

$$\mathrm{GELU}(x) = \mathbb{E}[x \cdot m] = x \cdot \mathbb{E}[m] = x \cdot \Phi(x)$$

$\Phi(x)$는 표준정규분포의 **누적분포함수(CDF)**:

$$\Phi(x) = P(X \leq x) = \frac{1}{2}\!\left(1 + \mathrm{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)$$

왜 표준정규분포인가? BatchNorm이나 LayerNorm 이후의 값은 대략 $\mathcal{N}(0, 1)$을 따른다. 이 분포에서 $x$ 이하일 확률이 $\Phi(x)$이므로, "평균보다 얼마나 큰가"가 자연스럽게 통과 확률이 된다.

#### tanh 근사

$\mathrm{erf}$는 계산이 비싸므로, GPT-2/3에서는 tanh 근사를 사용:

$$\mathrm{GELU}(x) \approx 0.5x\!\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$

이 근사의 유도:

$\mathrm{erf}$와 $\tanh$는 둘 다 원점 대칭(기함수), $(-1, 1)$ 범위, S자 곡선이라는 공통 성질을 가진다. $\mathrm{erf}(x) \approx \tanh(cx)$에서 시작하여, $x^3$ 보정 항을 추가하면 정밀도가 크게 올라간다:

$$\mathrm{erf}(x) \approx \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x + ax^3\right)\right)$$

$\sqrt{2/\pi}$는 $x \to 0$에서 $\mathrm{erf}'(0) = 2/\sqrt{\pi}$와 $\tanh'(0) = 1$을 매칭한 것. $a = 0.044715$는 최소제곱 피팅으로 결정. 최대 오차는 $10^{-4}$ 이하.

#### 값 비교: GELU vs ReLU

| $x$ | ReLU | GELU |
|---|---|---|
| -1.0 | 0.000 | **-0.159** |
| -0.5 | 0.000 | **-0.154** |
| 0.0 | 0.000 | 0.000 |
| 0.5 | 0.500 | 0.346 |
| 1.0 | 1.000 | 0.841 |

핵심 차이:
- **GELU는 음수 영역에서도 값이 있다** ($x=-0.5$에서 약 $-0.154$)
- ReLU는 음수를 완전히 0으로 만들지만, GELU는 약간의 음수 값을 허용
- 이로 인해 음수 영역에서도 기울기가 흐를 수 있어 학습이 더 원활

#### 역전파 유도

$\mathrm{GELU}(x) = 0.5x(1 + \tanh(u))$에서 $u = \sqrt{2/\pi}(x + 0.044715x^3)$

곱의 미분법 적용:

$$\frac{d}{dx}\mathrm{GELU}(x) = \underbrace{0.5(1 + \tanh(u))}_{\text{첫째 항의 미분}} + \underbrace{0.5x \cdot \mathrm{sech}^2(u) \cdot \frac{du}{dx}}_{\text{둘째 항의 미분}}$$

여기서 $\mathrm{sech}^2(u) = 1 - \tanh^2(u)$ ($\tanh$의 미분).

$u$의 미분:

$$\frac{du}{dx} = \sqrt{\frac{2}{\pi}}\!\left(1 + 3 \cdot 0.044715 \cdot x^2\right) = \sqrt{\frac{2}{\pi}}\!\left(1 + 0.134145 x^2\right)$$

종합:

$$g_x = g_y \cdot \left[0.5(1 + \tanh(u)) + 0.5x(1 - \tanh^2(u))\sqrt{\frac{2}{\pi}}(1 + 0.134145x^2)\right]$$

원소별 연산이므로 구현이 직관적:

```rust
fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
    let gx = Zip::from(&gy).and(&x).map_collect(|&gy_v, &x_v| {
        let u = sqrt_2_pi * (x_v + 0.044715 * x_v.powi(3));
        let tanh_u = u.tanh();
        let sech2_u = 1.0 - tanh_u * tanh_u;
        let du_dx = sqrt_2_pi * (1.0 + 0.134145 * x_v * x_v);
        gy_v * (0.5 * (1.0 + tanh_u) + 0.5 * x_v * sech2_u * du_dx)
    });
}
```

수치 미분과 일치 확인됨 (오차 < $10^{-4}$).

#### 왜 Transformer에서 GELU를 쓰는가

Transformer의 FFN(Feed-Forward Network)은:

$$\mathrm{FFN}(x) = \mathrm{GELU}(x W_1 + b_1) W_2 + b_2$$

여기서 $W_1$은 보통 hidden size를 4배로 확장 ($D \to 4D$), $W_2$는 다시 축소 ($4D \to D$).

GELU가 ReLU보다 선호되는 이유:
1. **부드러운 전환**: 기울기가 연속적이어서 학습이 더 안정적
2. **음수 허용**: dead neuron 문제 완화
3. **실험적 성능**: GPT, BERT 등에서 ReLU보다 일관되게 더 나은 성능 보고

---

### Transformer 블록에서의 위치

#### Pre-LN vs Post-LN

원래 Transformer (Vaswani et al., 2017)는 **Post-LN** — Attention/FFN 뒤에 정규화:

```
x → Attention → + x → LayerNorm → FFN → + → LayerNorm → out
```

GPT-2부터 사용하는 **Pre-LN** — Attention/FFN 앞에 정규화:

```
x → LayerNorm → Attention → + x → LayerNorm → FFN → + x → out
```

Pre-LN이 선호되는 이유: Post-LN에서는 residual 경로를 타고 온 값과 정규화된 값이 더해진 후에야 정규화가 적용된다. 깊은 네트워크에서 이 누적이 학습 불안정을 일으킨다. Pre-LN은 각 서브레이어 입력이 항상 정규화된 상태이므로 학습이 안정적이고, warmup 없이도 수렴한다.

#### Residual Connection (잔차 연결)

```
x ─────────────────────────────┐
↓                              │
LayerNorm → Attention/FFN → +──┘→ out = x + F(x)
```

$\mathrm{out} = x + F(x)$에서 역전파:

$$\frac{\partial \mathrm{out}}{\partial x} = I + \frac{\partial F}{\partial x}$$

항등 행렬 $I$가 있으므로, $\frac{\partial F}{\partial x}$가 아무리 작아도 기울기가 **최소 1**로 보장된다. 이것이 100개 이상의 층을 쌓아도 기울기가 소실되지 않는 핵심 메커니즘이다 (ResNet, He et al. 2015에서 최초 도입).

LSTM의 forget gate와 비교하면: LSTM은 $c_t = f \odot c_{t-1} + \ldots$ (곱셈적 조절), Residual은 $y = x + F(x)$ (항상 덧셈). Residual이 더 직접적으로 기울기를 보존한다.

#### FFN 확장 비율: 왜 4배인가

$$\mathrm{FFN}(x) = \mathrm{GELU}(x W_1 + b_1) W_2 + b_2$$

$W_1 \in \mathbb{R}^{D \times 4D}$, $W_2 \in \mathbb{R}^{4D \times D}$로 hidden dimension을 4배 확장했다 축소한다.

Attention이 **토큰 간 정보 교환**(mixing)을 담당한다면, FFN은 **각 토큰 내에서의 비선형 변환**(processing)을 담당한다. 4배 확장은 이 변환의 용량을 키우는 것이다.

왜 하필 4배인가? Transformer 원 논문에서 $D = 512$, $D_{\mathrm{ff}} = 2048$으로 설정한 것이 관례가 되었다. 이후 연구에서 2배~8배까지 실험되었으나, 4배가 성능-효율의 좋은 균형점으로 확인됨. GPT-2/3, BERT 모두 4배를 사용한다.

이 블록을 $N$번 반복하면 GPT의 본체가 된다.

---

### nanoGPT 로드맵

```
[x] Embedding, AdamW          ← step 61
[x] Transpose(axes)           ← step 62
[x] Batched Matmul            ← step 62
[x] Softmax(axis)             ← step 63
[x] Causal Mask               ← step 63
[x] LayerNorm                 ← step 64 (이번)
[x] GELU                      ← step 64 (이번)
[ ] Multi-Head Attention
[ ] GPT Block 통합
```

다음은 **Multi-Head Attention** — 지금까지 만든 모든 부품을 조합하여 Attention 모듈을 완성.
