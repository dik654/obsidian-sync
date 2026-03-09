# Step 76: GAN (Generative Adversarial Network)

> Phase 5 (생성모델)의 두 번째 스텝. Goodfellow et al. (2014)가 제안한 GAN은 **Generator**와 **Discriminator**의 적대적 학습을 통해 데이터를 생성하는 모델이다. VAE가 변분 추론으로 데이터의 확률 분포를 명시적으로 근사했다면, GAN은 **암묵적 생성 모델**(implicit generative model)로 분포를 직접 모델링하지 않고도 분포에서 샘플을 생성한다.

---

## 1. Minimax 게임

### 1.1 두 플레이어

GAN은 두 신경망의 **경쟁**으로 정의된다:

- **Generator** $G$: 랜덤 노이즈 $\mathbf{z} \sim p(\mathbf{z})$를 받아 가짜 데이터 $G(\mathbf{z})$를 생성. 목표: Discriminator를 속여서 가짜를 진짜로 판별하게 만들기.
- **Discriminator** $D$: 데이터 $\mathbf{x}$를 받아 "진짜일 확률" $D(\mathbf{x}) \in (0, 1)$을 출력. 목표: 진짜 데이터와 가짜 데이터를 정확히 구분하기.

### 1.2 목적 함수

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$$

이것은 **zero-sum 게임**이다:
- $D$는 $V$를 **최대화** — 진짜에 높은 확률, 가짜에 낮은 확률을 할당
- $G$는 $V$를 **최소화** — $D(G(\mathbf{z}))$를 높여서 가짜가 진짜로 판별되게

### 1.3 각 항의 의미

$D$의 관점에서 $V$를 최대화:
- $\mathbb{E}[\log D(\mathbf{x})]$: 진짜 데이터에 $D(\mathbf{x}) \to 1$을 출력하면 $\log 1 = 0$ (최대)
- $\mathbb{E}[\log(1-D(G(\mathbf{z})))]$: 가짜 데이터에 $D(G(\mathbf{z})) \to 0$을 출력하면 $\log 1 = 0$ (최대)

$D$가 완벽할 때 $V$의 상한:

$$V_{\max} = \mathbb{E}[\log 1] + \mathbb{E}[\log 1] = 0$$

$D$가 랜덤($D = 0.5$)일 때:

$$V = \log 0.5 + \log 0.5 = -2\log 2 \approx -1.386$$

### 1.4 위조지폐범 비유

| 역할 | GAN | 목표 |
|------|-----|------|
| 위조지폐범 | Generator | 진짜처럼 보이는 가짜 지폐 제작 |
| 경찰 | Discriminator | 진짜와 가짜 지폐 구분 |

위조지폐범이 실력을 키우면 경찰도 더 정교해져야 하고, 경찰이 정교해지면 위조지폐범도 더 나아져야 한다 — 이 경쟁적 과정이 양쪽 모두를 발전시킨다.

### 1.5 암묵적 생성 모델

GAN은 $p_g(\mathbf{x})$의 밀도 함수를 명시적으로 정의하지 않는다. 대신 $G$가 $\mathbf{z} \sim p(\mathbf{z})$를 변환하여 **암묵적으로** $p_g$를 정의한다:

$$p_g(\mathbf{x}) = p(\mathbf{z}) \left|\det \frac{\partial G^{-1}}{\partial \mathbf{x}}\right| \quad (\text{$G$가 가역일 때})$$

일반적으로 $G$는 가역이 아니므로 $p_g$를 직접 계산할 수 없다. 그러나 $p_g$에서 **샘플링**은 가능하다: $\mathbf{z} \sim p(\mathbf{z})$를 뽑고 $G(\mathbf{z})$를 계산하면 된다.

이것이 GAN과 VAE/Flow의 근본적 차이다:
- **VAE**: 명시적 밀도 $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$ (하한 계산 가능)
- **Flow**: 정확한 밀도 $p_\theta(\mathbf{x}) = p(\mathbf{z}) |\det J|^{-1}$ (change of variables)
- **GAN**: 밀도 함수 없음. 샘플링만 가능.

---

## 2. 최적 Discriminator

### 2.1 고정된 $G$에서 최적 $D^*$ 유도

$G$가 고정되었을 때, $V(D, G)$를 $D$에 대해 최대화한다. 기댓값을 적분으로 바꾸면:

$$V(D, G) = \int \left[ p_{\mathrm{data}}(\mathbf{x}) \log D(\mathbf{x}) + p_g(\mathbf{x}) \log(1 - D(\mathbf{x})) \right] d\mathbf{x}$$

여기서 $p_g$는 $G$가 정의하는 생성 분포: $\mathbf{x} = G(\mathbf{z})$, $\mathbf{z} \sim p(\mathbf{z})$.

각 점 $\mathbf{x}$에서 피적분함수를 독립적으로 최대화할 수 있다. $D(\mathbf{x}) = y$로 놓으면:

$$f(y) = a \log y + b \log(1-y), \quad a = p_{\mathrm{data}}(\mathbf{x}), \; b = p_g(\mathbf{x})$$

**1차 조건**:

$$f'(y) = \frac{a}{y} - \frac{b}{1-y} = \frac{a(1-y) - by}{y(1-y)} = 0$$

$$a(1-y) = by \implies a - ay = by \implies y = \frac{a}{a+b}$$

**2차 조건** (오목함수 확인):

$$f''(y) = -\frac{a}{y^2} - \frac{b}{(1-y)^2} < 0 \quad \forall y \in (0, 1)$$

$f$가 순오목(strictly concave)이므로 1차 조건의 해가 유일한 **전역 최대점**이다.

$$\boxed{D^*_G(\mathbf{x}) = \frac{p_{\mathrm{data}}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x}) + p_g(\mathbf{x})}}$$

### 2.2 최적 $D^*$의 해석

$D^*_G(\mathbf{x})$는 **Bayes 최적 분류기**이다. 진짜($C = 1$)와 가짜($C = 0$)를 동일 확률로 뽑은 혼합 데이터에서:

$$P(C=1|\mathbf{x}) = \frac{P(\mathbf{x}|C=1) P(C=1)}{P(\mathbf{x})} = \frac{p_{\mathrm{data}}(\mathbf{x}) \cdot \frac{1}{2}}{p_{\mathrm{data}}(\mathbf{x}) \cdot \frac{1}{2} + p_g(\mathbf{x}) \cdot \frac{1}{2}} = \frac{p_{\mathrm{data}}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x}) + p_g(\mathbf{x})} = D^*_G(\mathbf{x})$$

즉, 최적 Discriminator는 **likelihood ratio**에 기반한 Bayes 결정 규칙이다.

### 2.3 평형에서의 최적 $D^*$

$p_g = p_{\mathrm{data}}$이면:

$$D^*(\mathbf{x}) = \frac{p_{\mathrm{data}}(\mathbf{x})}{2 p_{\mathrm{data}}(\mathbf{x})} = \frac{1}{2}$$

평형에서 Discriminator는 **동전 던지기** 수준이 된다 — 진짜와 가짜를 전혀 구분하지 못한다.

---

## 3. 전역 최적해와 Jensen-Shannon Divergence

### 3.1 $D^*_G$를 $V$에 대입 — 단계별 유도

$D^*_G$를 $V(D, G)$에 대입하면:

$$C(G) = V(D^*_G, G) = \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}}\!\left[\log \frac{p_{\mathrm{data}}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x}) + p_g(\mathbf{x})}\right] + \mathbb{E}_{\mathbf{x} \sim p_g}\!\left[\log \frac{p_g(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x}) + p_g(\mathbf{x})}\right]$$

**Step 1**: 분모에 $\frac{1}{2}$를 곱하고 나누어 KL divergence 형태를 만든다.

$$\log \frac{p_{\mathrm{data}}}{p_{\mathrm{data}} + p_g} = \log \frac{p_{\mathrm{data}}}{\frac{p_{\mathrm{data}} + p_g}{2}} - \log 2$$

이유: $\frac{p_{\mathrm{data}}}{p_{\mathrm{data}} + p_g} = \frac{p_{\mathrm{data}}}{\frac{p_{\mathrm{data}} + p_g}{2}} \cdot \frac{1}{2}$이고 $\log\frac{1}{2} = -\log 2$.

마찬가지로:

$$\log \frac{p_g}{p_{\mathrm{data}} + p_g} = \log \frac{p_g}{\frac{p_{\mathrm{data}} + p_g}{2}} - \log 2$$

**Step 2**: $C(G)$에 대입한다.

$$C(G) = \mathbb{E}_{p_{\mathrm{data}}}\!\left[\log \frac{p_{\mathrm{data}}}{\frac{p_{\mathrm{data}} + p_g}{2}} - \log 2\right] + \mathbb{E}_{p_g}\!\left[\log \frac{p_g}{\frac{p_{\mathrm{data}} + p_g}{2}} - \log 2\right]$$

$$= \mathbb{E}_{p_{\mathrm{data}}}\!\left[\log \frac{p_{\mathrm{data}}}{\frac{p_{\mathrm{data}} + p_g}{2}}\right] + \mathbb{E}_{p_g}\!\left[\log \frac{p_g}{\frac{p_{\mathrm{data}} + p_g}{2}}\right] - 2\log 2$$

**Step 3**: $M = \frac{p_{\mathrm{data}} + p_g}{2}$로 놓으면, 각 기댓값은 KL divergence의 정의와 정확히 일치한다:

$$\mathbb{E}_{p_{\mathrm{data}}}\!\left[\log \frac{p_{\mathrm{data}}}{M}\right] = D_{\mathrm{KL}}(p_{\mathrm{data}} \| M)$$

$$\mathbb{E}_{p_g}\!\left[\log \frac{p_g}{M}\right] = D_{\mathrm{KL}}(p_g \| M)$$

**Step 4**: JSD의 정의에 의해:

$$D_{\mathrm{JS}}(p \| q) = \frac{1}{2} D_{\mathrm{KL}}(p \| M) + \frac{1}{2} D_{\mathrm{KL}}(q \| M), \quad M = \frac{p+q}{2}$$

따라서:

$$D_{\mathrm{KL}}(p_{\mathrm{data}} \| M) + D_{\mathrm{KL}}(p_g \| M) = 2 D_{\mathrm{JS}}(p_{\mathrm{data}} \| p_g)$$

**결론**:

$$\boxed{C(G) = -\log 4 + 2 \cdot D_{\mathrm{JS}}(p_{\mathrm{data}} \| p_g)}$$

### 3.2 전역 최솟값

$D_{\mathrm{JS}} \geq 0$이고 등호는 $p_{\mathrm{data}} = p_g$일 때만 성립하므로:

$$C(G) \geq -\log 4$$

등호 조건: $p_g = p_{\mathrm{data}}$ (Generator가 진짜 데이터 분포를 완벽히 학습).

이 때 $V(D^*, G^*) = -\log 4 = -2\ln 2 \approx -1.386$.

### 3.3 D loss의 이론적 평형값

구현에서 사용하는 D loss는 $-V(D, G)$ (부호 반전, 최소화 대상):

$$D_{\mathrm{loss}} = -V(D, G) = -\mathbb{E}[\log D(\mathbf{x})] - \mathbb{E}[\log(1 - D(G(\mathbf{z})))]$$

평형에서 $D^* = \frac{1}{2}$이므로:

$$D_{\mathrm{loss}}^* = -\log\frac{1}{2} - \log\frac{1}{2} = 2\log 2 = \log 4 \approx 1.386$$

```rust
// test_training_convergence 결과
epoch   1: d_loss = 1.3604  // ← 초기부터 ~1.386 근처
epoch 100: d_loss = 1.3901  // ← 이론적 평형 log(4)≈1.386에 수렴
```

### 3.4 Jensen-Shannon Divergence의 성질

#### 정의

$$D_{\mathrm{JS}}(p \| q) = \frac{1}{2} D_{\mathrm{KL}}\!\left(p \Big\| \frac{p+q}{2}\right) + \frac{1}{2} D_{\mathrm{KL}}\!\left(q \Big\| \frac{p+q}{2}\right)$$

#### 대칭성 증명

$M = \frac{p+q}{2}$로 놓으면, $D_{\mathrm{JS}}(p \| q) = \frac{1}{2}D_{\mathrm{KL}}(p \| M) + \frac{1}{2}D_{\mathrm{KL}}(q \| M)$.

$p \leftrightarrow q$를 교환하면 $M' = \frac{q+p}{2} = M$ (불변). 따라서:

$$D_{\mathrm{JS}}(q \| p) = \frac{1}{2}D_{\mathrm{KL}}(q \| M) + \frac{1}{2}D_{\mathrm{KL}}(p \| M) = D_{\mathrm{JS}}(p \| q) \quad \blacksquare$$

KL divergence와 달리 JSD는 **대칭**이다. 이는 GAN의 목적함수가 $p_{\mathrm{data}}$와 $p_g$를 대등하게 다루는 것과 일치한다.

#### 유계성 증명: $0 \leq D_{\mathrm{JS}} \leq \log 2$

**하한**: $D_{\mathrm{KL}} \geq 0$ (Gibbs 부등식)이므로 $D_{\mathrm{JS}} \geq 0$. 등호는 $p = q$일 때.

**상한**: $M = \frac{p+q}{2} \geq \frac{p}{2}$이므로:

$$D_{\mathrm{KL}}(p \| M) = \int p \log \frac{p}{M} \leq \int p \log \frac{p}{p/2} = \int p \log 2 = \log 2$$

마찬가지로 $D_{\mathrm{KL}}(q \| M) \leq \log 2$. 따라서:

$$D_{\mathrm{JS}} = \frac{1}{2}D_{\mathrm{KL}}(p \| M) + \frac{1}{2}D_{\mathrm{KL}}(q \| M) \leq \frac{1}{2}\log 2 + \frac{1}{2}\log 2 = \log 2 \quad \blacksquare$$

등호는 $p$와 $q$의 support가 완전히 분리될 때 성립한다.

#### $\sqrt{D_{\mathrm{JS}}}$는 metric

Endres & Schindelin (2003)의 결과: $\sqrt{D_{\mathrm{JS}}}$는 확률분포 공간 위의 **metric**(거리 함수)이다.

즉, 삼각 부등식 $\sqrt{D_{\mathrm{JS}}(p \| r)} \leq \sqrt{D_{\mathrm{JS}}(p \| q)} + \sqrt{D_{\mathrm{JS}}(q \| r)}$을 만족한다.

KL divergence는 삼각 부등식을 만족하지 않는다 — JSD가 더 "거리"다운 성질을 가진다.

#### KLD vs JSD 비교

| 성질 | $D_{\mathrm{KL}}(p \| q)$ | $D_{\mathrm{JS}}(p \| q)$ |
|------|--------------------------|--------------------------|
| 대칭성 | 비대칭 | **대칭** |
| 범위 | $[0, +\infty)$ | $[0, \log 2]$ |
| $p$와 $q$ 겹치지 않을 때 | $+\infty$ | $\log 2$ (유한) |
| 삼각 부등식 | 만족 안 함 | $\sqrt{D_{\mathrm{JS}}}$가 만족 |
| 볼록성 | $(p,q)$에 대해 볼록 | $(p,q)$에 대해 볼록 |

---

## 4. f-divergence 프레임워크

### 4.1 GAN을 f-divergence로 일반화

GAN의 minimax 목적함수는 **Jensen-Shannon divergence**를 최소화한다. 이를 일반화하면, 다양한 **f-divergence**를 최소화하는 GAN 변형을 설계할 수 있다 (Nowozin et al., 2016, f-GAN).

f-divergence의 정의:

$$D_f(p \| q) = \int q(\mathbf{x}) \, f\!\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}\right) d\mathbf{x}$$

여기서 $f$는 볼록 함수이고 $f(1) = 0$.

### 4.2 특수 케이스

| f-divergence | $f(t)$ | GAN 변형 |
|-------------|--------|---------|
| KL divergence | $t \log t$ | — |
| Reverse KL | $-\log t$ | — |
| **JSD** | $-(1+t)\log\frac{1+t}{2} + t\log t$ | **원래 GAN** |
| Pearson $\chi^2$ | $(t-1)^2$ | Least Squares GAN |
| Total Variation | $\frac{1}{2}|t - 1|$ | — |

### 4.3 Fenchel Conjugate를 통한 변분 하한

f-divergence를 직접 계산할 수 없으므로, **Fenchel conjugate** $f^*$를 이용한 변분 하한을 사용한다:

$$D_f(p \| q) \geq \sup_T \left[\mathbb{E}_p[T(\mathbf{x})] - \mathbb{E}_q[f^*(T(\mathbf{x}))]\right]$$

여기서 $T: \mathcal{X} \to \mathbb{R}$은 임의의 함수 (= Discriminator). $T$를 신경망으로 매개변수화하면 f-GAN이 된다.

JSD의 경우: $f^*(t) = -\log(2 - e^t)$이고, $T = \log D$로 놓으면 원래 GAN 목적함수가 복원된다.

---

## 5. Non-Saturating Loss

### 5.1 원래 G loss의 문제

원래 목적 함수에서 G의 loss:

$$\mathcal{L}_G^{\mathrm{minimax}} = \mathbb{E}_{\mathbf{z}}[\log(1 - D(G(\mathbf{z})))]$$

학습 초기에 $G$가 나쁜 상태면 $D(G(\mathbf{z})) \approx 0$이므로:

$$\log(1 - D(G(\mathbf{z}))) \approx \log 1 = 0$$

Gradient도 거의 0: $\frac{d}{dy}\log(1-y)\big|_{y=0} = -1$ (절대값이 작다).

즉, **G가 가장 개선이 필요한 순간에 gradient가 가장 약하다** — saturation 문제.

### 5.2 Non-Saturating 대안

Goodfellow (2014)가 제안한 실용적 대안:

$$\mathcal{L}_G^{\mathrm{NS}} = -\mathbb{E}_{\mathbf{z}}[\log D(G(\mathbf{z}))]$$

$D(G(\mathbf{z})) \approx 0$이면 $-\log(D(G(\mathbf{z}))) \to +\infty$ — gradient가 **강하다**.

### 5.3 Gradient 비교 — 정량적 분석

$y = D(G(\mathbf{z}))$로 놓고 두 G loss의 gradient를 비교:

**Minimax**: $\mathcal{L}_G = \log(1-y)$

$$\frac{\partial \mathcal{L}_G}{\partial y} = -\frac{1}{1-y}$$

**Non-saturating**: $\mathcal{L}_G = -\log y$

$$\frac{\partial \mathcal{L}_G}{\partial y} = -\frac{1}{y}$$

| $y = D(G(\mathbf{z}))$ | minimax $\left|-\frac{1}{1-y}\right|$ | NS $\left|-\frac{1}{y}\right|$ | 비율 NS/minimax |
|-------------------------|-------|------|------|
| 0.01 (G 나쁨) | 1.01 | **100** | ×99 |
| 0.1 | 1.11 | **10** | ×9 |
| 0.5 (평형) | 2.0 | 2.0 | ×1 |
| 0.9 (G 좋음) | **10** | 1.11 | ×0.1 |

핵심: 학습 초기($y \approx 0$)에 NS는 minimax보다 **100배** 강한 gradient를 제공하고, 학습 후기($y \approx 1$)에는 반대로 약해진다 — 학습 시작을 효과적으로 부트스트랩한다.

### 5.4 Non-Saturating Loss가 최소화하는 divergence

Non-saturating G loss:

$$\mathcal{L}_G^{\mathrm{NS}} = -\mathbb{E}_{\mathbf{z}}[\log D^*(G(\mathbf{z}))]$$

$D^* = \frac{p_{\mathrm{data}}}{p_{\mathrm{data}} + p_g}$를 대입하면:

$$= -\mathbb{E}_{p_g}\!\left[\log \frac{p_{\mathrm{data}}}{p_{\mathrm{data}} + p_g}\right]$$

$$= \mathbb{E}_{p_g}\!\left[\log \frac{p_{\mathrm{data}} + p_g}{p_{\mathrm{data}}}\right] = \mathbb{E}_{p_g}\!\left[\log\!\left(1 + \frac{p_g}{p_{\mathrm{data}}}\right)\right]$$

이것은 KLD도 JSD도 아닌, 다른 divergence에 해당한다. 정확히는:

$$\mathcal{L}_G^{\mathrm{NS}} = D_{\mathrm{KL}}(p_g \| p_{\mathrm{data}}) - 2 D_{\mathrm{JS}}(p_{\mathrm{data}} \| p_g) + \log 4$$

이 관계에서 NS loss의 최솟값도 $p_g = p_{\mathrm{data}}$에서 달성된다 — 같은 평형점을 공유한다.

### 5.5 BCE로의 표현

Non-saturating G loss = BCE(D(G(z)), 1):

$$\mathcal{L}_G^{\mathrm{NS}} = -\mathbb{E}[\log D(G(\mathbf{z}))] = \mathrm{BCE}(D(G(\mathbf{z})), 1)$$

"가짜를 진짜라고 주장" — Discriminator가 가짜를 1(진짜)로 판별하도록 유도.

---

## 6. Binary Cross-Entropy

### 6.1 정의

$$\mathrm{BCE}(\mathbf{p}, \mathbf{t}) = -\frac{1}{N}\sum_{i=1}^N [t_i \log p_i + (1-t_i)\log(1-p_i)]$$

- $p_i \in (0, 1)$: 예측 확률 (sigmoid 출력)
- $t_i \in \{0, 1\}$: 타겟 라벨

### 6.2 확률적 해석 — MLE 유도

$D(\mathbf{x})$를 베르누이 분포의 파라미터로 모델링:

$$p(\text{real} | \mathbf{x}) = D(\mathbf{x})^{t} (1 - D(\mathbf{x}))^{1-t}$$

$N$개 관측의 log-likelihood:

$$\log L = \sum_{i=1}^N [t_i \log D(\mathbf{x}_i) + (1-t_i) \log(1 - D(\mathbf{x}_i))]$$

이를 최대화하는 것은 $-\frac{1}{N}\log L = \mathrm{BCE}$를 **최소화**하는 것과 동치. 따라서 **BCE 최소화 = MLE**이다.

> **BCE와 KL divergence의 관계**: $\mathrm{BCE}(p, t) = H(t, p) = H(t) + D_{\mathrm{KL}}(t \| p)$. $H(t)$는 타겟의 엔트로피(상수)이므로, BCE 최소화 $\iff$ KL 최소화.

### 6.3 D loss를 BCE로 표현

$$\mathcal{L}_D = \mathrm{BCE}(D(\mathbf{x}_{\mathrm{real}}), 1) + \mathrm{BCE}(D(G(\mathbf{z})), 0)$$

전개하면:

$$= -\frac{1}{N}\sum_i \log D(\mathbf{x}_i) - \frac{1}{M}\sum_j \log(1 - D(G(\mathbf{z}_j)))$$

이것은 원래 minimax 목적함수의 $-V(D, G)$와 동치 (상수 계수 차이).

### 6.4 수치 안정성: Clamp

$\log(0) = -\infty$이므로, sigmoid 출력이 극단적일 때 문제가 생길 수 있다. Clamp로 해결:

$$\tilde{p} = \mathrm{clamp}(p, \epsilon, 1-\epsilon), \quad \epsilon = 10^{-7}$$

> **대안: logits 기반 BCE**. 더 안정적인 방법은 sigmoid 적용 전의 **logits** $l$에 대해 직접 계산하는 것이다:
> $$\mathrm{BCE}_{\mathrm{logits}}(l, t) = \max(l, 0) - l \cdot t + \log(1 + e^{-|l|})$$
> 이 공식은 log-sum-exp trick을 사용하여 overflow/underflow를 방지한다. 본 구현에서는 clamp 방식을 사용했으나, 실전 프레임워크에서는 logits 기반이 표준이다.

### 6.5 구현

```rust
pub fn binary_cross_entropy(p: &Variable, t: &Variable) -> Variable {
    let eps = 1e-7;
    let p_clamped = clamp(p, eps, 1.0 - eps);
    let n = p.len() as f64;
    let term1 = t * &log(&p_clamped);
    let term2 = &(1.0 - t) * &log(&(1.0 - &p_clamped));
    &sum(&(&term1 + &term2)) * (-1.0 / n)
}
```

기존 프리미티브(log, sum, clamp)를 조합한 합성 함수. autograd가 backward를 자동 처리한다.

### 6.6 BCE의 Gradient — 유도

$$\mathrm{BCE} = -\frac{1}{N}\sum_i [t_i \log p_i + (1-t_i)\log(1-p_i)]$$

$p_i$에 대해 편미분:

$$\frac{\partial \mathrm{BCE}}{\partial p_i} = -\frac{1}{N}\left(\frac{t_i}{p_i} + (1-t_i) \cdot \frac{-1}{1-p_i}\right) = -\frac{1}{N}\left(\frac{t_i}{p_i} - \frac{1-t_i}{1-p_i}\right)$$

통분하면:

$$= -\frac{1}{N} \cdot \frac{t_i(1-p_i) - (1-t_i)p_i}{p_i(1-p_i)} = -\frac{1}{N} \cdot \frac{t_i - p_i}{p_i(1-p_i)}$$

| $t_i$ | $p_i$ | gradient | 의미 |
|-------|-------|----------|------|
| 1 | 0.2 (낮음) | $-\frac{1}{N}\cdot\frac{0.8}{0.16} = -\frac{5}{N}$ | 강한 증가 압력 |
| 1 | 0.9 (높음) | $-\frac{1}{N}\cdot\frac{0.1}{0.09} \approx -\frac{1.1}{N}$ | 약한 증가 압력 |
| 0 | 0.8 (높음) | $-\frac{1}{N}\cdot\frac{-0.8}{0.16} = \frac{5}{N}$ | 강한 감소 압력 |
| 0 | 0.1 (낮음) | $-\frac{1}{N}\cdot\frac{-0.1}{0.09} \approx \frac{1.1}{N}$ | 약한 감소 압력 |

예측이 타겟에서 멀수록 gradient가 커진다 — **자연스러운 adaptive learning rate**.

---

## 7. GAN 아키텍처

### 7.1 구조

```
[Generator]
z ∈ ℝ^{latent_dim} ~ N(0,I)
    ↓ Linear(latent_dim → g_hidden) + tanh
    ↓ Linear(g_hidden → g_hidden) + tanh
    ↓ Linear(g_hidden → data_dim) + sigmoid
    ↓
x_fake ∈ (0,1)^{data_dim}

[Discriminator]
x ∈ ℝ^{data_dim}
    ↓ Linear(data_dim → d_hidden) + tanh
    ↓ Linear(d_hidden → d_hidden) + tanh
    ↓ Linear(d_hidden → 1) + sigmoid
    ↓
prob ∈ (0,1)    ← "진짜일 확률"
```

### 7.2 설계 결정

| 결정 | 선택 | 이유 |
|------|------|------|
| G/D 은닉 활성화 | tanh | sigmoid는 vanishing gradient, tanh가 GAN 표준 |
| G 출력 활성화 | sigmoid | [0,1] 데이터, BCE와 호환 |
| D 출력 활성화 | sigmoid | 확률 출력 → BCE와 호환 |
| G loss | non-saturating | 학습 초기 gradient 문제 해결 |
| 학습 패턴 | 별도 optimizer | cleargrads로 G/D 분리 업데이트 |
| RNG | LCG + Box-Muller | 프로젝트 관례 |

### 7.3 왜 tanh인가?

sigmoid vs tanh 비교:
- **Sigmoid**: $\sigma(x) \in (0, 1)$, 출력이 항상 양수 → gradient가 한 방향으로만 흐름 (zig-zagging)
- **tanh**: $\tanh(x) \in (-1, 1)$, 출력이 **0 중심** → gradient가 양방향으로 균형 있게 흐름

또한 $\tanh(x) = 2\sigma(2x) - 1$이므로 sigmoid의 rescaled 버전이지만, 0 중심화가 학습 동역학을 크게 개선한다.

실전 GAN에서는 LeakyReLU가 더 선호되지만, 본 구현에서는 기존 활성화 함수만 사용.

### 7.4 구현

```rust
pub struct GAN {
    g_l1: Linear, g_l2: Linear, g_out: Linear,   // Generator
    d_l1: Linear, d_l2: Linear, d_out: Linear,   // Discriminator
    pub latent_dim: usize,
    pub data_dim: usize,
    rng_state: Cell<u64>,
}
```

6개 Linear × (W + b) = **12개 학습 파라미터** (G 6개 + D 6개).

---

## 8. 학습 알고리즘

### 8.1 교대 학습 (Alternating Training)

```
for epoch in epochs:
    for batch in data:
        // --- Step 1: D 학습 (G 고정) ---
        d_real = D(x_real)
        fake = G(z)            // z ~ N(0,I)
        d_fake = D(fake)
        d_loss = bce(d_real, 1) + bce(d_fake, 0)
        cleargrads_all()
        d_loss.backward()
        d_optimizer.update(D_params)   // D만 업데이트

        // --- Step 2: G 학습 (D 고정) ---
        fake2 = G(z2)          // 새로운 noise
        d_fake2 = D(fake2)
        g_loss = bce(d_fake2, 1)
        cleargrads_all()
        g_loss.backward()
        g_optimizer.update(G_params)   // G만 업데이트
```

### 8.2 "고정"의 구현 — Optimizer 분리

파라미터 freezing이 필요 없다. 핵심: **어떤 파라미터를 업데이트하는가**만 제어하면 된다.

D 학습 시:
- `d_loss.backward()`는 D와 G 모두에 gradient를 계산
- `d_optimizer.update(&gan.discriminator_params())`는 **D의 파라미터만 업데이트**
- G의 gradient는 계산되지만 **사용되지 않음**

G 학습 시:
- `g_loss.backward()`는 G → D → G 전체 경로의 gradient를 계산
- 중요: gradient가 **D를 통과하여 G까지 흘러야** 한다
- `g_optimizer.update(&gan.generator_params())`는 **G의 파라미터만 업데이트**

### 8.3 Gradient 흐름: G 학습 시

```
z → G_l1 → tanh → G_l2 → tanh → G_out → sigmoid → x_fake
                                                       ↓
                                           D_l1 → tanh → D_l2 → tanh → D_out → sigmoid → prob
                                                                                            ↓
                                                                              g_loss = bce(prob, 1)
```

`g_loss.backward()` 호출 시 chain rule:

$$\frac{\partial \mathcal{L}_G}{\partial \mathbf{W}_G} = \underbrace{\frac{\partial \mathcal{L}_G}{\partial \mathrm{prob}}}_{\text{BCE grad}} \cdot \underbrace{\frac{\partial \mathrm{prob}}{\partial \mathbf{x}_{\mathrm{fake}}}}_{\text{D의 입력에 대한 grad}} \cdot \underbrace{\frac{\partial \mathbf{x}_{\mathrm{fake}}}{\partial \mathbf{W}_G}}_{\text{G의 파라미터에 대한 grad}}$$

핵심 중간 항 $\frac{\partial \mathrm{prob}}{\partial \mathbf{x}_{\mathrm{fake}}}$: D는 "어떤 방향으로 바꾸면 더 진짜처럼 보이는지"를 G에게 알려주는 역할을 한다.

```rust
// test_gradient_flow_generator에서 검증
g_loss.backward(false, false);
let g_params = gan.generator_params();
let has_grads = g_params.iter().filter(|p| p.grad().is_some()).count();
// G params with gradient: 6/6 ✓
```

### 8.4 학습 결과

```rust
// test_training_convergence 결과
epoch   1: d_loss = 1.3604, g_loss = 0.7119
epoch  21: d_loss = 1.3595, g_loss = 0.7413
epoch  41: d_loss = 1.3774, g_loss = 0.7443
epoch  61: d_loss = 1.3762, g_loss = 0.7391
epoch  81: d_loss = 1.3886, g_loss = 0.7002
epoch 100: d_loss = 1.3901, g_loss = 0.7093
```

D loss ≈ 1.39 ≈ $\log 4 \approx 1.386$ — 이론적 Nash 평형에 근접.
G loss ≈ 0.71 ≈ $-\log(0.5) = \log 2 \approx 0.693$ — $D(G(\mathbf{z})) \approx 0.5$에 대응.

---

## 9. GAN의 수렴 이론

### 9.1 Nash 균형

GAN의 학습은 **Nash 균형**(Nash Equilibrium)을 찾는 문제다:

$$(G^*, D^*)$$가 Nash 균형 $\iff$ $G^*$가 $D^*$에 대한 최적 응답 AND $D^*$가 $G^*$에 대한 최적 응답

Goodfellow (2014)의 정리: $G$가 충분한 표현력을 가지면, 유일한 Nash 균형은:

$$p_g = p_{\mathrm{data}}, \quad D^*(\mathbf{x}) = \frac{1}{2} \; \forall \mathbf{x}$$

### 9.2 수렴 보장의 한계

이론적 결과는 **함수 공간에서의 최적화**를 가정한다. 실제로는:

1. G와 D가 신경망이므로 **유한한 표현력** — 진짜 $p_{\mathrm{data}}$를 완벽히 표현 불가
2. SGD 기반 교대 최적화는 Nash 균형 수렴을 **보장하지 않음**
3. 함수 공간에서 $V$는 $D$에 대해 볼록, $G$에 대해 오목이지만 (볼록-오목), 신경망 파라미터화 후에는 **비볼록-비오목**

### 9.3 학습 동역학 — 진동 문제

$G$와 $D$의 파라미터를 $\theta$와 $\phi$로 놓으면, 교대 경사하강법:

$$\phi_{t+1} = \phi_t + \eta \nabla_\phi V(\phi_t, \theta_t)$$
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta V(\phi_{t+1}, \theta_t)$$

이 동역학은 볼록-오목 게임에서조차 **진동**할 수 있다.

**구체적 예시**: $V(\phi, \theta) = \phi \cdot \theta$ (가장 간단한 bilinear game)

Gradient: $\nabla_\phi V = \theta$, $\nabla_\theta V = \phi$.

동시 gradient descent/ascent의 연속 시간 동역학:

$$\dot{\phi} = \theta, \qquad \dot{\theta} = -\phi$$

이 시스템의 야코비안:

$$J = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$$

**고유값**: $\det(J - \lambda I) = \lambda^2 + 1 = 0 \implies \lambda = \pm i$.

고유값이 **순허수** — 원점 주위를 **등속 회전**하며, 수렴하지도 발산하지도 않는다.

해를 구하면: $\phi(t) = r\cos(t + \alpha)$, $\theta(t) = r\sin(t + \alpha)$. 초기 조건에 따른 반경 $r$의 원을 그린다.

이산화(gradient descent)하면 상황이 더 나빠진다. 학습률 $\eta$를 사용한 이산 업데이트:

$$\begin{pmatrix} \phi_{t+1} \\ \theta_{t+1} \end{pmatrix} = \begin{pmatrix} 1 & \eta \\ -\eta & 1 \end{pmatrix} \begin{pmatrix} \phi_t \\ \theta_t \end{pmatrix}$$

이 행렬의 고유값: $1 \pm i\eta$, 절대값 $\sqrt{1 + \eta^2} > 1$.

고유값의 절대값이 1보다 크므로 — **발산**한다! 이산화가 진동을 **발산으로 악화**시킨다.

이것이 GAN 학습이 근본적으로 어려운 이유다. 최적화 문제가 아니라 **게임**이기 때문.

---

## 10. GAN 학습의 문제와 해결

### 10.1 Mode Collapse

**현상**: Generator가 데이터 분포의 **일부 모드**만 생성하는 현상. 예: MNIST에서 숫자 "1"만 생성.

**수학적 원인**: Minimax GAN은 이론적으로 $\min_G \max_D V$를 풀지만, 실전에서는 $\max_D$와 $\min_G$를 **교대로** 수행한다. 이 순서가 결과를 바꾼다:

- 이론: $\min_G \max_D V(D, G)$ — G가 $D$의 **최악 응답**을 견뎌야 하므로 모든 모드를 커버해야
- 실전: $G$는 **현재 $D$**를 속이면 충분 — $D$가 약한 모드에만 집중하면 됨

$D$를 충분히 학습시키지 않으면 (내부 loop이 불완전하면), $G$는 단일 모드 "속이기 쉬운" 출력만 생성하게 된다.

**Cycling 해석**: 시간 $t$에서 $G$가 모드 $A$를 잘 생성 → $D$가 모드 $A$를 학습하여 구분 → $G$가 모드 $B$로 전환 → $D$가 모드 $B$ 학습 → ... → $G$가 다시 모드 $A$로 회귀. 이 **순환**(cycling)이 mode collapse의 동역학적 구조다.

**해결 방법**:

| 방법 | 설명 | 핵심 아이디어 |
|------|------|------------|
| Minibatch discrimination | D가 배치 내 다양성을 평가 | $G$가 같은 출력만 내면 D가 탐지 |
| Unrolled GAN | D의 미래 $k$-step을 예상하여 G 학습 | $G$가 "지금 D"가 아닌 "미래 D"를 속여야 함 |
| Feature matching | D의 중간 특징 통계를 맞추도록 G 학습 | 단일 출력이 아닌 통계적 다양성 유도 |
| WGAN | Wasserstein distance 사용 | 이론적으로 mode collapse 발생 안 함 |

### 10.2 학습 불안정성

**현상**: D loss → 0 (D가 너무 강해짐) 또는 loss가 진동.

**원인**: D가 G보다 훨씬 빨리 학습하면, $D(\mathbf{x}_{\mathrm{fake}}) \approx 0$이 되어 G의 gradient가 vanish (minimax) 또는 explode (non-saturating).

**해결 방법**:

| 방법 | 설명 | 수식 |
|------|------|------|
| Label smoothing | 진짜 라벨을 1 대신 0.9 사용 | D가 "절대적 확신"을 못하게 → 과적합 방지 |
| Instance noise | 입력에 노이즈 추가 | support 겹침을 인위적으로 만들어 gradient 유지 |
| Spectral normalization | D의 Lipschitz 상수 제한 (§10.4) | 안정적 gradient |
| Gradient penalty | D의 gradient 크기 제한 (§10.5) | WGAN-GP |
| Learning rate 차이 | D에 작은 lr, G에 큰 lr | D와 G의 학습 속도 균형 |

### 10.3 Vanishing Gradient (JSD의 한계)

$p_{\mathrm{data}}$와 $p_g$의 support가 겹치지 않으면 (고차원에서 거의 항상 그러함):

$$D_{\mathrm{JS}}(p_{\mathrm{data}} \| p_g) = \log 2 = \mathrm{const}$$

**증명**: $\mathrm{supp}(p_{\mathrm{data}}) \cap \mathrm{supp}(p_g) = \emptyset$이면, 임의의 $\mathbf{x}$에서 $p_{\mathrm{data}}(\mathbf{x}) \cdot p_g(\mathbf{x}) = 0$.

$M = \frac{p_{\mathrm{data}} + p_g}{2}$에 대해:

- $p_{\mathrm{data}}(\mathbf{x}) > 0$인 점에서: $M(\mathbf{x}) = \frac{p_{\mathrm{data}}(\mathbf{x})}{2}$이므로 $\frac{p_{\mathrm{data}}(\mathbf{x})}{M(\mathbf{x})} = 2$
- $p_g(\mathbf{x}) > 0$인 점에서: $M(\mathbf{x}) = \frac{p_g(\mathbf{x})}{2}$이므로 $\frac{p_g(\mathbf{x})}{M(\mathbf{x})} = 2$

따라서:

$$D_{\mathrm{KL}}(p_{\mathrm{data}} \| M) = \int p_{\mathrm{data}} \log \frac{p_{\mathrm{data}}}{M} = \int p_{\mathrm{data}} \log 2 = \log 2$$

마찬가지로 $D_{\mathrm{KL}}(p_g \| M) = \log 2$.

$$D_{\mathrm{JS}} = \frac{1}{2}\log 2 + \frac{1}{2}\log 2 = \log 2 \quad \blacksquare$$

$D_{\mathrm{JS}}$가 상수이면 $\nabla_\theta D_{\mathrm{JS}} = 0$ — **G가 학습할 수 없다**.

**직관**: 진짜 데이터 매니폴드와 가짜 데이터 매니폴드가 고차원 공간에서 "스쳐가기만" 하면, 완벽한 $D$가 그 사이에 날카로운 결정 경계를 놓을 수 있다. 이 때 JSD는 최대값 $\log 2$에 고정되어 gradient 정보를 제공하지 못한다.

**왜 고차원에서 거의 항상 발생하는가?** 두 저차원 매니폴드 (실제 데이터와 생성 데이터)가 고차원 공간 $\mathbb{R}^D$에 임베딩되면, 교차할 확률은 $D$가 커질수록 0에 수렴한다 (measure zero). MNIST에서도 784차원 공간의 두 ~20차원 매니폴드가 겹칠 확률은 사실상 0이다.

### 10.4 Spectral Normalization

Miyato et al. (2018)의 방법: D의 각 레이어의 **스펙트럼 노름**(최대 특이값)으로 가중치를 정규화:

$$\bar{\mathbf{W}} = \frac{\mathbf{W}}{\sigma(\mathbf{W})}, \quad \sigma(\mathbf{W}) = \max_{\mathbf{h}: \|\mathbf{h}\| \leq 1} \|\mathbf{W}\mathbf{h}\|$$

각 레이어가 Lipschitz 상수 1을 가지면, D 전체의 Lipschitz 상수는 레이어 수의 곱으로 제한된다.

실전에서 $\sigma(\mathbf{W})$는 **power iteration** 1~2회로 근사한다:

$$\mathbf{v} \leftarrow \frac{\mathbf{W}^\top \mathbf{u}}{\|\mathbf{W}^\top \mathbf{u}\|}, \quad \mathbf{u} \leftarrow \frac{\mathbf{W}\mathbf{v}}{\|\mathbf{W}\mathbf{v}\|}, \quad \sigma(\mathbf{W}) \approx \mathbf{u}^\top \mathbf{W} \mathbf{v}$$

### 10.5 WGAN과 Wasserstein Distance

#### Wasserstein-1 Distance (Earth Mover's Distance)

$$W_1(p_{\mathrm{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\mathrm{data}}, p_g)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma}[\|\mathbf{x} - \mathbf{y}\|]$$

**직관**: $p_{\mathrm{data}}$를 "흙더미", $p_g$를 "구멍"으로 보면, 흙을 옮겨 구멍을 메우는 **최소 비용**이 $W_1$이다. 각 단위 흙을 거리 $\|\mathbf{x} - \mathbf{y}\|$만큼 옮기는 비용.

#### 1D 예시: $W_1$과 JSD의 차이

$p = \delta_0$ (점 질량, $x = 0$), $q = \delta_\theta$ ($x = \theta$)로 놓자.

- $W_1(p, q) = |\theta|$ — $\theta$에 대해 **연속, 미분 가능** ($\frac{\partial W_1}{\partial \theta} = \mathrm{sign}(\theta)$)
- $D_{\mathrm{JS}}(p \| q) = \begin{cases} \log 2 & \theta \neq 0 \\ 0 & \theta = 0 \end{cases}$ — $\theta \neq 0$에서 **상수**, gradient 0

$W_1$은 "얼마나 멀리 떨어져 있는지"에 대한 연속적 정보를 제공한다. JSD는 "겹치는가, 안 겹치는가"만 알려주는 이진 정보.

#### Kantorovich-Rubinstein Duality

직접 $\inf_\gamma$를 구하는 것은 계산적으로 불가능하므로, 쌍대 형태를 사용:

$$W_1(p, q) = \sup_{\|f\|_L \leq 1} \left[\mathbb{E}_{p}[f(\mathbf{x})] - \mathbb{E}_{q}[f(\mathbf{x})]\right]$$

여기서 $\|f\|_L \leq 1$은 $f$가 **1-Lipschitz** 함수라는 제약: $|f(\mathbf{x}) - f(\mathbf{y})| \leq \|\mathbf{x} - \mathbf{y}\|$ $\forall \mathbf{x}, \mathbf{y}$.

**유도 스케치** (강쌍대정리):

원시 문제: $\inf_\gamma \int c(\mathbf{x}, \mathbf{y}) d\gamma(\mathbf{x}, \mathbf{y})$ subject to 주변 분포 제약 $\gamma_X = p$, $\gamma_Y = q$.

라그랑지안: $L(\gamma, f, g) = \int c \, d\gamma + \int f(d\gamma_X - dp) + \int g(d\gamma_Y - dq)$

$c(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$이면 쌍대 변수 제약이 $f(\mathbf{x}) + g(\mathbf{y}) \leq c(\mathbf{x}, \mathbf{y})$가 되고, 최적에서 $g = -f$, $f$는 1-Lipschitz가 된다.

#### WGAN의 학습

1-Lipschitz 제약을 강제하는 방법:

| 방법 | 설명 | 문제점 |
|------|------|--------|
| **Weight clipping** (WGAN) | $\mathbf{W} \leftarrow \mathrm{clamp}(\mathbf{W}, -c, c)$ | capacity 감소, 최적 critic이 단순 함수에 제한 |
| **Gradient penalty** (WGAN-GP) | $\lambda \mathbb{E}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\| - 1)^2]$ | 더 안정적, 표준 방법 |

WGAN-GP에서 $\hat{\mathbf{x}} = \alpha \mathbf{x}_{\mathrm{real}} + (1-\alpha)\mathbf{x}_{\mathrm{fake}}$, $\alpha \sim U(0,1)$은 진짜와 가짜 사이의 보간점이다. 이 점에서 $D$의 gradient 노름이 1이 되도록 강제한다.

---

## 11. GAN Loss 함수

### 11.1 Loss 변형 총정리

| 이름 | D loss | G loss | 최소화하는 divergence |
|------|--------|--------|-------------------|
| **Minimax** | $-\mathbb{E}[\log D(\mathbf{x})] - \mathbb{E}[\log(1-D(G(\mathbf{z})))]$ | $\mathbb{E}[\log(1-D(G(\mathbf{z})))]$ | JSD |
| **Non-saturating** | 동일 | $-\mathbb{E}[\log D(G(\mathbf{z}))]$ | (다른 divergence) |
| **LSGAN** | $\mathbb{E}[(D(\mathbf{x})-1)^2] + \mathbb{E}[D(G(\mathbf{z}))^2]$ | $\mathbb{E}[(D(G(\mathbf{z}))-1)^2]$ | Pearson $\chi^2$ |
| **WGAN** | $\mathbb{E}[D(G(\mathbf{z}))] - \mathbb{E}[D(\mathbf{x})]$ | $-\mathbb{E}[D(G(\mathbf{z}))]$ | $W_1$ |
| **Hinge** | $\mathbb{E}[\max(0, 1-D(\mathbf{x}))] + \mathbb{E}[\max(0, 1+D(G(\mathbf{z})))]$ | $-\mathbb{E}[D(G(\mathbf{z}))]$ | — |

### 11.2 gan_loss 구현

```rust
pub fn gan_loss(d_real: &Variable, d_fake: &Variable) -> (Variable, Variable) {
    let batch_real = d_real.shape()[0];
    let batch_fake = d_fake.shape()[0];

    let ones_real = Variable::new(ArrayD::ones(ndarray::IxDyn(&[batch_real, 1])));
    let zeros_fake = Variable::new(ArrayD::zeros(ndarray::IxDyn(&[batch_fake, 1])));
    let ones_fake = Variable::new(ArrayD::ones(ndarray::IxDyn(&[batch_fake, 1])));

    let d_loss_real = binary_cross_entropy(d_real, &ones_real);
    let d_loss_fake = binary_cross_entropy(d_fake, &zeros_fake);
    let d_loss = &d_loss_real + &d_loss_fake;

    let g_loss = binary_cross_entropy(d_fake, &ones_fake);

    (d_loss, g_loss)
}
```

### 11.3 Loss 분해 검증

```rust
// test_d_loss_decomposition 결과
d_loss = 1.350945, manual = 1.350945, diff = 0.00e0  ✓
```

---

## 12. Clamp 함수

### 12.1 정의

$$\mathrm{clamp}(x, a, b) = \begin{cases} a & \text{if } x \leq a \\ x & \text{if } a < x < b \\ b & \text{if } x \geq b \end{cases}$$

### 12.2 Backward

$$\frac{\partial \mathrm{clamp}}{\partial x} = \begin{cases} 1 & \text{if } a < x < b \\ 0 & \text{otherwise} \end{cases}$$

이것은 **Straight-Through Estimator**의 특수 케이스다. 범위 밖에서 gradient를 0으로 차단하여 수치적 불안정을 방지한다.

### 12.3 구현

```rust
struct ClampFn { min_val: f64, max_val: f64 }

impl Function for ClampFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![xs[0].mapv(|v| v.clamp(self.min_val, self.max_val))]
    }
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let x = xs[0].data();
        let mask = x.mapv(|v| {
            if v > self.min_val && v < self.max_val { 1.0 } else { 0.0 }
        });
        vec![&gys[0] * &Variable::new(mask)]
    }
}
```

---

## 13. GAN 평가 지표

### 13.1 Inception Score (IS)

$$\mathrm{IS} = \exp\!\left(\mathbb{E}_{\mathbf{x} \sim p_g}\left[D_{\mathrm{KL}}(p(y|\mathbf{x}) \| p(y))\right]\right)$$

- $p(y|\mathbf{x})$: 분류기(Inception Net)의 조건부 분포. 좋은 생성이면 **sharp** (특정 클래스에 집중)
- $p(y) = \mathbb{E}_{p_g}[p(y|\mathbf{x})]$: 주변 분포. 다양한 생성이면 **uniform**
- KL이 크면 → IS가 높다 → 좋은 생성

**한계**: 모드 내 품질만 평가하고, $p_{\mathrm{data}}$와의 비교가 없다. 모든 클래스를 고품질로 생성하지만 클래스 비율이 틀려도 높은 점수.

### 13.2 Fréchet Inception Distance (FID)

$$\mathrm{FID} = \|\boldsymbol{\mu}_r - \boldsymbol{\mu}_g\|^2 + \mathrm{tr}\!\left(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g)^{1/2}\right)$$

- $(\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r)$: 실제 데이터의 Inception 특징의 평균/공분산
- $(\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g)$: 생성 데이터의 Inception 특징의 평균/공분산
- **낮을수록 좋다** (0 = 완벽)

FID는 두 가우시안 사이의 **Fréchet distance** (2-Wasserstein distance)이다:

$$W_2^2(\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1), \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)) = \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|^2 + \mathrm{tr}(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2 - 2(\boldsymbol{\Sigma}_1^{1/2}\boldsymbol{\Sigma}_2 \boldsymbol{\Sigma}_1^{1/2})^{1/2})$$

$\boldsymbol{\Sigma}_1$, $\boldsymbol{\Sigma}_2$가 대각이면 더 간단해진다.

IS보다 FID가 선호되는 이유: (1) 실제 데이터와 비교, (2) mode collapse 감지 가능, (3) 인간 평가와 높은 상관.

---

## 14. VAE vs GAN

### 14.1 수학적 비교

| | VAE | GAN |
|--|-----|-----|
| 목적함수 | ELBO 최대화 | Minimax game |
| 분포 근사 방식 | KL divergence | Jensen-Shannon divergence |
| 밀도 추정 | 가능 (ELBO) | 불가능 (implicit model) |
| 학습 안정성 | 안정적 | 불안정 (mode collapse, vanishing grad) |
| 샘플 품질 | 흐릿 (평균 효과) | 선명 (adversarial pressure) |
| 잠재 공간 | 구조화 (KL regularization) | 비구조화 |

### 14.2 흐릿함 vs 선명함의 근본 원인

- **VAE**: $\min \mathbb{E}[\|\mathbf{x} - \hat{\mathbf{x}}\|^2]$ → 다중 모드의 **평균** 출력 → 흐릿
- **GAN**: D가 흐릿한 이미지에 "가짜" 판정 → G가 **특정 모드** 선택 → 선명하지만 다양성 부족 가능

수학적으로: VAE의 디코더는 $\mathbb{E}_{p(\mathbf{x}|\mathbf{z})}[\mathbf{x}]$를 출력 — 이는 조건부 기댓값이므로 여러 가능한 $\mathbf{x}$의 **가중 평균**이다. GAN의 generator는 결정적 함수 $G(\mathbf{z})$로 특정 한 점을 출력한다.

### 14.3 KL vs JSD의 mode-seeking vs mode-covering

- **$D_{\mathrm{KL}}(q \| p)$ (forward KL)**: $p(\mathbf{x}) > 0$인 곳에서 $q(\mathbf{x}) > 0$이어야 함 → **mode-covering** → 모든 모드를 커버하되 흐릿
- **$D_{\mathrm{KL}}(p \| q)$ (reverse KL)**: $q(\mathbf{x}) > 0$인 곳에서 $p(\mathbf{x}) > 0$이어야 함 → **mode-seeking** → 일부 모드만 선명하게
- **$D_{\mathrm{JS}}$**: 대칭, 중간 성질

VAE는 forward KL을 근사적으로 최소화하므로 mode-covering (흐릿), GAN은 JSD를 최소화하므로 중간이지만 실전에서는 mode-seeking 경향 (mode collapse).

### 14.4 수렴성 비교

- **VAE**: ELBO가 단조 증가하는 것을 관찰 가능. Loss가 줄면 모델이 개선되었다는 의미.
- **GAN**: D loss와 G loss가 동시에 줄어들면 안 됨. 이론적으로는 D loss → $\log 4$, G loss → $\log 2$에 수렴해야 하지만, 실전에서는 진동이 흔하다. Loss 값만으로 생성 품질을 판단할 수 없다.

---

## 15. GAN의 확장

### 15.1 DCGAN (Radford et al., 2016)

Conv/Deconv 레이어를 사용하여 이미지 생성에 특화. 안정적 학습을 위한 아키텍처 가이드라인:

| 규칙 | 이유 |
|------|------|
| Pooling → Strided Conv | 학습 가능한 다운샘플링 |
| FC 제거 (global average pooling) | 파라미터 감소 |
| Batch Normalization (G, D 모두) | 학습 안정화 |
| G: ReLU (은닉) + Tanh (출력) | 더 나은 gradient 흐름 |
| D: LeakyReLU (전체) | 죽은 뉴런 방지 |
| Adam ($\beta_1 = 0.5$, $\beta_2 = 0.999$) | 기본값 $\beta_1 = 0.9$보다 안정 |

### 15.2 Conditional GAN (cGAN, Mirza & Osadchy, 2014)

조건 변수 $\mathbf{c}$ (예: 클래스 라벨)를 G와 D에 추가:

$$\min_G \max_D \; \mathbb{E}[\log D(\mathbf{x}, \mathbf{c})] + \mathbb{E}[\log(1 - D(G(\mathbf{z}, \mathbf{c}), \mathbf{c}))]$$

응용: 텍스트 → 이미지, 클래스 조건부 생성, 이미지 변환 (pix2pix).

### 15.3 주요 GAN 계보

| 모델 | 핵심 아이디어 | 년도 |
|------|------------|------|
| **GAN** | 최초의 적대적 학습 | 2014 |
| **cGAN** | 조건부 생성 | 2014 |
| **DCGAN** | CNN 기반 아키텍처 가이드라인 | 2016 |
| **WGAN** | Wasserstein distance, weight clipping | 2017 |
| **WGAN-GP** | Gradient penalty로 Lipschitz 제약 | 2017 |
| **SNGAN** | Spectral normalization | 2018 |
| **SAGAN** | Self-attention + SN | 2018 |
| **BigGAN** | 대규모 학습, class-conditional | 2019 |
| **StyleGAN** | 스타일 기반 Generator, AdaIN | 2019 |
| **StyleGAN2** | Weight demodulation, 경로 정규화 | 2020 |
| **StyleGAN3** | Alias-free generation | 2021 |

---

## 16. 테스트 요약

| # | 테스트 | 검증 내용 |
|---|--------|----------|
| 1 | `test_gan_construction` | 생성, G params 6개, D params 6개, total 12개 |
| 2 | `test_generator_output` | generate(10) shape (10, data_dim), sigmoid ∈ (0,1) |
| 3 | `test_discriminator_output` | discriminate(x) shape (B, 1), sigmoid ∈ (0,1) |
| 4 | `test_bce_known_values` | 수동 계산과 비교 (단일, 배치) |
| 5 | `test_bce_gradient` | backward 후 grad 존재, shape 정확 |
| 6 | `test_d_loss_decomposition` | d_loss = bce(real,1) + bce(fake,0) |
| 7 | `test_g_loss_computation` | g_loss > 0, finite |
| 8 | `test_gradient_flow_discriminator` | D backward 후 D params 6/6에 gradient |
| 9 | `test_gradient_flow_generator` | G backward (D 통과) 후 G params 6/6에 gradient |
| 10 | `test_training_convergence` | 100 epoch, d_loss ≈ 1.39 ≈ log(4), bounded |

전체 10/10 통과.

---

## 17. 다음 단계

Step 77에서는 **DDPM** (Denoising Diffusion Probabilistic Model)을 구현할 예정이다. GAN이 적대적 학습으로 한 번에 데이터를 생성했다면, Diffusion 모델은 **점진적 노이즈 제거** 과정으로 데이터를 생성한다.

$$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \xrightarrow{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)} \cdots \xrightarrow{} \mathbf{x}_0 \sim p_{\mathrm{data}}$$
