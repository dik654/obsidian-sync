# Step 77: DDPM (Denoising Diffusion Probabilistic Models)

> Phase 5 (생성모델)의 세 번째 스텝. Ho et al. (2020)이 제안한 DDPM은 데이터에 **점진적으로 noise를 추가**하는 forward process와, 학습된 신경망이 **noise를 역방향으로 제거**하는 reverse process로 구성된 생성 모델이다. VAE가 잠재 변수의 사후 분포를 근사하고, GAN이 적대적 학습으로 생성했다면, DDPM은 **마르코프 체인의 역과정을 학습**하여 순수 noise에서 데이터를 생성한다.

---

## 1. 핵심 아이디어

### 1.1 확산과 역확산

물리적 확산(diffusion)에서 영감을 받은 모델이다:

- **Forward (확산)**: 잉크 한 방울을 물에 떨어뜨리면 점점 퍼져서 결국 균일한 분포가 된다
- **Reverse (역확산)**: 그 과정을 영상으로 되감으면, 균일한 분포에서 잉크 방울이 다시 모인다

DDPM은 이 "되감기"를 신경망으로 학습한다:

$$\underbrace{\mathbf{x}_0}_{\text{데이터}} \xrightarrow{\text{noise 추가}} \mathbf{x}_1 \xrightarrow{} \cdots \xrightarrow{} \underbrace{\mathbf{x}_T}_{\text{순수 noise}} \xrightarrow{\text{학습된 제거}} \mathbf{x}_{T-1} \xrightarrow{} \cdots \xrightarrow{} \hat{\mathbf{x}}_0$$

### 1.2 왜 "점진적"인가

한 번에 모든 noise를 추가/제거하는 것이 아니라, $T$개의 작은 스텝으로 나눈다. 각 스텝에서 추가되는 noise가 아주 작으면, **역과정도 가우시안으로 근사 가능**하다는 것이 핵심 통찰이다 (Feller, 1949):

> 충분히 작은 스텝의 확산 과정의 역과정은 원래 과정과 같은 함수형(가우시안)을 갖는다.

이 성질 덕분에 역과정의 각 스텝을 가우시안 $\mathcal{N}(\boldsymbol{\mu}_\theta, \sigma_t^2 \mathbf{I})$로 매개변수화할 수 있다.

### 1.3 세 가지 생성 모델 비교

| | VAE | GAN | DDPM |
|--|-----|-----|------|
| 생성 방식 | 잠재 변수 → 디코더 | noise → Generator | noise → 반복 denoising |
| 학습 목표 | ELBO 최대화 | minimax game | noise 예측 MSE |
| 밀도 추정 | 하한 계산 가능 | 불가 | 하한 계산 가능 |
| 샘플 품질 | 흐릿함 | 선명하지만 불안정 | 선명하고 안정적 |
| 샘플링 속도 | 한 번 (fast) | 한 번 (fast) | $T$번 반복 (slow) |
| 모드 커버리지 | 높음 | 낮음 (mode collapse) | 높음 |

---

## 2. Forward Process (확산)

### 2.1 스텝별 정의

Forward process $q$는 데이터 $\mathbf{x}_0$에 점진적으로 가우시안 noise를 추가하는 **고정된** 마르코프 체인이다:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{1 - \beta_t}\, \mathbf{x}_{t-1},\; \beta_t \mathbf{I}\right)$$

여기서 $\beta_t \in (0, 1)$은 **noise schedule**로, 각 스텝에서 추가되는 noise의 양을 결정한다.

이를 sampling 관점에서 쓰면:

$$\mathbf{x}_t = \sqrt{1 - \beta_t}\, \mathbf{x}_{t-1} + \sqrt{\beta_t}\, \boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**해석**: $\mathbf{x}_{t-1}$을 $\sqrt{1-\beta_t}$만큼 **축소**(신호 감쇠)하고, $\sqrt{\beta_t}$만큼의 **noise를 추가**한다.

### 2.2 분산 보존 (Variance-Preserving)

$\mathbf{x}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$라고 가정하면:

$$\mathrm{Var}[\mathbf{x}_t] = (1 - \beta_t) \cdot \mathrm{Var}[\mathbf{x}_{t-1}] + \beta_t \cdot \mathrm{Var}[\boldsymbol{\epsilon}_t] = (1 - \beta_t) + \beta_t = 1$$

즉, 총 분산이 1로 보존된다. 이것이 **Variance-Preserving (VP)** SDE라 불리는 이유다.

### 2.3 Closed-Form $q(\mathbf{x}_t | \mathbf{x}_0)$ 유도

$T$번 순차적으로 noise를 추가하지 않고도, $\mathbf{x}_0$에서 **임의의 $t$로 직접** 점프할 수 있다. $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$로 정의하면:

**유도 (수학적 귀납법)**:

**Base case** ($t = 1$):

$$\mathbf{x}_1 = \sqrt{\alpha_1}\, \mathbf{x}_0 + \sqrt{1 - \alpha_1}\, \boldsymbol{\epsilon}_1$$

$$\mathbf{x}_1 \sim \mathcal{N}(\sqrt{\bar{\alpha}_1}\, \mathbf{x}_0,\; (1 - \bar{\alpha}_1) \mathbf{I}) \quad \checkmark \quad (\bar{\alpha}_1 = \alpha_1)$$

**Inductive step**: $\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\, \bar{\boldsymbol{\epsilon}}$이 성립한다고 가정하면:

$$\mathbf{x}_t = \sqrt{\alpha_t}\, \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\, \boldsymbol{\epsilon}_t$$

$$= \sqrt{\alpha_t}\!\left(\sqrt{\bar{\alpha}_{t-1}}\, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\, \bar{\boldsymbol{\epsilon}}\right) + \sqrt{1 - \alpha_t}\, \boldsymbol{\epsilon}_t$$

$$= \sqrt{\alpha_t \bar{\alpha}_{t-1}}\, \mathbf{x}_0 + \sqrt{\alpha_t(1 - \bar{\alpha}_{t-1})}\, \bar{\boldsymbol{\epsilon}} + \sqrt{1 - \alpha_t}\, \boldsymbol{\epsilon}_t$$

두 독립 가우시안의 합: $\mathcal{N}(0, \sigma_1^2) + \mathcal{N}(0, \sigma_2^2) = \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)$이므로:

$$\sigma^2 = \alpha_t(1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t) = \alpha_t - \alpha_t \bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \alpha_t \bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$$

따라서:

$$\boxed{q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0,\; (1 - \bar{\alpha}_t) \mathbf{I}\right)}$$

**샘플링**:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### 2.4 경계 조건

- $t = 0$: $\bar{\alpha}_0 = \alpha_0 \approx 1$이므로 $\mathbf{x}_0 \approx \mathbf{x}_0$ (noise 거의 없음)
- $t = T$: $\bar{\alpha}_T \approx 0$이므로 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (순수 noise)

이것은 "잉크가 완전히 퍼진" 상태다.

### 2.5 Signal-to-Noise Ratio (SNR)

$q(\mathbf{x}_t | \mathbf{x}_0)$에서 신호 대 잡음비를 정의할 수 있다:

$$\mathrm{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

- $t = 0$: $\mathrm{SNR} \gg 1$ (신호 우세)
- $t = T$: $\mathrm{SNR} \approx 0$ (noise 우세)

SNR은 단조 감소하며, noise schedule의 설계를 분석하는 데 유용하다. 로그 SNR:

$$\lambda_t = \log \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

는 $+\infty$에서 $-\infty$로 감소하는 것이 이상적이다.

---

## 3. Reverse Process (역확산)

### 3.1 목표

Forward process의 **역방향**을 학습한다:

$$p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$$

여기서 $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$이고, 각 역방향 스텝을 가우시안으로 매개변수화:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \mathbf{I}\right)$$

### 3.2 왜 가우시안인가

$\beta_t$가 충분히 작으면, $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$도 가우시안이 된다. 이는 **중심극한정리**의 연속 버전으로 이해할 수 있다: 충분히 작은 확산 스텝의 역과정은 forward와 같은 함수형을 갖는다.

따라서 reverse 스텝을 가우시안으로 매개변수화하는 것은 근사가 아니라 **이론적으로 정당**하다.

### 3.3 True Reverse Process

실제 역조건부 분포는:

$$q(\mathbf{x}_{t-1} | \mathbf{x}_t) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}) \, q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}$$

이것은 $q(\mathbf{x}_{t-1})$과 $q(\mathbf{x}_t)$를 알아야 하므로 직접 계산할 수 없다 (모든 데이터에 대한 적분이 필요). 그러나 $\mathbf{x}_0$까지 조건을 주면 계산 가능해진다 — 이것이 §4의 posterior다.

---

## 4. Tractable Posterior $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$

### 4.1 베이즈 정리 적용

$\mathbf{x}_0$를 알고 있을 때, reverse posterior를 정확히 계산할 수 있다:

$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) \, q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}$$

마르코프 성질에 의해 $q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t | \mathbf{x}_{t-1})$이므로:

$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}) \, q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}$$

### 4.2 세 가우시안의 결합

세 항은 모두 가우시안이다:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\, \mathbf{x}_{t-1},\; (1-\alpha_t) \mathbf{I})$$

$$q(\mathbf{x}_{t-1} | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}\, \mathbf{x}_0,\; (1-\bar{\alpha}_{t-1}) \mathbf{I})$$

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\, \mathbf{x}_0,\; (1-\bar{\alpha}_t) \mathbf{I})$$

가우시안의 비율도 가우시안이므로 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t \mathbf{I})$.

### 4.3 Posterior Variance $\tilde{\beta}_t$ 유도

가우시안 $\mathcal{N}(\boldsymbol{\mu}, \sigma^2 \mathbf{I})$의 로그는 $-\frac{1}{2\sigma^2}\|\mathbf{x} - \boldsymbol{\mu}\|^2$에 비례한다. $\mathbf{x}_{t-1}$에 대한 이차항의 계수를 모으면:

분자의 두 가우시안에서 $\mathbf{x}_{t-1}$의 이차항 계수:
- $q(\mathbf{x}_t | \mathbf{x}_{t-1})$에서: $-\frac{\alpha_t}{2(1-\alpha_t)} \|\mathbf{x}_{t-1}\|^2$
- $q(\mathbf{x}_{t-1} | \mathbf{x}_0)$에서: $-\frac{1}{2(1-\bar{\alpha}_{t-1})} \|\mathbf{x}_{t-1}\|^2$

정밀도(precision, 분산의 역수)를 더하면:

$$\frac{1}{\tilde{\beta}_t} = \frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} = \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t)}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}$$

분자:

$$\alpha_t - \alpha_t \bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \bar{\alpha}_t$$

따라서:

$$\frac{1}{\tilde{\beta}_t} = \frac{1 - \bar{\alpha}_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})} = \frac{1-\bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}$$

$$\boxed{\tilde{\beta}_t = \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}}$$

### 4.4 Posterior Mean $\tilde{\boldsymbol{\mu}}_t$ 유도

$\mathbf{x}_{t-1}$의 일차항을 모아 평균을 구한다. 정밀도-가중 평균(precision-weighted mean):

$$\tilde{\boldsymbol{\mu}}_t = \tilde{\beta}_t \left( \frac{\alpha_t}{1-\alpha_t} \cdot \frac{\mathbf{x}_t}{\sqrt{\alpha_t}} + \frac{1}{1-\bar{\alpha}_{t-1}} \cdot \sqrt{\bar{\alpha}_{t-1}}\, \mathbf{x}_0 \right)$$

$q(\mathbf{x}_t | \mathbf{x}_{t-1})$에서 $\mathbf{x}_{t-1}$의 일차항: $\frac{\alpha_t}{1-\alpha_t} \cdot \frac{\mathbf{x}_t}{\sqrt{\alpha_t}} = \frac{\sqrt{\alpha_t}}{1-\alpha_t} \mathbf{x}_t$

$q(\mathbf{x}_{t-1} | \mathbf{x}_0)$에서 $\mathbf{x}_{t-1}$의 일차항: $\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}} \mathbf{x}_0$

$$\tilde{\boldsymbol{\mu}}_t = \tilde{\beta}_t \left( \frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \right)$$

$\tilde{\beta}_t = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$를 대입하여 정리하면:

$$\boxed{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0}$$

### 4.5 $\mathbf{x}_0$를 $\boldsymbol{\epsilon}$으로 치환

$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\, \boldsymbol{\epsilon}$에서:

$$\mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\, \boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

§4.4에 대입하면 (대수적 정리는 아래):

$$\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$

**유도**:

$$\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t} \cdot \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

$\mathbf{x}_t$의 계수:

$$\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}_t}}$$

$\sqrt{\bar{\alpha}_t} = \sqrt{\alpha_t \bar{\alpha}_{t-1}}$이므로 $\frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}} = \frac{1}{\sqrt{\alpha_t}}$:

$$= \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1}) + \beta_t / \sqrt{\alpha_t}}{1-\bar{\alpha}_t} = \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\sqrt{\alpha_t}(1-\bar{\alpha}_t)}$$

분자: $\alpha_t - \bar{\alpha}_t + 1 - \alpha_t = 1 - \bar{\alpha}_t$

따라서 $\mathbf{x}_t$의 계수 $= \frac{1}{\sqrt{\alpha_t}}$.

$\boldsymbol{\epsilon}$의 계수:

$$-\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{(1-\bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} \cdot \sqrt{1-\bar{\alpha}_t} = -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}}$$

합치면:

$$\boxed{\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)}$$

이 공식이 DDPM의 핵심이다. $\boldsymbol{\epsilon}$을 신경망 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$로 예측하면 된다.

---

## 5. 변분 하한 (ELBO) 유도

### 5.1 Log-Likelihood 하한

데이터 로그 가능도의 변분 하한:

$$\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \right] = -L_{\mathrm{VLB}}$$

### 5.2 VLB 분해

$L_{\mathrm{VLB}}$를 분해하면 세 종류의 항이 나타난다:

$$L_{\mathrm{VLB}} = \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_T | \mathbf{x}_0) \| p(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^{T} \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t))}_{L_{t-1}} - \underbrace{\log p_\theta(\mathbf{x}_0 | \mathbf{x}_1)}_{L_0}$$

**유도**: joint를 전개한다.

$$\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} = \log p(\mathbf{x}_T) + \sum_{t=1}^T \log p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) - \sum_{t=1}^T \log q(\mathbf{x}_t | \mathbf{x}_{t-1})$$

$q$의 마르코프 체인을 $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \, q(\mathbf{x}_t|\mathbf{x}_0)}{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}$로 바꾸면 텔레스코핑 소거가 일어나 위 결과를 얻는다.

### 5.3 각 항의 의미

| 항 | 의미 | 계산 방법 |
|----|------|----------|
| $L_T$ | prior matching | $\bar{\alpha}_T \approx 0$이면 $\approx 0$ (고정, 학습 불필요) |
| $L_{t-1}$ | denoising matching | 두 가우시안의 KL → closed-form |
| $L_0$ | reconstruction | 디코더의 데이터 복원 능력 |

### 5.4 $L_{t-1}$의 Closed-Form

두 가우시안의 KL divergence:

$$D_{\mathrm{KL}}(\mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t \mathbf{I}) \| \mathcal{N}(\boldsymbol{\mu}_\theta, \sigma_t^2 \mathbf{I})) = \frac{1}{2\sigma_t^2} \|\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_\theta\|^2 + \frac{d}{2}\left(\frac{\tilde{\beta}_t}{\sigma_t^2} - 1 - \log \frac{\tilde{\beta}_t}{\sigma_t^2}\right)$$

$\sigma_t^2 = \tilde{\beta}_t$로 설정하면 (DDPM의 선택), 두 번째 항이 0이 되고:

$$L_{t-1} = \frac{1}{2\tilde{\beta}_t} \|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\|^2$$

---

## 6. ε-Prediction으로의 단순화

### 6.1 μ-Prediction에서 ε-Prediction으로

$\boldsymbol{\mu}_\theta$를 직접 예측하는 대신, §4.5의 공식을 활용한다. $\boldsymbol{\mu}_\theta$를 다음과 같이 매개변수화:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$

$\tilde{\boldsymbol{\mu}}_t$와 $\boldsymbol{\mu}_\theta$의 차이:

$$\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_\theta = \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \boldsymbol{\epsilon})$$

$L_{t-1}$에 대입:

$$L_{t-1} = \frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t (1-\bar{\alpha}_t)} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$$

### 6.2 가중치 단순화 (Ho et al.)

Ho et al. (2020)의 핵심 발견: 시간별 가중치 $\frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t (1-\bar{\alpha}_t)}$를 **모두 1로 치환**해도 (심지어 더 좋은) 성능을 보인다:

$$\boxed{L_{\mathrm{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]}$$

여기서 $t \sim \mathrm{Uniform}\{1, \ldots, T\}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\, \boldsymbol{\epsilon}$.

### 6.3 왜 가중치를 버려도 되는가

원래 가중치를 분석하면:

$$w_t = \frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t (1-\bar{\alpha}_t)}$$

$\tilde{\beta}_t = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$를 대입:

$$w_t = \frac{\beta_t}{2\alpha_t(1-\bar{\alpha}_{t-1})}$$

이 가중치는 $t$가 작을 때 (noise가 적을 때) 매우 크고, $t$가 클 때 작다. 직관적으로:

- 큰 $t$: noise가 많아 쉬운 문제 → 낮은 가중치가 적절
- 작은 $t$: noise가 적어 정밀한 예측 필요 → 높은 가중치

그러나 실험적으로 **균등 가중치**가 다운-웨이팅된 고주파 디테일을 더 잘 학습시켜, 샘플 품질이 향상된다. 이론적으로는 VLB에서 벗어나지만, **경험적으로 우월**하다.

### 6.4 세 가지 예측 대상 비교

| 예측 대상 | 네트워크 출력 | 손실 |
|-----------|-------------|------|
| $\boldsymbol{\epsilon}$-prediction | $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \approx \boldsymbol{\epsilon}$ | $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$ |
| $\mathbf{x}_0$-prediction | $\hat{\mathbf{x}}_0(\mathbf{x}_t, t) \approx \mathbf{x}_0$ | $\|\mathbf{x}_0 - \hat{\mathbf{x}}_0\|^2$ |
| $\mathbf{v}$-prediction | $\mathbf{v}_\theta \approx \sqrt{\bar{\alpha}_t}\boldsymbol{\epsilon} - \sqrt{1-\bar{\alpha}_t}\mathbf{x}_0$ | $\|\mathbf{v} - \mathbf{v}_\theta\|^2$ |

$\boldsymbol{\epsilon}$-prediction이 DDPM의 원본이고, $\mathbf{v}$-prediction은 Imagen과 Stable Diffusion에서 사용된다. 세 가지는 수학적으로 등가이지만, 학습 동역학이 다르다.

---

## 7. Noise Schedule

### 7.1 Linear Schedule (DDPM 원본)

$$\beta_t = \beta_{\mathrm{start}} + \frac{\beta_{\mathrm{end}} - \beta_{\mathrm{start}}}{T-1} \cdot (t-1), \quad t = 1, \ldots, T$$

Ho et al.의 기본값: $\beta_{\mathrm{start}} = 10^{-4}$, $\beta_{\mathrm{end}} = 0.02$, $T = 1000$.

```rust
// 구현 (DDPM::new 내부)
let betas: Vec<f64> = (0..timesteps)
    .map(|t| beta_start + (beta_end - beta_start) * t as f64 / (timesteps - 1) as f64)
    .collect();
```

### 7.2 Cosine Schedule (Nichol & Dhariwal, 2021)

Linear schedule의 문제: $t$가 클 때 $\bar{\alpha}_t$가 너무 빨리 0에 도달하여, 마지막 스텝들에서 정보가 완전히 파괴된다.

Cosine schedule은 $\bar{\alpha}_t$를 직접 정의:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

$s = 0.008$은 $t=0$에서 $\beta_t$가 너무 작아지는 것을 방지.

$\beta_t$는 역으로 계산:

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \quad \beta_t \in [\epsilon, 1-\epsilon]$$

Cosine schedule은 SNR이 보다 균일하게 감소하여, 모든 timestep에서 유용한 학습이 일어난다.

### 7.3 Schedule 비교

| | Linear | Cosine |
|--|--------|--------|
| $\bar{\alpha}_t$ 감소 패턴 | 초반 느림, 후반 급격 | 전 구간 균일 |
| SNR 분포 | 불균형 | 균형 |
| $t$가 큰 구간 | 정보 완전 파괴 | 일부 신호 보존 |
| 구현 난이도 | 단순 | $\bar{\alpha}_t$에서 역산 필요 |

---

## 8. Sinusoidal Time Embedding

### 8.1 왜 time conditioning이 필요한가

Denoiser $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$는 **같은 네트워크**가 모든 timestep $t$에서의 noise를 예측한다. 그러나 $t=1$에서의 noise 패턴과 $t=999$에서의 noise 패턴은 완전히 다르다:

- $t$ 작을 때: 미세한 noise, 고주파 디테일 복원 필요
- $t$ 클 때: 대규모 구조 결정, 저주파 패턴 생성 필요

따라서 네트워크는 $t$를 입력으로 받아 **어떤 레벨의 noise를 제거해야 하는지** 알아야 한다.

### 8.2 Transformer의 위치 인코딩 차용

DDPM은 Transformer (Vaswani et al., 2017)의 sinusoidal positional encoding을 시간 인코딩으로 사용:

$$\mathrm{PE}(t, 2k) = \sin\!\left(\frac{t}{10000^{2k/d}}\right), \quad \mathrm{PE}(t, 2k+1) = \cos\!\left(\frac{t}{10000^{2k/d}}\right)$$

여기서 $d$는 embedding 차원, $k = 0, 1, \ldots, d/2 - 1$.

**왜 작동하는가**:
- 각 차원은 다른 주파수($10000^{-2k/d}$)로 진동 — **다중 스케일 표현**
- $k$ 작음 → 고주파 → 인접 timestep 구분
- $k$ 큼 → 저주파 → 먼 timestep 간 유사성 포착
- 값이 항상 $[-1, 1]$ 범위 → 안정적인 입력

### 8.3 Fourier Feature 관점

Sinusoidal embedding은 **Random Fourier Features** (Rahimi & Recht, 2007)의 결정론적 변형이다. 스칼라 $t$를 고차원 공간에 매핑하여 **비선형 관계**를 선형 모델로도 포착 가능하게 만든다:

$$\gamma(t) = [\sin(\omega_0 t), \cos(\omega_0 t), \sin(\omega_1 t), \cos(\omega_1 t), \ldots]$$

주파수 $\omega_k = 10000^{-2k/d}$는 기하급수적으로 감소하여 넓은 주파수 대역을 커버한다.

```rust
// 구현 (DDPM::sinusoidal_embedding)
pub fn sinusoidal_embedding(&self, t: &[usize]) -> Variable {
    let batch = t.len();
    let half = self.t_emb_dim / 2;
    let mut data = vec![0.0f64; batch * self.t_emb_dim];
    for (i, &ti) in t.iter().enumerate() {
        for k in 0..half {
            let freq = 1.0 / (10000.0_f64).powf(k as f64 / half as f64);
            let angle = ti as f64 * freq;
            data[i * self.t_emb_dim + 2 * k] = angle.sin();
            data[i * self.t_emb_dim + 2 * k + 1] = angle.cos();
        }
    }
    Variable::new(ArrayD::from_shape_vec(IxDyn(&[batch, self.t_emb_dim]), data).unwrap())
}
```

### 8.4 Time Injection 방식

Embedding을 네트워크에 주입하는 방식:

| 방식 | 수식 | 장단점 |
|------|------|--------|
| Concatenation | $\mathbf{h} = f([\mathbf{x}; \mathbf{t}])$ | 단순, 차원 증가 |
| **Additive** | $\mathbf{h} = f(\mathbf{x}) + g(\mathbf{t})$ | 표준, 차원 보존 |
| FiLM | $\mathbf{h} = \gamma(\mathbf{t}) \odot f(\mathbf{x}) + \beta(\mathbf{t})$ | 표현력 높음, 복잡 |

DDPM의 표준은 **Additive injection** — time embedding을 MLP로 변환한 후 각 블록의 출력에 더한다.

---

## 9. Denoiser 아키텍처

### 9.1 U-Net (원본 DDPM)

원본 DDPM은 이미지용 **U-Net** 아키텍처를 사용:
- Encoder: 해상도를 줄이며 특징 추출
- Decoder: 해상도를 복원하며 noise 예측
- Skip connections: encoder와 decoder를 연결
- Self-attention: 중간 해상도에서 전역 정보 통합
- Time embedding: 각 ResBlock에 additive injection

### 9.2 MLP Denoiser (본 구현)

이미지가 아닌 저차원 데이터에는 MLP로 충분하다:

```
t → sinusoidal_embedding → GELU(t_mlp1) → t_mlp2 → t_hidden
x_t → x_proj ──→ (+t_hidden) → GELU
                  → block1 ──→ (+t_hidden) → GELU
                  → block2 ──→ (+t_hidden) → GELU
                  → out_proj → predicted ε
```

```rust
// 구현 (DDPM::forward)
pub fn forward(&self, x_t: &Variable, t: &[usize]) -> Variable {
    let t_emb = self.sinusoidal_embedding(t);
    let t_hidden = self.t_mlp2.forward(&gelu(&self.t_mlp1.forward(&t_emb)));

    let h = gelu(&(&self.x_proj.forward(x_t) + &t_hidden));
    let h = gelu(&(&self.block1.forward(&h) + &t_hidden));
    let h = gelu(&(&self.block2.forward(&h) + &t_hidden));
    self.out_proj.forward(&h)
}
```

### 9.3 왜 GELU인가

| 활성화 | 수식 | DDPM에서의 적합성 |
|--------|------|-------------------|
| ReLU | $\max(0, x)$ | 0에서 미분 불연속, gradient 흐름 단절 |
| tanh | $\tanh(x)$ | 포화 영역에서 vanishing gradient |
| **GELU** | $x \cdot \Phi(x)$ | 부드러운 게이팅, Transformer 표준 |

GELU는 입력의 크기에 따라 **부드럽게 게이팅**하여, noise 예측에 적합한 정밀한 출력을 생성한다.

---

## 10. Training Algorithm

### 10.1 알고리즘 (Pseudo-code)

```
repeat:
    x_0 ~ p_data                           # 데이터 샘플
    t ~ Uniform{1, ..., T}                 # 랜덤 timestep
    ε ~ N(0, I)                            # 랜덤 noise
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε      # forward diffusion
    loss = ||ε - ε_θ(x_t, t)||²           # noise 예측 MSE
    θ ← θ - η∇_θ loss                     # gradient descent
until converged
```

### 10.2 핵심 관찰

1. **Random timestep**: 매 iteration마다 $t$를 랜덤으로 뽑아 **모든 noise 레벨을 균등하게 학습**
2. **Closed-form noising**: $q(\mathbf{x}_t | \mathbf{x}_0)$의 closed-form 덕분에, $T$번 순차 noise 추가 없이 **한 번에** $\mathbf{x}_t$ 생성
3. **Self-supervised**: 라벨 불필요 — noise $\boldsymbol{\epsilon}$이 자동으로 타겟
4. **단일 네트워크**: 같은 $\boldsymbol{\epsilon}_\theta$가 모든 $t$에서 동작 (time conditioning으로 구분)

### 10.3 구현

```rust
pub fn ddpm_loss(ddpm: &DDPM, x_0: &Variable, t: &[usize]) -> Variable {
    let (x_t, noise) = ddpm.q_sample(x_0, t);
    let predicted = ddpm.forward(&x_t, t);
    mean_squared_error(&predicted, &noise)
}
```

학습 루프:

```rust
for epoch in 0..200 {
    let t = ddpm.sample_timesteps(batch_size);    // uniform random
    let loss = ddpm_loss(&ddpm, &x_batch, &t);
    ddpm.cleargrads();
    loss.backward(false, false);
    optimizer.update(&ddpm.params());
}
```

---

## 11. Sampling Algorithm

### 11.1 알고리즘 (Pseudo-code)

```
x_T ~ N(0, I)                                    # 순수 noise에서 시작
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1, else z = 0
    ε_pred = ε_θ(x_t, t)                          # noise 예측
    μ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_pred)    # posterior mean
    x_{t-1} = μ + σ_t · z                          # stochastic step
return x_0
```

### 11.2 역방향 스텝의 해석

$$\mathbf{x}_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}}_{\text{스케일 복원}} \left( \mathbf{x}_t - \underbrace{\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}}_{\text{noise 제거 강도}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \underbrace{\sigma_t \mathbf{z}}_{\text{확률적 noise}}$$

- $\frac{1}{\sqrt{\alpha_t}}$: forward에서 $\sqrt{\alpha_t}$로 축소된 신호를 복원
- $\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta$: 예측된 noise의 기여분을 제거
- $\sigma_t \mathbf{z}$: 역과정의 **확률성** 유지 (DDPM은 stochastic sampler)

### 11.3 마지막 스텝 ($t = 1$)

$t = 1$에서는 $\mathbf{z} = \mathbf{0}$ — noise를 추가하지 않고 평균만 반환한다. 이는 최종 출력이 깨끗한 데이터가 되도록 보장한다.

### 11.4 구현

```rust
pub fn p_sample(&self, x_t: &Variable, t: usize) -> Variable {
    let _guard = no_grad();
    let predicted_noise = self.forward(x_t, &vec![t; batch]);

    let coeff = self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t];
    let mean = &(x_t - &(&predicted_noise * coeff)) * (1.0 / (1.0 - self.betas[t]).sqrt());

    if t == 0 { mean }
    else { &mean + &(&z * self.posterior_variance[t].sqrt()) }
}

pub fn sample(&self, num_samples: usize) -> Variable {
    let _guard = no_grad();
    let mut x = /* N(0, I) of shape [num_samples, data_dim] */;
    for t in (0..self.timesteps).rev() {
        x = self.p_sample(&x, t);
    }
    x
}
```

---

## 12. Score Matching과의 연결

### 12.1 Score Function

**Score function**은 로그 밀도의 gradient:

$$\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

직관: 데이터 밀도가 높은 방향을 가리키는 **벡터장**.

### 12.2 ε-Prediction = Score Estimation

$q(\mathbf{x}_t | \mathbf{x}_0)$의 score:

$$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0}{1 - \bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$$

따라서 noise 예측과 score estimation은 상수 차이:

$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) = -\sqrt{1 - \bar{\alpha}_t} \cdot \mathbf{s}_\theta(\mathbf{x}_t, t)$$

$$\boxed{\mathbf{s}_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t)}$$

이것이 DDPM과 **Score-Based Generative Models** (Song & Ermon, 2019)의 등가성이다.

### 12.3 Langevin Dynamics

Score를 알면 **Langevin dynamics**로 샘플링 가능:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\eta}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_k) + \sqrt{\eta}\, \mathbf{z}_k$$

이는 $p(\mathbf{x})$를 향해 gradient를 따라가면서 확률적 noise를 추가하는 MCMC 방법이다. DDPM의 sampling은 사실 **시간-의존 Langevin dynamics**의 특수한 형태이다.

### 12.4 SDE 통합 관점 (Song et al., 2021)

Forward diffusion을 연속 시간 SDE로 일반화:

$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\, dt + g(t)\, d\mathbf{w}$$

DDPM의 경우 (VP-SDE):

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\, dt + \sqrt{\beta(t)}\, d\mathbf{w}$$

Reverse SDE (Anderson, 1982):

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + g(t)\, d\bar{\mathbf{w}}$$

Score $\nabla_{\mathbf{x}} \log p_t$를 신경망으로 추정하면 reverse SDE를 풀 수 있다 — 이것이 DDPM의 연속 시간 해석이다.

---

## 13. DDIM (Denoising Diffusion Implicit Models)

### 13.1 DDPM의 한계: 느린 샘플링

DDPM은 $T = 1000$ 스텝을 역방향으로 순차 실행해야 한다. 이미지 한 장 생성에 수십 초가 걸린다.

### 13.2 DDIM의 핵심 아이디어

Song et al. (2021b)의 DDIM은 **non-Markovian** forward process를 설계하여, 같은 학습된 모델 $\boldsymbol{\epsilon}_\theta$를 사용하면서 **적은 스텝**으로 샘플링:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\, \boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } \mathbf{x}_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_\theta + \sigma_t \mathbf{z}$$

- $\sigma_t = 0$: **deterministic** DDIM (ODE solver)
- $\sigma_t = \sqrt{\tilde{\beta}_t}$: DDPM과 동일

### 13.3 Subsequence Sampling

$T = 1000$ 중 $S$개의 부분집합 $\{\tau_1, \tau_2, \ldots, \tau_S\}$만 사용:

$$\tau = [1, 51, 101, 151, \ldots, 951] \quad (S = 20)$$

DDPM에서 학습한 $\boldsymbol{\epsilon}_\theta$를 그대로 사용하면서 $20$배 빠르게 샘플링. 샘플 품질은 소폭 저하.

---

## 14. Classifier-Free Guidance

### 14.1 Conditional Generation

클래스 라벨 $y$로 조건부 생성: $p(\mathbf{x} | y)$.

$$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y): \text{조건부 noise 예측}$$

### 14.2 Guidance의 원리

Score의 베이즈 분해:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x} | y) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \nabla_{\mathbf{x}} \log p(y | \mathbf{x})$$

$$= \underbrace{\text{unconditional score}}_{\text{구조}} + \underbrace{\text{classifier gradient}}_{\text{조건 방향}}$$

### 14.3 Classifier-Free 방식 (Ho & Salimans, 2022)

별도 classifier 없이, **무조건부/조건부 모델을 하나의 네트워크**로 학습:

- 학습 시: 일정 확률(e.g. 10%)로 $y = \varnothing$ (unconditional)으로 학습
- 추론 시: 두 예측을 보간

$$\hat{\boldsymbol{\epsilon}} = (1 + w) \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)$$

$w$는 **guidance scale**:
- $w = 0$: 순수 조건부 모델
- $w > 0$: 조건 방향으로 "밀기" → 다양성 ↓, 품질/일치도 ↑
- $w = 7.5$: Stable Diffusion의 기본값

---

## 15. DDPM vs Score SDE 통합

### 15.1 통합 프레임워크

| 관점 | DDPM (이산) | Score SDE (연속) |
|------|------------|-----------------|
| Forward | $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ | $d\mathbf{x} = \mathbf{f}\,dt + g\,d\mathbf{w}$ |
| Reverse | $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ | Anderson reverse SDE |
| 학습 | $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$ | Score matching |
| 샘플링 | Ancestral sampling | SDE/ODE solver |

### 15.2 Probability Flow ODE

모든 SDE에 대응하는 **결정론적 ODE**가 존재한다 (같은 marginal 분포 유지):

$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt$$

이 ODE를 풀면 **정확한 log-likelihood** 계산 가능 (continuous normalizing flow와 등가).

---

## 16. 확산 모델의 계보

```
Sohl-Dickstein et al. (2015) ── 확산 기반 생성의 최초 제안
        │
Ho et al. (2020) ── DDPM: ε-prediction, simplified loss
        │
   ┌────┼────────────────────────┐
   │    │                        │
Song et al. (2021a)    Song et al. (2021b)    Nichol & Dhariwal (2021)
Score SDE              DDIM                    Improved DDPM
   │                   (빠른 샘플링)              (cosine schedule)
   │                        │
   └───────┬────────────────┘
           │
    Dhariwal & Nichol (2021) ── ADM: classifier guidance, GAN 능가
           │
    Ho & Salimans (2022) ── Classifier-Free Guidance
           │
    ┌──────┼──────┐
    │      │      │
 DALL-E 2  Imagen  Stable Diffusion (Rombach et al., 2022)
                   Latent Diffusion = VAE encoder + DDPM in latent space
```

---

## 17. 구현 요약

### 17.1 구조

```rust
pub struct DDPM {
    t_mlp1: Linear, t_mlp2: Linear,         // Time embedding MLP
    x_proj: Linear, block1: Linear,          // Denoiser
    block2: Linear, out_proj: Linear,
    // Schedule (사전 계산, f64)
    betas, alphas_cumprod, sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod, posterior_variance: Vec<f64>,
    pub data_dim, pub timesteps, t_emb_dim: usize,
    rng_state: Cell<u64>,
}
```

### 17.2 핵심 메서드

| 메서드 | 역할 | 수식 |
|--------|------|------|
| `q_sample` | Forward diffusion | $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ |
| `forward` | Noise 예측 | $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ |
| `p_sample` | Reverse 한 스텝 | $\mu + \sigma_t \mathbf{z}$ |
| `sample` | 전체 역방향 | $\mathbf{x}_T \to \mathbf{x}_0$ |

### 17.3 학습 결과 (step77 테스트)

```
epoch   1: loss = 4.237176
epoch  41: loss = 2.393737
epoch  81: loss = 1.696241
epoch 121: loss = 1.784118
epoch 161: loss = 1.467109
epoch 200: loss = 1.860047
first 5 avg = 3.888030, last 5 avg = 1.895683
```

Loss가 초기 ~3.9에서 ~1.9로 감소. data_dim=4, hidden=32, T=50, 200 epoch.

---

## 18. VAE-GAN-DDPM 삼각 비교

### 18.1 생성 파이프라인

```
VAE:   x → encoder → μ,σ → z ~ N(μ,σ²) → decoder → x̂
GAN:   z ~ N(0,I) → G(z) → fake     vs     real → D → [0,1]
DDPM:  x₀ → (+noise) → x_t → ε_θ(x_t,t) → (-noise) → x_{t-1} → ... → x̂₀
```

### 18.2 손실 함수 비교

| 모델 | 손실 | 수학적 해석 |
|------|------|------------|
| VAE | $\mathrm{MSE} + \beta \cdot D_{\mathrm{KL}}$ | ELBO 최대화 |
| GAN | $\mathrm{BCE}(D(\cdot), \cdot)$ | JS divergence 최소화 |
| DDPM | $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$ | Denoising score matching |

### 18.3 안정성과 품질 트레이드오프

| 기준 | VAE | GAN | DDPM |
|------|-----|-----|------|
| 학습 안정성 | 높음 | 낮음 (mode collapse) | 높음 |
| 샘플 품질 | 보통 (흐릿) | 높음 (선명) | 매우 높음 |
| 다양성 | 높음 | 낮음 | 높음 |
| 샘플링 속도 | 빠름 | 빠름 | 느림 ($T$번 반복) |
| 밀도 추정 | 가능 (하한) | 불가 | 가능 (하한) |
| Latent space | 의미 있음 | 없음 (직접) | 없음 (noise) |

---

## 19. 다음 단계: DiT (Diffusion Transformer)

DDPM의 U-Net denoiser를 **Vision Transformer** (ViT)로 대체:
- Peebles & Xie (2023)의 DiT
- Latent space에서 patch를 토큰으로 처리
- AdaLN (Adaptive Layer Normalization)으로 time/class conditioning
- U-Net 대비 스케일링 효율 우수

이것이 Step 78의 주제가 된다.
