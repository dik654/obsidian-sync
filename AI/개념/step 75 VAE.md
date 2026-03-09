# Step 75: VAE (Variational Autoencoder)

> Phase 5 (생성모델)의 첫 번째 스텝. Kingma & Welling (2014)이 제안한 VAE는 **확률적 잠재 변수 모델**과 **신경망 기반 추론**을 결합하여, 데이터를 생성하면서도 의미 있는 잠재 표현을 학습하는 대표적 생성 모델이다.

---

## 1. 생성 모델의 목표

### 1.1 생성 모델이란

생성 모델(Generative Model)의 목표는 데이터의 **진짜 분포** $p_{\mathrm{data}}(\mathbf{x})$를 근사하는 모델 분포 $p_\theta(\mathbf{x})$를 학습하는 것이다:

$$p_\theta(\mathbf{x}) \approx p_{\mathrm{data}}(\mathbf{x})$$

학습된 모델에서 **새로운 샘플**을 생성할 수 있다:

$$\mathbf{x}_{\mathrm{new}} \sim p_\theta(\mathbf{x})$$

### 1.2 판별 모델과의 비교

| | 판별 모델 | 생성 모델 |
|--|---------|---------|
| 목표 | $p(y|\mathbf{x})$ 학습 | $p(\mathbf{x})$ 또는 $p(\mathbf{x}, y)$ 학습 |
| 예시 | 분류기, 회귀 | VAE, GAN, Diffusion |
| 용도 | 라벨 예측 | 데이터 생성, 밀도 추정, 표현 학습 |
| 결정 경계 | 필요한 것만 모델링 | 전체 데이터 분포 모델링 |

생성 모델은 판별 모델보다 **더 어려운 문제**를 푼다 — 결정 경계만 찾는 것이 아니라 데이터의 전체 구조를 이해해야 하기 때문이다.

### 1.3 잠재 변수 모델 (Latent Variable Model)

고차원 데이터 $\mathbf{x} \in \mathbb{R}^D$를 직접 모델링하기 어렵다. **매니폴드 가설**(manifold hypothesis)에 따르면, 고차원 데이터는 실제로 저차원 매니폴드 위에 집중되어 있다. 예를 들어 28×28 MNIST 이미지는 $\mathbb{R}^{784}$에 살지만, 실질적 자유도는 수십 차원에 불과하다 (숫자 종류, 기울기, 굵기, 크기 등).

이 직관을 수학적으로 구현하기 위해, 저차원 **잠재 변수** $\mathbf{z} \in \mathbb{R}^d$ ($d \ll D$)를 도입하여 2단계로 생성한다:

$$p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}$$

- $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$: 잠재 공간의 사전 분포 (prior)
- $p_\theta(\mathbf{x} | \mathbf{z})$: 디코더가 학습하는 조건부 분포
- $\mathbf{z}$: 데이터의 "요약" — 예컨대 손글씨 숫자의 기울기, 굵기, 스타일 등

문제: 이 적분은 **intractable** (해석적으로 계산 불가능)하다.

### 1.4 왜 Intractable인가?

사후 분포(posterior)를 Bayes 정리로 쓰면:

$$p_\theta(\mathbf{z} | \mathbf{x}) = \frac{p_\theta(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z})}{p_\theta(\mathbf{x})} = \frac{p_\theta(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z})}{\int p_\theta(\mathbf{x} | \mathbf{z}') \, p(\mathbf{z}') \, d\mathbf{z}'}$$

분모의 적분이 $\mathbf{z}$의 전체 공간에 대해 이루어지므로, 잠재 공간이 연속적이고 고차원이면 직접 계산이 불가능하다.

**구체적 이유**: $p_\theta(\mathbf{x}|\mathbf{z})$가 신경망(디코더)으로 매개변수화되면, $\mathbf{z}$에 대해 비선형적으로 의존한다. 적분 $\int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$는 closed-form이 존재하지 않으며, Monte Carlo 추정도 고차원에서는 분산이 너무 커서 실용적이지 않다.

수치적으로 보자. $d = 20$차원 잠재 공간에서 각 축을 $k = 100$개 점으로 이산화하면, 총 격자점 수는 $100^{20} = 10^{40}$개다. 이를 직접 합산하는 것은 현존하는 어떤 컴퓨터로도 불가능하다. 단순한 Monte Carlo도 이 차원에서는 분산이 지수적으로 커져서 비실용적이다.

VAE는 이 문제를 **변분 추론**(Variational Inference)으로 우회한다.

---

## 2. 변분 추론과 ELBO

### 2.1 핵심 아이디어

사후 분포 $p_\theta(\mathbf{z} | \mathbf{x})$를 직접 구할 수 없으므로, 파라미터 $\phi$로 근사하는 **인식 모델**(recognition model) $q_\phi(\mathbf{z} | \mathbf{x})$를 도입한다:

$$q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \, \mathrm{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$$

이것이 VAE의 **Encoder**이다. 입력 $\mathbf{x}$를 받아 $\boldsymbol{\mu}$와 $\boldsymbol{\sigma}^2$ (또는 $\log \boldsymbol{\sigma}^2$)를 출력한다.

> **왜 대각 가우시안인가?** 풀 공분산 $\boldsymbol{\Sigma}$는 $d(d+1)/2$개 파라미터가 필요하지만, 대각 가우시안은 $2d$개만 필요하다. 또한 대각 가정 하에서 KL divergence가 closed-form으로 계산되어 학습이 효율적이다. 표현력이 부족할 경우 Normalizing Flow 등으로 보완할 수 있다 (§12.1).

### 2.2 ELBO 유도 — 방법 1: Jensen 부등식

로그 주변 우도(log marginal likelihood)에서 시작:

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$$

$q_\phi(\mathbf{z} | \mathbf{x})$를 이용한 importance sampling 변환:

$$\log p_\theta(\mathbf{x}) = \log \int \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \, q_\phi(\mathbf{z} | \mathbf{x}) \, d\mathbf{z} = \log \, \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \right]$$

Jensen의 부등식 ($\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$, $\log$가 오목함수):

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \right] \equiv \mathcal{L}(\theta, \phi; \mathbf{x})$$

이 하한 $\mathcal{L}$이 **ELBO** (Evidence Lower Bound)이다.

> **Jensen 등호 조건**: $\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}$가 $\mathbf{z}$에 대해 상수일 때, 즉 $q_\phi(\mathbf{z}|\mathbf{x}) \propto p_\theta(\mathbf{x},\mathbf{z})$일 때 등호가 성립한다. 이는 $q_\phi(\mathbf{z}|\mathbf{x}) = p_\theta(\mathbf{z}|\mathbf{x})$와 동치다.

### 2.3 ELBO 유도 — 방법 2: KL 분해

로그 우도를 $q_\phi$에 대한 기댓값으로 쓴다. $\log p_\theta(\mathbf{x})$는 $\mathbf{z}$에 의존하지 않으므로:

$$\log p_\theta(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x})]$$

$p_\theta(\mathbf{x}) = \frac{p_\theta(\mathbf{x},\mathbf{z})}{p_\theta(\mathbf{z}|\mathbf{x})}$를 대입:

$$= \mathbb{E}_{q_\phi} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z} | \mathbf{x})} \right]$$

분자·분모에 $q_\phi(\mathbf{z}|\mathbf{x})$를 곱하고 나눈다:

$$= \mathbb{E}_{q_\phi} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \cdot \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]$$

$$= \mathbb{E}_{q_\phi} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] + \mathbb{E}_{q_\phi} \left[ \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]$$

$$\boxed{\log p_\theta(\mathbf{x}) = \underbrace{\mathcal{L}(\theta, \phi; \mathbf{x})}_{\mathrm{ELBO}} + \underbrace{D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))}_{\geq \, 0}}$$

$D_{\mathrm{KL}} \geq 0$이므로 $\mathcal{L} \leq \log p_\theta(\mathbf{x})$이 자동으로 성립한다. 이 유도는 Jensen 부등식을 사용하지 않으며, ELBO가 하한인 이유를 더 명확하게 보여준다.

### 2.4 ELBO를 Reconstruction + KL로 분해

ELBO를 두 항으로 분해한다:

$$\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} | \mathbf{x}) \right]$$

$p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} | \mathbf{z}) \, p(\mathbf{z})$를 대입하면:

$$\mathcal{L} = \mathbb{E}_{q_\phi} [\log p_\theta(\mathbf{x} | \mathbf{z}) + \log p(\mathbf{z}) - \log q_\phi(\mathbf{z} | \mathbf{x})]$$

$$= \mathbb{E}_{q_\phi} [\log p_\theta(\mathbf{x} | \mathbf{z})] + \mathbb{E}_{q_\phi}\!\left[ \log \frac{p(\mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \right]$$

두 번째 항은 KL divergence의 **부호 반전**이다:

$$\mathbb{E}_{q_\phi}\!\left[ \log \frac{p(\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right] = -\mathbb{E}_{q_\phi}\!\left[ \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p(\mathbf{z})} \right] = -D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

따라서:

$$\boxed{\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x} | \mathbf{z})]}_{\text{Reconstruction}} - \underbrace{D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL Divergence}}}$$

- **Reconstruction term**: 디코더가 잠재 표현에서 원본을 잘 복원하는가? 이 항을 최대화하면 $q_\phi$가 $\mathbf{x}$의 정보를 $\mathbf{z}$에 많이 담도록 유도.
- **KL term**: 인코더의 사후 근사가 사전 분포 $\mathcal{N}(\mathbf{0}, \mathbf{I})$에서 얼마나 벗어나는가? 이 항을 최소화하면 잠재 공간이 정규화되어 샘플링 가능해짐.

두 항은 **본질적으로 상충**한다: reconstruction은 $\mathbf{z}$에 정보를 담기를 원하고, KL은 $\mathbf{z}$를 uninformative한 사전 분포로 밀어넣으려 한다.

### 2.5 ELBO와 로그 우도의 관계

$$\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi; \mathbf{x}) + D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))$$

- KL divergence $\geq 0$이므로 $\mathcal{L} \leq \log p_\theta(\mathbf{x})$ (하한)
- $q_\phi = p_\theta(\mathbf{z}|\mathbf{x})$이면 $\mathrm{KL} = 0$이고 $\mathcal{L} = \log p_\theta(\mathbf{x})$ (등호)
- ELBO를 최대화하면 동시에: (1) 로그 우도 $\uparrow$, (2) 근사 정확도 $\uparrow$

### 2.6 VAE와 EM 알고리즘의 관계

VAE의 학습은 **amortized variational EM**으로 볼 수 있다.

전통적 EM:
- **E-step**: $q(\mathbf{z}) \leftarrow p_\theta(\mathbf{z}|\mathbf{x})$ (정확한 사후 분포 계산)
- **M-step**: $\theta \leftarrow \arg\max_\theta \mathbb{E}_{q(\mathbf{z})}[\log p_\theta(\mathbf{x},\mathbf{z})]$

VAE의 차이점:
1. **Amortized inference**: 각 $\mathbf{x}$마다 별도 최적화 대신, 인코더 $q_\phi(\mathbf{z}|\mathbf{x})$가 **모든 $\mathbf{x}$에 대해 동시에** 사후 분포를 근사한다. 이를 **amortization**이라 하며, 학습/추론 시 encoder를 한번 forward하면 되므로 효율적이다.
2. **Approximate E-step**: $q_\phi \neq p_\theta(\mathbf{z}|\mathbf{x})$이므로 근사적이다.
3. **동시 최적화**: E-step과 M-step을 분리하지 않고 $\theta, \phi$를 동시에 경사 하강법으로 업데이트한다.

> **Amortization gap**: 전통 VI는 각 데이터 포인트에 별도 $q_i(\mathbf{z})$를 최적화하므로 더 정확하다. Amortized inference는 이를 $q_\phi(\mathbf{z}|\mathbf{x})$로 공유하므로 근사 품질이 떨어질 수 있다. 이 차이를 amortization gap이라 한다:
> $$\mathrm{gap}_{\mathrm{amort}} = \max_{q_i} \mathcal{L}_i - \mathcal{L}(q_\phi(\cdot|\mathbf{x}_i))$$

---

## 3. Reparameterization Trick

### 3.1 문제: 샘플링을 통한 역전파

ELBO의 reconstruction term은 $q_\phi(\mathbf{z}|\mathbf{x})$에 대한 기댓값:

$$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x} | \mathbf{z})]$$

Monte Carlo 추정으로 $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$를 샘플링하여 근사하려는데, **샘플링 연산은 미분 불가능**하다. $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}_\phi, \boldsymbol{\sigma}_\phi^2)$에서 직접 샘플링하면 $\phi$에 대한 gradient를 구할 수 없다.

**왜 미분 불가능인가?** $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$에서 샘플링하는 것은 확률적 프로세스다. $\boldsymbol{\mu}$를 살짝 바꾸면 $\mathbf{z}$의 **분포**는 변하지만, 이미 뽑힌 **특정 샘플** $\mathbf{z}$와 $\boldsymbol{\mu}$ 사이에는 결정적 함수 관계가 없다. 역전파는 결정적 함수의 chain rule에 기반하므로, 확률적 노드를 관통할 수 없다.

### 3.2 대안: REINFORCE (Score Function Estimator)

Reparameterization이 없을 때의 대안은 **REINFORCE** (Williams, 1992):

$$\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[f(\mathbf{z})] = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[f(\mathbf{z}) \nabla_\phi \log q_\phi(\mathbf{z}|\mathbf{x})]$$

**유도**:

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})] = \nabla_\phi \int f(\mathbf{z}) q_\phi(\mathbf{z}|\mathbf{x}) \, d\mathbf{z} = \int f(\mathbf{z}) \nabla_\phi q_\phi(\mathbf{z}|\mathbf{x}) \, d\mathbf{z}$$

Log-derivative trick: $\nabla_\phi q = q \cdot \nabla_\phi \log q$를 대입:

$$= \int f(\mathbf{z}) q_\phi(\mathbf{z}|\mathbf{x}) \nabla_\phi \log q_\phi(\mathbf{z}|\mathbf{x}) \, d\mathbf{z} = \mathbb{E}_{q_\phi}[f(\mathbf{z}) \nabla_\phi \log q_\phi(\mathbf{z}|\mathbf{x})]$$

**문제**: 이 추정량은 **분산이 매우 크다**. $f(\mathbf{z})$가 스칼라인데 $\nabla_\phi \log q$와 곱해지므로, $f$의 절대값이 크면 gradient 추정이 매우 노이즈하다. 실전에서 학습이 불안정하다.

### 3.3 핵심 아이디어: 확률성의 분리

확률적 노드 $\mathbf{z}$를 **결정적 함수** + **외부 노이즈**로 분리한다:

$$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
$$\mathbf{z} = g_\phi(\boldsymbol{\epsilon}, \mathbf{x}) = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}$$

이제 $\mathbf{z}$는 $\boldsymbol{\mu}$와 $\boldsymbol{\sigma}$의 **결정적 함수**이므로, chain rule로 gradient가 흐를 수 있다:

$$\frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}} = \mathbf{I}, \qquad \frac{\partial \mathbf{z}}{\partial \boldsymbol{\sigma}} = \mathrm{diag}(\boldsymbol{\epsilon})$$

기댓값의 gradient가 다음과 같이 바뀐다:

$$\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[f(\mathbf{z})] = \nabla_\phi \mathbb{E}_{p(\boldsymbol{\epsilon})}[f(g_\phi(\boldsymbol{\epsilon}, \mathbf{x}))] = \mathbb{E}_{p(\boldsymbol{\epsilon})}[\nabla_\phi f(g_\phi(\boldsymbol{\epsilon}, \mathbf{x}))]$$

$p(\boldsymbol{\epsilon}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$는 $\phi$에 의존하지 않으므로, **미분과 기댓값의 교환**이 정당화된다.

### 3.4 분포 보존 증명

$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$이고 $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$이면, $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2))$임을 보인다.

가우시안의 **아핀 변환 성질**: $\mathbf{X} \sim \mathcal{N}(\mathbf{m}, \boldsymbol{\Sigma})$이면 $\mathbf{A}\mathbf{X} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\mathbf{m} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$.

$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$에 $\mathbf{A} = \mathrm{diag}(\boldsymbol{\sigma})$, $\mathbf{b} = \boldsymbol{\mu}$를 적용하면:

$$\mathbf{z} = \mathrm{diag}(\boldsymbol{\sigma}) \cdot \boldsymbol{\epsilon} + \boldsymbol{\mu} \sim \mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma})\mathbf{I}\mathrm{diag}(\boldsymbol{\sigma})^\top) = \mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2))$$

따라서 reparameterized 샘플의 분포는 원래 $q_\phi(\mathbf{z}|\mathbf{x})$와 정확히 동일하다. ∎

### 3.5 Gradient 분산 비교: REINFORCE vs Reparameterization

두 추정량의 분산 차이는 실전에서 극적이다:

$$\mathrm{Var}[\hat{g}_{\mathrm{REINFORCE}}] = \mathrm{Var}[f(\mathbf{z}) \nabla_\phi \log q_\phi(\mathbf{z}|\mathbf{x})]$$

$$\mathrm{Var}[\hat{g}_{\mathrm{reparam}}] = \mathrm{Var}[\nabla_\phi f(g_\phi(\boldsymbol{\epsilon}, \mathbf{x}))]$$

REINFORCE에서 $f(\mathbf{z})$는 전체 loss 값 (큰 스칼라)과 log-likelihood의 gradient의 **곱**이므로, 분산이 $\mathcal{O}(\|f\|^2)$로 스케일된다. 반면 reparameterization은 $f$를 통한 **직접 미분**이므로, $f$의 Lipschitz 상수에 비례하는 훨씬 작은 분산을 갖는다.

실험적으로 reparameterization gradient의 분산은 REINFORCE보다 **수십~수백 배** 작으며, 이것이 VAE가 안정적으로 학습되는 핵심 이유다 (Kingma & Welling, 2014, Table 1).

### 3.6 log_var 파라미터화

실전에서는 $\boldsymbol{\sigma}$ 대신 $\log \boldsymbol{\sigma}^2$을 출력한다:

$$\mathbf{z} = \boldsymbol{\mu} + \exp\!\left(\tfrac{1}{2} \log \boldsymbol{\sigma}^2\right) \cdot \boldsymbol{\epsilon} = \boldsymbol{\mu} + \boldsymbol{\sigma} \cdot \boldsymbol{\epsilon}$$

이유:
- **제약 해소**: $\boldsymbol{\sigma} > 0$이어야 하지만, $\log \boldsymbol{\sigma}^2 \in (-\infty, +\infty)$로 제약 없는 출력
- **수치 안정성**: $\sigma$가 매우 작을 때 $\log \sigma^2$이 음의 큰 값이 되어 부드럽게 처리됨. $\sigma$를 직접 출력하면 softplus 등 추가 변환이 필요
- **KL 공식과의 직접 대응**: KL 공식에서 $\log \boldsymbol{\sigma}^2$이 직접 등장하여 계산이 깔끔

### 3.7 Gradient 흐름 상세 분석

```
x → Encoder → μ ──┐
              → log σ² ─→ σ = exp(0.5·log σ²) ─→ z = μ + σ·ε → Decoder → x_recon
                                                  ↑
                                            ε ~ N(0,I) (leaf, no grad)
```

$\boldsymbol{\epsilon}$은 creator가 없는 leaf Variable → backward 시 gradient가 흘러오지 않음.

**$\mathbf{z}$에서 $\boldsymbol{\mu}$로의 gradient 흐름**:

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \cdot \frac{\partial \mathbf{z}}{\partial \boldsymbol{\mu}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \cdot \mathbf{I} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}$$

**$\mathbf{z}$에서 $\log \boldsymbol{\sigma}^2$로의 gradient 흐름**:

$\gamma_j = \log \sigma_j^2$라 놓으면 $\sigma_j = e^{\gamma_j / 2}$이고 $z_j = \mu_j + e^{\gamma_j/2} \cdot \epsilon_j$:

$$\frac{\partial z_j}{\partial \gamma_j} = \frac{\partial}{\partial \gamma_j} e^{\gamma_j/2} \cdot \epsilon_j = \frac{1}{2} e^{\gamma_j/2} \cdot \epsilon_j = \frac{1}{2} \sigma_j \epsilon_j$$

따라서:

$$\frac{\partial \mathcal{L}}{\partial \gamma_j} = \frac{\partial \mathcal{L}}{\partial z_j} \cdot \frac{1}{2} \sigma_j \epsilon_j$$

이 gradient는 $\epsilon_j$에 의존하므로 **확률적**이지만, 기댓값이 올바른 gradient 방향을 가리킨다.

### 3.8 구현

```rust
pub fn reparameterize(&self, mu: &Variable, log_var: &Variable) -> Variable {
    let shape = mu.shape();
    let n: usize = shape.iter().product();
    let eps_data: Vec<f64> = (0..n).map(|_| self.next_normal()).collect();
    let eps = Variable::new(
        ArrayD::from_shape_vec(ndarray::IxDyn(&shape), eps_data).unwrap(),
    );
    // z = mu + exp(0.5 * log_var) * eps
    mu + &(&exp(&(log_var * 0.5)) * &eps)
}
```

`eps`는 `Variable::new()`로 생성 — creator가 설정되지 않으므로 backward에서 자동으로 leaf 역할을 한다. 별도의 `detach()` 없이 reparameterization trick이 성립하는 이유다.

---

## 4. KL Divergence 해석적 계산

### 4.1 KL Divergence 정의

두 확률분포 $q$, $p$에 대해:

$$D_{\mathrm{KL}}(q \| p) = \int q(\mathbf{z}) \log \frac{q(\mathbf{z})}{p(\mathbf{z})} \, d\mathbf{z} = \mathbb{E}_q\!\left[\log \frac{q(\mathbf{z})}{p(\mathbf{z})}\right]$$

기본 성질:
- $D_{\mathrm{KL}} \geq 0$ (Gibbs 부등식, 등호는 $q = p$ a.e.일 때)
- **비대칭**: $D_{\mathrm{KL}}(q \| p) \neq D_{\mathrm{KL}}(p \| q)$

### 4.2 두 다변량 가우시안의 KL — 완전한 유도

$q = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$, $p = \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$에 대해 KL을 유도한다.

다변량 가우시안의 로그 밀도:

$$\log \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{1}{2}(\mathbf{z} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{z} - \boldsymbol{\mu})$$

KL의 정의에 대입:

$$D_{\mathrm{KL}}(q \| p) = \mathbb{E}_q[\log q(\mathbf{z}) - \log p(\mathbf{z})]$$

$$= \mathbb{E}_q\!\left[-\frac{1}{2}\log|\boldsymbol{\Sigma}_1| - \frac{1}{2}(\mathbf{z}-\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_1^{-1}(\mathbf{z}-\boldsymbol{\mu}_1) + \frac{1}{2}\log|\boldsymbol{\Sigma}_0| + \frac{1}{2}(\mathbf{z}-\boldsymbol{\mu}_0)^\top\boldsymbol{\Sigma}_0^{-1}(\mathbf{z}-\boldsymbol{\mu}_0)\right]$$

각 항을 개별적으로 계산한다.

**항 1**: $\mathbb{E}_q[(\mathbf{z}-\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}_1^{-1}(\mathbf{z}-\boldsymbol{\mu}_1)]$

$\mathbf{z} \sim q$일 때 $\mathbf{z} - \boldsymbol{\mu}_1$의 공분산이 $\boldsymbol{\Sigma}_1$이므로:

$$= \mathrm{tr}(\boldsymbol{\Sigma}_1^{-1} \mathbb{E}_q[(\mathbf{z}-\boldsymbol{\mu}_1)(\mathbf{z}-\boldsymbol{\mu}_1)^\top]) = \mathrm{tr}(\boldsymbol{\Sigma}_1^{-1}\boldsymbol{\Sigma}_1) = \mathrm{tr}(\mathbf{I}_d) = d$$

> **사용된 항등식**: $\mathbb{E}[\mathbf{a}^\top \mathbf{M} \mathbf{a}] = \mathrm{tr}(\mathbf{M} \, \mathbb{E}[\mathbf{a}\mathbf{a}^\top])$ (이차형식의 기댓값 = trace trick)

**항 2**: $\mathbb{E}_q[(\mathbf{z}-\boldsymbol{\mu}_0)^\top\boldsymbol{\Sigma}_0^{-1}(\mathbf{z}-\boldsymbol{\mu}_0)]$

$\mathbf{z} - \boldsymbol{\mu}_0 = (\mathbf{z} - \boldsymbol{\mu}_1) + (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$로 분해:

$$\mathbb{E}_q[(\mathbf{z}-\boldsymbol{\mu}_0)(\mathbf{z}-\boldsymbol{\mu}_0)^\top] = \boldsymbol{\Sigma}_1 + (\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)^\top$$

따라서:

$$\mathbb{E}_q[(\mathbf{z}-\boldsymbol{\mu}_0)^\top\boldsymbol{\Sigma}_0^{-1}(\mathbf{z}-\boldsymbol{\mu}_0)] = \mathrm{tr}(\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)^\top\boldsymbol{\Sigma}_0^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)$$

모두 합치면:

$$D_{\mathrm{KL}}(q \| p) = \frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_0|}{|\boldsymbol{\Sigma}_1|} - d + \mathrm{tr}(\boldsymbol{\Sigma}_0^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)^\top\boldsymbol{\Sigma}_0^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_0)\right]$$

### 4.3 VAE 특수 케이스: $p = \mathcal{N}(\mathbf{0}, \mathbf{I})$

VAE에서 사전 분포가 표준 정규분포이므로 $\boldsymbol{\mu}_0 = \mathbf{0}$, $\boldsymbol{\Sigma}_0 = \mathbf{I}$:

$$\boldsymbol{\Sigma}_0^{-1} = \mathbf{I}, \quad |\boldsymbol{\Sigma}_0| = 1, \quad \log|\boldsymbol{\Sigma}_0| = 0$$

대입하면:

$$D_{\mathrm{KL}} = \frac{1}{2}\left[-\log|\boldsymbol{\Sigma}_1| - d + \mathrm{tr}(\boldsymbol{\Sigma}_1) + \boldsymbol{\mu}_1^\top \boldsymbol{\mu}_1\right]$$

$q_\phi$의 공분산이 대각행렬 $\boldsymbol{\Sigma}_1 = \mathrm{diag}(\sigma_1^2, \ldots, \sigma_d^2)$이면:

- $\mathrm{tr}(\boldsymbol{\Sigma}_1) = \sum_{j=1}^d \sigma_j^2$
- $|\boldsymbol{\Sigma}_1| = \prod_{j=1}^d \sigma_j^2$ (대각행렬의 행렬식 = 대각 원소의 곱)
- $\log|\boldsymbol{\Sigma}_1| = \sum_{j=1}^d \log \sigma_j^2$

따라서:

$$D_{\mathrm{KL}} = \frac{1}{2} \sum_{j=1}^d \left(\sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2\right)$$

### 4.4 log_var로 표현

$\log \sigma_j^2 \equiv \gamma_j$로 놓으면 $\sigma_j^2 = e^{\gamma_j}$:

$$\boxed{D_{\mathrm{KL}} = -\frac{1}{2} \sum_{j=1}^d \left(1 + \gamma_j - \mu_j^2 - e^{\gamma_j}\right)}$$

이것이 코드에서 사용하는 공식이다. 배치 평균을 취하면:

$$\mathrm{KL}_{\mathrm{batch}} = \frac{1}{B} \sum_{b=1}^B D_{\mathrm{KL}}^{(b)} = \frac{-1}{2B} \sum_{b=1}^B \sum_{j=1}^d (1 + \gamma_{bj} - \mu_{bj}^2 - e^{\gamma_{bj}})$$

### 4.5 KL의 gradient

최적화에 필요한 $\mu_j$, $\gamma_j$에 대한 편미분:

$$\frac{\partial D_{\mathrm{KL}}}{\partial \mu_j} = \mu_j$$

$$\frac{\partial D_{\mathrm{KL}}}{\partial \gamma_j} = \frac{1}{2}(e^{\gamma_j} - 1)$$

**최적해**: $\nabla D_{\mathrm{KL}} = 0$이면 $\mu_j = 0$, $e^{\gamma_j} = 1 \Rightarrow \gamma_j = 0$, 즉 $q_\phi = \mathcal{N}(\mathbf{0}, \mathbf{I}) = p(\mathbf{z})$. 이 때 $D_{\mathrm{KL}} = 0$.

### 4.6 검증: 표준 정규분포끼리의 KL

$\boldsymbol{\mu} = \mathbf{0}$, $\gamma = \mathbf{0}$ (즉 $\boldsymbol{\sigma} = \mathbf{1}$)이면:

$$D_{\mathrm{KL}} = -\frac{1}{2} \sum_j (1 + 0 - 0 - e^0) = -\frac{1}{2} \sum_j (1 - 1) = 0$$

이것은 같은 분포 사이의 KL이 0이라는 자명한 사실과 일치한다.

```rust
// test_kl_standard_normal에서 검증
let mu = Variable::new(ArrayD::zeros(IxDyn(&[batch, latent])));
let log_var = Variable::new(ArrayD::zeros(IxDyn(&[batch, latent])));
let (_, _, kl) = vae_loss(&x, &x_recon, &mu, &log_var, 1.0);
assert!(kl_val.abs() < 1e-10); // KL ≈ 0 ✓
```

### 4.7 KL 항의 기하학적 의미

각 잠재 차원 $j$에서의 KL 기여:

$$\mathrm{KL}_j = \frac{1}{2}(\sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2)$$

| 조건 | $\mu_j$ | $\sigma_j$ | $\mathrm{KL}_j$ | 해석 |
|------|---------|------------|-----------|------|
| 완벽 일치 | 0 | 1 | 0 | 표준 정규 |
| 평균 이동 | $\neq 0$ | 1 | $\frac{1}{2}\mu_j^2$ | 중심 이동 비용 |
| 분산 축소 | 0 | $< 1$ | $\frac{1}{2}(\sigma^2 - 1 - \log\sigma^2)$ | 확신 비용 |
| 분산 확대 | 0 | $> 1$ | $\frac{1}{2}(\sigma^2 - 1 - \log\sigma^2)$ | 불확실성 비용 |

$\mu$ 항 $\frac{1}{2}\mu_j^2$: 가우시안의 중심이 원점에서 멀어질수록 **이차적**으로 페널티가 증가. 이는 L2 정규화와 동일한 형태다.

$\sigma$ 항 $f(\sigma^2) = \frac{1}{2}(\sigma^2 - 1 - \log\sigma^2)$: $\sigma^2 = 1$에서 최솟값 0을 가지는 볼록 함수다.

**볼록성 증명**:

$$f'(s) = \frac{1}{2}\left(1 - \frac{1}{s}\right), \quad f''(s) = \frac{1}{2s^2} > 0 \quad (\forall s > 0)$$

$f'(s) = 0 \Leftrightarrow s = 1$이고 $f''(1) = \frac{1}{2} > 0$이므로, $s = 1$은 **극소점**(볼록함수이므로 전역 최소). $f(1) = \frac{1}{2}(1 - 1 - 0) = 0$. ∎

**근사**: $\sigma^2 = 1 + \delta$ ($|\delta| \ll 1$)이면 Taylor 전개로:

$$f(1+\delta) \approx \frac{1}{2}\left(\delta - \delta + \frac{\delta^2}{2}\right) = \frac{\delta^2}{4}$$

분산이 1에서 조금 벗어나면 KL은 **편차의 제곱**에 비례 — 부드러운 정규화.

---

## 5. 손실 함수

### 5.1 전체 손실

ELBO를 최대화하는 것은 $-\mathcal{L}$을 최소화하는 것과 동치:

$$\mathcal{L}_{\mathrm{VAE}} = \underbrace{\mathrm{Recon}(\mathbf{x}, \hat{\mathbf{x}})}_{\text{Reconstruction Loss}} + \beta \cdot \underbrace{D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{Regularization}}$$

### 5.2 Reconstruction Loss와 확률적 해석

Reconstruction loss의 선택은 **디코더의 확률적 가정**에서 유도된다.

#### 가우시안 디코더 → MSE

$p_\theta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \hat{\mathbf{x}}_\theta(\mathbf{z}), \sigma^2_{\mathrm{dec}} \mathbf{I})$로 가정하면:

$$\log p_\theta(\mathbf{x}|\mathbf{z}) = -\frac{D}{2}\log(2\pi\sigma^2_{\mathrm{dec}}) - \frac{1}{2\sigma^2_{\mathrm{dec}}}\|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

$\sigma^2_{\mathrm{dec}}$을 상수로 고정하면:

$$\max_\theta \, \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] \iff \min_\theta \, \mathbb{E}_{q_\phi}[\|\mathbf{x} - \hat{\mathbf{x}}\|^2] = \min_\theta \, \mathrm{MSE}$$

> **$\sigma^2_{\mathrm{dec}}$의 역할**: 실제로 $\sigma^2_{\mathrm{dec}}$는 reconstruction과 KL의 **상대적 가중치**를 결정한다. ELBO에서 reconstruction 항의 계수가 $\frac{1}{2\sigma^2_{\mathrm{dec}}}$이므로, $\sigma^2_{\mathrm{dec}}$가 작으면 reconstruction 강조, 크면 KL 강조. $\beta$-VAE의 $\beta$와 수학적으로 동치: $\beta \sim \sigma^2_{\mathrm{dec}}$.

#### 베르누이 디코더 → BCE

$p_\theta(\mathbf{x}|\mathbf{z}) = \prod_{i=1}^D \hat{x}_i^{x_i}(1-\hat{x}_i)^{1-x_i}$로 가정하면:

$$\log p_\theta(\mathbf{x}|\mathbf{z}) = \sum_{i=1}^D [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$$

이를 최대화하는 것은 **BCE**(Binary Cross-Entropy)를 최소화하는 것과 동치다.

BCE는 이진 데이터 (예: 이진화된 MNIST)에 이론적으로 더 적합하지만, 실전에서는 MSE도 잘 동작한다.

| Loss | 확률적 가정 | 적합한 데이터 | 출력 활성화 |
|------|-----------|------------|----------|
| MSE | 가우시안 $\mathcal{N}(\hat{\mathbf{x}}, \sigma^2\mathbf{I})$ | 연속값 $\mathbb{R}^D$ | 없음 또는 sigmoid |
| BCE | 베르누이 $\mathrm{Bern}(\hat{x}_i)$ | 이진값 $\{0,1\}^D$ | sigmoid |

본 구현에서는 MSE를 사용 — 이미 구현된 `mean_squared_error`를 활용하고, sigmoid 출력과 [0,1] 범위 데이터에 충분히 잘 동작한다.

### 5.3 구현

```rust
pub fn vae_loss(
    x: &Variable, x_recon: &Variable,
    mu: &Variable, log_var: &Variable, beta: f64,
) -> (Variable, Variable, Variable) {
    let recon_loss = mean_squared_error(x, x_recon);
    let batch_size = mu.shape()[0] as f64;
    let kl_elem = &(&(log_var + 1.0) - &mu.pow(2.0)) - &exp(log_var);
    let kl_loss = &sum(&kl_elem) * (-0.5 / batch_size);
    let total_loss = &recon_loss + &(&kl_loss * beta);
    (total_loss, recon_loss, kl_loss)
}
```

3-tuple 반환으로 각 loss 성분을 개별 추적할 수 있다.

---

## 6. VAE 아키텍처

### 6.1 구조

```
Input x ∈ ℝ^{input_dim}
    ↓
[Encoder]
    ↓ Linear(input_dim → hidden_dim) + sigmoid
    ↓
h ∈ ℝ^{hidden_dim}
    ├── Linear(hidden_dim → latent_dim) → μ
    └── Linear(hidden_dim → latent_dim) → log σ²
                                    ↓
            z = μ + exp(0.5·log σ²)·ε,  ε ~ N(0,I)
                                    ↓
[Decoder]
    ↓ Linear(latent_dim → hidden_dim) + sigmoid
    ↓ Linear(hidden_dim → input_dim) + sigmoid
    ↓
x_recon ∈ (0, 1)^{input_dim}
```

### 6.2 설계 결정

| 결정 | 선택 | 이유 |
|------|------|------|
| 은닉 활성화 | Sigmoid | ReLU 미구현; sigmoid/tanh로 충분 |
| 출력 활성화 | Sigmoid | [0,1] 출력 보장 (MSE와 호환) |
| log_var | log σ² 직접 출력 | KL 공식과 직접 대응, 제약 없는 파라미터 |
| RNG | LCG + Box-Muller | 프로젝트 관례, 외부 의존성 없음 |
| Lazy init | Linear의 W | 첫 forward에서 입력 차원 자동 결정 |

### 6.3 구현

```rust
pub struct VAE {
    encoder_h: Linear,         // input_dim → hidden_dim
    mu_layer: Linear,          // hidden_dim → latent_dim
    log_var_layer: Linear,     // hidden_dim → latent_dim
    decoder_h: Linear,         // latent_dim → hidden_dim
    decoder_out: Linear,       // hidden_dim → input_dim
    pub latent_dim: usize,
    pub beta: f64,
    rng_state: Cell<u64>,
}
```

5개 Linear 레이어 × (W + b) = **10개 학습 파라미터** (lazy init 후).

### 6.4 파라미터 수 분석

MNIST 전형적 설정 ($D = 784$, $H = 256$, $d_z = 20$):

| 레이어 | W 크기 | b 크기 | 파라미터 수 |
|--------|--------|--------|-----------|
| encoder_h | 784 × 256 | 256 | 200,960 |
| mu_layer | 256 × 20 | 20 | 5,140 |
| log_var_layer | 256 × 20 | 20 | 5,140 |
| decoder_h | 20 × 256 | 256 | 5,376 |
| decoder_out | 256 × 784 | 784 | 201,488 |
| **합계** | | | **418,104** |

Encoder와 decoder가 거의 대칭적이며, 병목은 latent_dim에서 발생한다.

---

## 7. Encoder

### 7.1 구조와 수식

$$\mathbf{h} = \sigma(\mathbf{W}_h \mathbf{x} + \mathbf{b}_h)$$
$$\boldsymbol{\mu} = \mathbf{W}_\mu \mathbf{h} + \mathbf{b}_\mu$$
$$\log \boldsymbol{\sigma}^2 = \mathbf{W}_\gamma \mathbf{h} + \mathbf{b}_\gamma$$

$\boldsymbol{\mu}$와 $\log \boldsymbol{\sigma}^2$는 활성화 함수 없이 출력 — 제약 없는 실수값이어야 하므로.

### 7.2 왜 두 헤드인가?

$\boldsymbol{\mu}$와 $\log \boldsymbol{\sigma}^2$는 **공유 은닉층** $\mathbf{h}$에서 갈라진다. 이를 두 개의 별도 encoder로 구현하면 파라미터가 2배가 되고, $\boldsymbol{\mu}$와 $\boldsymbol{\sigma}$가 독립적으로 학습되어 비효율적이다.

공유 은닉층은:
- 파라미터 효율적
- $\mathbf{x}$의 유용한 특징을 한번 추출하여 $\boldsymbol{\mu}$와 $\boldsymbol{\sigma}$ 모두에 활용
- 학습 안정성 향상

### 7.3 구현

```rust
pub fn encode(&self, x: &Variable) -> (Variable, Variable) {
    let h = sigmoid(&self.encoder_h.forward(x));
    let mu = self.mu_layer.forward(&h);
    let log_var = self.log_var_layer.forward(&h);
    (mu, log_var)
}
```

---

## 8. Decoder

### 8.1 구조와 수식

$$\mathbf{h}' = \sigma(\mathbf{W}_{h'} \mathbf{z} + \mathbf{b}_{h'})$$
$$\hat{\mathbf{x}} = \sigma(\mathbf{W}_{\mathrm{out}} \mathbf{h}' + \mathbf{b}_{\mathrm{out}})$$

출력에 sigmoid을 적용하여 $\hat{\mathbf{x}} \in (0, 1)^D$를 보장한다.

### 8.2 구현

```rust
pub fn decode(&self, z: &Variable) -> Variable {
    let h = sigmoid(&self.decoder_h.forward(z));
    sigmoid(&self.decoder_out.forward(&h))
}
```

---

## 9. 샘플 생성

### 9.1 학습 후 생성 과정

학습된 VAE에서 새 데이터를 생성하려면:

1. 사전 분포에서 잠재 벡터 샘플: $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. 디코더에 입력: $\hat{\mathbf{x}} = \mathrm{Decoder}(\mathbf{z})$

학습이 잘 되었다면, KL regularization 덕분에 $q_\phi(\mathbf{z}|\mathbf{x}) \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$이므로, 사전 분포에서 샘플링한 $\mathbf{z}$가 의미 있는 데이터를 생성할 수 있다.

### 9.2 Prior Hole 문제

KL regularization이 부족하면, 사전 분포에서 샘플된 $\mathbf{z}$가 encoder가 학습한 영역 밖에 놓여 **의미 없는 출력**을 생성할 수 있다. 이를 **prior hole** 또는 **aggregate posterior gap**이라 한다.

형식적으로, **집계 사후 분포**(aggregated posterior)를 정의한다:

$$q_\phi(\mathbf{z}) = \frac{1}{N} \sum_{i=1}^N q_\phi(\mathbf{z} | \mathbf{x}_i) = \mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})}[q_\phi(\mathbf{z}|\mathbf{x})]$$

좋은 생성을 위해서는 $q_\phi(\mathbf{z}) \approx p(\mathbf{z})$여야 한다. 하지만 ELBO의 KL 항은 **각 데이터 포인트별** KL을 최소화할 뿐, 집계 수준의 정합을 직접 보장하지 않는다.

문제 시나리오: $q_\phi(\mathbf{z}|\mathbf{x}_i)$들이 서로 다른 좁은 영역에 집중되면, 집계 분포 $q_\phi(\mathbf{z})$는 여러 "섬"으로 이루어진 불연속적 분포가 된다. 이 섬들 사이의 빈 공간(hole)에서 샘플된 $\mathbf{z}$는 의미 있는 데이터를 생성하지 못한다.

### 9.3 구현

```rust
pub fn sample(&self, num_samples: usize) -> Variable {
    let z_data: Vec<f64> = (0..num_samples * self.latent_dim)
        .map(|_| self.next_normal())
        .collect();
    let z = Variable::new(
        ArrayD::from_shape_vec(
            ndarray::IxDyn(&[num_samples, self.latent_dim]),
            z_data,
        ).unwrap(),
    );
    self.decode(&z)
}
```

---

## 10. $\beta$-VAE

### 10.1 동기

표준 VAE ($\beta = 1$)에서 KL 항과 reconstruction 항의 균형이 항상 최적은 아니다. Higgins et al. (2017)은 KL 가중치 $\beta$를 조절하여 **disentangled representation**을 학습하는 $\beta$-VAE를 제안했다:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathrm{Recon} + \beta \cdot D_{\mathrm{KL}}$$

### 10.2 $\beta$의 효과

| $\beta$ | Reconstruction | KL | 특성 |
|---------|---------------|-----|------|
| $\ll 1$ | 강조 | 무시 | 정확한 복원, 잠재 공간 비정규 |
| $= 1$ | 균형 | 균형 | 표준 VAE |
| $\gg 1$ | 희생 | 강조 | 흐릿한 복원, 잠재 공간 잘 정규화 |

```rust
// test_beta_effect 결과
beta= 0.01: total=1.6538, recon=1.6480, kl=0.5796  // recon 지배
beta= 1.00: total=2.2276, recon=1.6480, kl=0.5796  // 균형
beta=10.00: total=7.4438, recon=1.6480, kl=0.5796  // KL 지배
```

같은 초기 가중치이므로 학습 전에는 recon, kl이 동일하고, total만 $\mathrm{recon} + \beta \cdot \mathrm{kl}$로 달라진다. 학습 과정에서 optimizer가 $\beta$에 맞게 recon과 kl의 균형을 조정한다.

### 10.3 Disentanglement와 Total Correlation

$\beta > 1$이 disentanglement를 유도하는 이유를 **Total Correlation**(TC) 분해로 이해할 수 있다.

KL 항을 세 부분으로 분해한다 (Chen et al., 2018):

$$D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = \underbrace{I_q(\mathbf{x}; \mathbf{z})}_{\text{Index-Code MI}} + \underbrace{D_{\mathrm{KL}}(q_\phi(\mathbf{z}) \| \prod_j q_\phi(z_j))}_{\text{Total Correlation}} + \underbrace{\sum_j D_{\mathrm{KL}}(q_\phi(z_j) \| p(z_j))}_{\text{Dimension-wise KL}}$$

각 항의 의미:

1. **Index-Code MI** $I_q(\mathbf{x}; \mathbf{z})$: $\mathbf{z}$가 $\mathbf{x}$에 대해 얼마나 정보를 담는가. 이 항을 줄이면 posterior collapse.

2. **Total Correlation**: $q_\phi(\mathbf{z})$와 그 주변 분포의 **곱** $\prod_j q_\phi(z_j)$ 사이의 KL. 잠재 차원들 사이의 **통계적 의존성**을 측정. **이 항이 disentanglement의 핵심**이다.

3. **Dimension-wise KL**: 각 차원 개별적으로 사전 분포와의 정합.

$\beta > 1$이면 TC 항에도 더 큰 가중치가 가해져 잠재 차원들이 **독립적**이 되도록 압력을 받는다:

$$q_\phi(\mathbf{z}) \approx \prod_j q_\phi(z_j)$$

이는 각 잠재 차원이 데이터의 독립적 변동 요인(factor of variation)을 포착하도록 유도한다. 예: 잠재 차원 1 = 숫자 기울기, 잠재 차원 2 = 선 굵기.

> **$\beta$-TC-VAE** (Chen et al., 2018): TC 항에만 가중치를 두는 변형:
> $$\mathcal{L} = \mathrm{Recon} - I_q(\mathbf{x};\mathbf{z}) - \beta \cdot \mathrm{TC} - \sum_j D_{\mathrm{KL}}(q(z_j) \| p(z_j))$$
> 이렇게 하면 disentanglement는 유지하면서 reconstruction 품질 저하를 최소화할 수 있다.

---

## 11. 학습 알고리즘

### 11.1 전체 흐름

```
for epoch in epochs:
    for batch in data:
        (x_recon, μ, log_var) = VAE.forward(batch)
        (total, recon, kl) = vae_loss(batch, x_recon, μ, log_var, β)
        VAE.cleargrads()
        total.backward()
        Adam.update(VAE.params())
```

### 11.2 학습 결과

```rust
// test_training_loss_decreases 결과
epoch  1: loss = 2.3110
epoch 11: loss = 1.4522
epoch 21: loss = 1.3891
epoch 30: loss = 1.3984
// first_loss=2.3110, last_loss=1.3984 → 감소 확인 ✓
```

30 epoch 학습으로 loss가 **2.3110 → 1.3984**로 39% 감소했다.

### 11.3 Posterior Collapse

#### 현상

VAE 학습 시 흔히 발생하는 문제: KL 항이 너무 빠르게 0으로 수렴하여 잠재 변수가 무시되는 현상.

$$q_\phi(\mathbf{z}|\mathbf{x}) \to p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \forall \mathbf{x}$$

이 경우 디코더가 $\mathbf{z}$를 무시하고 입력 $\mathbf{x}$의 marginal 분포만 모델링한다. Reconstruction은 autoregressive decoder가 $\mathbf{z}$ 없이도 강력하게 수행할 수 있기 때문이다.

#### 왜 발생하는가?

학습 초기에 디코더가 아직 $\mathbf{z}$를 유용하게 활용하지 못한다. 이 때 KL 항의 gradient가 $q_\phi$를 $p(\mathbf{z})$로 밀어넣는다. 일단 $q_\phi \approx p(\mathbf{z})$가 되면, $\mathbf{z}$가 $\mathbf{x}$에 대한 정보를 담지 않으므로 디코더도 $\mathbf{z}$를 무시하는 법을 배운다 — **악순환**이 형성된다.

수학적으로, ELBO를 $I_q(\mathbf{x}; \mathbf{z})$ (mutual information) 관점에서 보면:

$$\mathcal{L} \leq \mathbb{E}[\log p_\theta(\mathbf{x}|\mathbf{z})] - I_q(\mathbf{x};\mathbf{z})$$

$I_q(\mathbf{x};\mathbf{z}) = 0$이면 $q_\phi(\mathbf{z}|\mathbf{x}) = q_\phi(\mathbf{z})$ — 인코더가 입력을 무시.

#### 해결 방법

| 방법 | 설명 | 수식 |
|------|------|------|
| **KL annealing** | $\beta$를 0에서 1로 점진 증가 | $\beta_t = \min(1, t/T_{\mathrm{warmup}})$ |
| **Free bits** | 차원별 KL에 최솟값 $\lambda$ 설정 | $\max(\mathrm{KL}_j, \lambda)$ |
| **$\delta$-VAE** | KL에 목표값 $\delta$ 설정 | $\max(D_{\mathrm{KL}}, \delta)$ |
| **Skip connection** | Encoder → Decoder 직접 연결 제거 | 아키텍처 수정 |
| **Aggressive training** | 각 M-step마다 Encoder K번 업데이트 | 학습 스케줄 변경 |

**KL annealing** 예시: 처음 10 epoch 동안 $\beta = 0$으로 시작하여 reconstruction만 학습. 디코더가 $\mathbf{z}$를 활용하는 법을 먼저 배운 후, 서서히 KL을 도입하면 collapse를 방지할 수 있다.

---

## 12. 수학적 성질

### 12.1 ELBO의 Tightness

ELBO와 로그 우도의 gap은 $D_{\mathrm{KL}}(q_\phi \| p_\theta(\mathbf{z}|\mathbf{x}))$에 의해 결정된다:

$$\mathrm{gap} = \log p_\theta(\mathbf{x}) - \mathcal{L} = D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}))$$

이 gap을 줄이는 방법:

#### Normalizing Flows (Rezende & Mohamed, 2015)

대각 가우시안 $q_\phi$에 일련의 가역 변환(flow)을 적용하여 더 풍부한 사후 분포를 구성:

$$\mathbf{z}_0 \sim q_\phi(\mathbf{z}_0|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi, \mathrm{diag}(\boldsymbol{\sigma}_\phi^2))$$
$$\mathbf{z}_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z}_0)$$

변수 변환 공식에 의해:

$$\log q_K(\mathbf{z}_K|\mathbf{x}) = \log q_0(\mathbf{z}_0|\mathbf{x}) - \sum_{k=1}^K \log\left|\det \frac{\partial f_k}{\partial \mathbf{z}_{k-1}}\right|$$

Planar flow의 경우 $f_k(\mathbf{z}) = \mathbf{z} + \mathbf{u}_k h(\mathbf{w}_k^\top \mathbf{z} + b_k)$이면:

$$\left|\det \frac{\partial f_k}{\partial \mathbf{z}}\right| = |1 + \mathbf{u}_k^\top h'(\mathbf{w}_k^\top \mathbf{z} + b_k) \mathbf{w}_k|$$

이렇게 하면 $O(d)$로 야코비안 행렬식을 계산할 수 있어 효율적이다.

#### IWAE (Burda et al., 2016)

$K$개 샘플의 importance-weighted bound:

$$\mathcal{L}_K = \mathbb{E}_{\mathbf{z}_1, \ldots, \mathbf{z}_K \sim q_\phi} \left[ \log \frac{1}{K} \sum_{k=1}^K \frac{p_\theta(\mathbf{x}, \mathbf{z}_k)}{q_\phi(\mathbf{z}_k | \mathbf{x})} \right]$$

**단조성**: $\mathcal{L}_1 \leq \mathcal{L}_K \leq \log p_\theta(\mathbf{x})$ ($K$가 클수록 더 tight한 bound).

**증명**: Jensen 부등식의 "덜 오목한" 적용. $\log$의 오목성과 $K$개 샘플의 평균에 의해, 샘플 수가 증가할수록 Jensen gap이 줄어든다. $K = 1$이면 표준 ELBO이고, $K \to \infty$이면 $\frac{1}{K}\sum_k w_k \to \mathbb{E}[w_k] = p_\theta(\mathbf{x})$이므로 $\mathcal{L}_K \to \log p_\theta(\mathbf{x})$. ∎

### 12.2 KL의 볼록성 — 완전한 증명

$D_{\mathrm{KL}}$는 $(\boldsymbol{\mu}, \boldsymbol{\gamma})$에 대해 **볼록**(convex)하다 ($\gamma_j = \log \sigma_j^2$).

$$D_{\mathrm{KL}} = \frac{1}{2}\sum_{j=1}^d (\mu_j^2 + e^{\gamma_j} - 1 - \gamma_j)$$

**1차 편미분**:

$$\frac{\partial D_{\mathrm{KL}}}{\partial \mu_j} = \mu_j, \qquad \frac{\partial D_{\mathrm{KL}}}{\partial \gamma_j} = \frac{1}{2}(e^{\gamma_j} - 1)$$

**2차 편미분** (Hessian):

$$\frac{\partial^2 D_{\mathrm{KL}}}{\partial \mu_j^2} = 1, \qquad \frac{\partial^2 D_{\mathrm{KL}}}{\partial \gamma_j^2} = \frac{1}{2} e^{\gamma_j}$$

교차 편미분: $\frac{\partial^2}{\partial \mu_j \partial \mu_k} = 0$ ($j \neq k$), $\frac{\partial^2}{\partial \gamma_j \partial \gamma_k} = 0$ ($j \neq k$), $\frac{\partial^2}{\partial \mu_j \partial \gamma_k} = 0$ (모든 $j, k$).

따라서 Hessian은 대각행렬:

$$\mathbf{H} = \mathrm{diag}\!\left(1, \ldots, 1, \frac{1}{2}e^{\gamma_1}, \ldots, \frac{1}{2}e^{\gamma_d}\right)$$

모든 대각 원소가 양수이므로 $\mathbf{H} \succ 0$ (양의 정부호). $D_{\mathrm{KL}}$는 **순볼록**(strictly convex)하다. ∎

**의미**: KL 최소화에 유일한 전역 최적해 $(\boldsymbol{\mu}^* = \mathbf{0}, \boldsymbol{\gamma}^* = \mathbf{0})$가 존재한다. 전체 ELBO는 reconstruction 항 때문에 비볼록이지만, KL 항 자체는 잘 정의된 볼록 문제다.

### 12.3 정보 이론적 해석

#### ELBO의 에너지-엔트로피 분해

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}, \mathbf{z})]}_{\text{Energy (negative)}} + \underbrace{H(q_\phi(\mathbf{z}|\mathbf{x}))}_{\text{Entropy}}$$

유도:

$$\mathcal{L} = \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x},\mathbf{z})] - \mathbb{E}_{q_\phi}[\log q_\phi(\mathbf{z}|\mathbf{x})]$$

두 번째 항은 $q_\phi$의 엔트로피: $H(q_\phi) = -\mathbb{E}_{q_\phi}[\log q_\phi]$.

- **Energy**: $q_\phi$가 joint $p_\theta(\mathbf{x},\mathbf{z})$의 높은 확률 영역에 집중하도록 유도
- **Entropy**: $q_\phi$가 넓게 퍼지도록 유도 (mode collapse 방지)

> 이 분해는 통계역학의 **자유 에너지** $F = E - TS$와 정확히 대응한다 ($T = 1$).

#### Mutual Information 분해

$$\mathbb{E}_{p_{\mathrm{data}}}[D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))] = I_q(\mathbf{x}; \mathbf{z}) + D_{\mathrm{KL}}(q_\phi(\mathbf{z}) \| p(\mathbf{z}))$$

유도:

$$\mathbb{E}_{p_{\mathrm{data}}}[D_{\mathrm{KL}}] = \mathbb{E}_{p_{\mathrm{data}}} \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\!\left[\log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p(\mathbf{z})}\right]$$

$\frac{q_\phi(\mathbf{z}|\mathbf{x})}{p(\mathbf{z})} = \frac{q_\phi(\mathbf{z}|\mathbf{x})}{q_\phi(\mathbf{z})} \cdot \frac{q_\phi(\mathbf{z})}{p(\mathbf{z})}$로 분해:

$$= \mathbb{E}_{p_{\mathrm{data}}} \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\!\left[\log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{q_\phi(\mathbf{z})}\right] + \mathbb{E}_{q_\phi(\mathbf{z})}\!\left[\log \frac{q_\phi(\mathbf{z})}{p(\mathbf{z})}\right]$$

$$= I_q(\mathbf{x}; \mathbf{z}) + D_{\mathrm{KL}}(q_\phi(\mathbf{z}) \| p(\mathbf{z}))$$

$I_q(\mathbf{x}; \mathbf{z})$가 크면 인코더가 $\mathbf{x}$에 대한 정보를 $\mathbf{z}$에 많이 담는다. KL 항을 최소화하면 이 mutual information이 줄어든다 — 이것이 reconstruction과 KL의 **정보 병목**(information bottleneck) 구조다.

---

## 13. Rate-Distortion 해석

### 13.1 VAE와 Rate-Distortion Theory의 연결

ELBO를 다시 쓰면:

$$\mathcal{L} = -\underbrace{\mathbb{E}_{q_\phi}[-\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\mathrm{Distortion}} - \underbrace{D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\mathrm{Rate}}$$

이는 Shannon의 rate-distortion 이론과 정확히 대응한다:

- **Rate** $R$: $\mathbf{z}$에 인코딩된 $\mathbf{x}$에 관한 정보량 (bits). $D_{\mathrm{KL}}(q_\phi \| p)$로 측정.
- **Distortion** $D$: 복원의 부정확성. $\mathbb{E}[-\log p_\theta(\mathbf{x}|\mathbf{z})]$로 측정.

### 13.2 Rate-Distortion 곡선

Rate-Distortion 함수 $R(D)$는 주어진 distortion level $D$ 하에서의 최소 rate:

$$R(D) = \min_{q: \mathbb{E}[d(\mathbf{x}, \hat{\mathbf{x}})] \leq D} I(\mathbf{x}; \mathbf{z})$$

$\beta$-VAE의 $\beta$는 이 곡선 위의 **동작점**(operating point)을 선택하는 Lagrange 승수에 해당한다:

$$\min_{\theta, \phi} \; D + \beta \cdot R$$

- $\beta \to 0$: $R \to \infty$, $D \to 0$ (완벽한 복원, 무한 정보)
- $\beta \to \infty$: $R \to 0$, $D \to D_{\max}$ (정보 없음, 최대 왜곡)

이 관점에서 $\beta$-VAE 학습은 **rate-distortion 최적화**의 한 형태다.

---

## 14. VAE vs 다른 생성 모델

### 14.1 비교표

| 모델 | 목적함수 | 장점 | 단점 |
|------|---------|------|------|
| **VAE** | ELBO 최대화 | 안정적 학습, 잠재 표현, 밀도 추정 | 흐릿한 샘플, posterior collapse |
| **GAN** | minimax 게임 | 선명한 샘플 | 모드 붕괴, 학습 불안정 |
| **Flow** | 정확한 $\log p(\mathbf{x})$ | 정확한 밀도, 가역 | 아키텍처 제약, 비용 |
| **Diffusion** | denoising score matching | 최고 샘플 품질, 안정 | 느린 생성 |

### 14.2 VAE 샘플이 흐릿한 이유

VAE 샘플의 **흐릿함**(blurriness)은 근본적인 수학적 원인이 있다.

MSE로 학습하면, 디코더가 출력하는 것은 $p_\theta(\mathbf{x}|\mathbf{z})$의 **평균**이다:

$$\hat{\mathbf{x}} = \mathbb{E}_{p_\theta(\mathbf{x}|\mathbf{z})}[\mathbf{x}] = \arg\min_{\hat{\mathbf{x}}} \mathbb{E}[\|\mathbf{x} - \hat{\mathbf{x}}\|^2]$$

하나의 $\mathbf{z}$에 대해 여러 가능한 $\mathbf{x}$가 있으면, 디코더는 그 **평균**을 출력 — 이것이 흐릿함의 원인이다.

예: $\mathbf{z}$가 "숫자 3"을 인코딩하지만 기울기 정보가 없으면, 디코더는 모든 기울기의 3을 평균내어 흐릿한 3을 출력한다.

GAN은 이 문제가 없다 — discriminator가 흐릿한 이미지에 "가짜" 판정을 내리므로, generator가 **선명한** 이미지를 생성하도록 압력받는다.

### 14.3 VAE의 핵심 강점

그럼에도 VAE가 중요한 이유:

1. **잠재 공간의 구조화**: 보간, 산술, disentanglement 등이 자연스럽게 가능
2. **밀도 추정**: $\log p_\theta(\mathbf{x})$의 하한 계산 가능 (GAN은 불가)
3. **안정적 학습**: mode collapse나 학습 발산 없음
4. **반지도 학습**: 잠재 변수를 라벨과 결합하면 소수 라벨로 학습 가능
5. **하이브리드**: VAE-GAN, VQ-VAE 등 다른 모델과 결합 용이

---

## 15. 잠재 공간 연산

### 15.1 잠재 공간 보간 (Linear Interpolation)

두 데이터 포인트 $\mathbf{x}_1, \mathbf{x}_2$의 잠재 표현 $\mathbf{z}_1 = \boldsymbol{\mu}(\mathbf{x}_1)$, $\mathbf{z}_2 = \boldsymbol{\mu}(\mathbf{x}_2)$ 사이를 보간하면 **의미 있는 중간 데이터**를 생성할 수 있다:

$$\mathbf{z}_t = (1-t) \mathbf{z}_1 + t \mathbf{z}_2, \quad t \in [0, 1]$$

KL regularization이 잠재 공간을 매끄럽게 정규화하므로, 선형 보간이 데이터 매니폴드 위에 머문다.

### 15.2 구형 보간 (SLERP)

잠재 공간이 가우시안이므로, 선형 보간보다 **구형 보간**(SLERP)이 더 자연스러울 수 있다:

$$\mathbf{z}_t = \frac{\sin((1-t)\Omega)}{\sin \Omega} \mathbf{z}_1 + \frac{\sin(t\Omega)}{\sin \Omega} \mathbf{z}_2$$

여기서 $\Omega = \arccos\left(\frac{\mathbf{z}_1 \cdot \mathbf{z}_2}{\|\mathbf{z}_1\|\|\mathbf{z}_2\|}\right)$.

**왜 SLERP가 더 나은가?**

고차원 가우시안에서 샘플 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$의 노름은 집중 현상에 의해 $\|\mathbf{z}\| \approx \sqrt{d}$에 집중한다:

$$\mathbb{E}[\|\mathbf{z}\|^2] = d, \qquad \mathrm{Var}[\|\mathbf{z}\|^2] = 2d$$

표준편차가 $\sqrt{2d}$이므로 상대 표준편차는 $\sqrt{2/d}$로, $d$가 크면 노름이 $\sqrt{d}$ 근처에 매우 집중된다. 즉 고차원 가우시안의 대부분의 확률 질량은 반경 $\sqrt{d}$의 **얇은 구각**(thin shell) 위에 있다.

선형 보간의 중점 $\mathbf{z}_{0.5} = \frac{1}{2}(\mathbf{z}_1 + \mathbf{z}_2)$는 이 구각 **안쪽**에 놓여 확률 밀도가 매우 낮은 영역을 통과한다. 반면 SLERP는 구각 위를 따라 보간하므로 항상 높은 밀도 영역을 유지한다.

### 15.3 잠재 공간 산술

잠재 공간에서의 벡터 산술이 의미적 연산에 대응:

$$\mathrm{Decode}(\mathbf{z}_{\text{smile}} - \mathbf{z}_{\text{neutral}} + \mathbf{z}_{\text{frown}}) \approx \text{smiling face with frown features}$$

이는 Word2Vec의 "king - man + woman = queen" 유추와 같은 원리.

왜 이것이 가능한가? KL regularization이 잠재 공간을 **지역적으로 선형**(locally linear)하게 만들기 때문이다. $\mathcal{N}(\mathbf{0}, \mathbf{I})$ prior는 원점 대칭이고 등방적(isotropic)이므로, 특정 의미적 방향이 대략 일정한 벡터에 대응하는 구조가 자연스럽게 형성된다.

---

## 16. Conditional VAE (CVAE)

### 16.1 동기

표준 VAE는 **무조건적** 생성만 가능하다. 특정 클래스(예: 숫자 "7")의 이미지를 생성하려면 잠재 공간에서 해당 영역을 찾아야 하는데, 이는 비효율적이다. CVAE (Sohn et al., 2015)는 조건 변수 $\mathbf{c}$를 도입하여 **조건부 생성**을 가능하게 한다.

### 16.2 수식

$$p_\theta(\mathbf{x} | \mathbf{c}) = \int p_\theta(\mathbf{x} | \mathbf{z}, \mathbf{c}) \, p(\mathbf{z}) \, d\mathbf{z}$$

ELBO:

$$\mathcal{L}_{\mathrm{CVAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x},\mathbf{c})}[\log p_\theta(\mathbf{x}|\mathbf{z},\mathbf{c})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x},\mathbf{c}) \| p(\mathbf{z}))$$

차이점:
- **Encoder**: $q_\phi(\mathbf{z}|\mathbf{x}, \mathbf{c})$ — 입력과 조건을 함께 인코딩
- **Decoder**: $p_\theta(\mathbf{x}|\mathbf{z}, \mathbf{c})$ — 잠재 변수와 조건으로 생성

생성 시: 원하는 $\mathbf{c}$ (예: one-hot "7")와 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$를 함께 디코더에 입력.

---

## 17. Box-Muller 변환

### 17.1 알고리즘

VAE의 reparameterization trick과 weight initialization에서 $\mathcal{N}(0,1)$ 샘플이 필요하다. 균일 분포 $U_1, U_2 \sim \mathrm{Uniform}(0,1)$로부터:

$$Z_0 = \sqrt{-2 \ln U_1} \cos(2\pi U_2)$$
$$Z_1 = \sqrt{-2 \ln U_1} \sin(2\pi U_2)$$

$Z_0, Z_1$은 독립인 $\mathcal{N}(0,1)$ 확률변수이다.

### 17.2 엄밀한 증명 — 야코비안을 이용한 변수 변환

**Step 1: 2차원 표준 정규분포의 극좌표 변환**

$(X, Y)$가 독립 $\mathcal{N}(0,1)$이면, 결합 밀도:

$$f_{X,Y}(x,y) = \frac{1}{2\pi} e^{-(x^2+y^2)/2}$$

극좌표 $x = r\cos\theta$, $y = r\sin\theta$ ($r > 0$, $0 \leq \theta < 2\pi$)로 변환. 야코비안:

$$\frac{\partial(x,y)}{\partial(r,\theta)} = \begin{pmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{pmatrix}$$

$$\left|\det \frac{\partial(x,y)}{\partial(r,\theta)}\right| = |r\cos^2\theta + r\sin^2\theta| = r$$

변환된 밀도:

$$f_{R,\Theta}(r,\theta) = f_{X,Y}(r\cos\theta, r\sin\theta) \cdot r = \frac{r}{2\pi} e^{-r^2/2}$$

$= \underbrace{r \, e^{-r^2/2}}_{f_R(r)} \cdot \underbrace{\frac{1}{2\pi}}_{f_\Theta(\theta)}$으로 **분리**되므로 $R$과 $\Theta$는 독립.

- $\Theta \sim \mathrm{Uniform}(0, 2\pi)$
- $R$의 밀도: $f_R(r) = r \, e^{-r^2/2}$ ($r > 0$) — 이는 **Rayleigh 분포**

**Step 2: $R^2$의 분포**

$S = R^2$으로 치환하면 $R = \sqrt{S}$, $\frac{dR}{dS} = \frac{1}{2\sqrt{S}}$:

$$f_S(s) = f_R(\sqrt{s}) \cdot \frac{1}{2\sqrt{s}} = \sqrt{s} \cdot e^{-s/2} \cdot \frac{1}{2\sqrt{s}} = \frac{1}{2} e^{-s/2}$$

이는 $\mathrm{Exp}(1/2)$ = $\mathrm{Gamma}(1, 2)$ = $\chi^2(2)$ 분포다.

**Step 3: 역변환으로 Box-Muller 유도**

$S \sim \mathrm{Exp}(1/2)$의 역 CDF: $F_S(s) = 1 - e^{-s/2}$이므로, $U_1 \sim \mathrm{Uniform}(0,1)$에 대해:

$$S = -2\ln(1 - U_1) \overset{d}{=} -2\ln U_1$$

($1 - U_1$과 $U_1$은 같은 분포이므로)

따라서 $R = \sqrt{S} = \sqrt{-2\ln U_1}$이고, $\Theta = 2\pi U_2$:

$$X = R\cos\Theta = \sqrt{-2\ln U_1}\cos(2\pi U_2) \sim \mathcal{N}(0,1)$$
$$Y = R\sin\Theta = \sqrt{-2\ln U_1}\sin(2\pi U_2) \sim \mathcal{N}(0,1)$$

이것이 Box-Muller 공식이다. ∎

### 17.3 구현

```rust
fn next_normal(&self) -> f64 {
    let u1 = self.next_f64();
    let u2 = self.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
```

cos 분기만 사용 ($Z_0$)하여 구현을 단순화했다. sin 분기($Z_1$)를 캐싱하면 효율을 2배로 높일 수 있지만, 교육적 구현에서는 불필요하다.

---

## 18. MNIST에서의 VAE

### 18.1 전형적 설정

| 하이퍼파라미터 | 값 |
|-------------|-----|
| input_dim | 784 (28×28) |
| hidden_dim | 256~512 |
| latent_dim | 2~32 |
| beta | 1.0 |
| optimizer | Adam (lr=1e-3) |
| epochs | 50~100 |

### 18.2 잠재 차원 2의 시각화

$d_z = 2$이면 잠재 공간을 2D 평면에 직접 시각화할 수 있다:

- **인코더 공간**: 각 숫자 클래스가 가우시안 클러스터를 형성. KL regularization 덕분에 클러스터 간 간격이 적당하고, 겹치는 영역에서는 두 숫자의 중간 형태가 나타남.
- **디코더 매니폴드**: 격자점 $\mathbf{z} \in [-3, 3]^2$를 디코딩하면 숫자가 연속적으로 변화하는 매니폴드. 경계 영역에서 숫자 간 부드러운 전환이 관찰됨.

$d_z = 32$ 등 높은 차원에서는 더 정밀한 복원이 가능하지만, 직접 시각화는 어렵다. t-SNE나 UMAP으로 투영하여 시각화할 수 있다.

---

## 19. VAE의 확장

### 19.1 VQ-VAE (van den Oord et al., 2017)

연속 잠재 공간 대신 **이산 코드북**을 사용:

$$z_q = \arg\min_{\mathbf{e}_k \in \mathrm{codebook}} \|\mathbf{z}_e - \mathbf{e}_k\|$$

KL divergence 대신 **commitment loss** + **codebook loss**를 사용. Posterior collapse 문제가 없고, PixelCNN 등과 결합하면 고품질 생성이 가능.

### 19.2 VAE-GAN

VAE의 decoder와 GAN의 generator를 공유하여 양쪽의 장점을 결합:

$$\mathcal{L} = \mathcal{L}_{\mathrm{ELBO}} + \lambda \cdot \mathcal{L}_{\mathrm{GAN}}$$

VAE의 잠재 구조 + GAN의 선명한 샘플.

### 19.3 Hierarchical VAE (Ladder VAE)

잠재 변수를 여러 층으로 쌓아 표현력을 높인다:

$$p(\mathbf{z}) = p(\mathbf{z}_L) \prod_{l=1}^{L-1} p(\mathbf{z}_l | \mathbf{z}_{l+1})$$

하위 층은 세부사항, 상위 층은 전역 구조를 포착. NVAE (Vahdat & Kohl, 2020)는 이 구조로 VAE가 GAN에 필적하는 이미지 품질을 달성했다.

---

## 20. 테스트 요약

| # | 테스트 | 검증 내용 |
|---|--------|----------|
| 1 | `test_vae_construction` | 생성, params 10개 (5W + 5b) |
| 2 | `test_encode_shape` | encode → (B, latent_dim) |
| 3 | `test_decode_shape_and_range` | decode → (B, input_dim), sigmoid ∈ (0,1) |
| 4 | `test_forward_full` | 전체 forward shapes, sigmoid 범위 |
| 5 | `test_reparameterization_gradient` | backward 후 10/10 params에 gradient |
| 6 | `test_kl_standard_normal` | μ=0, log_var=0 → KL≈0 |
| 7 | `test_vae_loss_decomposition` | total = recon + β·kl (diff < 1e-8) |
| 8 | `test_training_loss_decreases` | 30 epoch 학습, loss 39% 감소 |
| 9 | `test_sample_generation` | sample(10) shape, sigmoid 범위 |
| 10 | `test_beta_effect` | β 변화에 따른 total loss 변화 |
| 11 | `test_cleargrads` | cleargrads 후 gradient 없음 |

전체 11/11 통과.

---

## 21. 다음 단계

Step 76에서는 **GAN** (Generative Adversarial Network)을 구현할 예정이다. VAE가 **변분 추론**으로 데이터를 생성했다면, GAN은 **적대적 학습**(adversarial training)으로 생성한다 — Generator와 Discriminator의 minimax 게임.

$$\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$$
