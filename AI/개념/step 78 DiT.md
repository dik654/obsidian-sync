# Step 78: DiT (Diffusion Transformer)

Phase 5 (생성모델)의 마지막 스텝.
**핵심 아이디어**: DDPM의 MLP denoiser를 **Transformer + Adaptive Layer Normalization**으로 교체.
Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023).

---

## 1. 동기: 왜 Transformer인가?

DDPM(Step 77)의 denoiser $\epsilon_\theta(x_t, t)$는 **MLP**였다:

$$
h = \mathrm{GELU}(W_1 x_t + W_2 \mathrm{emb}(t)), \quad \hat{\epsilon} = W_{\mathrm{out}} h
$$

이 구조의 한계:

| 한계 | 설명 |
|------|------|
| **Global receptive field만 존재** | FC 레이어는 모든 입력을 한꺼번에 처리 → 공간적 locality 무시 |
| **Scaling 비효율** | 파라미터 수를 늘려도 표현력이 선형적으로만 증가 |
| **Conditioning 한계** | time embedding을 단순히 더하는 방식 → 레이어별 적응 불가 |

**Transformer**를 도입하면:
- **Self-Attention**: 토큰 간 관계를 동적으로 학습 → 이미지 패치 간 의존성 포착
- **Scalability**: ViT에서 입증된 것처럼, 모델 크기에 따라 성능이 일관되게 향상
- **Flexible conditioning**: **AdaLN**으로 모든 레이어에서 time 정보를 주입

DiT의 핵심 발견: U-Net이 아닌 순수 Transformer도 DDPM의 denoiser로 동작하며, **Gflops-FID frontier에서 U-Net을 능가**한다.

---

## 2. 전체 아키텍처

```
Input x_t ∈ ℝ^{B×D}
    │
    ▼ Patchify
[B, n_tokens, token_dim]
    │
    ▼ Linear (patch projection)
[B, n_tokens, hidden]
    │
    ▼ + Positional Embedding (학습 가능)
    │
    │    Timestep t
    │        │
    │        ▼ sinusoidal embedding
    │        ▼ MLP (GELU)
    │        ▼ conditioning c ∈ ℝ^{B×hidden}
    │        │
    ▼────────┘
DiTBlock × N
    │
    ▼ Final AdaLN
    ▼ Linear (token_dim 복원)
[B, n_tokens, token_dim]
    │
    ▼ Unpatchify
Output ε̂ ∈ ℝ^{B×D}
```

---

## 3. Patchify / Unpatchify

이미지에서 DiT는 ViT처럼 이미지를 **패치**로 분할한다. 우리 구현에서는 1D 데이터를 토큰으로 분할:

$$
x \in \mathbb{R}^{B \times D} \xrightarrow{\mathrm{patchify}} x \in \mathbb{R}^{B \times N \times d}
$$

여기서 $N = \frac{D}{d}$, $d$는 토큰 차원(token_dim).

**예시**: $D = 16$, $N = 4$이면 $d = 4$.

$$
\underbrace{[x_1, x_2, x_3, x_4}_{토큰\,1}, \underbrace{x_5, x_6, x_7, x_8}_{토큰\,2}, \underbrace{x_9, \ldots, x_{12}}_{토큰\,3}, \underbrace{x_{13}, \ldots, x_{16}}_{토큰\,4}]
$$

Patchify 후 Linear로 hidden 차원에 사영:

$$
h_i = W_{\mathrm{proj}} \cdot p_i + b_{\mathrm{proj}}, \quad p_i \in \mathbb{R}^d,\, h_i \in \mathbb{R}^{h}
$$

최종 출력은 역순으로:

$$
\mathbb{R}^{B \times N \times d} \xrightarrow{\mathrm{reshape}} \mathbb{R}^{B \times D}
$$

**이미지에서의 Patchify**: 실제 DiT에서는 $256 \times 256 \times 3$ 이미지를 $p \times p$ 패치로 분할.
$p = 2$이면 VAE latent $32 \times 32 \times 4$에서 $16 \times 16 = 256$개 토큰 생성 → sequence length = 256.

---

## 4. Positional Embedding

토큰의 위치 정보를 주입하기 위한 학습 가능 파라미터:

$$
E_{\mathrm{pos}} \in \mathbb{R}^{1 \times N \times h}
$$

Patch projection 후 broadcasting으로 더함:

$$
h_i \leftarrow h_i + E_{\mathrm{pos}}[i]
$$

**왜 학습 가능 embedding인가?**

| 방식 | 장점 | 단점 |
|------|------|------|
| **Sinusoidal (고정)** | 긴 시퀀스 일반화 | 최적이 아닐 수 있음 |
| **학습 가능** | 데이터에 맞게 최적화 | 고정 길이만 지원 |

DiT 원 논문은 학습 가능 embedding을 사용한다. 우리 구현에서는 4개 토큰만 있으므로 학습 가능 embedding이 자연스럽다. Xavier 초기화:

$$
E_{\mathrm{pos}} \sim \mathcal{N}\!\left(0, \frac{1}{h}\right)
$$

---

## 5. Adaptive Layer Normalization (AdaLN)

### 5.1 일반 LayerNorm vs AdaLN

**일반 LayerNorm** (Step 65):

$$
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

여기서 $\gamma, \beta$는 **학습 가능하지만 모든 샘플에 대해 동일** (input-independent).

**AdaLN**: $\gamma, \beta$를 conditioning 벡터 $c$로부터 **per-sample로 생성**:

$$
\gamma = W_\gamma c + b_\gamma, \quad \beta = W_\beta c + b_\beta
$$

그러면:

$$
\mathrm{AdaLN}(x, c) = (1 + \gamma) \odot \mathrm{LN}(x) + \beta
$$

여기서 $(1 + \gamma)$를 사용하는 이유: 초기화 시 $W_\gamma = 0$이면 scale = 1, 즉 표준 LayerNorm에서 시작한다. 이는 학습 안정성을 높인다.

### 5.2 AdaLN의 핵심 통찰

**왜 per-sample affine parameter인가?**

DDPM의 MLP에서 time conditioning은 **additive injection**이었다:

$$
h = \mathrm{act}(W_x x_t + W_t \mathrm{emb}(t))
$$

이 방식은 time 정보가 activation 전에 **한 번만** 주입된다. 반면 AdaLN은:

1. **모든 레이어에서** conditioning이 주입됨 (각 DiTBlock에 독립적인 $\gamma, \beta$)
2. **Normalization 자체를 제어** → activation의 scale과 shift를 직접 조절
3. **Per-sample**: 배치 내 각 샘플이 서로 다른 timestep을 가질 때, 각각 다른 normalization 적용

이는 **Feature-wise Linear Modulation (FiLM)**의 특수한 경우이다:

$$
\mathrm{FiLM}(x, c) = \gamma(c) \odot x + \beta(c)
$$

AdaLN = FiLM applied to LayerNorm output.

### 5.3 구현 트릭: $(1 + \gamma)x = x + \gamma x$

직접 $(1 + \gamma) \odot x$를 계산하려면 스칼라 1과 Variable $\gamma$를 더해야 한다. 우리 프레임워크에서는 분배법칙을 사용:

$$
(1 + \gamma) \odot x_{\mathrm{norm}} = x_{\mathrm{norm}} + \gamma \odot x_{\mathrm{norm}}
$$

```rust
// 분배법칙: (1+γ)*x_norm = x_norm + γ*x_norm
&(&x_norm + &(gamma * &x_norm)) + beta
```

### 5.4 브로드캐스팅 차원

$c \in \mathbb{R}^{B \times h}$에서 생성된 $\gamma, \beta$를 $x \in \mathbb{R}^{B \times T \times h}$에 적용하려면:

$$
\gamma = \mathrm{reshape}(W_\gamma c, [B, 1, h])
$$

가운데 축(시퀀스 차원)에 1을 넣어 broadcasting으로 모든 토큰에 동일한 scale/shift를 적용.

---

## 6. DiT Block 구조

### 6.1 아키텍처

```
Input x ∈ ℝ^{B×T×h}     Conditioning c ∈ ℝ^{B×h}
    │                         │
    │    ┌────────────────────┤
    │    │  γ₁ = Linear(c)   │  [B, 1, h]
    │    │  β₁ = Linear(c)   │  [B, 1, h]
    │    │                    │
    ▼    ▼                    │
 AdaLN(x, γ₁, β₁)           │
    │                         │
    ▼                         │
 SelfAttention                │
    │                         │
    + ←── x (residual)        │
    │                         │
    │    ┌────────────────────┤
    │    │  γ₂ = Linear(c)   │
    │    │  β₂ = Linear(c)   │
    │    │                    │
    ▼    ▼                    │
 AdaLN(h, γ₂, β₂)           │
    │                         │
    ▼                         │
 FFN (GELU)                  │
    │                         │
    + ←── h (residual)        │
    │                         │
    ▼
Output ∈ ℝ^{B×T×h}
```

### 6.2 표준 Transformer Block과의 비교

| 구성요소 | Standard (GPT, BERT) | DiT Block |
|----------|---------------------|-----------|
| **Norm 방식** | `LN(x)` 고정 γ,β | `AdaLN(x,c)` adaptive γ,β |
| **Attention** | Causal (GPT) / Bidirectional (BERT) | **Bidirectional** (생성 모델) |
| **FFN** | GELU | GELU (동일) |
| **Conditioning** | 없음 또는 cross-attention | **AdaLN** |
| **Residual** | Pre-norm | Pre-norm (동일) |

### 6.3 왜 Bidirectional Attention인가?

GPT는 autoregressive하므로 **causal mask** 필수. 하지만 DiT는:
- 전체 노이즈 이미지를 **한 번에** denoising
- 모든 패치가 서로를 참조해야 최적의 noise 예측 가능
- $\epsilon_\theta(x_t, t)$에서 각 패치의 noise는 다른 모든 패치의 정보에 의존

따라서 `SelfAttention::new_with_mask(hidden_dim, n_heads, 0.0, seed, false)` — `false`로 causal mask 비활성화.

### 6.4 파라미터 수

DiTBlock 1개:
- SelfAttention: $W_Q, W_K, W_V, W_O$ 각 2(W+b) = **8**
- FFN: $W_{\mathrm{fc}}, W_{\mathrm{proj}}$ 각 2 = **4**
- AdaLN: $W_{\gamma_1}, W_{\beta_1}, W_{\gamma_2}, W_{\beta_2}$ 각 2 = **8**
- **소계: 20 params/block**

---

## 7. DiT 전체 Forward Pass 유도

### 7.1 단계별 텐서 흐름

입력 $x_t \in \mathbb{R}^{B \times D}$, timestep $t \in \{0, \ldots, T-1\}^B$:

**Step 1: Patchify**
$$
X = \mathrm{reshape}(x_t, [B, N, d])
$$

**Step 2: Patch Projection**
$$
X_{\mathrm{flat}} = \mathrm{reshape}(X, [BN, d])
$$
$$
H = \mathrm{reshape}(W_{\mathrm{proj}} X_{\mathrm{flat}} + b_{\mathrm{proj}}, [B, N, h])
$$

**Step 3: Positional Embedding**
$$
H \leftarrow H + E_{\mathrm{pos}}
$$

**Step 4: Time Conditioning**
$$
e = \mathrm{SinusoidalEmbed}(t) \in \mathbb{R}^{B \times h}
$$
$$
c = W_2 \cdot \mathrm{GELU}(W_1 \cdot e + b_1) + b_2 \in \mathbb{R}^{B \times h}
$$

**Step 5: DiT Blocks (×N)**
$$
H \leftarrow \mathrm{DiTBlock}_i(H, c), \quad i = 1, \ldots, L
$$

**Step 6: Final AdaLN**
$$
\gamma_f = \mathrm{reshape}(W_{\gamma_f} c + b_{\gamma_f}, [B, 1, h])
$$
$$
\beta_f = \mathrm{reshape}(W_{\beta_f} c + b_{\beta_f}, [B, 1, h])
$$
$$
H = \mathrm{LN}(H) + \gamma_f \odot \mathrm{LN}(H) + \beta_f
$$

**Step 7: Output Projection**
$$
O_{\mathrm{flat}} = W_{\mathrm{out}} \cdot \mathrm{reshape}(H, [BN, h]) + b_{\mathrm{out}} \in \mathbb{R}^{BN \times d}
$$
$$
O = \mathrm{reshape}(O_{\mathrm{flat}}, [B, N, d])
$$

**Step 8: Unpatchify**
$$
\hat{\epsilon} = \mathrm{reshape}(O, [B, D])
$$

### 7.2 전체 파라미터 수

| 컴포넌트 | params |
|----------|--------|
| patch_proj $(W, b)$ | 2 |
| pos_emb | 1 |
| t_mlp1, t_mlp2 $(W, b)$ × 2 | 4 |
| DiTBlock × 2 | 40 |
| final_adaln_gamma, final_adaln_beta $(W, b)$ × 2 | 4 |
| final_proj $(W, b)$ | 2 |
| **합계** | **53** |

---

## 8. AdaLN의 대안들: Conditioning 방식 비교

DiT 논문은 여러 conditioning 방식을 비교 실험했다:

### 8.1 In-Context Conditioning

Timestep $t$를 **추가 토큰**으로 시퀀스에 append:

$$
[\mathrm{token}_1, \ldots, \mathrm{token}_N, \mathrm{emb}(t)]
$$

장점: 아키텍처 변경 없음 (표준 Transformer).
단점: 모든 레이어에서 동일한 conditioning → 층별 적응 불가.

### 8.2 Cross-Attention

Stable Diffusion의 U-Net에서 사용하는 방식. $Q$는 이미지 feature, $K, V$는 conditioning에서:

$$
\mathrm{Attn}(Q_x, K_c, V_c) = \mathrm{softmax}\!\left(\frac{Q_x K_c^\top}{\sqrt{d_k}}\right) V_c
$$

장점: 풍부한 conditioning (텍스트 등 시퀀스 조건).
단점: 추가 파라미터 ($W_K^c, W_V^c$), 계산량 증가.

### 8.3 AdaLN (DiT 선택)

장점:
- **가장 적은 추가 파라미터** (Linear 4개/block)
- **FID 최고 성능** (DiT 논문 실험 결과)
- Timestep처럼 **단일 벡터** conditioning에 최적

단점: 시퀀스 형태 conditioning (텍스트 프롬프트)에는 부적합 → cross-attention 필요.

### 8.4 AdaLN-Zero

DiT 논문의 최종 선택. AdaLN에 **gate parameter** $\alpha$를 추가:

$$
x \leftarrow x + \alpha \cdot \mathrm{SubLayer}(\mathrm{AdaLN}(x, c))
$$

$\alpha$도 conditioning에서 생성, **0으로 초기화** → 학습 초기에 각 block이 identity function처럼 동작 → 안정적 학습.

우리 구현에서는 교육적 단순화를 위해 $\alpha$ gate를 **생략**했다. 핵심 개념인 per-sample adaptive normalization은 AdaLN만으로 충분히 전달된다.

---

## 9. Time Embedding

DDPM과 동일한 sinusoidal positional encoding을 time embedding으로 사용:

$$
\mathrm{emb}(t)_{2k} = \sin\!\left(\frac{t}{10000^{2k/d}}\right), \quad \mathrm{emb}(t)_{2k+1} = \cos\!\left(\frac{t}{10000^{2k/d}}\right)
$$

이 고정 embedding을 MLP로 변환:

$$
c = W_2 \cdot \mathrm{GELU}(W_1 \cdot \mathrm{emb}(t) + b_1) + b_2
$$

**Sinusoidal → MLP 파이프라인의 역할**:

| 단계 | 역할 |
|------|------|
| Sinusoidal | 각 timestep에 고유한 주파수 패턴 부여 → 연속적 위치 인코딩 |
| $W_1 + \mathrm{GELU}$ | 비선형 변환으로 주파수 혼합 → 학습 가능한 time feature |
| $W_2$ | 원하는 차원으로 사영 → conditioning 벡터 $c$ |

$c$는 모든 DiTBlock에서 **공유**된다 (각 block의 AdaLN Linear가 독립적으로 $\gamma, \beta$를 생성하므로 block별 다른 modulation 가능).

---

## 10. 원 논문의 DiT vs U-Net

### 10.1 U-Net 기반 Diffusion (DDPM, Stable Diffusion)

```
x_t → Encoder (downsampling) → Bottleneck → Decoder (upsampling) → ε̂
                        ↕ skip connections ↕
```

- **Inductive bias**: 공간적 계층 구조 (CNN 기반)
- **Conditioning**: cross-attention 또는 additive
- **스케일링**: 채널 수 증가로 → 비효율적 FLOPs 사용

### 10.2 DiT (Transformer 기반)

```
x_t → Patchify → Transformer Blocks → Unpatchify → ε̂
```

- **Inductive bias 최소**: 순수 attention으로 관계 학습
- **Conditioning**: AdaLN — 효율적이고 성능 우수
- **스케일링**: depth/width 증가 → 일관된 FID 개선

### 10.3 Scaling Law

DiT 논문의 핵심 결과:

| 모델 | Gflops | FID-50K ↓ |
|------|--------|-----------|
| DiT-S/2 | 6 | 68.4 |
| DiT-B/2 | 24 | 43.5 |
| DiT-L/2 | 80 | 24.2 |
| DiT-XL/2 | 119 | **2.27** |

**Gflops vs FID가 거의 log-linear** → 계산량을 늘리면 일관되게 품질 향상. U-Net은 이런 clean한 scaling law를 보이지 않는다.

이는 LLM에서 관찰된 scaling law (Kaplan et al.)의 이미지 생성 버전이다. DiT가 중요한 이유: **Transformer의 scaling property가 생성 모델에도 적용됨**을 입증.

---

## 11. Latent DiT: VAE와의 결합

실제 DiT(논문)는 **pixel space**가 아닌 **latent space**에서 동작한다:

$$
\underbrace{x \xrightarrow{\mathrm{VAE\,Encoder}} z}_{\text{한 번만}} \xrightarrow{\mathrm{DiT}} \hat{z}_0 \xrightarrow{\mathrm{VAE\,Decoder}} \hat{x}
$$

1. VAE encoder로 이미지 $256 \times 256 \times 3$을 latent $32 \times 32 \times 4$로 압축
2. Latent에서 DiT로 diffusion 수행
3. 생성된 latent를 VAE decoder로 이미지 복원

**장점**:
- **계산 효율**: $256^2 \times 3 = 196608$ → $32^2 \times 4 = 4096$ (48배 압축)
- **의미적 공간**: pixel noise가 아닌 semantic feature 수준에서 denoising
- 이것이 **Stable Diffusion** (LDM)과 동일한 패러다임

우리 구현에서는 VAE 없이 raw data에 직접 적용하지만, 개념적으로 Step 75(VAE)의 latent space에서 동작시키는 것도 가능하다.

---

## 12. DiT 변형들과 후속 연구

### 12.1 DiT의 영향

DiT는 이미지 생성의 **패러다임 전환**을 이끌었다:

| 모델 | 년도 | 구조 | Conditioning |
|------|------|------|-------------|
| DDPM | 2020 | U-Net | Additive |
| Stable Diffusion 1/2 | 2022 | U-Net + Cross-Attn | Cross-Attention |
| **DiT** | 2023 | **Transformer + AdaLN** | **AdaLN** |
| Stable Diffusion 3 | 2024 | DiT 기반 (MM-DiT) | AdaLN + Cross-Attn |
| FLUX | 2024 | DiT 기반 | AdaLN |
| Sora | 2024 | Spatial-temporal DiT | AdaLN |

### 12.2 MM-DiT (Multimodal DiT)

Stable Diffusion 3에서 사용. Text와 Image 토큰을 **하나의 시퀀스로 concat**하여 joint attention:

$$
[\mathrm{text\,tokens}, \mathrm{image\,tokens}] \xrightarrow{\mathrm{Attention}} [\mathrm{text'}, \mathrm{image'}]
$$

각 modality에 **독립적인 AdaLN** 적용.

### 12.3 Sora (Spatial-Temporal DiT)

비디오 생성으로 확장. 3D 패치 (spatial + temporal):

$$
\mathrm{Video} \in \mathbb{R}^{T \times H \times W \times 3} \xrightarrow{\mathrm{3D\,patchify}} \text{tokens}
$$

DiT의 scalability가 비디오까지 확장 가능함을 보여준 사례.

---

## 13. Diffusion Framework (DDPM 복습)

DiT의 **Transformer 부분**만 새롭고, diffusion framework는 DDPM(Step 77)과 **완전히 동일**하다.

### Forward Diffusion

$$
q(x_t | x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t) I\right)
$$

한 번에 샘플링:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### Training Objective

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

DDPM: $\epsilon_\theta$ = MLP
DiT: $\epsilon_\theta$ = **Transformer + AdaLN**

**손실 함수가 동일**하므로 `dit_loss`와 `ddpm_loss`는 구조가 같다:

```rust
pub fn dit_loss(dit: &DiT, x_0: &Variable, t: &[usize]) -> Variable {
    let (x_t, noise) = dit.q_sample(x_0, t);
    let predicted = dit.forward(&x_t, t);
    mean_squared_error(&predicted, &noise)
}
```

### Reverse Sampling

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\right) + \sigma_t z
$$

$x_T \sim \mathcal{N}(0, I)$에서 시작하여 $T \to 0$으로 iterative denoising.

---

## 14. 코드 발췌

### DiTBlock: AdaLN + Attention + FFN

```rust
pub struct DiTBlock {
    attn: SelfAttention,       // bidirectional (causal=false)
    mlp_fc: Linear,            // hidden → 4*hidden
    mlp_proj: Linear,          // 4*hidden → hidden
    adaln_gamma1: Linear,      // c → γ₁
    adaln_beta1: Linear,       // c → β₁
    adaln_gamma2: Linear,      // c → γ₂
    adaln_beta2: Linear,       // c → β₂
    hidden_dim: usize,
}
```

### AdaLN 구현

```rust
fn adaln(&self, x: &Variable, gamma: &Variable, beta: &Variable) -> Variable {
    let ones = Variable::new(ArrayD::ones(IxDyn(&[self.hidden_dim])));
    let zeros = Variable::new(ArrayD::zeros(IxDyn(&[self.hidden_dim])));
    let x_norm = layer_norm(x, &ones, &zeros, 1e-5);
    // (1 + γ) * x_norm + β = x_norm + γ * x_norm + β
    &(&x_norm + &(gamma * &x_norm)) + beta
}
```

### DiTBlock Forward

```rust
pub fn forward(&self, x: &Variable, c: &Variable) -> Variable {
    // Modulation params from conditioning
    let gamma1 = reshape(&self.adaln_gamma1.forward(c), &[b, 1, self.hidden_dim]);
    let beta1 = reshape(&self.adaln_beta1.forward(c), &[b, 1, self.hidden_dim]);
    // Sub-block 1: AdaLN → Attention → Residual
    let h = x + &self.attn.forward(&self.adaln(x, &gamma1, &beta1));
    // Sub-block 2: AdaLN → FFN → Residual
    let h_mod = self.adaln(&h, &gamma2, &beta2);
    let ffn_out = self.mlp_proj.forward(&gelu(&self.mlp_fc.forward(&h_mod_2d)));
    &h + &ffn_out
}
```

### DiT Forward (전체)

```rust
pub fn forward(&self, x_t: &Variable, t: &[usize]) -> Variable {
    // 1. Patchify: [B, D] → [B, N, d]
    let x = reshape(x_t, &[batch, self.n_tokens, self.token_dim]);
    // 2. Patch projection → [B, N, h]
    let h = reshape(&self.patch_proj.forward(&x_flat), &[batch, self.n_tokens, self.hidden_dim]);
    // 3. + positional embedding
    let h = &h + &self.pos_emb;
    // 4. Time conditioning → c ∈ [B, h]
    let c = self.t_mlp2.forward(&gelu(&self.t_mlp1.forward(&t_emb)));
    // 5. DiT blocks
    for block in &self.blocks { h = block.forward(&h, &c); }
    // 6. Final AdaLN + projection → [B, N, d]
    // 7. Unpatchify → [B, D]
    reshape(&out, &[batch, self.data_dim])
}
```

---

## 15. 학습 결과

```
=== DiT 학습 ===
  epoch   1: loss = 24.184771
  epoch  41: loss = 7.989466
  epoch  81: loss = 7.407999
  epoch 121: loss = 8.541864
  epoch 161: loss = 6.845520
  epoch 200: loss = 7.859448
  first 5 avg = 17.310639, last 5 avg = 7.785999
```

| 지표 | 값 |
|------|-----|
| 초기 loss (5 epoch 평균) | 17.31 |
| 최종 loss (5 epoch 평균) | 7.79 |
| 감소율 | 55% |
| 총 파라미터 | 53 |
| gradient 전파 | 53/53 (100%) |

DDPM(Step 77)의 최종 loss 1.90 대비 DiT가 높은 이유:
- **data_dim이 4배** (DDPM: 4, DiT: 16) → 예측해야 할 noise 차원이 4배
- **Transformer overhead**: attention, AdaLN 등 복잡한 구조가 toy scale에서는 오히려 불리
- 실제 대규모 실험에서는 DiT가 U-Net/MLP를 압도 (Section 10 참조)

---

## 16. 테스트 설계

| # | 테스트 | 검증 내용 |
|---|--------|----------|
| 1 | `test_dit_construction` | 53개 파라미터, data_dim/timesteps 설정 |
| 2 | `test_dit_block_forward` | DiTBlock shape 보존 [B,T,D] → [B,T,D] |
| 3 | `test_adaln_conditioning` | 다른 $c$ → 다른 출력 (adaptive 동작 확인) |
| 4 | `test_patchify_unpatchify` | 입출력 shape 동일 [B,16] → forward → [B,16] |
| 5 | `test_dit_forward` | 전체 forward, predicted noise shape 확인 |
| 6 | `test_dit_loss` | loss > 0, finite |
| 7 | `test_gradient_flow` | backward 후 53/53 params에 gradient |
| 8 | `test_sampling` | T=20 reverse process, finite samples |
| 9 | `test_ddpm_vs_dit_api` | DDPM과 동일한 diffusion API 공유 확인 |
| 10 | `test_training_convergence` | 200 epoch loss 감소 + sample 생성 |

---

## 17. 생성모델 Phase 5 계보

```
Step 75: VAE        ← 잠재 공간 학습 (encoder-decoder)
    │
Step 76: GAN        ← 적대적 학습 (generator vs discriminator)
    │
Step 77: DDPM       ← 확산 모델 기초 (MLP denoiser)
    │
Step 78: DiT        ← Transformer denoiser + AdaLN ← 현재
```

| 모델 | Denoiser | Conditioning | Scalability |
|------|----------|-------------|-------------|
| VAE | Decoder (FC) | Latent $z$ | 제한적 |
| GAN | Generator (FC) | Latent $z$ | 불안정 |
| DDPM | MLP | Additive time | 제한적 |
| **DiT** | **Transformer** | **AdaLN** | **Log-linear** |

**DiT의 역사적 의의**: 생성 모델에서 **"Transformer가 답"**이라는 LLM의 교훈이 **이미지/비디오 생성에도 적용됨**을 입증. Sora, Stable Diffusion 3, FLUX 등 2024-2025 최신 모델들의 공통 backbone.
