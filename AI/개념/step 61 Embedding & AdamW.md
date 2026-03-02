## Step 61: Embedding과 AdamW — nanoGPT의 첫 부품

Step 60까지의 DeZero 책이 끝나고, 이제부터는 Transformer(nanoGPT) 구현을 향해 필요한 부품을 하나씩 만든다. 첫 번째는 **Embedding**(토큰 → 벡터)과 **AdamW**(Transformer 표준 옵티마이저).

---

### 한마디 직관

- **Embedding = 사전(Dictionary)** — 단어 ID로 찾아보면 해당 의미 벡터가 나온다
- **AdamW = Adam + 올바른 다이어트** — 모델이 가중치를 과도하게 키우지 않도록 감쇠를 분리 적용

---

### Embedding: 정수 인덱스 → 밀집 벡터

#### 왜 필요한가?

신경망은 **실수 벡터**를 입력으로 받는다. 하지만 자연어의 입력은 **정수 토큰 ID**(예: "cat"=3, "dog"=5)다. 정수 3은 "cat"이 "dog"보다 작다는 의미가 아니다 — 단지 사전에서의 번호일 뿐이다. 이 의미 없는 정수를 의미 있는 벡터 공간으로 변환하는 것이 Embedding이다.

$$\text{Embedding}(3) = W[3] = [0.2, -0.5, 0.8, \ldots]$$

내부적으로는 **(vocab_size, embed_dim)** 크기의 가중치 행렬 $W$에서 해당 행을 꺼내는 **룩업(lookup)** 연산이다.

**용어 정리:**
- **vocab_size**: 어휘 크기. 모델이 알고 있는 고유 토큰(단어/서브워드)의 수. GPT-2는 50257개
- **embed_dim**: 임베딩 벡터의 차원. 각 토큰을 표현하는 벡터의 길이. GPT-2는 768차원
- **밀집 벡터(dense vector)**: 대부분의 원소가 0이 아닌 벡터. [0.2, -0.5, 0.8]처럼 모든 차원에 의미 있는 값이 있다
- **희소 벡터(sparse vector)**: 대부분의 원소가 0인 벡터. one-hot [0, 0, 0, 1, 0]처럼 하나만 1이고 나머지가 전부 0

#### one-hot 인코딩과의 관계

가장 단순한 정수→벡터 변환은 **one-hot 인코딩**이다. 인덱스 $i$를 vocab_size 차원의 벡터로 변환하되, $i$번째만 1이고 나머지는 0으로 채운다:

$$e_3 = [0, 0, 0, 1, 0, \ldots] \quad \text{(vocab\_size 차원)}$$

이 one-hot 벡터에 가중치 행렬 $W$를 곱하면:

$$e_3 \cdot W = [0, 0, 0, 1, 0, \ldots] \cdot \begin{bmatrix} w_{00} & w_{01} \\ w_{10} & w_{11} \\ w_{20} & w_{21} \\ w_{30} & w_{31} \\ w_{40} & w_{41} \end{bmatrix} = [w_{30}, w_{31}] = W[3]$$

3번 행만 살아남는다. 즉 **one-hot @ W는 W에서 해당 행을 꺼내는 것과 수학적으로 동일**하다.

그렇다면 왜 one-hot을 만들지 않고 직접 인덱싱하는가?
- vocab_size = 50257이면, one-hot 벡터 하나가 50257차원 → batch × seq_len개를 만들면 메모리 폭발
- 행렬곱 $e_i \cdot W$를 하면 대부분 0과의 곱셈 → 연산 낭비
- 직접 인덱싱(`W[i]`)으로 한 줄 복사하면 $O(\text{embed\_dim})$으로 끝

---

#### 순전파: 룩업 (lookup)

인덱스 배열의 각 원소에 해당하는 $W$의 행 벡터를 복사:

```
W = [[0.1, 0.2],    idx = [2, 0, 2]
     [0.3, 0.4],
     [0.5, 0.6]]    → output = [[0.5, 0.6],   ← W[2]
                                 [0.1, 0.2],   ← W[0]
                                 [0.5, 0.6]]   ← W[2]
```

출력 shape: `idx.shape() + [embed_dim]`. 예를 들어 `idx = (batch=2, seq=3)` → 출력 `(2, 3, embed_dim)`.

---

#### 역전파: scatter-add

**scatter-add란?** "흩뿌려서(scatter) 더한다(add)"는 의미. 순전파에서 $W$의 특정 행들을 **모아왔으므로(gather)**, 역전파에서는 기울기를 원래 위치로 **되돌려 뿌린다(scatter)**.

유도 과정. 순전파를 수식으로 쓰면:

$$y_k = W[\text{idx}[k]] \quad \text{(k번째 출력 = idx[k]번째 행)}$$

손실 $L$에 대한 $W$의 기울기를 구하면:

$$\frac{\partial L}{\partial W[j]} = \sum_{k : \text{idx}[k] = j} \frac{\partial L}{\partial y_k}$$

즉, **인덱스 $j$를 사용한 모든 위치 $k$에서 온 기울기($\frac{\partial L}{\partial y_k}$)를 합산**한다. 이것이 scatter-add다.

구체적인 예:

```
idx = [2, 0, 2]   (인덱스 2가 두 번 등장)

gy = [[g₀₀, g₀₁],     ← y₀의 기울기 (idx=2에서 옴)
      [g₁₀, g₁₁],     ← y₁의 기울기 (idx=0에서 옴)
      [g₂₀, g₂₁]]     ← y₂의 기울기 (idx=2에서 옴)

gW[0] = [g₁₀, g₁₁]              ← idx=0은 1번 등장 → 그대로
gW[1] = [0.0, 0.0]               ← idx=1은 미사용 → 기울기 0
gW[2] = [g₀₀+g₂₀, g₀₁+g₂₁]     ← idx=2는 2번 등장 → 합산
```

**핵심**: 동일 인덱스가 여러 번 등장하면 기울기가 **누적**된다. 미사용 인덱스의 기울기는 0. 이것은 one-hot @ W의 역전파(`gW = one_hot.T @ gy`)와 정확히 동일한 결과다.

---

#### Xavier 초기화

Embedding의 가중치 $W$를 어떻게 초기화하는가? 모든 원소를 0으로 하면 모든 토큰이 같은 벡터 → 대칭성이 깨지지 않아 학습이 안 된다. 너무 크면 값이 폭발하고, 너무 작으면 신호가 사라진다.

**Xavier 초기화** (Glorot & Bengio, 2010)는 입출력 차원에 맞춰 분산을 조절한다:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{1}{n_{\text{in}}}\right)$$

**유도**: 입력 $x$의 분산이 $\text{Var}(x)$이고, 가중치 $W$의 분산이 $\text{Var}(W)$일 때, 출력 $y = \sum_{i=1}^{n} x_i w_i$의 분산은:

$$\text{Var}(y) = n_{\text{in}} \cdot \text{Var}(x) \cdot \text{Var}(W)$$

출력의 분산을 입력과 동일하게 유지하려면 $\text{Var}(y) = \text{Var}(x)$:

$$n_{\text{in}} \cdot \text{Var}(W) = 1 \quad \Rightarrow \quad \text{Var}(W) = \frac{1}{n_{\text{in}}}$$

따라서 표준편차 $\sigma = \sqrt{1/n_{\text{in}}}$인 정규분포에서 샘플링. Embedding에서 $n_{\text{in}} = \text{embed\_dim}$이므로:

```rust
let scale = (1.0 / embed_dim as f64).sqrt();
let w_ij = normal(0, 1) * scale;   // N(0, 1/embed_dim)
```

---

#### 구현

```rust
struct EmbeddingFn {
    vocab_size: usize,
    idx_data: Vec<usize>,   // 순전파 시 저장 (backward의 scatter 위치 결정)
    input_shape: Vec<usize>,
}

impl Function for EmbeddingFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let w = &xs[0];  // (vocab_size, embed_dim)
        let idx = &xs[1]; // 정수 인덱스 (f64로 전달)
        let embed_dim = w.shape()[1];
        let indices: Vec<usize> = idx.iter().map(|&v| v as usize).collect();

        // 각 인덱스의 행을 순서대로 복사
        let mut out_data = Vec::with_capacity(indices.len() * embed_dim);
        for &i in &indices {
            out_data.extend(w.slice(s![i, ..]).iter());
        }
        // 출력 shape: idx.shape() + [embed_dim]
        let mut out_shape = idx.shape().to_vec();
        out_shape.push(embed_dim);
        vec![ArrayD::from_shape_vec(IxDyn(&out_shape), out_data).unwrap()]
    }

    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gy = &gys[0];
        let gy_data = gy.data();
        let embed_dim = gy_data.shape()[gy_data.ndim() - 1];

        // scatter-add: gW[idx[i]] += gy[i]
        let mut gw = ArrayD::zeros(IxDyn(&[self.vocab_size, embed_dim]));
        let gy_2d = gy_data.into_shape((self.idx_data.len(), embed_dim)).unwrap();
        for (i, &idx) in self.idx_data.iter().enumerate() {
            gw.slice_mut(s![idx, ..]) += &gy_2d.slice(s![i, ..]);
        }
        vec![
            Variable::new(gw),
            Variable::new(ArrayD::zeros(IxDyn(&self.input_shape))),  // idx의 기울기는 0
        ]
    }
}
```

**설계 포인트:**
- **f64로 정수 전달**: Function trait의 인터페이스가 `ArrayD<f64>`이므로, 정수를 f64로 인코딩하고 내부에서 `as usize`로 변환. f64는 정수를 $2^{53}$까지 오차 없이 표현하므로 vocab_size < 9조이면 안전
- **idx_data 저장**: backward에서 scatter 위치를 알아야 하므로, forward 시점에 인덱스를 미리 캐시
- **idx에 대한 기울기 = 0**: 정수 인덱스는 이산값(discrete value)이므로 미분 불가능. 기울기 개념 자체가 성립하지 않음. 형식적으로 0을 반환

---

#### Embedding 레이어

```rust
pub struct Embedding {
    w: Variable,  // (vocab_size, embed_dim) — 학습 가능한 룩업 테이블
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        // Xavier 초기화: N(0, 1/embed_dim)
    }

    pub fn forward(&self, idx: &Variable) -> Variable {
        embedding(&self.w, idx)  // W와 idx를 EmbeddingFn에 전달
    }
}
```

2D 인덱스도 지원: `(batch, seq_len)` → `(batch, seq_len, embed_dim)`. Transformer에서 배치 입력 처리에 필수적이다.

---

### AdamW: 분리된 가중치 감쇠

#### 가중치 감쇠(Weight Decay)란?

모델이 학습 데이터에 과적합(overfitting)하면, 가중치가 불필요하게 커지는 경향이 있다. 큰 가중치 → 작은 입력 변화에도 출력이 크게 변동 → 일반화 성능 하락.

**가중치 감쇠**는 매 업데이트마다 가중치를 일정 비율만큼 줄이는 것이다:

$$p \leftarrow p - \mathrm{lr} \cdot g - \mathrm{lr} \cdot \lambda \cdot p$$

$\lambda$가 감쇠율(decay rate). 매 스텝마다 $p$의 크기가 $(1 - \mathrm{lr} \cdot \lambda)$배로 줄어든다. 이름 그대로 가중치가 점점 "감쇠(decay)"한다.

직관: 모델에게 "꼭 필요한 만큼만 가중치를 키워라, 아니면 줄여버리겠다"는 제약을 거는 것. 과적합을 방지하는 **정규화(regularization)** 기법의 하나.

---

#### L2 정규화와 가중치 감쇠: SGD에서는 동일

L2 정규화는 손실 함수에 가중치의 제곱합을 더한다:

$$L_{\text{reg}} = L + \frac{\lambda}{2} \|p\|^2$$

이것의 기울기를 구하면:

$$\frac{\partial L_{\text{reg}}}{\partial p} = g + \lambda \cdot p$$

SGD 업데이트에 대입하면:

$$p \leftarrow p - \mathrm{lr} \cdot (g + \lambda \cdot p) = p - \mathrm{lr} \cdot g - \mathrm{lr} \cdot \lambda \cdot p$$

**가중치 감쇠와 정확히 같은 결과!** 따라서 SGD에서는 "L2 정규화 = 가중치 감쇠"이다.

---

#### Adam에서는 달라지는 이유

Adam은 기울기를 $\frac{m}{\sqrt{v}}$로 **정규화**한 뒤 업데이트한다. L2 정규화를 Adam에 적용하면:

$$g' = g + \lambda \cdot p$$

이 $g'$가 Adam의 모멘트에 들어간다:

$$m \leftarrow \beta_1 m + (1-\beta_1) \cdot \underbrace{(g + \lambda p)}_{g'}$$
$$v \leftarrow \beta_2 v + (1-\beta_2) \cdot (g + \lambda p)^2$$

업데이트: $p \leftarrow p - \mathrm{lr}_t \cdot \frac{m}{\sqrt{v} + \epsilon}$

**문제**: $\frac{m}{\sqrt{v}}$가 정규화 항 $\lambda p$도 함께 스케일링한다.

- 기울기 $g$가 큰 파라미터: $v$도 크다 → $\frac{1}{\sqrt{v}}$가 작다 → 정규화 효과가 **약해짐**
- 기울기 $g$가 작은 파라미터: $v$도 작다 → $\frac{1}{\sqrt{v}}$가 크다 → 정규화 효과가 **강해짐**

결과: 파라미터마다 감쇠 강도가 달라진다 → $\lambda$가 의도한 대로 작동하지 않음.

---

#### AdamW의 해결법: 감쇠를 밖으로 분리

Loshchilov & Hutter (2019)의 핵심 아이디어: **가중치 감쇠를 기울기가 아닌 파라미터 업데이트 단계에서 직접 적용**한다.

$$m \leftarrow \beta_1 m + (1-\beta_1) g \quad \text{(순수 기울기만 사용)}$$
$$v \leftarrow \beta_2 v + (1-\beta_2) g^2$$
$$p \leftarrow p - \underbrace{\mathrm{lr}_t \cdot \frac{m}{\sqrt{v}+\epsilon}}_{\text{Adam 업데이트}} - \underbrace{\mathrm{lr} \cdot \lambda \cdot p}_{\text{분리된 감쇠}}$$

$\frac{m}{\sqrt{v}}$에 $\lambda p$가 포함되지 않으므로, 적응적 스케일링의 영향을 받지 않는다. 모든 파라미터에 **균일한 비율** $\lambda$로 감쇠가 적용된다.

---

#### 바이어스 보정 (Bias Correction)

Adam/AdamW에서 $m$과 $v$는 0으로 초기화된다. 학습 초기에는 실제 값보다 0에 가깝게 편향된다.

$t=1$ 일 때를 보면:

$$m_1 = \beta_1 \cdot 0 + (1-\beta_1) \cdot g_1 = (1-\beta_1) \cdot g_1$$

기대값을 구하면:

$$\mathbb{E}[m_1] = (1-\beta_1) \cdot \mathbb{E}[g_1]$$

$\beta_1 = 0.9$이면 $\mathbb{E}[m_1] = 0.1 \cdot \mathbb{E}[g_1]$ — 실제 기울기의 10%만 반영. 이 편향을 보정하기 위해:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$t=1$: $\frac{m_1}{1-0.9^1} = \frac{m_1}{0.1} = g_1$ → 편향 제거.
$t \to \infty$: $\beta_1^t \to 0$ → $\hat{m}_t \approx m_t$ → 보정이 자연스럽게 사라짐.

코드에서는 보정된 학습률로 구현:

$$\mathrm{lr}_t = \mathrm{lr} \times \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$$

```rust
let fix1 = 1.0 - self.beta1.powf(t);   // 1 - β₁ᵗ
let fix2 = 1.0 - self.beta2.powf(t);   // 1 - β₂ᵗ
let lr_t = self.lr * fix2.sqrt() / fix1;
```

---

#### 구현

```rust
pub struct AdamW {
    lr: f64,              // 학습률 (기본 0.001)
    beta1: f64,           // 1차 모멘트 감쇠율 (0.9)
    beta2: f64,           // 2차 모멘트 감쇠율 (0.999)
    eps: f64,             // 0 나누기 방지 (1e-8)
    weight_decay: f64,    // 가중치 감쇠율 (0.01~0.1)
    ms: RefCell<Vec<ArrayD<f64>>>,   // 1차 모멘트 (파라미터별)
    vs: RefCell<Vec<ArrayD<f64>>>,   // 2차 모멘트 (파라미터별)
    t: Cell<u32>,                     // 타임스텝
}

pub fn update(&self, params: &[Variable]) {
    self.t.set(self.t.get() + 1);
    let t = self.t.get() as f64;
    let lr_t = self.lr * (1.0 - self.beta2.powf(t)).sqrt()
                       / (1.0 - self.beta1.powf(t));

    for (i, p) in params.iter().enumerate() {
        if let Some(grad) = p.grad() {
            // m ← β₁·m + (1-β₁)·g
            ms[i] = &ms[i] * self.beta1 + &grad * (1.0 - self.beta1);
            // v ← β₂·v + (1-β₂)·g²
            vs[i] = &vs[i] * self.beta2 + &(&grad * &grad) * (1.0 - self.beta2);
            // Adam 업데이트
            let adam_update = ms[i].mapv(|m| m * lr_t)
                            / vs[i].mapv(|v| v.sqrt() + self.eps);
            // 분리된 가중치 감쇠
            let wd_update = p.data().mapv(|w| w * self.lr * self.weight_decay);
            // p ← p - adam_update - wd_update
            p.set_data(&(&p.data() - &adam_update) - &wd_update);
        }
    }
}
```

---

#### Adam vs AdamW 비교

| | Adam | AdamW |
|---|---|---|
| 정규화 방식 | L2: $g' = g + \lambda p$ | 분리: $p \leftarrow p(1-\mathrm{lr} \cdot \lambda)$ |
| $\frac{m}{\sqrt{v}}$ 스케일링 | 정규화 항도 스케일링됨 | 감쇠는 스케일링 무관 |
| 파라미터별 감쇠 강도 | 기울기 크기에 따라 달라짐 | 모든 파라미터에 **균일한** $\lambda$ |
| 사용처 | 일반 학습 | **Transformer 표준** (GPT, BERT) |

테스트 결과 (동일 학습, 10 스텝):
```
Adam  W norm: 0.939601
AdamW W norm: 0.411091    ← 가중치 감쇠로 노름이 절반 이하
```

---

### 학습 결과: Embedding이 학습되는가?

4개 토큰을 타겟 벡터로 매핑하는 학습:

```
idx 0 → target [1, 0]
idx 1 → target [0, 1]
idx 2 → target [-1, 0]
idx 3 → target [0, -1]
```

```
epoch   1 | loss 6.433415
epoch  10 | loss 0.139724
epoch  30 | loss 0.000032
epoch  50 | loss 0.000000

idx 0 → [1.000, 0.000]   (target: [1.0, 0.0])
idx 1 → [-0.000, 1.000]  (target: [0.0, 1.0])
idx 2 → [-1.000, -0.000] (target: [-1.0, 0.0])
idx 3 → [0.000, -1.000]  (target: [0.0, -1.0])
```

50 에폭만에 loss ≈ 0. scatter-add backward를 통해 각 인덱스의 임베딩 벡터가 정확히 타겟으로 수렴함을 확인.

---

### nanoGPT 로드맵에서의 위치

```
[x] Embedding     ← step 61 (이번)
[x] AdamW         ← step 61 (이번)
[ ] Transpose(axes), Batched Matmul
[ ] Softmax(axis), Causal Mask
[ ] LayerNorm, GELU
[ ] Multi-Head Attention
[ ] GPT Block 통합
```

Embedding은 Transformer의 **입력 변환**을 담당하고, AdamW는 **학습 전체**를 담당한다. 다음은 Attention 계산에 필요한 텐서 조작(Transpose, Batched Matmul)을 구현할 예정이다.
