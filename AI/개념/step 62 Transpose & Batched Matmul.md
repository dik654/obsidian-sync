## Step 62: Transpose(axes)와 Batched Matmul — Attention의 텐서 조작

### 한마디 직관

- **Transpose(axes) = 텐서의 축 재배치** — 데이터를 바꾸지 않고 "보는 방향"만 바꾼다
- **Batched Matmul = 여러 행렬곱을 한꺼번에** — 배치 차원을 유지한 채 마지막 두 차원에서 행렬곱

이 두 연산은 Multi-Head Attention에서 **가장 핵심적인 텐서 조작**이다.

---

### 왜 필요한가: Attention의 데이터 흐름

Attention에서 Q, K, V는 `(B, H, T, D)` shape을 갖는다:

| 차원 | 의미 | 예시 |
|------|------|------|
| B | 배치 크기 | 문장 몇 개를 동시에 처리 |
| H | 헤드 수 | 서로 다른 "관점"으로 어텐션 계산 |
| T | 시퀀스 길이 | 토큰 수 |
| D | 헤드 차원 | 각 헤드에서의 벡터 크기 |

Attention의 계산:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{D}}\right) V$$

이 과정에서 필요한 연산:
1. $K^T$: `(B,H,T,D)` → `(B,H,D,T)` — **마지막 두 축만 전치** → `transpose_axes`
2. $Q K^T$: `(B,H,T,D)` @ `(B,H,D,T)` → `(B,H,T,T)` — **배치 행렬곱** → `batched_matmul`
3. $\text{scores} \cdot V$: `(B,H,T,T)` @ `(B,H,T,D)` → `(B,H,T,D)` — 다시 배치 행렬곱

기존 dezero의 `transpose`는 2D `(M,N) → (N,M)`만, `matmul`은 2D `(M,K) @ (K,N)`만 지원한다. 4D 텐서를 다루려면 새로운 연산이 필요하다.

---

### Transpose(axes): 임의 축 순열

#### 순열(Permutation)이란?

축의 순서를 재배열하는 것. `axes = [0, 2, 1, 3]`은 "0번 축은 그대로, 1번과 2번을 교환, 3번은 그대로"를 의미한다.

$$\text{shape}(B, H, T, D) \xrightarrow{\text{axes}=[0,2,1,3]} \text{shape}(B, T, H, D)$$

**데이터는 바뀌지 않는다.** 같은 메모리의 데이터를 다른 순서로 읽는 것뿐이다. `x[b, h, t, d]`는 `y[b, t, h, d]`와 같은 값을 가리킨다.

구체적 예:

```
x: shape (2, 3, 4)         axes = [0, 2, 1]
x[0, 1, 2] = 6.0     →    y[0, 2, 1] = 6.0
 ↑  ↑  ↑                    ↑  ↑  ↑
 0번 1번 2번                 0번 2번→1번 1번→2번
```

#### 역전파: 역순열 (Inverse Permutation)

순전파에서 $y = \mathrm{transpose}_\pi(x)$이면, $y$의 원소와 $x$의 원소는 일대일 대응이다 — 데이터 자체는 변하지 않고 위치만 바뀌기 때문이다. 따라서:

$$\frac{\partial \mathcal{L}}{\partial x} = \mathrm{transpose}_{\pi^{-1}}\left(\frac{\partial \mathcal{L}}{\partial y}\right)$$

직관: 기울기도 같은 데이터를 다른 축 순서로 볼 뿐이므로, 원래 축 순서로 되돌리면 된다. 축을 $\pi$로 바꿨으면 $\pi^{-1}$로 되돌려야 원래 shape이 복원된다.

순열 $\pi$의 역순열 $\pi^{-1}$: $\pi^{-1}[\pi[i]] = i$

예: $\pi = [0, 2, 1, 3]$의 역순열은?
- $\pi[0]=0$ → $\pi^{-1}[0]=0$
- $\pi[1]=2$ → $\pi^{-1}[2]=1$
- $\pi[2]=1$ → $\pi^{-1}[1]=2$
- $\pi[3]=3$ → $\pi^{-1}[3]=3$

→ $\pi^{-1} = [0, 2, 1, 3]$ (이 경우 자기 자신의 역)

다른 예: $\pi = [2, 0, 1]$
- $\pi^{-1}[2]=0$, $\pi^{-1}[0]=1$, $\pi^{-1}[1]=2$
→ $\pi^{-1} = [1, 2, 0]$

```rust
// 역순열 계산
let mut inv_axes = vec![0; axes.len()];
for (i, &a) in axes.iter().enumerate() {
    inv_axes[a] = i;
}
```

#### 구현

```rust
struct TransposeAxesFn {
    axes: Vec<usize>,      // 순전파 축 순열
    inv_axes: Vec<usize>,  // 역전파용 역순열
}

impl Function for TransposeAxesFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let permuted = xs[0].clone().permuted_axes(IxDyn(&self.axes));
        // permuted_axes 후 비연속 메모리 → 연속화
        vec![permuted.as_standard_layout().into_owned()]
    }
    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        // 역순열을 적용하여 원래 shape으로 복원
        vec![transpose_axes(&gys[0], &self.inv_axes)]
    }
}
```

**메모리 레이아웃 문제**: `permuted_axes`는 실제 데이터를 복사하지 않고 stride만 바꾼다. 이 때문에 메모리가 비연속(non-contiguous)이 되어 이후 `into_shape_with_order`가 실패할 수 있다. `as_standard_layout()`으로 C-연속 메모리로 변환하여 해결한다.

**연속(contiguous) 메모리란?** 배열의 원소가 메모리에 빈 틈 없이 순서대로 나열된 상태. ndarray에서 배열의 메모리 위치는 **stride**로 결정된다:

```
shape = (2, 3)의 C-연속(row-major) 배열:
  strides = (3, 1)  → a[i,j]의 메모리 위치 = i*3 + j*1
  메모리: [a00, a01, a02, a10, a11, a12]

transpose 후 shape = (3, 2):
  strides = (1, 3)  → a[i,j]의 메모리 위치 = i*1 + j*3
  메모리 순서가 바뀌지 않음! stride만 바꿔서 "보는 방향"을 변경
  a[0,0]=a00, a[0,1]=a10, a[1,0]=a01, a[1,1]=a11, ...
  → 실제 메모리에서 a01이 a10보다 앞에 있음 (비연속)
```

이 비연속 상태에서 `into_shape_with_order`를 호출하면, ndarray가 원소를 재배치해야 하는지 판단할 수 없어 실패한다. `as_standard_layout()`은 데이터를 새 메모리에 연속적으로 복사하여 stride를 정상화한다.

---

### Batched Matmul: 배치 행렬곱

#### 기본 아이디어

`(..., M, K) @ (..., K, N) → (..., M, N)`

앞쪽 차원(...)은 배치로 취급하고, **마지막 두 차원에서만 행렬곱**을 수행한다.

```
x: (2, 4, 3, 5)   w: (2, 4, 5, 3)   → y: (2, 4, 3, 3)
    B  H  T  D        B  H  D  T         B  H  T  T

각 (b, h)마다 독립적으로:
  x[b,h] @ w[b,h] = (3,5) @ (5,3) = (3,3)
```

총 $B \times H = 2 \times 4 = 8$번의 독립적인 행렬곱.

#### 왜 배치 차원이 독립적인가

배치 행렬곱의 핵심 통찰: 배치 차원은 행렬곱에 참여하지 않는다. 수학적으로 표현하면:

$$Y[b_1, b_2, \ldots, :, :] = X[b_1, b_2, \ldots, :, :] \cdot W[b_1, b_2, \ldots, :, :]$$

각 배치 인덱스 조합 $(b_1, b_2, \ldots)$에서 독립적인 2D 행렬곱이 수행된다. 이것은 블록 대각 행렬의 곱과 동일하다:

$$\begin{pmatrix} X_1 & & \\ & X_2 & \\ & & \ddots \end{pmatrix} \begin{pmatrix} W_1 & & \\ & W_2 & \\ & & \ddots \end{pmatrix} = \begin{pmatrix} X_1 W_1 & & \\ & X_2 W_2 & \\ & & \ddots \end{pmatrix}$$

블록 대각 행렬의 역전파도 블록별로 독립적이므로, 2D 역전파 공식을 각 블록(배치)에 그대로 적용할 수 있다.

#### 역전파 유도

2D 행렬곱 $Y = X W$의 역전파를 원소 수준에서 유도한다.

$Y_{ij} = \sum_k X_{ik} W_{kj}$이므로:

$$\frac{\partial \mathcal{L}}{\partial X_{ik}} = \sum_j \frac{\partial \mathcal{L}}{\partial Y_{ij}} \cdot W_{kj} = \sum_j (gY)_{ij} \cdot (W^T)_{jk} = (gY \cdot W^T)_{ik}$$

$$\frac{\partial \mathcal{L}}{\partial W_{kj}} = \sum_i \frac{\partial \mathcal{L}}{\partial Y_{ij}} \cdot X_{ik} = \sum_i (X^T)_{ki} \cdot (gY)_{ij} = (X^T \cdot gY)_{kj}$$

따라서:

$$gX = gY \cdot W^T, \quad gW = X^T \cdot gY$$

배치 행렬곱에서는 이 공식이 각 배치 원소 $(b_1, b_2, \ldots)$에 독립적으로 적용된다. $W^T$는 "마지막 두 축만 전치"하면 된다 — 이것이 `swap_last_two`가 필요한 이유이다.

```rust
fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
    // gx = gy @ w^T  (마지막 두 축 전치)
    let w_t = swap_last_two(&xs[1]);
    let gx = batched_matmul(&gys[0], &w_t);
    // gw = x^T @ gy  (마지막 두 축 전치)
    let x_t = swap_last_two(&xs[0]);
    let gw = batched_matmul(&x_t, &gys[0]);
    vec![gx, gw]
}

/// 마지막 두 차원을 교환: (..., M, N) → (..., N, M)
fn swap_last_two(x: &Variable) -> Variable {
    let ndim = x.shape().len();
    let mut axes: Vec<usize> = (0..ndim).collect();
    axes.swap(ndim - 2, ndim - 1);
    transpose_axes(x, &axes)
}
```

#### 구현

```rust
impl Function for BatchedMatMulFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0];
        let w = &xs[1];
        let ndim = x.ndim();
        let m = x.shape()[ndim - 2];
        let k = x.shape()[ndim - 1];
        let n = w.shape()[ndim - 1];

        // 배치 크기: 마지막 2차원을 제외한 모든 차원의 곱
        let batch: usize = x.shape()[..ndim - 2].iter().product();

        // (..., M, K) → (batch, M, K)로 flatten
        let x_flat = x.as_standard_layout()
            .into_shape((batch, m, k)).unwrap();
        let w_flat = w.as_standard_layout()
            .into_shape((batch, k, n)).unwrap();

        // 각 배치에서 독립적으로 행렬곱
        let mut out = Vec::with_capacity(batch * m * n);
        for b in 0..batch {
            let yb = x_flat.slice(s![b, .., ..]).dot(&w_flat.slice(s![b, .., ..]));
            out.extend(yb.iter());
        }

        // 원래 배치 shape + (M, N)으로 복원
        let mut out_shape = x.shape()[..ndim - 2].to_vec();
        out_shape.extend([m, n]);
        vec![ArrayD::from_shape_vec(IxDyn(&out_shape), out).unwrap()]
    }
}
```

---

### Attention 시뮬레이션 테스트

step62의 마지막 테스트에서 실제 Attention 계산 흐름을 시뮬레이션:

```rust
// Q, K, V: (B=1, H=2, T=4, D=3)
let k_t = transpose_axes(&k, &[0, 1, 3, 2]);  // (1,2,4,3) → (1,2,3,4)
let scores = batched_matmul(&q, &k_t);          // (1,2,4,3)@(1,2,3,4) → (1,2,4,4)
let scaled = &scores / d_k.sqrt();               // scaled dot-product
let out = batched_matmul(&scaled, &v);           // (1,2,4,4)@(1,2,4,3) → (1,2,4,3)

// backward: Q, K, V 모두 grad shape이 원래와 동일
assert_eq!(q.grad().shape(), [1, 2, 4, 3]);
```

```
Attention pattern: Q@K^T → scores → scores@V
  Q shape:      [1, 2, 4, 3]
  K^T shape:    [1, 2, 3, 4]
  scores shape: [1, 2, 4, 4]
  output shape: [1, 2, 4, 3]
  Q grad shape: [1, 2, 4, 3]
```

---

### 2D vs N-D 연산 비교

| | 기존 (2D) | 신규 (N-D) |
|---|---|---|
| Transpose | `(M,N) → (N,M)` | `axes`로 임의 순열 |
| MatMul | `(M,K) @ (K,N)` | `(...,M,K) @ (...,K,N)` |
| 역전파 Transpose | 다시 전치 | 역순열 $\pi^{-1}$ |
| 역전파 MatMul | `gy@W^T`, `X^T@gy` | 동일 공식, 배치 유지 |
| 용도 | MLP, Linear | **Attention, Transformer** |

---

### nanoGPT 로드맵

```
[x] Embedding, AdamW          ← step 61
[x] Transpose(axes)           ← step 62 (이번)
[x] Batched Matmul            ← step 62 (이번)
[ ] Softmax(axis), Causal Mask
[ ] LayerNorm, GELU
[ ] Multi-Head Attention
[ ] GPT Block 통합
```

다음은 **Softmax(임의 축)**과 **Causal Mask** — Attention 계산의 나머지 핵심 부품이다.
