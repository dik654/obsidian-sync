
# Step 55: CNN — 합성곱 출력 크기 계산

## 개요

Step 55부터 CNN(Convolutional Neural Network) 구현이 시작된다. 이 단계에서는 코드 구현에 앞서 합성곱 연산의 **출력 크기를 계산하는 공식**을 도입한다. 이 공식은 이후 step 57의 `conv2d` 구현에서 핵심적으로 사용된다.

---

## 합성곱(Convolution) 연산이란?

### MLP vs CNN

지금까지 사용한 MLP(Fully Connected)는 입력을 1차원 벡터로 펼쳐서 처리한다:

$$
\text{MNIST: } 28 \times 28 \text{ 이미지} \xrightarrow{\text{flatten}} 784\text{차원 벡터} \xrightarrow{W \cdot x + b} \text{출력}
$$

**문제:** 이미지의 **공간적 구조**(인접 픽셀의 관계, 패턴의 위치)가 완전히 무시된다. 왼쪽 위의 "3"이나 가운데의 "3"을 완전히 다른 패턴으로 인식한다.

CNN은 **커널(필터)**을 이미지 위에서 슬라이딩하며 **지역적 패턴**을 감지한다:

$$
\text{28×28 이미지} \xrightarrow{\text{3×3 커널 슬라이딩}} \text{특성 맵(feature map)} \xrightarrow{} \text{출력}
$$

### 합성곱의 직관적 이해

3×3 커널이 4×4 입력 위를 슬라이딩하는 과정:

```
입력 (4×4)          커널 (3×3)          출력 (2×2)
┌─────────────┐     ┌───────┐
│ 1  2  3  0  │     │ 1 0 1 │     ┌───────┐
│ 0  1  2  3  │  *  │ 0 1 0 │  =  │ 8  6  │
│ 3  0  1  2  │     │ 1 0 1 │     │ 4  8  │
│ 2  3  0  1  │     └───────┘     └───────┘
└─────────────┘
```

커널이 입력의 각 위치에서 **원소별 곱의 합**(내적)을 계산:

$$
y_{ij} = \sum_{m=0}^{K_H-1} \sum_{n=0}^{K_W-1} x_{(i+m)(j+n)} \cdot k_{mn}
$$

---

## 출력 크기 공식

### 1차원 합성곱

입력 크기 $I$, 커널 크기 $K$, 스트라이드 $S$, 패딩 $P$일 때:

$$
O = \left\lfloor \frac{I + 2P - K}{S} \right\rfloor + 1
$$

### 2차원 합성곱

높이와 너비에 독립적으로 적용:

$$
O_H = \left\lfloor \frac{H + 2P_H - K_H}{S_H} \right\rfloor + 1
$$

$$
O_W = \left\lfloor \frac{W + 2P_W - K_W}{S_W} \right\rfloor + 1
$$

### 공식 유도

패딩 후 실제 입력 크기:

$$
I' = I + 2P
$$

커널이 첫 위치에서 차지하는 크기: $K$

남은 공간: $I' - K = I + 2P - K$

스트라이드 $S$로 이동할 수 있는 횟수: $\left\lfloor \frac{I' - K}{S} \right\rfloor$

첫 위치(+1) 포함:

$$
O = \left\lfloor \frac{I + 2P - K}{S} \right\rfloor + 1
$$

### Rust 구현

```rust
/// 합성곱 출력 크기 계산
/// (input_size + 2*pad - kernel_size) / stride + 1
pub fn get_conv_outsize(
    input_size: usize, kernel_size: usize,
    stride: usize, pad: usize,
) -> usize {
    (input_size + pad * 2 - kernel_size) / stride + 1
}
```

Python의 `//` (정수 나눗셈)와 Rust의 `usize /` 가 동일한 동작 (자동 floor).

---

## 파라미터별 역할

### 커널 크기 (Kernel Size, $K$)

커널은 감지할 **지역 패턴의 크기**를 결정한다:

```
3×3 커널: 작은 패턴 (에지, 코너)
5×5 커널: 중간 패턴 (텍스처)
7×7 커널: 큰 패턴 (구조)
```

커널이 클수록 넓은 범위를 보지만, 파라미터 수가 $K^2$로 증가한다.

### 스트라이드 (Stride, $S$)

커널이 이동하는 **칸 수**:

```
stride=1: 한 칸씩 이동 → 출력이 큼 (세밀)
stride=2: 두 칸씩 이동 → 출력이 절반 (다운샘플링 효과)
```

$$
S = 1: \quad O = I + 2P - K + 1
$$

$$
S = 2: \quad O = \left\lfloor \frac{I + 2P - K}{2} \right\rfloor + 1 \approx \frac{I}{2}
$$

### 패딩 (Padding, $P$)

입력 테두리에 **0을 추가**:

```
pad=0 (valid):  출력이 입력보다 작음
pad=1 (same):   출력이 입력과 같음 (3×3 커널, stride=1일 때)
```

패딩을 추가하는 이유:
1. **테두리 정보 보존**: 패딩 없으면 모서리 픽셀은 한 번만 사용됨
2. **출력 크기 유지**: 깊은 네트워크에서 특성 맵이 너무 작아지는 것 방지

---

## 대표적인 케이스

### Same Padding (출력 = 입력)

$K = 3, S = 1, P = 1$일 때:

$$
O = \frac{I + 2 \cdot 1 - 3}{1} + 1 = \frac{I - 1}{1} + 1 = I
$$

일반화: $S = 1$일 때 same padding을 위한 패딩 크기:

$$
P = \frac{K - 1}{2}
$$

| 커널 크기 | Same Padding |
|---|---|
| $3 \times 3$ | $P = 1$ |
| $5 \times 5$ | $P = 2$ |
| $7 \times 7$ | $P = 3$ |

### Valid Padding (패딩 없음)

$P = 0, S = 1$일 때:

$$
O = I - K + 1
$$

입력보다 $K - 1$만큼 작아진다. 3×3 커널이면 상하좌우 1씩 줄어서 $O = I - 2$.

### Stride 2 다운샘플링

$S = 2, P = 1, K = 3$일 때:

$$
O = \frac{I + 2 - 3}{2} + 1 = \frac{I - 1}{2} + 1 \approx \frac{I}{2}
$$

풀링(pooling) 대신 stride=2 합성곱으로 해상도를 절반으로 줄이는 현대적 기법.

---

## 테스트 코드와 검증

### step55.rs 전체 코드

```rust
use dezero::get_conv_outsize;

#[test]
fn test_conv_outsize() {
    let (h, w) = (4, 4);       // 입력 크기
    let (kh, kw) = (3, 3);     // 커널 크기
    let (sh, sw) = (1, 1);     // 스트라이드
    let (ph, pw) = (1, 1);     // 패딩

    let oh = get_conv_outsize(h, kh, sh, ph);
    let ow = get_conv_outsize(w, kw, sw, pw);
    // oh = 4, ow = 4 → same padding
    assert_eq!(oh, 4);
    assert_eq!(ow, 4);

    // pad=0, stride=1, kernel=3 → valid padding
    let oh = get_conv_outsize(4, 3, 1, 0);
    assert_eq!(oh, 2); // (4 + 0 - 3) / 1 + 1 = 2

    // stride=2 → 출력이 절반
    let oh = get_conv_outsize(8, 3, 2, 1);
    assert_eq!(oh, 4); // (8 + 2 - 3) / 2 + 1 = 4

    // 7×7 입력, 5×5 커널 → im2col의 출력 행 수
    let oh = get_conv_outsize(7, 5, 1, 0);
    assert_eq!(oh, 3); // (7 + 0 - 5) / 1 + 1 = 3
}
```

### 계산 검증 표

| 입력 | 커널 | Stride | Pad | 출력 | 설명 |
|---|---|---|---|---|---|
| 4 | 3 | 1 | 1 | **4** | Same padding |
| 4 | 3 | 1 | 0 | **2** | Valid padding |
| 8 | 3 | 2 | 1 | **4** | Stride 2 다운샘플링 |
| 7 | 5 | 1 | 0 | **3** | im2col 출력 행 수 |

---

## CNN에서 텐서의 차원 규약

### NCHW 형식

CNN에서는 4차원 텐서를 사용한다:

$$
\text{Tensor shape: } (N, C, H, W)
$$

| 차원 | 의미 | 예시 |
|---|---|---|
| $N$ | 배치 크기 | 100 (미니배치) |
| $C$ | 채널 수 | 1 (그레이스케일), 3 (RGB) |
| $H$ | 높이 | 28 (MNIST) |
| $W$ | 너비 | 28 (MNIST) |

### 합성곱 후 차원 변화

입력 $(N, C_{\mathrm{in}}, H, W)$에 커널 $(C_{\mathrm{out}}, C_{\mathrm{in}}, K_H, K_W)$를 적용하면:

$$
\text{출력: } (N, C_{\mathrm{out}}, O_H, O_W)
$$

여기서 $O_H, O_W$가 바로 `get_conv_outsize`로 계산되는 값이다.

---

## VGG16에서의 실제 크기 변화

참고: step 58에서 구현하는 VGG16 네트워크의 크기 변화:

```
입력:     (N,   3, 224, 224)    ← RGB 이미지
conv1:    (N,  64, 224, 224)    ← K=3, S=1, P=1 (same)
conv2:    (N,  64, 224, 224)    ← K=3, S=1, P=1 (same)
pool1:    (N,  64, 112, 112)    ← 2×2 max pooling (절반)
conv3:    (N, 128, 112, 112)    ← K=3, S=1, P=1 (same)
conv4:    (N, 128, 112, 112)    ← K=3, S=1, P=1 (same)
pool2:    (N, 128,  56,  56)    ← 2×2 max pooling
...
pool5:    (N, 512,   7,   7)    ← 최종 특성 맵
flatten:  (N, 25088)            ← 512 × 7 × 7
fc:       (N, 1000)             ← 1000 클래스 분류
```

모든 `conv` 레이어의 출력 크기가 `get_conv_outsize`로 계산된다.

---

## Python vs Rust 비교

```python
# Python — 정수 나눗셈 //
def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1
```

```rust
// Rust — usize 나눗셈은 자동으로 floor
pub fn get_conv_outsize(
    input_size: usize, kernel_size: usize,
    stride: usize, pad: usize,
) -> usize {
    (input_size + pad * 2 - kernel_size) / stride + 1
}
```

| | Python | Rust |
|---|---|---|
| 정수 나눗셈 | `//` 연산자 | `/` (usize는 자동 floor) |
| 타입 | int (무한 정밀도) | usize (64비트 양수) |
| 음수 언더플로 | 음수 반환 | **panic** (usize 언더플로) |

Rust의 usize 언더플로 주의: `input_size + 2*pad < kernel_size`이면 panic 발생. 유효하지 않은 합성곱 파라미터를 컴파일 타임에는 잡지 못하지만, 런타임에 즉시 발견된다.

---

## 이 Step의 의의

1. **CNN의 첫 단계**: 합성곱 출력 크기 공식은 CNN 설계의 기본
2. **공식 하나가 핵심**: `get_conv_outsize`는 단순하지만, 이후 `im2col`, `conv2d`, `pooling` 모두에서 사용
3. **공간적 구조 보존**: MLP와 달리 CNN은 이미지의 2D 구조를 유지하며 처리
4. **다음 단계 준비**: step 56(이론), step 57(im2col + conv2d 구현)의 기초

> 합성곱 출력 크기 공식은 CNN 논문을 읽거나 네트워크를 설계할 때
> 가장 자주 사용하는 공식이다. VGG, ResNet, EfficientNet 등
> 모든 CNN 아키텍처의 각 레이어 차원은 이 공식으로 결정된다.



# Step 57: im2col과 conv2d_simple

## 개요

Step 55에서 합성곱 출력 크기 공식을 준비했고, Step 56(이론)을 거쳐 Step 57에서 **실제 합성곱 연산**을 구현한다. 핵심 아이디어는 합성곱을 직접 루프로 계산하지 않고, **im2col 변환 + 행렬 곱셈**으로 대체하는 것이다.

---

## im2col이 필요한 이유

### 합성곱의 나이브 구현

합성곱을 직접 구현하면 **6중 루프**가 필요하다:

```
for n in 0..N:          ← 배치
  for oc in 0..OC:      ← 출력 채널
    for oh in 0..OH:    ← 출력 높이
      for ow in 0..OW:  ← 출력 너비
        for c in 0..C:  ← 입력 채널
          for kh, kw:   ← 커널 순회
            y[n,oc,oh,ow] += x[n,c,oh+kh,ow+kw] * W[oc,c,kh,kw]
```

이 구현은 정확하지만 **매우 느리다**. CPU/GPU는 루프보다 행렬 곱셈에 최적화되어 있기 때문이다.

### im2col: 합성곱 → 행렬 곱셈

**im2col(image to column)**은 입력 이미지의 각 패치를 행으로 펼쳐서 2D 행렬로 변환한다:

$$
\underbrace{(N, C, H, W)}_{\text{4D 이미지}} \xrightarrow{\mathrm{im2col}} \underbrace{(N \cdot O_H \cdot O_W,\ C \cdot K_H \cdot K_W)}_{\text{2D 패치 행렬}}
$$

그러면 합성곱이 **한 번의 행렬 곱셈**으로 바뀐다:

$$
\underbrace{\mathrm{col}}_{(N \cdot O_H \cdot O_W,\ C \cdot K_H \cdot K_W)} \times \underbrace{W^T}_{(C \cdot K_H \cdot K_W,\ O_C)} = \underbrace{Y}_{(N \cdot O_H \cdot O_W,\ O_C)}
$$

### 시각적 이해

3×3 커널이 4×4 입력을 슬라이딩 (stride=1, pad=0, OH=OW=2):

```
입력 (1, 1, 4, 4):                im2col 결과 (4, 9):
┌──────────────┐
│ a  b  c  d   │                   패치0: [a, b, c, e, f, g, i, j, k]
│ e  f  g  h   │    im2col →      패치1: [b, c, d, f, g, h, j, k, l]
│ i  j  k  l   │                   패치2: [e, f, g, i, j, k, m, n, o]
│ m  n  o  p   │                   패치3: [f, g, h, j, k, l, n, o, p]
└──────────────┘
                                   각 행 = 커널이 보는 3×3 영역을 펼친 것
```

커널도 1D로 펼치면:

$$
W_{\mathrm{flat}} = [w_0, w_1, \ldots, w_8] \quad (1 \times 9)
$$

합성곱 출력의 각 원소:

$$
y_i = \mathrm{col}[i, :] \cdot W_{\mathrm{flat}}^T = \sum_{j=0}^{8} \mathrm{col}[i, j] \cdot w_j
$$

---

## im2col 구현

### 알고리즘

입력 $(N, C, H, W)$에서 각 패치를 추출하여 행으로 배치:

$$
x[n, c, i \cdot S_H + k_h, j \cdot S_W + k_w] \rightarrow \mathrm{col}[n \cdot O_H \cdot O_W + i \cdot O_W + j,\ c \cdot K_H \cdot K_W + k_h \cdot K_W + k_w]
$$

행 인덱스: 배치 내 위치 $(n, i, j)$를 1D로 인코딩
열 인덱스: 채널 내 커널 위치 $(c, k_h, k_w)$를 1D로 인코딩

### Rust 구현

```rust
fn im2col_data(
    x: &ArrayD<f64>, kh: usize, kw: usize,
    sh: usize, sw: usize, ph: usize, pw: usize,
) -> ArrayD<f64> {
    let shape = x.shape();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let oh = get_conv_outsize(h, kh, sh, ph);
    let ow = get_conv_outsize(w, kw, sw, pw);

    // 패딩 적용: (N,C,H,W) → (N,C,H+2P,W+2P)
    let h_pad = h + 2 * ph;
    let w_pad = w + 2 * pw;
    let mut x_pad = ArrayD::zeros(ndarray::IxDyn(&[n, c, h_pad, w_pad]));
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    x_pad[[ni, ci, hi + ph, wi + pw]] = x[[ni, ci, hi, wi]];
                }
            }
        }
    }

    // 각 패치를 행으로 펼침
    let rows = n * oh * ow;        // 총 패치 수
    let cols = c * kh * kw;        // 패치 하나의 크기
    let mut col = vec![0.0; rows * cols];

    for ni in 0..n {
        for i in 0..oh {
            for j in 0..ow {
                let row = ni * oh * ow + i * ow + j;
                let h_start = i * sh;
                let w_start = j * sw;
                for ci in 0..c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let col_idx = ci * kh * kw + ki * kw + kj;
                            col[row * cols + col_idx] =
                                x_pad[[ni, ci, h_start + ki, w_start + kj]];
                        }
                    }
                }
            }
        }
    }

    ArrayD::from_shape_vec(ndarray::IxDyn(&[rows, cols]), col).unwrap()
}
```

### 출력 shape 검증

입력 $(1, 3, 7, 7)$, 커널 $5 \times 5$, stride=1, pad=0:

$$
O_H = \frac{7 - 5}{1} + 1 = 3, \quad O_W = 3
$$

$$
\mathrm{rows} = 1 \times 3 \times 3 = 9
$$

$$
\mathrm{cols} = 3 \times 5 \times 5 = 75
$$

$$
\mathrm{col.shape} = (9, 75)
$$

배치 10개: $\mathrm{rows} = 10 \times 3 \times 3 = 90 \rightarrow (90, 75)$

---

## col2im: im2col의 역연산

### 필요한 이유

역전파에서 기울기를 원래 이미지 shape으로 복원해야 한다. im2col에서 **같은 픽셀이 여러 패치에 포함**될 수 있으므로, 역변환 시에는 **겹치는 위치를 합산(scatter-add)**한다.

### 겹침이 발생하는 이유

stride=1, kernel=3일 때 입력의 중앙 픽셀은 **최대 9개 패치**에 포함된다:

```
패치 (0,0)에서 [2,2] 위치로 참조
패치 (0,1)에서 [2,1] 위치로 참조
패치 (1,0)에서 [1,2] 위치로 참조
... 등 최대 KH × KW = 9번 참조됨
```

따라서 col2im은 단순 복사가 아니라 **+=** (scatter-add)여야 한다.

### Rust 구현

```rust
fn col2im_data(
    col: &ArrayD<f64>, x_shape: &[usize],
    kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize,
) -> ArrayD<f64> {
    let (n, c, h, w) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    let oh = get_conv_outsize(h, kh, sh, ph);
    let ow = get_conv_outsize(w, kw, sw, pw);
    let h_pad = h + 2 * ph;
    let w_pad = w + 2 * pw;

    // 패딩 포함한 크기로 scatter-add
    let mut x_pad = ArrayD::zeros(ndarray::IxDyn(&[n, c, h_pad, w_pad]));

    for ni in 0..n {
        for i in 0..oh {
            for j in 0..ow {
                let row = ni * oh * ow + i * ow + j;
                let h_start = i * sh;
                let w_start = j * sw;
                for ci in 0..c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let col_idx = ci * kh * kw + ki * kw + kj;
                            // += : 겹치는 위치는 합산
                            x_pad[[ni, ci, h_start + ki, w_start + kj]] +=
                                col[[row, col_idx]];
                        }
                    }
                }
            }
        }
    }

    // 패딩 제거: (N,C,H+2P,W+2P) → (N,C,H,W)
    if ph == 0 && pw == 0 {
        x_pad
    } else {
        let mut result = ArrayD::zeros(ndarray::IxDyn(&[n, c, h, w]));
        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        result[[ni, ci, hi, wi]] = x_pad[[ni, ci, hi + ph, wi + pw]];
                    }
                }
            }
        }
        result
    }
}
```

---

## Im2colFn: 역전파를 지원하는 im2col

im2col을 계산 그래프에 포함시켜 역전파가 자동으로 col2im을 호출하도록 한다.

```rust
struct Im2colFn {
    kh: usize, kw: usize,
    sh: usize, sw: usize,
    ph: usize, pw: usize,
    x_shape: Vec<usize>,   // backward에서 원래 shape 복원용
}

impl Function for Im2colFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        vec![im2col_data(&xs[0], self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)]
    }
    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gy_data = gys[0].data();
        let gx = col2im_data(
            &gy_data, &self.x_shape,
            self.kh, self.kw, self.sh, self.sw, self.ph, self.pw,
        );
        vec![Variable::new(gx)]
    }
    fn name(&self) -> &str { "Im2col" }
}

pub fn im2col(
    x: &Variable, kh: usize, kw: usize,
    sh: usize, sw: usize, ph: usize, pw: usize,
) -> Variable {
    let x_shape = x.shape();
    Func::new(Im2colFn { kh, kw, sh, sw, ph, pw, x_shape }).call(&[x])
}
```

---

## conv2d_simple: 합성곱 = im2col + matmul

### Forward 과정

$$
x: (N, C, H, W), \quad W: (O_C, C, K_H, K_W)
$$

1. **im2col**: $\mathrm{col} = \mathrm{im2col}(x) \in \mathbb{R}^{N \cdot O_H \cdot O_W \times C \cdot K_H \cdot K_W}$
2. **W 변형**: $W_{\mathrm{flat}} = W.\mathrm{reshape}(O_C, C \cdot K_H \cdot K_W) \in \mathbb{R}^{O_C \times C \cdot K_H \cdot K_W}$
3. **행렬 곱**: $Y_{\mathrm{2d}} = \mathrm{col} \times W_{\mathrm{flat}}^T \in \mathbb{R}^{N \cdot O_H \cdot O_W \times O_C}$
4. **축 재배치**: $(N \cdot O_H \cdot O_W, O_C) \rightarrow (N, O_C, O_H, O_W)$

### Backward 과정: 유도

기울기 $\frac{\partial \mathcal{L}}{\partial Y}$가 $(N, O_C, O_H, O_W)$로 주어질 때, 이를 축 재배치하여 $\frac{\partial \mathcal{L}}{\partial Y_{\mathrm{2d}}} \in \mathbb{R}^{N \cdot O_H \cdot O_W \times O_C}$로 변환한다.

#### 왜 이 공식이 나오는가: 행렬곱의 chain rule

순전파는 행렬곱이다:

$$Y_{\mathrm{2d}} = \mathrm{col} \times W_{\mathrm{flat}}^T$$

$R = N \cdot O_H \cdot O_W$, $K = C \cdot K_H \cdot K_W$로 놓으면, 이것은 $(R \times K) \times (K \times O_C) = (R \times O_C)$ 형태의 행렬곱이다.

행렬곱 $Y = A B$의 역전파는:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial Y} \cdot B^T, \quad \frac{\partial \mathcal{L}}{\partial B} = A^T \cdot \frac{\partial \mathcal{L}}{\partial Y}$$

이 공식이 성립하는 이유를 원소 수준에서 확인한다. $Y_{ij} = \sum_k A_{ik} B_{kj}$이므로:

$$\frac{\partial \mathcal{L}}{\partial A_{ik}} = \sum_j \frac{\partial \mathcal{L}}{\partial Y_{ij}} \cdot \frac{\partial Y_{ij}}{\partial A_{ik}} = \sum_j \frac{\partial \mathcal{L}}{\partial Y_{ij}} \cdot B_{kj} = \sum_j (gY)_{ij} \cdot (B^T)_{jk}$$

이것은 행렬곱 $gY \cdot B^T$의 $(i,k)$ 원소이다. 마찬가지로 $gB = A^T \cdot gY$.

#### conv2d에 적용

순전파에서 $A = \mathrm{col}$, $B = W_{\mathrm{flat}}^T$이므로:

**입력 기울기** (col에 대한 기울기):

$$\frac{\partial \mathcal{L}}{\partial \mathrm{col}} = \frac{\partial \mathcal{L}}{\partial Y_{\mathrm{2d}}} \cdot (W_{\mathrm{flat}}^T)^T = \frac{\partial \mathcal{L}}{\partial Y_{\mathrm{2d}}} \cdot W_{\mathrm{flat}} \in \mathbb{R}^{R \times K}$$

col은 im2col의 출력이므로, col에 대한 기울기를 원래 이미지 shape으로 복원하려면 col2im을 적용한다:

$$\frac{\partial \mathcal{L}}{\partial x} = \mathrm{col2im}\left(\frac{\partial \mathcal{L}}{\partial \mathrm{col}}\right) \in \mathbb{R}^{N \times C \times H \times W}$$

**가중치 기울기** ($W_{\mathrm{flat}}^T$에 대한 기울기):

$$\frac{\partial \mathcal{L}}{\partial W_{\mathrm{flat}}^T} = \mathrm{col}^T \cdot \frac{\partial \mathcal{L}}{\partial Y_{\mathrm{2d}}} \in \mathbb{R}^{K \times O_C}$$

전치하면:

$$\frac{\partial \mathcal{L}}{\partial W_{\mathrm{flat}}} = \left(\frac{\partial \mathcal{L}}{\partial Y_{\mathrm{2d}}}\right)^T \cdot \mathrm{col} \in \mathbb{R}^{O_C \times K}$$

reshape으로 원래 커널 shape 복원:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W_{\mathrm{flat}}}.\mathrm{reshape}(O_C, C, K_H, K_W)$$

#### col2im이 im2col의 "전치 연산"인 이유

im2col을 선형 변환 $T: \mathbb{R}^{N \times C \times H \times W} \to \mathbb{R}^{R \times K}$로 볼 수 있다. 이 변환은 같은 픽셀을 여러 패치에 복사하는 연산이다.

chain rule에서 기울기 전파는 항상 순전파의 **전치(adjoint)**를 사용한다. im2col의 전치란 "각 패치에서 온 기울기를 원래 픽셀에 합산"하는 것 — 이것이 바로 col2im의 scatter-add이다.

$$\langle \mathrm{im2col}(x),\, g_{\mathrm{col}} \rangle = \langle x,\, \mathrm{col2im}(g_{\mathrm{col}}) \rangle$$

이 내적 관계가 성립하므로 col2im은 im2col의 수학적 전치(adjoint)이다.

### Rust 구현

```rust
struct Conv2dSimpleFn {
    kh: usize, kw: usize,
    sh: usize, sw: usize,
    ph: usize, pw: usize,
    x_shape: Vec<usize>,
    w_shape: Vec<usize>,
    col: RefCell<ArrayD<f64>>,  // forward에서 저장 → backward에서 dw 계산
}

impl Function for Conv2dSimpleFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = &xs[0]; // (N, C, H, W)
        let w = &xs[1]; // (OC, C, KH, KW)

        let n = self.x_shape[0];
        let oc = self.w_shape[0];
        let oh = get_conv_outsize(self.x_shape[2], self.kh, self.sh, self.ph);
        let ow = get_conv_outsize(self.x_shape[3], self.kw, self.sw, self.pw);

        // im2col: (N*OH*OW, C*KH*KW)
        let col = im2col_data(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw);

        // W → (OC, C*KH*KW)
        let ckk = self.w_shape[1] * self.kh * self.kw;
        let w_2d = w.to_shape((oc, ckk)).unwrap();

        // col @ W^T → (N*OH*OW, OC)
        let col_2d = col.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let y_2d = col_2d.dot(&w_2d.t());

        // (N*OH*OW, OC) → (N, OC, OH, OW) 축 재배치
        let mut y = ArrayD::zeros(ndarray::IxDyn(&[n, oc, oh, ow]));
        for ni in 0..n {
            for oci in 0..oc {
                for hi in 0..oh {
                    for wi in 0..ow {
                        y[[ni, oci, hi, wi]] =
                            y_2d[[ni * oh * ow + hi * ow + wi, oci]];
                    }
                }
            }
        }

        *self.col.borrow_mut() = col;  // backward에서 사용
        vec![y]
    }

    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let gy = gys[0].data(); // (N, OC, OH, OW)
        let n = self.x_shape[0];
        let oc = self.w_shape[0];
        let ckk = self.w_shape[1] * self.kh * self.kw;
        let oh = get_conv_outsize(self.x_shape[2], self.kh, self.sh, self.ph);
        let ow = get_conv_outsize(self.x_shape[3], self.kw, self.sw, self.pw);

        // gy (N,OC,OH,OW) → gy_2d (N*OH*OW, OC)
        let mut gy_2d = ndarray::Array2::zeros((n * oh * ow, oc));
        for ni in 0..n {
            for oci in 0..oc {
                for hi in 0..oh {
                    for wi in 0..ow {
                        gy_2d[[ni * oh * ow + hi * ow + wi, oci]] =
                            gy[[ni, oci, hi, wi]];
                    }
                }
            }
        }

        let col = self.col.borrow();
        let col_2d = col.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let w_data = _xs[1].data();
        let w_2d = w_data.to_shape((oc, ckk)).unwrap();

        // dx: dcol = gy_2d @ W → col2im → (N,C,H,W)
        let dcol = gy_2d.dot(&w_2d);
        let gx = col2im_data(
            &dcol.into_dyn(), &self.x_shape,
            self.kh, self.kw, self.sh, self.sw, self.ph, self.pw,
        );

        // dw: col^T @ gy_2d → (C*KH*KW, OC) → transpose → reshape
        let dw_2d = col_2d.t().dot(&gy_2d);       // (C*KH*KW, OC)
        let dw_t = dw_2d.t().to_owned();           // (OC, C*KH*KW)
        let dw = ArrayD::from_shape_vec(
            ndarray::IxDyn(&self.w_shape),
            dw_t.into_raw_vec_and_offset().0,
        ).unwrap();

        vec![Variable::new(gx), Variable::new(dw)]
    }

    fn name(&self) -> &str { "Conv2d" }
}
```

### 공개 함수

```rust
pub fn conv2d_simple(
    x: &Variable, w: &Variable, b: Option<&Variable>,
    stride: usize, pad: usize,
) -> Variable {
    let x_shape = x.shape();
    let w_shape = w.shape();
    let kh = w_shape[2];
    let kw = w_shape[3];

    let y = Func::new(Conv2dSimpleFn {
        kh, kw, sh: stride, sw: stride, ph: pad, pw: pad,
        x_shape, w_shape,
        col: RefCell::new(ArrayD::zeros(ndarray::IxDyn(&[]))),
    }).call(&[x, w]);

    match b {
        Some(b) => &y + b,  // broadcast: (N,OC,OH,OW) + (OC,) or (1,OC,1,1)
        None => y,
    }
}
```

---

## RefCell로 col을 저장하는 이유

### 문제

Conv2dSimpleFn의 backward에서 가중치 기울기를 계산하려면 forward에서 만든 `col` 행렬이 필요하다:

$$
\frac{\partial \mathcal{L}}{\partial W} = \mathrm{col}^T \times \frac{\partial \mathcal{L}}{\partial Y}
$$

하지만 Function trait의 `forward(&self, ...)`는 불변 참조이다.

### 해결

```rust
struct Conv2dSimpleFn {
    col: RefCell<ArrayD<f64>>,  // interior mutability
}

fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
    let col = im2col_data(...);
    *self.col.borrow_mut() = col;  // &self로도 쓰기 가능
    // ...
}

fn backward(&self, ...) -> Vec<Variable> {
    let col = self.col.borrow();   // forward에서 저장한 col 읽기
    let dw = col.t().dot(&gy_2d);  // dw 계산에 사용
    // ...
}
```

이는 Step 54의 DropoutFn에서 mask를 저장하는 것과 동일한 패턴이다.

---

## 축 재배치: (N\*OH\*OW, OC) → (N, OC, OH, OW)

행렬 곱셈 결과는 $(N \cdot O_H \cdot O_W, O_C)$이지만, CNN의 출력 규약은 $(N, O_C, O_H, O_W)$ (NCHW)이다.

단순히 reshape만 하면 안 된다 — **축 순서가 다르기** 때문이다:

```
y_2d[n*OH*OW + oh*OW + ow, oc]  →  y[n, oc, oh, ow]
```

Rust에서는 명시적 루프로 원소를 배치한다:

```rust
for ni in 0..n {
    for oci in 0..oc {
        for hi in 0..oh {
            for wi in 0..ow {
                y[[ni, oci, hi, wi]] = y_2d[[ni * oh * ow + hi * ow + wi, oci]];
            }
        }
    }
}
```

Python에서는 `reshape + transpose(0, 3, 1, 2)`로 한 줄이지만, Rust ndarray의 동적 차원(`ArrayD`)은 임의 축 치환이 번거로워서 루프로 처리한다.

---

## 테스트와 검증

### im2col shape 검증

```rust
#[test]
fn test_im2col() {
    // (1, 3, 7, 7) → kernel=5, stride=1, pad=0
    // OH = (7-5)/1+1 = 3, OW = 3
    // 출력: (1*3*3, 3*5*5) = (9, 75)
    let x1 = Variable::new(/* (1,3,7,7) */);
    let col1 = im2col(&x1, 5, 5, 1, 1, 0, 0);
    assert_eq!(col1.shape(), vec![9, 75]);

    // (10, 3, 7, 7) → 배치 10
    // 출력: (10*3*3, 3*5*5) = (90, 75)
    let x2 = Variable::new(/* (10,3,7,7) */);
    let col2 = im2col(&x2, 5, 5, 1, 1, 0, 0);
    assert_eq!(col2.shape(), vec![90, 75]);
}
```

```
col1.shape = [9, 75]     ← 1 샘플, 3×3 출력 위치, 75차원 패치
col2.shape = [90, 75]    ← 10 샘플, 각 9개 위치 = 90행
```

### conv2d forward + backward 검증

```rust
#[test]
fn test_conv2d_simple() {
    // x: (1, 5, 15, 15), W: (8, 5, 3, 3), stride=1, pad=1
    let x = Variable::new(/* (1,5,15,15) */);
    let w = Variable::new(/* (8,5,3,3) */);

    let y = conv2d_simple(&x, &w, None, 1, 1);
    assert_eq!(y.shape(), vec![1, 8, 15, 15]);  // same padding

    y.backward(false, false);
    let x_grad = x.grad().unwrap();
    assert_eq!(x_grad.shape(), &[1, 5, 15, 15]);  // 원래 x와 같은 shape
}
```

```
y.shape = [1, 8, 15, 15]        ← OC=8, same padding (pad=1)
x.grad.shape = [1, 5, 15, 15]   ← col2im으로 복원된 입력 기울기
```

---

## 전체 데이터 흐름 정리

### Forward

```
x (1, 5, 15, 15)
    │
    ▼ im2col (K=3, S=1, P=1)
col (225, 45)              ← 1×15×15 = 225 패치, 5×3×3 = 45차원
    │
    ▼ col @ W^T
y_2d (225, 8)              ← 225 패치 × 8 출력 채널
    │
    ▼ 축 재배치
y (1, 8, 15, 15)           ← NCHW 형식 출력
```

### Backward

```
gy (1, 8, 15, 15)
    │
    ▼ 축 재배치
gy_2d (225, 8)
    │
    ├──▶ gy_2d @ W = dcol (225, 45)  ──▶ col2im ──▶ gx (1, 5, 15, 15)
    │
    └──▶ col^T @ gy_2d = dw (45, 8) ──▶ reshape ──▶ gw (8, 5, 3, 3)
```

---

## 이 Step의 의의

1. **im2col**: 합성곱을 행렬 곱셈으로 변환하는 핵심 기법. 모든 딥러닝 프레임워크의 CNN 구현 기초
2. **col2im**: im2col의 역연산 (scatter-add). 역전파에서 입력 기울기를 복원
3. **conv2d_simple**: im2col + matmul 조합으로 완전한 합성곱 (forward + backward)
4. **메모리-속도 트레이드오프**: im2col은 패치를 중복 복사하므로 메모리를 더 쓰지만, 행렬 곱셈 최적화(BLAS)를 활용할 수 있어 훨씬 빠름

> im2col은 Caffe(2014)에서 대중화된 기법으로, cuDNN, PyTorch, TensorFlow 등
> 거의 모든 딥러닝 프레임워크가 합성곱 구현에 사용한다.
> 최근에는 Winograd 변환 등 더 효율적인 방법도 있지만,
> im2col은 여전히 가장 직관적이고 범용적인 접근법이다.