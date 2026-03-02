
# Step 52: GPU 모드와 에폭 타이밍

## 개요

Step 51에서 MNIST 학습 파이프라인을 완성했다. Step 52는 **학습 속도 최적화**에 초점을 맞춘다:
- **GPU 가속**: CuPy를 이용한 투명한 CPU→GPU 전환 (Python DeZero)
- **에폭 타이밍**: 에폭별 소요 시간 측정으로 성능 벤치마크

Rust 구현에서는 ndarray가 CPU 전용이므로 GPU 부분은 개념 학습에 그치고, 타이밍 측정만 실제 구현한다.

---

## GPU 가속이 필요한 이유

### 딥러닝의 연산 병목

MNIST MLP의 총 파라미터:

$$
\underbrace{784 \times 1000}_{W_1} + \underbrace{1000}_{b_1} + \underbrace{1000 \times 10}_{W_2} + \underbrace{10}_{b_2} = 795{,}010\mathrm{개}
$$

배치 1개(100샘플)의 forward 연산량:

$$
\underbrace{(100 \times 784) \cdot (784 \times 1000)}_{\text{1층: } 100 \times 784 \times 1000 = 78{,}400{,}000\text{ 곱셈}} + \underbrace{(100 \times 1000) \cdot (1000 \times 10)}_{\text{2층: } 100 \times 1000 \times 10 = 1{,}000{,}000\text{ 곱셈}}
$$

에폭당 600배치 × forward + backward ≈ **수십억 회의 부동소수점 연산**.

### CPU vs GPU 아키텍처

| | CPU | GPU |
|---|---|---|
| 코어 수 | 8~24개 (고성능) | 수천 개 (단순) |
| 설계 철학 | 복잡한 작업을 빠르게 | 단순한 작업을 대량으로 |
| 적합한 연산 | 분기, 순차 로직 | 행렬 곱셈, 원소별 연산 |
| 메모리 | RAM (수십~수백 GB) | VRAM (8~80 GB) |
| 대역폭 | ~50 GB/s | ~900 GB/s |

행렬 곱셈 $C = A \cdot B$에서 각 원소:

$$
C_{ij} = \sum_{k=1}^{K} A_{ik} \cdot B_{kj}
$$

모든 $C_{ij}$는 **서로 독립적**이므로 완벽한 병렬화 가능:

$$
\text{CPU: } C_{00}, C_{01}, C_{02}, \ldots \text{ 순차적으로 계산}
$$

$$
\text{GPU: } C_{00}, C_{01}, C_{02}, \ldots \text{ 수천 개 코어가 동시에 계산}
$$

---

## Python DeZero의 GPU 지원

### CuPy: NumPy의 GPU 쌍둥이

CuPy는 NumPy와 **완전히 동일한 API**를 GPU 위에서 구현한 라이브러리:

```python
import numpy as np
import cupy as cp

# CPU (NumPy)
a_cpu = np.random.randn(1000, 1000)   # RAM에 저장
c_cpu = a_cpu @ b_cpu                   # CPU 코어로 계산

# GPU (CuPy) — 코드가 동일하고 접두사만 다름
a_gpu = cp.random.randn(1000, 1000)   # VRAM에 저장
c_gpu = a_gpu @ b_gpu                   # GPU 커널로 계산
````

### 투명한 백엔드 전환

```python
# dezero가 xp 변수로 백엔드를 선택
xp = cupy if gpu_enable else numpy

# 이후 모든 연산이 xp를 통해 수행
x = xp.array([1, 2, 3])      # GPU면 VRAM, CPU면 RAM
y = xp.dot(W, x) + b          # GPU면 GPU 커널, CPU면 CPU 명령어
```

Step 52의 Python 코드:

```python
# GPU 사용 가능하면 데이터와 모델을 GPU로 전송
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()    # 배치 데이터를 VRAM으로
    model.to_gpu()           # 파라미터를 VRAM으로

# 이후 학습 코드는 한 글자도 안 바뀜!
for x, t in train_loader:
    y = model(x)             # GPU면 GPU에서, CPU면 CPU에서 실행
    loss = F.softmax_cross_entropy(y, t)
```

이것이 가능한 핵심 원리: **덕 타이핑(Duck Typing)**

```python
# Python은 타입이 아니라 "행동"으로 판단
# __add__, __matmul__ 등 매직 메서드만 있으면 어떤 객체든 연산 가능
def forward(self, x):
    return x @ self.W + self.b   # x가 numpy든 cupy든 동작
```

---

## Rust에서 투명한 GPU 전환이 어려운 이유

### 1. 정적 타입 시스템

Python은 런타임에 타입을 결정하지만, Rust는 컴파일 타임에 모든 타입이 확정된다:

```rust
// ndarray: 내부 데이터가 Vec<f64> (RAM)
let a: ArrayD<f64> = ArrayD::zeros(IxDyn(&[1000, 1000]));

// GPU 텐서: 내부 데이터가 VRAM 포인터
// → 타입 자체가 다름 → 같은 함수에 넣을 수 없음
```

### 2. 메모리 모델의 근본적 차이

CPU 연산 (`ndarray`):

```
① RAM에서 데이터 읽기
② CPU 명령어로 연산
③ RAM에 결과 쓰기
```

GPU 연산:

```
① RAM → VRAM으로 데이터 전송 (PCIe 버스, 느림)
② GPU 커널 컴파일 및 제출
③ GPU가 비동기로 실행 (수천 코어 병렬)
④ 동기화 대기
⑤ VRAM → RAM으로 결과 회수 (필요시)
```

메모리 전송, 커널 디스패치, 동기화 등이 추가되므로 내부 구현이 완전히 다르다.

### 3. ndarray에 GPU를 사후 추가할 수 없는 이유

```rust
// ndarray의 ArrayD<f64>는 내부적으로:
struct ArrayD<A> {
    data: Vec<A>,     // ← RAM에 고정된 데이터
    shape: Shape,
    strides: Strides,
}

// GPU 텐서는 내부적으로:
struct GpuTensor {
    device_ptr: *mut c_void,  // ← VRAM 포인터
    shape: Shape,
    device_id: usize,         // ← 어떤 GPU인지
}
```

`Vec<f64>`와 VRAM 포인터는 근본적으로 다른 것이라, 같은 제네릭으로 추상화하려면 **처음부터** 그렇게 설계해야 한다.

### 4. Rust에서의 해결 방법: Trait 추상화

처음부터 백엔드를 추상화하여 설계하면 가능하다:

```rust
// burn 프레임워크의 접근법
trait Backend {
    type Tensor;
    fn matmul(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn add(a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
}

struct CpuBackend;   // ndarray 기반
struct CudaBackend;  // CUDA 기반

// 제네릭으로 백엔드 교체
fn forward<B: Backend>(x: &B::Tensor, w: &B::Tensor) -> B::Tensor {
    B::matmul(x, w)   // 컴파일 타임에 백엔드 결정 → 제로 오버헤드
}
```

### Python vs Rust 비교

||Python (CuPy)|Rust|
|---|---|---|
|타입 결정|런타임 (동적)|컴파일타임 (정적)|
|API 호환 방식|덕 타이핑으로 투명 교체|trait 추상화 필요|
|백엔드 전환|`xp = cupy` 한 줄|제네릭 타입 파라미터 변경|
|성능 비용|런타임 디스패치 오버헤드|제로 오버헤드 (monomorphization)|
|구현 난이도|CuPy가 NumPy API 복제|처음부터 설계 필요|
|관련 프레임워크|CuPy, PyTorch|burn, candle|

---

## 에폭 타이밍 측정

### Python: `time.time()`

```python
import time

start = time.time()          # Unix 타임스탬프 (float, 초)
# ... 학습 ...
elapsed = time.time() - start  # 경과 시간 (초)
```

- Wall clock 기반 — 시스템 시간 변경에 영향받을 수 있음
- 정밀도: 마이크로초 수준 (OS 의존)

### Rust: `std::time::Instant`

```rust
use std::time::Instant;

let start = Instant::now();     // Monotonic clock (단조 증가)
// ... 학습 ...
let elapsed = start.elapsed();  // Duration 타입
println!("{:.4}[sec]", elapsed.as_secs_f64());
```

- **Monotonic clock**: 시스템 시간 변경에 영향받지 않음 (NTP 조정 등)
- 정밀도: 나노초 수준 (하드웨어 의존)
- `Duration` 타입으로 안전한 시간 연산

### Monotonic Clock vs Wall Clock

$$ \text{Wall Clock: } t = \mathrm{현재\ 시각} - \mathrm{시작\ 시각} $$

문제: NTP 동기화로 시스템 시간이 뒤로 갈 수 있음 → 음수 경과 시간 가능

$$ \text{Monotonic Clock: } t = \mathrm{tick}_{현재} - \mathrm{tick}_{시작} \quad (항상\ t \geq 0) $$

벤치마크에는 항상 monotonic clock을 사용해야 정확하다.

---

## 실행 결과

```
epoch: 1, loss: 1.9233, time: 6.0968[sec]
epoch: 2, loss: 1.2935, time: 6.0796[sec]
epoch: 3, loss: 0.9312, time: 6.1532[sec]
epoch: 4, loss: 0.7447, time: 6.1287[sec]
epoch: 5, loss: 0.6391, time: 6.1967[sec]
```

- 에폭당 약 **6.1초** (release 빌드, CPU)
- 에폭 간 시간이 거의 일정 → 연산량이 고정적이므로 당연
- loss가 step51과 동일 → 타이밍 추가가 학습에 영향 없음

### Debug vs Release 빌드 성능

|빌드 모드|에폭당 시간|비고|
|---|---|---|
|`--release` (LLVM 최적화)|~6초|실용적|
|debug (최적화 없음)|~10분+|784×1000 matmul이 극도로 느림|

`--release`의 LLVM 최적화:

- 루프 언롤링 (loop unrolling)
- SIMD 자동 벡터화 (SSE/AVX)
- 인라이닝 (함수 호출 제거)
- 상수 전파, 데드 코드 제거

---

## Rust의 CPU 최적화 전략

GPU가 없는 Rust에서 성능을 높이는 방법들:

### 1. BLAS 연동

ndarray는 OpenBLAS, Intel MKL 등의 BLAS 라이브러리와 연동 가능:

```toml
ndarray = { version = "0.16", features = ["blas"] }
blas-src = { version = "0.10", features = ["openblas"] }
```

BLAS가 제공하는 `sgemm`/`dgemm`은 CPU에서의 행렬 곱셈을 극한까지 최적화:

- 캐시 라인에 맞춘 타일링
- SIMD 명령어 직접 사용
- 멀티스레드 병렬화

### 2. Rayon 병렬 이터레이터

```rust
use rayon::prelude::*;
// 데이터 전처리, 배치 조립 등을 멀티코어로 병렬화
```

### 3. 컴파일러 타겟 최적화

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
# → 현재 CPU의 AVX2, FMA 등 모든 명령어 활용
```

---

## 이 Step의 의의

Step 52는 **코드 변경 없이 하드웨어 가속을 적용하는 추상화**의 중요성을 보여준다:

1. **Python**: 덕 타이핑 덕분에 `to_gpu()` 한 줄로 GPU 전환
2. **Rust**: 정적 타입이므로 trait 추상화를 처음부터 설계해야 함
3. **현실**: burn, candle 같은 Rust 프레임워크가 이 문제를 해결
4. **학습 목적**: 우리 dezero는 원리 학습이 목표이므로 CPU로 충분

> 핵심 교훈: 딥러닝 프레임워크의 핵심 가치 중 하나는 **동일한 코드로 다양한 하드웨어에서 실행**할 수 있는 추상화 레이어다. NumPy/CuPy, PyTorch의 `.to('cuda')`, TensorFlow의 device placement 모두 이 철학을 따른다.