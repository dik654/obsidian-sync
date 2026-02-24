# Step 54: Dropout 정규화

## 개요

Step 53까지의 MLP는 파라미터가 795,010개나 되지만, 과적합을 방지하는 장치가 없다. Step 54는 **Dropout**(Srivastava et al., 2014)을 구현하여 훈련 시 뉴런을 무작위로 비활성화함으로써 과적합을 억제한다.

추가로, 훈련/추론 모드를 전환하는 **`test_mode()`** RAII 가드를 도입한다. Dropout은 훈련과 추론에서 동작이 달라지는 최초의 연산이다.

---

## 과적합(Overfitting) 문제

### 과적합이란?

$$
\mathcal{L}_{\mathrm{train}} \ll \mathcal{L}_{\mathrm{test}}
$$

학습 데이터의 손실은 매우 낮지만, 처음 보는 테스트 데이터에서는 손실이 높은 현상. 모델이 데이터의 일반적인 패턴이 아니라 **개별 샘플의 노이즈까지 암기**한 것이다.

### 왜 발생하는가?

MLP(784 → 1000 → 10)의 파라미터:

$$
|\theta| = \underbrace{784 \times 1000}_{W_1} + \underbrace{1000}_{b_1} + \underbrace{1000 \times 10}_{W_2} + \underbrace{10}_{b_2} = 795{,}010
$$

MNIST 학습 데이터: 60,000개 × 784차원.

모델의 자유도(795,010)가 충분히 크면, 학습 데이터를 **완벽히 외울 수 있다**. 하지만 이것은 일반화가 아니라 암기에 불과하다.

### 정규화(Regularization) 기법들

| 기법 | 원리 |
|---|---|
| **L2 정규화** (Weight Decay) | 큰 가중치에 페널티: $\mathcal{L} + \lambda\|\theta\|^2$ |
| **L1 정규화** | 희소한 가중치 유도: $\mathcal{L} + \lambda\|\theta\|_1$ |
| **Dropout** | 뉴런을 무작위 비활성화 → 앙상블 효과 |
| **Early Stopping** | 검증 손실이 증가하기 시작하면 학습 중단 |
| **Data Augmentation** | 학습 데이터를 변형하여 양 늘리기 |

Step 54는 이 중 **Dropout**을 구현한다.

---

## Dropout의 원리

### 기본 아이디어

훈련의 각 이터레이션에서 뉴런을 확률 $p$로 **무작위 비활성화**(출력을 0으로 설정):

$$
\mathrm{mask}_i \sim \mathrm{Bernoulli}(1 - p)
$$

$$
\mathrm{mask}_i = \begin{cases} 0 & \text{확률 } p \text{ (비활성화)} \\ 1 & \text{확률 } 1-p \text{ (활성화)} \end{cases}
$$

매번 다른 뉴런이 비활성화되므로, 모델은 **어떤 뉴런이 꺼져도 작동하도록** 학습된다. 특정 뉴런에 과도하게 의존하는 것을 방지한다.

### Vanilla Dropout vs Inverted Dropout

#### Vanilla Dropout

훈련:

$$
y_{\mathrm{train}} = x \odot \mathrm{mask}
$$

추론:

$$
y_{\mathrm{test}} = x \cdot (1 - p)
$$

훈련 시 평균적으로 $(1-p)$만큼의 뉴런만 활성화되므로, 추론 시 전체 뉴런을 사용하면 출력이 $\frac{1}{1-p}$배 커진다. 이를 보정하기 위해 추론 시 $(1-p)$를 곱한다.

**문제:** 추론 시 매번 스케일링을 해야 함 → 추론 코드가 학습 설정(dropout_ratio)에 의존

#### Inverted Dropout (실제 사용)

훈련:

$$
y_{\mathrm{train}} = \frac{x \odot \mathrm{mask}}{1 - p}
$$

추론:

$$
y_{\mathrm{test}} = x
$$

훈련 시 $\frac{1}{1-p}$로 스케일링하여 보정을 미리 수행. 추론 시에는 아무것도 하지 않고 입력을 그대로 통과시킨다.

**장점:** 추론 코드가 단순해짐. dropout_ratio를 몰라도 추론 가능.

### 기댓값 보존 증명

Inverted Dropout의 출력 기댓값이 입력과 같은지 확인:

$$
\mathbb{E}[y_i] = \mathbb{E}\left[\frac{x_i \cdot \mathrm{mask}_i}{1-p}\right] = \frac{x_i}{1-p} \cdot \mathbb{E}[\mathrm{mask}_i]
$$

$\mathrm{mask}_i \sim \mathrm{Bernoulli}(1-p)$이므로:

$$
\mathbb{E}[\mathrm{mask}_i] = 1 \cdot (1-p) + 0 \cdot p = 1-p
$$

따라서:

$$
\mathbb{E}[y_i] = \frac{x_i}{1-p} \cdot (1-p) = x_i
$$

스케일링 후 기댓값이 원래 입력 $x_i$와 동일 → 추론 시 보정 불필요.

### 앙상블 해석

뉴런 $n$개의 네트워크에서 Dropout은 $2^n$개의 서브 네트워크를 샘플링하는 것과 같다:

$$
\text{5개 뉴런, } p = 0.5 \Rightarrow 2^5 = 32\text{개 서브 네트워크}
$$

각 이터레이션에서 다른 서브 네트워크를 학습하고, 추론 시에는 이들의 **평균 예측**을 사용하는 것과 등가:

$$
y_{\mathrm{test}} \approx \frac{1}{2^n} \sum_{m \in \text{masks}} f(x; \theta \odot m)
$$

Inverted Dropout은 이 앙상블 평균을 **정확히 근사**한다.

---

## Dropout의 역전파

### Forward

$$
y = \frac{x \odot \mathrm{mask}}{1 - p}
$$

각 원소:

$$
y_i = \begin{cases} \frac{x_i}{1-p} & \text{if } \mathrm{mask}_i = 1 \\ 0 & \text{if } \mathrm{mask}_i = 0 \end{cases}
$$

### Backward

$\mathrm{mask}$는 상수이므로:

$$
\frac{\partial y_i}{\partial x_i} = \frac{\mathrm{mask}_i}{1 - p}
$$

따라서:

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \frac{\mathrm{mask}_i}{1 - p}
$$

구현에서는 forward 시 mask에 $\frac{1}{1-p}$를 미리 곱해두므로:

$$
\mathrm{scaled\_mask}_i = \begin{cases} \frac{1}{1-p} & \text{if } \mathrm{mask}_i = 1 \\ 0 & \text{if } \mathrm{mask}_i = 0 \end{cases}
$$

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \odot \mathrm{scaled\_mask}
$$

비활성화된 뉴런은 forward에서 출력이 0이었으므로, backward에서도 기울기가 0 → 해당 뉴런은 이 이터레이션에서 학습되지 않음.

---

## test_mode() RAII 가드

### 훈련/추론 모드가 필요한 이유

Dropout은 **훈련과 추론에서 동작이 다른 최초의 연산**:

| 모드 | Dropout 동작 |
|---|---|
| 훈련 (TRAINING=true) | 랜덤 마스크 적용 + 스케일링 |
| 추론 (TRAINING=false) | 입력 그대로 통과 (identity) |

향후 Batch Normalization 등도 모드에 따라 동작이 달라진다.

### 기존 no_grad() 패턴 재사용

`test_mode()`는 `no_grad()`와 완전히 동일한 RAII 패턴:

```
no_grad():   ENABLE_BACKPROP = false → 스코프 종료 시 복원
test_mode(): TRAINING = false        → 스코프 종료 시 복원
```

### 구현

```rust
thread_local! {
    static TRAINING: Cell<bool> = const { Cell::new(true) };  // 기본: 훈련 모드
}

pub struct TestModeGuard { prev: bool }

pub fn test_mode() -> TestModeGuard {
    let prev = TRAINING.with(|c| c.get());
    TRAINING.with(|c| c.set(false));
    TestModeGuard { prev }
}

impl Drop for TestModeGuard {
    fn drop(&mut self) {
        TRAINING.with(|c| c.set(self.prev));  // 이전 값 복원
    }
}
```

### RAII(Resource Acquisition Is Initialization) 패턴

```
{
    let _guard = test_mode();  // ① TRAINING = false
    // 이 스코프 안에서는 추론 모드
    let y = dropout(&x, 0.5);  // → x 그대로 통과
}                              // ② _guard.drop() → TRAINING = true 복원
// 다시 훈련 모드
let y = dropout(&x, 0.5);      // → 마스크 적용
```

Python의 `with test_mode():` 컨텍스트 매니저와 동일한 역할이지만, Rust는 Drop trait으로 **컴파일러가 복원을 보장**한다:

| | Python `with` | Rust RAII |
|---|---|---|
| 진입 | `__enter__()` | 생성자에서 상태 변경 |
| 복원 | `__exit__()` | `Drop::drop()` |
| 예외 안전성 | try/finally 필요 | 자동 (unwinding 시에도 drop 호출) |
| 복원 누락 가능성 | 있음 (with 없이 호출 시) | 없음 (컴파일러 보장) |

---

## thread_local과 Cell

### 왜 전역 상태가 필요한가?

`dropout()` 함수는 `TRAINING` 플래그를 확인해야 하지만, 함수 시그니처로 전달하면 모든 함수에 `training: bool` 파라미터가 추가되어야 한다:

```rust
// 나쁜 예: 모든 함수에 training 파라미터 전파
fn dropout(x: &Variable, ratio: f64, training: bool) -> Variable
fn forward(x: &Variable, training: bool) -> Variable
fn model_forward(x: &Variable, training: bool) -> Variable
```

thread_local로 암묵적(implicit) 컨텍스트를 전달:

```rust
// 좋은 예: thread_local로 전역 상태 관리
fn dropout(x: &Variable, ratio: f64) -> Variable {
    if TRAINING.with(|c| c.get()) { /* 마스크 적용 */ }
    else { x.clone() }
}
```

### thread_local + Cell 조합

```rust
thread_local! {
    static TRAINING: Cell<bool> = const { Cell::new(true) };
}
```

| 구성 요소 | 역할 |
|---|---|
| `thread_local!` | 스레드마다 독립적인 값 → 멀티스레드 안전 |
| `Cell<bool>` | 공유 참조(`&`)를 통해서도 값 변경 가능 (interior mutability) |
| `const { Cell::new(true) }` | 컴파일 타임 초기화 → 런타임 비용 없음 |

---

## Dropout의 난수 생성

### LCG (Linear Congruential Generator)

Dropout 마스크 생성에 기존 프로젝트의 LCG 패턴을 재사용:

$$
\mathrm{state}_{n+1} = \mathrm{state}_n \times 6364136223846793005 + 1442695040888963407 \pmod{2^{64}}
$$

$$
r_n = \frac{\mathrm{state}_{n+1} \gg 11}{2^{53}} \in [0, 1)
$$

마스크 생성 로직:

$$
\mathrm{scaled\_mask}_i = \begin{cases} \frac{1}{1-p} & \text{if } r_i > p \\ 0 & \text{if } r_i \leq p \end{cases}
$$

### thread_local RNG

```rust
thread_local! {
    static DROPOUT_RNG: Cell<u64> = const { Cell::new(1234567890) };
}
```

함수 호출마다 상태가 변하므로, 매번 **다른 마스크**가 생성된다. 동일한 RNG 상태에서는 재현 가능(deterministic).

---

## 구현에서 RefCell이 필요한 이유

### 문제: forward에서 생성한 mask를 backward에서 사용

```rust
trait Function {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>>;   // &self (불변)
    fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable>; // &self (불변)
}
```

`forward`와 `backward` 모두 `&self` (불변 참조). 하지만 forward에서 생성한 mask를 backward에서 읽어야 한다.

### 해결: RefCell로 interior mutability

```rust
struct DropoutFn {
    dropout_ratio: f64,
    mask: RefCell<ArrayD<f64>>,  // &self로도 변경 가능
}

impl Function for DropoutFn {
    fn forward(&self, xs: &[ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        // ...
        *self.mask.borrow_mut() = mask;  // &self지만 RefCell로 쓰기 가능
        vec![y]
    }
    fn backward(&self, _xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        let mask = self.mask.borrow();    // forward에서 저장한 mask 읽기
        vec![&gys[0] * &Variable::new(mask.clone())]
    }
}
```

이는 `Linear`의 `w: RefCell<Option<Variable>>`(lazy init)과 동일한 패턴이다.

---

## 실행 결과

```
x = [1.0, 1.0, 1.0, 1.0, 1.0]
train mode: y = [0.0, 0.0, 0.0, 2.0, 2.0]     ← 3개 비활성화, 2개 ×2.0
test mode:  y = [1.0, 1.0, 1.0, 1.0, 1.0]     ← 전부 통과
train mode (restored): y = [2.0, 2.0, 2.0, 0.0, 0.0]  ← 가드 해제 후 다시 적용
```

- **훈련 모드**: dropout_ratio=0.5 → 약 절반이 0, 나머지는 $\frac{1}{1-0.5} = 2.0$으로 스케일링
- **추론 모드**: `test_mode()` 가드 안 → 입력 그대로 통과
- **가드 해제 후**: RAII로 자동 복원 → 다시 훈련 모드로 dropout 적용

---

## 이 Step의 의의

1. **Dropout**: 과적합 방지의 핵심 기법. 훈련 시 뉴런을 무작위 비활성화하여 앙상블 효과
2. **Inverted Dropout**: 추론 시 보정 불필요. 기댓값이 자동 보존됨을 수학적으로 증명
3. **test_mode()**: 훈련/추론에서 동작이 다른 연산을 위한 모드 전환 (향후 Batch Normalization에도 사용)
4. **RAII 가드**: Rust의 Drop trait으로 상태 복원을 컴파일러가 보장

> Dropout은 2012년 AlexNet에서 사용되어 ImageNet 대회 우승에 기여했다.
> 이후 거의 모든 딥러닝 모델에서 표준 정규화 기법으로 자리 잡았으며,
> Batch Normalization(2015)의 등장 이후에도 여전히 널리 사용된다.