##   
Step 2-4: Montgomery 곱셈

# Montgomery 곱셈 - 기호 정의 및 알고리즘

## 1. 기호 정의

### 1.1 기본 파라미터
- $p$: 소수 (modulus)
  - 예: secp256k1의 `0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F`
- $R$: Montgomery 상수 = $2^{256}$
- $a, b$: 일반 형식(normal form)의 숫자

---

### 1.2 Montgomery 형식
- $a_{\text{mont}} = a \cdot R \bmod p$: $a$를 Montgomery 형식으로 변환
- $b_{\text{mont}} = b \cdot R \bmod p$: $b$를 Montgomery 형식으로 변환
- $(a \cdot b)_{\text{mont}} = a \cdot b \cdot R \bmod p$: 곱셈 결과의 Montgomery 형식

---

### 1.3 역원 (Modular Inverses)

#### $p^{-1}$: $p$의 $\bmod R$ 역원
$$p \cdot p^{-1} \equiv 1 \pmod{R}$$
- **용도**: REDC에서 $m$ 계산 시 사용
- **계산**: 초기화 때 한 번만 (확장 유클리드 알고리즘)

#### $R^{-1}$: $R$의 $\bmod p$ 역원
$$R \cdot R^{-1} \equiv 1 \pmod{p}$$
- **용도**: REDC 결과의 수학적 의미 ($T \cdot R^{-1} \bmod p$)
- **직접 사용 안 함**: 알고리즘 증명에만 등장

---

### 1.4 REDC 알고리즘 변수

| 기호 | 정의 | 크기 | 의미 |
|------|------|------|------|
| $T$ | $a_{\text{mont}} \times b_{\text{mont}}$ | 512비트 | 곱셈 중간 결과 |
| $p'$ | $-p^{-1} \bmod R$ | 256비트 | 미리 계산된 상수 |
| $m$ | $T \cdot p' \bmod R$ | 256비트 | 보정값 |
| $t$ | $(T + m \cdot p) / R$ | 256비트 | 최종 결과 |

---

## 2. REDC 알고리즘

### 2.1 목표
$$\text{REDC}(T) = T \cdot R^{-1} \bmod p$$

입력 $T$ (512비트)를 받아서 $T \cdot R^{-1} \bmod p$ (256비트)를 계산

---

### 2.2 계산 단계

#### Step 1: $m$ 계산
$$m = (T \cdot p') \bmod R$$

**의미**: $(T + m \cdot p)$가 $R$의 배수가 되도록 하는 값

[목표]
T * R^(-1) mod p를 빠르게 계산

↓

[아이디어]
T를 R로 나누면 되는데, R = 2^256이니까 shift 쓰자!

↓

[문제]
T는 R의 배수가 아니라서 shift만으로는 안 됨

↓

[해결책 1]
T에 뭔가를 더해서 R의 배수로 만들자

↓

[해결책 2]
더하는 값은 "p의 배수" 형태여야 mod p에서 값이 안 바뀜

↓

[결론]
T + m*p 형태를 찾자!
- (T + m*p) ≡ 0 (mod R)  ← shift 가능
- (T + m*p) ≡ T (mod p)  ← 값 보존

**증명**:
$$
\begin{align}
p' &\equiv -p^{-1} \pmod{R} \\
m &\equiv -T \cdot p^{-1} \pmod{R} \\
m \cdot p &\equiv -T \pmod{R} \\
T + m \cdot p &\equiv 0 \pmod{R} \quad \checkmark
\end{align}
$$

**구현**: 하위 256비트만 취함 (비트 마스킹)
```rust
let m = (T.low_256() * P_PRIME) & MASK_256;
```

---

#### Step 2: $t$ 계산
$$t = \frac{T + m \cdot p}{R}$$

**의미**: $(T + m \cdot p)$는 $R$의 배수이므로 정확히 나누어떨어짐

**구현**: $R = 2^{256}$이므로 256비트 우측 시프트
```rust
let t = (T + (m as U512) * (p as U512)) >> 256;
```

---

#### Step 3: 범위 조정
$$
\text{result} = 
\begin{cases}
t - p & \text{if } t \geq p \\
t & \text{otherwise}
\end{cases}
$$

**이유**: $t$는 $[0, 2p)$ 범위일 수 있으므로 $[0, p)$로 조정

---

### 2.3 정확성 증명

**증명**: $t \equiv T \cdot R^{-1} \pmod{p}$

$$
\begin{align}
t &= \frac{T + m \cdot p}{R} \\
\\
t \cdot R &= T + m \cdot p \\
\\
t \cdot R &\equiv T \pmod{p} 
\quad (\because m \cdot p \equiv 0 \pmod{p}) \\
\\
t &\equiv T \cdot R^{-1} \pmod{p} \quad \checkmark
\end{align}
$$

---

## 3. Montgomery 곱셈 전체

### 3.1 입력
- $a_{\text{mont}} = a \cdot R \bmod p$
- $b_{\text{mont}} = b \cdot R \bmod p$

### 3.2 출력
- $(a \cdot b)_{\text{mont}} = a \cdot b \cdot R \bmod p$

### 3.3 과정

**Step 1**: 일반 곱셈
$$
\begin{align}
T &= a_{\text{mont}} \times b_{\text{mont}} \\
&= (a \cdot R) \times (b \cdot R) \\
&= a \cdot b \cdot R^2 \bmod p
\end{align}
$$

**Step 2**: REDC로 $R$ 하나 제거
$$
\begin{align}
\text{REDC}(T) &= T \cdot R^{-1} \bmod p \\
&= (a \cdot b \cdot R^2) \cdot R^{-1} \bmod p \\
&= a \cdot b \cdot R \bmod p \\
&= (a \cdot b)_{\text{mont}} \quad \checkmark
\end{align}
$$

---

## 4. 핵심 요약

### 4.1 두 가지 역원

| 역원 | 정의 | 모듈로 | 사용처 |
|------|------|--------|--------|
| $p^{-1}$ | $p \cdot p^{-1} \equiv 1$ | $\bmod R$ | $m$ 계산 (실제 코드) |
| $R^{-1}$ | $R \cdot R^{-1} \equiv 1$ | $\bmod p$ | 수학적 의미 (증명) |

### 4.2 REDC의 두 얼굴

| 관점 | 표현 |
|------|------|
| **수학적 의미** | $T \cdot R^{-1} \bmod p$ |
| **계산 방법** | $m = T \cdot p' \bmod R$, then $(T + m \cdot p) / R$ |

### 4.3 왜 빠른가?

| 일반 곱셈 | Montgomery 곱셈 |
|-----------|-----------------|
| 512비트 ÷ 256비트 나눗셈 | 256비트 마스킹 + shift + 뺄셈 1회 |
| 수십~수백 사이클 | 몇 사이클 |

**속도 차이: 10~100배**

---

## 5. 구현 예시

```rust
// 초기화 (한 번만)
const P: U256 = /* 소수 */;
const R: U256 = U256::MAX; // 2^256 - 1 (mask용)
const P_PRIME: U256 = compute_p_inverse(P); // -p^(-1) mod R

// REDC
fn redc(T: U512, p: U256, p_prime: U256) -> U256 {
    // Step 1: m = (T * p') mod R
    let m = (T.low_256() * p_prime) & R;
    
    // Step 2: t = (T + m*p) / R
    let t = (T + (m as U512) * (p as U512)) >> 256;
    
    // Step 3: 범위 조정
    if t >= p { 
        (t - p) as U256 
    } else { 
        t as U256 
    }
}

// Montgomery 곱셈
fn mont_mul(a_mont: U256, b_mont: U256) -> U256 {
    let T = (a_mont as U512) * (b_mont as U512);
    redc(T, P, P_PRIME)
}
```
### 핵심 질문

> `(a * b) mod p`를 하려면 나눗셈이 필요한데, 나눗셈은 곱셈보다 10배 이상 느리다. 어떻게 피하지?

### 일반 모듈러 곱셈의 문제

```
일반 방법:
1. a * b = product          (곱셈: 빠름)
2. product mod p = result    (나눗셈: 느림!)

예: 6 * 7 mod 11 = 42 mod 11 = 42 - 3*11 = 9
                                    ↑ 나눗셈/나머지 연산
```

ZK 증명에서는 모듈러 곱셈을 **수백만 번** 한다. 매번 나눗셈하면 느리다.

## Montgomery의 아이디어

Peter Montgomery (1985):
> "숫자를 저장할 때 $a$ 대신 $a \cdot R \bmod p$를 저장하면, 나눗셈 대신 비트 시프트로 곱셈할 수 있다."

$R = 2^{256}$ (= $2^{64 \times 4}$, limb 수 × 64)

$$
\begin{align}
\text{일반 표현 (normal form):} \quad & a \\
\text{Montgomery 표현 (mont form):} \quad & a_{\text{mont}} = a \cdot R \bmod p
\end{align}
$$

---

## 일반 곱셈 vs Montgomery 곱셈의 mod p 방식

### 1. 일반 곱셈의 mod p

$$c = (a \cdot b) \bmod p$$

**문제점:**
- $a \cdot b$는 최대 512비트 (256비트 × 2)
- 이걸 256비트 소수 $p$로 나눠야 함
- **512비트 ÷ 256비트 나눗셈** 필요 → 매우 느림

---

### 2. Montgomery 곱셈의 mod p (REDC)

## REDC 알고리즘

### 목표 (수학적 의미)
$$\text{출력} = T \cdot R^{-1} \bmod p$$

### 입력
$$T \quad \text{(최대 512비트)}$$

### 계산 방법

**1단계:** $m$을 계산
$$m \equiv -T \cdot p^{-1} \pmod{R}$$

**2단계:** $t$를 계산
$$t = \frac{T + m \cdot p}{R}$$

**3단계:** 범위 조정
$$
\text{if } t \geq p: \text{ return } t - p \\
\text{else: return } t
$$

### 왜 이게 $T \cdot R^{-1} \bmod p$가 되나?

**증명:**
$$
\begin{align}
t &= \frac{T + m \cdot p}{R} \\
t \cdot R &= T + m \cdot p \\
t \cdot R &\equiv T \pmod{p} \quad (\because m \cdot p \equiv 0 \pmod{p}) \\
t &\equiv T \cdot R^{-1} \pmod{p}
\end{align}
$$

REDC 알고리즘의 핵심:
REDC 전체 알고리즘 = R^(-1) 곱셈의 효과
$$
\begin{align}
\text{입력:} \quad & T = a_{\text{mont}} \cdot b_{\text{mont}} \text{ (최대 512비트)} \\
\text{목표:} \quad & T \cdot R^{-1} \bmod p
\end{align}
$$

**단계:**

$$
\begin{align}
1. \quad & m = (T \cdot p') \bmod R \quad \text{where } p' = -p^{-1} \bmod R \\
2. \quad & t = \frac{T + m \cdot p}{R} \quad \text{(이 나눗셈이 정확히 떨어짐!)} \\
3. \quad & \text{if } t \geq p: \text{ return } t - p \text{ else: return } t
\end{align}
$$

### 원하는 조건

$$T + m \cdot p \equiv 0 \pmod{R}$$

### 양변 정리

$$m \cdot p \equiv -T \pmod{R}$$

### 양변에 $p^{-1}$ 곱하기

$p^{-1}$은 "$p$의 $\bmod R$에서의 역원"입니다.

즉: 
$$p \cdot p^{-1} \equiv 1 \pmod{R}$$

따라서:

$$
\begin{align}
m \cdot p \cdot p^{-1} &\equiv -T \cdot p^{-1} \pmod{R} \\
m \cdot 1 &\equiv -T \cdot p^{-1} \pmod{R} \\
m &\equiv -T \cdot p^{-1} \pmod{R}
\end{align}
$$

**끝!** 이게 바로 $m$의 공식입니다.

**핵심 차이점:**

| 단계 | 연산 | 비용 |
|------|------|------|
| $m = (T \cdot p') \bmod R$ | **256비트만 취함** (하위 비트) | 비트 마스킹 (거의 공짜) |
| $t = (T + m \cdot p) / R$ | $R$로 나눗셈 = **256비트 시프트** | 거의 공짜 |
| $\text{if } t \geq p$ | 비교 + 뺄셈 1회 | 매우 빠름 |

---

### $(T + m \cdot p)$가 $R$의 배수인 이유

$$
\begin{align}
m &\equiv -T \cdot p^{-1} \pmod{R} \\
\\
\therefore T + m \cdot p &\equiv T - T \cdot p^{-1} \cdot p \pmod{R} \\
&\equiv T - T \pmod{R} \\
&\equiv 0 \pmod{R}
\end{align}
$$

즉, $(T + m \cdot p)$는 **수학적으로 보장되게** $R$의 배수이므로, $/R$은 **나머지가 0인 정확한 나눗셈**입니다.

---

### 결론

- **"mod p를 안 한다"가 아니라 "느린 나눗셈을 안 한다"**
- 일반 mod: 512÷256 긴 나눗셈
- Montgomery: 256비트 마스킹 + 256비트 시프트 + 조건부 뺄셈 1회
- **속도 차이: 약 10~100배**

---

### $R^{-1}$ 표기의 의미

### 모듈러 역원의 정의

$$R \cdot R^{-1} \equiv 1 \pmod{p}$$

**예시:** $p = 17$, $R = 2^8 = 256$일 때:

$$
\begin{align}
256 \cdot R^{-1} &\equiv 1 \pmod{17} \\
256 &\equiv 1 \pmod{17} \quad (\because 256 = 15 \times 17 + 1) \\
\therefore R^{-1} &\equiv 1 \pmod{17}
\end{align}
$$

---

### 왜 "$/R$"이 아니라 "$\cdot R^{-1}$"로 쓰나?

**모듈러 연산에서는 나눗셈이 정의되지 않음**

일반 산술:
$$10 / 2 = 5$$

모듈러 산술:
$$
\begin{align}
10 / 2 \pmod{7} &= \text{ ?  ← 의미 없음!} \\
\\
\text{대신:} \quad 10 \cdot 2^{-1} &\pmod{7} \\
&= 10 \cdot 4 \pmod{7} \quad (\because 2 \cdot 4 \equiv 1 \pmod{7}) \\
&= 40 \pmod{7} \\
&= 5
\end{align}
$$

---

## Montgomery 전체 흐름

### 변환: normal → Montgomery
$$a \rightarrow a \cdot R \bmod p$$

### Montgomery 곱셈
$$
\begin{align}
(a \cdot R) \cdot (b \cdot R) &= a \cdot b \cdot R^2 \\
\downarrow \text{ REDC} &\text{ (수학적으로 } R^{-1} \text{ 곱셈)} \\
a \cdot b \cdot R &
\end{align}
$$

### 변환: Montgomery → normal
$$(a \cdot R) \cdot R^{-1} \bmod p = a$$

---

## 저장 vs 연산

### ✅ "a 대신 $a \cdot R \bmod p$를 저장한다"
→ **데이터 표현 방식** 설명  
→ 완벽히 정확

### ✅ "REDC는 $T \cdot R^{-1} \bmod p$를 계산한다"
→ **알고리즘의 수학적 의미** 설명  
→ 완벽히 정확

| 표현                            | 맥락     | 정확성  |
| ----------------------------- | ------ | ---- |
| "$a \cdot R \bmod p$로 저장"     | 데이터 표현 | ✅ 정확 |
| "$T \cdot R^{-1} \bmod p$ 계산" | 연산 의미  | ✅ 정확 |

**둘 다 맞고, 서로 다른 측면을 설명하는 것입니다.**

### 필요한 상수 3개

#### R = 2^256 mod p

```rust
const R: [u64; 4] = [
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
];
```

1의 Montgomery form. `Fp::ONE = Fp(R)`이 되는 이유:

```
1_mont = 1 * R mod p = R mod p
```

#### R^2 mod p

```rust
const R2: [u64; 4] = [
    0xf32cfc5b538afa89,
    0xb5e71911d44501fb,
    0x47ab1eff0a417ff6,
    0x06d89f71cab8351f,
];
```

normal → Montgomery 변환에 사용:

```
from_raw(a) = mont_mul(a, R^2)
            = a * R^2 * R^{-1}    (mont_mul은 R^{-1}을 곱해주므로)
            = a * R
            = a_mont ✓
```

> [!tip] R^2를 미리 계산해두는 이유 변환할 때마다 $R^2 \bmod p$를 구하면 느리다. 한 번 계산해서 상수로 박아두면 변환은 그냥 mont_mul 한 번.

#### INV = -p^{-1} mod 2^64

```rust
const INV: u64 = {
    let p0 = MODULUS[0];
    let mut inv = 1u64;
    inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
    // ... 6번 반복
    inv.wrapping_neg()
};
```

REDC에서 "하위 limb을 0으로 만드는" 마법 상수. 아래 REDC 섹션에서 상세 설명.

# Montgomery Reduction - 왜 limb별로 계산하는가

## 핵심 질문

> `p`가 256비트인데 왜 `INV`는 64비트(`mod 2^64`)로 충분한가?

---

## 직관: "빚을 한 줄씩 갚는다"

256비트 정수를 4개의 64비트 limb로 생각하자.

```
p = [p0, p1, p2, p3]   ← 각각 64비트
T = [T0, T1, T2, T3, T4, T5, T6, T7]  ← 곱셈 결과, 최대 8 limb
```

Montgomery reduction의 목표: $$T \cdot R^{-1} \mod p \quad \text{where } R = 2^{256}$$

이걸 한 번에 계산하면 256비트 나눗셈이 필요하다. **비싸다.**

대신 이런 트릭을 쓴다:

> **"T의 최하위 64비트를 0으로 만드는 배수를 p에서 찾아서 빼라. 4번 반복하면 하위 256비트가 전부 0이 되고, 256비트 오른쪽 시프트로 나눗셈이 공짜가 된다."**

---

## Step-by-step: 하위 64비트를 0으로 만들기

### 1라운드: T[0]을 없앤다

`T[0]`을 0으로 만들고 싶다. 어떤 `m`을 찾아서 `T - m*p`의 최하위 64비트가 0이 되게 하면 된다.

$$T[0] + m \cdot p[0] \equiv 0 \pmod{2^{64}}$$

$$m \equiv -T[0] \cdot p[0]^{-1} \pmod{2^{64}}$$

여기서 `INV = -p[0]^{-1} mod 2^64` 라고 정의하면:

$$m = T[0] \cdot \text{INV} \pmod{2^{64}}$$

**포인트**: `m`을 구할 때 `p[0]`(하위 64비트)만 쓴다. `p[1], p[2], p[3]`은 필요 없다.

```
T' = T + m * p
   = [0, T1', T2', T3', T4', T5', T6', T7']
         ↑ carry가 올라갔을 뿐, T[0]은 0이 됨
```

이제 `T' >> 64` → 7 limb짜리 숫자가 됨.

### 2라운드: T'[0]을 없앤다

완전히 같은 과정. 새로운 최하위 limb에 대해 같은 `INV`를 쓴다.

$$m = T'[0] \cdot \text{INV} \pmod{2^{64}}$$

### 3, 4라운드 반복

4번 반복하면 하위 256비트가 전부 0 → `>> 256`은 그냥 상위 limb들만 취하면 됨.

---

## 왜 INV는 64비트로 충분한가

각 라운드에서 우리가 실제로 계산하는 건:

```
m = T_current[0] * INV  mod 2^64
```

오직 **64비트 곱셈 하나**. 256비트짜리 역원이 필요하지 않다.

`p[0]`(하위 limb)만으로 `m`을 결정하고, `m * p` 전체를 더해서 carry를 위로 올리는 구조이기 때문이다.



```
p[0] * INV
= p[0] * (-p[0]^{-1})
= -(p[0] * p[0]^{-1})
= -1  (mod 2^64)

필요한 조건:  p[0] * INV ≡ -1 (mod 2^64)
충분한 이유:  이 조건만 있으면 매 라운드에서 최하위 limb를 정확히 소거 가능
```

---

## 왜 결과가 +1이 아니라 -1인가

`m`을 구하는 식을 다시 보자:

$$T[0] + m \cdot p[0] \equiv 0 \pmod{2^{64}}$$ $$m = -T[0] \cdot p[0]^{-1} \pmod{2^{64}}$$

이를 `m = T[0] * INV` 형태로 쓰려면:

$$\text{INV} = -p[0]^{-1} \pmod{2^{64}}$$

Newton's method로 먼저 `p[0]^{-1}`을 구하고, 마지막에 `.wrapping_neg()`로 부호를 뒤집는 것이 바로 이 때문이다.

```rust
// Newton's method → p[0]^{-1} mod 2^64
let mut inv = 1u64;
inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
// ... 6번 반복 (64비트 수렴에 필요)

// 마지막에 부호 반전 → -p[0]^{-1} mod 2^64
inv.wrapping_neg()
```

---

## 전체 구조 요약

```
목표:  T * R^{-1} mod p  (R = 2^256)

방법:
  Round 1: m₀ = T[0] * INV mod 2^64
           T ← (T + m₀*p) >> 64

  Round 2: m₁ = T[0] * INV mod 2^64
           T ← (T + m₁*p) >> 64

  Round 3: 동일
  Round 4: 동일

  결과: T mod p  (조건부 subtraction 1번)
```

|항목|값|이유|
|---|---|---|
|`R`|`2^256`|4 limb × 64비트|
|`INV` 의 mod|`2^64`|매 라운드에서 64비트 소거만 필요|
|`INV` 의 값|`-p[0]^{-1}`|`m = T[0] * INV`로 바로 계산하기 위해|
|Newton 반복 횟수|6회|`1→2→4→8→16→32→64` 비트 수렴|

---

## 관련 개념

- [[Montgomery Multiplication]]
- [[Modular Arithmetic]]
- [[CIOS Algorithm]] ← 실제 구현에서 많이 쓰는 변형

### INV를 Newton's method로 구하는 과정

목표: $p_0^{-1} \bmod 2^{64}$를 구하고 negate

Newton's method: $x_{n+1} = x_n \cdot (2 - p_0 \cdot x_n) \bmod 2^{64}$

```
x_0 = 1                          (1-bit 정밀도)
x_1 = 1 * (2 - p0 * 1)           (2-bit 정밀도)
x_2 = x_1 * (2 - p0 * x_1)       (4-bit 정밀도)
x_3 = ...                        (8-bit)
x_4 = ...                        (16-bit)
x_5 = ...                        (32-bit)
x_6 = ...                        (64-bit) ← 완성!
```

> [!note] 왜 6번이면 충분한가? 각 반복마다 정밀도가 2배. $2^6 = 64$이므로 6번이면 64-bit 전체를 커버한다.

> [!note] 왜 const로 컴파일 타임에? 런타임에 계산할 필요 없이 컴파일 시점에 확정된다. `wrapping_mul`, `wrapping_sub`은 const fn에서 사용 가능.

### INV의 검증

```rust
MODULUS[0].wrapping_mul(INV).wrapping_add(1) == 0
// 즉: p0 * INV ≡ -1 (mod 2^64)
```

왜 이 성질이 필요한지는 REDC에서 설명.

### REDC 알고리즘 상세

8-limb 곱셈 결과 `t[0..8]`에서 4-limb 결과를 추출한다.

```
목표: t * R^{-1} mod p를 구하는데, 나눗셈 없이!

아이디어: t에 p의 적절한 배수를 더해서 하위 4 limb을 전부 0으로 만든다
         → 그러면 상위 4 limb이 t/R mod p가 된다 (하위가 0이니까 그냥 시프트)
```

한 반복의 동작 (i번째 limb 처리):

```
1. m = r[i] * INV mod 2^64
   → m을 이렇게 고르면 (r[i] + m * p0)의 하위 64-bit가 정확히 0이 된다!

2. r += m * p
   → r[i]가 0이 되고, 나머지는 상위로 전파

3. i를 증가시키며 4번 반복
   → 하위 4 limb 전부 0
```

> [!important] 왜 m = r[i] * INV로 하면 하위가 0이 되나?
> 
> `r[i] + m * p0 ≡ 0 (mod 2^64)` 이려면:
> 
> `m ≡ -r[i] * p0^{-1} (mod 2^64)`
> 
> `INV = -p0^{-1} mod 2^64` 이므로:
> 
> `m = r[i] * INV = r[i] * (-p0^{-1}) = -r[i] / p0`
> 
> → `r[i] + m * p0 = r[i] - r[i] = 0` ✓ (mod 2^64)

4번 반복 후:

```
전: [r0, r1, r2, r3, r4, r5, r6, r7]
후: [ 0,  0,  0,  0, r4, r5, r6, r7]  ← 하위 4개가 0
결과: [r4, r5, r6, r7]               ← 상위 4개가 답
```

마지막에 `sub_if_gte`로 결과가 p 이상이면 p를 뺀다.

### Schoolbook 4×4 곱셈

REDC 전에 먼저 4-limb × 4-limb = 8-limb 곱셈이 필요하다.

초등학교 세로 곱셈과 동일. 밑이 10 대신 $2^{64}$:

```
              a[3]  a[2]  a[1]  a[0]
           ×  b[3]  b[2]  b[1]  b[0]
           ─────────────────────────
              a[0]*b[0]  ← t[0]에
        a[0]*b[1]        ← t[1]에 누적
  a[0]*b[2]              ← t[2]에 누적
  ...
        a[1]*b[0]        ← t[1]에 누적 (여기서 mac의 acc 필요!)
  ...
```

`mac(acc, a, b, carry)`가 여기서 진가를 발휘한다:

- `acc`: 이전 단계에서 이 위치에 이미 쌓인 값
- `a * b`: 현재 곱셈
- `carry`: 이전 곱셈의 상위 64-bit (올림)

### from_raw: normal → Montgomery

```rust
pub fn from_raw(v: [u64; 4]) -> Self {
    Fp(v).mont_mul(&Fp(R2))
}
```

왜 이게 되나?

```
mont_mul(a, R^2) = a * R^2 * R^{-1} mod p
                 = a * R mod p
                 = a_mont ✓
```

### to_repr: Montgomery → normal

```rust
pub fn to_repr(&self) -> [u64; 4] {
    let mut t = [0u64; 8];
    t[0..4] = self.0;  // 하위 4 limb에 넣기
    mont_reduce_inner(&t)
}
```

8-limb 중 하위 4개만 채우고 REDC를 돌린다:

```
REDC([a_mont, 0, 0, 0, 0, 0, 0, 0])
= a_mont * R^{-1} mod p
= (a * R) * R^{-1} mod p
= a ✓
```

### Montgomery form에서 덧셈/뺄셈이 그대로 되는 이유

```
a_mont + b_mont = (a*R) + (b*R) = (a+b)*R = (a+b)_mont ✓
```

덧셈과 뺄셈은 Montgomery 변환 없이 그대로 쓸 수 있다. 곱셈만 REDC가 필요.

### 정리: 비용 비교

||일반|Montgomery|
|---|---|---|
|변환 비용|없음|from_raw/to_repr 각 1번|
|덧셈|mod p|동일 (그대로)|
|**곱셈**|**나눗셈 필요**|**REDC (시프트 + 덧셈)**|
|적합한 경우|곱셈 1~2번|곱셈 수백만 번 (ZK 증명)|

> [!tip] 핵심 통찰 Montgomery form은 "입장료"(from_raw)와 "퇴장료"(to_repr)를 내는 대신, 안에서 곱셈을 할 때마다 나눗셈을 아낀다. ZK 증명처럼 곱셈을 수백만 번 하는 상황에서는 압도적으로 유리하다.

## Step 2-5: square, pow, inv

### 핵심 질문

> mont_mul 하나만 있으면 제곱, 거듭제곱, 역원을 전부 만들 수 있다?

그렇다. 세 함수 모두 mont_mul의 조합일 뿐이다.

### square: 제곱

```rust
pub fn square(&self) -> Fp {
    self.mont_mul(self)
}
```

`a * a`를 하는 것. 끝.

> [!note] 왜 별도 함수로 분리하나? 지금은 `mont_mul(self, self)` 호출이지만, 나중에 성능 최적화할 때 제곱 전용 알고리즘으로 교체할 수 있다. `a[i] * a[j]`와 `a[j] * a[i]`가 같으므로 곱셈 횟수를 거의 절반으로 줄일 수 있다. 인터페이스를 미리 분리해두면 내부만 바꾸면 된다.

### pow: 거듭제곱 (Square-and-Multiply)

$a^{10}$을 구한다고 하자.

**순진한 방법**: a를 10번 곱하기 → 곱셈 9번

**Square-and-multiply**: 지수를 이진수로 보기

```
10 = 1010₂

비트를 오른쪽(LSB)부터 순회:
  bit 0 = 0: skip          base = a^1 → square → a^2
  bit 1 = 1: result *= a^2  base = a^2 → square → a^4
  bit 2 = 0: skip          base = a^4 → square → a^8
  bit 3 = 1: result *= a^8

result = a^2 * a^8 = a^10 ✓
```

곱셈 횟수: 매 비트마다 square 1번 + (비트가 1이면) multiply 1번 → 254-bit 지수에서 최대 **254 + 254 = 508번** (vs 순진하게 하면 $2^{254}$번)

```rust
pub fn pow(&self, exp: &[u64; 4]) -> Fp {
    let mut result = Fp::ONE;  // 누적 결과, 1부터 시작
    let mut base = *self;      // 현재 a^{2^i}

    for &limb in exp.iter() {      // limb 4개 순회
        for j in 0..64 {           // 각 limb의 64비트 순회
            if (limb >> j) & 1 == 1 {
                result = result.mont_mul(&base);  // 비트가 1이면 곱하기
            }
            base = base.square();  // 매번 제곱
        }
    }
    result
}
```

> [!tip] 왜 LSB부터 보나? MSB부터 보는 방법(left-to-right)도 있다. LSB 방식은 base를 매번 제곱하면서 result에 곱하는 구조라 구현이 더 단순하다. 성능 차이는 거의 없다.

#### 동작 예시: 3^10

```
exp = [10, 0, 0, 0]
10 = 0b1010

limb = 10:
  j=0: bit=0, skip.       base = 3^1 → 3^2
  j=1: bit=1, result *= 3^2 = 3^2.    base = 3^2 → 3^4
  j=2: bit=0, skip.       base = 3^4 → 3^8
  j=3: bit=1, result *= 3^8 = 3^2 * 3^8 = 3^10.  base = 3^8 → 3^16
  j=4~63: 전부 0, skip.

나머지 3개 limb도 전부 0.
result = 3^10 = 59049 ✓
```

### inv: 역원 (Fermat의 소정리)

$a^{-1}$을 구하는 방법은 여러 가지가 있다:

1. 확장 유클리드 알고리즘 (Extended GCD)
2. **Fermat의 소정리**: $a^{p-1} \equiv 1 \pmod{p}$

Fermat을 쓰면:

```
a^{p-1} ≡ 1 (mod p)

양변을 a로 나누면:
a^{p-2} ≡ a^{-1} (mod p)
```

이미 `pow`가 있으니 그냥 호출하면 된다:

```rust
pub fn inv(&self) -> Option<Fp> {
    if self.is_zero() {
        return None;  // 0에는 역원이 없다
    }
    Some(self.pow(&[MODULUS[0] - 2, MODULUS[1], MODULUS[2], MODULUS[3]]))
}
```

> [!important] 왜 Option을 반환하나? 유한체에서 0을 제외한 모든 원소는 역원을 가진다. 0은 역원이 없다 ($0 \cdot x = 0 \neq 1$). `None`으로 이를 표현한다.

> [!note] p - 2를 limb으로 표현 `p - 2`는 `p`에서 최하위 limb만 2를 빼면 된다: `[MODULUS[0] - 2, MODULUS[1], MODULUS[2], MODULUS[3]]`
> 
> p의 최하위 limb이 `0x3c208c16d87cfd47`이므로 2를 빼도 underflow가 일어나지 않는다.

### inv의 비용

254-bit 지수로 pow → 약 254번 square + ~127번 multiply ≈ **381번의 mont_mul**

비싸지만, ZK에서 역원은 곱셈보다 훨씬 드물게 쓰인다. 대부분의 연산은 add/mul이고 inv는 가끔.

### 검증: p-1의 제곱이 1인 이유

```rust
fn p_minus_one_squared_is_one() {
    let p_minus_1 = Fp::from_raw([MODULUS[0] - 1, MODULUS[1], MODULUS[2], MODULUS[3]]);
    assert_eq!(p_minus_1.square(), Fp::ONE);
}
```

$p - 1 \equiv -1 \pmod{p}$이므로 $(-1)^2 = 1$.

이 테스트는 from_raw, mont_mul, REDC가 큰 값에서도 정확히 동작하는지 한꺼번에 검증한다. `(p-1) * (p-1)`은 거의 최대 크기의 곱셈이라 edge case 테스트로 좋다.

### 정리: 의존 관계

```
mac (Step 2-2)
 └→ mont_mul (Step 2-4)
     ├→ square = mont_mul(self, self)
     ├→ pow = square + mont_mul 반복
     └→ inv = pow(p-2)
```

모든 것이 `mac` → `mont_mul` 위에 쌓여 있다.