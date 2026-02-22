## Step 2-1: 254-bit 소수를 `[u64; 4]`로 표현하기

### 핵심 질문

> 컴퓨터는 최대 64-bit 정수까지 네이티브로 다루는데, 254-bit 소수를 어떻게 저장하지?

### BN254란?

- **BN** = Barreto-Naehrig, 페어링에 적합한 타원곡선 패밀리
- **254** = base field 소수 p의 bit 길이
- 이 p 위에서 타원곡선 $y^2 = x^3 + 3$을 정의한다
- Ethereum의 `ecAdd`, `ecMul`, `ecPairing` precompile이 이 커브를 사용

### 왜 `[u64; 4]`인가?

```
p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
```

이 숫자는 254-bit. 한 개의 `u64`는 64-bit만 담을 수 있다.

```
254 / 64 = 3.96... → 최소 4개 필요
4 x 64 = 256 >= 254 ✓
```

### limb 란?
맞네요, 갑자기 limb이란 단어를 설명 없이 썼습니다. 보충합니다.

---

## Limb이란?

"limb"은 원래 "팔다리"라는 뜻인데, 큰 숫자의 **한 부분**이라는 비유에서 온 용어다. GMP(GNU Multiple Precision) 라이브러리에서 시작되어 업계 표준 용어가 됐다.

### 비유: 10진수로 생각하기

계산기가 4자리까지만 다룰 수 있다고 하자.

```
123,456,789를 저장하려면?

→ 4자리씩 쪼갠다:
  limb[0] = 6789  (최하위)
  limb[1] = 2345
  limb[2] = 1     (최상위)

복원: 1 * 10000^2 + 2345 * 10000^1 + 6789 * 10000^0
    = 123,456,789
```

### 64-bit limb

실제 컴퓨터에서는 CPU가 64-bit 연산을 네이티브로 지원하므로, 64-bit 단위로 쪼갠다:

```
254-bit 소수 p를 저장:

limb[0] = 0x3c208c16d87cfd47   ← 최하위 64-bit
limb[1] = 0x97816a916871ca8d
limb[2] = 0xb85045b68181585d
limb[3] = 0x30644e72e131a029   ← 최상위 64-bit

복원: limb[0] + limb[1] * 2^64 + limb[2] * 2^128 + limb[3] * 2^192
```

10진수에서 "자릿수"에 해당하는 것이 limb이다. 다만 밑(base)이 10이 아니라 $2^{64}$인 것.

### 관련 용어

| 용어                      | 의미                                       |
| ----------------------- | ---------------------------------------- |
| **limb**                | 큰 수의 한 조각 (여기서는 u64 하나)                  |
| **limb count**          | 조각 개수 (여기서는 4)                           |
| **little-endian limbs** | limb[0]이 최하위. carry가 낮은 인덱스 → 높은 인덱스로 전파 |
| **big-endian limbs**    | limb[0]이 최상위. 사람이 읽기엔 자연스럽지만 산술엔 불편      |

### Little-endian limb 배치

```
MODULUS = [
    0x3c208c16d87cfd47,  // limbs[0]: 최하위 64-bit
    0x97816a916871ca8d,  // limbs[1]
    0xb85045b68181585d,  // limbs[2]
    0x30644e72e131a029,  // limbs[3]: 최상위 64-bit
]
```

숫자로 복원하면:

```
p = limbs[0] + limbs[1] * 2^64 + limbs[2] * 2^128 + limbs[3] * 2^192
```

> [!tip] 왜 little-endian? carry(올림)가 낮은 인덱스 → 높은 인덱스로 전파되므로, `for i in 0..4` 순회가 자연스럽다. CPU의 덧셈 방향과 일치.

### 254-bit 확인

```rust
let top = MODULUS[3]; // 0x30644e72e131a029
let bit_len = 64 - top.leading_zeros(); // 64 - 2 = 62
let total_bits = 192 + bit_len; // 192 + 62 = 254 ✓
```

`0x30...`은 이진수로 `0011 0000...` → 상위 2비트가 0 → 62-bit 사용.

### Fp 구조체

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fp(pub(crate) [u64; 4]);
```

|derive|이유|
|---|---|
|`Clone, Copy`|256-bit는 충분히 작아서 스택 복사 OK|
|`PartialEq, Eq`|`==` 비교. Montgomery form에서는 내부 limb 비교만으로 충분|
|`Debug`|디버깅 출력|

`pub(crate)` — 같은 크레이트 내에서만 내부 limb 직접 접근 가능. 외부에서는 `from_raw`/`to_repr`로만 접근.

### ZERO

```rust
pub const ZERO: Fp = Fp([0, 0, 0, 0]);
```

나중에 Montgomery form을 쓰게 되면:

- `0`의 Montgomery form = `0 * R mod p = 0` → 그대로 `[0,0,0,0]`
- 즉 ZERO는 Montgomery 도입 전후로 값이 안 변한다

---

## Step 2-2: 큰 수 산술의 빌딩 블록 (adc, sbb, mac)

### 핵심 질문

> `[u64; 4]` 두 개를 더하면 어떻게 되지? 각 limb끼리 더하면 넘칠 수 있잖아.

초등학교 때 세로 덧셈에서 올림하던 것과 같다:

```
  37
+ 85
----
  22  (7+5=12, 2를 쓰고 1을 올림)
 1    (3+8+1=12)
= 122
```

64-bit에서도 똑같이 carry를 전파한다.

### adc (Add with Carry)

```rust
fn adc(a: u64, b: u64, carry: bool) -> (u64, bool)
```

**역할**: `a + b + carry_in` → `(합, carry_out)`

```
limb[0]: 0x3c208c16d87cfd47 + 0xffffffffffffffff = ?
         ↓ 넘침!
         (하위 64-bit 결과, carry = true)

limb[1]: 다음 limb 덧셈 시 carry를 같이 더한다
```

**구현 원리**:

```rust
let (s1, c1) = a.overflowing_add(b);     // a + b
let (s2, c2) = s1.overflowing_add(carry as u64); // + carry_in
(s2, c1 | c2)  // 어느 쪽이든 넘치면 carry_out = true
```

> [!note] `overflowing_add` Rust 표준 메서드. `(결과, 넘침여부)` 튜플 반환. `wrapping_add`는 결과만 반환하고 넘침 여부를 버린다.

**예시**:

```
adc(u64::MAX, 1, false)
= (0, true)   // 2^64 → 하위 0, carry 1

adc(3, 5, false)
= (8, false)   // 평범한 덧셈
```

### sbb (Subtract with Borrow)

```rust
fn sbb(a: u64, b: u64, borrow: bool) -> (u64, bool)
```

**역할**: `a - b - borrow_in` → `(차, borrow_out)`

adc의 뺄셈 버전. carry 대신 borrow(빌림)를 전파한다.

```
sbb(0, 1, false)
= (u64::MAX, true)  // 0 - 1 = underflow → 2^64 - 1, borrow 발생

sbb(10, 3, false)
= (7, false)        // 평범한 뺄셈
```

> [!important] borrow의 의미 borrow = true면 "이전 limb에서 1을 빌려왔다"는 뜻. 현재 limb에서 추가로 1을 빼야 한다.

### mac (Multiply-ACcumulate)

```rust
fn mac(acc: u64, a: u64, b: u64, carry: u64) -> (u64, u64)
```

**역할**: `acc + a * b + carry` → `(하위 64-bit, 상위 64-bit)`

이건 adc/sbb와 다르다. 두 64-bit를 곱하면 최대 **128-bit** 결과가 나온다:

```
u64::MAX * u64::MAX = (2^64 - 1)^2 = 2^128 - 2^65 + 1
```

그래서 `u128`로 확장해서 계산한다:

```rust
let wide = acc as u128 + (a as u128) * (b as u128) + carry as u128;
(wide as u64, (wide >> 64) as u64)
//  └ 하위 64-bit    └ 상위 64-bit
```

> [!tip] 왜 carry가 `u64`이고 `bool`이 아닌가? 곱셈의 carry는 최대 `u64` 크기다 (상위 64-bit 전체가 carry). 덧셈의 carry는 0 또는 1이라 `bool`.

**예시**:

```
mac(0, 3, 7, 0) = (21, 0)           // 평범한 곱셈
mac(0, u64::MAX, u64::MAX, 0) = (1, 0xFFFFFFFFFFFFFFFE)
// (2^64-1)^2 = 2^128 - 2^65 + 1
// lo = 1, hi = 2^64 - 2
```

### 세 함수의 관계

```
[u64; 4] 덧셈   →  adc를 4번 체이닝
[u64; 4] 뺄셈   →  sbb를 4번 체이닝
[u64; 4] 곱셈   →  mac으로 schoolbook 곱셈 (다음 단계에서)
```

```
   limbs[0]  limbs[1]  limbs[2]  limbs[3]
      +         +         +         +
   limbs[0]  limbs[1]  limbs[2]  limbs[3]
   ────────  ────────  ────────  ────────
   result[0] result[1] result[2] result[3]
       └─carry─┘  └─carry─┘  └─carry─┘
```

### `#[inline(always)]`를 붙인 이유

이 함수들은 한두 줄짜리 초경량 함수인데, 모듈러 곱셈 한 번에 수십 번 호출된다. 함수 호출 오버헤드를 없애기 위해 항상 인라인.

## Step 2-3: 모듈러 덧셈/뺄셈

### 핵심 질문

> 두 Fp 값을 더하면 결과가 p를 넘을 수 있다. 어떻게 `[0, p)` 범위를 유지하지?

### 일반 덧셈 vs 모듈러 덧셈

일반 정수에서는:

```
7 + 11 = 18 (끝)
```

유한체 $\mathbb{F}_p$에서는:

```
7 + 11 = 18          (18 < p이므로 그대로)
(p-1) + 1 = p        (p >= p이므로 p를 빼서 → 0)
(p-1) + (p-1) = 2p-2 (>= p이므로 p를 빼서 → p-2)
```

**규칙: 결과가 p 이상이면 p를 딱 한 번만 빼면 된다.**

왜 한 번이면 충분한가?

- 두 피연산자 모두 `[0, p)` 범위
- 최대 합 = `(p-1) + (p-1) = 2p - 2`
- `2p - 2 < 2p` → p를 한 번 빼면 `p - 2 < p` ✓

### add 구현: adc 4번 체이닝

```rust
pub fn add(&self, rhs: &Fp) -> Fp {
    let (d0, carry) = self.0[0].overflowing_add(rhs.0[0]);
    let (d1, carry) = adc(self.0[1], rhs.0[1], carry);
    let (d2, carry) = adc(self.0[2], rhs.0[2], carry);
    let (d3, _) = adc(self.0[3], rhs.0[3], carry);

    sub_if_gte([d0, d1, d2, d3])
}
```

그림으로 보면:

```
  self.0[0]   self.0[1]   self.0[2]   self.0[3]
+  rhs.0[0]    rhs.0[1]    rhs.0[2]    rhs.0[3]
  ─────────   ─────────   ─────────   ─────────
       d0    ←carry→ d1  ←carry→ d2  ←carry→ d3
```

> [!note] 첫 limb만 `overflowing_add`를 쓰는 이유 `adc`는 이전 carry를 받는 버전이고, 첫 limb은 이전 carry가 없으므로 `overflowing_add(rhs.0[0])`로 시작한다. `adc(self.0[0], rhs.0[0], false)`와 동일하다.

> [!note] 마지막 carry를 `_`로 버리는 이유 두 254-bit 수의 합은 최대 255-bit. `[u64; 4]` = 256-bit이므로 4개 limb에 항상 담긴다. 최상위 carry가 발생할 수 있지만, 어차피 `sub_if_gte`에서 p를 빼면 범위 안에 들어온다.

### sub_if_gte: "일단 빼보기" 트릭

```rust
fn sub_if_gte(v: [u64; 4]) -> Fp {
    let (d0, borrow) = v[0].overflowing_sub(MODULUS[0]);
    let (d1, borrow) = sbb(v[1], MODULUS[1], borrow);
    let (d2, borrow) = sbb(v[2], MODULUS[2], borrow);
    let (d3, borrow) = sbb(v[3], MODULUS[3], borrow);

    if borrow {
        Fp(v)                   // v < p → 원래 값
    } else {
        Fp([d0, d1, d2, d3])    // v >= p → 뺀 값
    }
}
```

비교 연산(`>=`) 없이 대소를 판단하는 방법:

1. 일단 `v - p`를 계산한다
2. borrow 발생 → `v < p` → 원래 v를 사용
3. borrow 없음 → `v >= p` → 뺀 결과를 사용

> [!tip] 왜 if문으로 비교 안 하나? `[u64; 4]`를 비교하려면 최상위 limb부터 하나씩 비교해야 한다 (4번 비교). sbb를 이미 돌려서 borrow 하나로 대소를 알 수 있으니, 뺄셈을 "시도"하는 게 더 간결하다.

### sub: underflow 처리

```rust
pub fn sub(&self, rhs: &Fp) -> Fp {
    let (d0, borrow) = self.0[0].overflowing_sub(rhs.0[0]);
    let (d1, borrow) = sbb(self.0[1], rhs.0[1], borrow);
    let (d2, borrow) = sbb(self.0[2], rhs.0[2], borrow);
    let (d3, borrow) = sbb(self.0[3], rhs.0[3], borrow);

    if borrow {
        // a < b → 음수 → p를 더한다
        ...
    }
}
```

예시: `3 - 7` in $\mathbb{F}_p$

```
3 - 7 = -4 (음수!)
-4 mod p = p - 4 (양수로 변환)

검증: (p - 4) + 7 = p + 3 ≡ 3 (mod p) ✓
```

borrow가 발생하면 결과에 p를 더해서 양수로 만든다. $-4$와 $p - 4$는 mod p에서 같은 값이다.

> [!important] 왜 한 번만 더하면 되나? add와 같은 논리. 두 피연산자가 `[0, p)` 범위이므로 최소 차이는 `0 - (p-1) = -(p-1)`. 여기에 p를 한 번 더하면 `1 > 0` ✓

### neg: 부정

```rust
pub fn neg(&self) -> Fp {
    if self.is_zero() {
        *self        // -0 = 0
    } else {
        // p - a     (= -a mod p)
    }
}
```

`-a mod p = p - a`

- `-0 mod p = 0` (특수 케이스: p - 0 = p인데, p mod p = 0이므로 그냥 0 반환)
- `-5 mod p = p - 5`
- 검증: `a + (-a) = a + (p - a) = p ≡ 0 (mod p)` ✓

### 정리

|함수|하는 일|핵심 트릭|
|---|---|---|
|`add`|`(a + b) mod p`|더하고, >= p이면 p를 뺀다|
|`sub`|`(a - b) mod p`|빼고, underflow면 p를 더한다|
|`neg`|`-a mod p`|`p - a`|
|`sub_if_gte`|정규화 헬퍼|"일단 빼보고 borrow로 판단"|

### 아직 남은 문제

> 덧셈/뺄셈은 됐는데 **곱셈**은? `(a * b) mod p`를 하려면 나눗셈이 필요하다. 나눗셈은 비싸다. → 다음 Step에서 Montgomery form으로 이 문제를 해결한다.