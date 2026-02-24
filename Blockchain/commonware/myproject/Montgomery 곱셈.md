##   
Step 2-4: Montgomery 곱셈

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

### Montgomery의 아이디어

Peter Montgomery (1985):

> "숫자를 저장할 때 $a$ 대신 $a \cdot R \bmod p$를 저장하면, 나눗셈 대신 비트 시프트로 곱셈할 수 있다."

$R = 2^{256}$ (= $2^{64 \times 4}$, limb 수 × 64)

```
일반 표현 (normal form):     a
Montgomery 표현 (mont form): a_mont = a * R mod p
```

### 왜 R = 2^256이면 나눗셈이 사라지나?

$2^{256}$으로 나누기 = **256-bit 오른쪽 시프트**. 컴퓨터에게 시프트는 거의 공짜.

단, `mod p`를 유지해야 하므로 순수 시프트만으로는 안 되고, **REDC** 알고리즘이 필요하다.

### Montgomery 곱셈의 흐름

```
입력: a_mont = a*R mod p,  b_mont = b*R mod p
목표: (a*b)_mont = a*b*R mod p

단계 1: a_mont * b_mont = a*b*R^2 mod p    (그냥 곱셈)
단계 2: REDC로 R을 하나 제거 → a*b*R mod p  (이게 (a*b)_mont!)
```

**REDC가 "R로 나누기"를 시프트로 해주는 것이 핵심.**

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