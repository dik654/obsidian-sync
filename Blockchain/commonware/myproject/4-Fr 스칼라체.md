# Step 3-1 · Fr 스칼라체 기본 구조

## Fp와 Fr — 왜 체가 두 개 필요한가?

BN254 커브에는 **소수가 두 개** 등장한다:

||Fp (base field)|Fr (scalar field)|
|---|---|---|
|**역할**|점의 **좌표** 범위|**스칼라** (점을 몇 번 더할지) 범위|
|**modulus**|p = 2188...08583|r = 2188...95617|
|**용도**|G1 좌표 `(x, y)` ∈ Fp|ZK witness, 회로 변수 ∈ Fr|

### 비유

> 지도의 좌표는 **위도/경도**(Fp)로 표현하고, "몇 km 이동"은 **거리**(Fr)로 표현한다. 같은 숫자 체계이지만 **의미가 다르다**.

### p와 r은 무엇이 다른가?

```
p = 0x30644e72e131a029 b85045b68181585d 97816a916871ca8d 3c208c16d87cfd47
r = 0x30644e72e131a029 b85045b68181585d 2833e84879b97091 43e1f593f0000001
    ^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^
    상위 2 limb 동일          하위 2 limb이 다름
```

둘 다 254-bit 소수. 상위 절반이 동일한 것은 BN254 커브 파라미터 구성 방식 때문이다.

### fr.rs — 최초 구조 (Step 3-1)

```rust
/// BN254 scalar field modulus r (curve order)
const MODULUS: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fr(pub(crate) [u64; 4]);

impl Fr {
    pub const ZERO: Fr = Fr([0, 0, 0, 0]);
    pub fn is_zero(&self) -> bool { self.0 == [0, 0, 0, 0] }
}
```

Fp와 **구조가 100% 동일**. `[u64; 4]` + little-endian. 다른 것은 MODULUS 값뿐이다.

---

# Step 3-2 · `define_prime_field!` 매크로로 코드 재사용

## 문제: 같은 코드를 복붙해야 하나?

Fr의 add, sub, mont_mul, inv... 전부 Fp와 **로직이 동일**하고 **상수만 다르다**:

|상수|의미|Fp 값|Fr 값|
|---|---|---|---|
|`MODULUS`|나누는 소수|p|r|
|`R`|2^256 mod MODULUS|`0xd35d...0d9d...`|`0xac96...fffb...`|
|`R2`|2^512 mod MODULUS|`0xf32c...a89...`|`0x1bb8...da7...`|
|`INV`|REDC용 상수|MODULUS[0]에서 계산|MODULUS[0]에서 계산|

## 해결: Rust의 `macro_rules!`

```rust
macro_rules! define_prime_field {
    (
        $name:ident,                    // ← 타입 이름 (Fp 또는 Fr)
        modulus: [$m0, $m1, $m2, $m3],  // ← 소수
        r: [$r0, $r1, $r2, $r3],        // ← 2^256 mod 소수
        r2: [$r20, $r21, $r22, $r23]    // ← 2^512 mod 소수
    ) => {
        // ↓ 여기에 구조체, 상수, 메서드, trait impl 전부 생성
    };
}
```

매크로는 **코드 템플릿**이다. `$name`에 `Fr`을 넣으면 `pub struct Fr(...)`, `impl Fr { ... }` 등이 자동으로 만들어진다.

## 매크로가 생성하는 것들

하나의 매크로 호출로 **16가지**가 한번에 생성된다:

```
상수: MODULUS, R, R2, INV (Newton's method로 자동 계산)
구조체: Fr
메서드: ZERO, ONE, is_zero, from_raw, from_u64, to_repr,
        mont_mul, mont_reduce_inner, add, sub, neg,
        square, pow, inv
연산자: +, -, *, - (unary), Display
헬퍼: sub_if_gte
```

## 헬퍼 함수 분리: mod.rs로 이동

`adc`, `sbb`, `mac`은 Fp와 Fr 모두에서 쓰이므로 **공용 모듈**(field/mod.rs)로 이동:

```rust
// field/mod.rs — 공용 헬퍼
pub(crate) fn adc(a: u64, b: u64, carry: bool) -> (u64, bool) { ... }
pub(crate) fn sbb(a: u64, b: u64, borrow: bool) -> (u64, bool) { ... }
pub(crate) fn mac(acc: u64, a: u64, b: u64, carry: u64) -> (u64, u64) { ... }
```

```rust
// fp.rs — 이제 import해서 사용
use super::{adc, sbb, mac};
```

## Fr 생성: 매크로 한 줄

```rust
// fr.rs
use super::{adc, sbb, mac};

super::define_prime_field!(
    Fr,
    modulus: [0x43e1f593f0000001, 0x2833e84879b97091,
              0xb85045b68181585d, 0x30644e72e131a029],
    r:  [0xac96341c4ffffffb, 0x36fc76959f60cd29,
         0x666ea36f7879462e, 0x0e0a77c19a07df2f],
    r2: [0x1bb8e645ae216da7, 0x53fe3ab1e35c59e3,
         0x8c49833d53bb8085, 0x0216d0b17f4e44a5]
);
```

**이것만으로** Fr은 Fp와 동일한 모든 기능을 갖는다:

- `Fr::from_u64(42)`, `Fr::ONE`, `Fr::ZERO`
- `a + b`, `a * b`, `a.inv()`
- Montgomery form 자동 변환

## INV 자동 계산

매크로 안에서 INV는 MODULUS[0]에서 **컴파일 타임에 자동 계산**된다:

```rust
const INV: u64 = {
    let p0 = MODULUS[0];
    let mut inv = 1u64;
    // Newton's method: 정밀도가 매 반복마다 2배
    inv = inv.wrapping_mul(2u64.wrapping_sub(p0.wrapping_mul(inv)));
    // ... 6회 반복 → 64-bit 정밀도 달성
    inv.wrapping_neg() // -p0^{-1} mod 2^64
};
```

→ 새 체를 추가할 때 INV를 직접 계산할 필요 없이, MODULUS만 넣으면 된다.

## 파일 구조 변화

```
field/
├── mod.rs   ← adc/sbb/mac 공용 헬퍼 + define_prime_field! 매크로
├── fp.rs    ← 수동 구현 (학습 참고용)
└── fr.rs    ← 매크로로 생성 (코드 재사용)
```

## 테스트 결과

```
running 45 tests
field::fp::tests ... 35 passed  ← 기존 Fp 테스트 그대로 통과
field::fr::tests ... 10 passed  ← Fr 새 테스트 전부 통과
```

Fr에서 검증한 것:

- `Fr::from_u64(6) * Fr::from_u64(7) == Fr::from_u64(42)` ← Montgomery 곱셈 정상
- `a.inv() * a == Fr::ONE` ← 역원 동작
- 교환법칙, 결합법칙, 분배법칙 ← 체 공리 성립

## 핵심 정리

> **같은 소수체 로직을 매크로로 추출** → 상수 3개(MODULUS, R, R2)만 바꾸면 새 유한체 생성. Fp는 손으로 짠 참고 코드, Fr은 매크로로 생성한 실전 코드. 나중에 다른 커브(BLS12-381 등)를 추가할 때도 같은 매크로를 재사용할 수 있다.