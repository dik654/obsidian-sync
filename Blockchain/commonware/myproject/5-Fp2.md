# Step 3-3 · Fp2 이차 확장체 — 구조와 덧셈/뺄셈

## 전체 그림: 왜 체를 3종류나 만드는가?

BN254 페어링 시스템에는 **세 종류의 체**가 각각 다른 역할을 한다:

```
e : G1 × G2 → GT

G1 = 타원곡선 위의 점, 좌표 ∈ Fp      ← "실수 좌표"
G2 = 트위스트 커브 위의 점, 좌표 ∈ Fp2  ← "복소수 좌표"  ★
GT = 페어링 결과가 사는 곳, ∈ Fp12

Fr = 스칼라체 — G1/G2 점을 "몇 번 더할지"의 범위
     ZK 회로의 witness도 Fr 원소
```

### 왜 G2에는 Fp 대신 Fp2가 필요한가?

BN254 커브 방정식: `y² = x³ + 3`

- **G1**: 이 방정식의 해 `(x, y)`에서 x, y ∈ Fp → Fp만으로 충분
- **G2**: 같은 방정식의 "트위스트" 버전으로, 해가 Fp 안에 존재하지 않음
    - 마치 `x² = -1`의 해가 실수에 없어서 복소수가 필요한 것과 같다
    - → Fp를 확장한 **Fp2** 위에서 해를 찾는다

### 타워 구조: 페어링까지 가는 길

```
Fp  →  Fp2  →  Fp6  →  Fp12
 │      │       │        │
 │      │       │        └── 페어링 결과 e(P,Q) ∈ Fp12
 │      │       └────────── Fp12를 효율적으로 구성하는 중간 단계
 │      └────────────────── G2 좌표 (지금 구현)
 └───────────────────────── G1 좌표 + 모든 확장의 기초
```

각 확장은 **이전 체 2~3개를 묶어서** 더 큰 체를 만든다:

- Fp2 = Fp × 2 (복소수)
- Fp6 = Fp2 × 3
- Fp12 = Fp6 × 2

## 복소수와의 대응

|복소수|Fp2|
|---|---|
|실수 ℝ|Fp|
|허수단위 i (i²=-1)|u (u²=-1)|
|복소수 a+bi|Fp2 원소 a₀+a₁u|
|켤레 a-bi|conjugate a₀-a₁u|
|\|z\|² = a²+b²|norm = a₀²+a₁²|

## 코드: Fp2 구조체

```rust
use super::fp::Fp;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fp2 {
    pub c0: Fp, // 실수부
    pub c1: Fp, // 허수부 (u의 계수)
}

impl Fp2 {
    pub const ZERO: Fp2 = Fp2 { c0: Fp::ZERO, c1: Fp::ZERO };
    pub const ONE: Fp2 = Fp2 { c0: Fp::ONE, c1: Fp::ZERO };  // 1 + 0·u
}
```

Fp2는 **Fp 원소 2개**를 묶은 것. `Fp::ONE`은 Montgomery form의 1이므로 Fp2::ONE도 자동으로 올바르다.

## 코드: 덧셈/뺄셈

```rust
/// (a₀+a₁u) + (b₀+b₁u) = (a₀+b₀) + (a₁+b₁)u
pub fn add(&self, rhs: &Fp2) -> Fp2 {
    Fp2 {
        c0: self.c0 + rhs.c0,
        c1: self.c1 + rhs.c1,
    }
}
```

복소수와 동일 — 각 성분끼리 연산. 덧셈에서는 u²가 개입하지 않는다.

---

# Step 3-4 · Fp2 곱셈, conjugate, norm, inv, frobenius

## 곱셈 — u² = -1이 작용하는 순간

```
(a₀ + a₁u)(b₀ + b₁u)
= a₀b₀ + a₀b₁u + a₁b₀u + a₁b₁u²
                              ^^^^
                              u² = -1 이므로 → -a₁b₁

= (a₀b₀ - a₁b₁) + (a₀b₁ + a₁b₀)u
```

복소수 곱셈 `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`와 **완전히 동일한 구조**다.

### Karatsuba 트릭: 곱셈 4회 → 3회

나이브하게 하면 Fp 곱셈 4번: a₀b₀, a₀b₁, a₁b₀, a₁b₁ Karatsuba로 3번에 가능:

```rust
pub fn mul(&self, rhs: &Fp2) -> Fp2 {
    let v0 = self.c0 * rhs.c0;  // a₀·b₀
    let v1 = self.c1 * rhs.c1;  // a₁·b₁

    // c₁ = (a₀+a₁)(b₀+b₁) - v₀ - v₁   ← 이것이 a₀b₁ + a₁b₀
    let c1 = (self.c0 + self.c1) * (rhs.c0 + rhs.c1) - v0 - v1;
    // c₀ = v₀ - v₁                       ← u² = -1이 여기서 작용
    let c0 = v0 - v1;

    Fp2 { c0, c1 }
}
```

**왜 3회가 가능한가?**

- `a₀b₁ + a₁b₀`를 직접 계산하면 곱셈 2회
- `(a₀+a₁)(b₀+b₁) = a₀b₀ + a₀b₁ + a₁b₀ + a₁b₁`에서 이미 구한 v₀, v₁을 빼면 → 곱셈 1회로 같은 결과
- 덧셈은 곱셈보다 훨씬 저렴하므로, 곱셈 1회를 아끼는 것이 이득

### 테스트로 검증

```rust
#[test]
fn mul_basic() {
    // (3 + 5u)(7 + 2u) = 21 + 6u + 35u + 10u²
    //                   = 21 + 41u - 10 = 11 + 41u
    let a = fp2(3, 5);
    let b = fp2(7, 2);
    assert_eq!(a * b, fp2(11, 41));
}
```

## Conjugate (켤레)

복소수의 켤레와 동일:

```rust
/// a₀ + a₁u → a₀ - a₁u
pub fn conjugate(&self) -> Fp2 {
    Fp2 { c0: self.c0, c1: -self.c1 }
}
```

## Norm (노름) — 차원이 내려간다

```rust
/// a · conj(a) = (a₀+a₁u)(a₀-a₁u) = a₀² + a₁²  ∈ Fp
pub fn norm(&self) -> Fp {
    self.c0 * self.c0 + self.c1 * self.c1
}
```

핵심: **Fp2 × Fp2 → Fp**. 결과가 Fp2가 아니라 Fp다. 복소수에서 |a+bi|² = a²+b²가 실수인 것과 같다.

```rust
#[test]
fn norm_basic() {
    // norm(3 + 4u) = 9 + 16 = 25
    let a = fp2(3, 4);
    assert_eq!(a.norm(), Fp::from_u64(25));
}

#[test]
fn a_times_conjugate_is_norm() {
    let a = fp2(3, 4);
    let product = a * a.conjugate();
    assert_eq!(product.c1, Fp::ZERO);       // 허수부 = 0
    assert_eq!(product.c0, Fp::from_u64(25)); // 실수부 = norm
}
```

## Inv (역원) — norm을 이용한 트릭

```rust
/// a⁻¹ = conj(a) · norm(a)⁻¹
pub fn inv(&self) -> Option<Fp2> {
    let n = self.norm();        // Fp 원소
    let n_inv = n.inv()?;       // Fp의 역원 (Fermat)
    let conj = self.conjugate();
    Some(Fp2 {
        c0: conj.c0 * n_inv,
        c1: conj.c1 * n_inv,
    })
}
```

**왜 이렇게 하는가?**

1. Fp2에서 직접 역원을 구하려면 p²-2승이 필요 → 매우 비쌈
2. 대신: norm(a) ∈ Fp → Fp의 inv는 이미 구현됨 (p-2승)
3. `a⁻¹ = conj(a) / norm(a)`는 Fp 역원 1번 + Fp 곱셈 2번으로 끝

## Frobenius 사상 — x → x^p

```rust
/// BN254에서 u^p = -u이므로 frobenius = conjugate
pub fn frobenius_map(&self) -> Fp2 {
    self.conjugate()
}
```

Frobenius는 유한체에서 **x를 p번 거듭제곱**하는 것. Fp2에서:

- `(a₀ + a₁u)^p = a₀^p + a₁^p · u^p`
- a₀, a₁ ∈ Fp이므로 `a₀^p = a₀` (Fermat)
- BN254에서 `u^p = -u` (p ≡ 3 mod 4이므로)
- 따라서 `frobenius(a₀+a₁u) = a₀ - a₁u = conjugate`

```rust
#[test]
fn frobenius_twice_is_identity() {
    // x^{p²} = x (Fp2 위에서 frobenius 2번 = 원래 값)
    let a = fp2(7, 11);
    assert_eq!(a.frobenius_map().frobenius_map(), a);
}
```

frobenius를 **2번** 적용하면 원래 값으로 돌아온다 → Fp2는 Fp의 **2차** 확장이므로.

## Fp, Fr, Fp2 역할 정리

|체|크기|용도|다음에 쓰이는 곳|
|---|---|---|---|
|**Fp**|254-bit 소수|G1 좌표, 모든 확장의 기초|Fp2, G1|
|**Fr**|254-bit 소수 (다른 값)|스칼라, ZK witness|회로(R1CS), 서명|
|**Fp2**|Fp × 2 = 508-bit|G2 좌표|Fp6, G2|

## 테스트 결과

```
running 62 tests
field::fp  ... 35 passed
field::fr  ... 10 passed
field::fp2 ... 17 passed  ← Step 3-3 (6개) + Step 3-4 (11개)
```