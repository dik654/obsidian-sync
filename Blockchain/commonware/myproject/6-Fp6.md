## Step 04: Fp6 — 3차 확장체 (타워의 3번째 층)

### 전체 타워 구조

```
Fp (254-bit 소수체)
 │  원소: 정수 하나
 │
 └─ Fp2 = Fp[u] / (u² + 1)
     │  원소: a₀ + a₁·u     (Fp 2개)
     │  u² = -1 (복소수)
     │
     └─ Fp6 = Fp2[v] / (v³ - β)     ← 이번 스탭
         │  원소: c₀ + c₁·v + c₂·v²  (Fp2 3개 = Fp 6개)
         │  v³ = β = 9 + u
         │
         └─ Fp12 = Fp6[w] / (w² - v)   ← 다음 스탭
              원소: d₀ + d₁·w           (Fp6 2개 = Fp 12개)
              w² = v
```

> [!info] 왜 "타워"인가? Fp12를 한 번에 만들 수도 있지만, 타워로 쌓으면:
> 
> 1. **각 층의 차수가 작아서** 구현이 단순
> 2. **Karatsuba 최적화**를 각 층에서 적용 가능
> 3. **Frobenius 사상**이 각 층에서 깔끔하게 분해
> 
> BN254의 표준 선택: 2 × 3 × 2 = 12

---

### 왜 Fp6가 필요한가?

페어링 `e(G1, G2) → GT`의 결과는 **Fp12의 원소**다.

Fp12를 효율적으로 만들려면 중간 단계가 필요:

- Fp2 위에 **3차 확장** → Fp6 (이번 스탭)
- Fp6 위에 **2차 확장** → Fp12 (다음 스탭)

> [!important] Fr과의 차이
> 
> - **Fr** (스칼라체): 확장 없음. 끝.
> - **Fp** (좌표체): Fp → Fp2 → Fp6 → Fp12로 확장
> 
> G2의 좌표가 Fp2에 살고, 페어링 결과가 Fp12에 사는 것이 이유다.

---

### Step 4-1: Fp6 기본 구조 + mul_by_nonresidue

#### 구조

```
Fp6 원소 = c₀ + c₁·v + c₂·v²

여기서:
  c₀, c₁, c₂ ∈ Fp2  (각각이 복소수)
  v는 새로운 미지수
  v³ = β = 9 + u ∈ Fp2  (non-residue)
```

> [!tip] 비유
> 
> - Fp = 실수
> - Fp2 = 복소수 (2차원)
> - Fp6 = "3차원 복소수" — Fp2 위에 새 축 v를 추가

#### 코드: 구조체

```rust
// fp6.rs
use super::fp2::Fp2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fp6 {
    pub c0: Fp2, // 상수항
    pub c1: Fp2, // v의 계수
    pub c2: Fp2, // v²의 계수
}

impl Fp6 {
    pub const ZERO: Fp6 = Fp6 {
        c0: Fp2::ZERO, c1: Fp2::ZERO, c2: Fp2::ZERO
    };
    pub const ONE: Fp6 = Fp6 {
        c0: Fp2::ONE, c1: Fp2::ZERO, c2: Fp2::ZERO
    };
}
```

> [!note] 패턴 반복 Fp2의 `{ c0: Fp, c1: Fp }`가 Fp6에서 `{ c0: Fp2, c1: Fp2, c2: Fp2 }`로 확장. 덧셈·뺄셈은 **각 성분끼리** — Fp2와 동일한 패턴이다.

#### mul_by_nonresidue: β = 9 + u 곱하기

Fp6에서 v³ 이상의 항이 나오면 v³ = β로 대체해야 한다. 이때 Fp2 원소에 β를 곱하는 연산이 반복적으로 필요하다.

```rust
// fp2.rs에 추가
/// (a + bu)(9 + u) = (9a - b) + (a + 9b)u
pub fn mul_by_nonresidue(&self) -> Fp2 {
    let nine = Fp::from_u64(9);
    Fp2 {
        c0: nine * self.c0 - self.c1,
        c1: self.c0 + nine * self.c1,
    }
}
```

수동 전개:

```
(a + bu)(9 + u) = 9a + au + 9bu + bu²
                = 9a + au + 9bu + b·(-1)    ← u² = -1
                = (9a - b) + (a + 9b)u
```

> [!question] β = 9 + u는 어디서 온 값인가? v³ = β가 되려면 β가 Fp2에서 **cubic non-residue**여야 한다. 즉, β^((p²-1)/3) ≠ 1이어야 Fp2[v]/(v³-β)가 체가 된다. BN254에서 β = 9 + u가 이 조건을 만족하는 표준 선택이다.

---

### Step 4-2: Fp6 곱셈 (Karatsuba) + 역원

#### 곱셈: 왜 특별한가?

```
(a₀ + a₁v + a₂v²)(b₀ + b₁v + b₂v²) 를 전개하면

= a₀b₀ + a₀b₁v + a₀b₂v²
+ a₁b₀v + a₁b₁v² + a₁b₂v³        ← v³ 등장!
+ a₂b₀v² + a₂b₁v³ + a₂b₂v⁴       ← v³, v⁴ 등장!
```

v³ = β, v⁴ = βv로 대체하면:

```
c₀ = a₀b₀ + β·(a₁b₂ + a₂b₁)
c₁ = a₀b₁ + a₁b₀ + β·a₂b₂
c₂ = a₀b₂ + a₁b₁ + a₂b₀
```

나이브하게 하면 Fp2 곱셈 **9회**. Karatsuba 트릭으로 **6회**로 줄인다:

```
v₀ = a₀b₀,  v₁ = a₁b₁,  v₂ = a₂b₂    ← 3회

c₀ = v₀ + β·((a₁+a₂)(b₁+b₂) - v₁ - v₂)   ← 1회
c₁ = (a₀+a₁)(b₀+b₁) - v₀ - v₁ + β·v₂      ← 1회
c₂ = (a₀+a₂)(b₀+b₂) - v₀ + v₁ - v₂         ← 1회
                                          합계: 6회
```

> [!tip] Karatsuba의 핵심 아이디어 `a₁b₂ + a₂b₁`을 직접 계산하면 곱셈 2회. 대신 `(a₁+a₂)(b₁+b₂) - a₁b₁ - a₂b₂`로 쓰면 곱셈 1회 (v₁, v₂는 이미 계산함). **덧셈은 싸고, 곱셈은 비싸다** — 이것이 Karatsuba가 작동하는 이유.

#### 코드: Fp6 곱셈

```rust
pub fn mul(&self, rhs: &Fp6) -> Fp6 {
    let v0 = self.c0 * rhs.c0;   // Fp2 곱셈 ①
    let v1 = self.c1 * rhs.c1;   // Fp2 곱셈 ②
    let v2 = self.c2 * rhs.c2;   // Fp2 곱셈 ③

    // c₀ = v₀ + β·(a₁b₂ + a₂b₁)
    let c0 = v0 + ((self.c1 + self.c2) * (rhs.c1 + rhs.c2) // ④
        - v1 - v2).mul_by_nonresidue();

    // c₁ = a₀b₁ + a₁b₀ + β·v₂
    let c1 = (self.c0 + self.c1) * (rhs.c0 + rhs.c1)       // ⑤
        - v0 - v1 + v2.mul_by_nonresidue();

    // c₂ = a₀b₂ + a₁b₁ + a₂b₀
    let c2 = (self.c0 + self.c2) * (rhs.c0 + rhs.c2)       // ⑥
        - v0 + v1 - v2;

    Fp6 { c0, c1, c2 }
}
```

---

#### 역원: Norm Reduction 패턴

> [!important] 타워 전체를 관통하는 핵심 패턴 **역원을 구할 때, 한 층 아래로 내려간다:**
> 
> |층|역원 계산|norm이 사는 곳|
> |---|---|---|
> |Fp|Fermat 소정리: a^(p-2)|—|
> |Fp2|conj(a) / norm(a)|norm ∈ **Fp**|
> |Fp6|cofactor / norm(a)|norm ∈ **Fp2**|
> |Fp12|conj(a) / norm(a)|norm ∈ **Fp6**|
> 
> 각 층에서 norm은 **한 단계 아래 체의 원소**가 되어, 아래 체의 inv를 재활용한다.

Fp6의 역원 공식:

```
t₀ = c₀² - β·c₁c₂
t₁ = β·c₂² - c₀c₁
t₂ = c₁² - c₀c₂

norm = c₀·t₀ + β·(c₂·t₁ + c₁·t₂)  ∈ Fp2  ← 차원이 내려감!

a⁻¹ = (t₀ + t₁·v + t₂·v²) · norm⁻¹
```

#### 코드: Fp6 역원

```rust
pub fn inv(&self) -> Option<Fp6> {
    let c0s = self.c0.square();
    let c1s = self.c1.square();
    let c2s = self.c2.square();
    let c01 = self.c0 * self.c1;
    let c02 = self.c0 * self.c2;
    let c12 = self.c1 * self.c2;

    let t0 = c0s - c12.mul_by_nonresidue();        // c₀² - β·c₁c₂
    let t1 = c2s.mul_by_nonresidue() - c01;         // β·c₂² - c₀c₁
    let t2 = c1s - c02;                             // c₁² - c₀c₂

    // norm ∈ Fp2 — 차원이 Fp6 → Fp2로 내려간다
    let norm = self.c0 * t0
        + (self.c2 * t1 + self.c1 * t2).mul_by_nonresidue();
    let norm_inv = norm.inv()?;  // Fp2의 inv 사용

    Some(Fp6 {
        c0: t0 * norm_inv,
        c1: t1 * norm_inv,
        c2: t2 * norm_inv,
    })
}
```

> [!note] inv 호출 체인
> 
> ```
> Fp6::inv()
>   └─ norm ∈ Fp2 계산
>       └─ Fp2::inv()
>           └─ norm ∈ Fp 계산
>               └─ Fp::inv()  (Fermat: a^(p-2))
> ```
> 
> 최종적으로 Fp의 pow 연산 하나로 귀결된다.

---

### 테스트로 검증되는 체 공리

```rust
#[test]
fn mul_identity() {
    let a = fp6(3, 5, 7, 11, 13, 17);
    assert_eq!(a * Fp6::ONE, a);       // 곱셈 항등원
}

#[test]
fn mul_commutativity() {
    let a = fp6(3, 5, 7, 11, 13, 17);
    let b = fp6(2, 4, 6, 8, 10, 12);
    assert_eq!(a * b, b * a);           // 교환법칙
}

#[test]
fn mul_associativity() {
    let a = fp6(1, 2, 3, 4, 5, 6);
    let b = fp6(7, 8, 9, 10, 11, 12);
    let c = fp6(13, 14, 15, 16, 17, 18);
    assert_eq!((a * b) * c, a * (b * c)); // 결합법칙
}

#[test]
fn inv_basic() {
    let a = fp6(3, 5, 7, 11, 13, 17);
    let a_inv = a.inv().unwrap();
    assert_eq!(a * a_inv, Fp6::ONE);    // a · a⁻¹ = 1
}
```