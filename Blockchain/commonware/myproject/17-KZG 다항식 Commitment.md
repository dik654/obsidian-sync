## Step 13: KZG 다항식 Commitment — 하나의 점으로 다항식 전체를 약정하다

### 핵심 질문: 왜 다항식 commitment이 필요한가?

```
Groth16까지의 파이프라인:

  프로그램 → R1CS → QAP → Groth16
                              │
                              ↓
                    "회로별 trusted setup"

  문제: 회로가 바뀌면 setup도 처음부터 다시!

    개발자: "AND 게이트 하나 추가했어."
    시스템: "τ, α, β, γ, δ 전부 새로 생성해야 합니다."
    개발자: "MPC 세레모니를 또...?"

  → 이 문제의 해결 = KZG (universal setup)
```

> [!important] KZG의 역할
> ```
> Groth16: 회로별(per-circuit) setup — 회로 구조가 바뀌면 새 setup
> KZG: 범용(universal) setup — τ 하나로 모든 다항식에 재사용
>
> KZG는 PLONK의 핵심 빌딩 블록:
>   PLONK = 범용 회로 시스템 + KZG polynomial commitment
>
> KZG 없이 PLONK은 불가능 — 다항식 관계를 검증할 도구가 없다
> ```

---

### KZG란?

```
KZG [Kate-Zaverucha-Goldberg, 2010]:
  - 다항식을 하나의 타원곡선 점으로 commit
  - 특정 점에서의 evaluation을 pairing check로 증명
  - PLONK, Danksharding(EIP-4844) 등의 핵심 프리미티브

핵심 수치 비교:
  ┌──────────────────┬──────────────────────────┐
  │ Commitment 크기   │ G1 1개 = 64 bytes        │
  │ Opening proof    │ G1 1개 = 64 bytes        │
  │ 검증 시간         │ O(1) — 페어링 2회         │
  │ Commit 시간       │ O(d) — d = 다항식 차수    │
  │ Setup            │ 1회 (universal)           │
  │ Setup 파라미터    │ τ 1개                     │
  │ Binding          │ O (DL 가정 하에)           │
  │ Hiding           │ O (τ 비밀 시)              │
  └──────────────────┴──────────────────────────┘

"Polynomial Commitment Scheme" 의 의미:
  Commit: 다항식 f(x)를 하나의 값 C로 약정
  Open: 특정 점 z에서 f(z) = y 임을 증명
  Verify: C와 증거만으로 f(z) = y 확인 (f(x) 전체를 모르고도!)
```

---

### Part 1: 핵심 아이디어 — 다항식 인수정리

#### 수학적 기반

```
다항식 인수정리 (Factor Theorem):
  f(z) = y  ⟺  (x - z) | (f(x) - y)

  즉, f(x) - y = q(x) · (x - z) 인 다항식 q(x)가 존재

증명:
  f(z) = y이면
  g(x) = f(x) - y 라 하자
  g(z) = f(z) - y = 0

  다항식 나눗셈 정리에 의해:
    g(x) = q(x) · (x - z) + r
    여기서 r은 상수 (deg(r) < deg(x-z) = 1)

  x = z 대입:
    g(z) = q(z) · 0 + r
    0 = r

  따라서 g(x) = q(x) · (x - z), 나머지 없이 나누어떨어짐!
```

#### KZG가 이것을 어떻게 활용하는가

```
아이디어:
  증명자: "f(z) = y 입니다"
  검증자: "증명해봐"
  증명자: q(x) = (f(x) - y) / (x - z) 를 계산하여 [q(τ)]₁ 을 제출

  검증자는 페어링으로 확인:
    f(τ) - y = q(τ) · (τ - z)

  ⟹ e([q(τ)]₁, [τ - z]₂) = e([f(τ) - y]₁, G₂)

  만약 f(z) ≠ y 라면?
    f(x) - y 는 (x - z)로 나누어떨어지지 않음
    → q(x)가 다항식이 아님
    → [q(τ)]₁ 을 SRS만으로 계산 불가능
    → 위조 불가 (under DL assumption)

직관:
  ┌─────────────────────────────────────────┐
  │ "나누어떨어짐" = "그 점에서의 값이 맞다"     │
  │                                          │
  │ 나누어떨어짐을 SRS 위에서 증명하면           │
  │ τ를 모르는 검증자도 확인 가능                │
  └─────────────────────────────────────────┘
```

---

### Part 2: Groth16과의 비교

```
┌────────────────────┬───────────────────────┬───────────────────────┐
│                    │ Groth16               │ KZG                   │
├────────────────────┼───────────────────────┼───────────────────────┤
│ Setup 유형          │ per-circuit           │ universal             │
│ Setup 파라미터       │ τ, α, β, γ, δ        │ τ 하나               │
│ 회로 변경 시         │ 새 setup 필요          │ 재사용 가능           │
│ 증명 대상           │ R1CS/QAP 만족성        │ 다항식 evaluation     │
│ 증명 크기           │ G1×2 + G2×1 = 192B    │ G1×1 = 64B           │
│ 검증 페어링 수       │ 3회                   │ 2회                   │
│ 사용처              │ 독립 ZK 증명           │ PLONK 빌딩 블록       │
│ SRS 구조            │ A, B, C별 인코딩       │ powers of τ          │
│ 건전성 근거          │ Schwartz-Zippel       │ 다항식 인수정리        │
└────────────────────┴───────────────────────┴───────────────────────┘

핵심 차이:
  Groth16의 SRS:
    PK에 L_i(τ), M_i(τ), N_i(τ) 등 회로 구조가 인코딩됨
    → 회로가 바뀌면 전부 다시 계산

  KZG의 SRS:
    [τ⁰]₁, [τ¹]₁, ..., [τᵈ]₁  +  [τ⁰]₂, [τ¹]₂
    → 순수한 τ의 거듭제곱만 저장
    → 어떤 다항식이든 이 위에서 commit 가능
```

---

### Part 3: 전체 흐름 — 4단계

```
┌─────────────────────────────────────────────────────────────────┐
│                   KZG 프로토콜 전체 흐름                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  Setup   │    │  Commit  │    │  Open    │    │  Verify  │   │
│  │          │    │          │    │          │    │          │   │
│  │ τ 생성   │    │ f(x)     │    │ 점 z     │    │ C, π     │   │
│  │ SRS 출력 │───→│ SRS      │───→│ f, SRS   │───→│ SRS      │   │
│  │          │    │ → C      │    │ → (y, π) │    │ → bool   │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│                                                                  │
│  1회 (유니버설)     매 다항식마다      매 평가마다       매 검증마다    │
│  ⚠️ τ 삭제         O(d)             O(d)             O(1)        │
│                                                                  │
│  출력:             출력:            출력:             출력:         │
│  SRS{g1,g2}       C ∈ G1          (y, π∈G1)        true/false   │
└─────────────────────────────────────────────────────────────────┘
```

---

### Part 4: 자료구조 — SRS, Commitment, Opening

#### SRS (Structured Reference String)

```rust
pub struct SRS {
    pub g1_powers: Vec<G1>,  // [τ⁰]₁, [τ¹]₁, ..., [τ^d]₁
    pub g2_powers: Vec<G2>,  // [τ⁰]₂, [τ¹]₂, ..., [τ^k]₂
}
```

```
SRS 메모리 레이아웃:

  g1_powers (max_degree + 1개):
  ┌─────────┬─────────┬─────────┬─────────┬─────────┐
  │ [τ⁰]₁  │ [τ¹]₁  │ [τ²]₁  │  ...    │ [τᵈ]₁  │
  │ = G₁    │ = τ·G₁ │ =τ²·G₁ │         │=τᵈ·G₁  │
  └─────────┴─────────┴─────────┴─────────┴─────────┘

  g2_powers (max_degree_g2 + 1개):
  ┌─────────┬─────────┬─────────┐
  │ [τ⁰]₂  │ [τ¹]₂  │  ...    │
  │ = G₂    │ = τ·G₂ │         │
  └─────────┴─────────┴─────────┘

왜 G1과 G2 모두 필요한가?

  단일 점 검증:
    e(π, [τ]₂) = e(C - [y]₁ + z·π, G₂)
    → [τ⁰]₂ = G₂ 와 [τ¹]₂ 만 있으면 충분

  다중 점 검증 (batch):
    e(π, [Z(τ)]₂) = e(C - [I(τ)]₁, G₂)
    → Z(x) = ∏(x - zᵢ) 의 degree = 점의 수 k
    → [Z(τ)]₂ 계산에 [τ⁰]₂, ..., [τᵏ]₂ 필요
    → max_degree_g2 ≥ k
```

#### Commitment

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment(pub G1);
```

```
Commitment = 하나의 G1 점

  다항식 f(x) = f₀ + f₁x + f₂x² + ... + fₐx^d

  C = f₀·[τ⁰]₁ + f₁·[τ¹]₁ + f₂·[τ²]₁ + ... + fₐ·[τᵈ]₁
    = [f₀·τ⁰ + f₁·τ¹ + ... + fₐ·τᵈ]₁
    = [f(τ)]₁

  크기: G1 점 1개 = 64 bytes (BN254)

  핵심 속성:
    - binding: 다른 f → 다른 C (DL 가정)
    - hiding: C에서 f를 복원 불가 (τ 비밀 시)
    - additivity: commit(f) + commit(g) = commit(f + g)
```

#### Opening / BatchOpening

```rust
pub struct Opening {
    pub point: Fr,   // z — 평가 점
    pub value: Fr,   // y = f(z) — 평가 값
    pub proof: G1,   // π = [q(τ)]₁ — 증거
}

pub struct BatchOpening {
    pub points: Vec<Fr>,   // [z₁, ..., zₖ]
    pub values: Vec<Fr>,   // [y₁, ..., yₖ]
    pub proof: G1,         // π = [q(τ)]₁
}
```

```
Opening 구조:
  ┌─────────────────────────────────────────┐
  │ Opening                                  │
  │                                          │
  │  point: z (어느 점에서 열었나)              │
  │  value: y = f(z) (값이 뭔가)              │
  │  proof: π = [q(τ)]₁ (quotient의 commit)  │
  │                                          │
  │  검증자에게 전달: (z, y, π)                │
  │  검증자는 C와 함께 pairing check           │
  └─────────────────────────────────────────┘

BatchOpening: k개의 점을 한 번에 증명
  proof는 여전히 G1 하나! (O(1) 크기 불변)
  검증도 페어링 2회 (O(1))
```

---

### Part 5: Trusted Setup — τ의 powers 생성

#### 알고리즘

```
Input:  max_degree (G1 powers 수), max_degree_g2 (G2 powers 수), RNG
Output: SRS

1. τ ← random nonzero element of Fr
   (τ = 0이면 모든 power가 identity가 되어 의미없음)

2. G1 powers 생성:
   tau_power ← 1
   for i = 0 to max_degree:
     g1_powers[i] ← tau_power · G₁
     tau_power ← tau_power · τ

   결과: [1·G₁, τ·G₁, τ²·G₁, ..., τᵈ·G₁]

3. G2 powers 생성:
   tau_power ← 1
   for j = 0 to max_degree_g2:
     g2_powers[j] ← tau_power · G₂
     tau_power ← tau_power · τ

   결과: [1·G₂, τ·G₂, τ²·G₂, ..., τᵏ·G₂]

4. τ 삭제 (toxic waste)
   → Rust에서는 함수 종료 시 스택에서 자동 소멸
```

#### 코드 대응

```rust
pub fn setup<R: Rng>(
    max_degree: usize,
    max_degree_g2: usize,
    rng: &mut R,
) -> SRS {
    let tau = random_nonzero_fr(rng);    // Step 1: τ 생성
    let g1 = G1::generator();
    let g2 = G2::generator();

    // Step 2: G1 powers
    let mut g1_powers = Vec::with_capacity(max_degree + 1);
    let mut tau_power = Fr::ONE;          // τ⁰ = 1
    for _ in 0..=max_degree {
        g1_powers.push(g1.scalar_mul(&tau_power.to_repr()));
        //               ↑ G₁ · τⁱ = [τⁱ]₁
        tau_power = tau_power * tau;      // τⁱ → τⁱ⁺¹
    }

    // Step 3: G2 powers (동일 패턴)
    let mut g2_powers = Vec::with_capacity(max_degree_g2 + 1);
    tau_power = Fr::ONE;
    for _ in 0..=max_degree_g2 {
        g2_powers.push(g2.scalar_mul(&tau_power.to_repr()));
        tau_power = tau_power * tau;
    }

    SRS { g1_powers, g2_powers }
    // Step 4: tau는 여기서 drop됨 (Rust ownership)
}
```

```
⚠️ to_repr() 호출이 필수인 이유:
  Fr 내부는 Montgomery form (연산 최적화)
  scalar_mul()은 raw [u64; 4] 기대
  to_repr()가 Montgomery → standard 변환

  이것 없이 직접 전달하면 완전히 다른 스칼라로 해석됨!

⚠️ Toxic Waste:
  τ를 아는 사람은 모든 commitment을 위조할 수 있다:
    임의의 C = [f(τ)]₁ 에 대해
    τ를 알면 q(τ)를 직접 계산 → 가짜 π 생성 가능

  프로덕션에서는 MPC (Multi-Party Computation) 세레모니:
    N명이 각각 τᵢ 기여 → τ = τ₁ · τ₂ · ... · τₙ
    한 명이라도 정직하면 (자신의 τᵢ를 삭제하면) 안전

    예: Zcash Powers of Tau (2017), EF KZG Ceremony (2023)
```

---

### Part 6: Commit — 다항식을 G1 점으로

#### 알고리즘

```
Input:  SRS, 다항식 f(x) = f₀ + f₁x + ... + fₐx^d
Output: C = [f(τ)]₁ ∈ G1

특수 케이스:
  f = 0 → C = identity (영점)

일반 케이스:
  C = Σᵢ fᵢ · [τⁱ]₁

  이것은 MSM (Multi-Scalar Multiplication):
    스칼라: [f₀, f₁, ..., fₐ]
    기저:   [[τ⁰]₁, [τ¹]₁, ..., [τᵈ]₁]

단계별:
  result ← identity (G1의 항등원)
  for i = 0 to d:
    if fᵢ ≠ 0:
      result ← result + fᵢ · g1_powers[i]
  return Commitment(result)
```

#### 코드 대응

```rust
pub fn commit(srs: &SRS, poly: &Polynomial) -> Commitment {
    if poly.is_zero() {
        return Commitment(G1::identity());    // 영다항식 특수 처리
    }
    assert!(
        poly.coeffs.len() <= srs.g1_powers.len(),
        "polynomial degree {} exceeds SRS max degree {}",
        poly.degree(), srs.g1_powers.len() - 1
    );

    let mut result = G1::identity();
    for (i, &coeff) in poly.coeffs.iter().enumerate() {
        if !coeff.is_zero() {                 // 0 계수 건너뛰기 (최적화)
            result = result + srs.g1_powers[i].scalar_mul(&coeff.to_repr());
            //               [τⁱ]₁ · fᵢ  =  fᵢ · [τⁱ]₁
        }
    }
    Commitment(result)
}
```

```
수치 예시 (개념적, 실제 커브 위가 아닌 추상):

  f(x) = 3 + 2x + x²
  SRS: [G₁, τ·G₁, τ²·G₁, ...]

  C = 3·G₁ + 2·(τ·G₁) + 1·(τ²·G₁)
    = (3 + 2τ + τ²)·G₁
    = f(τ)·G₁
    = [f(τ)]₁

  검증자는 C만 봄 → f(x)가 뭔지 알 수 없음
  하지만 나중에 f(z) = y 를 증명/검증 가능!
```

---

### Part 7: Open — 단일 점 evaluation proof 생성

#### 알고리즘 상세

```
Input:  SRS, 다항식 f(x), 평가 점 z
Output: Opening { point: z, value: y, proof: π }

단계:
  1. y = f(z) 계산

  2. 분자 다항식 구성:
     n(x) = f(x) - y

     n(z) = f(z) - y = 0  ← 핵심!

  3. 분모 다항식:
     d(x) = x - z = -z + 1·x

  4. 다항식 나눗셈:
     q(x), r = n(x) / d(x)

     n(z) = 0 이므로 r = 0 (나머지 없이 나누어떨어짐!)

  5. 증거 생성:
     π = commit(q) = [q(τ)]₁

의미:
  q(x) = (f(x) - y) / (x - z)

  이 나눗셈이 "정확히" 나누어떨어진다는 것 자체가
  f(z) = y의 증거

  SRS 위에서 q를 commit하면 검증자가 pairing으로 확인 가능
```

#### 코드 대응

```rust
pub fn open(srs: &SRS, poly: &Polynomial, point: Fr) -> Opening {
    let value = poly.eval(point);                // Step 1: y = f(z)

    // Step 2: n(x) = f(x) - y
    let y_poly = Polynomial::constant(value);
    let numerator = poly - &y_poly;

    // Step 3: d(x) = x - z
    let denominator = Polynomial::from_coeffs(vec![-point, Fr::ONE]);
    //                                              -z      1·x

    // Step 4: q(x) = n(x) / d(x)
    let (quotient, remainder) = numerator.div_rem(&denominator);
    debug_assert!(
        remainder.is_zero(),                     // r = 0 확인 (디버그 모드)
        "f(z) = y should make (f(x) - y) divisible by (x - z)"
    );

    // Step 5: π = commit(q)
    let proof = commit(srs, &quotient).0;

    Opening { point, value, proof }
}
```

#### 수치 예시: f(x) = 2x + 3, z = 5

```
Step 1: y = f(5) = 2·5 + 3 = 13

Step 2: n(x) = f(x) - 13 = (2x + 3) - 13 = 2x - 10

Step 3: d(x) = x - 5

Step 4: q(x) = (2x - 10) / (x - 5)

  다항식 나눗셈:
    2x - 10 = q(x) · (x - 5) + r

    2x - 10 = 2 · (x - 5)  + 0
    2x - 10 = 2x - 10       ✓

    q(x) = 2 (상수 다항식)
    r = 0 ✓

Step 5: π = commit(srs, q) = 2 · [τ⁰]₁ = 2 · G₁

결과: Opening { point: 5, value: 13, proof: 2·G₁ }
```

---

### Part 8: Verify — 2-pairing 최적화 검증

#### 검증 방정식 유도 (상세)

```
출발점:
  q(x) = (f(x) - y) / (x - z)
  ⟹ f(x) - y = q(x) · (x - z)              ... (*)

(*) 에 x = τ 대입:
  f(τ) - y = q(τ) · (τ - z)

양변을 커브 포인트로 해석:
  [f(τ)]₁ - [y]₁ = q(τ) · [τ - z]            ... 스칼라 관계

페어링으로 검증 (naive 형태):
  e([q(τ)]₁, [τ - z]₂) = e([f(τ) - y]₁, G₂)

  여기서 [τ - z]₂ = [τ]₂ - z·G₂

  문제: z·G₂ 계산에 G2에서의 scalar_mul 필요
        G2 scalar_mul은 G1보다 2~3배 비쌈 (Fp2 위의 연산)

2-pairing 최적화:
  e(π, [τ]₂ - z·G₂) = e(C - [y]₁, G₂)

  bilinearity로 전개:
    e(π, [τ]₂) · e(π, -z·G₂) = e(C - [y]₁, G₂)
    e(π, [τ]₂) · e(-z·π, G₂) = e(C - [y]₁, G₂)
    e(π, [τ]₂) = e(C - [y]₁, G₂) · e(z·π, G₂)
    e(π, [τ]₂) = e(C - [y]₁ + z·π, G₂)         ... (**) 최종 형태

  (**) 에서는:
    LHS: e(π, [τ]₂) — G1, G2 모두 이미 있는 값
    RHS: e(C - [y]₁ + z·π, G₂) — G1에서만 scalar_mul

  → G2 scalar_mul 완전 회피!

비교:
  naive:      e(π, [τ]₂ - z·G₂) = e(C - [y]₁, G₂)
              ↑ G2 scalar_mul 1회

  optimized:  e(π, [τ]₂) = e(C - [y]₁ + z·π, G₂)
              ↑ G1 scalar_mul 2회 (y·G₁, z·π) + G1 덧셈

  G1 연산이 G2보다 훨씬 빠르므로 최적화 형태 사용
```

#### 코드 대응

```rust
pub fn verify(srs: &SRS, commitment: &Commitment, opening: &Opening) -> bool {
    assert!(
        srs.g2_powers.len() >= 2,
        "SRS needs at least [τ⁰]₂ and [τ¹]₂ for verification"
    );

    let g1 = G1::generator();
    let g2 = srs.g2_powers[0];    // G₂ = [τ⁰]₂
    let tau_g2 = srs.g2_powers[1]; // [τ]₂ = [τ¹]₂

    let z = opening.point;
    let y = opening.value;
    let pi = opening.proof;

    // LHS: e(π, [τ]₂)
    let lhs = pairing(&pi, &tau_g2);

    // RHS: e(C - [y]₁ + z·π, G₂)
    let y_g1 = g1.scalar_mul(&y.to_repr());    // [y]₁
    let z_pi = pi.scalar_mul(&z.to_repr());    // z · π
    let rhs_g1 = commitment.0 + (-y_g1) + z_pi; // C - [y]₁ + z·π
    let rhs = pairing(&rhs_g1, &g2);

    lhs == rhs
}
```

```
코드 흐름 시각화:

  Input: C, (z, y, π), SRS

  ┌─── LHS 계산 ─────────────────────┐
  │                                    │
  │  π ───────┐                        │
  │           │ pairing                │
  │  [τ]₂ ───┘──→ LHS ∈ GT           │
  │                                    │
  └────────────────────────────────────┘

  ┌─── RHS 계산 ─────────────────────┐
  │                                    │
  │  y → y·G₁ = [y]₁                  │
  │  z → z·π                          │
  │                                    │
  │  C - [y]₁ + z·π ─┐                │
  │                   │ pairing        │
  │  G₂ ─────────────┘──→ RHS ∈ GT   │
  │                                    │
  └────────────────────────────────────┘

  LHS == RHS ?  →  true / false
```

---

### Part 9: 수치 트레이스 — F₁₃ 위에서 완전 추적

```
체: F₁₃ (소수체, p = 13)
생성원: g (추상적)

⚠️ 실제로는 BN254 커브를 사용하지만,
   개념 이해를 위해 F₁₃ 위의 스칼라 연산만 추적

═══════════════════════════════════════════
다항식: f(x) = 2x + 3
평가 점: z = 5
기대 값: y = f(5) = 2·5 + 3 = 13 ≡ 0 (mod 13)
═══════════════════════════════════════════

Step 1: Setup (τ = 7 이라 가정)

  G1 powers (스칼라 부분):
    τ⁰ = 1
    τ¹ = 7
    τ² = 49 ≡ 10 (mod 13)
    τ³ = 70 ≡ 5 (mod 13)
    τ⁴ = 35 ≡ 9 (mod 13)

  G2 powers:
    τ⁰ = 1
    τ¹ = 7

Step 2: Commit

  C = f₀ · [τ⁰]₁ + f₁ · [τ¹]₁
    = 3 · [1]₁ + 2 · [7]₁
    = [3]₁ + [14]₁
    = [3]₁ + [1]₁    (14 mod 13 = 1)
    = [4]₁

  또는 직접: f(τ) = f(7) = 2·7 + 3 = 17 ≡ 4 (mod 13)
  C = [4]₁ ✓ 일치!

Step 3: Open (z = 5)

  y = f(5) = 10 + 3 = 13 ≡ 0 (mod 13)

  n(x) = f(x) - y = f(x) - 0 = f(x) = 2x + 3
  d(x) = x - 5 = x + 8  (−5 ≡ 8 mod 13)

  q(x) = (2x + 3) / (x + 8)

  다항식 나눗셈 in F₁₃:
    2x + 3 = q · (x + 8) + r

    q = 2 (최고차 계수 맞추기: 2x / x = 2)
    2 · (x + 8) = 2x + 16 ≡ 2x + 3 (mod 13)

    r = (2x + 3) - (2x + 3) = 0 ✓

  q(x) = 2 (상수)
  π = commit(q) = 2 · [τ⁰]₁ = 2 · [1]₁ = [2]₁

Step 4: Verify

  LHS: e(π, [τ]₂) = e([2]₁, [7]₂)
       = e(G₁, G₂)^(2·7)
       = e(G₁, G₂)^14
       = e(G₁, G₂)^1    (14 mod 13 = 1)

  RHS: C - [y]₁ + z·π
       = [4]₁ - [0]₁ + 5·[2]₁
       = [4]₁ + [10]₁
       = [14]₁
       = [1]₁    (14 mod 13 = 1)

       e([1]₁, G₂) = e(G₁, G₂)^1

  LHS = RHS = e(G₁, G₂)^1  ✓ 검증 성공!
```

---

### Part 10: 검증 실패 트레이스 — 값 위조

```
═══════════════════════════════════════════
같은 설정: f(x) = 2x + 3, τ = 7, z = 5
정직한 y = 0, 위조된 y' = 3
═══════════════════════════════════════════

공격자가 y = 3 이라고 주장

공격자의 문제:
  n'(x) = f(x) - 3 = 2x + 3 - 3 = 2x
  d(x) = x - 5 = x + 8

  q'(x) = 2x / (x + 8)

  나눗셈:
    2x = 2 · (x + 8) + r
    2x = 2x + 16 ≡ 2x + 3
    r = 2x - (2x + 3) = -3 ≡ 10 (mod 13)

  r ≠ 0!  나누어떨어지지 않음!

  q'(x)가 다항식이 아니므로
  SRS만으로는 [q'(τ)]₁ 을 계산할 수 없음

공격자가 아무 π'를 제출한다면?
  π' = [k]₁ (임의의 k)

  LHS: e([k]₁, [7]₂) = e(G₁, G₂)^(7k)

  RHS: C - [y']₁ + z·π'
       = [4]₁ - [3]₁ + 5·[k]₁
       = [1 + 5k]₁
       e([1+5k]₁, G₂) = e(G₁, G₂)^(1+5k)

  LHS = RHS ⟺ 7k ≡ 1 + 5k (mod 13)
              ⟺ 2k ≡ 1 (mod 13)
              ⟺ k ≡ 7 (mod 13)  (2의 역원 = 7, 2·7=14≡1)

  공격자가 k = 7을 찾으면 검증 통과?

  하지만! F₁₃에서는 전수 탐색 가능
  실제 BN254에서 p ≈ 2²⁵⁴ → DL 문제:
    공격자는 τ = 7을 모르므로 k를 찾을 수 없음
    (이산로그 문제의 어려움)

⚠️ F₁₃ 은 교육용 — 실제 보안은 큰 소수체에서 성립
```

---

### Part 11: Batch Open — 다중 점 증명

#### 핵심 아이디어

```
단일 점:
  f(z) = y  ⟺  (x - z) | (f(x) - y)

다중 점으로 확장:
  f(z₁) = y₁, f(z₂) = y₂, ..., f(zₖ) = yₖ

  이것을 하나로 합치려면?

  Step 1: vanishing polynomial
    Z(x) = (x - z₁)(x - z₂)···(x - zₖ)
    Z(zᵢ) = 0 for all i

  Step 2: interpolation polynomial
    I(x) s.t. I(zᵢ) = yᵢ for all i
    → Lagrange 보간으로 구성

  Step 3: 핵심 관찰
    g(x) = f(x) - I(x)
    g(zᵢ) = f(zᵢ) - I(zᵢ) = yᵢ - yᵢ = 0  for all i

    → g(x)는 z₁, z₂, ..., zₖ 를 모두 근으로 가짐
    → Z(x) | g(x)
    → q(x) = g(x) / Z(x) 가 다항식

  Step 4: 증명
    π = commit(q) = [q(τ)]₁

흐름도:
  f(x), [z₁...zₖ]
       │
       ├──→ yᵢ = f(zᵢ)  ──→  {(zᵢ, yᵢ)} ──→ I(x) (Lagrange)
       │                                        │
       ├──→ Z(x) = ∏(x - zᵢ)                   │
       │         │                              │
       │         ↓                              ↓
       │    f(x) - I(x)  ──÷── Z(x) ──→ q(x)
       │                                   │
       │                                   ↓
       └──────────────────────────→ π = commit(q)
```

#### 코드 대응

```rust
fn vanishing_poly(points: &[Fr]) -> Polynomial {
    let mut result = Polynomial::from_coeffs(vec![Fr::ONE]);  // Z = 1
    for &z in points {
        let factor = Polynomial::from_coeffs(vec![-z, Fr::ONE]); // (x - z)
        result = &result * &factor;  // Z *= (x - z)
    }
    result
}

pub fn batch_open(srs: &SRS, poly: &Polynomial, points: &[Fr]) -> BatchOpening {
    assert!(!points.is_empty(), "at least one point required");

    // 1. 평가값 계산
    let values: Vec<Fr> = points.iter().map(|&z| poly.eval(z)).collect();

    // 2. vanishing polynomial: Z(x) = ∏(x - zᵢ)
    let vanishing = vanishing_poly(points);

    // 3. interpolation polynomial: I(zᵢ) = yᵢ
    let interp_points: Vec<(Fr, Fr)> = points.iter()
        .zip(values.iter())
        .map(|(&z, &y)| (z, y))
        .collect();
    let interpolation = Polynomial::lagrange_interpolate(&interp_points);

    // 4. quotient: (f(x) - I(x)) / Z(x)
    let numerator = poly - &interpolation;
    let (quotient, remainder) = numerator.div_rem(&vanishing);
    debug_assert!(remainder.is_zero());

    // 5. π = commit(q)
    let proof = commit(srs, &quotient).0;

    BatchOpening {
        points: points.to_vec(),
        values,
        proof,
    }
}
```

---

### Part 12: Batch Verify — 검증 방정식

#### 방정식 유도

```
q(x) = (f(x) - I(x)) / Z(x)
⟹ f(x) - I(x) = q(x) · Z(x)

τ에서 평가:
  f(τ) - I(τ) = q(τ) · Z(τ)

커브 포인트로:
  [f(τ)]₁ - [I(τ)]₁ = q(τ) · Z(τ)    (스칼라 관계)

페어링으로:
  e([q(τ)]₁, [Z(τ)]₂) = e([f(τ)]₁ - [I(τ)]₁, G₂)

  즉:
  e(π, [Z(τ)]₂) = e(C - [I(τ)]₁, G₂)     ... (★)

(★) 에서 필요한 것:
  - π: 증명자가 제공 (G1 점)
  - [Z(τ)]₂: 검증자가 SRS의 G2 powers로 계산
    Z(x) = ∏(x - zᵢ) 의 계수로 commit_g2 수행
    degree(Z) = k (점의 수) → g2_powers가 k+1개 필요
  - C: 원래 commitment (G1 점)
  - [I(τ)]₁: 검증자가 SRS의 G1 powers로 계산
    I(x)를 Lagrange 보간으로 구성 → commit 수행

단일 점과의 비교:
  단일 점: e(π, [τ]₂) = e(C - [y]₁ + z·π, G₂)
           → [τ]₂ 하나만 필요 (G2 powers 2개)
           → G2 commit 불필요

  다중 점: e(π, [Z(τ)]₂) = e(C - [I(τ)]₁, G₂)
           → [Z(τ)]₂ 필요 → G2 powers k+1개 필요
           → G2에서의 commit 필요
```

#### 코드 대응

```rust
pub fn batch_verify(
    srs: &SRS,
    commitment: &Commitment,
    opening: &BatchOpening,
) -> bool {
    assert!(!opening.points.is_empty());

    let g2 = srs.g2_powers[0]; // G₂

    // 1. vanishing polynomial: Z(x)
    let vanishing = vanishing_poly(&opening.points);
    assert!(
        vanishing.coeffs.len() <= srs.g2_powers.len(),
        "vanishing polynomial degree exceeds SRS G2 max degree"
    );

    // 2. [Z(τ)]₂ — G2에서 commit
    let z_g2 = commit_g2(srs, &vanishing);

    // 3. interpolation polynomial: I(x)
    let interp_points: Vec<(Fr, Fr)> = opening.points.iter()
        .zip(opening.values.iter())
        .map(|(&z, &y)| (z, y))
        .collect();
    let interpolation = Polynomial::lagrange_interpolate(&interp_points);

    // 4. [I(τ)]₁ — G1에서 commit
    let i_commitment = commit(srs, &interpolation);

    // 5. 검증: e(π, [Z(τ)]₂) = e(C - [I(τ)]₁, G₂)
    let lhs = pairing(&opening.proof, &z_g2);
    let rhs_g1 = commitment.0 + (-i_commitment.0);
    let rhs = pairing(&rhs_g1, &g2);

    lhs == rhs
}
```

```
Batch Verify 흐름도:

  Input: C, (points[], values[], π), SRS

  ┌─── LHS ──────────────────────────────┐
  │                                        │
  │  points → Z(x) = ∏(x-zᵢ)             │
  │                    │                    │
  │                    ↓ commit_g2          │
  │  π ──────┐   [Z(τ)]₂                  │
  │          │       │                      │
  │          └───┬───┘                      │
  │              │ pairing                  │
  │              ↓                          │
  │           LHS ∈ GT                     │
  └────────────────────────────────────────┘

  ┌─── RHS ──────────────────────────────┐
  │                                        │
  │  (points, values) → I(x)  (Lagrange)  │
  │                       │                 │
  │                       ↓ commit          │
  │  C ──────┐       [I(τ)]₁              │
  │          │           │                  │
  │  C - [I(τ)]₁ ←──────┘                 │
  │          │                              │
  │          └───┬────┐                    │
  │              │    │ pairing            │
  │  G₂ ────────┘    ↓                    │
  │               RHS ∈ GT                │
  └────────────────────────────────────────┘

  LHS == RHS ?  →  true / false
```

---

### Part 13: Batch 수치 트레이스 — F₁₃

```
═══════════════════════════════════════════
체: F₁₃, τ = 7
다항식: f(x) = x² + 1
평가 점: z₁ = 1, z₂ = 2, z₃ = 3
기대 값: y₁ = 2, y₂ = 5, y₃ = 10
═══════════════════════════════════════════

Step 1: 평가값 확인
  f(1) = 1 + 1 = 2   ✓
  f(2) = 4 + 1 = 5   ✓
  f(3) = 9 + 1 = 10  ✓

Step 2: Vanishing polynomial
  Z(x) = (x-1)(x-2)(x-3)

  (x-1)(x-2) = x² - 3x + 2
  (x² - 3x + 2)(x-3) = x³ - 3x² - 3x² + 9x + 2x - 6
                       = x³ - 6x² + 11x - 6

  mod 13:
  Z(x) = x³ + 7x² + 11x + 7

  검산: Z(1) = 1 + 7 + 11 + 7 = 26 ≡ 0 ✓
        Z(2) = 8 + 28 + 22 + 7 = 65 ≡ 0 ✓
        Z(3) = 27 + 63 + 33 + 7 = 130 ≡ 0 ✓

Step 3: Interpolation polynomial
  I(zᵢ) = yᵢ 를 만족하는 degree ≤ 2 다항식

  Lagrange:
    L₁(x) = (x-2)(x-3) / (1-2)(1-3) = (x-2)(x-3) / 2
    L₂(x) = (x-1)(x-3) / (2-1)(2-3) = (x-1)(x-3) / (-1)
    L₃(x) = (x-1)(x-2) / (3-1)(3-2) = (x-1)(x-2) / 2

  I(x) = 2·L₁(x) + 5·L₂(x) + 10·L₃(x)

  mod 13에서 2의 역원 = 7 (2·7 = 14 ≡ 1)
  mod 13에서 -1의 역원 = -1 = 12

  L₁(x) = 7·(x-2)(x-3) = 7·(x²-5x+6) = 7x² + 9x + 42 ≡ 7x² + 9x + 3
  L₂(x) = 12·(x-1)(x-3) = 12·(x²-4x+3) = 12x² + 4x + 36 ≡ 12x² + 4x + 10
  L₃(x) = 7·(x-1)(x-2) = 7·(x²-3x+2) = 7x² + 5x + 14 ≡ 7x² + 5x + 1

  I(x) = 2·(7x²+9x+3) + 5·(12x²+4x+10) + 10·(7x²+5x+1)
       = (14x²+18x+6) + (60x²+20x+50) + (70x²+50x+10)
       = 144x² + 88x + 66

  mod 13:
  I(x) = 1x² + 10x + 1    (144=11·13+1, 88=6·13+10, 66=5·13+1)

  검산:
    I(1) = 1 + 10 + 1 = 12 ≡ 12

  ⚠️ 계산을 다시 해보자...

  사실 f(x) = x² + 1 이므로 I(x) = x² + 1 = f(x) 이어야 함
  (f 자체가 degree 2이고, 3개의 점이 f 위에 있으므로)

  I(x) = x² + 1

  검산: I(1) = 2 ✓, I(2) = 5 ✓, I(3) = 10 ✓

Step 4: Quotient
  n(x) = f(x) - I(x) = (x² + 1) - (x² + 1) = 0

  q(x) = 0 / Z(x) = 0

  π = commit(0) = identity

Step 5: Verify
  LHS: e(identity, [Z(τ)]₂) = 1_GT  (항등원과의 pairing = GT 항등원)

  RHS: C - [I(τ)]₁
       C = [f(τ)]₁ = [τ² + 1]₁ = [10 + 1]₁ = [11]₁
       [I(τ)]₁ = [f(τ)]₁ = [11]₁   (I = f 이므로)
       C - [I(τ)]₁ = [11]₁ - [11]₁ = identity
       e(identity, G₂) = 1_GT

  LHS = RHS = 1_GT  ✓

비고: 이 경우 deg(f) = deg(I) = 2, k = 3이므로
      q = 0이 됨. deg(f) > k-1 인 경우 q ≠ 0.
```

---

### Part 14: Batch — deg(f) > k-1 인 경우

```
═══════════════════════════════════════════
f(x) = x³ + 1 (degree 3)
점: z₁ = 1, z₂ = 2 (k = 2)
═══════════════════════════════════════════

f(1) = 2, f(2) = 9

Z(x) = (x-1)(x-2) = x² - 3x + 2

I(x): I(1) = 2, I(2) = 9
  degree 1 다항식: I(x) = 7x - 5 (기울기 = (9-2)/(2-1) = 7)
  I(1) = 7-5 = 2 ✓
  I(2) = 14-5 = 9 ✓

n(x) = f(x) - I(x) = x³ + 1 - (7x - 5) = x³ - 7x + 6

mod 13: n(x) = x³ + 6x + 6

검산: n(1) = 1 + 6 + 6 = 13 ≡ 0 ✓
      n(2) = 8 + 12 + 6 = 26 ≡ 0 ✓

q(x) = n(x) / Z(x) = (x³ - 7x + 6) / (x² - 3x + 2)

다항식 나눗셈:
  x³ - 7x + 6 ÷ (x² - 3x + 2)

  첫 항: x³ / x² = x
  x · (x² - 3x + 2) = x³ - 3x² + 2x
  나머지: 3x² - 9x + 6

  둘째 항: 3x² / x² = 3
  3 · (x² - 3x + 2) = 3x² - 9x + 6
  나머지: 0

q(x) = x + 3

검산: q(x) · Z(x) = (x+3)(x²-3x+2)
  = x³ - 3x² + 2x + 3x² - 9x + 6
  = x³ - 7x + 6 ✓

이 경우 q(x) ≠ 0, π ≠ identity
→ 비자명한 batch proof
```

---

### Part 15: commit_g2 — G2에서의 Commit

```
Batch verify에서 [Z(τ)]₂ 계산에 필요

commit_g2는 commit과 동일한 로직이지만 G2에서 수행:
  G2 = srs.g2_powers[0]
  [Z(τ)]₂ = Σᵢ zᵢ · g2_powers[i]
```

```rust
fn commit_g2(srs: &SRS, poly: &Polynomial) -> G2 {
    if poly.is_zero() {
        return G2::identity();
    }
    assert!(
        poly.coeffs.len() <= srs.g2_powers.len(),
        "polynomial degree {} exceeds SRS G2 max degree {}",
        poly.degree(), srs.g2_powers.len() - 1
    );

    let mut result = G2::identity();
    for (i, &coeff) in poly.coeffs.iter().enumerate() {
        if !coeff.is_zero() {
            result = result + srs.g2_powers[i].scalar_mul(&coeff.to_repr());
        }
    }
    result
}
```

```
왜 commit_g2는 pub이 아니고 fn인가?

  commit_g2는 내부 구현 세부사항:
  - 외부 사용자는 G2 commitment이 필요 없음
  - batch_verify 내부에서만 vanishing polynomial의 G2 commit에 사용
  - API 표면을 최소화하는 Rust 관행
```

---

### Part 16: Commitment 속성 — Binding, Hiding, Additivity

#### Binding (바인딩)

```
정의: 같은 commitment C에 대해 두 가지 다른 다항식 f₁ ≠ f₂ 를
      열 수 없어야 한다.

증명 (sketch):
  C = [f₁(τ)]₁ = [f₂(τ)]₁
  ⟹ f₁(τ) = f₂(τ)
  ⟹ (f₁ - f₂)(τ) = 0

  f₁ ≠ f₂ 이면 g(x) = f₁(x) - f₂(x) 는 영 아닌 다항식
  degree(g) ≤ d 이면 g는 최대 d개의 근을 가짐

  τ가 g의 근이 되려면 τ ∈ {d개의 특정 값}
  |Fr| ≈ 2²⁵⁴ 이므로 이 확률 ≈ d / 2²⁵⁴ ≈ 0

  → 다항식이 다르면 commitment이 다름 (overwhelming probability)

코드에서 확인:

  #[test]
  fn commitment_binding() {
      // f₁ = 1 + 2x, f₂ = 3 + 4x → C₁ ≠ C₂
      let poly1 = Polynomial::from_coeffs(vec![Fr::from_u64(1), Fr::from_u64(2)]);
      let poly2 = Polynomial::from_coeffs(vec![Fr::from_u64(3), Fr::from_u64(4)]);
      let c1 = commit(&srs, &poly1);
      let c2 = commit(&srs, &poly2);
      assert_ne!(c1, c2);
  }
```

#### Hiding (은닉)

```
정의: C = [f(τ)]₁ 에서 f(x)를 복원할 수 없어야 한다.

조건: τ가 비밀이어야 함

직관:
  C = [f(τ)]₁ = f(τ) · G₁

  공격자가 아는 것: C ∈ G1 (하나의 점)
  복원하려면: f(τ) = discrete_log(C, G₁) 필요
  → DL 문제: 불가능 (BN254 위에서)

  설사 f(τ)를 알아내더라도:
  f(x) = f₀ + f₁x + ... + fₐxᵈ 에서
  f(τ)는 하나의 값 → d+1개의 미지수에 1개의 방정식
  → 무한히 많은 해 → f(x) 복원 불가

코드에서 확인:

  #[test]
  fn different_srs_different_commitment() {
      // 같은 다항식, 다른 τ → 다른 commitment
      let srs1 = setup(4, 1, &mut rng1);  // τ₁
      let srs2 = setup(4, 1, &mut rng2);  // τ₂
      let c1 = commit(&srs1, &poly);
      let c2 = commit(&srs2, &poly);
      assert_ne!(c1, c2);
  }
```

#### Additivity (동형 속성)

```
핵심: commit(f) + commit(g) = commit(f + g)

증명:
  commit(f) = [f(τ)]₁ = f(τ) · G₁
  commit(g) = [g(τ)]₁ = g(τ) · G₁

  commit(f) + commit(g) = (f(τ) + g(τ)) · G₁
                        = (f+g)(τ) · G₁
                        = [(f+g)(τ)]₁
                        = commit(f + g)

응용:
  - 다항식의 선형 결합을 commitment 수준에서 검증
  - PLONK에서 gate polynomial의 결합에 활용

코드에서 확인:

  #[test]
  fn commitment_additivity() {
      let f = Polynomial::from_coeffs(vec![Fr::from_u64(1), Fr::from_u64(2)]);
      let g = Polynomial::from_coeffs(vec![Fr::from_u64(3), Fr::from_u64(4)]);
      let fg = &f + &g;

      assert_eq!(cf.0 + cg.0, cfg.0);  // commit(f) + commit(g) = commit(f+g)
  }
```

---

### Part 17: SRS 일관성 검증 — Pairing으로 τ powers 확인

```
문제: 외부에서 받은 SRS가 정말 같은 τ에서 생성되었는가?

검증 방법:
  e([τ¹]₁, [τ¹]₂) = e([τ²]₁, [τ⁰]₂)

왜 성립하는가:
  LHS = e(τ·G₁, τ·G₂) = e(G₁, G₂)^(τ²)
  RHS = e(τ²·G₁, G₂)  = e(G₁, G₂)^(τ²)

  같은 τ라면 반드시 일치

일반화:
  e([τᵃ]₁, [τᵇ]₂) = e([τᵃ⁺ᵇ]₁, G₂)  for any a, b

  이것으로 모든 power가 일관된 τ에서 왔는지 검증 가능

코드에서 확인:

  #[test]
  fn srs_g1_powers_consistent() {
      let lhs = pairing(&srs.g1_powers[1], &srs.g2_powers[1]);
      // e([τ¹]₁, [τ¹]₂) = e(G₁, G₂)^(τ²)

      let rhs = pairing(&srs.g1_powers[2], &srs.g2_powers[0]);
      // e([τ²]₁, G₂) = e(G₁, G₂)^(τ²)

      assert_eq!(lhs, rhs);
  }

프로덕션 응용:
  MPC 세레모니에서 각 참가자의 기여가 유효한지 검증
  → 이 pairing check로 SRS의 무결성 확인
```

---

### Part 18: 검증 실패 케이스 분석

#### 케이스 1: 값 위조 (value tampering)

```
시나리오: f(5) = 13인데 y = 99로 주장

  정직한 opening: π = [q(τ)]₁ where q = (f(x)-13)/(x-5)

  위조: opening.value = 99

  검증식에서:
    LHS = e(π, [τ]₂)  (변하지 않음)
    RHS = e(C - [99]₁ + 5·π, G₂)
        ≠ e(C - [13]₁ + 5·π, G₂)  ([99]₁ ≠ [13]₁)

  → LHS ≠ RHS → false

직관: y 값이 바뀌면 RHS의 G1 점이 변해서 pairing 결과 불일치
```

#### 케이스 2: 점 변조 (point tampering)

```
시나리오: z=5에서 열었지만 z=7로 변조

  정직한 π는 q(x) = (f(x)-13)/(x-5) 에서 생성

  위조: opening.point = 7

  검증식:
    RHS = e(C - [13]₁ + 7·π, G₂)  ← z=7이므로 7·π
    vs   e(C - [13]₁ + 5·π, G₂)  ← 정상이면 5·π

  7·π ≠ 5·π → RHS 변경 → LHS ≠ RHS → false
```

#### 케이스 3: Proof 변조

```
시나리오: π에 G₁ 을 더함

  위조: opening.proof = π + G₁

  LHS = e(π + G₁, [τ]₂) ≠ e(π, [τ]₂)
  RHS = e(C - [y]₁ + z·(π + G₁), G₂) ≠ e(C - [y]₁ + z·π, G₂)

  양쪽 모두 변하지만 다르게 변함 → false
```

#### 케이스 4: 다른 commitment으로 검증

```
시나리오: f₁으로 open, f₂의 commitment으로 verify

  C₂ = commit(f₂) ≠ commit(f₁) = C₁
  π = open(f₁, z) 의 proof

  LHS = e(π, [τ]₂)  (f₁ 기반)
  RHS = e(C₂ - [y]₁ + z·π, G₂)  (C₂ 사용)

  C₂ ≠ C₁ → RHS 변경 → false
```

---

### Part 19: 계산 복잡도 분석

```
┌────────────────────┬────────────────────────────────────────┐
│ 연산               │ 복잡도                                  │
├────────────────────┼────────────────────────────────────────┤
│ setup(d, k)        │ O(d) G1 scalar_mul + O(k) G2 scalar_mul│
│ commit(f)          │ O(d) G1 scalar_mul (MSM)               │
│ open(f, z)         │ O(d) 다항식 나눗셈 + O(d) commit       │
│ verify             │ 2 pairings + 2 G1 scalar_mul           │
│ batch_open(f, k점) │ O(k²) 보간 + O(d) 나눗셈 + O(d) commit │
│ batch_verify       │ 2 pairings + O(k) G2 MSM + O(k) G1 MSM│
└────────────────────┴────────────────────────────────────────┘

핵심 관찰:
  - 검증은 항상 O(1) (2 pairings) — 다항식 차수 무관!
  - Commitment 크기도 O(1) (G1 1개 = 64 bytes) — 차수 무관!
  - 증명 크기도 O(1) (G1 1개 = 64 bytes) — 차수 무관!

  이것이 "succinct"의 핵심:
    degree 1000 다항식이나 degree 10 다항식이나
    commitment, proof, verification 모두 같은 크기/시간

병목:
  - setup: τ의 power 계산 — degree에 비례
  - commit: MSM — degree에 비례 (Pippenger로 O(d/log d) 가능)
  - open: 다항식 나눗셈 — degree에 비례
  - 모든 것은 "증명자 측" 비용
  - "검증자 측"은 항상 O(1)

Groth16과 비교:
  ┌──────────────┬────────────────┬────────────────┐
  │              │ KZG            │ Groth16        │
  ├──────────────┼────────────────┼────────────────┤
  │ Proof 크기    │ 1 G1 = 64B    │ 2G1+1G2 = 192B │
  │ Verify 시간   │ 2 pairings    │ 3 pairings     │
  │ Setup        │ universal      │ per-circuit     │
  │ Prover       │ O(d) MSM      │ O(n) MSM       │
  └──────────────┴────────────────┴────────────────┘
```

---

### Part 20: KZG와 PLONK의 관계

```
PLONK이 KZG를 사용하는 방법:

  PLONK의 핵심:
    1. 회로를 다항식들의 관계로 표현
    2. 증명자가 이 다항식들을 commit
    3. 검증자가 랜덤 점 ζ에서 evaluation 요청
    4. 증명자가 KZG opening으로 응답

  PLONK 증명 구조:
    ┌────────────────────────────────────────────┐
    │ PLONK Proof                                 │
    │                                              │
    │  Round 1: [a(τ)]₁, [b(τ)]₁, [c(τ)]₁       │ ← KZG commit
    │  Round 2: [z(τ)]₁                           │ ← KZG commit
    │  Round 3: [t_lo(τ)]₁, [t_mid(τ)]₁, [t_hi]₁│ ← KZG commit
    │  Round 4: ā, b̄, c̄, s̄σ₁, s̄σ₂, z̄ω        │ ← evaluations
    │  Round 5: [W_ζ(τ)]₁, [W_ζω(τ)]₁           │ ← KZG opening
    │                                              │
    │  검증: KZG verify (batch)                    │
    └────────────────────────────────────────────┘

  KZG 없이 PLONK은 불가능:
    - 다항식 commit에 KZG commit 사용
    - evaluation 증명에 KZG open/verify 사용
    - batch opening으로 효율적 검증

  KZG의 universal setup 덕분에:
    - PLONK도 universal setup 달성
    - 회로가 바뀌어도 SRS 재사용 가능
    - "updateable" SRS — 나중에 참가자 추가 가능
```

---

### Part 21: KZG vs 다른 Polynomial Commitment 비교

```
┌──────────────────┬─────────────┬──────────────┬──────────────┐
│                  │ KZG         │ FRI (STARK)  │ Bulletproofs │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Trusted Setup    │ 필요 (τ)    │ 불필요       │ 불필요       │
│ Proof 크기       │ O(1)        │ O(log²d)     │ O(log d)     │
│ Verify 시간      │ O(1)        │ O(log²d)     │ O(d)         │
│ Prover 시간      │ O(d)        │ O(d log d)   │ O(d)         │
│ 암호학 가정       │ DL + pairing│ hash only    │ DL           │
│ 양자 저항         │ ✗           │ ✓            │ ✗            │
│ 대표 사용처       │ PLONK       │ STARK        │ Monero       │
└──────────────────┴─────────────┴──────────────┴──────────────┘

KZG의 장점:
  - 가장 작은 proof 크기 (64 bytes!)
  - 가장 빠른 검증 (2 pairings)
  - Ethereum EIP-4844 채택

KZG의 단점:
  - Trusted setup 필요 (MPC ceremony)
  - Pairing 필요 (특정 커브에 제한)
  - 양자 컴퓨터에 취약

FRI (STARK) 는:
  - 해시만 사용 → 양자 저항
  - Trusted setup 불필요
  - 하지만 proof 크기가 큼 (~100KB)

선택 기준:
  on-chain 검증이 중요 → KZG (작은 proof, 빠른 verify)
  양자 저항이 중요 → FRI
  범용성이 중요 → KZG (PLONK 호환)
```

---

### Part 22: Trusted Setup 심화 — MPC Ceremony

```
왜 MPC가 필요한가:

  단일 참가자 setup의 문제:
    τ를 생성한 사람이 τ를 알고 있음
    → 모든 commitment을 위조할 수 있음
    → 시스템 전체가 그 한 사람에게 의존

  MPC (Multi-Party Computation):
    N명의 참가자가 각각 τᵢ를 기여
    τ = τ₁ · τ₂ · ... · τₙ

    한 명이라도 자신의 τᵢ를 삭제하면 → τ 복원 불가
    "1-of-N" 신뢰 가정

MPC 프로토콜 (Powers of Tau):

  Phase 1: Powers of tau
    참가자 1:
      τ₁ 생성
      [τ₁⁰]₁, [τ₁¹]₁, ..., [τ₁ᵈ]₁ 출력
      τ₁ 삭제

    참가자 2:
      τ₂ 생성
      기존 [τ₁ⁱ]₁ 에 τ₂ 적용:
      [τ₂ⁱ · τ₁ⁱ]₁ = [(τ₁τ₂)ⁱ]₁ 출력
      τ₂ 삭제

    ...참가자 N까지 반복

    결과: [(τ₁·τ₂·...·τₙ)ⁱ]₁ = [τⁱ]₁

    누구도 τ = τ₁·τ₂·...·τₙ 를 알 수 없음!

  검증:
    각 참가자의 기여가 유효한지 pairing check로 확인
    e([τ_new^i]₁, [τ_old^1]₂) = e([τ_old^i]₁, [τ_new^1]₂)
    → SRS 일관성 검증과 동일한 원리

실제 사례:
  ┌────────────────────┬──────────────┬─────────────┐
  │ 프로젝트            │ 참가자 수     │ 연도        │
  ├────────────────────┼──────────────┼─────────────┤
  │ Zcash Powers of Tau│ 87명         │ 2017        │
  │ Hermez             │ 1,088명      │ 2021        │
  │ EF KZG Ceremony    │ 141,416명    │ 2023        │
  └────────────────────┴──────────────┴─────────────┘

  EF KZG Ceremony:
    - EIP-4844 (Proto-Danksharding) 을 위해 실시
    - 141,416명 참가 — 역대 최대 규모
    - 한 명이라도 정직하면 τ 비밀 보장
```

---

### Part 23: random_fr / random_nonzero_fr — 랜덤 생성

```rust
fn random_fr<R: Rng>(rng: &mut R) -> Fr {
    let limbs: [u64; 4] = [
        rng.gen(), rng.gen(), rng.gen(), rng.gen(),
    ];
    Fr::from_raw(limbs)
}

fn random_nonzero_fr<R: Rng>(rng: &mut R) -> Fr {
    loop {
        let f = random_fr(rng);
        if !f.is_zero() { return f; }
    }
}
```

```
랜덤 생성 로직:

  1. 4개의 u64 (256비트) 랜덤 생성
  2. Fr::from_raw()로 Montgomery form 변환
     from_raw 내부에서 mod r 수행 (r = BN254 스칼라 체 order)

  왜 nonzero가 필요한가?
    τ = 0이면:
      g1_powers = [G₁, O, O, O, ...]  (identity)
      commit(f) = f₀ · G₁  (상수항만 남음)
      → 다항식의 정보가 완전히 손실

    확률적으로 τ = 0일 확률 ≈ 1/r ≈ 1/2²⁵⁴ ≈ 0
    하지만 명시적 방어를 위해 loop으로 보장

  Fr 크기: r ≈ 2²⁵⁴ (약 77자리 십진수)
    랜덤 충돌 확률: 2⁻²⁵⁴ ≈ 무시 가능
```

---

### Part 24: 테스트 해설

#### 테스트 1: srs_structure

```rust
#[test]
fn srs_structure() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let srs = setup(4, 2, &mut rng);
    assert_eq!(srs.g1_powers.len(), 5); // degree 0..4 → 5개
    assert_eq!(srs.g2_powers.len(), 3); // degree 0..2 → 3개
}
```

```
setup(max_degree=4, max_degree_g2=2) 호출 시:
  g1_powers: [τ⁰, τ¹, τ², τ³, τ⁴]₁ → 5개
  g2_powers: [τ⁰, τ¹, τ²]₂ → 3개

이 SRS로 가능한 것:
  - degree ≤ 4 다항식 commit
  - 단일 점 verify (g2_powers ≥ 2 필요 → ✓)
  - batch verify 최대 2점 (g2_powers ≥ k+1 필요 → 3 ≥ 3 ✓)
```

#### 테스트 5: open_verify_linear

```rust
#[test]
fn open_verify_linear() {
    let poly = Polynomial::from_coeffs(vec![Fr::from_u64(3), Fr::from_u64(2)]);
    // f(x) = 3 + 2x
    let c = commit(&srs, &poly);
    let opening = open(&srs, &poly, Fr::from_u64(5));
    // z = 5, y = f(5) = 3 + 10 = 13

    assert_eq!(opening.value, Fr::from_u64(13));
    assert!(verify(&srs, &c, &opening));
}
```

```
내부 동작 추적:

  1. commit:
     C = 3·[τ⁰]₁ + 2·[τ¹]₁ = [3 + 2τ]₁

  2. open:
     y = f(5) = 13
     n(x) = (2x + 3) - 13 = 2x - 10
     d(x) = x - 5
     q(x) = (2x - 10)/(x - 5) = 2
     π = 2·[τ⁰]₁ = [2]₁

  3. verify:
     LHS = e([2]₁, [τ]₂) = e(G₁, G₂)^(2τ)

     RHS: C - [13]₁ + 5·π
        = [3+2τ]₁ - [13]₁ + [10]₁
        = [3 + 2τ - 13 + 10]₁
        = [2τ]₁
     e([2τ]₁, G₂) = e(G₁, G₂)^(2τ)

     LHS = RHS ✓
```

#### 테스트 12: batch_open_verify

```rust
#[test]
fn batch_open_verify() {
    let srs = setup(4, 3, &mut rng);  // G2 degree 3 (3점 batch용)
    let poly = Polynomial::from_coeffs(vec![
        Fr::from_u64(1), Fr::ZERO, Fr::from_u64(1),
    ]); // f(x) = x² + 1
    let c = commit(&srs, &poly);
    let points = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
    let opening = batch_open(&srs, &poly, &points);

    assert_eq!(opening.values[0], Fr::from_u64(2));   // 1+1
    assert_eq!(opening.values[1], Fr::from_u64(5));   // 4+1
    assert_eq!(opening.values[2], Fr::from_u64(10));  // 9+1
    assert!(batch_verify(&srs, &c, &opening));
}
```

```
내부 동작 추적:

  f(x) = x² + 1
  점: [1, 2, 3] → 값: [2, 5, 10]

  Z(x) = (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6

  I(x) = x² + 1  (3점을 보간하면 degree ≤ 2, f 자체가 degree 2)

  n(x) = f(x) - I(x) = 0
  q(x) = 0 / Z(x) = 0
  π = identity

  verify:
    LHS = e(identity, [Z(τ)]₂) = 1_GT
    RHS = e(C - [I(τ)]₁, G₂) = e(identity, G₂) = 1_GT
    LHS = RHS ✓

  왜 G2 degree 3이 필요한가:
    Z(x) = x³ - 6x² + 11x - 6 → degree 3
    [Z(τ)]₂ = -6·[τ⁰]₂ + 11·[τ¹]₂ - 6·[τ²]₂ + 1·[τ³]₂
    → g2_powers[3] 필요 → max_degree_g2 ≥ 3
```

#### 테스트 14: batch_single_point_consistency

```rust
#[test]
fn batch_single_point_consistency() {
    let single = open(&srs, &poly, Fr::from_u64(5));
    let batch = batch_open(&srs, &poly, &[Fr::from_u64(5)]);

    assert_eq!(single.value, batch.values[0]);
    assert!(verify(&srs, &c, &single));
    assert!(batch_verify(&srs, &c, &batch));
}
```

```
의미: batch_open([z]) 는 single open(z) 와 의미적으로 동일

  batch 1점:
    Z(x) = (x - z)
    I(x) = y (상수)
    q(x) = (f(x) - y) / (x - z)

  single:
    q(x) = (f(x) - y) / (x - z)

  동일한 quotient → 동일한 π
  단, batch_verify는 commit_g2(Z)를 추가로 계산하므로
  약간 더 비쌈
```

#### 테스트 19: commitment_additivity

```
이 테스트는 KZG commitment의 동형 속성을 확인:

  f(x) = 1 + 2x
  g(x) = 3 + 4x
  f+g = 4 + 6x

  commit(f) + commit(g)
  = [1 + 2τ]₁ + [3 + 4τ]₁
  = [4 + 6τ]₁
  = commit(f + g)

  이 속성은 KZG의 핵심 — PLONK에서 광범위하게 활용됨
```

---

### Part 25: Groth16에서 KZG로의 진화

```
코드 구조 비교:

  Groth16 (groth16.rs):
    setup()  → ProvingKey, VerifyingKey
                (QAP 인코딩, 회로 구조 포함)
    prove()  → Proof { a: G1, b: G2, c: G1 }
    verify() → 3 pairings

  KZG (kzg.rs):
    setup()  → SRS { g1_powers, g2_powers }
                (순수 τ powers, 회로 구조 없음)
    commit() → Commitment(G1)
    open()   → Opening { point, value, proof: G1 }
    verify() → 2 pairings

공유 의존성:
  ┌─────────────┐
  │ field::Fr   │──→ 스칼라체 (from_u64, to_repr, inv 등)
  ├─────────────┤
  │ curve::G1   │──→ 타원곡선 그룹 (scalar_mul, generator)
  │ curve::G2   │
  │ curve::pairing│→ 최적 Ate pairing
  ├─────────────┤
  │ qap::Polynomial│→ 다항식 산술 (eval, div_rem, lagrange)
  └─────────────┘

Groth16은 QAP 의존성이 있지만,
KZG는 Polynomial만 사용 (R1CS/QAP 불필요)

→ KZG는 더 범용적인 프리미티브
```

---

### Part 26: EIP-4844와 KZG — 실제 사용 사례

```
EIP-4844 (Proto-Danksharding):
  Ethereum의 blob 데이터 가용성 솔루션

  작동 방식:
    1. L2 (rollup)가 blob 데이터를 다항식으로 인코딩
    2. KZG commitment으로 blob을 commit
    3. Ethereum L1이 commitment만 저장 (blob은 일시적)
    4. 누구든 KZG opening으로 blob의 특정 값 검증 가능

  왜 KZG가 선택되었나:
    - O(1) 크기의 commitment (blob 크기 무관)
    - O(1) 검증 시간 (on-chain 비용 절약)
    - 향후 Danksharding (DAS)로 확장 가능
    - erasure coding과 잘 어우러짐

  KZG Ceremony (2023):
    - EIP-4844용 SRS 생성
    - 141,416명 참가
    - max_degree = 4096 (blob 크기)
    - 결과: 4097개의 G1 powers, 2개의 G2 powers

구현에서의 대응:
  setup(4096, 1, &mut rng)
  → g1_powers: 4097개 = [τ⁰]₁ ... [τ⁴⁰⁹⁶]₁
  → g2_powers: 2개 = [τ⁰]₂, [τ¹]₂
  → 단일 점 검증만 사용 (batch 불필요)
```

---

### Part 27: 전체 의존성 그래프

```
Step 13: KZG 의존성

  ┌───────────────────────────────────────────────────────────┐
  │                                                            │
  │  Step 1: Fp (소수체)                                       │
  │    │                                                       │
  │    ├──→ Step 2: Fp2 (이차 확장체)                          │
  │    │      │                                                │
  │    │      ├──→ Step 3: Fp6 (삼차 확장)                     │
  │    │      │      │                                         │
  │    │      │      └──→ Step 4: Fp12 (이차 확장)             │
  │    │      │             │                                  │
  │    │      │             └──→ GT (페어링 타겟)               │
  │    │      │                    │                            │
  │    │      └──→ Step 6: G2 (트위스트 커브)                   │
  │    │             │             │                            │
  │    ├──→ Step 5: G1 (BN254 커브)│                            │
  │    │      │      │             │                            │
  │    │      │      └─────────────┴──→ Step 7: pairing        │
  │    │      │                              │                 │
  │    ├──→ Fr (스칼라체)                     │                 │
  │    │      │                              │                 │
  │    │      └──→ Step 11: Polynomial       │                 │
  │    │             │                       │                 │
  │    │             └───────────────────────┴──→ ★ Step 13:   │
  │    │                                         KZG           │
  │    │                                                       │
  │    │  (Groth16은 별도 경로: R1CS → QAP → Groth16)          │
  │    │   KZG는 QAP/R1CS 불필요 — 다항식에 직접 작용            │
  │    │                                                       │
  │    │                                                       │
  │    └──→ Step 14~17: PLONK (KZG를 사용)                     │
  │                                                            │
  └───────────────────────────────────────────────────────────┘

KZG의 직접 의존성:
  1. Fr — 스칼라체 (계수, 평가점, 평가값)
  2. G1 — 커밋 공간 (commitment, proof)
  3. G2 — 검증 공간 (SRS의 G2 powers)
  4. pairing — 검증 방정식 (e: G1 × G2 → GT)
  5. Polynomial — 다항식 산술 (eval, div_rem, lagrange)

KZG가 사용하지 않는 것:
  - R1CS (회로 없음)
  - QAP (circuit-to-polynomial 변환 불필요)
  - circuits (gadget 없음)
  - hash, merkle, commitment(Pedersen), signature
```

---

### Part 28: 코드 패턴 분석 — Groth16 vs KZG

```
패턴 1: to_repr() 브릿지

  Groth16:
    pk.alpha_g1.scalar_mul(&s.to_repr())

  KZG:
    srs.g1_powers[i].scalar_mul(&coeff.to_repr())

  동일한 패턴! Montgomery → raw 변환 후 scalar_mul

패턴 2: pairing 검증

  Groth16: 3개의 pairing
    e(A, B) = e(alpha, beta) · e(L, gamma) · e(C, delta)
    → product of pairings

  KZG: 2개의 pairing
    e(π, [τ]₂) = e(C - [y]₁ + z·π, G₂)
    → LHS == RHS 비교

패턴 3: 랜덤 생성

  Groth16:
    random_fr()로 r, s 생성 (blinding factors)

  KZG:
    random_nonzero_fr()로 τ 생성
    → τ = 0 방지가 추가됨

패턴 4: 다항식 활용

  Groth16:
    QAP의 Polynomial: eval_domain, lagrange, div_rem
    → R1CS 행렬을 다항식으로 변환하는 데 사용

  KZG:
    Polynomial: eval, from_coeffs, constant, div_rem, lagrange
    → 직접 다항식을 commit/open/verify하는 데 사용

패턴 5: Identity 처리

  Groth16:
    G1::identity()는 사용하지 않음 (generator 기반)

  KZG:
    G1::identity() — 영다항식 commit, 누적 시작점
    G2::identity() — commit_g2의 시작점
```

---

### Part 29: vanishing_poly — 소거 다항식

```
vanishing_poly(points) = ∏(x - zᵢ)

구현 방식: 점진적 곱셈

  result = 1 (단항식)

  for z in points:
    factor = (x - z)
    result = result * factor

예시: points = [1, 2, 3]

  시작: result = 1

  z = 1: result = 1 · (x - 1) = x - 1
  z = 2: result = (x-1) · (x-2) = x² - 3x + 2
  z = 3: result = (x²-3x+2) · (x-3) = x³ - 6x² + 11x - 6

  최종: Z(x) = x³ - 6x² + 11x - 6

성질:
  degree(Z) = |points| = k
  Z(zᵢ) = 0 for all i (각 zᵢ가 근)

  이것은 batch opening의 핵심:
    f(x) - I(x) 가 모든 zᵢ에서 0
    → Z(x) | (f(x) - I(x))
    → quotient q(x) 존재
```

---

### Part 30: 정리 — KZG의 완전성과 건전성

```
완전성 (Completeness):
  정직한 증명자가 올바른 f(z) = y를 증명하면
  → verify는 항상 true 반환

  증명:
    q(x) = (f(x) - y) / (x - z) 가 다항식 (인수정리)
    → π = [q(τ)]₁ 은 유효한 G1 점
    → 검증 방정식: e(π, [τ]₂) = e(C - [y]₁ + z·π, G₂)

    LHS = e([q(τ)]₁, [τ]₂) = e(G₁, G₂)^(q(τ)·τ)

    RHS = e([f(τ) - y + z·q(τ)]₁, G₂) = e(G₁, G₂)^(f(τ)-y+z·q(τ))

    f(τ) - y = q(τ)·(τ-z) = q(τ)·τ - z·q(τ)
    → f(τ) - y + z·q(τ) = q(τ)·τ

    → LHS = RHS ✓

건전성 (Soundness):
  위조자가 f(z) ≠ y인데 verify를 통과하는 것은
  DL 문제를 푸는 것과 동등

  직관:
    f(z) ≠ y → (f(x) - y)는 (x-z)로 나누어떨어지지 않음
    → q(x)가 유리함수 (다항식 아님)
    → SRS의 다항식 basis로는 [q(τ)]₁ 계산 불가
    → τ 자체를 알아야 함 → DL 문제

  형식적으로: q-SDH (q-Strong Diffie-Hellman) 가정

지식 건전성 (Knowledge Soundness):
  유효한 opening을 생성할 수 있다면
  → 증명자는 실제로 다항식 f(x)를 "알고 있다"
  → extraction: 증명자를 여러 번 실행하여 f(x) 추출 가능
```

---

### Part 31: 코드 전체 구조 정리

```
kzg.rs 구성 (약 690줄):

  ┌─ imports (3줄)
  │   Fr, G1, G2, pairing, Polynomial, Rng
  │
  ├─ random helpers (18줄)
  │   random_fr(): 4×u64 → Fr::from_raw
  │   random_nonzero_fr(): loop until !is_zero()
  │
  ├─ data structures (56줄)
  │   SRS { g1_powers, g2_powers }
  │   Commitment(G1)
  │   Opening { point, value, proof }
  │   BatchOpening { points, values, proof }
  │
  ├─ setup() (24줄)
  │   τ 생성 → powers of τ in G1, G2
  │
  ├─ commit() (18줄)
  │   MSM: Σ fᵢ · [τⁱ]₁
  │
  ├─ commit_g2() (18줄)
  │   동일 로직, G2 위에서 (batch용, private)
  │
  ├─ open() (21줄)
  │   y=f(z), q=(f-y)/(x-z), π=commit(q)
  │
  ├─ verify() (23줄)
  │   e(π,[τ]₂) == e(C-[y]₁+z·π, G₂)
  │
  ├─ vanishing_poly() (8줄)
  │   Z(x) = ∏(x - zᵢ)
  │
  ├─ batch_open() (33줄)
  │   Z, I 구성 → q=(f-I)/Z → π=commit(q)
  │
  ├─ batch_verify() (36줄)
  │   e(π,[Z(τ)]₂) == e(C-[I(τ)]₁, G₂)
  │
  └─ tests (280줄, 19개)
      SRS 구조/일관성, commit, open/verify,
      실패 케이스, binding/hiding/additivity, batch
```

---

### Part 32: 다음 단계 — PLONK으로의 확장

```
Step 13 (KZG) 완성 → Step 14~17 (PLONK) 진행

PLONK이 KZG 위에 추가하는 것:

  1. Arithmetization (회로 → 다항식)
     R1CS 대신 "gate equations"과 "copy constraints" 사용
     → 더 유연한 회로 표현

  2. Permutation argument
     wiring 관계를 다항식으로 인코딩
     → σ(x) permutation polynomial

  3. Quotient polynomial
     모든 constraints를 하나의 다항식 관계로 합침
     → t(x) = (gate_eq + perm_eq + ...) / Z_H(x)

  4. Linearization
     검증자의 연산을 줄이기 위한 최적화
     → 여러 polynomial commitment을 하나로 합침

  5. KZG opening
     검증자가 랜덤 ζ 선택
     → 증명자가 모든 다항식의 ζ에서의 값을 KZG로 증명

  흐름:
    PLONK 회로
      ↓ arithmetize
    다항식들 {a(x), b(x), c(x), z(x), t(x), ...}
      ↓ KZG commit
    commitments {[a]₁, [b]₁, [c]₁, [z]₁, [t]₁, ...}
      ↓ KZG open (at ζ)
    evaluations + opening proofs
      ↓ KZG verify (batch)
    true / false
```

---

> [!summary] Step 13 요약
> ```
> KZG = 다항식 인수정리 + 타원곡선 + 페어링
>
> 핵심 방정식:
>   f(z) = y  ⟺  (x-z) | (f(x)-y)
>   검증: e(π, [τ]₂) = e(C - [y]₁ + z·π, G₂)
>
> 코드: kzg.rs (690줄, 19 테스트)
>   setup → commit → open → verify (단일 점)
>   batch_open → batch_verify (다중 점)
>
> 의존성: Fr, G1, G2, pairing, Polynomial
> 독립성: R1CS, QAP 불필요 (범용 프리미티브)
>
> 다음: PLONK (KZG를 사용하는 범용 ZK 시스템)
> ```
