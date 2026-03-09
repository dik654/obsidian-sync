## Step 14: PLONKish Arithmetization — 하나의 게이트로 모든 연산을 표현하다

### 핵심 질문: R1CS의 한계는 무엇인가?

```
R1CS의 문제:

  R1CS 게이트: (Σ aᵢsᵢ)(Σ bᵢsᵢ) = (Σ cᵢsᵢ)

  이것은 본질적으로 "곱셈 전용":
    x * y = z  →  1개 제약
    x + y = z  →  1개 제약 (a=x+y, b=1, c=z)

  하지만: x * y + x = z 를 표현하려면?
    보조변수 필요: t = x*y (1개 제약), t + x = z (1개 제약)
    → 2개 제약으로 분해해야 함

  PLONKish는 이것을 하나의 제약으로:
    q_M·x·y + q_L·x + q_O·z = 0
    → 1·x·y + 1·x + (-1)·z = 0

  "곱셈도 덧셈도 하나의 범용 게이트에!"
```

> [!important] PLONKish의 핵심 아이디어
> ```
> R1CS: 곱셈 하나 = 제약 하나, 행렬 3개 (A, B, C)
> PLONKish: selector로 게이트 유형 선택, 범용 게이트 하나
>
> 범용 게이트 방정식:
>   q_L·a + q_R·b + q_O·c + q_M·a·b + q_C = 0
>
> selector 값에 따라:
>   덧셈: q_L=1, q_R=1, q_O=-1 → a+b=c
>   곱셈: q_M=1, q_O=-1 → a·b=c
>   상수: q_L=1, q_C=-v → a=v
> ```

---

### PLONKish란?

```
PLONK [Gabizon-Williamson-Ciobotaru, 2019]:
  Permutations over Lagrange-bases for Oecumenical
  Noninteractive arguments of Knowledge

특징:
  ┌──────────────────┬──────────────────────────────┐
  │ 게이트 유형       │ 범용 (selector로 결정)        │
  │ 제약 형태         │ q_L·a + q_R·b + q_O·c +     │
  │                  │ q_M·a·b + q_C = 0            │
  │ 와이어 연결       │ copy constraint (permutation) │
  │ Setup            │ universal (KZG 기반)          │
  │ Custom gate      │ selector 추가로 확장 가능      │
  │ 건전성           │ Schwartz-Zippel + permutation │
  └──────────────────┴──────────────────────────────┘

이 단계의 범위:
  ✓ arithmetization (게이트 + 제약 시스템)
  ✓ permutation argument (copy constraint + grand product)
  ✗ prover/verifier (Step 16)
  ✗ Plookup (Step 15)
```

---

### Part 1: 범용 게이트 — 5개의 selector

#### 게이트 방정식

```
q_L·a + q_R·b + q_O·c + q_M·a·b + q_C = 0

각 기호의 의미:
  a, b, c — 3개의 wire 값 (left, right, output)
  q_L     — left wire의 선형 계수
  q_R     — right wire의 선형 계수
  q_O     — output wire의 선형 계수
  q_M     — 곱셈 항의 계수
  q_C     — 상수 항

시각화:
  ┌─────────────────────────────────────────────┐
  │                 PLONK Gate                    │
  │                                               │
  │   a ──→ ×q_L ──┐                             │
  │                 │                             │
  │   b ──→ ×q_R ──┼──→ [ + ] ──→ = 0           │
  │                 │       ↑                     │
  │   c ──→ ×q_O ──┘       │                     │
  │                         │                     │
  │   a·b ──→ ×q_M ────────┘                     │
  │                         │                     │
  │   q_C ─────────────────┘                     │
  └─────────────────────────────────────────────┘
```

#### 게이트 종류별 selector 값

```
┌────────────────┬─────┬─────┬─────┬─────┬─────┬─────────────────┐
│ 게이트          │ q_L │ q_R │ q_O │ q_M │ q_C │ 의미             │
├────────────────┼─────┼─────┼─────┼─────┼─────┼─────────────────┤
│ addition       │  1  │  1  │ -1  │  0  │  0  │ a + b = c       │
│ multiplication │  0  │  0  │ -1  │  1  │  0  │ a · b = c       │
│ constant(v)    │  1  │  0  │  0  │  0  │ -v  │ a = v           │
│ boolean        │  1  │  0  │  0  │ -1  │  0  │ a·(1-a) = 0     │
│ dummy          │  0  │  0  │  0  │  0  │  0  │ 0 = 0 (패딩용)   │
│ custom         │ ... │ ... │ ... │ ... │ ... │ 자유 정의        │
└────────────────┴─────┴─────┴─────┴─────┴─────┴─────────────────┘

예시 — 덧셈 게이트 검증:
  q_L·a + q_R·b + q_O·c + q_M·a·b + q_C
  = 1·3 + 1·4 + (-1)·7 + 0·3·4 + 0
  = 3 + 4 - 7
  = 0 ✓

예시 — 곱셈 게이트 검증:
  = 0·3 + 0·4 + (-1)·12 + 1·3·4 + 0
  = -12 + 12
  = 0 ✓
```

#### Boolean Gate의 유도

```
목표: a ∈ {0, 1} 이라는 조건을 게이트 방정식으로 표현

Step 1: 대수적 변환
  a ∈ {0, 1}
  ⟺ a = 0 또는 a = 1
  ⟺ a·(1 - a) = 0           ← 유일한 근이 0, 1인 2차 다항식
  ⟺ a - a² = 0

Step 2: PLONKish 게이트로 매핑
  범용 게이트: q_L·a + q_R·b + q_O·c + q_M·a·b + q_C = 0

  핵심 아이디어: b = a 로 설정! (copy constraint로 강제)

  그러면 a·b = a·a = a², 따라서:
    q_L·a + q_M·a·a = 0  (q_R=0, q_O=0, q_C=0)

  a - a² = 0 을 얻으려면:
    q_L = 1, q_M = -1

  대입 검증:
    1·a + (-1)·a·a = a - a² = a(1-a) = 0 ✓

Step 3: 구체적 값으로 확인
  a=0: 1·0 + (-1)·0·0 = 0 - 0 = 0   ✓ (boolean)
  a=1: 1·1 + (-1)·1·1 = 1 - 1 = 0   ✓ (boolean)
  a=2: 1·2 + (-1)·2·2 = 2 - 4 = -2  ✗ (not boolean!)
  a=½: 1·½ + (-1)·½·½ = ½ - ¼ = ¼   ✗ (not boolean!)

주의: b = a 는 반드시 copy constraint로 보장해야 함!
  cs.add_gate(PlonkGate::boolean_gate(), x, x, dummy);
  cs.copy_constraint((A, i), (B, i));   // ← 이것이 없으면 불건전

  copy constraint 없이는 prover가 b ≠ a 인 값을 넣어
  부등식을 우회할 수 있음:
    a=5, b=1/5 → q_L·5 + q_M·5·(1/5) = 5 - 1 = 4 ≠ 0
    하지만 a=2, b=½ → 1·2 + (-1)·2·½ = 2 - 1 = 1 ≠ 0
    ... 우연히 0이 되는 (a,b) 쌍이 존재 가능!
```

#### 왜 5개의 selector면 충분한가

```
범용 게이트 q_L·a + q_R·b + q_O·c + q_M·a·b + q_C = 0 은
degree 2 이하의 모든 다항식 조건을 표현할 수 있다.

증명:
  3변수 a, b, c에 대한 degree ≤ 2 단항식:
    degree 0: 1          → q_C
    degree 1: a, b, c    → q_L, q_R, q_O
    degree 2: a·b        → q_M
    (a·c, b·c, a², b², c² 는 직접 불가 → copy constraint + 보조게이트)

  왜 a·b만 있고 a·c, b·c는 없는가?
    → 게이트 하나에 selector 5개로 제한하여 prover의 부담을 줄임
    → 필요하면 보조변수 + 게이트를 추가하여 분해 가능
    → 이것이 PLONKish의 "충분히 범용적이면서 효율적인" 설계

  예: a·c = z 를 표현하려면
    Gate i: a·b = z (b에 c값, copy constraint (B,i) ↔ (C,j))
    → c를 wire B에 라우팅하면 됨!
```

---

### Part 2: R1CS vs PLONKish — 구체적 비교

```
예시 회로: x² + x + 5 = y (x=3 → y=17)

═══ R1CS 방식 ═══
  보조변수: t = x·x

  제약 1: x · x = t         (곱셈)
  제약 2: t + x + 5 = y     (덧셈 → 1·(t+x+5) · 1 = y)

  행렬: A, B, C 각각 2×5

═══ PLONKish 방식 ═══
  게이트 0: mul, a=x, b=x, c=t
    0·x + 0·x + (-1)·t + 1·x·x + 0 = x²-t = 0 ✓

  게이트 1: custom, a=t, b=x, c=y
    1·t + 1·x + (-1)·y + 0·t·x + 5 = t+x+5-y = 0 ✓

  Copy constraint: gate0.c(=t) ↔ gate1.a(=t)

  장점:
    - "t + x + 5 = y"를 하나의 게이트로! (R1CS는 변환 필요)
    - selector로 직관적으로 표현
    - copy constraint가 wire 연결을 명시적으로 관리
```

---

### Part 3: Wire Position과 Copy Constraint

#### Wire Position의 개념

```
n개의 게이트가 있으면 총 3n개의 wire position:

  게이트 0: (A,0)  (B,0)  (C,0)
  게이트 1: (A,1)  (B,1)  (C,1)
  ...
  게이트 n-1: (A,n-1)  (B,n-1)  (C,n-1)

  Column: A(left), B(right), C(output)
  Row: 게이트 번호

예시 — x³+x+5=y 회로 (x=3):

  변수: x=3, v1=9(x²), v2=27(x³), y=35

  Gate 0: mul  a=x   b=x   c=v1   → x·x=v1
  Gate 1: mul  a=v1  b=x   c=v2   → v1·x=v2
  Gate 2: add  a=v2  b=x   c=y    → v2+x+5=y
  Gate 3: dummy                     (padding)

  Wire positions:
        A     B     C
  Row 0: x     x     v1
  Row 1: v1    x     v2
  Row 2: v2    x     y
  Row 3: 0     0     0     (dummy)

  같은 변수가 여러 position에 등장:
    x: (A,0), (B,0), (B,1), (B,2)
    v1: (C,0), (A,1)
    v2: (C,1), (A,2)

  → Copy constraints로 이들을 연결!
```

#### Copy Constraint

```
Copy constraint는 "두 wire position의 값이 같아야 한다"를 강제

사용 예:
  cs.copy_constraint((A,0), (B,0))   // 둘 다 x
  cs.copy_constraint((B,0), (B,1))   // 둘 다 x
  cs.copy_constraint((B,1), (B,2))   // 둘 다 x
  cs.copy_constraint((C,0), (A,1))   // 둘 다 v1
  cs.copy_constraint((C,1), (A,2))   // 둘 다 v2

결과: 같은 변수를 공유하는 position들이 equivalence class를 형성

  x의 class:  {(A,0), (B,0), (B,1), (B,2)}
  v1의 class: {(C,0), (A,1)}
  v2의 class: {(C,1), (A,2)}
  y의 class:  {(C,2)}  (단독)

검증자는 이 equivalence class 관계가 만족되는지를
grand product로 확인한다.
```

---

### Part 4: Permutation σ — Cycle 표현

#### 왜 Permutation인가

```
Copy constraint를 인코딩하는 방법:
  각 equivalence class를 cycle로 표현

  예: x의 class {(A,0), (B,0), (B,1), (B,2)}

  σ(A,0) = (B,0)
  σ(B,0) = (B,1)
  σ(B,1) = (B,2)
  σ(B,2) = (A,0)    ← cycle 완성

  시각화:
  (A,0) → (B,0) → (B,1) → (B,2) → (A,0)
    ↑                                 │
    └─────────────────────────────────┘

  Copy constraint가 없는 position: self-loop
  σ(C,2) = (C,2)  (y는 한 곳에서만 사용)
```

#### Cycle 구성 알고리즘

```
Union-Find 기반:

1. 초기화: 각 position이 자기 자신의 parent
   parent[(A,0)] = (A,0)
   parent[(B,0)] = (B,0)
   ...

2. Copy constraint 처리:
   copy_constraint((A,0), (B,0)) → union(A0, B0)
   copy_constraint((B,0), (B,1)) → union(B0, B1)
   ...

3. Equivalence class 그룹핑:
   같은 root를 가진 position끼리 모음

4. 각 class 내에서 cycle 생성:
   [p₁, p₂, ..., pₖ] → σ(p₁)=p₂, σ(p₂)=p₃, ..., σ(pₖ)=p₁
```

---

### Part 5: Position Tag — 코셋으로 Column 구분

```
각 wire position에 고유한 Fr 값(tag) 부여:

  (A, i) → 1 · ωⁱ         column A = {ωⁱ}
  (B, i) → K1 · ωⁱ        column B = K1·{ωⁱ}
  (C, i) → K2 · ωⁱ        column C = K2·{ωⁱ}

여기서:
  ω = primitive n-th root of unity (ωⁿ = 1)
  K1 = 2, K2 = 3 (코셋 생성원)

왜 3개의 코셋이 서로소(disjoint)인가?

  H = {1, ω, ω², ..., ω^(n-1)} 는 Fr*의 부분군 (multiplicative subgroup)
  K1·H = {K1, K1·ω, K1·ω², ...}  (H의 left coset)
  K2·H = {K2, K2·ω, K2·ω², ...}  (H의 left coset)

═══ 정리: K ∉ H 이면 H ∩ K·H = ∅ ═══

  증명 (귀류법):
    H ∩ K·H ≠ ∅ 라고 가정하면,
    ∃ h₁, h₂ ∈ H such that h₁ = K·h₂

    그러면 K = h₁ · h₂⁻¹

    H는 군이므로 h₁ · h₂⁻¹ ∈ H
    따라서 K ∈ H — 모순! ∎

  같은 논리로 K1·H ∩ K2·H = ∅:
    K1·h₁ = K2·h₂ 라면 K2/K1 = h₁/h₂ ∈ H
    즉 K2·K1⁻¹ ∈ H 이면 교차, 아니면 서로소

═══ K1=2, K2=3이 H에 속하지 않는 이유 ═══

  H의 원소는 ωⁱ 형태 (i = 0, ..., n-1)
  ω는 primitive n-th root of unity, 즉 ord(ω) = n

  K1 = 2 ∈ H ⟺ ∃ i: 2 = ωⁱ ⟺ 2ⁿ = (ωⁱ)ⁿ = ωⁿⁱ = 1 (mod r)

  BN254에서 r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

  ord(2) in Fr* 는?
    ord(2) | r-1 (Fermat)
    r-1 = 2²⁸ · t (t는 홀수, ≈ 2²²⁶)

    n ≤ 2²⁸ 이므로 n | 2²⁸
    2ⁿ ≡ 1 (mod r) 이려면 ord(2) | n | 2²⁸
    → ord(2) 는 2의 거듭제곱이어야 함

    하지만 ord(2) 는 r-1 ≈ 2²⁵⁴ 에 가까운 매우 큰 수
    (2는 Fr*의 "거의 생성원"에 가까움)
    ord(2) ≫ 2²⁸ 이므로 ord(2) ∤ n

    따라서 2ⁿ ≢ 1 (mod r), 즉 2 ∉ H ✓

  K2 = 3에 대해서도 동일한 논증 적용

  K2·K1⁻¹ = 3·2⁻¹ = 3·(r+1)/2 mod r 도 H에 속하지 않음
  (이유: ord(3·2⁻¹) 도 매우 큼)

  → H, K1·H, K2·H 는 pairwise disjoint
  → 3n개의 tag가 모두 distinct ✓

예시 (n=4):
  H = {1, ω, ω², ω³}

  Column A tags: 1, ω, ω², ω³
  Column B tags: 2, 2ω, 2ω², 2ω³
  Column C tags: 3, 3ω, 3ω², 3ω³

  총 12개, 모두 서로 다름
```

---

### Part 6: Domain — Roots of Unity

```
PLONK은 evaluation domain H = {1, ω, ω², ..., ω^(n-1)} 위에서 동작

BN254 Fr:
  r - 1 = 2²⁸ · t (t는 홀수)
  TWO_ADICITY = 28
  → 최대 2²⁸ = 268,435,456 크기 도메인 지원

ω 계산 방법:
  1. 기본 상수: ROOT_OF_UNITY_2_28 = 5^((r-1)/2^28) mod r
     이것은 2²⁸-th primitive root of unity

  2. n = 2^k 크기 도메인의 ω:
     ω = ROOT_OF_UNITY_2_28 ^ (2^(28-k))

     예: n = 4 (k=2)
     ω₄ = ROOT_2_28 ^ (2²⁶)
     → ω₄⁴ = ROOT_2_28 ^ (4·2²⁶) = ROOT_2_28 ^ (2²⁸) = 1 ✓
     → ω₄² = ROOT_2_28 ^ (2²⁷) = -1 ≠ 1 ✓ (primitive)

검증 속성:
  ω^n = 1              (n-th root)
  ω^(n/2) = -1 ≠ 1    (primitive)
  원소 {1, ω, ..., ω^(n-1)} 모두 서로 다름
```

```rust
pub struct Domain {
    pub n: usize,           // 도메인 크기 (2의 거듭제곱)
    pub omega: Fr,          // primitive n-th root of unity
    pub omega_inv: Fr,      // ω⁻¹
    pub elements: Vec<Fr>,  // [1, ω, ω², ..., ω^(n-1)]
}

impl Domain {
    pub fn new(n: usize) -> Self {
        let root = Fr::from_raw(ROOT_OF_UNITY_2_28_RAW);
        let mut omega = root;
        for _ in 0..(28 - n.trailing_zeros() as usize) {
            omega = omega * omega;  // 2^(28-k)번 제곱
        }
        // ... elements 생성, omega^n = 1 검증
    }
}
```

---

### Part 6.5: Vanishing Polynomial과 Lagrange 기저 — Roots of Unity의 특수 구조

#### Vanishing Polynomial Z_H(x)

```
정의:
  Z_H(x) = ∏_{i=0}^{n-1} (x - ωⁱ) = x^n - 1

이 등식이 성립하는 이유:

  Step 1: ω가 primitive n-th root of unity이면
    ωⁿ = 1, ω^k ≠ 1 (0 < k < n)

  Step 2: 다항식 x^n - 1 의 근은?
    x^n = 1 의 해 = {ω⁰, ω¹, ..., ω^(n-1)} = H

  Step 3: 근과 다항식의 관계
    degree n 다항식이 n개의 서로 다른 근을 가짐
    → 완전 인수분해: x^n - 1 = ∏_{i=0}^{n-1} (x - ωⁱ) ∎

특수 성질:
  Z_H(ωⁱ) = 0  for all i ∈ {0, ..., n-1}    ← 도메인 위에서 영
  Z_H(x) ≠ 0  for x ∉ H                     ← 도메인 밖에서 비영
  Z_H'(ωⁱ) = n · ω^(-i)                      ← 미분값 (아래 유도)

Z_H'(x) 계산:
  Z_H(x) = x^n - 1
  Z_H'(x) = n·x^(n-1)
  Z_H'(ωⁱ) = n·(ωⁱ)^(n-1) = n·ω^(i(n-1)) = n·ω^(in)·ω^(-i) = n·1·ω^(-i)
            = n · ω^(-i) ✓
```

#### Lagrange 기저 다항식 L_i(x)

```
정의:
  L_i(ωʲ) = δ_{ij} = { 1  if i = j
                       { 0  if i ≠ j

L_i(x) 의 닫힌 형태:

  표준 Lagrange 공식:
    L_i(x) = ∏_{j≠i} (x - ωʲ) / ∏_{j≠i} (ωⁱ - ωʲ)

  roots of unity에서의 단순화:

  Step 1: 분자
    ∏_{j≠i} (x - ωʲ) = Z_H(x) / (x - ωⁱ) = (x^n - 1) / (x - ωⁱ)

  Step 2: 분모
    ∏_{j≠i} (ωⁱ - ωʲ)

    이것을 계산하기 위해 Z_H'(ωⁱ) 를 사용:
    Z_H(x) = (x - ωⁱ) · ∏_{j≠i} (x - ωʲ)

    양변을 미분:
    Z_H'(x) = ∏_{j≠i} (x - ωʲ) + (x - ωⁱ) · [...]

    x = ωⁱ 대입:
    Z_H'(ωⁱ) = ∏_{j≠i} (ωⁱ - ωʲ) = n · ω^(-i)

  Step 3: 결합
    L_i(x) = (x^n - 1) / (n · ω^(-i) · (x - ωⁱ))
           = ω^i · (x^n - 1) / (n · (x - ωⁱ))

  ↓ 정리

  ┌─────────────────────────────────────────────┐
  │  L_i(x) = ωⁱ/n · (x^n - 1)/(x - ωⁱ)      │
  └─────────────────────────────────────────────┘

  검증 (x = ωⁱ):
    직접 대입하면 0/0 형태 → L'Hôpital:
    lim_{x→ωⁱ} (x^n - 1)/(x - ωⁱ) = n·(ωⁱ)^(n-1) / 1 = n·ω^(-i)
    L_i(ωⁱ) = ωⁱ/n · n·ω^(-i) = 1 ✓

  검증 (x = ωʲ, j ≠ i):
    L_i(ωʲ) = ωⁱ/n · ((ωʲ)^n - 1)/(ωʲ - ωⁱ) = ωⁱ/n · 0/(ωʲ - ωⁱ) = 0 ✓
```

#### 수치 예시 (n=4)

```
n=4, ω = i (허수단위... 실제로는 Fr의 원소지만 개념 설명용)

도메인 H = {1, ω, ω², ω³} = {1, i, -1, -i}

Z_H(x) = x⁴ - 1

L_0(x) = 1/4 · (x⁴-1)/(x-1)
       = 1/4 · (x³ + x² + x + 1)
       → L_0(1) = 1/4·4 = 1 ✓
       → L_0(i) = 1/4·(−i−1+i+1) = 0 ✓

L_1(x) = ω/4 · (x⁴-1)/(x-ω) = i/4 · (x⁴-1)/(x-i)

임의의 다항식 f(x)를 도메인 위의 값으로 표현:
  f(x) = Σᵢ f(ωⁱ) · L_i(x)

  이것이 Lagrange 보간의 본질:
  n개의 (ωⁱ, yᵢ) 점을 지나는 degree < n 다항식을
  L_i(x) 기저의 선형결합으로 구성
```

#### 왜 Roots of Unity가 유리한가

```
일반적인 Lagrange 보간 vs Roots of Unity 위의 보간:

  ┌──────────────────────┬──────────────────────────────┐
  │ 일반 도메인            │ Roots of Unity 도메인          │
  ├──────────────────────┼──────────────────────────────┤
  │ Z(x) 계산: O(n)      │ Z_H(x) = x^n - 1  (즉시!)   │
  │ L_i(x) 분모: O(n)    │ 분모 = n·ω^(-i) (상수 시간!)  │
  │ 보간: O(n²)           │ FFT: O(n log n) !            │
  │ 멀티 eval: O(n²)      │ FFT: O(n log n) !            │
  └──────────────────────┴──────────────────────────────┘

  프로덕션 PLONK에서는:
    - Lagrange → coefficient: IFFT
    - coefficient → evaluation: FFT
    - 다항식 곱셈: FFT → pointwise mul → IFFT

  교육용 구현(우리 코드)에서는:
    - O(n²) Lagrange 보간 직접 사용
    - 개념적으로 동일, 성능만 다름
```

---

### Part 7: Selector 다항식과 Wire 다항식

#### 아이디어

```
n개의 게이트가 있을 때, selector와 wire 값을
도메인 점에서의 다항식 평가값으로 인코딩:

  q_L(ωⁱ) = gate_i의 q_l 값
  q_R(ωⁱ) = gate_i의 q_r 값
  ...
  a(ωⁱ)   = gate_i의 wire A 값
  b(ωⁱ)   = gate_i의 wire B 값
  c(ωⁱ)   = gate_i의 wire C 값

Lagrange 보간으로 이 값들을 지나는 다항식을 구성

결과:
  q_L(x), q_R(x), q_O(x), q_M(x), q_C(x)  — selector 다항식 (5개)
  a(x), b(x), c(x)                          — wire 다항식 (3개)

이 8개의 다항식이 게이트 방정식을 인코딩:
  ∀i: q_L(ωⁱ)·a(ωⁱ) + q_R(ωⁱ)·b(ωⁱ) + q_O(ωⁱ)·c(ωⁱ)
     + q_M(ωⁱ)·a(ωⁱ)·b(ωⁱ) + q_C(ωⁱ) = 0
```

#### 다항식 수준의 게이트 방정식

```
정의:
  G(x) = q_L(x)·a(x) + q_R(x)·b(x) + q_O(x)·c(x)
        + q_M(x)·a(x)·b(x) + q_C(x)

G(x)의 degree 분석:
  각 selector, wire 다항식의 degree ≤ n-1 (n개 점을 보간)

  선형 항: q_L(x)·a(x)  → degree ≤ 2(n-1)
  곱셈 항: q_M(x)·a(x)·b(x)  → degree ≤ 3(n-1)

  따라서 deg(G) ≤ 3(n-1) = 3n - 3
```

```
═══ 정리: G(ωⁱ) = 0 ∀i ⟺ Z_H(x) | G(x) ═══

  여기서 Z_H(x) = x^n - 1 = ∏_{i=0}^{n-1} (x - ωⁱ)

  (⟸) 자명: Z_H(x) | G(x) 이면 G(x) = Q(x)·Z_H(x)
       G(ωⁱ) = Q(ωⁱ)·Z_H(ωⁱ) = Q(ωⁱ)·0 = 0 ✓

  (⟹) 증명:
    G(ωⁱ) = 0 for all i = 0, ..., n-1

    다항식 나눗셈 (유클리드 알고리즘):
      G(x) = Q(x)·Z_H(x) + R(x),  deg(R) < deg(Z_H) = n

    모든 ωⁱ에서:
      G(ωⁱ) = Q(ωⁱ)·Z_H(ωⁱ) + R(ωⁱ)
      0 = Q(ωⁱ)·0 + R(ωⁱ)
      R(ωⁱ) = 0

    R(x)는 degree < n 다항식인데 n개의 서로 다른 근을 가짐
    → R(x) ≡ 0 (영다항식)

    따라서 G(x) = Q(x)·Z_H(x), 즉 Z_H(x) | G(x) ∎

  Q(x)의 degree:
    deg(Q) = deg(G) - deg(Z_H) ≤ (3n-3) - n = 2n - 3

왜 이것이 중요한가?

  Verifier의 전략:
    1. Prover가 G(x) = Q(x)·Z_H(x) 라고 주장
    2. Verifier가 랜덤 ζ ∈ Fr 선택
    3. G(ζ) = Q(ζ)·Z_H(ζ) 인지 확인

  Schwartz-Zippel:
    G(x) ≠ Q(x)·Z_H(x) 인데 G(ζ) = Q(ζ)·Z_H(ζ) 일 확률
    ≤ deg(G - Q·Z_H) / |Fr| ≤ (3n-3) / 2²⁵⁴ ≈ 0

  → 하나의 점 확인으로 모든 게이트를 검증! (Step 16)

이 관계를 KZG commit + opening으로 검증하면 = PLONK (Step 16)
```

---

### Part 8: Permutation 다항식 — σ를 다항식으로 인코딩

```
σ_A(ωⁱ) = position_tag(σ(A, i))
σ_B(ωⁱ) = position_tag(σ(B, i))
σ_C(ωⁱ) = position_tag(σ(C, i))

예시 (n=2, copy: (C,0) ↔ (A,1)):

  σ(A,0) = (A,0)  (self-loop)    → tag = ω⁰ = 1
  σ(A,1) = (C,0)                 → tag = K2·ω⁰ = 3
  σ(B,0) = (B,0)  (self-loop)    → tag = K1·ω⁰ = 2
  σ(B,1) = (B,1)  (self-loop)    → tag = K1·ω¹ = 2ω
  σ(C,0) = (A,1)                 → tag = ω¹ = ω
  σ(C,1) = (C,1)  (self-loop)    → tag = K2·ω¹ = 3ω

Lagrange 보간으로 σ_A(x), σ_B(x), σ_C(x) 구성:
  σ_A: σ_A(1) = 1, σ_A(ω) = 3  → degree 1 다항식
  σ_B: σ_B(1) = 2, σ_B(ω) = 2ω → degree 1 다항식
  σ_C: σ_C(1) = ω, σ_C(ω) = 3ω → degree 1 다항식

Identity permutation (copy constraint 없음):
  σ_A(ωⁱ) = ωⁱ         (자기 자신)
  σ_B(ωⁱ) = K1·ωⁱ      (자기 자신)
  σ_C(ωⁱ) = K2·ωⁱ      (자기 자신)
```

---

### Part 9: Grand Product Z(x) — 핵심 메커니즘

#### 아이디어

```
목표: permutation σ가 만족되는지 (copy constraint가 성립하는지) 증명

핵심 관찰:
  σ가 만족되면, "wire 값 + position tag" 의 multiset이
  "wire 값 + sigma tag" 의 multiset과 일치

  이 일치를 grand product로 확인:
    ∏ᵢ (value + β·identity_tag + γ) / (value + β·sigma_tag + γ) = 1

  ↑ 이것이 telescope하여 1이 되려면
    각 equivalence class 내의 값이 모두 같아야 함
```

#### Z(x) 계산

```
Z(ω⁰) = 1    (초기값)

Z(ω^(i+1)) = Z(ωⁱ) · num(i) / den(i)

  num(i) = (aᵢ + β·ωⁱ + γ)
         · (bᵢ + β·K1·ωⁱ + γ)
         · (cᵢ + β·K2·ωⁱ + γ)

  den(i) = (aᵢ + β·σ_A(ωⁱ) + γ)
         · (bᵢ + β·σ_B(ωⁱ) + γ)
         · (cᵢ + β·σ_C(ωⁱ) + γ)

여기서:
  β, γ — 랜덤 챌린지 (Fiat-Shamir로 결정, 이 단계에서는 직접 전달)
  aᵢ, bᵢ, cᵢ — wire 값
  ωⁱ, K1·ωⁱ, K2·ωⁱ — identity tag
  σ_A(ωⁱ), σ_B(ωⁱ), σ_C(ωⁱ) — sigma tag

최종 검증:
  Z(ω^(n-1)) · num(n-1) / den(n-1) = 1  ← "closes"
```

#### 왜 Z가 1로 돌아오면 permutation이 만족되는가

```
═══ 정리: Multiset Equality ↔ Grand Product = 1 ═══

두 multiset S = {s₁, ..., s_m} 과 T = {t₁, ..., t_m} 에 대해:

  S = T (as multisets)
  ⟺ ∏_{j=1}^{m} (γ + s_j) = ∏_{j=1}^{m} (γ + t_j)  for random γ

  (⟸ 증명은 Schwartz-Zippel로, 아래 참조)

  (⟹ 직관):
    S = T 이면 좌변과 우변은 같은 인수들의 곱 (순서만 다름)
    곱은 교환법칙 → 값이 같음 ✓
```

```
═══ PLONK에서의 적용 ═══

두 multiset을 정의:

  S = {(wⱼ, idⱼ) | j = 1, ..., 3n}
    = {(a₀, ω⁰), (a₁, ω¹), ..., (b₀, K1·ω⁰), ..., (c₀, K2·ω⁰), ...}

  T = {(wⱼ, σⱼ) | j = 1, ..., 3n}
    = {(a₀, σ_A(ω⁰)), ..., (b₀, σ_B(ω⁰)), ..., (c₀, σ_C(ω⁰)), ...}

여기서 wⱼ = wire 값, idⱼ = identity tag, σⱼ = sigma tag

쌍 (value, tag)를 하나의 원소로 합치기:
  s_j = w_j + β·id_j     (β로 선형 결합)

그러면:
  ∏ (γ + w_j + β·id_j) = ∏ (γ + w_j + β·σ_j)
```

```
═══ 왜 S = T ⟺ copy constraints 만족? ═══

핵심 관찰:
  σ는 {1, ..., 3n} 위의 permutation (전단사 함수)
  σ는 tag만 재배열, wire 값은 움직이지 않음

  S = T 가 되려면:
    각 j에 대해 (wⱼ, idⱼ) 쌍이 T에 존재해야 함
    T의 j번째 원소는 (wⱼ, σⱼ)

    σ가 j를 k로 보내면 (σ(j) = k):
      T에서 j번째 원소: (wⱼ, σ_tag(j)) = (wⱼ, id_tag(k))
      S에서 k번째 원소: (w_k, id_tag(k))

      이 두 쌍이 같으려면: wⱼ = w_k
      즉, σ(j) = k 이면 wⱼ = w_k!

    이것이 바로 copy constraint의 정의:
      같은 cycle에 있는 모든 wire가 같은 값을 가짐

  역으로, copy constraint가 만족되면:
    σ(j) = k 인 모든 j, k에 대해 wⱼ = w_k
    → S의 원소들과 T의 원소들이 정확히 일치 (σ가 bijection이므로)
    → S = T ✓

  따라서:
    grand product = 1 ⟺ S = T ⟺ copy constraints 만족 ∎
```

```
═══ Schwartz-Zippel 건전성 (Soundness) 분석 ═══

  정리 (Schwartz-Zippel Lemma):
    F(x₁, ..., x_k) 가 total degree d의 비영 다항식이면,
    x₁, ..., x_k를 유한체 F_q에서 균일 랜덤으로 선택할 때:
      Pr[F(x₁, ..., x_k) = 0] ≤ d / q

  적용:
    F(β, γ) = ∏ (wⱼ + β·idⱼ + γ) - ∏ (wⱼ + β·σⱼ + γ)

    S ≠ T 이면 F(β, γ)는 비영 다항식
    degree 분석:
      각 인수는 β, γ에 대해 degree 1
      3n개의 인수의 곱 → degree 3n (β와 γ 각각에 대해)
      total degree ≤ 3n (실제로는 약간 더 작지만 상한)

    따라서:
      Pr[F(β, γ) = 0] ≤ 3n / |Fr| ≤ 3·2²⁸ / 2²⁵⁴ ≈ 2⁻²²⁶

    이 확률은 무시할 수 있을 만큼 작음 (negligible)

  결론:
    copy constraint가 위반되면 (S ≠ T),
    랜덤 β, γ에 대해 grand product ≠ 1 일 확률 ≥ 1 - 2⁻²²⁶

β와 γ가 각각 왜 필요한가:
  β만 사용: (wⱼ + β·tagⱼ) → w와 tag를 분리
  γ만 사용: (wⱼ + γ) → tag 정보 없음, 순서만 확인
  둘 다 사용: (wⱼ + β·tagⱼ + γ) → 완전한 분리 + 영점 회피

  γ의 추가 역할:
    β·tagⱼ = 0 이 되는 경우 방지 (tag가 0인 경우 등)
    γ가 랜덤이므로 (wⱼ + 0 + γ) ≠ 0 with high probability
    → 분모가 0이 되어 나눗셈 실패하는 것을 방지
```

---

### Part 10: 수치 트레이스 — Identity Permutation

```
═══════════════════════════════════════════
n=2 (도메인 크기 2), ω = -1
게이트 0: add, a=3, b=4, c=7
게이트 1: dummy, a=0, b=0, c=0
Copy constraint: 없음
β=7, γ=11
═══════════════════════════════════════════

Position tags:
  (A,0): 1·ω⁰ = 1        (A,1): 1·ω¹ = -1
  (B,0): 2·ω⁰ = 2        (B,1): 2·ω¹ = -2
  (C,0): 3·ω⁰ = 3        (C,1): 3·ω¹ = -3

σ = identity → sigma tags = identity tags

Z(ω⁰) = 1

i=0:
  num = (3 + 7·1 + 11)(4 + 7·2 + 11)(7 + 7·3 + 11)
      = (3+7+11)(4+14+11)(7+21+11)
      = 21 · 29 · 39

  den = (3 + 7·1 + 11)(4 + 7·2 + 11)(7 + 7·3 + 11)
      = 21 · 29 · 39     ← identity와 동일!

  num/den = 1
  Z(ω¹) = 1 · 1 = 1

→ Z(x) = 1 (상수 다항식)

검증: Z(ω⁰)·num(0)/den(0) · Z(ω¹)·num(1)/den(1) ... = 1 ✓

Identity permutation에서는 num = den이므로
Z는 항상 1이다.
```

---

### Part 11: 수치 트레이스 — Copy Constraint 있는 경우

```
═══════════════════════════════════════════
n=2, ω = -1
게이트 0: add, a=3, b=4, c=7
게이트 1: mul, a=7, b=2, c=14
Copy: (C,0) ↔ (A,1)  → c₀=7 = a₁=7 ✓
β=7, γ=11
═══════════════════════════════════════════

σ mapping:
  σ(C,0) = (A,1)  → sigma_C(ω⁰) = tag(A,1) = ω¹ = -1
  σ(A,1) = (C,0)  → sigma_A(ω¹) = tag(C,0) = 3·ω⁰ = 3
  나머지: identity

Identity tags vs Sigma tags:
  Position  Identity_tag  Sigma_tag
  (A,0)     1             1         (self-loop)
  (A,1)     -1            3         ← changed!
  (B,0)     2             2         (self-loop)
  (B,1)     -2            -2        (self-loop)
  (C,0)     3             -1        ← changed!
  (C,1)     -3            -3        (self-loop)

Z(ω⁰) = 1

══ i=0 ══════════════════════════════════
  a₀=3, b₀=4, c₀=7

  num = (a₀ + β·ω⁰ + γ)(b₀ + β·K1·ω⁰ + γ)(c₀ + β·K2·ω⁰ + γ)
      = (3 + 7·1 + 11)(4 + 7·2 + 11)(7 + 7·3 + 11)
      = (3+7+11) · (4+14+11) · (7+21+11)
      = 21 · 29 · 39
      = 23,751

  den = (a₀ + β·σ_A(ω⁰) + γ)(b₀ + β·σ_B(ω⁰) + γ)(c₀ + β·σ_C(ω⁰) + γ)

  σ_A(ω⁰) = 1 (self), σ_B(ω⁰) = 2 (self), σ_C(ω⁰) = -1 (changed!)

  den = (3 + 7·1 + 11)(4 + 7·2 + 11)(7 + 7·(-1) + 11)
      = 21 · 29 · (7-7+11)
      = 21 · 29 · 11
      = 6,699

  Z(ω¹) = Z(ω⁰) · num/den = 1 · 23751/6699 = 23751/6699

  약분: 21·29 = 609 가 공통 인수
    23751 = 609 · 39
    6699  = 609 · 11

  Z(ω¹) = 39/11

══ i=1 (closing check) ══════════════════
  a₁=7, b₁=2, c₁=14

  num = (a₁ + β·ω¹ + γ)(b₁ + β·K1·ω¹ + γ)(c₁ + β·K2·ω¹ + γ)
      = (7 + 7·(-1) + 11)(2 + 7·(-2) + 11)(14 + 7·(-3) + 11)
      = (7-7+11) · (2-14+11) · (14-21+11)
      = 11 · (-1) · 4
      = -44

  den = (a₁ + β·σ_A(ω¹) + γ)(b₁ + β·σ_B(ω¹) + γ)(c₁ + β·σ_C(ω¹) + γ)

  σ_A(ω¹) = 3 (changed!), σ_B(ω¹) = -2 (self), σ_C(ω¹) = -3 (self)

  den = (7 + 7·3 + 11)(2 + 7·(-2) + 11)(14 + 7·(-3) + 11)
      = (7+21+11) · (2-14+11) · (14-21+11)
      = 39 · (-1) · 4
      = -156

  closing = Z(ω¹) · num(1)/den(1)
           = (39/11) · (-44)/(-156)
           = (39/11) · (44/156)

  약분: 44/156 = 4·11 / 4·39 = 11/39

  closing = (39/11) · (11/39)
           = (39·11) / (11·39)
           = 1 ✓ 🎉

══ 검증: 왜 정확히 1이 되는가 ══════════
  전체 곱을 전개하면:

  ∏_{i=0}^{1} num(i)/den(i) = (21·29·39)/(21·29·11) · (11·(-1)·4)/(39·(-1)·4)

  공통 인수를 확인:
    분자 전체: 21·29·39 · 11·(-1)·4
    분모 전체: 21·29·11 · 39·(-1)·4

    ↑ 분자와 분모가 같은 인수들의 곱! (순서만 다름)

  이것이 permutation의 본질:
    σ가 (C,0)↔(A,1) 교환 → 39와 11, 11과 39 가 교차
    → 전체 곱에서 상쇄 → 1

  copy constraint 만족 시:
    c₀ = 7 = a₁ → 같은 값이 identity/sigma tag에 대응
    → num의 인수 집합 = den의 인수 집합 (multiset으로 동일)
    → 곱이 1 ✓
```

---

### Part 12: Grand Product 실패 — Copy Constraint 위반

```
═══════════════════════════════════════════
같은 구조, but:
  Gate 0: add, a=3, b=4, c=7
  Gate 1: mul, a=99, b=4, c=396    ← a₁=99 ≠ c₀=7
  Copy: (C,0) ↔ (A,1)  → c₀=7, a₁=99 ← 불일치!
═══════════════════════════════════════════

σ는 동일 (구조만 보고 만듦):
  σ(C,0) = (A,1), σ(A,1) = (C,0)

하지만 wire 값이 다르므로:
  num(0)의 c₀ 항: (7 + 7·3 + 11) = 39
  den(0)의 c₀ 항: (7 + 7·(-1) + 11) = 11

  num(1)의 a₁ 항: (99 + 7·(-1) + 11) = 103  ← 99가 아닌 값
  den(1)의 a₁ 항: (99 + 7·3 + 11) = 131     ← σ tag = 3

  최종 곱: Z(ω¹) · num(1)/den(1) ≠ 1

  값이 다르면 identity tag와 sigma tag의 교환이
  multiset을 보존하지 않음 → telescope 실패 → ≠ 1

→ verify_grand_product_closes() returns false ✓
```

---

### Part 13: PlonkConstraintSystem — 코드 워크스루

```rust
pub struct PlonkConstraintSystem {
    pub values: Vec<Fr>,           // 변수 풀
    gates: Vec<GateInstance>,       // 게이트 목록
    copy_constraints: Vec<(WirePosition, WirePosition)>,
    pub num_public_inputs: usize,
}
```

```
사용 흐름:

  1. cs = PlonkConstraintSystem::new()

  2. x = cs.alloc_variable(Fr::from_u64(3))     → index 0
     y = cs.alloc_variable(Fr::from_u64(4))     → index 1
     z = cs.alloc_variable(Fr::from_u64(7))     → index 2

  3. cs.add_gate(PlonkGate::addition_gate(), x, y, z)
     → gates[0] = { gate: add, a: 0, b: 1, c: 2 }

  4. cs.copy_constraint((C,0), (A,1))
     → copy_constraints에 추가

  5. n = cs.pad_to_power_of_two()
     → dummy 변수(0) 할당, dummy 게이트 추가, n 반환

  6. domain = Domain::new(n)

  7. selectors = cs.selector_polynomials(&domain)
     → q_L(ωⁱ), q_R(ωⁱ), ... Lagrange 보간

  8. (a, b, c) = cs.wire_polynomials(&domain)
     → a(ωⁱ), b(ωⁱ), c(ωⁱ) Lagrange 보간

메모리 레이아웃:

  values: [x=3, y=4, z=7, ..., dummy=0]

  gates:
  ┌──────────────────────────────────────┐
  │ gate_0: {add, a=0, b=1, c=2}        │
  │ gate_1: {mul, a=2, b=3, c=4}        │
  │ gate_2: {dummy, a=5, b=5, c=5}      │  (padding)
  │ gate_3: {dummy, a=5, b=5, c=5}      │  (padding)
  └──────────────────────────────────────┘

  wire_value(WirePosition{A, 0}) = values[gates[0].a] = values[0] = 3
  wire_value(WirePosition{B, 1}) = values[gates[1].b] = values[3] = ...
```

---

### Part 14: 패딩 전략

```
PLONK은 도메인 크기가 2의 거듭제곱이어야 함 (FFT 호환)

pad_to_power_of_two():
  1. 현재 게이트 수 n에서 next_power_of_two(n) 계산
  2. 부족한 만큼 dummy 게이트 추가
  3. dummy 변수(값=0) 할당하여 사용

  3 gates → 4 (1개 dummy)
  5 gates → 8 (3개 dummy)
  4 gates → 4 (패딩 불필요)
  1 gate  → 2 (최소 도메인 = 2)

Dummy 게이트가 안전한 이유:
  q_L = q_R = q_O = q_M = q_C = 0
  → 0·a + 0·b + 0·c + 0·a·b + 0 = 0
  → 항상 만족 (어떤 wire 값이든)

  Dummy wire의 값 = 0
  → permutation에서도 무해 (self-loop)
```

---

### Part 15: Permutation 다항식 계산 — 코드 워크스루

```rust
pub fn compute_permutation_polynomials(
    cs: &PlonkConstraintSystem,
    domain: &Domain,
) -> (Polynomial, Polynomial, Polynomial) {
    let n = domain.n;
    let sigma = compute_sigma(n, cs.copy_constraints());

    // σ_A(ωⁱ) = tag of σ(A, i)
    let sigma_a_points: Vec<(Fr, Fr)> = (0..n)
        .map(|i| {
            let from_idx = pos_to_idx(WirePosition{column: Column::A, row: i}, n);
            let to = idx_to_pos(sigma[from_idx], n);
            (domain.elements[i], position_tag(to.column, to.row, domain))
        })
        .collect();

    // Lagrange 보간
    let sigma_a_poly = Polynomial::lagrange_interpolate(&sigma_a_points);
    // ... sigma_b, sigma_c 도 동일
}
```

```
내부 인덱스 변환:

  pos_to_idx: (column, row) → 선형 인덱스
    A: 0..n
    B: n..2n
    C: 2n..3n

  예 (n=4):
    (A,0)=0, (A,1)=1, (A,2)=2, (A,3)=3
    (B,0)=4, (B,1)=5, (B,2)=6, (B,3)=7
    (C,0)=8, (C,1)=9, (C,2)=10, (C,3)=11

  idx_to_pos: 역변환
    0~3 → Column::A
    4~7 → Column::B
    8~11 → Column::C
```

---

### Part 16: Grand Product Z(x) — 코드 워크스루

```rust
pub fn compute_grand_product(
    cs: &PlonkConstraintSystem,
    domain: &Domain,
    sigma_a: &Polynomial, sigma_b: &Polynomial, sigma_c: &Polynomial,
    beta: Fr, gamma: Fr,
) -> Polynomial {
    let n = domain.n;
    let k1 = Fr::from_u64(K1);
    let k2 = Fr::from_u64(K2);

    let mut z_values = Vec::with_capacity(n);
    z_values.push(Fr::ONE); // Z(ω⁰) = 1

    for i in 0..n - 1 {
        let omega_i = domain.elements[i];
        let a_i = cs.wire_value(WirePosition{column: Column::A, row: i});
        // ...

        let num = (a_i + beta * omega_i + gamma)
                * (b_i + beta * k1 * omega_i + gamma)
                * (c_i + beta * k2 * omega_i + gamma);

        let den = (a_i + beta * sigma_a.eval(omega_i) + gamma)
                * (b_i + beta * sigma_b.eval(omega_i) + gamma)
                * (c_i + beta * sigma_c.eval(omega_i) + gamma);

        z_values.push(z_values[i] * num * den.inv().unwrap());
    }

    // Lagrange 보간으로 Z(x) 다항식 구성
    Polynomial::lagrange_interpolate(...)
}
```

```
Z(x) 계산 흐름:

  z[0] = 1
  ┌─ i=0 ─────────────────────────────────┐
  │ num = (a₀+β·ω⁰+γ)(b₀+β·K1·ω⁰+γ)    │
  │       (c₀+β·K2·ω⁰+γ)                 │
  │ den = (a₀+β·σ_A(ω⁰)+γ)(b₀+β·σ_B...) │
  │ z[1] = z[0] · num/den                 │
  └────────────────────────────────────────┘
  ┌─ i=1 ─────────────────────────────────┐
  │ z[2] = z[1] · num/den                 │
  └────────────────────────────────────────┘
  ...
  z[n-1] 계산까지

  closing check:
  z[n-1] · num(n-1)/den(n-1) = 1?
```

---

### Part 17: PlonkCircuit trait — 회로 인터페이스

```rust
pub trait PlonkCircuit {
    fn synthesize(&self, cs: &mut PlonkConstraintSystem);
}

// x³ + x + 5 = y 회로 구현 예시
struct CubicCircuit { x: u64 }

impl PlonkCircuit for CubicCircuit {
    fn synthesize(&self, cs: &mut PlonkConstraintSystem) {
        let x_val = Fr::from_u64(self.x);
        let v1_val = x_val * x_val;
        let v2_val = v1_val * x_val;
        let y_val = v2_val + x_val + Fr::from_u64(5);

        let x  = cs.alloc_variable(x_val);
        let v1 = cs.alloc_variable(v1_val);
        let v2 = cs.alloc_variable(v2_val);
        let y  = cs.alloc_variable(y_val);

        // Gate 0: x * x = v1
        cs.add_gate(PlonkGate::multiplication_gate(), x, x, v1);
        // Gate 1: v1 * x = v2
        cs.add_gate(PlonkGate::multiplication_gate(), v1, x, v2);
        // Gate 2: v2 + x + 5 = y (custom gate)
        cs.add_gate(PlonkGate { q_l: ONE, q_r: ONE, q_o: -ONE,
                                q_m: ZERO, q_c: Fr::from_u64(5) },
                    v2, x, y);

        // Copy constraints
        cs.copy_constraint((A,0), (B,0));  // x
        cs.copy_constraint((B,0), (B,1));  // x
        cs.copy_constraint((B,1), (B,2));  // x
        cs.copy_constraint((C,0), (A,1));  // v1
        cs.copy_constraint((C,1), (A,2));  // v2
    }
}
```

```
R1CS 버전과의 비교:

  R1CS (Step 10):
    x * x = t1          (제약 1)
    t1 * x = t2         (제약 2)
    t2 + x + 5 = y      (제약 3, 변환 필요)
    → 3 제약 + 행렬 A, B, C

  PLONKish:
    Gate 0: mul x*x=v1   (selector로)
    Gate 1: mul v1*x=v2  (selector로)
    Gate 2: v2+x+5=y     (custom selector로!)
    + 5 copy constraints
    → 3 게이트, 더 직관적!

  PLONKish의 장점:
    Gate 2를 하나의 게이트로 표현 (R1CS에서는 보조변수 필요할 수 있음)
    회로 구조가 시각적으로 명확
```

---

### Part 17.5: Custom Gate 설계 패턴

#### 기본 원리

```
PLONKish의 강점: selector 값만 바꾸면 새로운 게이트 유형을 정의할 수 있음

범용 게이트: q_L·a + q_R·b + q_O·c + q_M·a·b + q_C = 0

설계 절차:
  1. 원하는 제약 조건을 수식으로 표현
  2. q_L, q_R, q_O, q_M, q_C 값을 결정
  3. 필요하면 copy constraint로 wire 연결 보장
  4. 부족하면 보조변수 + 다중 게이트로 분해
```

#### 패턴 1: 선형 조합 (Linear Combination)

```
제약: α·a + β·b = c

PLONKish: α·a + β·b + (-1)·c = 0
  q_L = α, q_R = β, q_O = -1, q_M = 0, q_C = 0

예시: 3a + 5b = c
  q_L = 3, q_R = 5, q_O = -1

검증: 3·2 + 5·4 = 6 + 20 = 26 = c ✓
```

#### 패턴 2: 상수 곱셈 (Scalar Multiplication)

```
제약: k·a = c   (a를 상수 k배)

PLONKish: k·a + (-1)·c = 0
  q_L = k, q_R = 0, q_O = -1, q_M = 0, q_C = 0

예시: 7·a = c, a=3 → c=21
  7·3 + (-1)·21 = 21 - 21 = 0 ✓
```

#### 패턴 3: 조건부 선택 (Conditional Select / MUX)

```
제약: c = s·a + (1-s)·b   (s ∈ {0,1})

이것은 2개 게이트로 분해:

  Gate i: boolean_gate on s
    q_L=1, q_M=-1, wire_a=s, wire_b=s  (copy constraint!)
    → s(1-s)=0, 즉 s ∈ {0,1}

  Gate i+1: c = s·a + (1-s)·b
    전개: c = s·a + b - s·b = s·(a-b) + b
    재배열: s·(a-b) + b - c = 0
    q_M·s·t + q_R·b + q_O·c = 0  (t = a-b, 보조변수)

  실제로는 3개 게이트:
    Gate 0: boolean(s)
    Gate 1: a - b = t   (뺄셈)
    Gate 2: s·t + b = c  → q_M=1, q_R=1, q_O=-1

    Copy constraints로 s, b, t 연결
```

#### 패턴 4: 범위 제약 (Range Check, 간소화)

```
제약: 0 ≤ a < 4  (2비트)

방법: a = 2·b₁ + b₀ (b₀, b₁ ∈ {0,1})

  Gate 0: boolean(b₀)     → b₀ ∈ {0,1}
  Gate 1: boolean(b₁)     → b₁ ∈ {0,1}
  Gate 2: b₁ · 2 = t      → q_L=2, q_O=-1 (상수 곱)
  Gate 3: t + b₀ = a      → q_L=1, q_R=1, q_O=-1 (덧셈)

  총 4 게이트 + copy constraints

  일반화: k비트 범위 → k boolean gates + k-1 결합 gates
  → O(k) 게이트   (Plookup은 이것을 O(1)로 줄임 → Step 15)
```

#### 패턴 5: 비교 (a = b 확인)

```
두 wire의 값이 같은지 확인하는 두 가지 방법:

  방법 1: Copy constraint (비용 0!)
    cs.copy_constraint((A, i), (B, j))
    → permutation argument가 자동으로 강제
    → 추가 게이트 불필요!

  방법 2: 뺄셈 게이트 (다른 게이트에서 결과를 사용해야 할 때)
    a - b = 0  →  q_L=1, q_R=-1, q_C=0
    wire_a=a, wire_b=b, wire_c=dummy

  원칙:
    순수하게 "같은 값"만 강제 → copy constraint (무료)
    뺄셈 결과를 다른 계산에 사용 → 게이트 (비용 1)
```

#### 실전 예시: 5차 다항식 f(x) = x⁵ + 3x² + 7

```
단계별 분해:

  보조변수:
    t1 = x²      (1 mul gate)
    t2 = t1 · t1  (= x⁴, 1 mul gate)
    t3 = t2 · x   (= x⁵, 1 mul gate)
    t4 = 3·t1     (1 scalar mul gate)
    y = t3 + t4 + 7  (1 custom gate)

  총 5 게이트:
    Gate 0: x · x = t1          (mul)
    Gate 1: t1 · t1 = t2        (mul)
    Gate 2: t2 · x = t3         (mul)
    Gate 3: 3·t1 = t4           (q_L=3, q_O=-1)
    Gate 4: t3 + t4 + 7 = y     (q_L=1, q_R=1, q_O=-1, q_C=7)

  Copy constraints (7개):
    (B,0) ↔ (A,0)   ← x in gate 0
    (A,1) ↔ (C,0)   ← t1
    (B,1) ↔ (C,0)   ← t1 (gate 1에서 양쪽 다)
    (A,2) ↔ (C,1)   ← t2
    (B,2) ↔ (A,0)   ← x again
    (A,3) ↔ (C,0)   ← t1 for 3·t1
    (A,4) ↔ (C,2)   ← t3
    (B,4) ↔ (C,3)   ← t4

  R1CS로는 같은 회로에 5개 제약이 필요 — 비슷하지만
  PLONKish는 custom gate로 Gate 4를 하나로 합침

  참고: PLONKish 확장 (TurboPLONK)에서는
  wider gate (wire 4-5개)를 써서 더 줄일 수 있음
```

---

### Part 18: 계산 복잡도

```
┌──────────────────────────┬──────────────────────────────┐
│ 연산                      │ 복잡도                        │
├──────────────────────────┼──────────────────────────────┤
│ alloc_variable            │ O(1)                         │
│ add_gate                  │ O(1)                         │
│ copy_constraint           │ O(1)                         │
│ is_satisfied              │ O(n)                         │
│ pad_to_power_of_two       │ O(n)                         │
│ selector_polynomials      │ O(n²) — Lagrange 보간        │
│ wire_polynomials          │ O(n²) — Lagrange 보간        │
│ compute_sigma             │ O(n · α(n)) — Union-Find     │
│ compute_permutation_polys │ O(n²) — Lagrange 보간        │
│ compute_grand_product     │ O(n²) — n번 eval (각 O(n))   │
│ verify_grand_product      │ O(n) — 3번 eval              │
└──────────────────────────┴──────────────────────────────┘

참고: Lagrange 보간이 O(n²)인 이유는 교육용 구현이기 때문.
프로덕션에서는 FFT 기반 O(n log n) 보간 사용.
```

---

### Part 19: PLONK 전체 파이프라인에서의 위치

```
PLONK 전체 흐름:

  ┌──────────────────────────────────────────────────────┐
  │                                                        │
  │  Step 14 (이 단계):                                    │
  │  ┌────────────────────┐  ┌───────────────────────┐    │
  │  │  Arithmetization   │  │  Permutation Argument │    │
  │  │                    │  │                       │    │
  │  │  PlonkGate         │  │  σ 다항식              │    │
  │  │  PlonkConstraintSys│  │  Grand Product Z(x)   │    │
  │  │  Selector polys    │  │                       │    │
  │  │  Wire polys        │  │                       │    │
  │  └────────────────────┘  └───────────────────────┘    │
  │            ↓                        ↓                  │
  │  Step 15: Plookup (lookup argument)                    │
  │            ↓                                           │
  │  Step 16: PLONK Prover/Verifier                        │
  │  ┌─────────────────────────────────────────────────┐  │
  │  │  Round 1-5: commit, challenge, open             │  │
  │  │  KZG commit/open/verify 사용                     │  │
  │  │  Fiat-Shamir transcript                         │  │
  │  └─────────────────────────────────────────────────┘  │
  │            ↓                                           │
  │  Step 17: FFLONK 최적화                                │
  │                                                        │
  └──────────────────────────────────────────────────────┘
```

---

### Part 20: 테스트 요약

```
21개 테스트 구성:

  Gate (4개):
    addition_gate_satisfied    — 3+4=7 ✓, 3+4=8 ✗
    multiplication_gate_satisfied — 3·4=12 ✓
    constant_gate_satisfied    — a=42 ✓
    boolean_gate_satisfied     — a∈{0,1} ✓, a=2 ✗

  CS (4개):
    alloc_variable             — 인덱스 순차, 값 저장
    add_gate_check_satisfied   — 2게이트 회로
    wrong_witness_fails        — 잘못된 값 → 불만족
    pad_to_power_of_two        — 3→4, dummy 안전

  Poly (3개):
    selector_polynomials_at_domain  — q_L(ωⁱ) 검증
    wire_polynomials_at_domain      — a(ωⁱ) 검증
    gate_equation_via_polynomials   — G(ωⁱ) = 0 검증

  Perm (4개):
    identity_permutation           — σ=id
    copy_constraint_permutation    — 2-cycle
    multiple_copy_constraints_chain — 3-cycle
    copy_constraint_violation      — 값 불일치

  Z(x) (3개):
    grand_product_identity         — Z=1
    grand_product_with_copy        — Z closes
    grand_product_violation        — Z 안 닫힘

  E2E (2개):
    plonk_circuit_trait   — a+b=c 전체 흐름
    cubic_circuit_plonk   — x³+x+5=y 전체 흐름

  Domain (1개):
    domain_roots_of_unity — ω^n=1, ω^(n/2)=-1
```

---

### Part 21: 전체 의존성 그래프

```
Step 14: PLONKish Arithmetization 의존성

  ┌───────────────────────────────────────────────────────┐
  │                                                        │
  │  Step 1: Fp (소수체)                                   │
  │    │                                                   │
  │    ├──→ Fr (스칼라체)                                  │
  │    │      │                                            │
  │    │      ├──→ Step 11: Polynomial                     │
  │    │      │      │                                     │
  │    │      │      └──→ ★ Step 14: PLONKish             │
  │    │      │             │                              │
  │    │      └─────────────┘                              │
  │    │                                                   │
  │    │  (Step 14는 G1, G2, pairing 불필요!)              │
  │    │  (KZG도 이 단계에서는 불필요)                      │
  │    │                                                   │
  │    │  Step 16에서 KZG + 이 단계의 결과를 결합:          │
  │    │    PLONKish CS → selector/wire/σ polys → KZG      │
  │    │                                                   │
  │    ├──→ Step 5-7: G1, G2, pairing (Step 16에서 사용)   │
  │    ├──→ Step 13: KZG (Step 16에서 사용)                │
  │    │                                                   │
  └───────────────────────────────────────────────────────┘

PLONKish의 직접 의존성:
  1. Fr — 스칼라체 (wire 값, selector 값, position tag)
  2. Polynomial — Lagrange 보간, eval

PLONKish가 사용하지 않는 것:
  - G1, G2, pairing (Step 16에서 사용)
  - KZG (Step 16에서 commit/open/verify)
  - R1CS (대체됨)
  - QAP (대체됨)
  - hash, merkle, signature
```

---

> [!summary] Step 14 요약
> ```
> PLONKish = 범용 게이트 + copy constraint + grand product
>
> 핵심 방정식:
>   게이트: q_L·a + q_R·b + q_O·c + q_M·a·b + q_C = 0
>   Grand product: Z(ω⁰)=1, Z(ω^(i+1)) = Z(ωⁱ)·num/den
>   검증: Z closes (telescope → 1)
>
> 코드: plonk/ (mod.rs + arithmetization.rs + permutation.rs)
>   약 770줄, 21 테스트
>
> 의존성: Fr, Polynomial
> 독립성: G1/G2/pairing/KZG 불필요 (Step 16에서 결합)
>
> 다음: Step 15 Plookup → Step 16 PLONK Prover/Verifier
> ```
