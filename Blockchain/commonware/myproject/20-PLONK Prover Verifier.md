## Step 16: PLONK Prover/Verifier — 5라운드로 범용 증명을 만들다

### 핵심 질문: 빌딩 블록이 있는데 왜 Prover가 더 필요한가?

```
Steps 13-15에서 구축한 것:

  Step 13 (KZG): 다항식을 하나의 점으로 commit, 특정 값 증명
  Step 14 (PLONKish): 범용 게이트 + permutation으로 회로 표현
  Step 15 (Plookup): 테이블 멤버십을 grand product로 증명

각각은 독립적인 도구:
  KZG는 "이 다항식의 이 점에서의 값이 맞다"를 증명
  PLONKish는 "이 게이트 방정식이 성립한다"를 인코딩
  Permutation은 "와이어가 올바르게 연결되어 있다"를 인코딩

하지만 이들을 어떻게 결합하는가?

  증명자: "나는 올바른 witness를 알고 있어.
           모든 게이트가 만족되고, 와이어가 올바르게 연결되어 있어."

  검증자: "witness를 보지 않고 어떻게 믿어?"

  → 이 질문에 대한 답 = PLONK Prover/Verifier

핵심 통찰:
  모든 제약을 다항식 등식으로 변환하고,
  "다항식이 도메인 위에서 0"임을 quotient polynomial로 증명하고,
  KZG로 commit하고, Fiat-Shamir로 비대화형 변환

  빌딩 블록들의 "접착제" 역할
```

> [!important] PLONK Prover의 역할
> ```
> KZG + PLONKish + Permutation = 재료
> PLONK Prover = 이 재료들을 결합하는 레시피
>
> 비유:
>   KZG = 용접기 (두 금속을 붙임)
>   PLONKish = 설계도 (어떤 구조인지)
>   Permutation = 배선도 (어디가 연결되는지)
>   PLONK Prover = 조립 공정 (실제로 만드는 과정)
>
> Groth16과의 차이:
>   Groth16: 회로별 setup, 증명 3개 원소 (최소)
>   PLONK: universal setup, 증명 9 G1 + 6 Fr (조금 크지만 범용적)
> ```

---

### Part 1: 전체 파이프라인 — Setup → Prove → Verify

```
┌──────────────────────────────────────────────────────────────────┐
│                      PLONK 전체 흐름                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐       ┌──────────┐       ┌─────────┐               │
│  │  Setup   │       │  Prove   │       │  Verify │               │
│  │          │       │          │       │         │               │
│  │ CS + RNG │       │ SRS      │       │ SRS     │               │
│  │    ↓     │──SRS─→│ CS       │─Proof→│ VK      │──bool         │
│  │ SRS + VK │       │ Domain   │       │ Proof   │               │
│  │          │──VK──→│          │       │         │               │
│  └─────────┘       └──────────┘       └─────────┘               │
│                                                                   │
│  Setup (1회):                                                     │
│    - SRS 생성: τ의 거듭제곱 [τⁱ]₁, [τʲ]₂                         │
│    - Selector 다항식 commit: [q_L]₁, [q_R]₁, [q_O]₁, [q_M]₁, [q_C]₁ │
│    - Permutation 다항식 commit: [σ_A]₁, [σ_B]₁, [σ_C]₁           │
│    → VerifyingKey (회로 구조에만 의존, witness에 독립)              │
│                                                                   │
│  Prove (5라운드):                                                 │
│    Round 1: Wire 다항식 commit → [a]₁, [b]₁, [c]₁                │
│    Round 2: Permutation grand product Z(x) commit → [Z]₁         │
│    Round 3: Quotient 다항식 T(x) 분할 commit                      │
│    Round 4: 챌린지 ζ에서 6개 평가값 계산                            │
│    Round 5: Linearization + batched opening proof                 │
│    → PlonkProof (9 G1 + 6 Fr)                                    │
│                                                                   │
│  Verify:                                                          │
│    Fiat-Shamir 재현 → 챌린지 동일성 확인                            │
│    Linearization commitment 구성 (스칼라 곱만!)                     │
│    Batched pairing check 1회                                      │
│    → bool                                                         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### Part 2: Fiat-Shamir Heuristic — 대화형을 비대화형으로

#### 왜 비대화형이어야 하는가?

```
대화형 프로토콜 (interactive):
  증명자                          검증자
    │                               │
    │── Round 1: [a]₁,[b]₁,[c]₁ ──→│
    │                               │── β, γ 랜덤 선택
    │←──── β, γ ────────────────────│
    │                               │
    │── Round 2: [Z]₁ ────────────→│
    │                               │── α 랜덤 선택
    │←──── α ───────────────────────│
    │                               │
    │── Round 3: [T_lo]₁,... ─────→│
    │                               │── ζ 랜덤 선택
    │←──── ζ ───────────────────────│
    │                               │
    │── Round 4: ā, b̄, c̄, ... ───→│
    │                               │── ν 랜덤 선택
    │←──── ν ───────────────────────│
    │                               │
    │── Round 5: W_ζ, W_{ζω} ────→│
    │                               │── 검증

  문제:
    - 5번 왕복 필요 → 지연 시간
    - 검증자가 온라인이어야 함
    - 블록체인에서는 검증자가 "컨트랙트" → 대화 불가!

  해결: Fiat-Shamir heuristic
    "검증자의 랜덤 챌린지를 해시 함수로 대체"
```

#### Fiat-Shamir의 핵심 아이디어

```
아이디어:
  검증자의 역할 = "지금까지의 transcript를 보고 랜덤 값 선택"

  이것을 해시 함수로 시뮬레이션:
    challenge = Hash(지금까지의 모든 메시지)

  증명자가 직접 해시를 계산해서 챌린지를 도출
  → 왕복 없이 증명 생성 가능!

안전성의 직관:
  Hash가 랜덤 오라클처럼 행동하면:
    - 증명자가 챌린지를 "예측"할 수 없음
    - 증명자가 transcript를 조작하면 챌린지가 완전히 바뀜
    → 대화형 프로토콜과 동일한 건전성

구현 — Transcript 자료구조:
  ┌──────────────────────────────────────────────────────┐
  │  struct Transcript {                                  │
  │      state: Fr     // 해시 체인의 현재 상태              │
  │  }                                                    │
  │                                                       │
  │  append_commitment(C):                                │
  │      // G1 점 → affine x좌표 → Fr로 변환               │
  │      state = Poseidon(state, x_coord(C))              │
  │                                                       │
  │  append_scalar(s):                                    │
  │      state = Poseidon(state, s)                       │
  │                                                       │
  │  challenge() → Fr:                                    │
  │      state = Poseidon(state, 1)    // domain separation│
  │      return state                                     │
  │                                                       │
  │  핵심: 같은 순서로 같은 데이터 → 같은 챌린지            │
  │       Prover와 Verifier가 동일한 transcript 유지!      │
  └──────────────────────────────────────────────────────┘
```

#### G1 → Fr 변환이 필요한 이유

```
문제: Transcript는 Fr 원소를 해시하는데, commitment는 G1 점

해결: G1 → affine x좌표 → Fr

  G1 점은 Jacobian 좌표 (X, Y, Z):
    affine x = X / Z²
    affine y = X / Z³

  변환 과정:
    1. Z의 역원 계산: z_inv = Z⁻¹
    2. affine x 계산: affine_x = X · z_inv²
    3. Fp → Fr 변환: Fr::from_raw(affine_x.to_repr())
       (Fp와 Fr은 같은 [u64; 4] 표현, mod r로 축소)

  특수 경우: identity (무한원점) → Fr::ZERO

왜 x좌표만?
  y좌표까지 쓰면 더 안전하지만, x좌표만으로도 충분:
  같은 x에 대해 y는 ±y 두 개뿐 → 2-to-1 매핑
  교육용 구현에서는 이 충돌이 문제되지 않음

왜 Poseidon?
  - ZK-friendly 해시 (체 연산에 최적화)
  - SHA-256 대비: 체 위에서 ~10배 적은 제약
  - 이미 Step 07에서 구현했으므로 재사용
```

---

### Part 3: 핵심 통찰 — "도메인에서 0" = "Z_H로 나누어떨어짐"

```
PLONK 증명의 핵심 등식:

  모든 i ∈ {0, 1, ..., n-1}에서:
    Gate(ωⁱ) = 0     (게이트 방정식 만족)
    Perm(ωⁱ) = 0     (permutation 제약 만족)

  이것을 다항식으로 보면:
    Gate(x)가 H = {1, ω, ..., ω^(n-1)} 에서 모두 0
    ⟺ Z_H(x) = ∏(x - ωⁱ) = x^n - 1 이 Gate(x)를 나눔

  즉:
    Gate(x) + α·Perm₁(x) + α²·Perm₂(x) = T(x) · Z_H(x)

  여기서 T(x) = quotient polynomial (몫 다항식)

왜 α로 결합하는가?
  여러 제약을 하나의 다항식 등식으로 합치기 위함.

  α가 랜덤이면:
    Gate(x) + α·Perm₁(x) + α²·Perm₂(x) ≡ 0 on H
    ⟺ Gate(x) ≡ 0 on H AND Perm₁(x) ≡ 0 on H AND Perm₂(x) ≡ 0 on H

  이유 (Schwartz-Zippel):
    만약 Gate(ωⁱ) ≠ 0 이지만 전체 합이 0이 되려면
    α가 정확히 -Gate(ωⁱ)/Perm₁(ωⁱ)를 만족해야 함
    → 랜덤 α에서 이 확률 ≤ deg/|Fr| ≈ 0

왜 Z_H(x) = x^n - 1 인가?
  H = {ω⁰, ω¹, ..., ω^(n-1)} 은 n차 단위근의 집합
  ω^n = 1 이므로:
    (ω^k)^n = (ω^n)^k = 1^k = 1
  모든 h ∈ H에 대해 h^n = 1
  → Z_H(x) = x^n - 1 은 H의 모든 원소를 근으로 가짐
```

---

### Part 4: Round 1 — Wire Commitments

```
입력: 만족된 제약 시스템 CS (게이트 + witness 값)

Step 1: CS에서 다항식 추출
  a_poly, b_poly, c_poly = cs.wire_polynomials(domain)

  이것은 Lagrange 보간:
    a(ωⁱ) = cs.gates[i].a값
    b(ωⁱ) = cs.gates[i].b값
    c(ωⁱ) = cs.gates[i].c값

  예시 (x³+x+5=y, x=3):
    ┌────────┬───────┬───────┬───────┐
    │ Gate i │  a    │  b    │  c    │
    ├────────┼───────┼───────┼───────┤
    │   0    │ x=3   │ x=3   │ v1=9  │  (x*x=v1)
    │   1    │ v1=9  │ x=3   │ v2=27 │  (v1*x=v2)
    │   2    │ v2=27 │ x=3   │ y=35  │  (v2+x+5=y)
    │   3    │  0    │  0    │  0    │  (dummy pad)
    └────────┴───────┴───────┴───────┘

    → a(ω⁰)=3, a(ω¹)=9, a(ω²)=27, a(ω³)=0  등

Step 2: KZG commit
  [a]₁ = Σ aᵢ · [τⁱ]₁    (a(x)의 계수 × SRS powers)
  [b]₁ = ...
  [c]₁ = ...

Step 3: Transcript에 추가
  transcript.append([a]₁)
  transcript.append([b]₁)
  transcript.append([c]₁)

왜 Round 1이 먼저인가?
  Wire 값은 witness에 의존 → 챌린지보다 먼저 commit해야
  "먼저 답을 제출하고, 그 다음에 시험 문제를 받는다"
  → 증명자가 문제를 보고 답을 조작할 수 없음
```

---

### Part 5: Round 2 — Permutation Grand Product

```
챌린지 도출:
  β = transcript.challenge()    // 코셋 분리용
  γ = transcript.challenge()    // 와이어 값 보호용

Permutation argument의 목적:
  같은 변수가 여러 wire에 나타날 때, 값이 실제로 같음을 증명

  예시 (x³+x+5=y):
    x는 (A,0), (B,0), (B,1), (B,2) 에 나타남
    → 이 4곳의 값이 모두 같아야 함

    v1은 (C,0), (A,1) 에 나타남
    → 이 2곳의 값이 같아야 함

Grand Product Z(x) 의 구성:
  Z(ω⁰) = 1

  Z(ω^(i+1)) = Z(ωⁱ) ·
    (a(ωⁱ)+β·ωⁱ+γ)(b(ωⁱ)+β·K₁·ωⁱ+γ)(c(ωⁱ)+β·K₂·ωⁱ+γ)
    ───────────────────────────────────────────────────────
    (a(ωⁱ)+β·σ_A(ωⁱ)+γ)(b(ωⁱ)+β·σ_B(ωⁱ)+γ)(c(ωⁱ)+β·σ_C(ωⁱ)+γ)

  분자: identity permutation (각 셀의 고유 ID)
    ωⁱ, K₁·ωⁱ, K₂·ωⁱ 는 column A, B, C의 i번째 셀 ID
    K₁=2, K₂=3 → 3개 코셋이 서로소

  분모: actual permutation (σ가 지정한 연결)
    σ_A(ωⁱ), σ_B(ωⁱ), σ_C(ωⁱ)는 해당 셀이 연결된 대상의 ID

  Z가 닫히는 조건: Z(ω^n) = Z(ω⁰) = 1
    ⟺ 분자 전체곱 = 분모 전체곱
    ⟺ identity permutation과 actual permutation이 같은 멀티셋
    ⟺ copy constraints가 만족됨

KZG commit:
  [Z]₁ = commit(srs, z_poly)
  transcript.append([Z]₁)
```

---

### Part 6: Round 3 — Quotient Polynomial (가장 핵심)

#### 세 가지 제약의 결합

```
챌린지: α = transcript.challenge()

제약 1: Gate constraint G(x)
  G(x) = q_L(x)·a(x) + q_R(x)·b(x) + q_O(x)·c(x)
        + q_M(x)·a(x)·b(x) + q_C(x)

  모든 ωⁱ에서 G(ωⁱ) = 0 이어야 함 (게이트 방정식 만족)

제약 2: Permutation constraint P₁(x)
  P₁(x) = Z(x)·(a+βx+γ)(b+βK₁x+γ)(c+βK₂x+γ)
         - Z(ωx)·(a+βσ_A+γ)(b+βσ_B+γ)(c+βσ_C+γ)

  모든 ωⁱ에서 P₁(ωⁱ) = 0
  (grand product의 재귀 관계가 매 행에서 성립)

제약 3: Permutation boundary P₂(x)
  P₂(x) = (Z(x) - 1) · L₁(x)

  L₁(x) = 첫 번째 Lagrange 기저: L₁(ω⁰)=1, L₁(ωⁱ)=0 (i≠0)

  이것은 "Z(ω⁰) = 1" 을 인코딩:
    P₂(ω⁰) = (Z(1)-1)·1 = 0  (Z(1)=1이면)
    P₂(ωⁱ) = (Z(ωⁱ)-1)·0 = 0  (i≠0이면 L₁=0)

결합:
  Numerator(x) = G(x) + α·P₁(x) + α²·P₂(x)

  Numerator(ωⁱ) = 0 for all i
  → Z_H(x) | Numerator(x)
  → T(x) = Numerator(x) / Z_H(x)
```

#### Z(ωx) — Shifted 다항식의 구성

```
문제: Z(ωx)는 Z(x)를 "한 칸 이동"한 다항식
  Z(x) = c₀ + c₁x + c₂x² + ...
  Z(ωx) = c₀ + c₁(ωx) + c₂(ωx)² + ...
         = c₀ + (c₁ω)x + (c₂ω²)x² + ...

구성 방법:
  Z_shifted(x)의 i번째 계수 = Z(x)의 i번째 계수 × ωⁱ

  z_shifted.coeffs[i] = z_poly.coeffs[i] * ω^i

  이렇게 하면:
    Z_shifted(ωⁱ) = Σⱼ (cⱼ·ωʲ)·(ωⁱ)ʲ = Σⱼ cⱼ·ω^(j(i+1)) = Z(ω^(i+1))

  즉 Z_shifted(ωⁱ) = Z(ω^(i+1)) — 정확히 "한 행 뒤의 Z"
```

#### T(x) 분할 — 왜 3조각인가?

```
차수 분석:
  G(x): deg = deg(q_M)·deg(a·b) ≈ (n-1)·2(n-1) = 3(n-1)...
    실제로 q_M, a, b 각각 deg ≤ n-1 → 곱 deg ≤ 3(n-1)

  P₁(x): Z(x) · 3개 선형 곱 → deg ≈ (n-1) + 3(n-1) = 4(n-1)...
    더 정확히: Z는 deg n, 3개 선형인자 각각 deg n → deg ≈ 4n
    a+βx+γ 는 deg n-1 + deg 1 → max(n-1, 1) 그러나 실제 곱은 더 복잡

  결론: Numerator(x)의 최대 차수 ≈ 3n + 작은 항

  T(x) = Numerator(x) / Z_H(x):
    deg T = deg Numerator - deg Z_H ≈ 3n - n ≈ 2n

  KZG commit은 deg ≤ n-1 크기의 SRS로 제한 (일반적)
  → T를 3조각으로 분할:

    T(x) = t_lo(x) + x^n · t_mid(x) + x^{2n} · t_hi(x)

    t_lo = T의 계수 [0..n)      → deg ≤ n-1
    t_mid = T의 계수 [n..2n)    → deg ≤ n-1
    t_hi = T의 계수 [2n..)      → deg ≤ n-1 (보통 더 작음)

  각 조각은 별도로 KZG commit:
    [t_lo]₁, [t_mid]₁, [t_hi]₁

  검증 시 ζ에서 재결합:
    T(ζ) = t_lo(ζ) + ζ^n · t_mid(ζ) + ζ^{2n} · t_hi(ζ)

SRS 크기:
  plonk_setup에서 max_degree = 3n + 5
  이것은 T(x)의 최대 차수 + 여유분
```

---

### Part 7: Round 4 — 평가값 계산

```
챌린지: ζ = transcript.challenge()

ζ는 도메인 H의 원소가 아닌 "랜덤 점"
  (확률적으로 ζ ∉ H — |H|/|Fr| ≈ n/2²⁵⁴ ≈ 0)

6개 평가값 계산:
  ā = a(ζ)           — left wire의 ζ에서의 값
  b̄ = b(ζ)           — right wire
  c̄ = c(ζ)           — output wire
  σ̄_A = σ_A(ζ)       — permutation A
  σ̄_B = σ_B(ζ)       — permutation B
  z̄_ω = Z(ζω)        — Z의 shifted evaluation

왜 이 6개인가?
  Round 5에서 linearization을 위해 필요한 스칼라값들.
  "다항식 곱"을 "스칼라 × 다항식"으로 변환하기 위함.

왜 σ_C(ζ)는 없는가?
  linearization에서 σ_C(x)는 다항식 형태로 남김
  (나머지 5개만 스칼라로 평가)
  → σ_C의 commitment은 verifier가 이미 VK에 가지고 있음

왜 Z(ζ) 대신 Z(ζω)인가?
  P₁(x)에서 Z(ωx)가 등장 → ζ에서 평가하면 Z(ζω) 필요
  Z(ζ)는 linearization에서 Z(x)를 다항식으로 남기므로 스칼라 불필요

transcript에 추가:
  ā, b̄, c̄, σ̄_A, σ̄_B, z̄_ω 모두 append
```

---

### Part 8: Round 5 — Linearization의 핵심 아이디어

#### 왜 Linearization이 필요한가?

```
문제: Verifier가 T(ζ) · Z_H(ζ) = G(ζ) + α·P₁(ζ) + α²·P₂(ζ) 를 확인하고 싶음

순진한 방법:
  Prover가 모든 다항식의 ζ에서의 값을 제공:
    a(ζ), b(ζ), c(ζ), q_L(ζ), q_R(ζ), ..., σ_A(ζ), σ_B(ζ), σ_C(ζ), Z(ζ), Z(ζω)

  Verifier가 직접 계산:
    G(ζ) = q_L(ζ)·a(ζ) + ...
    P₁(ζ) = Z(ζ)·(...) - Z(ζω)·(...)

  그리고 각 평가값이 맞는지 KZG로 확인

  문제: 평가값 하나당 opening proof 하나 → 너무 많은 페어링!
    q_L(ζ) → 1 proof, q_R(ζ) → 1 proof, ...
    총 ~15개 opening proof → 15번 페어링 → 비효율적

해결: Linearization
  "다항식 곱"을 "스칼라 × 다항식"으로 분해

  핵심 관찰:
    q_L(x) · a(x) 에서 a(ζ) = ā 를 스칼라로 대체하면
    → q_L(x) · ā  (이것은 q_L(x)의 스칼라 곱)

    이것의 ζ에서의 값: q_L(ζ) · ā  (원래와 동일!)
    하지만 commitment은: ā · [q_L]₁  (VK에 이미 있음!)

  → q_L(ζ)의 opening proof가 불필요해짐!

linearization polynomial r(x):
  Gate 부분:
    r_gate(x) = ā·q_L(x) + b̄·q_R(x) + c̄·q_O(x) + (ā·b̄)·q_M(x) + q_C(x)

  Permutation 부분:
    P₁에서 Z(x)는 다항식으로, Z(ωx)→z̄_ω는 스칼라로:
    σ_C(x)도 다항식으로 남기고, σ_A(ζ), σ_B(ζ)는 스칼라로

  이렇게 하면:
    r(x)의 commitment = [r]₁ 은 VK와 proof의 commitment들의 선형 결합
    → Verifier가 scalar multiplication만으로 계산 가능!
    → 페어링 없이 [r]₁ 구성 가능!
```

#### Linearization 수식 — 완전한 유도

```
Gate constraint at ζ:
  G(ζ) = q_L(ζ)·ā + q_R(ζ)·b̄ + q_O(ζ)·c̄ + q_M(ζ)·ā·b̄ + q_C(ζ)

  Linearize: 상수 ā, b̄, c̄ 를 계수로, q_*(x) 를 다항식으로:
    r_gate(x) = ā·q_L(x) + b̄·q_R(x) + c̄·q_O(x) + (ā·b̄)·q_M(x) + q_C(x)
    r_gate(ζ) = G(ζ)  ✓

Permutation constraint 1 at ζ:
  P₁(ζ) = Z(ζ)·(ā+βζ+γ)(b̄+βK₁ζ+γ)(c̄+βK₂ζ+γ)
         - Z(ζω)·(ā+βσ̄_A+γ)(b̄+βσ̄_B+γ)(c̄+βσ_C(ζ)+γ)

  여기서:
    perm_num = (ā+βζ+γ)(b̄+βK₁ζ+γ)(c̄+βK₂ζ+γ)   ← 스칼라 (ā,b̄,c̄,ζ 모두 알려짐)
    perm_den_partial = (ā+βσ̄_A+γ)(b̄+βσ̄_B+γ)      ← 스칼라

  Linearize: Z(x)와 σ_C(x)를 다항식으로 남김:
    r_perm(x) = perm_num · Z(x)
              - z̄_ω · perm_den_partial · β · σ_C(x)

  r_perm(ζ)를 확인하면:
    = perm_num · Z(ζ)
      - z̄_ω · perm_den_partial · β · σ_C(ζ)

  원래 P₁(ζ)와 비교:
    P₁(ζ) = Z(ζ) · perm_num
           - z̄_ω · perm_den_partial · (c̄ + β·σ_C(ζ) + γ)

  차이를 보자:
    r_perm(ζ) = perm_num · Z(ζ) - z̄_ω · perm_den_partial · β · σ_C(ζ)

    P₁(ζ) = perm_num · Z(ζ)
           - z̄_ω · perm_den_partial · [c̄ + β·σ_C(ζ) + γ]

    차이 = z̄_ω · perm_den_partial · (c̄ + γ)

  즉: r_perm(ζ) = P₁(ζ) + z̄_ω · perm_den_partial · (c̄ + γ)  ← 잔여!

Permutation boundary at ζ:
  P₂(ζ) = (Z(ζ) - 1) · L₁(ζ)

  Linearize: Z(x)를 다항식으로, L₁(ζ)는 스칼라:
    L₁(ζ) = (ζⁿ - 1) / (n · (ζ - 1))   ← Verifier가 직접 계산

    r_boundary(x) = L₁(ζ) · Z(x)
    r_boundary(ζ) = L₁(ζ) · Z(ζ)

  원래 P₂(ζ) = (Z(ζ)-1) · L₁(ζ) = L₁(ζ)·Z(ζ) - L₁(ζ)
  차이: r_boundary(ζ) - P₂(ζ) = L₁(ζ)  ← 잔여!
```

#### r_constant — 상수 잔여의 보정

```
위 분석에서:
  r_gate(ζ) = G(ζ)                                    (정확)
  r_perm(ζ) = P₁(ζ) + z̄_ω · perm_den_partial · (c̄+γ)  (잔여 있음)
  r_boundary(ζ) = P₂(ζ) + L₁(ζ)                        (잔여 있음)

전체 linearization (보정 전):
  r_pre(x) = r_gate(x) + α·r_perm(x) + α²·r_boundary(x) - Z_H(ζ)·T_combined(x)

  r_pre(ζ) = G(ζ) + α·[P₁(ζ) + z̄_ω·den·(c̄+γ)] + α²·[P₂(ζ) + L₁(ζ)]
           - Z_H(ζ)·T(ζ)

  = [G(ζ) + α·P₁(ζ) + α²·P₂(ζ) - Z_H(ζ)·T(ζ)]    ← 이것은 0!
    + α · z̄_ω · perm_den_partial · (c̄+γ)             ← 잔여 1
    + α² · L₁(ζ)                                      ← 잔여 2

  따라서:
    r_pre(ζ) = 0 + r_constant

    r_constant = α · z̄_ω · perm_den_partial · (c̄ + γ) + α² · L₁(ζ)

보정:
  r(x) = r_pre(x) - r_constant

  r(ζ) = r_pre(ζ) - r_constant = r_constant - r_constant = 0  ✓

핵심: r(ζ) = 0 이 보장됨!
  이것이 검증에서 매우 중요:
  Verifier는 r(ζ) = 0 임을 알기에,
  batched evaluation에서 r의 평가값을 포함할 필요 없음

Verifier 쪽 보정:
  Verifier도 [r]₁에서 r_constant·G₁을 빼야 함:
    [r]₁ = ... (기존 선형 결합) ... - r_constant · G₁

  G₁ = [τ⁰]₁ = generator
  → commit(constant(r_constant)) = r_constant · G₁
```

---

### Part 9: Round 5 — Batched Opening

#### 왜 Batch가 필요한가?

```
상황:
  Round 4에서 6개 평가값을 주장함:
    a(ζ)=ā, b(ζ)=b̄, c(ζ)=c̄, σ_A(ζ)=σ̄_A, σ_B(ζ)=σ̄_B

  그리고 1개 다른 점에서의 평가값:
    Z(ζω) = z̄_ω

  + linearization: r(ζ) = 0 (implicit)

순진한 방법:
  각 평가값에 대해 KZG opening proof 1개
  → 7개 proof → 7번 페어링 → 비효율적

Batch 아이디어:
  "여러 다항식의 같은 점에서의 evaluation을 하나의 proof로"

  KZG opening: f(z) = y ⟺ (x-z) | (f(x) - y)

  여러 다항식 f₁, f₂, ..., fₖ 에 대해:
    νⁱ로 가중합을 만들면:
    F(x) = f₁(x) + ν·f₂(x) + ... + νᵏ·fₖ(x)
    F(z) = f₁(z) + ν·f₂(z) + ... + νᵏ·fₖ(z) = Y

    (x-z) | (F(x) - Y)
    ⟹ W(x) = (F(x) - Y) / (x - z) 하나로 충분!

  Schwartz-Zippel:
    만약 fᵢ(z) ≠ yᵢ 이면 F(z) ≠ Y (ν가 랜덤이므로)
    → batch opening도 건전
```

#### 두 개의 개구점: ζ와 ζω

```
PLONK에는 두 개의 서로 다른 평가점이 있음:
  점 ζ: a, b, c, σ_A, σ_B, r  (6개 다항식)
  점 ζω: Z                     (1개 다항식)

이것을 어떻게 합치는가?

Step 1: ζ에서의 batch opening
  ν = transcript.challenge()

  W_ζ(x) = [r(x) + ν·(a(x)-ā) + ν²·(b(x)-b̄) + ν³·(c(x)-c̄)
           + ν⁴·(σ_A(x)-σ̄_A) + ν⁵·(σ_B(x)-σ̄_B)] / (x - ζ)

  r(ζ)=0 이므로 r(x) - 0 = r(x), 상수항 빼기 불필요

Step 2: ζω에서의 opening
  W_{ζω}(x) = [Z(x) - z̄_ω] / (x - ζω)

Step 3: KZG commit
  [W_ζ]₁ = commit(srs, W_ζ(x))
  [W_{ζω}]₁ = commit(srs, W_{ζω}(x))

증명 출력:
  PlonkProof = {
    [a]₁, [b]₁, [c]₁,           // Round 1
    [Z]₁,                        // Round 2
    [t_lo]₁, [t_mid]₁, [t_hi]₁, // Round 3
    ā, b̄, c̄, σ̄_A, σ̄_B, z̄_ω,  // Round 4
    W_ζ, W_{ζω}                  // Round 5
  }
```

---

### Part 10: Verification — Batched Pairing Check 유도

#### KZG Opening에서 Pairing Check로

```
KZG 기본 검증:
  f(z) = y 의 opening proof π에 대해:
    e(π, [τ-z]₂) = e([f]₁ - y·G₁, G₂)

  이것을 변형:
    e(π, [τ]₂ - z·G₂) = e([f]₁ - y·G₁, G₂)
    e(π, [τ]₂) - e(z·π, G₂) = e([f]₁ - y·G₁, G₂)
    e(π, [τ]₂) = e(z·π + [f]₁ - y·G₁, G₂)
```

#### 두 개의 KZG 검증을 하나로

```
PLONK에는 두 개의 KZG 검증이 있음:

  (1) W_ζ 검증:
    e(W_ζ, [τ]₂) = e(ζ·W_ζ + F_ζ - E_ζ·G₁, G₂)

    여기서 F_ζ = [r]₁ + ν·[a]₁ + ν²·[b]₁ + ν³·[c]₁ + ν⁴·[σ_A]₁ + ν⁵·[σ_B]₁
          E_ζ = 0 + ν·ā + ν²·b̄ + ν³·c̄ + ν⁴·σ̄_A + ν⁵·σ̄_B
               (r(ζ)=0 이므로 r의 기여 없음)

  (2) W_{ζω} 검증:
    e(W_{ζω}, [τ]₂) = e(ζω·W_{ζω} + [Z]₁ - z̄_ω·G₁, G₂)

랜덤 u로 결합:
  u = transcript.challenge()

  (1) + u·(2):
    e(W_ζ + u·W_{ζω}, [τ]₂)
      = e(ζ·W_ζ + u·ζω·W_{ζω} + F_ζ + u·[Z]₁ - (E_ζ + u·z̄_ω)·G₁, G₂)

  정리:
    LHS: e(W_ζ + u·W_{ζω}, [τ]₂)

    RHS: e(ζ·W_ζ + u·ζω·W_{ζω} + F - [E]₁, G₂)

    여기서:
      F = F_ζ + u·[Z]₁
        = [r]₁ + ν·[a]₁ + ν²·[b]₁ + ν³·[c]₁ + ν⁴·[σ_A]₁ + ν⁵·[σ_B]₁ + u·[Z]₁

      E = E_ζ + u·z̄_ω
        = ν·ā + ν²·b̄ + ν³·c̄ + ν⁴·σ̄_A + ν⁵·σ̄_B + u·z̄_ω

  하나의 페어링 등식! (2회 페어링으로 검증)

왜 u·[Z]₁가 중요한가?
  결합에서 (2)번 등식의 [Z]₁ 항이 u 가중치로 합류
  u·z̄_ω도 마찬가지로 E에 합류

  단순 회로 (Z=1, copy constraint 없음):
    [Z]₁ = G₁, z̄_ω = 1
    u·G₁ - u·1·G₁ = 0 → 상쇄됨 → 빠져도 통과!

  복잡한 회로 (Z≠1, copy constraint 있음):
    u·[Z]₁ ≠ u·z̄_ω·G₁ → 빠지면 실패!

  구현 디버깅에서 발견된 실제 버그:
    처음에 u·[Z]₁ 과 u·z̄_ω를 빠뜨림
    → 단순 회로는 통과, cubic 회로는 실패
    → 원인: Z=1일 때만 우연히 상쇄
```

---

### Part 11: Verifier 알고리즘 — 전체 흐름

```
입력: SRS, VK, Proof, public_inputs (현재 비어있음)

┌─────────────────────────────────────────────────────────────┐
│  Step 1: Fiat-Shamir 재현                                    │
│    Prover와 동일한 순서로 transcript 구성                      │
│    → β, γ, α, ζ, ν, u 도출                                  │
│                                                              │
│  Step 2: 보조값 계산                                          │
│    ζⁿ (반복 제곱), ζ²ⁿ = (ζⁿ)²                               │
│    Z_H(ζ) = ζⁿ - 1                                          │
│    L₁(ζ) = Z_H(ζ) / (n · (ζ - 1))                           │
│                                                              │
│  Step 3: [r]₁ 구성 (스칼라 곱만!)                              │
│    Gate:  ā·[q_L]₁ + b̄·[q_R]₁ + c̄·[q_O]₁ + (ā·b̄)·[q_M]₁ + [q_C]₁  │
│    Perm:  (α·perm_num + α²·L₁(ζ))·[Z]₁                      │
│           - (α·perm_den_partial·z̄_ω·β)·[σ_C]₁               │
│    Quot:  -Z_H(ζ)·[t_lo]₁ - Z_H(ζ)·ζⁿ·[t_mid]₁             │
│           - Z_H(ζ)·ζ²ⁿ·[t_hi]₁                              │
│    보정:  -r_constant · G₁                                    │
│                                                              │
│  Step 4: F 구성                                               │
│    F = [r]₁ + ν·[a]₁ + ν²·[b]₁ + ν³·[c]₁                    │
│      + ν⁴·[σ_A]₁ + ν⁵·[σ_B]₁ + u·[Z]₁                      │
│                                                              │
│  Step 5: E 계산                                               │
│    E = ν·ā + ν²·b̄ + ν³·c̄ + ν⁴·σ̄_A + ν⁵·σ̄_B + u·z̄_ω       │
│    [E]₁ = E · G₁                                             │
│                                                              │
│  Step 6: 페어링 체크                                           │
│    LHS = e(W_ζ + u·W_{ζω}, [τ]₂)                             │
│    RHS = e(ζ·W_ζ + u·ζω·W_{ζω} + F - [E]₁, G₂)             │
│    return LHS == RHS                                         │
└─────────────────────────────────────────────────────────────┘

Verifier 복잡도:
  - Transcript: O(1) 해시 호출 (~15회)
  - 스칼라 곱: O(1) scalar_mul (~15회)
  - 페어링: 정확히 2회
  - 총: O(1) — 회로 크기에 무관!
```

---

### Part 12: 수치 트레이스 — a + b = c 회로

```
간단한 회로: 3 + 4 = 7

── 회로 구성 ──────────────────────────────────────

  Gate 0: a + b = c  (q_L=1, q_R=1, q_O=-1, q_M=0, q_C=0)
    a₀=3, b₀=4, c₀=7

  Pad to n=2 (최소 2의 거듭제곱):
  Gate 1: dummy (q_L=0, q_R=0, q_O=0, q_M=0, q_C=0)
    a₁=0, b₁=0, c₁=0

── Domain H = {1, ω} ─────────────────────────────

  n = 2, ω = n차 원시 단위근 (ω² = 1, ω ≠ 1)
  H = {ω⁰=1, ω¹=ω}

── Wire 다항식 ────────────────────────────────────

  a(ω⁰)=3, a(ω¹)=0 → a(x) = Lagrange 보간
  b(ω⁰)=4, b(ω¹)=0 → b(x) = ...
  c(ω⁰)=7, c(ω¹)=0 → c(x) = ...

── KZG Commit ─────────────────────────────────────

  [a]₁ = a₀·[τ⁰]₁ + a₁·[τ¹]₁ + ...  (계수 기반)
  [b]₁, [c]₁ 동일

── Gate Check (ζ에서) ─────────────────────────────

  G(ζ) = q_L(ζ)·a(ζ) + q_R(ζ)·b(ζ) + q_O(ζ)·c(ζ) + q_M(ζ)·a(ζ)·b(ζ) + q_C(ζ)

  H 위 (ω⁰):
    1·3 + 1·4 + (-1)·7 + 0·3·4 + 0 = 3+4-7 = 0  ✓

  H 위 (ω¹):
    0·0 + 0·0 + 0·0 + 0·0·0 + 0 = 0  ✓ (dummy)

  → G(x)는 H 위에서 0 → Z_H(x) = x²-1 로 나누어떨어짐

── Permutation ─────────────────────────────────────

  Copy constraints: 없음 (각 변수 1회만 사용)
  → σ = identity → Z(x) = 1 (상수)

── Quotient T(x) ──────────────────────────────────

  T(x) = [G(x) + α·P₁(x) + α²·P₂(x)] / Z_H(x)

  Z=1이면 P₁=0 (분자=분모), P₂=(1-1)·L₁=0
  → T(x) = G(x) / Z_H(x)

── 검증 ───────────────────────────────────────────

  Verifier:
    1. transcript에서 β,γ,α,ζ,ν,u 재현
    2. [r]₁ 계산 (VK의 commitments에 스칼라 곱)
    3. F, E 구성
    4. e(W_ζ+u·W_{ζω}, [τ]₂) = e(ζ·W_ζ+u·ζω·W_{ζω}+F-[E]₁, G₂)
    → true  ✓
```

---

### Part 13: 증명 크기 분석

```
PlonkProof 구성:
  ┌──────────────────────┬──────────┬───────────┐
  │ 항목                   │ 유형     │ 크기       │
  ├──────────────────────┼──────────┼───────────┤
  │ Round 1: [a]₁,[b]₁,[c]₁ │ G1 × 3  │ 192 bytes │
  │ Round 2: [Z]₁          │ G1 × 1  │  64 bytes │
  │ Round 3: [t]₁ × 3      │ G1 × 3  │ 192 bytes │
  │ Round 4: 6 evaluations │ Fr × 6  │ 192 bytes │
  │ Round 5: W_ζ, W_{ζω}   │ G1 × 2  │ 128 bytes │
  ├──────────────────────┼──────────┼───────────┤
  │ 합계                   │ 9G1+6Fr │ 768 bytes │
  └──────────────────────┴──────────┴───────────┘

  G1 점: 64 bytes (x, y 각 32 bytes, 압축 시 33 bytes)
  Fr 스칼라: 32 bytes

비교:
  ┌──────────────┬────────────────┬─────────────┐
  │              │ 증명 크기        │ 검증 페어링  │
  ├──────────────┼────────────────┼─────────────┤
  │ Groth16      │ 2G1+1G2 = 192B │ 3회         │
  │ PLONK        │ 9G1+6Fr = 768B │ 2회         │
  │ PLONK (압축)  │ ~500B          │ 2회         │
  └──────────────┴────────────────┴─────────────┘

  PLONK 증명이 ~4배 크지만:
    - Universal setup (회로 변경 시 재사용)
    - 검증 페어링 1회 적음
    - Custom gate 확장 용이

SRS 크기 (Prover가 보관):
  G1 powers: 3n + 6개 → O(n)
  G2 powers: 2개 (G₂, [τ]₂)
  VK: 8 commitments + 2 스칼라 → 고정 크기
```

---

### Part 14: 건전성 분석

```
PLONK의 건전성: "거짓 증명이 검증을 통과할 확률은?"

1. Schwartz-Zippel (α 결합):
   Gate, Perm1, Perm2 중 하나라도 H에서 0이 아니면
   α·Perm1 + α²·Perm2가 Gate을 우연히 상쇄할 확률:
   ≤ 2(n-1) / |Fr| ≈ n/2²⁵³

2. Quotient polynomial (T 분할 검증):
   T가 올바르게 분할되었는지는 ζ에서의 재결합으로 확인
   잘못된 분할이 ζ에서 우연히 맞을 확률:
   ≤ 3n / |Fr|

3. KZG opening soundness:
   DL 가정 하에 다항식이 아닌 함수에 대해
   올바른 opening proof를 생성할 확률 ≈ 0

4. Fiat-Shamir (Random Oracle):
   Hash가 랜덤 오라클이면 증명자가 챌린지를 예측할 확률 ≈ 0
   실제로는 hash collision 저항성에 의존

총 건전성 오류:
  ε ≤ O(n) / |Fr| + negl(λ)

  n = 2²⁰ (백만 게이트)이면:
    ε ≈ 2²⁰ / 2²⁵⁴ = 2⁻²³⁴ ≈ 0

  사실상 무시 가능!

5. Knowledge extraction:
   증명자가 실제로 witness를 "알고 있음"을 보장
   Algebraic Group Model (AGM) 가정 하에:
   증명에서 G1 원소를 추출하면 → SRS의 선형 결합
   → 다항식 계수를 복원 → witness 추출
```

---

### Part 15: Blinding과 Zero-Knowledge

```
이 구현에서 생략된 것: Zero-Knowledge 속성

ZK가 없으면?
  검증자가 증명에서 witness 정보를 추출할 수 있음
  예: a(ζ) = ā 가 직접 노출됨
  여러 ζ에 대한 ā 값으로 a(x) 복원 가능

Blinding으로 ZK 달성:
  Wire 다항식에 랜덤 항 추가:
    a_blinded(x) = a(x) + (r₁ + r₂·x) · Z_H(x)

  Z_H(ωⁱ) = 0 이므로:
    a_blinded(ωⁱ) = a(ωⁱ)  ← 도메인 위에서 동일!
    a_blinded(ζ) = a(ζ) + (r₁ + r₂·ζ) · Z_H(ζ)  ← ζ에서는 다름

  랜덤 r₁, r₂가 a(ζ)를 마스킹 → 검증자가 정보 추출 불가

  Z(x)에도 동일한 blinding 적용

이 구현이 blinding을 생략한 이유:
  교육용 — SNARK의 ARK (Argument of Knowledge)에 집중
  S (Succinct), N (Non-interactive), AR (Argument) 모두 구현됨
  ZK 속성만 생략 → 핵심 프로토콜 이해에 집중
```

---

### Part 16: PLONK vs Groth16 — 설계 철학 비교

```
┌───────────────────┬──────────────────────┬──────────────────────┐
│                   │ Groth16              │ PLONK                │
├───────────────────┼──────────────────────┼──────────────────────┤
│ 제약 시스템        │ R1CS                 │ PLONKish             │
│ 다항식 기반        │ QAP                  │ Lagrange + KZG       │
│ Setup             │ Per-circuit          │ Universal            │
│ Setup 파라미터     │ τ, α, β, γ, δ        │ τ 하나               │
│ 증명 크기          │ 192 bytes (최소)      │ ~768 bytes           │
│ 검증 페어링        │ 3회                  │ 2회                  │
│ 프로토콜 라운드     │ 1 (non-interactive)  │ 5 (Fiat-Shamir)      │
│ Custom gate       │ 불가                 │ Selector 추가         │
│ Lookup            │ 불가                 │ Plookup 통합 가능     │
│ 증명 생성 비용      │ O(n) MSM            │ O(n log n) FFT+MSM   │
│ 코드 복잡도        │ 비교적 단순           │ 복잡 (5라운드)        │
├───────────────────┼──────────────────────┼──────────────────────┤
│ 최적 사용처        │ 고정 회로,            │ 범용, 변경 빈번,      │
│                   │ 증명 크기 중요         │ 유연성 중요           │
└───────────────────┴──────────────────────┴──────────────────────┘

핵심 트레이드오프:
  Groth16: 작은 증명 ↔ 회로별 setup
  PLONK: 큰 증명 ↔ universal setup + 확장성

실무에서의 선택:
  zkSync Era: PLONK 기반 (회로 업그레이드 빈번)
  Zcash Sapling: Groth16 (고정 회로, 증명 크기 중요)
  Polygon Hermez: Groth16 + PLONK 하이브리드
```

---

### Part 17: 구현에서 발견된 미묘한 버그와 교훈

#### 버그 1: r_constant 누락

```
증상: 모든 테스트에서 "batch opening remainder at zeta must be zero" 실패

원인:
  W_ζ(x) = batch_poly(x) / (x - ζ) 에서 나머지가 0이 아님
  → batch_poly(ζ) ≠ 0
  → r(ζ) ≠ 0 이었음!

디버깅 과정:
  1. batch_poly(ζ) = r(ζ) + ν·0 + ν²·0 + ... = r(ζ)
  2. r(ζ) ≠ 0 → linearization이 정확하지 않았음
  3. 수식 추적:
     r_perm(ζ)에서 σ_C(x)를 다항식으로 남길 때
     (c̄ + β·σ_C(ζ) + γ) 중 β·σ_C(ζ)만 다항식에서 나오고
     (c̄ + γ)는 상수로 남아 잔여 발생

수정:
  r_constant = α · z̄_ω · perm_den_partial · (c̄ + γ) + α² · L₁(ζ)
  r(x) = r_pre(x) - r_constant

교훈:
  Linearization에서 다항식을 스칼라로 "부분 대체"하면
  반드시 상수 잔여가 발생한다.
  원래 등식의 양변을 기호적으로 전개하여 잔여를 정확히 계산해야 한다.
```

#### 버그 2: u·[Z]₁ 누락

```
증상: 단순 회로(AddCircuit, MulCircuit)는 통과,
      copy constraint가 있는 회로(CubicCircuit, LargerCircuit)만 실패

원인:
  Verifier의 F에서 u·[Z]₁ 누락
  E에서 u·z̄_ω 누락

왜 단순 회로는 통과했나?
  Copy constraint 없음 → Z(x) = 1 (상수)
  [Z]₁ = 1 · G₁ = G₁
  z̄_ω = Z(ζω) = 1

  누락된 항: u·G₁ - u·1·G₁ = 0 → 상쇄!

  하지만 Z ≠ 1이면:
  u·[Z]₁ ≠ u·z̄_ω·G₁ → 상쇄 안 됨 → 실패

수정:
  F에 + u·[Z]₁ 추가
  E에 + u·z̄_ω 추가

교훈:
  두 개의 KZG 개구점을 u로 결합할 때,
  두 번째 개구점(ζω)의 다항식(Z)과 평가값(z̄_ω)이
  반드시 F와 E에 각각 포함되어야 한다.

  단순 테스트만으로는 이런 버그를 놓칠 수 있다.
  항상 non-trivial case (Z≠1)를 테스트해야 한다.
```

---

### Part 18: 코드 구조와 의존성

```
── 파일 구조 ────────────────────────────────────────

crates/primitives/src/plonk/
  ├── mod.rs              -- Domain, 코셋 K1/K2, PlonkCircuit trait
  ├── arithmetization.rs  -- PlonkConstraintSystem, 게이트, selector
  ├── permutation.rs      -- σ 순열, grand product Z(x)
  ├── lookup.rs           -- Plookup (Step 15)
  └── prover.rs           -- ★ PLONK Prover/Verifier (이 단계)

── prover.rs 내부 구조 ─────────────────────────────

  g1_to_fr()        — G1 → Fr 변환 (transcript 용)
  Transcript         — Fiat-Shamir (Poseidon 기반)
  VerifyingKey        — 8 commitments + n + ω
  PlonkSetupParams    — SRS + VK
  PlonkProof          — 9 G1 + 6 Fr
  plonk_setup()       — SRS 생성 + selector/permutation commit
  prove()             — 5-round Fiat-Shamir → PlonkProof
  verify()            — Fiat-Shamir 재현 + batched pairing check

── 의존성 그래프 ────────────────────────────────────

  Fr, Fp (체 연산)
    │
    ├──→ G1, G2, pairing (타원곡선)
    │      │
    │      └──→ KZG (setup, commit, open, verify)
    │             │
    │             └──→ ★ PLONK Prover
    │
    ├──→ Polynomial (QAP 모듈에서)
    │      │
    │      ├──→ PLONKish Arithmetization (selector/wire 다항식)
    │      │      │
    │      │      └──→ ★ PLONK Prover
    │      │
    │      └──→ Permutation (σ 다항식, grand product)
    │             │
    │             └──→ ★ PLONK Prover
    │
    └──→ Poseidon Hash
           │
           └──→ ★ PLONK Prover (Fiat-Shamir Transcript)

── 핵심 API 사용 ────────────────────────────────────

  kzg::setup(max_g1, max_g2, rng)   — SRS 생성
  kzg::commit(srs, poly)            — 다항식 commit
  cs.selector_polynomials(domain)    — q_L, q_R, q_O, q_M, q_C
  cs.wire_polynomials(domain)        — a(x), b(x), c(x)
  compute_permutation_polynomials()  — σ_A, σ_B, σ_C
  compute_grand_product()            — Z(x)
  Polynomial::lagrange_interpolate() — 점 → 다항식
  poly.eval(point)                   — 다항식 평가
  poly.div_rem(divisor)              — 나눗셈 + 나머지
  poseidon_hash(a, b)                — Fiat-Shamir 해시
  G1::scalar_mul(scalar)             — 스칼라 곱
  pairing(g1, g2)                    — 페어링 연산
```

---

### Part 19: 테스트 요약

```
11개 테스트 구성:

  Transcript (2개):
    transcript_deterministic    — 같은 입력 → 같은 챌린지
    transcript_different_inputs — 다른 입력 → 다른 챌린지

  E2E Prove/Verify (5개):
    prove_verify_addition       — a+b=c (1 게이트, copy constraint 없음)
    prove_verify_multiplication — a*b=c (1 게이트, copy constraint 없음)
    prove_verify_cubic          — x³+x+5=y (3 게이트, 5 copy constraints)
    prove_verify_boolean        — a·(1-a)=0 (a=0, a=1 둘 다)
    prove_verify_larger_circuit — 8 게이트 체인 (7 copy constraints)

  Soundness (4개):
    tampered_commitment_fails    — [a]₁ 변조 → verify 실패
    tampered_evaluation_fails    — ā 변조 → verify 실패
    tampered_opening_fails       — W_ζ 변조 → verify 실패
    cross_circuit_verification   — 회로 A 증명을 회로 B VK로 → 실패

테스트 설계 의도:
  - E2E: 단순→복잡 순서로 정확성 확인
    단순 회로 (Z=1): linearization 기본 동작 검증
    복잡 회로 (Z≠1): u·[Z]₁ 포함 여부 검증
  - Soundness: 증명의 각 구성요소를 개별 변조하여 탐지 확인
    commitment 변조 → Fiat-Shamir 챌린지가 바뀜 → 실패
    evaluation 변조 → KZG opening과 불일치 → 실패
    opening 변조 → 페어링 등식 불일치 → 실패
    cross-circuit → VK 불일치 → 실패
```

---

### Part 20: 전체 파이프라인에서의 위치

```
┌──────────────────────────────────────────────────────────────────┐
│                        PLONK 전체 스택                             │
│                                                                    │
│  Step 13: KZG ─── 다항식 commitment                               │
│    │  setup, commit, open, verify, batch                          │
│    │                                                               │
│  Step 14: PLONKish ─── 제약 시스템                                 │
│    │  게이트 방정식, selector, wire, copy constraint                │
│    │                                                               │
│  Step 14: Permutation ─── 와이어 연결 증명                         │
│    │  σ 다항식, grand product Z(x)                                 │
│    │                                                               │
│  Step 15: Plookup ─── 테이블 멤버십 증명                           │
│    │  sorted list, h1/h2, Z_lookup                                │
│    │                                                               │
│  ★ Step 16: PLONK Prover/Verifier ─── 통합                       │
│    │  5-round Fiat-Shamir + linearization + batched pairing       │
│    │                                                               │
│    │  이 단계가 "접착제":                                           │
│    │    KZG로 commit → Fiat-Shamir로 챌린지 도출                   │
│    │    → quotient로 제약 만족 증명 → linearization으로 검증 효율화  │
│    │    → batched pairing으로 최종 검증                             │
│    │                                                               │
│  Step 17: FFLONK ─── 최적화                                       │
│       단일 opening proof, 더 작은 증명, custom gate 확장           │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

PLONK이 Groth16과 같은 수준의 "완전한 증명 시스템"을 구성:
  Circuit → Setup → Prove → Verify → bool

차이점:
  Groth16: R1CS → QAP → 3-페어링 검증
  PLONK: PLONKish → Lagrange → KZG → 2-페어링 검증

다음 Step 17 (FFLONK):
  PLONK의 증명 크기를 더 줄이고
  custom gate를 통한 회로 최적화
```

---

> [!summary] Step 16 요약
> ```
> PLONK Prover/Verifier = KZG + PLONKish + Permutation의 통합
>
> 5-Round Fiat-Shamir 프로토콜:
>   R1: Wire commit [a]₁,[b]₁,[c]₁
>   R2: β,γ → Grand product Z(x) → [Z]₁
>   R3: α → Quotient T(x) → [t_lo]₁,[t_mid]₁,[t_hi]₁
>   R4: ζ → 6개 평가값 (ā,b̄,c̄,σ̄_A,σ̄_B,z̄_ω)
>   R5: ν → Linearization + batch opening → W_ζ, W_{ζω}
>
> 핵심 기법:
>   - Linearization: 다항식 곱 → 스칼라 × 다항식 (Verifier 효율화)
>   - r_constant 보정: r(ζ)=0 보장 (상수 잔여 제거)
>   - Batched pairing: 2번의 KZG 검증을 1번의 페어링 등식으로
>   - u·[Z]₁: 두 번째 개구점(ζω)의 commitment을 F에 포함
>
> 증명 크기: 9 G1 + 6 Fr ≈ 768 bytes
> 검증 비용: O(1) — 페어링 2회, 스칼라 곱 ~15회
>
> 코드: plonk/prover.rs (~1000줄, 11 테스트)
> 총 테스트: 306개 (기존 295 + 신규 11)
>
> 다음: Step 17 FFLONK 최적화
> ```
