## Step 17: FFLONK — Combined Opening으로 증명을 최적화하다

### 핵심 질문: PLONK Round 5는 왜 비효율적인가?

```
Step 16에서 구축한 PLONK Prover의 Round 5를 복기하면:

  ζ에서 6개 다항식 open:
    W_ζ(x) = [r(x) + ν·(a(x)-ā) + ν²·(b(x)-b̄) + ... + ν⁵·(σ_B(x)-σ̄_B)] / (x - ζ)

  ζω에서 1개 다항식 open:
    W_{ζω}(x) = [Z(x) - z̄_ω] / (x - ζω)

  → 2개의 G1 opening proof

  검증:
    u = transcript.challenge()
    e(W_ζ + u·W_{ζω}, [τ]₂) = e(ζ·W_ζ + u·ζω·W_{ζω} + F - [E]₁, G₂)

  → custom pairing 등식, 직접 유도 필요

문제점:
  1. 2개의 서로 다른 evaluation point → 2개의 quotient polynomial
  2. u 챌린지로 결합 → 복잡한 pairing 등식 유도
  3. 검증 코드가 ~50줄의 수동 G1 연산
  4. 기존 KZG 모듈의 batch_verify 재사용 불가

FFLONK의 질문:
  "이 7개 다항식을 하나로 합쳐서,
   기존 kzg::batch_open + batch_verify를 그대로 쓸 수는 없을까?"
```

> [!important] FFLONK의 핵심 아이디어
> ```
> PLONK Round 5:
>   "6개 다항식을 ζ에서, 1개를 ζω에서"
>   → 2개의 opening proof + custom pairing 등식
>
> FFLONK Round 5:
>   "7개를 1개 combined polynomial로 합치고, 그것을 {ζ, ζω} 2점에서"
>   → kzg::batch_open 1번 → 1개의 opening proof
>   → kzg::batch_verify 1번 → 검증 완료
>
> 비유:
>   PLONK: 6과목 시험지를 한 봉투에, 1과목을 다른 봉투에 넣어 제출
>          → 시험관이 두 봉투를 따로 열어 채점
>   FFLONK: 7과목을 하나의 종합 시험지로 합쳐 제출
>          → 시험관이 한 번에 채점
> ```

---

### Part 1: 왜 Combined Polynomial이 가능한가?

#### KZG의 가법 동형성(Additive Homomorphism)

```
KZG commitment의 핵심 속성:

  commit(f) = [f(τ)]₁ = Σᵢ fᵢ · [τⁱ]₁

  이것은 선형 연산이므로:
    commit(f + g) = commit(f) + commit(g)              ... (1)
    commit(α · f) = α · commit(f)                       ... (2)

  일반화:
    commit(f + ν·g + ν²·h) = commit(f) + ν·commit(g) + ν²·commit(h)

왜 이것이 중요한가?

  Prover가 combined(x) = r(x) + ν·a(x) + ν²·b(x) + ... 을 만들면

  Verifier는 개별 commitment [r]₁, [a]₁, [b]₁, ... 로부터:
    commit(combined) = [r]₁ + ν·[a]₁ + ν²·[b]₁ + ...

  을 스칼라 곱과 덧셈만으로 재구성할 수 있다!

  → combined polynomial을 Prover에게서 직접 받을 필요 없음
  → commitment의 정직성은 KZG의 binding 속성이 보장

제약: 곱에 대해서는 동형이 아님!
  commit(f · g) ≠ commit(f) · commit(g)  (이런 연산 자체가 정의되지 않음)

  왜? G1 × G1 → G1 연산은 없음 (G1은 덧셈군)
  pairing e: G1 × G2 → GT 는 있지만, GT로 가면 돌아올 수 없음

  이것이 linearization이 필요한 이유:
    q_L(x) · a(x) 같은 다항식 곱은 commitment에서 재구성 불가
    → a(ζ) = ā 를 스칼라로 빼서 ā · q_L(x) 로 변환 (곱 → 스칼라곱)
    → 이 과정이 linearization = r(x)
```

#### 곱과 선형 결합의 차이 — 구체적 예시

```
f(x) = 3x + 1,  g(x) = 2x + 5

선형 결합 (가능):
  h(x) = f(x) + 7·g(x) = (3x+1) + 7(2x+5) = 17x + 36

  commit(h) = commit(f) + 7·commit(g)
            = [f(τ)]₁ + 7·[g(τ)]₁
            = [(f + 7g)(τ)]₁
            = [h(τ)]₁  ✓

  Verifier가 [f]₁과 [g]₁을 알면 [h]₁을 계산 가능!

곱 (불가능):
  p(x) = f(x) · g(x) = 6x² + 17x + 5

  commit(p) = [6τ² + 17τ + 5]₁
  commit(f) · commit(g) = ???  ← G1 점끼리의 "곱"은 정의되지 않음

  pairing을 쓰면?
    e([f(τ)]₁, [g(τ)]₂) = e(G₁, G₂)^{f(τ)·g(τ)} ∈ GT
    → GT에서 "f·g가 맞는지"는 확인 가능
    → 하지만 commit(f·g) ∈ G1을 복원할 수는 없음
    → 그래서 PLONK verify는 pairing을 직접 사용하는 것

이것이 FFLONK의 설계를 결정함:
  combined(x) = r(x) + ν·a(x) + ν²·b(x) + ...
  → 모든 항이 "선형 결합" → Verifier가 commitment 재구성 가능
  → r(x) 안에서 이미 모든 곱이 linearization으로 해소됨
```

---

### Part 2: PLONK Round 5 vs FFLONK Round 5 — 상세 비교

#### PLONK 방식 (Step 16)

```
PLONK Round 5의 논리:

  7개 다항식의 evaluation을 검증해야 함:
    ζ에서: r(ζ)=0, a(ζ)=ā, b(ζ)=b̄, c(ζ)=c̄, σ_A(ζ)=σ̄_A, σ_B(ζ)=σ̄_B
    ζω에서: Z(ζω)=z̄_ω

  Step 1: ζ에서의 6개를 ν로 결합하여 하나의 quotient
    batch_numerator_ζ(x) = r(x) + ν·(a(x)-ā) + ν²·(b(x)-b̄) + ...
    W_ζ(x) = batch_numerator_ζ(x) / (x - ζ)

    r(ζ)=0 이므로 r(x)에서 상수 빼기 불필요

  Step 2: ζω에서의 1개는 별도
    W_{ζω}(x) = (Z(x) - z̄_ω) / (x - ζω)

  Step 3: 검증 시 u 챌린지로 2개를 다시 결합
    u = transcript.challenge()
    → custom pairing 등식

  → 3단계 결합: ν(같은 점 batch) → 별도(다른 점) → u(두 proof 결합)
```

#### FFLONK 방식

```
FFLONK Round 5의 논리:

  핵심 전환:
    "점별로 다항식을 나누지 말고, 먼저 다항식을 합치자"

  Step 1: 7개 다항식을 ν로 결합
    combined(x) = r(x) + ν·a(x) + ν²·b(x) + ν³·c(x)
                + ν⁴·σ_A(x) + ν⁵·σ_B(x) + ν⁶·Z(x)

    → 이것은 1개의 다항식!

  Step 2: 2개 점에서 평가
    combined(ζ) = r(ζ) + ν·ā + ν²·b̄ + ... + ν⁶·Z(ζ)
    combined(ζω) = r(ζω) + ν·a(ζω) + ... + ν⁶·Z(ζω)

    주의: 이 평가값은 "개별 평가의 가중합"과 정확히 일치
          (다항식 평가의 선형성!)

  Step 3: batch_open으로 2점 동시 개구
    batch_opening = kzg::batch_open(srs, &combined, &[ζ, ζω])
    → 1개의 G1 proof!

  → 1단계 결합: ν(다항식 합체) → batch_open(2점 동시)

왜 순서를 바꿀 수 있는가?

  PLONK: "같은 점끼리 batch → 다른 점은 따로 → u로 재결합"
  FFLONK: "먼저 다항식을 합체 → 합체된 하나를 여러 점에서 open"

  핵심 관찰:
    combined polynomial을 {ζ, ζω}에서 batch_open하면
    내부적으로 vanishing poly Z(x) = (x-ζ)(x-ζω) 로 나눔

    이것은 PLONK의 (x-ζ)와 (x-ζω) 두 나눗셈을
    하나의 (x-ζ)(x-ζω) 나눗셈으로 통합한 것!

  건전성:
    ν가 랜덤이면:
      combined(ζ) = ŷ 이고 combined(ζω) = ŷ' 일 때
      r(ζ) + ν·a(ζ) + ... = ŷ  이면서
      r(ζω) + ν·a(ζω) + ... = ŷ' 이면
      → 각 f_i(ζ), f_i(ζω) 가 올바름 (Schwartz-Zippel)
```

---

### Part 3: Combined Polynomial — 수학적 유도

#### 정의

```
입력:
  다항식 7개: r(x), a(x), b(x), c(x), σ_A(x), σ_B(x), Z(x)
  랜덤 챌린지: ν (Fiat-Shamir에서 도출)

정의:
  combined(x) = Σᵢ₌₀⁶ νⁱ · fᵢ(x)

  여기서:
    f₀ = r(x)     (linearization polynomial)
    f₁ = a(x)     (left wire)
    f₂ = b(x)     (right wire)
    f₃ = c(x)     (output wire)
    f₄ = σ_A(x)   (permutation A)
    f₅ = σ_B(x)   (permutation B)
    f₆ = Z(x)     (grand product)
```

#### 왜 r(x)이 맨 앞인가?

```
r(x)의 특별한 성질: r(ζ) = 0

  r(x)은 linearization으로 구성되었고, r_constant 보정으로
  ζ에서 정확히 0이 되도록 만들었음 (Step 16 Part 8 참조)

PLONK에서 이 성질이 활용된 방식:
  W_ζ(x) = [r(x) + ν·(a(x)-ā) + ...] / (x - ζ)
  → r(x)는 이미 r(ζ)=0이므로 r(x)-0 = r(x), 상수 빼기 불필요

FFLONK에서는?
  combined(ζ) = r(ζ) + ν·a(ζ) + ν²·b(ζ) + ... + ν⁶·Z(ζ)
             = 0 + ν·ā + ν²·b̄ + ... + ν⁶·Z(ζ)

  r(ζ)=0 이므로 combined(ζ)에서 r의 기여는 0
  → 검증자는 r의 ζ에서의 값을 몰라도 됨
  → 검증자가 아는 스칼라(ā, b̄, c̄, σ̄_A, σ̄_B, z̄_ω)와 Z(ζ)로 combined(ζ) 재계산?

  잠깐, Z(ζ)는 검증자가 모름!

  실제 설계:
    combined(ζ)와 combined(ζω)는 Prover가 직접 계산하여 제공
    → FflonkProof에 combined_eval_zeta, combined_eval_zeta_omega 포함

    검증자는:
      1. combined_comm을 commitment 선형 결합으로 재구성
      2. batch_verify(combined_comm, {ζ→v₁, ζω→v₂}, proof)
      → "combined(ζ)=v₁ 이고 combined(ζω)=v₂" 를 KZG로 확인
```

#### 건전성(Soundness) 유도

```
Claim: ν가 랜덤이면, combined polynomial의 opening 검증은
       개별 다항식의 opening 검증과 동등하다.

증명 (Schwartz-Zippel에 의한 귀류법):

  가정: 악의적 Prover가 어떤 i에 대해 fᵢ(ζ) ≠ yᵢ (올바른 값) 인데도
        combined(ζ) = Σ νⁱ·yᵢ 가 성립하도록 만들고 싶다.

  그러면:
    Σ νⁱ · fᵢ(ζ) = Σ νⁱ · yᵢ
    Σ νⁱ · (fᵢ(ζ) - yᵢ) = 0

  eᵢ = fᵢ(ζ) - yᵢ 라 하면 (적어도 하나의 eᵢ ≠ 0):
    Σ νⁱ · eᵢ = 0

  이것은 ν에 대한 6차 다항식 = 0:
    e₀ + e₁·ν + e₂·ν² + ... + e₆·ν⁶ = 0

  eᵢ 중 적어도 하나가 ≠ 0 이면,
  이 등식은 ν에 대한 비자명 다항식 = 0.

  Schwartz-Zippel: 6차 다항식의 근은 최대 6개.
  ν가 |Fr| ≈ 2²⁵⁴ 크기의 체에서 랜덤으로 선택되면:
    확률 ≤ 6 / 2²⁵⁴ ≈ 2⁻²⁵¹ ≈ 0

  마찬가지로 ζω에서도 동일한 논증 적용.

  따라서: batch_open + batch_verify가 통과하면
          → 각 fᵢ(ζ)과 fᵢ(ζω)가 올바를 확률 ≥ 1 - negl(λ)

결론:
  PLONK의 2-proof 방식과 동일한 건전성 보장,
  하지만 proof 개수는 2 → 1로 줄어듦.
```

---

### Part 4: SRS의 G2 확장 — 왜 degree 2가 필요한가?

#### PLONK vs FFLONK의 G2 요구사항

```
KZG batch_verify의 검증 등식:

  e(π, [Z(τ)]₂) = e(C - [I(τ)]₁, G₂)

  여기서 Z(x) = vanishing polynomial = ∏(x - zᵢ)

  [Z(τ)]₂ 계산에는 G2에서의 다항식 commit이 필요:
    Z(x) = z₀ + z₁x + z₂x² + ...
    [Z(τ)]₂ = z₀·[1]₂ + z₁·[τ]₂ + z₂·[τ²]₂ + ...

  → deg(Z) 만큼의 G2 powers 필요

PLONK의 경우:
  검증에 batch_verify를 쓰지 않음 (custom pairing 등식 사용)
  G2는 {[1]₂, [τ]₂} 2개면 충분
  → max_degree_g2 = 1

FFLONK의 경우:
  batch_open에서 evaluation points = {ζ, ζω} → k=2개 점
  Z(x) = (x - ζ)(x - ζω) → degree 2
  → [Z(τ)]₂ = z₀·[1]₂ + z₁·[τ]₂ + z₂·[τ²]₂
  → G2 powers 3개 필요: {[1]₂, [τ]₂, [τ²]₂}
  → max_degree_g2 = 2

코드에서:
  // PLONK
  let srs = kzg::setup(3 * n + 5, 1, rng);  // G2: [1]₂, [τ]₂

  // FFLONK
  let srs = kzg::setup(3 * n + 5, 2, rng);  // G2: [1]₂, [τ]₂, [τ²]₂
                                     ↑
                               이 차이가 전부!
```

#### SRS 크기에 미치는 영향

```
SRS 구성 비교:

  ┌──────────────┬────────────────┬────────────────────────────┐
  │              │ PLONK          │ FFLONK                     │
  ├──────────────┼────────────────┼────────────────────────────┤
  │ G1 powers    │ 3n+6개         │ 3n+6개 (동일)               │
  │ G2 powers    │ 2개            │ 3개 (+1)                    │
  │ G1 총 크기   │ (3n+6)·64B     │ (3n+6)·64B                 │
  │ G2 총 크기   │ 2·128B = 256B  │ 3·128B = 384B              │
  │ SRS 추가 비용 │ —              │ +128B (G2 1개)             │
  └──────────────┴────────────────┴────────────────────────────┘

  n = 1024 (2¹⁰) 일 때:
    G1: (3·1024+6)·64 = 196,992 bytes ≈ 192 KB
    G2 추가: +128 bytes

  G2 1개 추가는 SRS 대비 0.065% — 무시할 수 있는 비용!
```

#### vanishing polynomial의 구체적 계산

```
batch_open에서 ζ와 ζω가 주어지면:

  Z(x) = (x - ζ)(x - ζω)
       = x² - (ζ + ζω)x + ζ·ζω
       = x² - ζ(1 + ω)x + ζ²ω

  계수:
    z₀ = ζ²ω           (상수항)
    z₁ = -ζ(1 + ω)     (1차항)
    z₂ = 1              (2차항)

  [Z(τ)]₂ = ζ²ω · [1]₂ + (-ζ(1+ω)) · [τ]₂ + 1 · [τ²]₂

  [τ²]₂ 가 없으면 이 계산이 불가능!
  → FFLONK에서 max_degree_g2 = 2 가 필수인 이유

batch_verify 내부 코드 (kzg.rs에서):
  fn commit_g2(srs: &SRS, poly: &Polynomial) -> G2 {
      let mut result = G2::identity();
      for (i, &coeff) in poly.coeffs.iter().enumerate() {
          result = result + srs.g2_powers[i].scalar_mul(&coeff.to_repr());
      }
      result
  }

  → poly.coeffs.len() = 3 (degree 2)
  → srs.g2_powers[0..3] 접근 → 3개 필요!
```

---

### Part 5: FFLONK Prover — Round 5 상세 유도

#### Round 1-4: PLONK과 완전히 동일

```
FFLONK은 Round 5만 변경. Round 1-4는 100% 동일:

  Round 1: [a]₁, [b]₁, [c]₁ commit → transcript
           → β, γ 도출

  Round 2: Z(x) grand product 계산 → [Z]₁ commit → transcript
           → α 도출

  Round 3: Gate + Permutation 제약 결합
           → T(x) = numerator / Z_H(x)
           → T 3분할 → [t_lo]₁, [t_mid]₁, [t_hi]₁ commit → transcript
           → ζ 도출

  Round 4: a(ζ), b(ζ), c(ζ), σ_A(ζ), σ_B(ζ), Z(ζω) 평가
           → transcript에 추가
           → ν 도출

여기까지 PLONK과 FFLONK의 차이 = 0
Fiat-Shamir transcript도 동일 → 같은 ν가 도출됨
```

#### Round 5: Linearization → Combined → Batch Open

```
PLONK Round 5 (비교용):
  ν 도출 후:
  1. W_ζ(x) 구성 (ζ에서의 batch quotient)
  2. W_{ζω}(x) 구성 (ζω에서의 quotient)
  3. 각각 commit → 2개의 G1

FFLONK Round 5:
  ν 도출 후:

  ── Step 5a: Linearization r(x) 구성 ──────────────────
    (PLONK과 동일한 공식)

    보조 스칼라 계산:
      ζⁿ = ζ · ζ · ... · ζ  (n번)
      ζ²ⁿ = (ζⁿ)²
      Z_H(ζ) = ζⁿ - 1
      L₁(ζ) = Z_H(ζ) / (n · (ζ - 1))

    Gate linearization:
      r_gate(x) = ā·q_L(x) + b̄·q_R(x) + c̄·q_O(x) + (ā·b̄)·q_M(x) + q_C(x)

    Permutation linearization:
      perm_num = (ā+βζ+γ)(b̄+βK₁ζ+γ)(c̄+βK₂ζ+γ)
      perm_den_partial = (ā+βσ̄_A+γ)(b̄+βσ̄_B+γ)

      r_perm(x) = (α·perm_num + α²·L₁(ζ)) · Z(x)
                 - α·β·perm_den_partial·z̄_ω · σ_C(x)

    Quotient linearization:
      T_combined(x) = t_lo(x) + ζⁿ·t_mid(x) + ζ²ⁿ·t_hi(x)
      r_quot(x) = Z_H(ζ) · T_combined(x)

    상수 잔여 보정:
      r_constant = α·z̄_ω·perm_den_partial·(c̄+γ) + α²·L₁(ζ)
      r(x) = r_gate(x) + r_perm(x) - r_quot(x) - r_constant

    검증: r(ζ) = 0 ✓

  ── Step 5b: Combined polynomial 구성 ─────────────────
    ν² = ν·ν
    ν³ = ν²·ν
    ν⁴ = ν³·ν
    ν⁵ = ν⁴·ν
    ν⁶ = ν⁵·ν

    combined(x) = r(x) + ν·a(x) + ν²·b(x) + ν³·c(x)
                + ν⁴·σ_A(x) + ν⁵·σ_B(x) + ν⁶·Z(x)

  ── Step 5c: 2점 평가 ─────────────────────────────────
    combined_eval_zeta = combined(ζ)
    combined_eval_zeta_omega = combined(ζω)

  ── Step 5d: Batch open ───────────────────────────────
    batch_opening = kzg::batch_open(srs, &combined, &[ζ, ζω])
    w = batch_opening.proof    ← 단 1개의 G1!

    batch_open 내부:
      1. values = [combined(ζ), combined(ζω)]
      2. Z(x) = (x-ζ)(x-ζω)                    ← degree 2
      3. I(x) = Lagrange interpolation from {(ζ,v₁), (ζω,v₂)}
      4. quotient = (combined(x) - I(x)) / Z(x)
      5. w = commit(quotient)
```

#### 왜 u 챌린지가 필요 없는가?

```
PLONK에서 u가 필요했던 이유:

  PLONK은 2개의 독립적인 opening proof를 생성:
    W_ζ: 6개 다항식의 ζ batch opening
    W_{ζω}: Z의 ζω opening

  이 2개를 "하나의 pairing 등식"으로 결합하기 위해
  u = transcript.challenge()로 랜덤 가중치를 부여:
    (등식1) + u · (등식2) → 하나의 등식

  u 없이 단순 합산하면?
    악의적 Prover가 (등식1)에서 δ 만큼 오차를 넣고
    (등식2)에서 -δ 를 넣어 상쇄시킬 수 있음
    → u가 랜덤이면 상쇄 불가 (Schwartz-Zippel)

FFLONK에서 u가 불필요한 이유:

  FFLONK은 "하나의 다항식"을 "하나의 batch_open"으로 처리
  → 결합할 2개의 독립 등식이 없음!

  kzg::batch_verify가 내부적으로:
    e(π, [Z(τ)]₂) = e(C - [I(τ)]₁, G₂)

  이것은 단일 등식. 2개를 u로 합칠 필요가 없음.

  2개 점 {ζ, ζω}의 동시 검증은
  vanishing polynomial Z(x) = (x-ζ)(x-ζω) 가 담당.

  Fiat-Shamir 챌린지 개수 비교:
    PLONK:  β, γ, α, ζ, ν, u → 6개
    FFLONK: β, γ, α, ζ, ν    → 5개 (u 없음)
```

---

### Part 6: FFLONK Verifier — batch_verify로의 환원

#### PLONK Verifier vs FFLONK Verifier

```
PLONK Verifier (Step 16):

  1. Fiat-Shamir: β, γ, α, ζ, ν, u 도출
  2. 보조값: ζⁿ, Z_H(ζ), L₁(ζ)
  3. [r]₁ 구성 (Gate + Perm + Quot + r_constant 보정)
  4. F = [r]₁ + ν·[a]₁ + ... + ν⁵·[σ_B]₁ + u·[Z]₁
  5. E = ν·ā + ... + ν⁵·σ̄_B + u·z̄_ω
  6. Pairing check:
     e(W_ζ + u·W_{ζω}, [τ]₂) = e(ζ·W_ζ + u·ζω·W_{ζω} + F - E·G₁, G₂)

  → 50줄의 G1 연산 코드
  → pairing 2회

FFLONK Verifier:

  1. Fiat-Shamir: β, γ, α, ζ, ν 도출 (u 불필요)
  2. 보조값: ζⁿ, Z_H(ζ), L₁(ζ) (동일)
  3. [r]₁ 구성 (동일)
  4. combined_comm 재구성:
     combined_comm = [r]₁ + ν·[a]₁ + ν²·[b]₁ + ν³·[c]₁
                   + ν⁴·[σ_A]₁ + ν⁵·[σ_B]₁ + ν⁶·[Z]₁
  5. BatchOpening 구성:
     opening = { points: [ζ, ζω],
                 values: [combined_eval_zeta, combined_eval_zeta_omega],
                 proof: w }
  6. kzg::batch_verify(srs, &combined_comm, &opening)

  → F, E, u 계산 불필요!
  → batch_verify가 pairing까지 처리
  → 코드가 더 단순하고, KZG 모듈의 검증 로직을 재사용
```

#### Verifier의 combined_comm 재구성 — 상세

```
FFLONK Verifier의 핵심 단계는 combined_comm 구성.
이것이 가능한 이유를 다시 한번 확인:

  Prover가 구성한 다항식:
    combined(x) = r(x) + ν·a(x) + ν²·b(x) + ν³·c(x)
                + ν⁴·σ_A(x) + ν⁵·σ_B(x) + ν⁶·Z(x)

  이것의 commitment:
    [combined(τ)]₁ = [r(τ)]₁ + ν·[a(τ)]₁ + ν²·[b(τ)]₁ + ν³·[c(τ)]₁
                   + ν⁴·[σ_A(τ)]₁ + ν⁵·[σ_B(τ)]₁ + ν⁶·[Z(τ)]₁

  각 항의 출처:
    [r(τ)]₁ → Verifier가 Step 3에서 구성 (r_comm)
    [a(τ)]₁ → proof.a_comm (Round 1에서 Prover 제공)
    [b(τ)]₁ → proof.b_comm
    [c(τ)]₁ → proof.c_comm
    [σ_A(τ)]₁ → vk.sigma_a_comm (Setup에서 생성, VK에 포함)
    [σ_B(τ)]₁ → vk.sigma_b_comm
    [Z(τ)]₁ → proof.z_comm (Round 2에서 Prover 제공)

  → 모든 항이 proof 또는 VK에서 제공됨
  → Verifier는 스칼라 곱과 덧셈만으로 combined_comm 계산 가능!

코드:
  let combined_comm = Commitment(
      r_comm                                           // [r]₁
      + proof.a_comm.0.scalar_mul(&nu.to_repr())       // + ν·[a]₁
      + proof.b_comm.0.scalar_mul(&nu2.to_repr())      // + ν²·[b]₁
      + proof.c_comm.0.scalar_mul(&nu3.to_repr())      // + ν³·[c]₁
      + vk.sigma_a_comm.0.scalar_mul(&nu4.to_repr())   // + ν⁴·[σ_A]₁
      + vk.sigma_b_comm.0.scalar_mul(&nu5.to_repr())   // + ν⁵·[σ_B]₁
      + proof.z_comm.0.scalar_mul(&nu6.to_repr())      // + ν⁶·[Z]₁
  );
```

#### batch_verify 내부에서 일어나는 일

```
kzg::batch_verify(srs, &combined_comm, &opening) 호출 시:

  1. vanishing polynomial 구성:
     Z(x) = (x - ζ)(x - ζω)

  2. [Z(τ)]₂ 계산 (G2에서 commit):
     Z(x) = ζ²ω + (-ζ-ζω)x + x²
     [Z(τ)]₂ = ζ²ω·[1]₂ + (-ζ-ζω)·[τ]₂ + 1·[τ²]₂
                                              ↑ max_degree_g2 = 2 필요!

  3. interpolation polynomial I(x) 구성:
     I(ζ) = combined_eval_zeta
     I(ζω) = combined_eval_zeta_omega
     → degree 1 다항식 (2점 Lagrange 보간)

  4. [I(τ)]₁ 계산 (G1에서 commit)

  5. pairing check:
     e(w, [Z(τ)]₂) = e(combined_comm - [I(τ)]₁, G₂)

     LHS: "proof w가 올바른 quotient polynomial의 commit인가?"
     RHS: "combined(x) - I(x)가 Z(x)로 나누어떨어지는가?"

     이것이 성립하면:
       combined(ζ) = I(ζ) = combined_eval_zeta      ✓
       combined(ζω) = I(ζω) = combined_eval_zeta_omega  ✓

     → combined polynomial의 두 점에서의 평가가 정직함이 증명됨
     → ν가 랜덤이므로 각 개별 다항식의 평가도 정직 (Part 3 건전성)
```

---

### Part 7: 증명 크기와 검증 비용 비교

#### 증명 크기 분석

```
┌──────────────────────┬──────────────────┬──────────────────┐
│ 항목                   │ PLONK            │ FFLONK           │
├──────────────────────┼──────────────────┼──────────────────┤
│ Round 1: wire comm   │ 3 G1             │ 3 G1 (동일)      │
│ Round 2: Z comm      │ 1 G1             │ 1 G1             │
│ Round 3: T comm      │ 3 G1             │ 3 G1             │
│ Round 4: evaluations │ 6 Fr             │ 6 Fr             │
│ Round 5: openings    │ 2 G1             │ 1 G1 + 2 Fr      │
├──────────────────────┼──────────────────┼──────────────────┤
│ 합계                   │ 9 G1 + 6 Fr      │ 8 G1 + 8 Fr      │
│ 크기 (비압축)          │ 9·64 + 6·32      │ 8·64 + 8·32      │
│                      │ = 768 bytes      │ = 768 bytes      │
└──────────────────────┴──────────────────┴──────────────────┘

총 바이트 수는 동일 (768B)!

  G1이 1개 줄고 (64B ↓), Fr이 2개 늘었음 (64B ↑)

왜 바이트가 같은데 의미가 있는가?

  1. G1 연산 vs Fr 연산의 비용 차이:
     G1 scalar_mul: ~수백 배 비쌈 (타원곡선 연산)
     Fr 곱셈: ~수 나노초 (64비트 정수 연산)
     → G1 1개 절약 = 검증 속도 향상

  2. 압축 시 차이:
     G1 비압축: 64 bytes (x, y)
     G1 압축: 33 bytes (x + 부호 비트)
     Fr: 32 bytes (항상)

     PLONK 압축: 9·33 + 6·32 = 297 + 192 = 489 bytes
     FFLONK 압축: 8·33 + 8·32 = 264 + 256 = 520 bytes

     → 압축 시에는 FFLONK이 31 bytes 더 큼!
     → 압축 안 할 경우 동일

  3. 교육적 가치:
     "여러 opening을 하나로 합치는" 기법의 시연
     실무에서는 더 많은 다항식을 결합할수록 절약 효과 증가
```

#### 검증 비용 분석

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│ 연산                 │ PLONK                │ FFLONK               │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Fiat-Shamir 해시   │ ~15회                │ ~14회 (u 없음)        │
│ G1 scalar_mul      │ ~15회                │ ~14회 (F,E 대신       │
│                    │ (r_comm + F + E)     │  combined_comm)      │
│ G2 scalar_mul      │ 0회                  │ 2회 (commit_g2)      │
│ Lagrange 보간      │ 0회                  │ 1회 (degree 1, 2점)  │
│ G1 commit (I)      │ 0회                  │ 1회 (degree 1)       │
│ Pairing            │ 2회                  │ 2회 (동일)            │
├────────────────────┼──────────────────────┼──────────────────────┤
│ 총 비용 (지배항)    │ 2 pairing +          │ 2 pairing +           │
│                    │ ~15 G1 MSM           │ ~14 G1 MSM +          │
│                    │                      │ 2 G2 MSM              │
└────────────────────┴──────────────────────┴──────────────────────┘

G2 scalar_mul은 G1보다 ~2배 비쌈 (Fp2 산술).
하지만 degree-1 polynomial이므로 2회 → 작은 overhead.

실질적 차이:
  - 코드 단순화: custom pairing 등식 → batch_verify 호출 1줄
  - 모듈 재사용: kzg.rs 변경 없이 기존 API 활용
  - 확장성: 점이 3개, 4개로 늘어도 batch_open/verify가 자동 처리

실무에서의 FFLONK (Polygon zkEVM 등):
  - 8개 이상의 다항식, 3-4개 evaluation point
  - FFLONK으로 opening proof 1개 → 검증 비용 대폭 절감
```

---

### Part 8: 수치 트레이스 — x³ + x + 5 = y (x=3, y=35)

```
CubicCircuit: 3개 게이트, 5개 copy constraints

── 회로 구성 ──────────────────────────────────────

  Gate 0: x * x = v1      → 3 * 3 = 9
  Gate 1: v1 * x = v2     → 9 * 3 = 27
  Gate 2: v2 + x + 5 = y  → 27 + 3 + 5 = 35
  Gate 3: dummy pad        → 0, 0, 0

  n = 4, ω = 4차 원시 단위근

── Round 1-4: PLONK과 동일 ─────────────────────

  [a]₁, [b]₁, [c]₁ commit  → β, γ
  Z(x) grand product       → [Z]₁ → α
  T(x) 3분할               → [t_lo]₁, [t_mid]₁, [t_hi]₁ → ζ
  6개 평가값                → ν

── Round 5 (FFLONK 차이점) ──────────────────────

  r(x) 구성 (PLONK과 동일한 공식):
    r_gate + r_perm - r_quot - r_constant
    r(ζ) = 0  ✓

  combined 구성:
    combined(x) = r(x) + ν·a(x) + ν²·b(x) + ν³·c(x)
                + ν⁴·σ_A(x) + ν⁵·σ_B(x) + ν⁶·Z(x)

    여기서 Z(x) ≠ 1 (copy constraints가 있으므로!)

    combined(ζ)를 Prover가 직접 계산:
      = 0 + ν·a(ζ) + ν²·b(ζ) + ν³·c(ζ) + ν⁴·σ_A(ζ) + ν⁵·σ_B(ζ) + ν⁶·Z(ζ)
        ↑ r(ζ)=0

    combined(ζω)도 Prover가 계산:
      = r(ζω) + ν·a(ζω) + ν²·b(ζω) + ... + ν⁶·Z(ζω)
        ↑ r(ζω) ≠ 0 일반적으로!

    주의: ζω에서는 r(ζω) ≠ 0.
          r(x)은 ζ에서만 0이 보장됨, ζω에서는 일반적인 값.

  batch_open:
    kzg::batch_open(srs, &combined, &[ζ, ζω])

    내부:
      Z(x) = (x-ζ)(x-ζω)                    → degree 2
      I(x): I(ζ) = combined(ζ), I(ζω) = combined(ζω)  → degree 1
      q(x) = (combined(x) - I(x)) / Z(x)
      w = commit(q)

    → FflonkProof {
        ...,
        combined_eval_zeta,
        combined_eval_zeta_omega,
        w,       ← G1 1개!
      }

── Verifier 측 ──────────────────────────────────

  1. Fiat-Shamir → β, γ, α, ζ, ν (PLONK과 동일, u 없음)

  2. 보조값: ζⁿ, Z_H(ζ), L₁(ζ)

  3. [r]₁ 구성 (PLONK과 동일):
     r_comm = ā·[q_L]₁ + b̄·[q_R]₁ + c̄·[q_O]₁ + (ā·b̄)·[q_M]₁ + [q_C]₁
            + (α·perm_num + α²·L₁)·[Z]₁
            - α·β·den·z̄_ω·[σ_C]₁
            - Z_H·[t_lo]₁ - Z_H·ζⁿ·[t_mid]₁ - Z_H·ζ²ⁿ·[t_hi]₁
            - r_const·G₁

  4. combined_comm:
     = r_comm + ν·[a]₁ + ν²·[b]₁ + ν³·[c]₁ + ν⁴·[σ_A]₁ + ν⁵·[σ_B]₁ + ν⁶·[Z]₁

  5. batch_verify:
     opening = { points: [ζ, ζω],
                 values: [combined_eval_zeta, combined_eval_zeta_omega],
                 proof: w }
     kzg::batch_verify(srs, &combined_comm, &opening) → true ✓

  PLONK과의 차이:
    PLONK: F, E 구성 → custom pairing → ~20줄의 G1 연산
    FFLONK: combined_comm → batch_verify 호출 → ~5줄
```

---

### Part 9: FflonkProof 자료구조 — 각 필드의 역할

```
pub struct FflonkProof {
    // ── Round 1: Wire Commitments ──────────────────
    // Prover가 witness 값을 commit
    // binding: commit 후 값 변경 불가 (DL 가정)
    pub a_comm: Commitment,    // [a(τ)]₁
    pub b_comm: Commitment,    // [b(τ)]₁
    pub c_comm: Commitment,    // [c(τ)]₁

    // ── Round 2: Permutation Grand Product ─────────
    // Z(ω⁰)=1에서 시작, copy constraint 만족 시 Z(ωⁿ)=1
    pub z_comm: Commitment,    // [Z(τ)]₁

    // ── Round 3: Quotient Polynomial 3분할 ──────────
    // T(x) = Numerator(x) / Z_H(x), degree ≈ 2n
    // 3분할: T = t_lo + x^n · t_mid + x^{2n} · t_hi
    pub t_lo_comm: Commitment,  // [t_lo(τ)]₁
    pub t_mid_comm: Commitment, // [t_mid(τ)]₁
    pub t_hi_comm: Commitment,  // [t_hi(τ)]₁

    // ── Round 4: Evaluations at ζ ──────────────────
    // Linearization을 위한 스칼라 값
    pub a_eval: Fr,        // a(ζ) = ā
    pub b_eval: Fr,        // b(ζ) = b̄
    pub c_eval: Fr,        // c(ζ) = c̄
    pub sigma_a_eval: Fr,  // σ_A(ζ) = σ̄_A
    pub sigma_b_eval: Fr,  // σ_B(ζ) = σ̄_B
    pub z_omega_eval: Fr,  // Z(ζω) = z̄_ω

    // ── Round 5 (FFLONK): Combined Opening ─────────
    // PLONK: w_zeta (G1) + w_zeta_omega (G1) = 2 G1
    // FFLONK: combined_eval × 2 (Fr) + w (G1) = 1 G1 + 2 Fr
    pub combined_eval_zeta: Fr,        // combined(ζ)
    pub combined_eval_zeta_omega: Fr,  // combined(ζω)
    pub w: G1,                         // batch opening proof
}

PLONK과의 구조적 차이 (Round 5만):

  PLONK:
    w_zeta: G1          → ζ에서의 batch opening proof
    w_zeta_omega: G1    → ζω에서의 opening proof
    (2 G1, 0 Fr)

  FFLONK:
    combined_eval_zeta: Fr        → combined(ζ) 평가값
    combined_eval_zeta_omega: Fr  → combined(ζω) 평가값
    w: G1                         → batch opening proof
    (1 G1, 2 Fr)

  변화: G1 1개 ↓, Fr 2개 ↑
  원인: 2개의 quotient를 1개로 합치면서 추가 평가값이 필요해짐
```

---

### Part 10: 코드 구조와 의존성

```
── 파일 구조 ────────────────────────────────────────

crates/primitives/src/plonk/
  ├── mod.rs              -- Domain, K1/K2, PlonkCircuit trait
  ├── arithmetization.rs  -- PlonkConstraintSystem, 게이트, selector
  ├── permutation.rs      -- σ 순열, grand product Z(x)
  ├── lookup.rs           -- Plookup (Step 15)
  ├── prover.rs           -- PLONK Prover/Verifier (Step 16)
  └── fflonk.rs           -- ★ FFLONK (이 단계)

── fflonk.rs 내부 구조 ─────────────────────────────

  g1_to_fr()            — G1 → Fr 변환 (prover.rs와 동일)
  Transcript             — Fiat-Shamir (prover.rs와 동일)
  FflonkSetupParams      — SRS + VerifyingKey
  FflonkProof            — 8 G1 + 8 Fr
  fflonk_setup()         — SRS(G2 degree 2) + VK 생성
  fflonk_prove()         — Round 1-4 (PLONK) + Round 5 (FFLONK)
  fflonk_verify()        — Fiat-Shamir + combined_comm + batch_verify

── 의존성 ─────────────────────────────────────────

  kzg (기존 코드 재사용, 수정 없음):
    ├── setup(max_g1, max_g2, rng) → SRS
    ├── commit(srs, poly) → Commitment
    ├── batch_open(srs, poly, points) → BatchOpening
    └── batch_verify(srs, commitment, opening) → bool

  plonk::prover (import만):
    └── VerifyingKey (pub struct, 재사용)

  plonk 기반 모듈 (재사용):
    ├── arithmetization::PlonkConstraintSystem
    ├── permutation::{compute_permutation_polynomials, compute_grand_product}
    ├── Domain, K1, K2
    └── PlonkCircuit trait

  기존 primitives (재사용):
    ├── field::{Fr, Fp}
    ├── curve::G1
    ├── qap::Polynomial
    └── hash::poseidon::poseidon_hash

── Transcript/g1_to_fr 복사 이유 ──────────────────

  prover.rs에서 Transcript과 g1_to_fr는 private (비공개):
    struct Transcript { ... }     // pub 아님
    fn g1_to_fr(...) { ... }     // pub 아님

  공유를 위해 pub으로 변경하면?
    → prover.rs의 내부 API 노출 → 모듈 경계 위반
    → 교육적 코드에서 각 모듈이 독립적으로 읽히는 것이 더 중요

  따라서 fflonk.rs에 동일한 구현을 복사:
    → 자체 완결적 (self-contained)
    → prover.rs 수정 없이 FFLONK 추가 가능
```

---

### Part 11: 건전성(Soundness) 공격 시나리오

#### 공격 1: commitment 변조

```
시나리오: Prover가 proof.a_comm을 [a(τ)]₁ + G₁ 로 변조

결과:
  1. Verifier가 Fiat-Shamir를 재현할 때,
     변조된 a_comm으로 해시 → β', γ' 이 원래와 다름
  2. 이후 모든 챌린지(α', ζ', ν')가 달라짐
  3. Verifier가 구성하는 combined_comm이 Prover의 것과 불일치
  4. batch_verify 실패

  → Fiat-Shamir가 commitment과 챌린지를 binding

테스트: fflonk_tampered_commitment_fails
  proof.a_comm = Commitment(proof.a_comm.0 + G1::generator());
  assert!(!fflonk_verify(...));
```

#### 공격 2: combined evaluation 변조

```
시나리오: Prover가 combined_eval_zeta를 1 증가시킴

결과:
  1. Fiat-Shamir 챌린지는 변하지 않음
     (combined_eval은 Round 5에서 생성, transcript에 포함 안 됨)
  2. Verifier가 combined_comm을 올바르게 재구성
  3. batch_verify에서:
     opening.values[0] = 변조된 값
     I(x) 보간 → I(ζ) = 변조된 값
     [I(τ)]₁ 이 잘못됨
     e(w, [Z(τ)]₂) ≠ e(combined_comm - [I(τ)]₁, G₂)
  4. pairing check 실패

  → KZG의 evaluation binding이 변조 탐지

테스트: fflonk_tampered_combined_eval_fails
  proof.combined_eval_zeta = proof.combined_eval_zeta + Fr::ONE;
  assert!(!fflonk_verify(...));
```

#### 공격 3: opening proof 변조

```
시나리오: Prover가 w를 w + G₁ 으로 변조

결과:
  1. Fiat-Shamir 챌린지 불변 (w는 transcript에 포함 안 됨)
  2. combined_comm 올바름
  3. opening의 values도 올바름
  4. 하지만 proof가 다름:
     e(w + G₁, [Z(τ)]₂) = e(w, [Z(τ)]₂) · e(G₁, [Z(τ)]₂)
                         ≠ e(combined_comm - [I(τ)]₁, G₂)
  5. pairing check 실패

  → KZG의 opening soundness (DL 가정)

테스트: fflonk_tampered_opening_fails
  proof.w = proof.w + G1::generator();
  assert!(!fflonk_verify(...));
```

#### 공격 4: 다른 회로의 증명 재사용

```
시나리오: 회로 A (a+b=c) 의 증명을 회로 B (a*b=c) 의 VK로 검증

결과:
  1. VK에 담긴 selector commitment가 다름:
     회로 A: q_L=1, q_M=0  →  [q_L]₁, [q_M]₁ ≠ 회로 B의 것
     회로 B: q_L=0, q_M=1
  2. Verifier가 r_comm을 구성할 때 VK_B의 commitments 사용
  3. combined_comm이 Prover(회로 A)의 것과 불일치
  4. batch_verify 실패

  → VK(Verifying Key)가 회로 구조를 인코딩
  → 다른 회로의 VK로 검증하면 commitment 불일치

테스트: fflonk_cross_circuit_fails
  params_a와 params_b를 다른 회로로 setup
  proof_a를 vk_b로 검증 → false
```

---

### Part 12: 실무에서의 FFLONK

```
FFLONK의 원래 논문:
  "FFLONK: a Fast-Fourier inspired verifier efficient version of PlonK"
  (Héctor Masip Ardevol, 2021)

  원래 FFLONK은 여기서 구현한 것보다 더 과감한 최적화:
    - FFT를 활용한 다항식 결합 (이름의 유래)
    - 모든 commitment을 하나로 합치는 "monolithic" 접근
    - round 수 자체를 줄임

  이 구현은 FFLONK의 "핵심 아이디어" (combined opening)를
  교육적으로 깔끔하게 추출한 버전.

Polygon zkEVM에서의 활용:
  Polygon Hermez는 FFLONK을 선택:
    - 8+ 다항식을 combined polynomial로 합침
    - 검증 비용: pairing 2회로 고정 (다항식 수에 무관)
    - L1 가스 비용 절감이 핵심 동기

  실무 수치 (참고):
    ┌──────────────────────┬──────────────┬──────────────┐
    │                      │ PLONK        │ FFLONK       │
    ├──────────────────────┼──────────────┼──────────────┤
    │ Opening proofs       │ 2 G1         │ 1 G1         │
    │ L1 검증 가스          │ ~300K gas    │ ~250K gas    │
    │ 증명 시간 (Prover)    │ 비슷          │ 비슷          │
    └──────────────────────┴──────────────┴──────────────┘

Shplonk과의 비교:
  Shplonk (Bünz-Fisch-Szepieniec, 2020):
    - 여러 다항식, 여러 점 → 단일 opening
    - 더 일반적인 framework
    - FFLONK은 Shplonk의 특수 케이스로 볼 수 있음

  이 구현의 접근:
    기존 kzg::batch_open/batch_verify를 재사용하는 방식
    → Shplonk의 아이디어를 "기존 API 위에" 구현한 셈
```

---

### Part 13: 전체 증명 시스템 비교

```
┌───────────────────┬────────────┬────────────┬────────────┐
│                   │ Groth16    │ PLONK      │ FFLONK     │
├───────────────────┼────────────┼────────────┼────────────┤
│ 제약 시스템        │ R1CS       │ PLONKish   │ PLONKish   │
│ Commitment        │ 내장       │ KZG        │ KZG        │
│ Setup             │ Per-circuit│ Universal  │ Universal  │
│ 증명 크기          │ 192B       │ 768B       │ 768B       │
│ 증명 크기 (압축)   │ ~128B      │ ~489B      │ ~520B      │
│ G1 원소 수         │ 2          │ 9          │ 8          │
│ Fr 원소 수         │ 0          │ 6          │ 8          │
│ Opening proofs    │ 0          │ 2          │ 1          │
│ 검증 pairing      │ 3          │ 2          │ 2          │
│ Fiat-Shamir 라운드│ -          │ 5          │ 5          │
│ 챌린지 수          │ -          │ 6          │ 5 (u 없음)  │
│ SRS G2 degree     │ -          │ 1          │ 2          │
│ Custom gate       │ ✗          │ ✓          │ ✓          │
│ Lookup            │ ✗          │ ✓          │ ✓          │
│ 코드 줄 수         │ ~350       │ ~1000      │ ~600       │
│ 테스트 수          │ 12         │ 11         │ 10         │
├───────────────────┼────────────┼────────────┼────────────┤
│ 핵심 이점          │ 최소 증명   │ 범용 setup  │ 검증 단순화 │
│ 핵심 단점          │ Per-circuit│ 2 opening   │ G2 추가    │
│                   │ setup      │ proof      │ (미미)     │
└───────────────────┴────────────┴────────────┴────────────┘

FFLONK의 위치:
  "PLONK의 검증을 KZG batch API로 단순화한 최적화"

  PLONK → FFLONK 변경 범위:
    - fflonk.rs 신규 (~600줄)
    - mod.rs 수정 (~4줄 추가)
    - 기존 코드 수정 없음 (kzg.rs, prover.rs 등 그대로)
```

---

### Part 14: 테스트 요약

```
10개 테스트 구성:

  Basic (1개):
    combined_polynomial_evaluations
      — combined = a + ν·b + ν²·c 의 평가가
        개별 평가의 ν-가중합과 일치하는지 확인
      — 다항식 평가의 선형성(linearity) 검증

  E2E Prove/Verify (5개):
    fflonk_prove_verify_addition
      — 1 게이트, copy constraint 없음 (Z=1)
      — 가장 단순한 케이스
    fflonk_prove_verify_multiplication
      — 1 게이트, copy constraint 없음
    fflonk_prove_verify_cubic
      — 3 게이트, 5 copy constraints (Z≠1!)
      — 가장 중요한 테스트: combined(ζω)에서 Z 기여가 비자명
    fflonk_prove_verify_boolean
      — a·(1-a)=0, a=0과 a=1 둘 다 테스트
    fflonk_prove_verify_larger
      — 8 게이트 체인, 7 copy constraints
      — domain size 16 (가장 큰 회로)

  Soundness (4개):
    fflonk_tampered_commitment_fails
      — a_comm 변조 → Fiat-Shamir binding으로 탐지
    fflonk_tampered_combined_eval_fails
      — combined_eval_zeta 변조 → KZG evaluation binding으로 탐지
    fflonk_tampered_opening_fails
      — w 변조 → KZG opening soundness로 탐지
    fflonk_cross_circuit_fails
      — 회로 A 증명 + 회로 B VK → VK 불일치로 탐지

테스트 설계 의도:
  - Z=1 (단순 회로)와 Z≠1 (복잡 회로) 모두 포함
    → combined(ζω)에서 ν⁶·Z(ζω) 항이 비자명한 경우 검증
  - 증명의 각 구성요소를 개별 변조하여
    어떤 보안 속성이 각 변조를 탐지하는지 명시
```

---

### Part 15: 전체 파이프라인에서의 위치

```
┌──────────────────────────────────────────────────────────────────┐
│                        완성된 PLONK 스택                             │
│                                                                    │
│  Step 13: KZG ─── 다항식 commitment (setup, commit, open, verify) │
│    │                                                               │
│  Step 14: PLONKish ─── 범용 게이트 + permutation argument         │
│    │                                                               │
│  Step 15: Plookup ─── 테이블 멤버십 증명                           │
│    │                                                               │
│  Step 16: PLONK Prover ─── 5-round Fiat-Shamir + linearization    │
│    │                                                               │
│  ★ Step 17: FFLONK ─── combined opening 최적화                   │
│                                                                    │
│  변경점 요약:                                                       │
│    Round 1-4: 동일 (wire commit → grand product → quotient → eval) │
│    Round 5: 7개 다항식 → ν로 합체 → batch_open 1회                  │
│    Verify: combined_comm 재구성 → batch_verify 1회                  │
│    SRS: G2 degree 1 → 2 (+[τ²]₂ 추가)                             │
│    u 챌린지: 불필요 (5개 챌린지로 충분)                               │
│                                                                    │
│  코드: fflonk.rs (~600줄), mod.rs (+4줄)                           │
│  테스트: 10개 (기존 306 + 10 = 316개 전체 통과)                      │
│  기존 코드 수정: 없음                                                │
│                                                                    │
│  다음: Step 18 (Mersenne31 유한체) → STARK로의 전환                  │
└──────────────────────────────────────────────────────────────────┘
```

---

> [!summary] Step 17 요약
> ```
> FFLONK = PLONK Round 5를 combined opening으로 최적화
>
> 핵심 기법:
>   1. 7개 다항식을 ν의 거듭제곱으로 선형 결합
>      combined(x) = r(x) + ν·a(x) + ... + ν⁶·Z(x)
>   2. KZG 가법 동형성으로 Verifier가 combined_comm 재구성
>      combined_comm = [r]₁ + ν·[a]₁ + ... + ν⁶·[Z]₁
>   3. kzg::batch_open(combined, {ζ, ζω}) → 1개 proof
>   4. kzg::batch_verify(combined_comm, opening) → 검증 완료
>
> 변경 사항:
>   - Opening proof: 2 G1 → 1 G1 (ν⁶ 결합)
>   - SRS G2: degree 1 → 2 (batch_verify용)
>   - u 챌린지: 불필요 (단일 polynomial)
>   - 기존 kzg API 재사용 (수정 없음)
>
> 증명 크기: 8 G1 + 8 Fr = 768 bytes (PLONK과 동일)
> 검증: batch_verify 1회 호출 (custom pairing 등식 불필요)
>
> 건전성: Schwartz-Zippel → ν 기반 결합의 soundness
>        + KZG batch_verify의 pairing soundness
>
> 코드: fflonk.rs (~600줄, 10 테스트)
> 총 테스트: 316개 (306 + 10)
> ```
