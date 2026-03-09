## Step 12: Groth16 — 3개의 페어링으로 증명을 검증하다

### 핵심 질문: QAP 이후에 무엇이 필요한가?

```
지금까지의 파이프라인:

  프로그램 → R1CS → QAP
                     │
                     │  a(x)·b(x) - c(x) = h(x)·t(x)
                     │
                     ↓
             "다항식 항등식은 확보했다"

  그런데... 검증자는 witness를 모른다!

    증명자: "나는 올바른 witness s를 알고 있어.
             a(τ)·b(τ) - c(τ) = h(τ)·t(τ) 가 성립해."

    검증자: "그 s를 보여줘."

    증명자: "안 돼! 그건 비밀이야. (ZK = Zero-Knowledge)"

    검증자: "그럼 어떻게 믿어?"

  → 이 질문에 대한 답 = Groth16
```

> [!important] Groth16의 역할
> ```
> QAP: "무엇을" 증명해야 하는지 정의 (다항식 관계)
> Groth16: "어떻게" 비밀을 드러내지 않고 증명하는지 (pairing-based)
>
> QAP 없이 Groth16은 무의미 — 증명할 관계가 없다
> Groth16 없이 QAP는 미완성 — 검증자가 witness를 알아야 한다
> ```

---

### Groth16이란?

```
Groth16 [Jens Groth, 2016]:
  - ZK-SNARK (Zero-Knowledge Succinct Non-interactive ARgument of Knowledge)
  - 가장 간결한 증명 크기: G1 2개 + G2 1개
  - 가장 빠른 검증: 페어링 3회 + MSM 1회
  - Ethereum의 zk-rollup 표준 (zkSync, Polygon zkEVM 등)

특징 요약:
  ┌──────────────────┬──────────────────────────┐
  │ 증명 크기        │ G1×2 + G2×1 = 256 bytes  │
  │ 검증 시간        │ O(1) — 페어링 3회         │
  │ 증명 생성        │ O(n) — MSM 크기           │
  │ Trusted Setup    │ 필요 (회로별 1회)          │
  │ 영지식성          │ 완전 (perfect ZK)         │
  │ 건전성           │ 계산적 (computational)     │
  └──────────────────┴──────────────────────────┘

"SNARK" 의 의미:
  S — Succinct: 증명이 짧다 (256 bytes, 회로 크기에 무관)
  N — Non-interactive: 1라운드 (증명자 → 검증자)
  AR — ARgument: 계산적 건전성 (정보 이론적 X)
  K — of Knowledge: 증명자가 실제로 witness를 "안다"
```

---

### 전체 흐름: 3단계 프로토콜

```
┌─────────────────────────────────────────────────────────────┐
│                    Groth16 프로토콜                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐       ┌─────────┐       ┌─────────┐           │
│  │ Setup   │       │ Prove   │       │ Verify  │           │
│  │         │       │         │       │         │           │
│  │ QAP     │       │ PK      │       │ VK      │           │
│  │   +     │──PK──→│   +     │─Proof→│   +     │──bool     │
│  │ RNG     │       │ witness │       │ public  │           │
│  │         │──VK──→│   +     │       │ inputs  │           │
│  └─────────┘       │ RNG     │       └─────────┘           │
│                    └─────────┘                              │
│                                                              │
│  1회 (오프라인)       매 증명마다          매 검증마다           │
│  ⚠️ toxic waste      비밀 유지            O(1) 시간            │
│                                                              │
│  출력:                출력:               출력:                │
│  - ProvingKey (PK)   - Proof (A, B, C)   - true / false      │
│  - VerifyingKey (VK)                                         │
└─────────────────────────────────────────────────────────────┘
```

---

### Part 1: 핵심 아이디어 — 페어링으로 다항식 관계 검증

#### 왜 페어링인가?

```
QAP의 핵심 등식:
  a(τ)·b(τ) - c(τ) = h(τ)·t(τ)

이것을 검증하려면 τ에서의 값을 알아야 하는데...
τ는 비밀이다! (toxic waste)

해법: 타원곡선 + 페어링

  핵심 성질:
    e(a·G₁, b·G₂) = e(G₁, G₂)^(ab)

  즉, "커브 위에서는 곱셈을 할 수 없지만,
       페어링을 통해 GT에서 곱셈 결과를 확인할 수 있다"

  예시:
    증명자가 [a(τ)]₁ = a(τ)·G₁ 을 보냄
    검증자는 a(τ)를 모르지만,
    e([a(τ)]₁, [b(τ)]₂) = e(G₁, G₂)^(a(τ)·b(τ))
    를 계산할 수 있다!
```

> [!tip] 페어링의 마법
> ```
> "값을 모르면서도 값들의 곱이 올바른지 확인할 수 있다"
>
> 이것이 ZK-SNARK를 가능하게 하는 수학적 핵심이다.
>
> 그런데 단순히 e([a(τ)]₁, [b(τ)]₂) = e([c(τ)]₁, [1]₂) · e([h(τ)]₁, [t(τ)]₂)
> 만으로는 부족하다:
>
>   문제 1: 증명자가 가짜 값을 넣을 수 있다
>   문제 2: witness가 노출될 수 있다 (영지식이 아님)
>   문제 3: A, B, C가 같은 witness에서 나왔는지 보장할 수 없다
>
> → α, β로 "구조적 일관성"을 강제
> → γ, δ로 "public/private 분리"
> → r, s로 "영지식성" 보장
> ```

---

#### 5개의 toxic waste 파라미터

```
τ (tau):    비밀 평가점
  → QAP 다항식을 τ에서 평가 → 커브 포인트로 인코딩
  → Schwartz-Zippel에 의해 하나의 점이면 충분

α (alpha): 지식 계수 (knowledge coefficient)
  → A와 B가 "올바른 구조"로 만들어졌는지 강제
  → 증명자가 a(τ)를 실제로 아는지 보장

β (beta):  교차항 계수
  → A, B, C가 "같은 witness에서" 나왔는지 결합
  → β·a(τ) + α·b(τ) + c(τ) 형태로 교차 검증

γ (gamma): public 구분자
  → 공개 변수의 commitment를 γ로 나눔
  → 검증 방정식에서 e(_, [γ]₂)로 소거

δ (delta): private 구분자
  → 비공개 변수 + h(τ)t(τ) 를 δ로 나눔
  → 검증 방정식에서 e(_, [δ]₂)로 소거
  → 블라인딩 팩터의 분모 역할
```

---

### Part 2: Trusted Setup — 커브 포인트로 인코딩

#### Setup이 하는 일

```
입력: QAP (a_polys, b_polys, c_polys, t, domain, num_instance, num_variables)
출력: ProvingKey (PK), VerifyingKey (VK)

Setup 과정:

  ① toxic waste 생성
     τ, α, β, γ, δ ← Fr* (0이 아닌 랜덤)
     γ_inv = γ⁻¹, δ_inv = δ⁻¹

  ② 기본 커브 포인트 계산
     [α]₁ = α·G₁,  [β]₁ = β·G₁,  [β]₂ = β·G₂
     [δ]₁ = δ·G₁,  [δ]₂ = δ·G₂,  [γ]₂ = γ·G₂

  ③ QAP 다항식 평가 (τ에서)
     각 변수 j = 0, 1, ..., n-1:
       aⱼ(τ), bⱼ(τ), cⱼ(τ) ∈ Fr

  ④ Query 벡터 생성
     a_query[j] = [aⱼ(τ)]₁ = aⱼ(τ)·G₁
     b_g1_query[j] = [bⱼ(τ)]₁
     b_g2_query[j] = [bⱼ(τ)]₂

  ⑤ LC 계산 및 분리
     lcⱼ = β·aⱼ(τ) + α·bⱼ(τ) + cⱼ(τ)

     공개 변수 (j = 0..num_instance):
       ic[j] = [lcⱼ / γ]₁

     비공개 변수 (j = num_instance+1..n-1):
       l_query[j'] = [lcⱼ / δ]₁

  ⑥ h_query 생성
     h_query[i] = [τⁱ · t(τ) / δ]₁    for i = 0, 1, ..., m-2

  ⑦ 사전 계산
     e(α, β) = e([α]₁, [β]₂) → VK에 저장
```

---

#### 변수 인덱싱 규칙

```
witness 벡터 s = [s₀, s₁, ..., sₙ₋₁]

  인덱스 0:           One (상수 1, s₀ = 1)
  인덱스 1..ℓ:        Instance 변수 (공개, ℓ = num_instance)
  인덱스 ℓ+1..n-1:   Witness 변수 (비공개)

  예: CubicCircuit (x³+x+5=y, x=3)
    변수 할당 순서:
      x = alloc_witness(3)   → Witness(0)
      y = alloc_instance(35) → Instance(0)
      t1 = alloc_witness(9)  → Witness(1)
      t2 = alloc_witness(27) → Witness(2)

    witness 벡터: s = [1, 35, 3, 9, 27]
      s[0] = 1   → One
      s[1] = 35  → Instance(0) = y     ← 공개
      s[2] = 3   → Witness(0) = x      ← 비공개
      s[3] = 9   → Witness(1) = t1     ← 비공개
      s[4] = 27  → Witness(2) = t2     ← 비공개

  Groth16에서의 분리:
    num_public = num_instance + 1 = 2  (One + y)
    num_private = n - num_public = 3   (x, t1, t2)

    IC: s[0], s[1]  → 검증자가 아는 값
    L:  s[2], s[3], s[4]  → 증명자만 아는 값
```

> [!important] 왜 public/private를 분리하는가?
> ```
> 검증자는 public_inputs (= Instance 변수)만 안다.
> 검증 방정식에서:
>
>   e(IC_sum, [γ]₂): 공개 변수가 올바르게 포함됐는지 확인
>   e(C, [δ]₂): 비공개 변수 + h(τ)t(τ)가 일관되는지 확인
>
> γ로 나눈 값은 [γ]₂와 페어링하면 γ가 소거
> δ로 나눈 값은 [δ]₂와 페어링하면 δ가 소거
>
> 이것이 Groth16의 "핵심 트릭":
>   public과 private를 별도의 "채널"로 분리하면서도
>   하나의 검증 방정식으로 통합 검증
> ```

---

#### ProvingKey 구조체 분석

```rust
pub struct ProvingKey {
    // 기본 파라미터
    alpha_g1: G1,       // [α]₁ — A 계산의 시작점
    beta_g1: G1,        // [β]₁ — C 내 B_g1 계산용
    beta_g2: G2,        // [β]₂ — B 계산의 시작점
    delta_g1: G1,       // [δ]₁ — 블라인딩 (A, C에 사용)
    delta_g2: G2,       // [δ]₂ — 블라인딩 (B에 사용)

    // 다항식 평가 쿼리
    a_query: Vec<G1>,       // n개: [aⱼ(τ)]₁
    b_g1_query: Vec<G1>,    // n개: [bⱼ(τ)]₁
    b_g2_query: Vec<G2>,    // n개: [bⱼ(τ)]₂

    // private 변수 전용
    l_query: Vec<G1>,       // (n - ℓ - 1)개: [lcⱼ/δ]₁

    // h 다항식 계수용
    h_query: Vec<G1>,       // (m-1)개: [τⁱ·t(τ)/δ]₁

    num_instance: usize,
    num_variables: usize,
}
```

```
각 필드의 역할:

  a_query[j]: A = [α]₁ + Σ wⱼ·a_query[j] + r·[δ]₁
    → 증명자가 A를 계산할 때 사용

  b_g2_query[j]: B = [β]₂ + Σ wⱼ·b_g2_query[j] + s·[δ]₂
    → B ∈ G2 계산에 사용

  b_g1_query[j]: B' = [β]₁ + Σ wⱼ·b_g1_query[j] + s·[δ]₁
    → C 계산에 필요한 B의 G1 버전

  l_query[j']: C += Σ_{private} wⱼ·l_query[j']
    → 비공개 변수의 기여분

  h_query[i]: C += Σ hᵢ·h_query[i]
    → QAP 만족의 증거 (h 다항식)

왜 B를 G1과 G2 둘 다 계산하는가?

  B 자체는 G2 원소 (검증 방정식에서 e(A, B) 필요)
  하지만 C = ... + r·B' 에서 B'는 G1 원소여야 함
    → C ∈ G1이므로, r·(G2 원소)는 계산 불가
    → B의 G1 버전 (B')이 별도로 필요
```

---

#### VerifyingKey 구조체 분석

```rust
pub struct VerifyingKey {
    alpha_beta_gt: Fp12,    // e(α, β)
    gamma_g2: G2,           // [γ]₂
    delta_g2: G2,           // [δ]₂
    ic: Vec<G1>,            // (ℓ+1)개: [lcⱼ/γ]₁
}
```

```
VK가 PK보다 훨씬 작은 이유:

  PK: ~4n+m개의 커브 포인트 (회로 크기에 비례)
  VK: ℓ+4개 (공개 입력 수에 비례)

  예: CubicCircuit (n=5, m=3, ℓ=1)
    PK: 5+5+5+3+2 = 20개의 커브 포인트 + 5개 기본 포인트
    VK: Fp12 1개 + G2 2개 + G1 2개 = 매우 작음

  이것이 "Succinct"의 의미:
    검증자가 필요한 정보량이 회로 크기에 무관
```

> [!tip] e(α, β)를 사전 계산하는 이유
> ```
> 검증 때마다 e(α, β)를 계산하면 페어링 4회 필요
> 사전 계산하면 페어링 3회로 감소
>
> 이 최적화는 안전하다:
>   e(α, β)는 VK에 포함된 "상수"
>   α, β는 이미 삭제됨 — e(α, β)로는 복원 불가 (ECDLP)
> ```

---

### Part 3: Prove — 증명 원소 A, B, C 구성

#### 증명 구조

```
Proof = (A, B, C) where:

  A ∈ G1:  α + a(τ) + r·δ   를 인코딩
  B ∈ G2:  β + b(τ) + s·δ   를 인코딩
  C ∈ G1:  private기여 + h기여 + 블라인딩  을 인코딩

  크기: BN254에서
    G1 점: 2 × 32 = 64 bytes
    G2 점: 2 × 64 = 128 bytes
    총: 64 + 128 + 64 = 256 bytes
```

---

#### A 계산

```
A = [α]₁ + Σⱼ wⱼ · [aⱼ(τ)]₁ + r · [δ]₁

  ┌──────────────────────────────────────────────────┐
  │ [α]₁             ← 구조적 태그 (α가 포함됨을 보장) │
  │ + Σ wⱼ·[aⱼ(τ)]₁  ← witness에 의한 QAP a(τ) 값    │
  │ + r·[δ]₁          ← 블라인딩 (영지식성)            │
  └──────────────────────────────────────────────────┘

  Σ wⱼ·[aⱼ(τ)]₁의 의미:
    = Σ wⱼ · aⱼ(τ) · G₁
    = (Σ wⱼ · aⱼ(τ)) · G₁
    = a(τ) · G₁
    = [a(τ)]₁

  즉, A = [α + a(τ) + rδ]₁
```

```rust
// A 계산 코드
let mut proof_a = pk.alpha_g1;     // [α]₁로 시작
for j in 0..pk.num_variables {
    if !witness[j].is_zero() {     // 0인 값 건너뛰기 (최적화)
        proof_a = proof_a
            + pk.a_query[j].scalar_mul(&witness[j].to_repr());
    }
}
proof_a = proof_a + pk.delta_g1.scalar_mul(&r.to_repr());  // + r·[δ]₁
```

```
수동 추적: CubicCircuit (x=3)
  witness = [1, 35, 3, 9, 27]

  proof_a = [α]₁                       ← 초기값
  j=0: w₀=1,  += 1·[a₀(τ)]₁           ← One 변수 기여
  j=1: w₁=35, += 35·[a₁(τ)]₁          ← y (Instance) 기여
  j=2: w₂=3,  += 3·[a₂(τ)]₁           ← x (Witness) 기여
  j=3: w₃=9,  += 9·[a₃(τ)]₁           ← t1 기여
  j=4: w₄=27, += 27·[a₄(τ)]₁          ← t2 기여
  마지막:     += r·[δ]₁                ← 블라인딩

  결과: A = [α + a(τ) + rδ]₁
```

---

#### B 계산

```
B ∈ G2:
  B = [β]₂ + Σⱼ wⱼ · [bⱼ(τ)]₂ + s · [δ]₂

B' ∈ G1 (C 계산용):
  B' = [β]₁ + Σⱼ wⱼ · [bⱼ(τ)]₁ + s · [δ]₁

왜 B를 두 번 계산하는가?
  B 자체: G2 원소 → 검증에서 e(A, B) 에 사용
  B':     G1 원소 → C = ... + r·B' 에서 G1 덧셈 필요

  G1과 G2 사이의 효율적인 변환은 불가능 (ECDLP)
  → 처음부터 두 버전을 따로 계산
```

---

#### C 계산

```
C = Σ_{j∈private} wⱼ · l_query[j']
  + Σᵢ hᵢ · h_query[i]
  + s·A + r·B' - r·s·[δ]₁

세 부분으로 구성:

  ┌─────────────────────────────────────────────────────────┐
  │ ① 비공개 변수 기여:                                      │
  │    Σ_{j∈private} wⱼ · [(β·aⱼ(τ)+α·bⱼ(τ)+cⱼ(τ))/δ]₁    │
  │    → 비공개 변수의 QAP 관계가 올바름을 인코딩             │
  │                                                          │
  │ ② h(x) 기여:                                             │
  │    Σᵢ hᵢ · [τⁱ·t(τ)/δ]₁ = [h(τ)·t(τ)/δ]₁               │
  │    → QAP 만족의 핵심 증거                                 │
  │                                                          │
  │ ③ 블라인딩:                                               │
  │    s·A + r·B' - r·s·[δ]₁                                 │
  │    → 영지식성 보장 (검증 방정식 유지하면서 랜덤화)         │
  └─────────────────────────────────────────────────────────┘
```

```rust
// C 계산 코드
let num_public = pk.num_instance + 1;
let mut proof_c = G1::identity();

// ① 비공개 변수 기여분
for (idx, j) in (num_public..pk.num_variables).enumerate() {
    if !witness[j].is_zero() {
        proof_c = proof_c
            + pk.l_query[idx].scalar_mul(&witness[j].to_repr());
    }
}

// ② h(x) 기여분
for (i, &h_coeff) in h.coeffs.iter().enumerate() {
    if !h_coeff.is_zero() && i < pk.h_query.len() {
        proof_c = proof_c
            + pk.h_query[i].scalar_mul(&h_coeff.to_repr());
    }
}

// ③ 블라인딩: s·A + r·B' - r·s·[δ]₁
proof_c = proof_c + proof_a.scalar_mul(&s.to_repr());
proof_c = proof_c + b_g1.scalar_mul(&r.to_repr());
let rs = r * s;
proof_c = proof_c + (-pk.delta_g1.scalar_mul(&rs.to_repr()));
```

> [!tip] 블라인딩 항의 역할
> ```
> s·A + r·B' - r·s·[δ]₁
>
> 전개:
>   s·A = s·(α + a(τ) + rδ) = sα + s·a(τ) + rsδ
>   r·B' = r·(β + b(τ) + sδ) = rβ + r·b(τ) + rsδ
>   -rs·δ
>
>   합계 = sα + s·a(τ) + rβ + r·b(τ) + rsδ
>
> 이 항이 검증 방정식에서 e(_, [δ]₂)를 통해 정확히 소거된다.
> 증명자가 선택한 r, s는 검증자에게 전혀 드러나지 않는다.
> ```

---

### Part 4: Verify — 검증 방정식

#### 검증 알고리즘

```
입력: VK, public_inputs = [s₁, ..., sℓ], Proof = (A, B, C)

  ① IC_sum 계산:
     IC_sum = ic[0] + Σⱼ₌₁ˡ sⱼ · ic[j]

     ic[0] = [lc₀/γ]₁  ← One 변수 (항상 1)
     sⱼ·ic[j] = sⱼ·[lcⱼ/γ]₁  ← Instance 변수

  ② 검증 방정식 확인:

     e(A, B) ?= e(α,β) · e(IC_sum, [γ]₂) · e(C, [δ]₂)
     ═══════    ══════   ════════════════   ═══════════
       LHS      상수      공개 입력 검증     나머지 전부

  → true / false 반환
```

```rust
pub fn verify(vk: &VerifyingKey, public_inputs: &[Fr], proof: &Proof) -> bool {
    // IC_sum = ic[0] + Σ public_inputs[j] · ic[j+1]
    let mut ic_sum = vk.ic[0];
    for (j, &input) in public_inputs.iter().enumerate() {
        if !input.is_zero() {
            ic_sum = ic_sum + vk.ic[j + 1].scalar_mul(&input.to_repr());
        }
    }

    // 검증 방정식
    let lhs = pairing(&proof.a, &proof.b);
    let rhs = vk.alpha_beta_gt
        * pairing(&ic_sum, &vk.gamma_g2)
        * pairing(&proof.c, &vk.delta_g2);

    lhs == rhs
}
```

> [!important] 검증의 O(1) 복잡도
> ```
> 연산 분석:
>   - IC_sum 계산: ℓ회 scalar_mul (공개 입력 수에 비례)
>   - 페어링 3회: 상수 시간
>   - Fp12 곱셈 2회: 상수 시간
>
> 공개 입력이 적으면 (보통 1~10개) 실질적으로 O(1)
> 이것이 온체인 검증에 적합한 이유:
>   - Ethereum L1에서 검증: 가스 비용 고정
>   - 회로가 아무리 커도 검증 비용 동일
> ```

---

### Part 5: 검증 방정식 완전 유도

#### 왜 e(A, B) = e(α,β) · e(IC_sum, [γ]₂) · e(C, [δ]₂) 인가?

```
Step 1: A·B 전개

  A = α + a(τ) + rδ
  B = β + b(τ) + sδ

  A·B = (α + a(τ) + rδ)(β + b(τ) + sδ)

  9개의 항으로 전개:
    = αβ                    ... (i)
    + α·b(τ)                ... (ii)
    + α·sδ                  ... (iii)
    + a(τ)·β                ... (iv)
    + a(τ)·b(τ)             ... (v)
    + a(τ)·sδ               ... (vi)
    + rδ·β                  ... (vii)
    + rδ·b(τ)               ... (viii)
    + rδ·sδ = rsδ²          ... (ix)

Step 2: QAP 조건 대입

  항 (v)에서: a(τ)·b(τ) = c(τ) + h(τ)·t(τ)

  → A·B = αβ + α·b(τ) + a(τ)·β + c(τ) + h(τ)·t(τ)
          + αsδ + a(τ)sδ + rβδ + rb(τ)δ + rsδ²

Step 3: 항 재그룹

  ❶ αβ                                     → e(α, β)

  ❷ α·b(τ) + β·a(τ) + c(τ)                → IC + L 기여
     = Σⱼ wⱼ·(α·bⱼ(τ) + β·aⱼ(τ) + cⱼ(τ))
     = Σⱼ wⱼ·lcⱼ

     public 부분: Σ_{j∈pub} wⱼ·lcⱼ         → IC_sum · γ
     private 부분: Σ_{j∈priv} wⱼ·lcⱼ       → C의 일부 · δ

  ❸ h(τ)·t(τ)                              → C의 일부 · δ

  ❹ sα + s·a(τ) + rβ + r·b(τ) + rsδ       → C의 블라인딩 · δ
     (= sA - sδ·r + rB' - rδ·s + rsδ = s·A + r·B' - rs·δ)

Step 4: 페어링에서의 검증

  e(A, B) = e(G₁, G₂)^(AB)

  = e(G₁, G₂)^(αβ)                         ... ❶
  · e(G₁, G₂)^(Σ_{pub} wⱼ·lcⱼ)             ... ❷a
  · e(G₁, G₂)^(Σ_{priv} wⱼ·lcⱼ)            ... ❷b
  · e(G₁, G₂)^(h(τ)·t(τ))                  ... ❸
  · e(G₁, G₂)^(블라인딩·δ)                   ... ❹

Step 5: 구조 매칭

  ❶ = e(α·G₁, β·G₂) = e([α]₁, [β]₂) = alpha_beta_gt  ✓

  ❷a: public 변수
    IC_sum = Σ_{j∈pub} wⱼ · [lcⱼ/γ]₁
    e(IC_sum, [γ]₂) = e(Σ wⱼ·lcⱼ/γ · G₁, γ·G₂)
                     = e(G₁, G₂)^(Σ wⱼ·lcⱼ)  ✓

  ❷b + ❸ + ❹: C에 포함
    C = Σ_{priv} wⱼ·[lcⱼ/δ]₁ + Σ hᵢ·[τⁱt(τ)/δ]₁
        + s·A + r·B' - rs·[δ]₁

    e(C, [δ]₂) = e(G₁, G₂)^(Σ_{priv} wⱼ·lcⱼ + h(τ)t(τ) + 블라인딩·δ)

    블라인딩 부분: (s·A + r·B' - rs·δ)·δ 에서 δ가 소거 안 됨!

    잠깐... 다시 계산:
    e(C, [δ]₂)에서 C의 각 항:
      Σ_{priv} wⱼ·(lcⱼ/δ) → e에서 δ와 만나 δ 소거 → wⱼ·lcⱼ
      Σ hᵢ·(τⁱt(τ)/δ) → δ 소거 → h(τ)t(τ)
      (s·A + r·B' - rs·δ) → δ와 만남

    여기서 (sA + rB' - rsδ)는 이미 G1 원소이므로:
    e((sA + rB' - rsδ)·G₁, δ·G₂)
    = e(G₁, G₂)^((sA + rB' - rsδ)·δ)

    A = α + a(τ) + rδ이므로:
    sA = sα + sa(τ) + rsδ
    B' = β + b(τ) + sδ이므로:
    rB' = rβ + rb(τ) + rsδ
    -rsδ

    합: sα + sa(τ) + rβ + rb(τ) + rsδ

    이것에 δ를 곱하면:
    (sα + sa(τ) + rβ + rb(τ) + rsδ)·δ
    = sαδ + sa(τ)δ + rβδ + rb(τ)δ + rsδ²

    이것은 ❹의 항과 정확히 일치!  ✓

  ∴ e(A, B) = e(α,β) · e(IC_sum, [γ]₂) · e(C, [δ]₂)  ✓✓✓
```

> [!important] 검증 방정식의 의미
> ```
> e(A, B):          "증명자가 제시한 A, B의 곱"
>   =
> e(α, β):          "구조적 일관성 기준값" (상수)
> · e(IC_sum, γ₂):  "공개 입력이 올바르게 포함됨" (검증자가 IC_sum 직접 계산)
> · e(C, δ₂):       "나머지 모든 것(private + h + blinding)이 일관됨"
>
> 이 등식이 성립하려면:
>   1. 증명자가 올바른 witness를 알아야 하고
>   2. QAP가 만족되어야 하고 (h(τ)t(τ) 항)
>   3. α, β가 올바르게 사용되어야 함 (구조적 일관성)
>
> 세 가지 중 하나라도 틀리면 → e(A,B) ≠ RHS
> ```

---

### Part 6: 영지식성 — r, s 블라인딩

#### 왜 증명에서 witness가 드러나지 않는가?

```
만약 r = s = 0이라면:
  A = [α + a(τ)]₁
  B = [β + b(τ)]₂
  C = [Σ_{priv} wⱼ·lcⱼ/δ + h(τ)t(τ)/δ]₁

  → A, B, C에서 a(τ), b(τ) 등이 "고정"됨
  → 같은 witness에 대해 항상 같은 증명
  → 여러 증명을 비교하면 witness 정보 유출 가능!

r, s ≠ 0이면:
  A = [α + a(τ) + rδ]₁         ← r이 매번 다름
  B = [β + b(τ) + sδ]₂         ← s가 매번 다름
  C = ... + s·A + r·B' - rs·δ   ← r, s에 의존

  → 같은 witness라도 매번 다른 (A, B, C) 생성
  → 개별 증명에서 witness 정보를 추출하는 것이 불가능
  → 이것이 "완전 영지식" (perfect zero-knowledge)

테스트로 확인:
  groth16_proof_independence:
    같은 회로, 같은 witness, 다른 rng
    → proof1.A ≠ proof2.A ✓
    → 둘 다 verify = true ✓
```

> [!tip] 영지식의 직관
> ```
> 영지식 증명 = "시뮬레이션 가능"
>
> 만약 악의적인 검증자가 증명을 받았을 때,
> "이 증명은 진짜 witness로 만든 건지,
>  아니면 r, s만 적절히 조작한 건지"
> 를 구별할 수 없어야 한다.
>
> r, s가 랜덤이면:
>   증명 (A, B, C)의 분포가 witness에 무관하게 균일
>   → "시뮬레이터"가 witness 없이도 같은 분포의 가짜 증명 생성 가능
>   → 따라서 진짜 증명에서 witness 정보를 추출하는 것은 불가능
> ```

---

### Part 7: 보안 분석

#### Trusted Setup의 위험성

```
toxic waste를 아는 자가 할 수 있는 일:

  τ를 알면:
    → QAP 다항식을 직접 평가 가능
    → 가짜 h(τ) 구성 가능

  α, β를 알면:
    → 구조적 태그를 위조 가능
    → 잘못된 witness에 대한 유효한 A, B 구성 가능

  δ를 알면:
    → l_query와 h_query의 δ를 소거하고 재구성 가능
    → 임의의 C를 만들 수 있음

  결론: 5개 파라미터 중 하나라도 노출되면 위조 가능

대응책 (프로덕션):
  MPC (Multi-Party Computation) 세레모니:
    - N명의 참여자가 각각 τᵢ, αᵢ, βᵢ, γᵢ, δᵢ를 생성
    - τ = τ₁ · τ₂ · ... · τₙ (곱)
    - N명 중 단 1명이라도 자기 값을 삭제하면 안전
    - "1-of-N 신뢰 모델"
```

#### 건전성 (Soundness)

```
"잘못된 witness로 증명을 만들 수 있는가?"

  잘못된 witness s' ≠ s:
    a'(τ)·b'(τ) - c'(τ) ≠ h'(τ)·t(τ)  (QAP 불만족)

    → compute_h() 에서 나머지가 0이 아님 → None 반환
    → prove 실패

  하지만... 증명자가 QAP를 우회하면?
    → α, β가 이를 방지
    → A에 α가 포함되어야 하므로,
      α를 모르는 증명자는 올바른 A를 구성할 수 없음
    → "지식 추출기"(knowledge extractor)로 형식적으로 증명 가능

테스트로 확인:
  groth16_wrong_witness: prove() == None ✓
  groth16_wrong_public_input: verify() == false ✓
  groth16_tampered_proof_a/b/c: verify() == false ✓
```

---

### Part 8: Setup 코드 분석

```rust
pub fn setup<R: Rng>(qap: &QAP, rng: &mut R) -> (ProvingKey, VerifyingKey) {
    let n = qap.num_variables;
    let m = qap.domain.len();

    // ① toxic waste 생성
    let tau = random_nonzero_fr(rng);
    let alpha = random_nonzero_fr(rng);
    let beta = random_nonzero_fr(rng);
    let gamma = random_nonzero_fr(rng);
    let delta = random_nonzero_fr(rng);

    let gamma_inv = gamma.inv().unwrap();
    let delta_inv = delta.inv().unwrap();
    ...
```

```
random_nonzero_fr의 구현:

  fn random_fr<R: Rng>(rng: &mut R) -> Fr {
      let limbs: [u64; 4] = [rng.gen(), rng.gen(), rng.gen(), rng.gen()];
      Fr::from_raw(limbs)
  }

  Fr::from_raw([u64; 4]):
    → 입력을 Montgomery form으로 변환
    → 내부적으로 (limbs × R²) mod r 계산
    → 어떤 256비트 값이든 유효한 Fr 원소로 변환

  2^256 > r이므로 약간의 bias가 존재하지만
  (2^256 / r ≈ 1.00...): 교육용으로 무시 가능
```

---

#### Fr ↔ scalar_mul 브릿지

```
핵심 문제:
  Fr 원소는 Montgomery form으로 저장
  scalar_mul은 raw (standard) representation을 기대

  예: Fr::from_u64(5)
    내부 저장: 5 × R mod r  (Montgomery form)
    to_repr(): 5를 raw [u64;4]로 반환 → [5, 0, 0, 0]

  브릿지:
    let point = G1::generator().scalar_mul(&fr_value.to_repr());

  to_repr()가 Montgomery → raw 변환을 수행:
    "디-몽고메리화" = 자기 자신 × R⁻¹ mod r

  주의: to_repr()를 빼먹으면 R배 스케일링된 잘못된 값으로 scalar_mul!
```

> [!important] Fr::to_repr()의 중요성
> ```
> Fr 내부: Montgomery form = value × R mod r
> scalar_mul 입력: raw form = value
>
> to_repr() 없이 scalar_mul하면:
>   실제 계산: (value × R) · G  (R배 틀림!)
>
> to_repr()로 변환 후:
>   실제 계산: value · G  (올바름)
>
> 이 변환은 Groth16의 모든 scalar_mul에서 필수적이다.
> ```

---

#### IC vs L 분리 코드

```rust
let num_public = qap.num_instance + 1;

// IC: 공개 변수 (j = 0..=num_instance)
let mut ic = Vec::with_capacity(num_public);
for j in 0..num_public {
    let lc = beta * a_at_tau[j] + alpha * b_at_tau[j] + c_at_tau[j];
    let val = lc * gamma_inv;    // ÷ γ
    ic.push(g1.scalar_mul(&val.to_repr()));
}

// L: 비공개 변수 (j = num_instance+1..n-1)
let num_private = n - num_public;
let mut l_query = Vec::with_capacity(num_private);
for j in num_public..n {
    let lc = beta * a_at_tau[j] + alpha * b_at_tau[j] + c_at_tau[j];
    let val = lc * delta_inv;    // ÷ δ
    l_query.push(g1.scalar_mul(&val.to_repr()));
}
```

```
수동 추적: CubicCircuit (n=5, num_instance=1)

  num_public = 2  (One, y)
  num_private = 3 (x, t1, t2)

  IC (2개):
    ic[0]: j=0 (One)
      lc₀ = β·a₀(τ) + α·b₀(τ) + c₀(τ)
      ic[0] = [lc₀ / γ]₁

    ic[1]: j=1 (y = Instance(0))
      lc₁ = β·a₁(τ) + α·b₁(τ) + c₁(τ)
      ic[1] = [lc₁ / γ]₁

  L (3개):
    l_query[0]: j=2 (x = Witness(0))
      lc₂ = β·a₂(τ) + α·b₂(τ) + c₂(τ)
      l_query[0] = [lc₂ / δ]₁

    l_query[1]: j=3 (t1 = Witness(1))
      l_query[1] = [lc₃ / δ]₁

    l_query[2]: j=4 (t2 = Witness(2))
      l_query[2] = [lc₄ / δ]₁
```

---

#### h_query 생성 코드

```rust
let t_at_tau = qap.t.eval(tau);
let t_delta_inv = t_at_tau * delta_inv;

let h_len = m.saturating_sub(1);
let mut h_query = Vec::with_capacity(h_len);
let mut tau_power = Fr::ONE;
for _i in 0..h_len {
    let val = tau_power * t_delta_inv;
    h_query.push(g1.scalar_mul(&val.to_repr()));
    tau_power = tau_power * tau;
}
```

```
h_query의 의미:

  h(x) = h₀ + h₁x + h₂x² + ... + h_{m-2}x^{m-2}

  h_query[i] = [τⁱ · t(τ) / δ]₁

  증명에서:
    Σᵢ hᵢ · h_query[i]
    = Σᵢ hᵢ · [τⁱ · t(τ) / δ]₁
    = [(Σᵢ hᵢ · τⁱ) · t(τ) / δ]₁
    = [h(τ) · t(τ) / δ]₁

  검증에서:
    e([h(τ)·t(τ)/δ]₁, [δ]₂) = e(G₁, G₂)^(h(τ)·t(τ))

  → δ가 소거되어 QAP 항등식의 h(τ)·t(τ) 항이 정확히 복원됨

h_query 길이:

  m = 0: h_len = 0 (제약 없음 → h = 0)
  m = 1: h_len = 0 (단일 제약 → h = 0)
  m = 2: h_len = 1 (h는 상수)
  m = 3: h_len = 2 (h는 1차)
  m = k: h_len = k-1

  일반적으로 deg(h) ≤ 2(m-1) - m = m-2
  → 계수 개수 = m-1
```

---

### Part 9: 수동 추적 — CubicCircuit 전체 과정

#### 회로 정의

```
f(x) = x³ + x + 5 = y

R1CS (3 제약):
  제약 1: x · x = t1
  제약 2: t1 · x = t2
  제약 3: (t2 + x + 5) · 1 = y

x = 3일 때:
  t1 = 9, t2 = 27, y = 27 + 3 + 5 = 35

변수: [One=1, y=35, x=3, t1=9, t2=27]
  인덱스: [0, 1, 2, 3, 4]
  분류:   [pub, pub, priv, priv, priv]
```

#### QAP 다항식 (이전 Step에서 계산)

```
도메인: D = {1, 2, 3}  (ω₁=1, ω₂=2, ω₃=3)

R1CS 행렬:
  제약 1 (ω₁=1): A=[0,0,1,0,0] B=[0,0,1,0,0] C=[0,0,0,1,0]
  제약 2 (ω₂=2): A=[0,0,0,1,0] B=[0,0,1,0,0] C=[0,0,0,0,1]
  제약 3 (ω₃=3): A=[5,0,1,0,1] B=[1,0,0,0,0] C=[0,1,0,0,0]

열별 보간으로 aⱼ(x), bⱼ(x), cⱼ(x) 생성:
  각각 degree ≤ 2 (3점 보간이므로)

소거 다항식:
  t(x) = (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
```

#### Setup 추적

```
toxic waste (가상 값, 실제로는 랜덤):
  τ = 42,  α = 7,  β = 11,  γ = 13,  δ = 17

① 다항식 평가 (τ=42에서):
  각 aⱼ(42), bⱼ(42), cⱼ(42) 계산
  → Fr 원소 5×3 = 15개

② lcⱼ 계산:
  lc₀ = β·a₀(42) + α·b₀(42) + c₀(42) = 11·a₀(42) + 7·b₀(42) + c₀(42)
  lc₁ = 11·a₁(42) + 7·b₁(42) + c₁(42)
  ...

③ IC와 L 분리:
  ic[0] = [lc₀ / 13]₁   (One)
  ic[1] = [lc₁ / 13]₁   (y)
  l_query[0] = [lc₂ / 17]₁  (x)
  l_query[1] = [lc₃ / 17]₁  (t1)
  l_query[2] = [lc₄ / 17]₁  (t2)

④ h_query:
  t(42) = 42³ - 6·42² + 11·42 - 6 = 74088 - 10584 + 462 - 6 = 63960
  t_delta_inv = 63960 / 17

  h_query[0] = [1 · t(42)/17]₁ = [63960/17]₁
  h_query[1] = [42 · t(42)/17]₁ = [42 · 63960/17]₁

⑤ e(α, β) 사전 계산:
  alpha_beta_gt = e([7]₁, [11]₂)
```

#### Prove 추적

```
witness = [1, 35, 3, 9, 27]
h(x) = QAP에서 계산 (degree ≤ 1)
r, s = 랜덤

A 계산:
  A = [α]₁
    + 1·[a₀(τ)]₁ + 35·[a₁(τ)]₁ + 3·[a₂(τ)]₁ + 9·[a₃(τ)]₁ + 27·[a₄(τ)]₁
    + r·[δ]₁
  = [α + a(τ) + rδ]₁

B 계산:
  B = [β]₂
    + 1·[b₀(τ)]₂ + 35·[b₁(τ)]₂ + 3·[b₂(τ)]₂ + 9·[b₃(τ)]₂ + 27·[b₄(τ)]₂
    + s·[δ]₂
  = [β + b(τ) + sδ]₂

C 계산:
  private_sum = 3·l_query[0] + 9·l_query[1] + 27·l_query[2]
  h_sum = h₀·h_query[0] + h₁·h_query[1]
  blinding = s·A + r·B' - rs·[δ]₁
  C = private_sum + h_sum + blinding
```

#### Verify 추적

```
public_inputs = [35]  (y의 값)

IC_sum = ic[0] + 35·ic[1]
       = [lc₀/γ]₁ + 35·[lc₁/γ]₁

LHS = e(A, B)
RHS = e(α,β) · e(IC_sum, [γ]₂) · e(C, [δ]₂)

LHS == RHS ?  → true ✓
```

---

### Part 10: Edge Case 분석

#### m = 0 (제약 없는 회로)

```
상황: ConstraintSystem::new()만 호출, enforce 없음
  → witness = [1], num_instance = 0, num_variables = 1

QAP:
  domain = []  (비어 있음)
  t(x) = 1  (빈 곱 = 1)
  a₀(x) = 0, b₀(x) = 0, c₀(x) = 0  (제약이 없으므로 전부 0)
  h(x) = (0·0 - 0) / 1 = 0

Setup:
  a_query = [identity]  (0·G₁ = O)
  IC: ic[0] = [(0+0+0)/γ]₁ = O
  L: 비어 있음
  h_query: 비어 있음

Prove:
  A = [α]₁ + 1·O + r·[δ]₁ = [α + rδ]₁
  B = [β]₂ + 1·O + s·[δ]₂ = [β + sδ]₂
  C = O + s·A + r·B' - rs·[δ]₁
    = [sα + rβ + rsδ]₁

Verify:
  IC_sum = O (identity)
  e(O, [γ]₂) = 1 (identity in GT)

  LHS = e(α+rδ, β+sδ)
      = e^(αβ + αsδ + rβδ + rsδ²)

  RHS = e(α,β) · 1 · e(sα+rβ+rsδ, δ)
      = e^(αβ) · e^(sαδ + rβδ + rsδ²)
      = e^(αβ + sαδ + rβδ + rsδ²)

  LHS = RHS ✓
```

> [!tip] 빈 회로도 유효한 증명이 존재
> ```
> 제약이 없으므로 "항상 참"인 증명.
> 이것은 프로토콜의 일관성을 보여준다:
>   0개의 제약을 만족 → 증명 가능 → 검증 통과
>
> 실용적 의미:
>   "아무것도 증명하지 않는 증명"이 가능하다는 것은
>   프로토콜의 수학적 구조가 건전하다는 반증.
> ```

---

#### m = 1 (단일 제약)

```
상황: x · y = z (제약 1개)
  witness = [1, 12, 3, 4]  (One=1, z=12, x=3, y=4)

QAP:
  domain = {1}
  t(x) = x - 1
  모든 aⱼ, bⱼ, cⱼ: degree 0 (상수)
  a(x) = 3 (상수), b(x) = 4, c(x) = 12
  a·b - c = 12 - 12 = 0
  h = 0 / t = 0

특이점:
  h_query.len() = max(0, 1-1) = 0
  → h 기여분이 없음
  → C는 private_sum + blinding만으로 구성

이것이 맞는 이유:
  단일 제약이면 a, b, c가 모두 상수
  → a·b = c가 어디서든 성립 (도메인 검사 1번으로 충분)
  → h = 0: t(x)로 나눌 "잉여"가 없음
```

---

#### witness에 0이 포함된 경우

```
상황: 0 · 5 = 0  (x=0, y=5, z=0)
  witness = [1, 0, 0, 5]

코드에서의 처리:
  for j in 0..pk.num_variables {
      if !witness[j].is_zero() {    ← 0인 값 건너뛰기
          proof_a = proof_a + pk.a_query[j].scalar_mul(&witness[j].to_repr());
      }
  }

  j=0: w₀=1  → 포함 (One은 항상 1)
  j=1: w₁=0  → 건너뜀 (z=0)
  j=2: w₂=0  → 건너뜀 (x=0)
  j=3: w₃=5  → 포함 (y=5)

수학적으로:
  0 · [aⱼ(τ)]₁ = O (항등원)
  → 더해도 결과 변화 없음
  → 건너뛰기는 순수한 최적화 (결과 동일)

테스트: groth16_zero_witness_values ✓
```

---

### Part 11: 랜덤 Fr 생성

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
랜덤 Fr 생성의 동작:

  1. 4개의 랜덤 u64 생성 → 256비트 정수
  2. Fr::from_raw()로 Montgomery form 변환
     내부: (limbs × R²) mod r

  편향성 분석:
    2^256 = r × q + rem  (q ≈ 1, rem 작음)
    편향: ≤ 1/2^254 ≈ 10^(-77)
    → 교육용으로 무시 가능

  프로덕션: 256비트 중 상위 2비트를 0으로 고정 후
            rejection sampling으로 균일 분포 보장

왜 nonzero인가?
  γ⁻¹, δ⁻¹을 계산해야 하므로 0이면 역원이 없다.
  Fr::ZERO가 나올 확률: 1/r ≈ 10^(-77) → 사실상 불가능
  하지만 loop으로 안전하게 처리.
```

---

### Part 12: Groth16 vs 다른 증명 시스템

```
┌──────────────┬─────────┬──────────┬────────────┬──────────────┐
│              │ Groth16 │ PLONK    │ STARK      │ Bulletproofs │
├──────────────┼─────────┼──────────┼────────────┼──────────────┤
│ 증명 크기     │ 256 B   │ ~4 KB    │ ~100 KB    │ ~700 B       │
│ 검증 시간     │ O(1)    │ O(1)     │ O(log²n)   │ O(n)         │
│ Trusted Setup│ 회로별   │ 범용     │ 불필요      │ 불필요       │
│ 양자 내성     │ ✗       │ ✗        │ ✓          │ ✗            │
│ 대수 구조     │ 페어링   │ 페어링    │ 해시       │ 이산로그     │
│ 주요 용도     │ zk-rollup│ 범용 ZK  │ zkVM       │ 기밀 트랜잭션│
└──────────────┴─────────┴──────────┴────────────┴──────────────┘

Groth16의 장점:
  - 가장 작은 증명 크기
  - 가장 빠른 검증 (온체인 검증에 최적)
  - 잘 검증된 보안 (2016년 이후 다수의 프로덕션 배포)

Groth16의 단점:
  - 회로별 trusted setup 필요
  - 회로 변경 시 새 setup 필요
  - 양자 컴퓨터에 취약 (페어링 기반)
```

---

### Part 13: 프로덕션 Groth16과의 차이

```
우리 구현 (교육용):
  - O(n²) 다항식 연산 (schoolbook)
  - 개별 scalar_mul (O(n) 곱셈)
  - 단순 랜덤 생성 (약간의 편향)
  - 비최적화 페어링 (모든 페어링 개별 계산)

프로덕션 (arkworks, bellman 등):
  - FFT 기반 O(n log n) 다항식 연산
  - MSM (Multi-Scalar Multiplication): Pippenger 알고리즘
  - 밀러 루프 배칭: 여러 페어링을 동시 계산
  - 증명 병렬화 (A, B, C를 병렬 계산)
  - 랜덤: OS 엔트로피 + rejection sampling

성능 차이:
  n = 10,000 (보통의 DeFi 회로):
    교육용: 수 분
    arkworks: 수 초

하지만 수학적 구조는 100% 동일하다.
우리의 구현으로 Groth16의 모든 본질을 이해할 수 있다.
```

---

### Part 14: 테스트 전략

```
13개의 테스트로 Groth16의 모든 측면을 커버:

기본 동작 (2):
  groth16_cubic_valid:           3차 회로, x=3 → y=35
  groth16_cubic_different_input: 3차 회로, x=5 → y=135

보안 검증 (5):
  groth16_wrong_public_input:  틀린 공개 입력 → verify false
  groth16_wrong_witness:       틀린 witness → prove None
  groth16_tampered_proof_a:    A 변조 → verify false
  groth16_tampered_proof_b:    B 변조 → verify false
  groth16_tampered_proof_c:    C 변조 → verify false

다양한 회로 (3):
  groth16_multiply_circuit:    x·y=z (m=1)
  groth16_pythagorean:         x²+y²=z² (m=3)
  groth16_conditional:         if b then x else y (m=3)

영지식성 (1):
  groth16_proof_independence:  같은 witness, 다른 rng → 다른 증명

Edge case (2):
  groth16_zero_witness_values: witness에 0 포함
  groth16_empty_circuit:       제약 0개 (m=0)
```

---

### Part 15: 전체 파이프라인 요약

```
┌──────────────────────────────────────────────────────────────┐
│            프로그램 → 증명 → 검증 전체 파이프라인               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  프로그램:  f(x) = x³ + x + 5 = y                             │
│     ↓                                                         │
│  R1CS:  3개의 제약                                             │
│     ↓  (Step 09)                                              │
│  QAP:  다항식 aⱼ(x), bⱼ(x), cⱼ(x), t(x)                     │
│     ↓  (Step 11)                                              │
│  Setup:  QAP + toxic waste → PK, VK                           │
│     ↓  (Step 12, 1회만)                                       │
│  Prove:  PK + witness + rng → Proof(A, B, C)                  │
│     ↓  (Step 12, 매 증명마다)                                  │
│  Verify: VK + public_inputs + Proof → true/false              │
│     ↓  (Step 12, 매 검증마다)                                  │
│                                                               │
│  "나는 x³ + x + 5 = 35를 만족하는 x를 안다"                     │
│  → x=3이라는 것을 밝히지 않고 검증자를 설득                     │
│                                                               │
│  증명 크기: 256 bytes                                          │
│  검증 시간: 페어링 3회 = O(1)                                   │
│  보안: QAP 만족 + 구조적 일관성 + 영지식성                      │
│                                                               │
│  기존 227개 테스트 + Groth16 13개 = 240개 모두 통과 ✅           │
└──────────────────────────────────────────────────────────────┘
```

> [!important] 프로그램 → R1CS → QAP → Groth16: 완결
> ```
> Step 09 (R1CS): 프로그램을 곱셈 제약으로 분해
> Step 10 (가젯): 반복 패턴을 재사용 가능한 회로로
> Step 11 (QAP):  m개의 제약을 하나의 다항식 항등식으로 압축
> Step 12 (Groth16): 다항식 항등식을 페어링으로 비밀 유지 검증
>
> 이 4단계가 Groth16 ZK-SNARK의 전부이다.
> 다음 스텝(PLONK)부터는 trusted setup 없는 범용 프로토콜로 진화한다.
> ```

---

### Part 16: 변조 증명이 실패하는 수학적 이유

#### proof.a 변조

```
정상 증명: (A, B, C) — verify 통과

변조: A' = A + G₁  (생성자를 더함)

검증 방정식:
  LHS' = e(A', B) = e(A + G₁, B)
       = e(A, B) · e(G₁, B)      ← 쌍선형성
       = LHS · e(G₁, B)

  RHS는 변하지 않음 (A가 RHS에 나타나지 않으므로)

  LHS' = LHS · e(G₁, B) ≠ LHS = RHS

  e(G₁, B) = 1 이 되려면 B = O (항등원) 이어야 함
  정상 증명에서 B ≠ O (β, b(τ) 등이 포함)

  ∴ LHS' ≠ RHS → verify false ✓
```

#### proof.b 변조

```
변조: B' = B + G₂

  LHS' = e(A, B') = e(A, B + G₂)
       = e(A, B) · e(A, G₂)
       = LHS · e(A, G₂)

  e(A, G₂) = 1 이 되려면 A = O
  정상 증명에서 A ≠ O

  ∴ LHS' ≠ RHS → verify false ✓
```

#### proof.c 변조

```
변조: C' = C + G₁

  LHS는 변하지 않음 (C가 LHS에 나타나지 않으므로)

  RHS' = e(α,β) · e(IC_sum, [γ]₂) · e(C', [δ]₂)
       = e(α,β) · e(IC_sum, [γ]₂) · e(C + G₁, [δ]₂)
       = e(α,β) · e(IC_sum, [γ]₂) · e(C, [δ]₂) · e(G₁, [δ]₂)
       = RHS · e(G₁, [δ]₂)

  e(G₁, [δ]₂) = e(G₁, G₂)^δ ≠ 1 (δ ≠ 0이므로)

  ∴ LHS = RHS ≠ RHS' → verify false ✓
```

> [!tip] 변조 감지의 수학적 원리
> ```
> 페어링의 쌍선형성이 "변조 증폭기" 역할을 한다:
>
>   e(A + Δ, B) = e(A, B) · e(Δ, B)
>
> 아무리 작은 Δ(=G₁)를 더해도:
>   e(Δ, B) ≠ 1 (B ≠ O인 한)
>
> → 변조가 GT에서 "배가"되어 검증 방정식을 깨뜨린다
> → 이것이 Groth16의 "건전성"(soundness)의 수학적 근거
> ```

---

### Part 17: α, β의 역할 — 지식 추출 (Knowledge Extraction)

#### 구조적 일관성 문제

```
만약 α, β 없이 프로토콜을 설계한다면:

  A = [a(τ)]₁
  B = [b(τ)]₂
  C에 h(τ)t(τ)를 포함

  → 증명자가 "가짜" a'(τ), b'(τ)를 만들 수 있다!

  예: a'(τ)·b'(τ) - c(τ) ≠ h(τ)·t(τ) 이지만,
      검증 방정식에 맞도록 A, B, C를 조작

  공격:
    A = [x]₁ (임의의 x)
    B = [y]₂ (임의의 y)
    C를 적절히 조작하여 e(A,B) = RHS 만족

α, β가 있으면:
  A = [α + a(τ) + rδ]₁

  α를 모르는 공격자는:
    "α가 포함된 올바른 A"를 만들 수 없다
    → A에 α가 없으면 검증 방정식 불만족

  이것을 형식적으로 증명하는 기법: "지식 추출기" (knowledge extractor)
    → "A를 만들 수 있다면, 그 안에 a(τ)가 들어있다"는 것을 수학적으로 보장
```

#### β의 교차 결합 역할

```
β가 없다면:
  A, B, C를 "각각 독립적으로" 구성 가능
  → A는 올바르지만 B는 다른 witness에서 온 것일 수 있음

β가 있으면:
  lcⱼ = β·aⱼ(τ) + α·bⱼ(τ) + cⱼ(τ)

  이 항은 A, B, C가 "같은 다항식"에서 나왔을 때만 올바르게 결합됨

  직관:
    β·aⱼ: A에서 온 정보
    α·bⱼ: B에서 온 정보
    cⱼ:   C에서 온 정보

    세 가지가 하나의 lcⱼ로 합쳐짐
    → A, B, C가 불일치하면 lcⱼ가 깨짐
    → 검증 방정식 불만족
```

---

### Part 18: γ와 δ의 분리 원리 — 두 개의 "채널"

```
왜 γ와 δ 두 개가 필요한가?

핵심 문제:
  공개 변수와 비공개 변수를 "분리"해야 한다.

  만약 하나의 분모 (예: δ)만 사용한다면:
    IC_sum과 C가 "같은 채널"에 있게 됨
    → 증명자가 IC_sum 부분을 조작하여 C를 보상할 수 있음!

  두 개의 분모 (γ, δ):
    공개 변수: lcⱼ/γ → e(_, [γ]₂)로 소거
    비공개 변수: lcⱼ/δ → e(_, [δ]₂)로 소거

    γ로 나눈 값은 [δ]₂와 페어링해도 γ/δ가 남아서 소거 불가
    → 두 채널이 완전히 독립

비유:
  γ 채널 = 공개 채널 (검증자가 IC_sum 직접 계산)
  δ 채널 = 비공개 채널 (증명에 묶여 있음)

  증명자가 공개 채널의 값을 바꾸면:
    → IC_sum이 달라짐 → 검증자가 직접 감지
  증명자가 비공개 채널의 값을 바꾸면:
    → C가 달라짐 → e(C, [δ]₂)가 달라짐 → LHS ≠ RHS
```

---

### Part 19: Prove 함수 상세 코드 분석

```rust
pub fn prove<R: Rng>(
    pk: &ProvingKey,
    qap: &QAP,
    witness: &[Fr],
    rng: &mut R,
) -> Option<Proof> {
    assert_eq!(witness.len(), pk.num_variables);
```

```
함수 시그니처 분석:

  &ProvingKey: Setup에서 생성된 키 (빌려옴, 소유권 이전 안 함)
  &QAP: h(x) 계산에 필요 (compute_h 호출)
  &[Fr]: witness 슬라이스 (cs.values)
  &mut R: 랜덤 생성기 (r, s 생성용)
  → Option<Proof>: QAP 불만족이면 None

assert: witness 길이 검증
  → 회로와 무관한 witness를 넣으면 즉시 패닉
  → 디버깅을 위한 안전장치
```

```rust
    // h(x) 계산 — QAP 불만족이면 None
    let h = qap.compute_h(witness)?;
```

```
compute_h 내부 동작:
  1. compute_witness_polys(witness):
     a(x) = Σⱼ wⱼ·aⱼ(x)
     b(x) = Σⱼ wⱼ·bⱼ(x)
     c(x) = Σⱼ wⱼ·cⱼ(x)

  2. p(x) = a(x)·b(x) - c(x)

  3. (h, rem) = p.div_rem(&t)

  4. rem이 0이면 Some(h), 아니면 None

  ? 연산자:
    None이면 → 즉시 None 반환 (증명 불가)
    Some(h)이면 → h에 바인딩

  실패하는 경우:
    잘못된 witness → R1CS 불만족 → a(ωᵢ)·b(ωᵢ) ≠ c(ωᵢ)
    → p(x)가 t(x)로 나누어떨어지지 않음
    → 나머지 ≠ 0 → None
```

```rust
    let r = random_fr(rng);
    let s = random_fr(rng);
```

```
r과 s는 0이어도 괜찮은가?

  r = 0이면: A = [α + a(τ)]₁  (블라인딩 없음)
  s = 0이면: B = [β + b(τ)]₂  (블라인딩 없음)

  검증 방정식은 여전히 성립! (블라인딩 항이 0이 될 뿐)

  하지만 영지식성이 약해진다:
    r = s = 0이면 같은 witness에 대해 항상 같은 증명
    → 여러 증명 비교로 정보 유출 가능

  실제로 r, s가 0일 확률: 1/r ≈ 10^(-77) → 무시 가능

  nonzero를 강제하지 않는 이유:
    - 0이어도 "약간" 영지식하지 않을 뿐, 건전성은 유지
    - 불필요한 loop 방지
    - 확률적으로 발생하지 않음
```

---

### Part 20: Verify 함수 상세 코드 분석

```rust
pub fn verify(vk: &VerifyingKey, public_inputs: &[Fr], proof: &Proof) -> bool {
    assert_eq!(public_inputs.len() + 1, vk.ic.len());
```

```
assert 분석:
  public_inputs.len() = num_instance
  vk.ic.len() = num_instance + 1  (One 포함)

  +1은 ic[0]이 One 변수(항상 1)에 대응하기 때문

  예: CubicCircuit
    public_inputs = [35]  (y값만)
    vk.ic = [ic₀, ic₁]   (One, y)
    assert: 1 + 1 = 2 ✓
```

```rust
    let mut ic_sum = vk.ic[0];
    for (j, &input) in public_inputs.iter().enumerate() {
        if !input.is_zero() {
            ic_sum = ic_sum + vk.ic[j + 1].scalar_mul(&input.to_repr());
        }
    }
```

```
IC_sum 계산의 의미:

  ic_sum = ic[0] + s₁·ic[1] + s₂·ic[2] + ...

  = [lc₀/γ]₁ + s₁·[lc₁/γ]₁ + s₂·[lc₂/γ]₁ + ...

  = [(lc₀ + s₁·lc₁ + s₂·lc₂ + ...)/γ]₁

  = [(Σ_{j∈pub} wⱼ · lcⱼ) / γ]₁

  여기서 w₀ = 1 (One), w₁ = s₁ (public_inputs[0]), etc.

  이것을 [γ]₂와 페어링하면:
    e(IC_sum, [γ]₂) = e(G₁, G₂)^(Σ_{pub} wⱼ · lcⱼ)
    → γ가 소거됨!
```

```rust
    let lhs = pairing(&proof.a, &proof.b);
    let rhs = vk.alpha_beta_gt
        * pairing(&ic_sum, &vk.gamma_g2)
        * pairing(&proof.c, &vk.delta_g2);
    lhs == rhs
```

```
페어링 3회의 연산 비용:

  pairing 1: e(A, B) — Miller loop + final exp
  pairing 2: e(IC_sum, [γ]₂) — Miller loop + final exp
  pairing 3: e(C, [δ]₂) — Miller loop + final exp

  각 페어링:
    Miller loop: O(log p) 곱셈 (BN254: ~64 iterations)
    Final exp: f^((p^12-1)/r)
    → BN254에서 약 1-5ms (최적화 시)

  총 검증 시간: ~5-15ms (최적화 구현)

  Fp12 곱셈 2회: vk.alpha_beta_gt * pairing₂ * pairing₃
    → Fp12 곱셈은 페어링에 비해 무시할 수 있을 정도로 빠름

  최적화 가능:
    batched pairing: 3개 페어링을 하나의 final exp로 통합
    → 실질적으로 Miller loop 3회 + final exp 1회
    → ~30% 속도 향상
```

---

### Part 21: 수동 추적 — 피타고라스 회로

```
회로: x² + y² = z²  (3 제약)
  x=3, y=4, z=5

변수 할당:
  x = alloc_witness(3)   → Witness(0)
  y = alloc_witness(4)   → Witness(1)
  z = alloc_instance(5)  → Instance(0)
  x_sq = alloc_witness(9)  → Witness(2)
  y_sq = alloc_witness(16) → Witness(3)

witness 벡터: s = [1, 5, 3, 4, 9, 16]
  s[0] = 1   → One
  s[1] = 5   → z (Instance, 공개)
  s[2] = 3   → x (Witness, 비공개)
  s[3] = 4   → y (Witness, 비공개)
  s[4] = 9   → x² (Witness, 비공개)
  s[5] = 16  → y² (Witness, 비공개)

R1CS 제약:
  1: x · x = x_sq        → A=[0,0,1,0,0,0] B=[0,0,1,0,0,0] C=[0,0,0,0,1,0]
  2: y · y = y_sq         → A=[0,0,0,1,0,0] B=[0,0,0,1,0,0] C=[0,0,0,0,0,1]
  3: z · z = x_sq + y_sq  → A=[0,1,0,0,0,0] B=[0,1,0,0,0,0] C=[0,0,0,0,1,1]

QAP:
  도메인 = {1, 2, 3}, m = 3
  t(x) = (x-1)(x-2)(x-3)
  각 aⱼ(x), bⱼ(x), cⱼ(x): degree ≤ 2

Groth16:
  num_public = 2 (One, z)
  num_private = 4 (x, y, x², y²)
  IC: 2개 (ic[0], ic[1])
  L: 4개 (l_query[0]..l_query[3])
  h_query: 2개 (degree(h) ≤ 1)

증명:
  public_inputs = [5]  (z의 값)
  → verify 통과 ✓

검증자가 아는 것: z = 5
검증자가 모르는 것: x = 3, y = 4 (무한히 많은 (x,y) 가능!)
  → 3² + 4² = 25 = 5² 이지만,
     검증자는 어떤 x, y인지 전혀 모른다
  → 이것이 영지식 증명의 핵심!
```

---

### Part 22: "if b then x else y" 회로의 Groth16

```
조건부 선택 회로 (3 제약):
  b=1, x=42, y=99 → result=42

변수 할당:
  b = alloc_witness(1)
  x = alloc_witness(42)
  y = alloc_witness(99)
  t = alloc_witness(-57)  // t = b*(x-y) = 1*(42-99) = -57
  result = alloc_instance(42)  // y + t = 99 + (-57) = 42

제약:
  1: b · (1-b) = 0        ← 부울 제약 (b ∈ {0, 1})
  2: b · (x-y) = t        ← 조건부 차이
  3: (y+t) · 1 = result   ← 결과

영지식의 의미:
  검증자가 아는 것: result = 42
  검증자가 모르는 것: b, x, y
    → "42가 x인지 y인지 모른다"
    → "b가 0인지 1인지 모른다"
    → 조건의 분기가 비밀로 유지됨

이것이 프라이버시 보존 스마트 컨트랙트의 기초:
  "어떤 조건을 선택했는지 밝히지 않고 결과만 공개"
```

---

### Part 23: Groth16 구현의 전체 구조

```
groth16.rs (356줄):

  줄   1- 45: 모듈 설명 + 검증 방정식 유도 (주석)
  줄  47- 50: import (Fr, Fp12, G1, G2, pairing, QAP, Rng)
  줄  54- 72: 랜덤 Fr 생성 (random_fr, random_nonzero_fr)
  줄  76-119: ProvingKey 구조체
  줄 123-137: VerifyingKey 구조체
  줄 141-152: Proof 구조체
  줄 156-236: setup() 함수
  줄 240-330: prove() 함수
  줄 334-365: verify() 함수
  줄 369-end: 테스트 (13개)

의존성 그래프:
  groth16.rs
    ├── field/fr.rs: Fr (스칼라체)
    ├── field/fp12.rs: Fp12 (GT)
    ├── curve/g1.rs: G1 (증명 원소 A, C)
    ├── curve/g2.rs: G2 (증명 원소 B)
    ├── curve/pairing.rs: pairing() (검증)
    └── qap.rs: QAP, Polynomial (h(x) 계산)

lib.rs:
  pub mod groth16;  ← 추가

기존 코드 재사용:
  새로 구현한 것: setup, prove, verify, 3개 struct
  재사용한 것: Fr, G1, G2, Fp12, pairing, QAP, Polynomial,
              ConstraintSystem, Circuit, LinearCombination, Variable
```

---

### Part 24: 검증 방정식의 다른 관점 — 행렬 형태

```
Groth16 검증을 행렬 관점에서 보면:

  QAP: a(τ)·b(τ) - c(τ) = h(τ)·t(τ)

  Groth16: 이것을 3개의 "독립적인 채널"로 분리

  ┌──────────────────────────────────────────┐
  │ 채널 1: e(α, β)                          │
  │   → "기준값" (상수)                       │
  │   → 항상 같은 값                          │
  │                                          │
  │ 채널 2: e(IC_sum, [γ]₂)                   │
  │   → 공개 입력의 기여분                    │
  │   → 검증자가 IC_sum을 직접 계산            │
  │   → γ로 보호 (공개 채널)                   │
  │                                          │
  │ 채널 3: e(C, [δ]₂)                        │
  │   → 비공개 기여 + h(τ)t(τ) + 블라인딩      │
  │   → δ로 보호 (비공개 채널)                 │
  │   → 증명자만 구성 가능                    │
  └──────────────────────────────────────────┘

  세 채널의 곱 = e(A, B)

  이 구조가 보장하는 것:
    1. A·B의 "총합"이 맞다 (건전성)
    2. 공개 부분은 검증자가 직접 확인 (공개 입력 정확성)
    3. 비공개 부분은 δ 뒤에 숨겨짐 (영지식성)
    4. α, β가 구조적 일관성 강제 (지식 건전성)
```

---

### Part 25: Groth16 논문 표기법과의 대응

```
Groth16 원논문 [Groth, 2016] 표기법  ↔  우리 구현

  논문                        구현
  ────────────────────────   ──────────────────
  σ = (α, β, γ, δ, {τⁱ})    toxic waste
  [a]₁ = a·G₁               g1.scalar_mul(&a.to_repr())
  [b]₂ = b·G₂               g2.scalar_mul(&b.to_repr())
  uⱼ, vⱼ, wⱼ               aⱼ, bⱼ, cⱼ (QAP 다항식)
  t(x)                       qap.t (소거 다항식)
  aₓ (public inputs)         public_inputs[j]
  ∑ aⱼ uⱼ(τ)                a(τ) = Σ wⱼ·aⱼ(τ)
  π = (A, B, C)              Proof { a, b, c }

  검증 방정식 (논문):
    e(A, B) = e([α]₁, [β]₂) · e(∑ aⱼ [Γⱼ]₁, [γ]₂) · e(C, [δ]₂)

  검증 방정식 (우리):
    e(A, B) = alpha_beta_gt · e(IC_sum, gamma_g2) · e(C, delta_g2)

  Γⱼ (논문) = lcⱼ/γ = (β·uⱼ(τ) + α·vⱼ(τ) + wⱼ(τ))/γ  = ic[j] (우리)
```

---

### Part 26: Trusted Setup 세레모니 심화

```
실제 프로덕션에서의 MPC 세레모니:

Phase 1: "Powers of Tau"
  - 범용 setup: τ의 거듭제곱 [τ⁰]₁, [τ¹]₁, ..., [τᴺ]₁ 생성
  - 회로에 무관하게 한 번만 수행
  - Zcash의 "Powers of Tau" 세레모니: 87명 참여

  참여자 i:
    τᵢ ← 랜덤
    [τ^k]₁ ← [τ^k]₁^τᵢ = [τᵢ · τ^k]₁  (이전 결과에 자기 τᵢ 곱함)
    τᵢ 삭제

  최종: τ = τ₁ · τ₂ · ... · τ₈₇
    87명 중 1명이라도 τᵢ를 삭제했으면 전체 τ는 안전

Phase 2: "회로 특화"
  - 특정 회로의 α, β, γ, δ와 IC, L, h_query 생성
  - Phase 1의 결과를 기반으로 수행

보안 모델:
  "1-of-N 신뢰": N명 중 단 1명만 정직하면 안전

  N을 충분히 크게 하면:
    - 지리적으로 분산 (세계 각지)
    - 이해관계가 다른 참여자 (경쟁사, 학계, 커뮤니티)
    - 물리적으로 다른 환경 (클라우드, 에어갭, 모바일)

  사실상 "한 명이라도 정직할 확률"은 거의 100%

우리 구현에서는?
  단일 rng로 모든 toxic waste 생성
  → 교육용으로는 충분
  → 프로덕션에서는 절대 이렇게 하면 안 됨!
```

---

### Part 27: Groth16의 한계와 PLONK으로의 전환

```
Groth16의 근본적 한계:

  1. 회로별 setup
     - 회로가 바뀌면 새 setup 필요
     - DeFi 프로토콜 업그레이드마다 세레모니 반복
     - 비용과 시간 소모

  2. R1CS 제약
     - 곱셈 1개 = 제약 1개
     - 범위 검사, 해시 등에 많은 제약 필요
     - PLONKish arithmetization이 더 효율적

  3. 양자 취약성
     - 페어링 기반 → 양자 컴퓨터의 Shor 알고리즘에 취약
     - STARK는 해시 기반이므로 양자 내성

PLONK의 장점 (다음 스텝):
  - 범용 setup: 한 번의 setup으로 모든 회로에 사용
  - KZG commitment: 다항식을 직접 commit
  - 더 유연한 게이트 (커스텀 게이트, lookup table)
  - Trusted setup도 더 간단 (Phase 1만 필요)

하지만:
  - 증명 크기: ~4KB (Groth16의 ~16배)
  - 검증 시간: 약간 느림 (하지만 여전히 O(1))

Groth16이 여전히 사용되는 이유:
  - 최소 증명 크기가 중요한 곳 (온체인 비용 최소화)
  - 이미 검증된 코드베이스와 보안 감사
  - L1 검증 비용이 핵심인 zk-rollup
```

> [!important] Groth16 → PLONK 전환의 의미
> ```
> Groth16: "이 특정 회로에 대한 증명"
>   → 회로 ∈ trusted setup
>
> PLONK: "어떤 회로에 대해서든 증명"
>   → 회로 ∉ trusted setup (범용)
>
> 우리의 다음 여정:
>   Step 13: KZG 다항식 commitment (PLONK의 기반)
>   Step 14: PLONKish arithmetization
>   Step 15: Plookup (lookup argument)
>   Step 16: PLONK prover/verifier
> ```

---

### Part 28: 시뮬레이터 구성 — 완전 영지식 (Perfect ZK) 증명

#### 영지식의 형식적 정의

```
영지식 (Zero-Knowledge)의 정의:

  "증명을 받은 검증자가 얻는 정보량 = 0"

  형식적으로:
    임의의 (악의적인) 검증자 V*에 대해,
    V*가 증명으로부터 계산할 수 있는 모든 것을
    witness 없이도 계산할 수 있는 "시뮬레이터" S가 존재

  즉:
    Real(π, V*) ≈ Simulated(S, V*)

  Groth16은 "완전" 영지식 (perfect ZK):
    Real 분포 = Simulated 분포  (통계적으로 동일)
    "≈"가 아니라 "="

왜 "시뮬레이션"이 핵심인가?

  반직관적 논증:
    "만약 시뮬레이터가 witness 없이 진짜와 구별 불가능한
     증명을 만들 수 있다면,
     진짜 증명에서도 witness 정보를 추출하는 것이 불가능하다"

  왜냐하면:
    정보가 추출 가능하다면 → 시뮬레이터의 가짜 증명에서도 추출 가능해야 함
    → 하지만 시뮬레이터는 witness를 모름
    → 모순!
    → ∴ 정보 추출 불가능
```

#### Groth16 시뮬레이터 구성

```
시뮬레이터 S의 동작 (toxic waste τ, α, β, γ, δ를 안다고 가정):

  주의: 시뮬레이터는 "보안 증명"을 위한 사고 실험이다.
        실제 세계에서 이 시뮬레이터를 실행하는 것은 불가능하다.
        (toxic waste가 삭제되었으므로)

  입력: VK, public_inputs (witness는 모름!)
  출력: 가짜 Proof (A', B', C') that verifies

  구성:
    1. 랜덤 a, b ∈ Fr 선택

    2. A' = a · G₁  (랜덤 G1 점)
       B' = b · G₂  (랜덤 G2 점)

    3. C'를 검증 방정식에서 역산:
       e(A', B') = e(α,β) · e(IC_sum, [γ]₂) · e(C', [δ]₂)

       → e(C', [δ]₂) = e(A', B') / (e(α,β) · e(IC_sum, [γ]₂))
       → e(C', [δ]₂) = e(A', B') · e(α,β)⁻¹ · e(IC_sum, [γ]₂)⁻¹

       C'를 구하려면:
       e(C', [δ]₂) = e(G₁, G₂)^(c' · δ)  (어떤 c'에 대해)

       시뮬레이터는 δ를 알므로:
       c' = (a·b - α·β - Σ pub_j · lc_j) · δ⁻¹  (in Fr)
       C' = c' · G₁

    4. Proof' = (A', B', C')

  검증:
    e(A', B') = e(G₁, G₂)^(a·b)
    e(C', [δ]₂) = e(G₁, G₂)^(c'·δ)
                 = e(G₁, G₂)^(a·b - αβ - Σ pub·lc)
    e(α,β) = e(G₁, G₂)^(αβ)
    e(IC_sum, [γ]₂) = e(G₁, G₂)^(Σ pub·lc)

    RHS = e^(αβ) · e^(Σ pub·lc) · e^(a·b - αβ - Σ pub·lc)
        = e^(a·b)
        = LHS  ✓✓✓
```

```
시뮬레이터가 증명하는 것:

  1. witness를 모르는 시뮬레이터도 유효한 증명을 만들 수 있음
  2. 시뮬레이터의 증명은 진짜 증명과 분포가 동일
     (둘 다 랜덤 원소로 구성)
  3. ∴ 검증자는 진짜/가짜를 구별할 수 없음
  4. ∴ 증명에서 witness 정보를 추출하는 것은 불가능

왜 시뮬레이터가 "현실에서" 위험하지 않은가?

  시뮬레이터는 toxic waste (τ, α, β, γ, δ)를 알아야 함
  trusted setup이 올바르게 수행되면 toxic waste는 삭제됨
  → 시뮬레이터를 실제로 실행하는 것은 불가능

  이것이 "trusted setup"의 핵심 의미:
    영지식성: toxic waste가 있으면 가짜 증명 가능 → ZK 증명 가능
    건전성: toxic waste가 없으면 가짜 증명 불가 → 위조 불가

  두 성질이 양립하는 이유:
    "이론적 시뮬레이터"의 존재 → 영지식성 증명
    "실제 세계"에서의 불가능성 → 건전성 유지
```

> [!important] 시뮬레이션 패러다임
> ```
> ZK 증명의 보안 증명은 항상 이 구조:
>
>   "검증자가 증명에서 f(witness)를 계산할 수 있다고 가정하자."
>   "그러면 시뮬레이터의 가짜 증명에서도 f(witness)를 계산할 수 있어야 한다."
>   "하지만 시뮬레이터는 witness를 모른다."
>   "시뮬레이터의 출력에는 f(witness)가 없다."
>   "→ 모순. ∴ 검증자는 f(witness)를 계산할 수 없다."
>
> 이것이 "시뮬레이션 기반 보안 증명"이며,
> Groth16뿐 아니라 모든 ZK 프로토콜의 표준 증명 기법이다.
> ```

---

### Part 29: 지식 추출기 — 건전성 (Knowledge Soundness) 증명

#### 건전성의 형식적 정의

```
지식 건전성 (Knowledge Soundness):

  "유효한 증명을 만들 수 있는 자는 반드시 witness를 안다"

  형식적으로:
    증명자 P가 verify(VK, x, π) = true인 π를 출력하면,
    P에서 witness w를 추출할 수 있는 "추출기" E가 존재한다.

  이것은 일반 건전성보다 강한 조건:
    건전성: "잘못된 statement에 대한 증명은 만들 수 없다"
    지식 건전성: "증명을 만들 수 있다면 witness를 실제로 안다"

  이 구분이 중요한 이유:
    건전성만으로는 "증명자가 witness를 알고 있다"를 보장하지 않음
    지식 건전성이 있어야 "증명 = witness 소유의 증거"가 됨
```

#### Groth16 지식 추출기 (AGM 모델)

```
Algebraic Group Model (AGM):

  가정: 증명자는 "대수적"이다
    즉, G1/G2의 원소를 출력할 때, 그것이
    이전에 받은 원소들의 "선형 결합"으로 표현되어야 함

  증명자가 받은 G1 원소들 (PK에서):
    [α]₁, [β]₁, [δ]₁,
    [aⱼ(τ)]₁ for all j,
    [bⱼ(τ)]₁ for all j,
    [lcⱼ/δ]₁ for private j,
    [τⁱ·t(τ)/δ]₁ for all i

  AGM에서 A는 이들의 선형결합이어야 함:
    A = ã·[α]₁ + b̃·[β]₁ + d̃·[δ]₁
      + Σⱼ ãⱼ·[aⱼ(τ)]₁
      + Σⱼ b̃ⱼ·[bⱼ(τ)]₁
      + Σⱼ' l̃ⱼ·[lcⱼ/δ]₁
      + Σᵢ h̃ᵢ·[τⁱt(τ)/δ]₁

  이것을 "τ의 다항식"로 해석하면:
    A = ã·α + b̃·β + d̃·δ
      + Σ ãⱼ·aⱼ(τ) + Σ b̃ⱼ·bⱼ(τ)
      + (Σ l̃ⱼ·lcⱼ + Σ h̃ᵢ·τⁱ·t(τ)) / δ

  마찬가지로 B (G2에서):
    B = b̂·β + d̂·δ + Σ b̂ⱼ·bⱼ(τ)

추출기 구성:

  Step 1: 검증 방정식 e(A, B) = e(α,β) · e(IC, γ₂) · e(C, δ₂)를
          τ에 대한 다항식 등식으로 해석

  Step 2: α, β, γ, δ가 독립적인 랜덤 변수이므로,
          각 변수의 계수를 비교 (coefficient extraction)

  Step 3: α의 계수 비교:
    A에 α가 포함 → ã ≠ 0
    α가 A에만 나타나고 B에는 없으므로
    → e(A, B)에서 α·(B의 내용)이 나타남
    → 이것은 e(α,β) = e(α·G₁, β·G₂) 항에 대응
    → ã = 1 이어야 함 (α가 정확히 1번 나타남)

  Step 4: β의 계수도 마찬가지로 비교하면:
    A 쪽의 β 기여와 B 쪽의 β 기여가 균형을 이루어야 함

  Step 5: δ의 계수 비교:
    A의 δ항 = r·δ, B의 δ항 = s·δ
    C에는 s·A + r·B' - rs·δ 항이 있으므로
    δ²으로 나누어떨어져야 함 → δ 항들이 일관되어야 함

  Step 6: 나머지 (α, β, γ, δ 없는 부분):
    Σ ãⱼ·aⱼ(τ) · Σ b̂ⱼ·bⱼ(τ) - cⱼ(τ) 항
    = h'(τ) · t(τ) 형태여야 함

  Step 7: Schwartz-Zippel로:
    τ가 랜덤이므로, 다항식으로서
    a'(x)·b'(x) - c'(x) = h'(x)·t(x)
    가 성립해야 함 (높은 확률로)

  Step 8: 이것은 곧 witness w = (ã₀, ã₁, ..., ãₙ₋₁)이
    R1CS를 만족한다는 것!

  ∴ 추출기 E는 AGM 표현의 계수로부터 witness를 추출 가능
```

```
요약: 건전성이 성립하는 이유

  1. AGM 모델에서 증명자의 출력은 PK 원소들의 선형결합
  2. 검증 방정식이 성립하면 → 계수가 QAP를 만족
  3. 계수 = witness
  4. ∴ 증명을 만들 수 있다면 witness를 반드시 알고 있다

  이것이 "지식의 증거"(proof of knowledge)의 의미:
    증명 π가 존재한다 → witness w가 존재한다
    (단순히 "statement가 참"이 아니라 "증명자가 w를 안다")

주의: AGM 모델의 한계
  AGM은 "대수적 조작만 가능하다"는 가정
  → 이것은 일반 모델(GGM)보다 약한 가정
  → Groth16은 AGM에서 증명됨
  → 실제 세계에서는 "충분히 안전"으로 간주
```

> [!tip] 영지식성 vs 건전성 — 미묘한 균형
> ```
> 영지식성: "toxic waste를 알면 가짜 증명 생성 가능"
>   → 시뮬레이터로 증명
>   → toxic waste가 삭제되었으므로 현실에서는 불가능
>
> 건전성: "toxic waste를 모르면 가짜 증명 생성 불가"
>   → 지식 추출기로 증명
>   → AGM 모델에서 계수 추출
>
> 두 성질이 모순되지 않는 이유:
>   시뮬레이터: toxic waste 필요 (현실에서 불가능)
>   추출기: toxic waste 불필요 (AGM 모델에서 작동)
>
>   → "toxic waste를 아는 세계"에서는 ZK
>   → "toxic waste를 모르는 세계"에서는 Sound
>   → trusted setup이 이 두 세계를 분리
> ```

---

### Part 30: 완전 수치 추적 — 작은 체에서의 Groth16

#### 목표: 실제 숫자로 Groth16 전체 과정을 추적

```
주의: BN254 Fr은 254비트 (매우 큰 소수)이므로
교육용으로 작은 소수체 F₁₃ (mod 13)에서 추적한다.

물론 실제 구현은 BN254 Fr을 사용하지만,
수학적 구조는 100% 동일하다.

체: F₁₃ = {0, 1, 2, ..., 12}
  덧셈, 곱셈: mod 13
  역원: a⁻¹ = a^(11) mod 13 (페르마 소정리)

역원 테이블:
  1⁻¹ = 1,   2⁻¹ = 7,   3⁻¹ = 9,   4⁻¹ = 10
  5⁻¹ = 8,   6⁻¹ = 11,  7⁻¹ = 2,   8⁻¹ = 5
  9⁻¹ = 3,   10⁻¹ = 4,  11⁻¹ = 6,  12⁻¹ = 12
```

#### 회로: a · b = c (단순 곱셈, 1 제약)

```
변수:
  a = 3 (witness, 비공개)
  b = 4 (witness, 비공개)
  c = 12 (instance, 공개) ← 3·4 = 12 mod 13

변수 인덱싱:
  j=0: One = 1
  j=1: c = 12  (Instance)
  j=2: a = 3   (Witness)
  j=3: b = 4   (Witness)

witness 벡터: s = [1, 12, 3, 4]
num_instance = 1, num_variables = 4
```

#### R1CS 행렬

```
1개의 제약: a · b = c

  A = [0, 0, 1, 0]  (a 계수)
  B = [0, 0, 0, 1]  (b 계수)
  C = [0, 1, 0, 0]  (c 계수)

확인:
  ⟨A, s⟩ · ⟨B, s⟩ = ⟨C, s⟩
  (0·1 + 0·12 + 1·3 + 0·4) · (0·1 + 0·12 + 0·3 + 1·4) = (0·1 + 1·12 + 0·3 + 0·4)
  3 · 4 = 12 ✓
```

#### QAP 변환

```
도메인: D = {1}  (제약 1개이므로 ω₁ = 1)

각 변수 j의 다항식 (degree 0 = 상수):
  a₀(x) = 0,   b₀(x) = 0,   c₀(x) = 0    (One)
  a₁(x) = 0,   b₁(x) = 0,   c₁(x) = 1    (c)
  a₂(x) = 1,   b₂(x) = 0,   c₂(x) = 0    (a)
  a₃(x) = 0,   b₃(x) = 1,   c₃(x) = 0    (b)

소거 다항식:
  t(x) = (x - 1)  (degree 1)

witness 다항식:
  a(x) = Σ sⱼ · aⱼ(x) = 1·0 + 12·0 + 3·1 + 4·0 = 3
  b(x) = Σ sⱼ · bⱼ(x) = 1·0 + 12·0 + 3·0 + 4·1 = 4
  c(x) = Σ sⱼ · cⱼ(x) = 1·0 + 12·1 + 3·0 + 4·0 = 12

h(x) = (a(x)·b(x) - c(x)) / t(x)
     = (3·4 - 12) / (x - 1)
     = (12 - 12) / (x - 1)
     = 0 / (x - 1)
     = 0

(단일 제약이므로 h = 0)
```

#### Trusted Setup (F₁₃에서)

```
toxic waste 선택:
  τ = 5,  α = 3,  β = 7,  γ = 2,  δ = 11

역원 계산:
  γ⁻¹ = 2⁻¹ = 7 (mod 13)
  δ⁻¹ = 11⁻¹ = 6 (mod 13)

① QAP 다항식을 τ=5에서 평가:
  (단일 제약이므로 다항식이 모두 상수 → τ 무관)

  a₀(5) = 0,   b₀(5) = 0,   c₀(5) = 0
  a₁(5) = 0,   b₁(5) = 0,   c₁(5) = 1
  a₂(5) = 1,   b₂(5) = 0,   c₂(5) = 0
  a₃(5) = 0,   b₃(5) = 1,   c₃(5) = 0

② lcⱼ = β·aⱼ(τ) + α·bⱼ(τ) + cⱼ(τ):
  lc₀ = 7·0 + 3·0 + 0 = 0
  lc₁ = 7·0 + 3·0 + 1 = 1
  lc₂ = 7·1 + 3·0 + 0 = 7
  lc₃ = 7·0 + 3·1 + 0 = 3

③ IC vs L 분리:
  num_public = 2 (One, c)

  IC (공개):
    ic[0] = lc₀ · γ⁻¹ = 0 · 7 = 0  → [0]₁ = O (항등원)
    ic[1] = lc₁ · γ⁻¹ = 1 · 7 = 7  → [7]₁

  L (비공개):
    l_query[0] = lc₂ · δ⁻¹ = 7 · 6 = 42 mod 13 = 3  → [3]₁
    l_query[1] = lc₃ · δ⁻¹ = 3 · 6 = 18 mod 13 = 5  → [5]₁

④ h_query:
  m = 1 → h_len = max(0, 1-1) = 0
  → h_query = []  (비어 있음)

⑤ 기본 커브 포인트:
  [α]₁ = [3]₁,   [β]₁ = [7]₁
  [β]₂ = [7]₂
  [δ]₁ = [11]₁,  [δ]₂ = [11]₂
  [γ]₂ = [2]₂

⑥ a_query (G1):
  a_query[0] = [a₀(τ)]₁ = [0]₁ = O
  a_query[1] = [a₁(τ)]₁ = [0]₁ = O
  a_query[2] = [a₂(τ)]₁ = [1]₁ = G₁
  a_query[3] = [a₃(τ)]₁ = [0]₁ = O

  b_g2_query (G2):
  b_g2_query[0] = [0]₂ = O
  b_g2_query[1] = [0]₂ = O
  b_g2_query[2] = [0]₂ = O
  b_g2_query[3] = [1]₂ = G₂

  b_g1_query (G1):
  b_g1_query[0] = [0]₁ = O
  b_g1_query[1] = [0]₁ = O
  b_g1_query[2] = [0]₁ = O
  b_g1_query[3] = [1]₁ = G₁

⑦ e(α, β) = e([3]₁, [7]₂) = e(G₁, G₂)^(3·7) = e(G₁, G₂)^21
   = e(G₁, G₂)^(21 mod 13) = e(G₁, G₂)^8
```

#### Prove (F₁₃에서)

```
witness = [1, 12, 3, 4]
h(x) = 0  (위에서 계산)

블라인딩 팩터 선택 (랜덤):
  r = 9,  s = 6

── A 계산 ──

  A = [α]₁ + Σ wⱼ · a_query[j] + r · [δ]₁

  = [3]₁
    + 1·O + 12·O + 3·[1]₁ + 4·O   (w₀ = 1, w₂ = 3만 기여)
    + 9·[11]₁

  = [3]₁ + [3]₁ + [99 mod 13]₁
  = [3]₁ + [3]₁ + [8]₁
  = [3 + 3 + 8]₁
  = [14 mod 13]₁
  = [1]₁

  검산: α + a(τ) + rδ = 3 + 3 + 9·11 = 3 + 3 + 99
        = 3 + 3 + 8 (mod 13) = 14 mod 13 = 1  ✓

── B ∈ G2 계산 ──

  B = [β]₂ + Σ wⱼ · b_g2_query[j] + s · [δ]₂

  = [7]₂ + 4·[1]₂ + 6·[11]₂   (w₃ = 4만 기여)
  = [7]₂ + [4]₂ + [66 mod 13]₂
  = [7]₂ + [4]₂ + [1]₂
  = [12]₂

  검산: β + b(τ) + sδ = 7 + 4 + 6·11 = 7 + 4 + 66
        = 7 + 4 + 1 (mod 13) = 12  ✓

── B' ∈ G1 계산 (C용) ──

  B' = [7]₁ + 4·[1]₁ + 6·[11]₁
     = [7 + 4 + 1]₁
     = [12]₁

── C 계산 ──

  C = Σ_{private} wⱼ · l_query + h기여 + 블라인딩

  private 기여:
    3 · l_query[0] + 4 · l_query[1]
    = 3 · [3]₁ + 4 · [5]₁
    = [9]₁ + [20 mod 13]₁
    = [9]₁ + [7]₁
    = [16 mod 13]₁
    = [3]₁

  h 기여: 0 (h_query 비어 있음)

  블라인딩: s·A + r·B' - rs·[δ]₁
    s·A = 6 · [1]₁ = [6]₁
    r·B' = 9 · [12]₁ = [108 mod 13]₁ = [4]₁
    rs·[δ]₁ = (9·6)·[11]₁ = 54·[11]₁ = (54 mod 13)·[11]₁
            = 2·[11]₁ = [22 mod 13]₁ = [9]₁

    블라인딩 = [6]₁ + [4]₁ - [9]₁
             = [10 - 9]₁ = [1]₁

  C = [3]₁ + [0]₁ + [1]₁ = [4]₁

Proof = (A=[1]₁, B=[12]₂, C=[4]₁)
```

#### Verify (F₁₃에서)

```
public_inputs = [12]  (c의 값)

── IC_sum 계산 ──
  IC_sum = ic[0] + 12 · ic[1]
         = O + 12 · [7]₁
         = [84 mod 13]₁
         = [6]₁

── LHS ──
  LHS = e(A, B) = e([1]₁, [12]₂) = e(G₁, G₂)^(1·12)
      = e(G₁, G₂)^12

── RHS ──
  e(α,β) = e(G₁, G₂)^8                     (위에서 계산)
  e(IC_sum, [γ]₂) = e([6]₁, [2]₂) = e(G₁, G₂)^(6·2) = e(G₁, G₂)^12
  e(C, [δ]₂) = e([4]₁, [11]₂) = e(G₁, G₂)^(4·11) = e(G₁, G₂)^(44 mod 13)
              = e(G₁, G₂)^5

  RHS = e(G₁, G₂)^(8 + 12 + 5) = e(G₁, G₂)^(25 mod 13) = e(G₁, G₂)^12

── 비교 ──
  LHS = e(G₁, G₂)^12
  RHS = e(G₁, G₂)^12
  LHS == RHS  ✓  → verify = true ✓✓✓
```

> [!important] 수치 추적의 의미
> ```
> 위의 F₁₃ 추적에서 모든 숫자가 실제로 계산되었다.
> BN254 위의 Groth16도 정확히 같은 구조이며,
> 차이는 "숫자가 매우 크다"는 것뿐이다.
>
> 핵심 확인:
>   α + a(τ) + rδ = 3 + 3 + 99 ≡ 1 (mod 13) → A의 스칼라
>   β + b(τ) + sδ = 7 + 4 + 66 ≡ 12 (mod 13) → B의 스칼라
>   1 · 12 = 12 → LHS의 지수
>   8 + 12 + 5 = 25 ≡ 12 (mod 13) → RHS의 지수
>
>   LHS 지수 = RHS 지수 → 검증 통과!
> ```

---

### Part 31: 틀린 공개 입력의 수치 추적

```
같은 증명 (A=[1]₁, B=[12]₂, C=[4]₁)에 대해
틀린 public_input = [10] (실제는 12)으로 검증하면?

── IC_sum' ──
  IC_sum' = ic[0] + 10 · ic[1]
          = O + 10 · [7]₁
          = [70 mod 13]₁
          = [5]₁    (원래는 [6]₁)

── LHS (변하지 않음) ──
  LHS = e(G₁, G₂)^12

── RHS' ──
  e(α,β) = e(G₁, G₂)^8
  e(IC_sum', [γ]₂) = e([5]₁, [2]₂) = e(G₁, G₂)^10
  e(C, [δ]₂) = e(G₁, G₂)^5

  RHS' = e(G₁, G₂)^(8 + 10 + 5) = e(G₁, G₂)^(23 mod 13) = e(G₁, G₂)^10

── 비교 ──
  LHS = e(G₁, G₂)^12
  RHS' = e(G₁, G₂)^10
  12 ≠ 10  → LHS ≠ RHS'  → verify = false ✓

정확히 "공개 입력 차이" (12 - 10 = 2)가
IC_sum의 차이 [6]₁ vs [5]₁ → e(G₁, G₂)^12 vs e(G₁, G₂)^10 으로 나타남
→ GT에서의 차이 = e(G₁, G₂)^2
→ 검증 방정식이 깨짐
```

---

### Part 32: 변조 증명의 수치 추적

```
A를 변조: A' = A + G₁ = [1]₁ + [1]₁ = [2]₁

── LHS' ──
  LHS' = e(A', B) = e([2]₁, [12]₂) = e(G₁, G₂)^(2·12) = e(G₁, G₂)^24
       = e(G₁, G₂)^(24 mod 13) = e(G₁, G₂)^11

── RHS (변하지 않음) ──
  RHS = e(G₁, G₂)^12

── 비교 ──
  11 ≠ 12 → LHS' ≠ RHS → verify = false ✓

차이 분석:
  LHS' - LHS (지수에서) = 24 - 12 = 12 = B의 스칼라
  → e(G₁, B) = e(G₁, [12]₂) = e(G₁, G₂)^12
  → 변조 Δ = G₁를 더하면 LHS에 e(G₁, G₂)^(B의 스칼라) 만큼 배가됨

이것은 Part 16의 일반 분석을 F₁₃에서 수치로 확인한 것이다.
```

---

### Part 33: MSM (Multi-Scalar Multiplication) 심화

#### 단순 구현 (우리의 코드)

```
A = [α]₁ + Σⱼ wⱼ · [aⱼ(τ)]₁ + r · [δ]₁

우리 코드의 구현:
  let mut proof_a = pk.alpha_g1;                    // P₀
  for j in 0..pk.num_variables {                     // n번 반복
      proof_a = proof_a + pk.a_query[j].scalar_mul(&witness[j].to_repr());
  }
  proof_a = proof_a + pk.delta_g1.scalar_mul(&r.to_repr());

연산량 분석:
  각 scalar_mul: O(256) 점 덧셈 (double-and-add)
  n개의 scalar_mul → 총 O(256n) 점 덧셈

  CubicCircuit (n=5): 5 × 256 = 1,280 점 연산
  실제 회로 (n=100,000): 25,600,000 점 연산

  → 실용적이지 않음!
```

#### MSM: Pippenger 알고리즘

```
MSM 문제:
  Σᵢ sᵢ · Pᵢ  를 계산하라 (n개의 스칼라-포인트 쌍)

Naive: O(256n)  — 각 scalar_mul을 독립적으로 계산

Pippenger (1976):
  핵심 아이디어: 같은 비트 위치의 포인트를 묶어서 처리

  1. 스칼라를 c비트 윈도우로 분할 (c ≈ log₂n)
     예: 256비트 스칼라, c=16이면 16개의 윈도우

  2. 각 윈도우에서 "버킷" 합산:
     버킷 b: sᵢ의 해당 윈도우가 b인 모든 Pᵢ를 합산
     → 2^c개의 버킷, 각각 n/2^c개의 점 합산

  3. 윈도우 결합:
     각 윈도우의 결과를 2^c배씩 곱하면서 합산

  복잡도: O(n / log n) 그룹 연산
    → n = 100,000이면 약 6,000배 빠름!

  ┌───────────────────────────────────────────────────┐
  │   스칼라 s₁ = | w₁₆ | w₁₅ | ... | w₁ |           │
  │   스칼라 s₂ = | w₁₆ | w₁₅ | ... | w₁ |           │
  │   ...                                             │
  │   스칼라 sₙ = | w₁₆ | w₁₅ | ... | w₁ |           │
  │                                                   │
  │   ↓ 각 윈도우별로 처리                              │
  │                                                   │
  │   윈도우 1: 모든 sᵢ의 하위 c비트로 버킷 분류         │
  │   윈도우 2: 다음 c비트로 버킷 분류                    │
  │   ...                                              │
  │   윈도우 k: 최상위 c비트로 버킷 분류                   │
  │                                                    │
  │   최종: 윈도우들을 (×2^c)씩 곱하며 합산                │
  └───────────────────────────────────────────────────┘
```

```
Pippenger의 직관적 이해:

  Naive: "각 스칼라 독립 처리"
    s₁·P₁ = s₁번 P₁ 더하기
    s₂·P₂ = s₂번 P₂ 더하기
    ...

  Pippenger: "같은 비트패턴끼리 묶기"
    예: s₁의 하위 4비트 = 0101, s₃의 하위 4비트 = 0101
    → P₁과 P₃을 "버킷 5"에 함께 넣고 한 번에 처리

    이렇게 묶으면 중복 작업이 대폭 줄어듦
    → O(n) → O(n/log n)

우리 구현에서는?
  교육용이므로 naive O(256n) 사용
  arkworks/bellman: Pippenger MSM 사용
  → 같은 수학적 결과, 다른 계산 복잡도
```

> [!tip] MSM이 Groth16 성능의 핵심
> ```
> Groth16 증명 시간의 ~80%가 MSM에 소비됨
>
>   setup: a_query, b_query, l_query, h_query 계산 → MSM
>   prove: A, B, C 계산 → MSM
>   verify: IC_sum 계산 → 작은 MSM (공개 입력 수만큼)
>
> MSM 최적화 기법들:
>   1. Pippenger (O(n/log n))
>   2. GLV 분해 (endomorphism 활용)
>   3. 병렬화 (멀티코어/GPU)
>   4. WNAF (Windowed NAF) 스칼라 표현
>
> 최신 GPU 구현 (cuMSM 등):
>   n = 2²⁰ (약 100만): ~200ms
>   n = 2²⁴ (약 1600만): ~3초
> ```

---

### Part 34: Groth16 on Ethereum — BN254 Precompile

#### EVM에서의 Groth16 검증

```
Ethereum은 Groth16을 위한 프리컴파일(precompile) 3개를 제공:

  EIP-196 (Constantinople, 2019):
    0x06: ecAdd(G1, G1) → G1          — 150 gas
    0x07: ecMul(G1, scalar) → G1      — 6,000 gas

  EIP-197 (Constantinople, 2019):
    0x08: ecPairing(pairs) → bool      — 113,000 + 80,000 × k gas
      k = 페어링 쌍의 수

  모두 BN254 (alt_bn128) 전용

실제 검증 비용:

  IC_sum 계산:
    ℓ회 ecMul + ℓ회 ecAdd
    비용: ℓ × (6,000 + 150) = ℓ × 6,150 gas

  검증 방정식:
    e(A, B) =? e(α,β) · e(IC_sum, γ₂) · e(C, δ₂)

    변환: e(A, B) · e(IC_sum, γ₂)⁻¹ · e(C, δ₂)⁻¹ =? e(α,β)

    → e(-A, B) · e(IC_sum, γ₂) · e(C, δ₂) =? e(α,β)

    하지만 프리컴파일은 "모든 페어링의 곱 = 1" 형태를 기대:
    e(A, B) · e(-IC_sum, γ₂) · e(-C, δ₂) · e(α₁, β₂) =? 1ₜ

    또는 동치:
    ecPairing([(A, B), (-IC_sum, γ₂), (-C, δ₂), (α₁, β₂)]) = true

    비용: 113,000 + 80,000 × 4 = 433,000 gas

  총 비용 (ℓ = 1):
    IC_sum: 6,150 gas
    페어링: 433,000 gas
    기타: ~5,000 gas
    ─────────────────
    합계: ~444,150 gas  (약 $10-50 at 30 gwei)

  비교:
    일반 ERC-20 전송: ~65,000 gas
    Groth16 검증: ~450,000 gas
    → 약 7배 비싸지만, 임의의 복잡한 계산을 검증!
```

#### Solidity 검증자 구조

```
Solidity에서의 Groth16 검증자 (의사 코드):

  contract Groth16Verifier {
      // VK 상수 (배포 시 고정)
      uint256[2] alpha_g1;      // G1 점 (x, y)
      uint256[4] beta_g2;       // G2 점 (x₀, x₁, y₀, y₁)
      uint256[4] gamma_g2;
      uint256[4] delta_g2;
      uint256[2][] ic;          // ℓ+1개의 G1 점

      function verify(
          uint256[] public_inputs,
          uint256[2] A,           // proof.a (G1)
          uint256[4] B,           // proof.b (G2)
          uint256[2] C            // proof.c (G1)
      ) returns (bool) {
          // 1. IC_sum = ic[0] + Σ public_inputs[i] · ic[i+1]
          G1 ic_sum = ic[0];
          for (uint i = 0; i < public_inputs.length; i++) {
              ic_sum = ecAdd(ic_sum, ecMul(ic[i+1], public_inputs[i]));
          }

          // 2. 페어링 검사
          //    e(A, B) · e(-IC_sum, gamma) · e(-C, delta) · e(alpha, beta) = 1
          return ecPairing([
              negate(A), B,
              ic_sum, negate(gamma_g2),
              negate(C), delta_g2,
              alpha_g1, beta_g2
          ]);
      }
  }

calldata 크기:
  A: 64 bytes (G1)
  B: 128 bytes (G2)
  C: 64 bytes (G1)
  public_inputs: 32 × ℓ bytes
  ─────────────────────
  총: 256 + 32ℓ bytes

  ℓ = 1이면: 288 bytes
  Groth16의 "Succinct"가 빛나는 순간!
```

> [!important] 왜 Groth16이 온체인 검증의 표준인가
> ```
> 1. 최소 증명 크기: 256 bytes → 낮은 calldata 비용
> 2. 고정 검증 비용: ~450K gas (회로 크기에 무관)
> 3. 네이티브 프리컴파일: ecPairing 지원
>
> 비교:
>   PLONK: 증명 ~4KB, 검증 ~500K gas (약간 비쌈)
>   STARK: 증명 ~100KB, 검증 ~2M gas (훨씬 비쌈)
>   FRI 기반: 프리컴파일 없음 → 비용 더 높음
>
> zk-rollup 경제학:
>   L1 검증 비용이 "모든 L2 트랜잭션"에 분산됨
>   배치에 1000 tx가 있다면:
>     450K gas / 1000 = 450 gas per tx
>     (일반 전송 65K gas의 ~0.7%)
>
>   이것이 zk-rollup의 확장성 핵심!
> ```

---

### Part 35: Groth16 증명의 Malleability (가변성)

#### 문제: 증명이 유일하지 않다

```
Groth16 증명의 가변성(malleability) 문제:

  정의: 유효한 증명 π = (A, B, C)가 주어지면,
        검증을 통과하는 다른 증명 π' = (A', B', C')를
        생성할 수 있다.

공격 1: 네게이션 공격

  π = (A, B, C) → verify 통과

  π' = (-A, -B, C)를 구성하면:

  e(-A, -B) = e(A, B)  (쌍선형성: e(-P, -Q) = e(P, Q))

  IC_sum, C, δ₂ 모두 변하지 않으므로:
  RHS도 변하지 않음

  ∴ e(A', B') = e(A, B) = RHS → verify 통과!

  하지만 (A, B, C) ≠ (-A, -B, C)  (다른 증명!)

공격 2: 스칼라 스케일링

  임의의 k ∈ Fr* 에 대해:
  A' = k·A
  B' = k⁻¹·B
  C는 수정 필요...

  e(k·A, k⁻¹·B) = e(A, B)^(k·k⁻¹) = e(A, B)^1 = e(A, B)

  하지만 C도 수정해야 검증 통과:
  C의 블라인딩이 A, B에 의존하므로 단순하지 않음
  → 일반적으로 이 공격은 부분적으로만 작동

왜 이것이 문제인가?

  대부분의 경우에는 문제가 아님:
    검증자는 "증명이 유효한지"만 확인
    같은 statement에 대한 다른 유효한 증명은 무해

  문제가 되는 경우:
    증명 해시를 식별자로 사용하는 경우
    → 같은 증명에 대해 다른 해시 → 이중 처리 가능

  예: zk-rollup에서 증명을 트랜잭션 ID로 사용하면
      공격자가 증명을 변조하여 중복 제출 가능
```

#### 대응책

```
1. 증명 정규화 (Normalization):
   A의 y 좌표가 양수(또는 특정 비트 조건)인 것만 수용
   → 네게이션 공격 방지

2. 증명 해시에 statement 포함:
   id = hash(public_inputs || proof)
   → 같은 statement에 대한 다른 증명은 다른 ID

3. 강한 해시 바인딩:
   증명을 제출할 때 sender의 서명과 함께 제출
   → 제3자가 증명을 변조할 수 없음

4. Groth16+ (확장):
   증명에 추가 원소를 넣어 유일성 보장
   → 증명 크기가 약간 증가하지만 malleability 방지

우리 구현에서는?
  교육용이므로 malleability 방지 미구현
  프로덕션에서는 반드시 위 대응책 중 하나를 적용해야 함
```

---

### Part 36: 배치 검증 (Batch Verification)

#### 여러 증명을 한 번에 검증

```
상황: N개의 서로 다른 증명을 동시에 검증해야 할 때

  Naive: N번 verify 호출
    비용: 3N 페어링

  배치 검증: N개를 하나의 등식으로 결합
    비용: 2N + 1 페어링 (약 33% 절약)

방법:

  각 증명 i: e(Aᵢ, Bᵢ) = αβ_gt · e(ICᵢ, γ₂) · e(Cᵢ, δ₂)

  랜덤 결합 계수 rᵢ ∈ Fr (Schwartz-Zippel 보안):

  Σᵢ rᵢ · [e(Aᵢ, Bᵢ) - αβ_gt · e(ICᵢ, γ₂) · e(Cᵢ, δ₂)] = 0

  변환:
  e(Σᵢ rᵢ·Aᵢ, Bᵢ) 형태는 안 됨 (B가 다름)

  올바른 변환:
  Πᵢ e(Aᵢ, Bᵢ)^rᵢ = (αβ_gt)^(Σrᵢ) · e(Σᵢ rᵢ·ICᵢ, γ₂) · e(Σᵢ rᵢ·Cᵢ, δ₂)

  RHS 분석:
    (αβ_gt)^(Σrᵢ): 1회 지수 연산
    e(Σᵢ rᵢ·ICᵢ, γ₂): 1회 MSM + 1회 pairing
    e(Σᵢ rᵢ·Cᵢ, δ₂): 1회 MSM + 1회 pairing

  LHS 분석:
    Πᵢ e(Aᵢ, Bᵢ)^rᵢ → Miller loop N회 + final exp 1회
    (배치 최적화: Miller loop만 N회, final exp는 1회로 통합)

  총 비용:
    Miller loop: N회 (LHS) + 2회 (RHS)
    Final exp: 1회 (모든 것을 곱한 후 한 번)
    MSM: 2회 (ICᵢ 합, Cᵢ 합)

    → 단일 검증: 3회 Miller loop + 3회 final exp
    → 배치 N개: (N+2)회 Miller loop + 1회 final exp

    final exp가 비싼 연산이므로:
    N이 클수록 절약 효과가 큼

  예: N = 100
    단일: 300 Miller + 300 final exp
    배치: 102 Miller + 1 final exp
    → 약 3배 빠름!
```

```
보안 분석:

  왜 랜덤 rᵢ가 필요한가?

  rᵢ = 1 (전부 1)로 하면:
    공격자가 "상쇄하는" 잘못된 증명 쌍을 만들 수 있음
    예: 증명 1은 LHS가 +Δ 크고, 증명 2는 LHS가 -Δ 크면
        합산했을 때 상쇄되어 검증 통과

  랜덤 rᵢ를 사용하면:
    각 증명의 오차에 랜덤 가중치가 곱해짐
    → 상쇄 확률 = 1/|Fr| ≈ 10^(-77)  (무시 가능)

  이것은 Schwartz-Zippel 보조정리의 응용:
    "랜덤 점에서 0이 아닌 다항식이 0일 확률은 무시할 수 있다"
```

> [!tip] 배치 검증의 실용적 의미
> ```
> zk-rollup 검증자:
>   L2 블록마다 1개의 증명 → 배치 불필요
>   여러 L2의 증명을 L1에서 함께 → 배치 유용
>
> 독립 검증자 (노드):
>   블록의 모든 트랜잭션 증명을 한 번에 검증
>   → 동기화 시간 단축
>
> 재귀 증명 (recursive SNARK):
>   "N개의 Groth16 증명이 유효하다"를 하나의 증명으로
>   → 배치 검증보다 더 강력하지만 복잡도 높음
> ```

---

### Part 37: Schwartz-Zippel 보조정리 — Groth16의 수학적 기반

#### 정리

```
Schwartz-Zippel Lemma (1979/1980):

  f(x₁, ..., xₙ)가 총차수 d인 0이 아닌 다항식이고,
  r₁, ..., rₙ이 유한체 F에서 균일 랜덤으로 선택되면:

    Pr[f(r₁, ..., rₙ) = 0] ≤ d / |F|

특수 경우 (단변수):
  f(x)가 degree d이면:
    Pr[f(τ) = 0] ≤ d / |Fr|

  BN254에서 |Fr| ≈ 2^254, 일반적인 d < 2^30이면:
    Pr ≤ 2^30 / 2^254 = 2^(-224) ≈ 10^(-67)

  → 사실상 0
```

#### Groth16에서의 적용

```
Schwartz-Zippel이 사용되는 3가지 핵심 포인트:

① QAP에서 다항식 항등식 검증

  R1CS는 m개의 제약이 모두 만족되어야 함
  QAP는 이것을 "하나의 다항식 항등식"으로 압축:

    a(x)·b(x) - c(x) = h(x)·t(x)

  하지만 검증자는 하나의 점 τ에서만 확인:
    a(τ)·b(τ) - c(τ) = h(τ)·t(τ)

  Schwartz-Zippel 보장:
    만약 다항식 관계가 성립하지 않으면 (f(x) ≠ 0),
    f(τ) = 0일 확률 ≤ d/|Fr| ≈ 0

    즉, τ에서 성립하면 전체 다항식이 같을 확률이 압도적

  직관:
    "degree d인 다항식은 최대 d개의 영점을 가진다.
     |Fr| 크기의 체에서 하나의 점을 랜덤으로 고르면
     그것이 영점일 확률은 d/|Fr| → 거의 0"

② 지식 추출에서의 다항식 비교

  AGM 모델에서 추출기가 계수를 비교할 때:
    "이 두 다항식이 같은가?"
    → τ에서의 값이 같으면 (e를 통해 확인)
    → Schwartz-Zippel에 의해 전체가 같을 확률이 압도적

③ 배치 검증에서의 랜덤 결합

  rᵢ를 랜덤으로 선택하면:
    Σ rᵢ · (LHSᵢ - RHSᵢ) = 0이 되려면
    각 LHSᵢ - RHSᵢ = 0이어야 함 (높은 확률로)

    이것은 다변수 Schwartz-Zippel의 적용
```

> [!important] Schwartz-Zippel 없이는 ZK-SNARK가 불가능
> ```
> Schwartz-Zippel이 보장하는 것:
>   "하나의 랜덤 점에서의 검사가 전체 다항식의 검사와 동치"
>
> 이것이 "Succinct"의 수학적 근거:
>   m개의 제약 → 1개의 다항식 등식 → τ 하나로 검증
>   O(m) 검사 → O(1) 검사로 압축
>
> 이 보조정리가 없었다면:
>   모든 도메인 점에서 검사해야 함 → O(m) 검증
>   → "Succinct"가 불가능
>   → SNARK가 아닌 일반 proof system만 가능
> ```

---

### Part 38: 다항식 나눗셈과 h(x) 계산의 심화

#### compute_h의 수학

```
h(x) = (a(x)·b(x) - c(x)) / t(x) 의 상세 분석

a(x)·b(x)의 차수:
  a(x) = Σⱼ wⱼ·aⱼ(x),  각 aⱼ: degree ≤ m-1
  b(x) = Σⱼ wⱼ·bⱼ(x),  각 bⱼ: degree ≤ m-1

  a(x): degree ≤ m-1
  b(x): degree ≤ m-1
  a(x)·b(x): degree ≤ 2(m-1)

c(x): degree ≤ m-1

p(x) = a(x)·b(x) - c(x): degree ≤ 2(m-1)

t(x) = ∏ᵢ₌₁ᵐ (x - ωᵢ): degree = m

h(x) = p(x) / t(x): degree ≤ 2(m-1) - m = m-2

h(x)의 계수 개수 = m-1  (degree m-2이므로 계수 0, 1, ..., m-2)

이것이 h_query의 길이를 결정:
  h_query.len() = m-1 = m.saturating_sub(1)
```

#### 다항식 나눗셈 알고리즘

```
schoolbook 다항식 나눗셈:

  p(x) ÷ t(x) = h(x) ... rem(x)

  Algorithm:
    remainder = p(x).clone()
    quotient = Polynomial::zero()

    while degree(remainder) ≥ degree(t):
      // 최고차항의 비율
      ratio = leading_coeff(remainder) / leading_coeff(t)
      deg_diff = degree(remainder) - degree(t)

      // ratio · x^deg_diff
      term = ratio · x^deg_diff

      quotient += term
      remainder -= term · t(x)

  return (quotient, remainder)

  복잡도: O(m²) (schoolbook)
  FFT 기반: O(m log m)

예: CubicCircuit (m=3)

  a(x)·b(x)의 차수 ≤ 4
  c(x)의 차수 ≤ 2
  t(x) = x³ - 6x² + 11x - 6 (차수 3)

  p(x) = a(x)·b(x) - c(x): 차수 ≤ 4
  h(x) = p(x) / t(x): 차수 ≤ 1 (1차 다항식)
  h_query.len() = 2 (h₀, h₁)

  h(x) = h₀ + h₁·x

  Σᵢ hᵢ · h_query[i] = h₀·[t(τ)/δ]₁ + h₁·[τ·t(τ)/δ]₁
                       = [(h₀ + h₁·τ)·t(τ)/δ]₁
                       = [h(τ)·t(τ)/δ]₁
```

#### QAP 불만족시의 나머지

```
잘못된 witness에서:
  a(ωᵢ)·b(ωᵢ) - c(ωᵢ) ≠ 0  (어떤 i에서)

  그러면 p(ωᵢ) ≠ 0
  → t(x) = ∏(x - ωᵢ)가 p(x)를 나누지 못함
  → 나머지 rem(x) ≠ 0
  → compute_h() 반환값: None

코드에서:
  let (h, rem) = p.div_rem(&qap.t);
  if rem.is_zero() {
      Some(h)
  } else {
      None
  }

직관:
  "QAP 만족 ⟺ p(x)가 t(x)의 배수"
  "나머지가 0이 아니면 증명 불가"
  → 이것이 Groth16의 "건전성"의 첫 번째 방어선
```

---

### Part 39: Groth16과 BLS 서명의 구조적 유사성

#### BLS 서명의 구조

```
BLS 서명 (Boneh-Lynn-Shacham, 2001):

  Setup:
    sk = 랜덤 Fr 원소 (비밀키)
    pk = sk · G₂          (공개키)

  Sign(sk, m):
    H = hash_to_G1(m)     (메시지를 G1 점으로 해시)
    σ = sk · H             (서명 = 비밀키 × 해시)

  Verify(pk, m, σ):
    H = hash_to_G1(m)
    e(σ, G₂) ?= e(H, pk)

  검증 이유:
    e(σ, G₂) = e(sk·H, G₂) = e(H, G₂)^sk
    e(H, pk) = e(H, sk·G₂) = e(H, G₂)^sk
    → 같다!
```

#### Groth16과의 구조적 비교

```
Groth16                          BLS
────────────────────────        ────────────────────
Secret:  witness w               Secret:  sk
         + toxic waste                    (= 비밀키)

Public:  VK + public_inputs      Public:  pk + message

Proof:   (A, B, C) ∈ G1×G2×G1   Signature:  σ ∈ G1

Verify:  e(A, B) = ...          Verify:  e(σ, G₂) = e(H, pk)
         (3 pairings)                     (2 pairings)

공통점:
  1. BN254 위에서 동작
  2. 페어링으로 검증
  3. 비밀(witness/sk)을 모르면서 관계를 확인
  4. 비밀이 증명/서명에서 드러나지 않음

차이점:
  BLS: "비밀키를 아는지" 검증 (단순)
  Groth16: "QAP를 만족하는 witness를 아는지" 검증 (복잡)

  BLS: trusted setup 불필요
  Groth16: trusted setup 필요

  BLS: 서명 집계 가능 (aggregate signatures)
  Groth16: 증명 집계는 재귀로만 가능

수학적 공통 기반:
  두 프로토콜 모두 "이산 로그의 어려움 + 쌍선형성" 활용
  페어링의 쌍선형성이 "곱셈 관계"를 "커브 위에서 검증" 가능하게 함
```

> [!tip] 페어링 기반 프로토콜의 통일적 이해
> ```
> 페어링은 "커브 위의 값들 사이의 곱셈 관계를 검증"하는 도구
>
>   BLS 서명:    e(sk·H, G₂) = e(H, sk·G₂)
>                "sk가 양쪽에 같다"
>
>   Groth16:     e(α+a(τ)+rδ, β+b(τ)+sδ) = e(α,β)·e(IC,γ₂)·e(C,δ₂)
>                "a(τ)·b(τ) = c(τ) + h(τ)·t(τ)"
>
>   KZG commit:  e([f(τ)]₁, G₂) = e(G₁, [τ]₂)^f(1)  (단순화)
>                "다항식이 올바르게 평가되었다"
>
> 모든 페어링 기반 프로토콜은 이 패턴:
>   "커브 포인트의 이산 로그 관계를 GT에서 확인"
> ```

---

### Part 40: h(x) = 0인 경우의 심화 분석

#### 언제 h(x) = 0인가?

```
h(x) = (a(x)·b(x) - c(x)) / t(x) = 0

이것은 a(x)·b(x) = c(x) 일 때 발생

a(x)·b(x) - c(x) = 0은 "영 다항식"
→ t(x)로 나누면 h(x) = 0

언제 이런 일이 발생하는가?

  Case 1: 단일 제약 (m=1)
    도메인 {ω₁}, t(x) = (x - ω₁)
    a(x), b(x), c(x)는 모두 상수 (degree 0)
    a(x)·b(x) - c(x) = 상수

    R1CS 만족이면: a(ω₁)·b(ω₁) = c(ω₁)
    상수끼리의 관계이므로 어디서든 성립
    → a(x)·b(x) - c(x) = 0 (영 다항식)
    → h(x) = 0

    예: x·y = z 회로 (m=1)
    a(x) = 3, b(x) = 4, c(x) = 12
    3·4 - 12 = 0 → h(x) = 0  ✓

  Case 2: 빈 회로 (m=0)
    도메인 = {}, t(x) = 1
    모든 다항식이 0
    h(x) = (0·0 - 0) / 1 = 0  ✓

  Case 3: 제약은 여러 개이지만 p(x)가 정확히 t(x)의 배수
    이 경우 나눗셈이 떨어지므로 h(x) ≠ 0일 수도 있고 = 0일 수도 있음
    일반적으로 m ≥ 2이면 h(x) ≠ 0

h(x) = 0이 Groth16에 미치는 영향:

  prove에서:
    h기여분 = Σᵢ hᵢ · h_query[i] = 0 (h가 영)
    → C에서 h 기여분 없음
    → C = private_sum + blinding만으로 구성

  검증에서:
    e(C, [δ]₂) 항에 h(τ)t(τ) 부분이 빠짐
    하지만 원래 h(τ)t(τ) = 0이므로 문제없음:
    검증 방정식은 여전히 성립

    이것은 F₁₃ 수치 추적 (Part 30)에서 이미 확인했다.
```

---

### Part 41: Groth16의 계산 복잡도 상세 분석

#### Setup 복잡도

```
setup(qap, rng) 연산 분석:

  n = num_variables, m = 제약 수

  ① toxic waste 생성:
    5 × random_fr → O(1)

  ② 기본 커브 포인트:
    7 × scalar_mul → O(7 × 256) = O(1)

  ③ QAP 다항식 평가:
    n개의 다항식을 τ에서 평가
    각 다항식 degree ≤ m-1
    → Horner: O(m) per polynomial
    → 총: O(nm)

    3 종류 (a, b, c) × n개 = 3n회
    → O(3nm) = O(nm)

  ④ Query 벡터:
    a_query: n × scalar_mul(G1) → O(256n)
    b_g1_query: n × scalar_mul(G1) → O(256n)
    b_g2_query: n × scalar_mul(G2) → O(256n × 2)  (G2는 Fp2)
    → O(n)

  ⑤ IC + L:
    n × (3 곱셈 + 2 덧셈 + 1 역원 곱셈 + 1 scalar_mul)
    → O(n)

  ⑥ h_query:
    (m-1) × scalar_mul → O(m)

  ⑦ e(α, β):
    1 × pairing → O(1)

  총 setup 복잡도: O(nm + n)
    (다항식 평가가 지배적)

  프로덕션: FFT로 다항식 평가 O(n log m) + MSM으로 query O(n/log n)
```

#### Prove 복잡도

```
prove(pk, qap, witness, rng) 연산 분석:

  ① h(x) 계산:
    compute_witness_polys: O(nm) (n개의 다항식 × m 계수)
    다항식 곱셈 a·b: O(m²) (schoolbook) / O(m log m) (FFT)
    다항식 나눗셈 p/t: O(m²) / O(m log m)
    → O(nm + m²)

  ② 블라인딩: O(1)

  ③ A = Σ wⱼ · a_query[j]: n × scalar_mul
    → O(n × 256) 점 덧셈

  ④ B = Σ wⱼ · b_g2_query[j]: n × scalar_mul(G2)
    → O(n × 256 × 2)

  ⑤ B' = Σ wⱼ · b_g1_query[j]: n × scalar_mul
    → O(n × 256)

  ⑥ C = Σ private + Σ h + blinding:
    private: (n-ℓ-1) × scalar_mul → O(n × 256)
    h: (m-1) × scalar_mul → O(m × 256)
    blinding: O(1)

  총 prove 복잡도: O(nm + m² + 4n × 256)
    → 실질적으로 MSM이 지배적: O(n)

  프로덕션: MSM O(n/log n) + FFT O(m log m)

크기별 예시:
  ┌──────────────┬──────────┬───────────┬────────────┐
  │ 회로 크기 (n) │ 우리 구현  │ arkworks   │ GPU (cuMSM)│
  ├──────────────┼──────────┼───────────┼────────────┤
  │ n = 100      │ ~10ms    │ ~1ms      │ ~0.1ms     │
  │ n = 10,000   │ ~1s      │ ~50ms     │ ~5ms       │
  │ n = 1,000,000│ ~100s    │ ~3s       │ ~200ms     │
  │ n = 10⁷      │ 불가능    │ ~30s      │ ~2s        │
  └──────────────┴──────────┴───────────┴────────────┘
  (추정치, 실제 시간은 하드웨어에 따라 다름)
```

#### Verify 복잡도

```
verify(vk, public_inputs, proof) 연산 분석:

  ① IC_sum:
    ℓ × scalar_mul(G1) + ℓ × 점 덧셈
    → O(ℓ × 256)

  ② 페어링 3회:
    각 페어링: Miller loop O(log p) + final exp
    → O(3)  (상수)

  ③ Fp12 곱셈 2회:
    → O(1)

  총 verify 복잡도: O(ℓ)
    ℓ이 작으면 (보통 1~10) → O(1)

  이것이 "Succinct verification"의 의미:
    n = 10⁷인 회로도 ℓ = 1이면 검증 시간 동일!

  BN254 페어링 벤치마크:
    Miller loop: ~0.5ms
    Final exp: ~0.3ms
    페어링 1회: ~0.8ms
    검증 전체: ~2.4ms (3회 × 0.8ms)

  Ethereum EVM에서:
    ecPairing: ~8ms (프리컴파일)
    → 블록 시간 12s의 0.07%
```

---

### Part 42: 재귀적 SNARK (Recursive SNARK) 개요

#### Groth16의 재귀

```
재귀적 SNARK의 아이디어:

  "Groth16 검증 자체를 Groth16 회로 안에 넣는다"

  즉: "이 증명이 유효하다"는 것의 증명을 만든다

  ┌──────────────────────────────────────────────────────┐
  │  외부 회로:                                            │
  │    입력: VK, public_inputs, proof (A, B, C)            │
  │    계산: verify(VK, public_inputs, proof) == true       │
  │    출력: true                                          │
  │                                                        │
  │  이것을 R1CS → QAP → Groth16으로 증명하면:              │
  │    "원래 증명이 유효하다"는 것의 증명 (meta-proof)        │
  │    크기: 여전히 256 bytes!                              │
  └──────────────────────────────────────────────────────┘

문제: 페어링을 회로 안에서 구현해야 함

  페어링의 R1CS 비용:
    BN254 페어링 1회 ≈ ~100만 제약
    Groth16 검증 = 3 페어링 ≈ ~300만 제약

  → 재귀 Groth16 회로가 매우 큼
  → 증명 생성 시간이 매우 김 (수 분 ~ 수십 분)

대안: 사이클 오브 커브 (Cycle of Curves)

  BN254 위의 Groth16 검증을 MNT4/MNT6 위에서 증명
  → MNT 위의 검증을 BN254 위에서 증명
  → "커브 사이클"로 재귀

  하지만 MNT 커브는 보안이 약함 (임베딩 차수가 작음)
  → 실용적이지 않음

현대적 접근:

  1. Groth16 + FFLONK:
     Groth16 증명을 FFLONK 내부에서 검증
     → FFLONK가 Groth16보다 재귀에 유리 (KZG 기반)

  2. IVC (Incrementally Verifiable Computation):
     Nova, SuperNova 등
     → 재귀를 위한 특화 구조

  3. STARK → SNARK 래핑:
     STARK으로 빠르게 증명 → Groth16/FFLONK으로 래핑
     → L1 검증 비용 최소화 (STARK 증명은 크지만 래핑 후 작음)
```

> [!important] 재귀적 증명의 실용적 의의
> ```
> 1. 증명 집계:
>    N개의 증명을 1개로 집약 → N배 절약
>
> 2. 증분 검증:
>    이전 상태의 증명 + 새 트랜잭션 → 새 상태의 증명
>    → 블록체인 전체 상태를 하나의 증명으로 표현
>
> 3. 크로스체인 증명:
>    "체인 A의 상태가 올바르다"를 체인 B에서 검증
>    → 브릿지의 보안 강화
>
> Groth16이 여기서 하는 역할:
>    "최종 래퍼"로서 L1에서 검증되는 최소 크기 증명
>    내부 구조는 STARK/PLONK → 마지막에 Groth16으로 래핑
> ```

---

### Part 43: 우리 구현의 코드 패턴 분석

#### 패턴 1: is_zero() 최적화

```rust
for j in 0..pk.num_variables {
    if !witness[j].is_zero() {        // ← 0이면 건너뛰기
        proof_a = proof_a
            + pk.a_query[j].scalar_mul(&witness[j].to_repr());
    }
}
```

```
이 패턴이 반복되는 곳:
  1. A 계산 (witness × a_query)
  2. B 계산 (witness × b_g2_query)
  3. B' 계산 (witness × b_g1_query)
  4. C의 private 기여 (witness × l_query)
  5. C의 h 기여 (h_coeffs × h_query)
  6. IC_sum 계산 (public_inputs × ic)

총 6곳에서 사용

왜 이 최적화가 효과적인가?

  scalar_mul은 O(256) 연산 (가장 비싼 연산)
  0 × P = O (항등원) → 계산할 필요 없음

  실제 회로에서 sparse한 경우가 많음:
    R1CS 행렬의 각 행에 0이 아닌 항이 3-5개
    → 각 변수의 다항식도 대부분 0
    → 많은 aⱼ(τ) = 0 또는 witness[j] = 0

  실측 효과:
    CubicCircuit (n=5): 최대 5번 scalar_mul 중 ~3번 건너뜀
    일반 회로: ~50-80% 건너뜀 (sparse R1CS의 특성)
```

#### 패턴 2: Fr ↔ scalar_mul 브릿지

```rust
pk.a_query[j].scalar_mul(&witness[j].to_repr())
//                                   ^^^^^^^^
//                                   Montgomery → raw 변환
```

```
이 패턴이 필수적인 이유:

  Fr 내부: Montgomery representation
    Fr::from_u64(5) → 내부 저장 = 5 × R mod r
    (R = 2^256 mod r)

  scalar_mul 입력: raw [u64; 4] representation
    "실제 값" 5를 비트로 표현한 것을 기대

  to_repr()이 하는 일:
    Montgomery form (value × R mod r)
    → 역변환 (× R⁻¹ mod r)
    → raw form (value)

  만약 to_repr()을 빼먹으면:
    scalar_mul에 value × R 이 입력됨
    → (value × R) · G  (R배 틀림!)
    → 검증 방정식 불만족

우리 코드에서 to_repr() 호출 횟수:
  setup: ~7 + 3n + (ℓ+1) + (n-ℓ-1) + (m-1) ≈ 4n + m
  prove: ~3n + (n-ℓ-1) + (m-1) + 4 ≈ 4n + m
  verify: ℓ회

  모든 scalar_mul마다 반드시 to_repr() 호출!
```

#### 패턴 3: Option을 활용한 에러 처리

```rust
pub fn prove<R: Rng>(...) -> Option<Proof> {
    let h = qap.compute_h(witness)?;    // ← ? 연산자
    ...
    Some(Proof { a: proof_a, b: proof_b, c: proof_c })
}
```

```
Rust의 ? 연산자 활용:

  compute_h(witness)가 None을 반환하면:
    → prove 함수도 즉시 None을 반환
    → "QAP 불만족 = 증명 불가" 를 자연스럽게 표현

  왜 panic이 아닌 Option인가?
    잘못된 witness는 "프로그래밍 에러"가 아닌 "논리적 실패"
    → 호출자가 처리할 수 있어야 함
    → Option이 적절

  반면, assert_eq!(witness.len(), pk.num_variables)는:
    길이 불일치는 "프로그래밍 에러" (항상 사전 검증 가능)
    → panic이 적절

디자인 원칙:
  "복구 가능한 실패" → Option/Result
  "프로그래밍 에러" → panic (assert)
  Groth16에서 이 둘의 경계가 명확하게 구현됨
```

---

### Part 44: Groth16과 Zcash — 역사적 맥락

```
Groth16의 역사와 채택:

  2016: Jens Groth 논문 "On the Size of Pairing-based Non-interactive
        Arguments" 발표 (EUROCRYPT 2016)

  2017: Zcash의 Sapling 프로토콜에 Groth16 채택 결정
        (기존 PHGR13 증명 시스템 대체)

  2018: Zcash Sapling 배포
        - Spend circuit: ~100,000 제약
        - Output circuit: ~50,000 제약
        - 증명 크기: 192 bytes (BLS12-381 사용)
        - 증명 시간: ~7초 (당시 하드웨어)
        - 검증 시간: ~10ms

  2019: Ethereum Constantinople에서 BN254 프리컴파일 추가
        → EVM에서 Groth16 검증 가능

  2020-현재:
        - zkSync: Groth16 기반 zk-rollup
        - Polygon zkEVM: Groth16 + FFLONK 하이브리드
        - Scroll: Groth16 래퍼
        - Tornado Cash: Groth16 기반 mixer

왜 BLS12-381 vs BN254?

  Zcash → BLS12-381:
    - 128비트 보안 수준 확실
    - Groth16 증명 192 bytes (G1 48B × 2 + G2 96B × 1)
    - EVM 프리컴파일 없음

  Ethereum → BN254:
    - 100비트 보안 (Kim-Barbulescu 공격 후 하향)
    - Groth16 증명 256 bytes (G1 64B × 2 + G2 128B × 1)
    - EVM 프리컴파일 있음 (EIP-196/197)
    - 실용적으로 충분한 보안

  우리 구현: BN254 (Ethereum 호환)
```

---

### Part 45: 전체 의존성 그래프와 모듈 아키텍처

```
Groth16 전체 모듈 의존성:

  groth16.rs
    │
    ├──→ field/fr.rs (Fr)
    │      │ from_raw, from_u64, to_repr
    │      │ 산술: +, -, *, inv, is_zero
    │      │ 상수: ZERO, ONE
    │      │
    │      └──→ field/mod.rs (define_prime_field! 매크로)
    │             └──→ Montgomery 연산 (adc, sbb, mac)
    │
    ├──→ field/fp12.rs (Fp12)
    │      │ GT 원소 (페어링 결과)
    │      │ 산술: *, ==
    │      │
    │      └──→ fp6, fp2, fp 타워
    │
    ├──→ curve/g1.rs (G1)
    │      │ scalar_mul, add, neg, identity, generator
    │      │ Jacobian → Affine 변환
    │      │
    │      └──→ field/fp.rs (좌표 연산)
    │
    ├──→ curve/g2.rs (G2)
    │      │ (G1과 동일 API, Fp2 좌표)
    │      │
    │      └──→ field/fp2.rs
    │
    ├──→ curve/pairing.rs
    │      │ pairing(G1, G2) → Fp12
    │      │ Miller loop + final exponentiation
    │      │
    │      └──→ G1Affine, G2Affine, Fp12, line_eval
    │
    └──→ qap.rs (QAP, Polynomial)
           │ from_r1cs, compute_h, compute_witness_polys
           │ Polynomial: eval, div_rem, +-*
           │
           └──→ r1cs.rs (ConstraintSystem)
                  │ ConstraintSystem, Circuit trait
                  │ Variable, LinearCombination
                  │
                  └──→ field/fr.rs

전체 호출 체인:

  사용자 코드:
    1. Circuit 구현 (synthesize)
    2. ConstraintSystem::new() + synthesize
    3. QAP::from_r1cs(&cs)
    4. groth16::setup(&qap, &mut rng)
    5. groth16::prove(&pk, &qap, &cs.values, &mut rng)
    6. groth16::verify(&vk, &public_inputs, &proof)

  내부 호출 (prove 기준):
    prove
    → qap.compute_h(witness)
      → compute_witness_polys(witness)
        → Polynomial::from_coeffs (n × m 연산)
      → Polynomial::mul (a × b)
      → Polynomial::sub (- c)
      → Polynomial::div_rem (/ t)
    → G1::scalar_mul (× n번)
    → G2::scalar_mul (× n번)
    → Fr::to_repr (× 4n번)
    → Fr 산술 (× m번)
```

```
파일 크기 비교:

  field/fp.rs:   ~700 줄 (수동 구현, 학습용)
  field/fr.rs:   ~38 줄 (매크로로 생성)
  field/fp2.rs:  ~300 줄
  field/fp6.rs:  ~400 줄
  field/fp12.rs: ~300 줄
  curve/g1.rs:   ~350 줄
  curve/g2.rs:   ~340 줄
  curve/pairing.rs: ~290 줄
  r1cs.rs:       ~300 줄
  circuits/:     ~200 줄
  qap.rs:        ~1060 줄
  groth16.rs:    ~850 줄

  전체: ~5,000줄

  이 5,000줄이 구현하는 것:
    유한체 → 확장체 → 타원곡선 → 페어링 → R1CS → QAP → Groth16
    = ZK-SNARK의 전체 스택!
```

> [!important] 5,000줄로 ZK-SNARK 전체 스택
> ```
> 1단계: Fp (700줄) — 기본 유한체 산술
> 2단계: Fr, Fp2, Fp6, Fp12 (1,038줄) — 확장체 타워
> 3단계: G1, G2 (690줄) — 타원곡선 점 연산
> 4단계: pairing (290줄) — Miller loop + final exp
> 5단계: R1CS + circuits (500줄) — 제약 시스템
> 6단계: QAP (1,060줄) — 다항식 변환
> 7단계: Groth16 (850줄) — 증명 시스템
>
> 각 단계가 이전 단계 위에 쌓이는 "타워 구조"
> 어떤 외부 라이브러리도 사용하지 않음 (rand/rand_chacha 제외)
>
> "교육용으로 처음부터 구현한 완전한 ZK-SNARK"
> 이것이 이 프로젝트의 핵심 가치이다.
> ```
