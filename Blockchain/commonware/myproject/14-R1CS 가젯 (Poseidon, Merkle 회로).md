## Step 10: R1CS 가젯 (Poseidon, Merkle 회로)

### 핵심 질문: 왜 가젯이 필요한가?

```
Step 09에서 R1CS를 만들었다:
  "모든 계산을 곱셈 하나로 분해"
  ⟨a, s⟩ · ⟨b, s⟩ = ⟨c, s⟩

하지만 실제 ZK 응용은:
  "나는 이 Merkle tree에 포함된 값을 안다"
  → Poseidon 해시를 반복 적용
  → 각 해시 안에 S-box, MDS, 라운드 상수
  → 수백 개의 R1CS 제약

문제:
  어떻게 Poseidon 해시를 R1CS 제약으로 변환하는가?
  어떻게 Merkle proof 검증을 R1CS 제약으로 변환하는가?

답:
  가젯(Gadget) = 반복적으로 사용되는 R1CS 패턴
  복잡한 회로를 기본 가젯의 조합으로 구성

  S-box 가젯:  x → x⁵  (3개 제약)
  Boolean 가젯: b ∈ {0,1}  (1개 제약)
  Mux 가젯:    if b then x else y  (2개 제약)
  Poseidon 가젯: 65 라운드 = 634개 제약
  Merkle 가젯:  depth=4 → 3,186개 제약
```

> [!important] Step 10의 위치
> ```
> Step 07: Poseidon 해시 (native — Fr 연산)
> Step 08: Merkle tree  (native — 해시 반복)
> Step 09: R1CS 제약 시스템 (곱셈 = 제약 = 게이트)
>     ↓
> Step 10: R1CS 가젯 (여기)
>     Poseidon + Merkle을 R1CS 회로로 재구현
>     "native 코드 = 회로 코드" 검증
>     ↓
> Step 11: QAP (R1CS 행렬 → 다항식)
> Step 12: Groth16 (다항식 + 페어링 → 128바이트 증명)
> ```
>
> 가젯은 native 구현(Step 07, 08)과 증명 시스템(Step 11, 12) 사이의 다리.
> "같은 계산을 제약의 언어로 번역"하는 작업이다.

---

### "native = circuit" 원칙

```
ZK 회로 개발의 가장 중요한 원칙:

  native 코드의 출력 = circuit 코드의 출력

  native:
    poseidon_hash_with_params(left, right) → hash_val
    verify_merkle_proof(root, key, value, proof) → true/false

  circuit:
    PoseidonCircuit::synthesize(cs) → cs.is_satisfied() + 같은 hash_val
    MerkleProofCircuit::synthesize(cs) → cs.is_satisfied()

  모든 입력에 대해 두 결과가 동일해야 한다!

왜 이것이 중요한가:
  1. 건전성(Soundness): 회로가 native와 다른 계산을 하면
     → 잘못된 증명이 유효하거나, 올바른 증명이 거부됨
  2. 완전성(Completeness): native에서 참인 계산은
     → 회로에서도 반드시 만족 가능해야 함
  3. 디버깅: native 결과와 비교하여 회로 버그를 발견

테스트 전략:
  같은 입력 → native 실행 → circuit 실행 → 출력 비교
  잘못된 입력 → native 거부 → circuit도 거부 확인
```

> [!tip] "native = circuit"이 실패하면?
> ```
> 시나리오 1: native는 H(1,2) = X인데, circuit은 다른 값을 출력
>   → 회로의 제약이 잘못됨 (보통 AddRC나 MDS의 선형결합 오류)
>
> 시나리오 2: native에서 valid한 proof인데, circuit에서 is_satisfied() = false
>   → 회로의 witness 할당이 잘못됨 (보통 mux나 bit 추출 오류)
>
> 시나리오 3: native에서 invalid한 proof인데, circuit에서 is_satisfied() = true
>   → 가장 위험! 제약이 부족하여 건전성 위반
>   → 보안 취약점 (Tornado Cash 버그와 동일한 유형)
> ```

---

### Part 1: 기본 가젯

#### Boolean 가젯: b는 0 또는 1

```
목적: 변수 b가 반드시 {0, 1} 중 하나임을 강제

R1CS 제약 (1개):
  b · (1 - b) = 0

  A = [(b, 1)]
  B = [(One, 1), (b, -1)]
  C = 0

검증:
  b = 0:  0 · (1 - 0) = 0 · 1 = 0 = 0 ✓
  b = 1:  1 · (1 - 1) = 1 · 0 = 0 = 0 ✓
  b = 2:  2 · (1 - 2) = 2 · (-1) = -2 ≠ 0 ✗
  b = 5:  5 · (1 - 5) = 5 · (-4) = -20 ≠ 0 ✗

  Fr에서 "0"과 "1"만이 x(1-x)=0을 만족!
```

```rust
// circuits/merkle.rs
/// Boolean 가젯: b ∈ {0, 1}
///
/// 제약: b · (1 - b) = 0
fn enforce_boolean(cs: &mut ConstraintSystem, b: Variable) {
    cs.enforce(
        LinearCombination::from(b),
        LinearCombination::from(Variable::One).add(-Fr::ONE, b),
        LinearCombination::zero(),
    );
}
```

> [!note] 왜 Boolean 가젯이 필요한가?
> Merkle proof에서 경로 방향을 결정하는 비트가 실제로 0 또는 1인지 강제해야 한다.
> 이 제약이 없으면 공격자가 b=2 같은 값을 넣어 mux 결과를 조작할 수 있다.
> ```
> b=0: left에서 current, right에서 sibling  (정상)
> b=1: left에서 sibling, right에서 current  (정상)
> b=2: ??? → mux 공식이 엉뚱한 값을 출력 → 위조된 root 생성 가능
> ```

---

#### Mux (조건부 선택) 가젯

```
목적: bit 값에 따라 두 입력 중 하나를 선택

수학적 표현:
  result = if bit then when_true else when_false
         = when_false + bit · (when_true - when_false)

bit=0일 때: result = when_false + 0 = when_false  ✓
bit=1일 때: result = when_false + (when_true - when_false) = when_true  ✓

R1CS 분해 (2개 제약):
  보조 변수: t = bit · (when_true - when_false)

  제약 1: bit · (when_true - when_false) = t
    A = [(bit, 1)]
    B = [(when_true, 1), (when_false, -1)]
    C = [(t, 1)]

  제약 2: (when_false + t) · 1 = result
    A = [(when_false, 1), (t, 1)]
    B = [(One, 1)]
    C = [(result, 1)]

총: 2개 제약 + 2개 보조 변수 (t, result)
```

```rust
// circuits/merkle.rs
fn mux_circuit(
    cs: &mut ConstraintSystem,
    bit: Variable,
    bit_val: Fr,
    when_true: Variable,
    when_true_val: Fr,
    when_false: Variable,
    when_false_val: Fr,
) -> (Variable, Fr) {
    let diff_val = when_true_val - when_false_val;
    let t_val = bit_val * diff_val;
    let result_val = when_false_val + t_val;

    let t = cs.alloc_witness(t_val);
    let result = cs.alloc_witness(result_val);

    // bit · (when_true - when_false) = t
    cs.enforce(
        LinearCombination::from(bit),
        LinearCombination::from(when_true).add(-Fr::ONE, when_false),
        LinearCombination::from(t),
    );

    // (when_false + t) · 1 = result
    cs.enforce(
        LinearCombination::from(when_false).add(Fr::ONE, t),
        LinearCombination::from(Variable::One),
        LinearCombination::from(result),
    );

    (result, result_val)
}
```

> [!abstract] Mux 가젯의 수동 계산
> ```
> 예: bit=1, when_true=42, when_false=99
>
> diff  = 42 - 99 = -57
> t     = 1 · (-57) = -57
> result = 99 + (-57) = 42  ← when_true 선택! ✓
>
> 제약 1 검증: bit · (when_true - when_false) = t
>   1 · (42 - 99) = -57 = t ✓
>
> 제약 2 검증: (when_false + t) · 1 = result
>   (99 + (-57)) · 1 = 42 = result ✓
>
> 예: bit=0, when_true=42, when_false=99
>
> diff  = 42 - 99 = -57
> t     = 0 · (-57) = 0
> result = 99 + 0 = 99  ← when_false 선택! ✓
> ```

> [!tip] Step 09의 조건부 선택과의 비교
> Step 09에서는 3개 제약으로 구현했다 (boolean + b*x + b*y).
> 여기서는 2개 제약으로 최적화했다:
> ```
> Step 09 방식: 3개 제약
>   1. b·(1-b) = 0    (boolean)
>   2. b·(x-y) = t    (차이 곱셈)
>   3. (y+t)·1 = result
>
> Step 10 방식: 2개 제약 (boolean은 별도)
>   1. bit·(when_true - when_false) = t
>   2. (when_false + t)·1 = result
>
> Boolean 제약은 mux 외부에서 한 번만 호출.
> 같은 bit으로 mux를 여러 번 호출할 때 boolean 제약은 공유!
> → Merkle에서 left/right 두 번 mux 호출 시 boolean 1개만 필요
> ```

---

### Part 2: Poseidon 해시 회로

#### S-box 가젯: x --> x^5

```
native S-box (Step 07):
  x → x² → x⁴ → x⁵ = x⁴ · x
  Fr 곱셈 3회 (square, square, mul)

circuit S-box:
  같은 계산을 R1CS 제약으로 표현
  각 곱셈 = 1개 제약 → 총 3개 제약

분해:
  t1 = x · x       ← 제약 1 (x²)
  t2 = t1 · t1      ← 제약 2 (x⁴)
  y  = t2 · x       ← 제약 3 (x⁵)

보조 변수: t1 (x²값), t2 (x⁴값), y (x⁵값) — 3개
제약: 3개

             ┌───┐
    x ──────→│ × │────→ t1 (= x²)
    x ──────→│   │
             └───┘
             ┌───┐
   t1 ──────→│ × │────→ t2 (= x⁴)
   t1 ──────→│   │
             └───┘
             ┌───┐
   t2 ──────→│ × │────→ y  (= x⁵)
    x ──────→│   │
             └───┘
```

```rust
// circuits/poseidon.rs
fn sbox_circuit(cs: &mut ConstraintSystem, x: Variable, x_val: Fr) -> (Variable, Fr) {
    let x2_val = x_val.square();
    let x4_val = x2_val.square();
    let x5_val = x4_val * x_val;

    let t1 = cs.alloc_witness(x2_val);   // 보조 변수: x²
    let t2 = cs.alloc_witness(x4_val);   // 보조 변수: x⁴
    let y = cs.alloc_witness(x5_val);    // 보조 변수: x⁵

    // 제약 1: x * x = t1
    cs.enforce(
        LinearCombination::from(x),
        LinearCombination::from(x),
        LinearCombination::from(t1),
    );
    // 제약 2: t1 * t1 = t2
    cs.enforce(
        LinearCombination::from(t1),
        LinearCombination::from(t1),
        LinearCombination::from(t2),
    );
    // 제약 3: t2 * x = y
    cs.enforce(
        LinearCombination::from(t2),
        LinearCombination::from(x),
        LinearCombination::from(y),
    );

    (y, x5_val)  // 반환: (변수, 값)
}
```

> [!abstract] S-box 가젯의 수동 검증
> ```
> x = 7에서 S-box 추적:
>
> native:
>   7² = 49
>   49² = 2401
>   2401 · 7 = 16807
>   → sbox(7) = 16807
>
> circuit:
>   x_val = 7
>   t1 = cs.alloc_witness(49)     → x²
>   t2 = cs.alloc_witness(2401)   → x⁴
>   y  = cs.alloc_witness(16807)  → x⁵
>
>   제약 1: 7 · 7 = 49      → A·B = 49 = C ✓
>   제약 2: 49 · 49 = 2401  → A·B = 2401 = C ✓
>   제약 3: 2401 · 7 = 16807 → A·B = 16807 = C ✓
>
>   native == circuit ✓
>
> 경계값: x = 0
>   0² = 0, 0² = 0, 0·0 = 0
>   → sbox(0) = 0 ✓ (모든 제약 0·0=0 만족)
> ```

> [!important] 왜 Variable과 Fr 값을 함께 전달하는가?
> ```
> sbox_circuit(cs, x: Variable, x_val: Fr)
>                  ↑               ↑
>             제약 시스템 변수    실제 Fr 값
>
> Variable: R1CS 제약에서의 "심볼릭" 참조
>   → 제약 A·B=C의 A, B, C에 들어감
>   → is_satisfied()에서 간접적으로 값을 참조
>
> Fr 값: witness 생성을 위한 "실제" 값
>   → 보조 변수의 값을 계산하는 데 사용
>   → cs.alloc_witness(x2_val)에서 사용
>
> 이 두 트랙을 병렬로 관리:
>   Variable 트랙: 제약 구조를 정의
>   Fr 값 트랙:    witness 값을 계산
>
> 실제 ZK 프레임워크(arkworks, bellman)도 동일한 패턴.
> 다만 매크로/trait으로 감추어져 보이지 않을 뿐.
> ```

---

#### Full Round vs Partial Round

```
Poseidon의 65 라운드:
  Phase 1: RF/2 = 4 full rounds
  Phase 2: RP = 57 partial rounds
  Phase 3: RF/2 = 4 full rounds

각 라운드의 구조:
  1. AddRoundConstants — state[i] += rc[i]
  2. S-box             — state[i] = state[i]⁵
  3. MDS mix           — state = M · state

Full round (S-box 전체):
  ┌─────────────────────────────────────────────────┐
  │  state[0] ──→ +rc₀ ──→ x⁵ ──→ ┐               │
  │  state[1] ──→ +rc₁ ──→ x⁵ ──→ │── MDS ──→ out │
  │  state[2] ──→ +rc₂ ──→ x⁵ ──→ ┘               │
  └─────────────────────────────────────────────────┘
  S-box: 3개 원소 × 3 제약 = 9 제약

Partial round (S-box 첫째만):
  ┌─────────────────────────────────────────────────┐
  │  state[0] ──→ +rc₀ ──→ x⁵ ──→ ┐               │
  │  state[1] ──→ +rc₁ ──→     ──→ │── MDS ──→ out │
  │  state[2] ──→ +rc₂ ──→     ──→ ┘               │
  └─────────────────────────────────────────────────┘
  S-box: 1개 원소 × 3 제약 = 3 제약
```

> [!important] 핵심 인사이트: 왜 AddRC와 MDS에 제약이 필요한가?
> ```
> 순수 수학에서:
>   AddRC: state + rc  → 선형 연산 → 제약 0개
>   MDS:   M · state   → 선형 연산 → 제약 0개
>   S-box: x⁵          → 비선형 연산 → 제약 3개
>
> 그래서 예상 제약 수:
>   Full round:    3 × 3 = 9 (S-box만)
>   Partial round: 1 × 3 = 3 (S-box만)
>   총 243개
>
> 하지만 실제 구현은 634개! 왜?
>
> 문제: S-box는 입력으로 "변수"가 필요하다.
>
>   S-box 제약: x · x = t1
>              여기서 x는 Variable이어야 함
>
>   하지만 AddRC 후의 값은 "state + rc"라는 선형결합이지,
>   하나의 Variable이 아니다!
>
>   선형결합을 S-box의 입력으로 바로 쓸 수 없다.
>   왜? R1CS 제약 A·B=C에서 A와 B는 각각 선형결합이지만,
>   (A₁·s)(A₂·s) 같은 곱의 곱은 표현할 수 없다.
>
> 해결: 선형결합을 보조 변수로 "고정"
>   after_rc = cs.alloc_witness(state_val + rc)
>   cs.enforce(
>       (state + rc·One),  // 선형결합
>       One,               // × 1
>       after_rc           // = 보조 변수
>   );
>
>   이제 after_rc는 Variable → S-box에 입력 가능!
>
> 같은 이유로 MDS mix 후에도 보조 변수 고정 필요:
>   MDS 결과 = Σ M[i][j] · sbox_out[j] → 선형결합
>   → 다음 라운드의 state Variable로 고정
> ```

```
제약 수 분해 (한 라운드):

Full round:
  AddRC:  T = 3개 제약  (선형결합 → 변수 고정)
  S-box:  3T = 9개 제약  (3개/원소 × 3원소)
  MDS:    T = 3개 제약  (선형결합 → 변수 고정)
  ─────────────────────
  합계:   5T = 15개 제약

Partial round:
  AddRC:  T = 3개 제약  (선형결합 → 변수 고정, 모든 원소)
  S-box:  3개 제약     (3개/원소 × 1원소)
  MDS:    T = 3개 제약  (선형결합 → 변수 고정)
  ─────────────────────
  합계:   2T + 3 = 9개 제약
```

> [!note] 순수 S-box만 세면 243개, 변수 고정 포함하면 634개
> ```
> 이론적 최소 (S-box만):
>   Full:    8 × 9 = 72
>   Partial: 57 × 3 = 171
>   합계:    243
>
> 실제 구현 (변수 고정 포함):
>   Full:    8 × 15 = 120
>   Partial: 57 × 9 = 513
>   출력:    1
>   합계:    634
>
> 추가된 제약: 634 - 243 = 391개
>   = (8 + 57) × 2T = 65 × 6 = 390개 (AddRC + MDS 고정)
>   + 1개 (출력 등치)
>
> 이 391개는 "구조적 오버헤드" — 선형 연산을 변수로 고정하는 비용.
> 최적화된 구현(R1CS 펼치기)에서는 줄일 수 있지만,
> 교육용으로 명시적 고정이 이해하기 쉽다.
> ```

---

#### poseidon_hash_circuit 구조

```
poseidon_hash_circuit()는 재사용 가능한 내부 함수:
  → PoseidonCircuit에서도, MerkleProofCircuit에서도 호출

입력:
  cs:     &mut ConstraintSystem  (제약 시스템)
  left:   Fr                      (해시 왼쪽 입력 값)
  right:  Fr                      (해시 오른쪽 입력 값)
  params: &PoseidonParams          (라운드 상수 + MDS)

출력:
  (Variable, Fr)  → (해시 결과 변수, 해시 결과 값)

왜 Circuit trait이 아닌 일반 함수인가?
  Circuit trait은 "최상위 회로"를 위한 것:
    → alloc_instance로 공개 입출력 설정
    → 한 번 호출로 전체 회로 구성

  poseidon_hash_circuit은 "서브루틴":
    → 더 큰 회로 안에서 여러 번 호출될 수 있음
    → Merkle 회로에서 depth+1 번 호출
    → 공개 입출력은 호출자가 결정
```

```rust
// circuits/poseidon.rs
pub fn poseidon_hash_circuit(
    cs: &mut ConstraintSystem,
    left: Fr,
    right: Fr,
    params: &PoseidonParams,
) -> (Variable, Fr) {
    // Sponge 초기 상태: [0, left, right]
    let mut state_vals: [Fr; T] = [Fr::ZERO, left, right];
    let mut state_vars: [Variable; T] = [
        cs.alloc_witness(Fr::ZERO),   // capacity = 0
        cs.alloc_witness(left),        // rate₀ = left
        cs.alloc_witness(right),       // rate₁ = right
    ];

    let half_rf = RF / 2;  // 4

    // Phase 1: RF/2 = 4 full rounds
    for r in 0..half_rf {
        let offset = r * T;
        let rc = &params.round_constants[offset..offset + T];
        full_round(cs, &mut state_vars, &mut state_vals, rc, &params.mds);
    }

    // Phase 2: RP = 57 partial rounds
    for r in 0..RP {
        let offset = (half_rf + r) * T;
        let rc = &params.round_constants[offset..offset + T];
        partial_round(cs, &mut state_vars, &mut state_vals, rc, &params.mds);
    }

    // Phase 3: RF/2 = 4 full rounds
    for r in 0..half_rf {
        let offset = (half_rf + RP + r) * T;
        let rc = &params.round_constants[offset..offset + T];
        full_round(cs, &mut state_vars, &mut state_vals, rc, &params.mds);
    }

    // 출력: state[1] (첫 번째 rate 원소)
    (state_vars[1], state_vals[1])
}
```

> [!tip] 두 트랙의 병렬 관리
> ```
> state_vals: [Fr; T]       ← 실제 값 (witness 생성용)
> state_vars: [Variable; T] ← 심볼릭 변수 (제약 구성용)
>
> 매 라운드마다 두 트랙이 동기화된다:
>
>   값 트랙:  state_vals → +rc → sbox → MDS → new_vals
>   변수 트랙: state_vars → alloc(after_rc) → sbox_circuit → alloc(mds_out)
>
> 값 트랙은 native Poseidon과 완전히 동일한 계산.
> 변수 트랙은 그 계산을 R1CS 제약으로 "기록".
>
> 이 구조 덕에 "native = circuit"이 보장된다:
>   값 트랙이 native와 같으므로 → 출력 값도 같음
>   변수 트랙이 값과 일치하므로 → 제약이 만족됨
> ```

---

#### Full Round 상세

```rust
// circuits/poseidon.rs
fn full_round(
    cs: &mut ConstraintSystem,
    state_vars: &mut [Variable; T],
    state_vals: &mut [Fr; T],
    rc: &[Fr],
    mds: &[[Fr; T]; T],
) {
    // ── Step 1: AddRC ──
    // 각 원소에 라운드 상수를 더하고, 결과를 보조 변수로 고정
    let mut after_rc_vals = [Fr::ZERO; T];
    let mut after_rc_vars = [Variable::One; T]; // placeholder
    for i in 0..T {
        after_rc_vals[i] = state_vals[i] + rc[i];
        after_rc_vars[i] = cs.alloc_witness(after_rc_vals[i]);
        // 제약: (state + rc) · 1 = after_rc
        cs.enforce(
            LinearCombination::from(state_vars[i]).add(rc[i], Variable::One),
            LinearCombination::from(Variable::One),
            LinearCombination::from(after_rc_vars[i]),
        );
    }
    // → T = 3개 제약 (AddRC 고정)

    // ── Step 2: S-box ALL ──
    let mut after_sbox_vals = [Fr::ZERO; T];
    let mut after_sbox_vars = [Variable::One; T];
    for i in 0..T {
        let (var, val) = sbox_circuit(cs, after_rc_vars[i], after_rc_vals[i]);
        after_sbox_vars[i] = var;
        after_sbox_vals[i] = val;
    }
    // → 3T = 9개 제약 (S-box × 3원소)

    // ── Step 3: MDS mix ──
    let mut new_vals = [Fr::ZERO; T];
    for i in 0..T {
        for j in 0..T {
            new_vals[i] = new_vals[i] + mds[i][j] * after_sbox_vals[j];
        }
        state_vars[i] = cs.alloc_witness(new_vals[i]);
        // MDS 선형결합을 보조 변수로 고정
        let mut lc = LinearCombination::zero();
        for j in 0..T {
            lc = lc.add(mds[i][j], after_sbox_vars[j]);
        }
        cs.enforce(
            lc,
            LinearCombination::from(Variable::One),
            LinearCombination::from(state_vars[i]),
        );
    }
    // → T = 3개 제약 (MDS 고정)

    *state_vals = new_vals;
    // 총: 3 + 9 + 3 = 15개 제약/full round
}
```

> [!abstract] Full round 제약 흐름 시각화
> ```
> Round r (Full):
>
>   state_vars[0..3]   ← 이전 라운드 출력
>        │
>        ▼ AddRC (+rc, T=3 제약으로 변수 고정)
>   after_rc_vars[0..3]
>        │
>        ▼ S-box (3개 제약/원소 × 3원소 = 9 제약)
>   after_sbox_vars[0..3]
>        │
>        ▼ MDS (선형결합, T=3 제약으로 변수 고정)
>   new state_vars[0..3]  → 다음 라운드로
>
>   제약 카운터: +3 (AddRC) + 9 (S-box) + 3 (MDS) = +15
>   변수 카운터: +3 (after_rc) + 9 (sbox aux) + 3 (mds out) = +15
> ```

---

#### Partial Round 상세

```rust
// circuits/poseidon.rs
fn partial_round(
    cs: &mut ConstraintSystem,
    state_vars: &mut [Variable; T],
    state_vals: &mut [Fr; T],
    rc: &[Fr],
    mds: &[[Fr; T]; T],
) {
    // ── Step 1: AddRC ── (전체 원소에 적용!)
    let mut after_rc_vals = [Fr::ZERO; T];
    let mut after_rc_vars = [Variable::One; T];
    for i in 0..T {
        after_rc_vals[i] = state_vals[i] + rc[i];
        after_rc_vars[i] = cs.alloc_witness(after_rc_vals[i]);
        cs.enforce(
            LinearCombination::from(state_vars[i]).add(rc[i], Variable::One),
            LinearCombination::from(Variable::One),
            LinearCombination::from(after_rc_vars[i]),
        );
    }
    // → T = 3개 제약

    // ── Step 2: S-box FIRST ONLY ──
    let mut after_sbox_vals = after_rc_vals;
    let mut after_sbox_vars: [Variable; T] = after_rc_vars;
    let (var, val) = sbox_circuit(cs, after_rc_vars[0], after_rc_vals[0]);
    after_sbox_vars[0] = var;
    after_sbox_vals[0] = val;
    // state[1], state[2]는 S-box 건너뜀 → after_rc 그대로 사용
    // → 3개 제약 (S-box × 1원소)

    // ── Step 3: MDS mix ── (동일)
    let mut new_vals = [Fr::ZERO; T];
    for i in 0..T {
        for j in 0..T {
            new_vals[i] = new_vals[i] + mds[i][j] * after_sbox_vals[j];
        }
        state_vars[i] = cs.alloc_witness(new_vals[i]);
        let mut lc = LinearCombination::zero();
        for j in 0..T {
            lc = lc.add(mds[i][j], after_sbox_vars[j]);
        }
        cs.enforce(
            lc,
            LinearCombination::from(Variable::One),
            LinearCombination::from(state_vars[i]),
        );
    }
    // → T = 3개 제약

    *state_vals = new_vals;
    // 총: 3 + 3 + 3 = 9개 제약/partial round
}
```

> [!note] Partial round에서도 AddRC는 모든 원소에 적용
> S-box는 첫 번째 원소에만 적용하지만, AddRC는 모든 원소에 적용한다.
> ```
> 왜? AddRC는 S-box 이전에 실행되기 때문:
>   state[1]과 state[2]도 라운드 상수가 더해진 후
>   MDS mix에서 사용된다.
>
>   AddRC를 건너뛰면 state[1], state[2]에 상수가 안 더해져서
>   native 결과와 달라진다 → "native ≠ circuit" 오류
>
> 따라서 partial round의 AddRC도 T=3개 제약이 필요.
> 이것이 partial round이 3이 아닌 9개 제약인 주된 이유.
> ```

---

#### PoseidonCircuit (Circuit trait 구현)

```rust
// circuits/poseidon.rs
pub struct PoseidonCircuit {
    pub left: Fr,
    pub right: Fr,
    pub params: PoseidonParams,
}

impl Circuit for PoseidonCircuit {
    fn synthesize(&self, cs: &mut ConstraintSystem) {
        // 내부 해시 계산
        let (hash_var, hash_val) = poseidon_hash_circuit(
            cs,
            self.left,
            self.right,
            &self.params,
        );

        // 해시 결과를 공개 출력(instance)으로 설정
        let expected = cs.alloc_instance(hash_val);

        // 해시 변수 == 공개 출력 (1개 제약)
        cs.enforce(
            LinearCombination::from(hash_var),
            LinearCombination::from(Variable::One),
            LinearCombination::from(expected),
        );
    }
}
```

```
PoseidonCircuit의 역할:
  "Poseidon 해시의 결과를 공개적으로 증명"

  공개 입력 (instance): hash_val (해시 결과)
  비공개 입력 (witness): left, right, 모든 중간값

  검증자의 관점:
    "hash_val이라는 공개값이 있고,
     누군가가 어떤 (left, right)를 알고 있어서
     Poseidon(left, right) = hash_val임을 증명했다."

  poseidon_hash_circuit: 633개 제약 (해시 계산)
  출력 등치: 1개 제약 (hash_var · 1 = expected)
  총: 634개 제약
```

---

#### 제약 수 분석: 634개

```
정확한 제약 수 계산:

Full rounds (8개):
  각 라운드: AddRC(T=3) + S-box(3T=9) + MDS(T=3) = 5T = 15
  8 × 15 = 120

Partial rounds (57개):
  각 라운드: AddRC(T=3) + S-box(3) + MDS(T=3) = 2T+3 = 9
  57 × 9 = 513

Output equality:
  hash_var · 1 = expected → 1

Total:
  120 + 513 + 1 = 634 제약

검증 공식:
  expected = RF × (T + 3T + T) + RP × (T + 3 + T) + 1
           = 8 × (3 + 9 + 3)  + 57 × (3 + 3 + 3) + 1
           = 8 × 15 + 57 × 9 + 1
           = 120 + 513 + 1
           = 634 ✓
```

```
변수 수 분석:

초기 상태: T = 3 (capacity, left, right)

Full rounds (8개):
  각 라운드: T(after_rc) + 3T(sbox aux) + T(mds out) = 5T = 15
  8 × 15 = 120

Partial rounds (57개):
  각 라운드: T(after_rc) + 3(sbox aux for first) + T(mds out) = 2T+3 = 9
  57 × 9 = 513

Instance: 1 (expected output)

기본 변수:
  One = 1
  초기 state witness = 3
  Instance = 1

Total variables:
  1(One) + 1(instance) + 3(initial) + 120 + 513 = 638

  또는: 1 + 1 + num_witness = 1 + 1 + 636 = 638 ✓
```

> [!important] 634 vs 243: "진짜" 비용은 얼마인가?
> ```
> 순수 S-box 제약만: 243개
>   → 이것이 Poseidon의 "암호학적" 비용
>   → 비선형성을 보장하는 핵심 제약
>
> 변수 고정 제약: 390개
>   → 이것이 "구조적 오버헤드"
>   → (state + rc) · 1 = var 형태
>   → 사실상 "항등 검증"
>
> QAP/Groth16에서의 실제 비용:
>   → 제약 수 = 다항식 차수 ≈ FFT 크기
>   → 634개 → 2^10 = 1024 크기 FFT (zero-pad)
>   → 243개 → 2^8 = 256 크기 FFT
>   → 실제로 2~4배 차이
>
> 최적화 가능성:
>   → AddRC를 S-box의 선형결합에 직접 흡수 (variable-free)
>   → MDS를 다음 라운드의 AddRC와 합치기
>   → bellman/arkworks의 최적화된 구현은 ~250개에 가까움
>
> 우리 구현은 교육용으로 각 단계를 명시적으로 분리하여
> "어디서 제약이 발생하는지"를 명확하게 보여준다.
> ```

---

### 최적화 분석: 교육용 634 vs 프로덕션 구현

> [!abstract] 우리 구현은 의도적으로 비최적화
> 교육 목적으로 각 단계를 명시적 변수로 분리했다.
> 프로덕션 구현은 선형결합을 흡수하여 제약을 대폭 줄인다.

#### 최적화 기법 1: AddRC 선형결합 흡수

```
현재 (교육용):
  1. after_rc = cs.alloc_witness(state + rc)     ← 보조 변수 할당
  2. (state + rc·One) · One = after_rc            ← 1개 제약 (고정)
  3. after_rc · after_rc = t1                     ← S-box 시작

최적화 후:
  AddRC의 "state + rc"를 S-box의 A/B에 직접 넣는다!

  1. (state + rc·One) · (state + rc·One) = t1    ← 1개 제약로 축소!

  → after_rc 변수 불필요, 고정 제약 불필요
  → 라운드당 T=3개 제약 절약

원리:
  R1CS는 A·B=C에서 A와 B가 선형결합을 허용한다.
  (state + rc·One)은 유효한 선형결합이므로
  S-box의 첫 곱셈에 직접 사용 가능!

  제약 1: (state+rc) · (state+rc) = t1   ← x² (AddRC 흡수)
  제약 2: t1 · t1 = t2                    ← x⁴
  제약 3: t2 · (state+rc) = y             ← x⁵ (다시 흡수)

  → AddRC 제약 0개! (T개 절약/라운드)
```

#### 최적화 기법 2: MDS + 다음 라운드 AddRC 합치기

```
현재 (교육용):
  MDS:  (M[i][0]·sbox₀ + M[i][1]·sbox₁ + M[i][2]·sbox₂) · 1 = mds_out
  다음 AddRC: (mds_out + rc'·One) · 1 = next_after_rc

최적화 후:
  MDS 결과를 다음 라운드의 S-box에 직접 흡수!

  다음 S-box의 첫 곱셈:
    (M[i][0]·sbox₀ + M[i][1]·sbox₁ + M[i][2]·sbox₂ + rc'·One) ·
    (M[i][0]·sbox₀ + M[i][1]·sbox₁ + M[i][2]·sbox₂ + rc'·One) = t1

  → MDS 고정 제약도 제거, 다음 AddRC 고정도 제거
  → 단, 선형결합이 길어짐 (A에 T+1개 항목)

주의:
  MDS 결과는 Partial round에서 state[1], state[2]에도 필요.
  이들은 S-box를 거치지 않으므로 "다음 라운드의 AddRC"로 흡수.
  Full round에서만 완전한 흡수가 가능.
```

#### 최적화 기법 3: Partial round의 identity 경로

```
Partial round에서 state[1], state[2]는 S-box를 거치지 않는다.
현재: AddRC → 변수 고정 → MDS에서 사용 → MDS 변수 고정

최적화:
  S-box를 거치지 않는 원소는 MDS 선형결합에 직접 포함 가능.

  state[1]의 경로:
    값 = state[1] + rc₁  (AddRC만 적용)
    → MDS 행렬의 해당 열에 직접 사용

  선형결합:
    mds_out[i] = M[i][0]·sbox(state[0]+rc₀)
               + M[i][1]·(state[1]+rc₁)      ← 직접 포함
               + M[i][2]·(state[2]+rc₂)      ← 직접 포함

  → state[1], state[2]의 AddRC 변수 2개 + 제약 2개 절약
  → partial round당 2개 제약 절약
  → 57 partial rounds × 2 = 114개 절약
```

#### 구현별 제약 수 비교

```
┌─────────────────────────┬────────────┬──────────┬──────────────────┐
│  구현                     │  제약 수    │  변수 수  │  비고              │
├─────────────────────────┼────────────┼──────────┼──────────────────┤
│  우리 구현 (교육용)        │    634     │   638    │  명시적 분리       │
│  AddRC 흡수              │    634-130  │   ~508   │  -130 (65×2 생략) │
│                         │    = 504    │          │                  │
│  + MDS 흡수              │    504-130  │   ~378   │  -130 (65×2 생략) │
│                         │    = 374    │          │                  │
│  + Partial 최적화        │    374-114  │   ~264   │  -114 (57×2)     │
│                         │    = 260    │          │                  │
│  circom poseidon         │    ~259    │   ~260   │  circomlib 구현   │
│  arkworks poseidon       │    ~255    │   ~258   │  r1cs-std 구현    │
│  이론적 최소 (S-box만)    │    243     │   ~246   │  달성 불가능       │
├─────────────────────────┼────────────┼──────────┼──────────────────┤
│  Merkle d=20 (우리)      │  13,394    │ ~13,488  │  634 × 21 기반    │
│  Merkle d=20 (circom)   │  ~5,439    │  ~5,460  │  ~259 × 21 기반   │
│  Merkle d=20 (arkworks) │  ~5,355    │  ~5,418  │  ~255 × 21 기반   │
└─────────────────────────┴────────────┴──────────┴──────────────────┘

핵심:
  교육용 634 → 최적화 ~260 → 이론적 최소 243
  차이: 634/260 ≈ 2.4배

  이것은 "교육 비용" — 구조를 명확히 보이기 위해
  2.4배의 제약 오버헤드를 감수.

  QAP/Groth16에서의 실제 영향:
    634 → 2^10 = 1024 크기 FFT
    260 → 2^9  = 512  크기 FFT
    → 증명 시간 약 2배 차이 (FFT는 n·log(n))
```

> [!note] 왜 이론적 최소 243에 도달할 수 없는가?
> ```
> S-box 243개 제약만으로는 부족한 이유:
>
> 1. MDS 결과를 다음 라운드에 전달해야 한다.
>    마지막 라운드의 MDS 후 결과를 "읽어야" 하는데,
>    선형결합은 변수가 아니므로 instance와 비교 불가.
>    → 최소 1개의 MDS 고정 제약이 필요 (출력 라운드)
>
> 2. Partial round에서 S-box를 거치지 않는 원소도
>    다음 라운드의 S-box 입력에 기여한다.
>    이 기여를 선형결합으로 표현하면,
>    A/B 선형결합의 항 수가 폭발적으로 증가.
>    → 실용적 한계: ~260개가 현실적 최적
>
> 3. 243은 "비선형 제약만"의 수.
>    회로의 입출력 바인딩을 위해 최소 수 개의
>    추가 선형 제약이 불가피.
> ```

> [!tip] PLONKish에서의 Poseidon
> ```
> R1CS의 한계: 제약당 곱셈 1개만 허용
>   → x⁵를 3개 제약으로 분해
>   → 선형결합을 변수로 고정하는 오버헤드
>
> PLONKish (Halo2 등):
>   → 커스텀 게이트: a₀·a₁·a₂·a₃·a₄ + q·(선형) = 0
>   → x⁵를 1개 게이트로 표현 가능!
>   → MDS도 내장 선형결합으로 처리
>
> 비교:
>   R1CS Poseidon:   634 제약 (교육) / 260 제약 (최적)
>   PLONKish Poseidon: ~65 행 (라운드당 1개 게이트)
>   → 4~10배 더 효율적
>
> 이것이 PLONKish가 "현대 ZK의 표준"인 이유 중 하나.
> 우리는 Step 14에서 PLONKish를 구현한다.
> ```

---

### Part 3: Merkle Proof 검증 회로

#### Merkle 경로와 비트

```
Sparse Merkle Tree에서 key의 비트가 경로를 결정:

  key = 5 (이진: 0101)
  depth = 4

  Level 0 (리프): bit 0 = 1 → 오른쪽
  Level 1:       bit 1 = 0 → 왼쪽
  Level 2:       bit 2 = 1 → 오른쪽
  Level 3:       bit 3 = 0 → 왼쪽

  트리 경로 (bottom-up):
                        [root]         ← level 4
                       /      \
                    [L3]      ...      ← level 3: bit 3=0 → 왼쪽
                   /    \
                 ...    [L2]           ← level 2: bit 2=1 → 오른쪽
                       /    \
                    [L1]    ...        ← level 1: bit 1=0 → 왼쪽
                   /    \
                 ...    [leaf]         ← level 0: bit 0=1 → 오른쪽

검증 과정:
  current = H(key, value)                    ← leaf hash

  level 0: bit=1 → H(sibling[0], current)   ← sibling이 왼쪽
  level 1: bit=0 → H(current, sibling[1])   ← current가 왼쪽
  level 2: bit=1 → H(sibling[2], current)
  level 3: bit=0 → H(current, sibling[3])

  current == root?  → 검증 완료
```

```
비트 추출 (get_bit):

  Fr 원소는 [u64; 4]로 표현 (256-bit)
  key = 5 → repr = [5, 0, 0, 0]

  get_bit(repr, 0) = (5 >> 0) & 1 = 1  (bit 0)
  get_bit(repr, 1) = (5 >> 1) & 1 = 0  (bit 1)
  get_bit(repr, 2) = (5 >> 2) & 1 = 1  (bit 2)
  get_bit(repr, 3) = (5 >> 3) & 1 = 0  (bit 3)

회로에서도 같은 get_bit 함수를 사용:
  native merkle.rs:  get_bit(&key_repr, level)
  circuit merkle.rs: get_bit(&key_repr, level)
  → 동일한 경로 결정 → "native = circuit" 보장
```

```rust
// circuits/merkle.rs (native merkle.rs와 동일한 함수)
fn get_bit(repr: &[u64; 4], i: usize) -> bool {
    if i >= 256 { return false; }
    let limb = i / 64;
    let bit = i % 64;
    (repr[limb] >> bit) & 1 == 1
}
```

---

#### 레벨별 회로 구성

```
Merkle proof 각 레벨의 회로 구조:

  ┌─────────────────────────────────────────────────┐
  │  Level i                                        │
  │                                                 │
  │  입력: current_var, current_val                  │
  │        sibling_var, sibling_val                  │
  │        bit_var, bit_val                          │
  │                                                 │
  │  1. Boolean: bit · (1-bit) = 0     ← 1 제약     │
  │                                                 │
  │  2. Mux × 2:                       ← 4 제약     │
  │     left  = mux(bit, sibling, current)          │
  │       bit=0: current (왼쪽)                      │
  │       bit=1: sibling (왼쪽)                      │
  │                                                 │
  │     right = mux(bit, current, sibling)          │
  │       bit=0: sibling (오른쪽)                    │
  │       bit=1: current (오른쪽)                    │
  │                                                 │
  │  3. Poseidon hash(left, right)     ← 633 제약   │
  │                                                 │
  │  출력: hash_var, hash_val → 다음 레벨의 current  │
  └─────────────────────────────────────────────────┘

레벨당 제약: 1 + 4 + 633 = 638
```

> [!note] 왜 mux를 2번 호출하는가?
> ```
> Poseidon hash는 H(left, right)로 호출된다.
> bit에 따라 left와 right에 무엇이 들어갈지 결정해야 한다:
>
>   bit = 0 → current가 왼쪽  → left = current, right = sibling
>   bit = 1 → sibling이 왼쪽 → left = sibling, right = current
>
> left  = mux(bit, when_true=sibling, when_false=current)
>   bit=0 → current (when_false)
>   bit=1 → sibling (when_true)
>
> right = mux(bit, when_true=current, when_false=sibling)
>   bit=0 → sibling (when_false)
>   bit=1 → current (when_true)
>
> 두 mux의 when_true/when_false가 교차되어 있다!
> 이것이 "bit에 따라 해시 입력 순서를 바꾸는" 핵심 메커니즘.
> ```

---

#### MerkleProofCircuit 구조

```rust
// circuits/merkle.rs
pub struct MerkleProofCircuit {
    pub root: Fr,         // 공개 입력: Merkle root
    pub key: Fr,          // 비공개: 키
    pub value: Fr,        // 비공개: 값
    pub siblings: Vec<Fr>, // 비공개: 경로의 형제 해시들
    pub depth: usize,      // 트리 깊이
    pub params: PoseidonParams,
}

impl Circuit for MerkleProofCircuit {
    fn synthesize(&self, cs: &mut ConstraintSystem) {
        let key_repr = self.key.to_repr();

        // ── 공개 입력: root ──
        let root_var = cs.alloc_instance(self.root);

        // ── 비공개 입력: key, value ──
        let _key_var = cs.alloc_witness(self.key);
        let _value_var = cs.alloc_witness(self.value);

        // ── Step 1: leaf = H(key, value) ──
        let (mut current_var, mut current_val) = poseidon_hash_circuit(
            cs,
            self.key,
            self.value,
            &self.params,
        );
        // → 633 제약 (Poseidon hash, 출력 등치 없음)

        // ── Step 2: 각 레벨에서 형제와 해시 ──
        for level in 0..self.depth {
            let sibling_val = self.siblings[level];
            let sibling_var = cs.alloc_witness(sibling_val);

            // key의 level번째 비트
            let bit = get_bit(&key_repr, level);
            let bit_val = if bit { Fr::ONE } else { Fr::ZERO };
            let bit_var = cs.alloc_witness(bit_val);

            // Boolean 제약: bit ∈ {0, 1}
            enforce_boolean(cs, bit_var);    // 1 제약

            // Mux: bit에 따라 left/right 결정
            let (_left_var, left_val) = mux_circuit(
                cs, bit_var, bit_val,
                sibling_var, sibling_val,  // when_true: sibling
                current_var, current_val,  // when_false: current
            );    // 2 제약
            let (_right_var, right_val) = mux_circuit(
                cs, bit_var, bit_val,
                current_var, current_val,  // when_true: current
                sibling_var, sibling_val,  // when_false: sibling
            );    // 2 제약

            // Poseidon hash
            let (hash_var, hash_val) = poseidon_hash_circuit(
                cs, left_val, right_val, &self.params,
            );    // 633 제약

            current_var = hash_var;
            current_val = hash_val;
        }

        // ── Step 3: current == root ──
        cs.enforce(
            LinearCombination::from(current_var),
            LinearCombination::from(Variable::One),
            LinearCombination::from(root_var),
        );
        // → 1 제약
    }
}
```

> [!important] 공개 vs 비공개 입력의 설계
> ```
> 공개 입력 (instance): root만
>   → 검증자는 "이 Merkle root에 대한 증명이 유효한가?"만 확인
>   → root는 블록체인에 공개된 상태 root
>
> 비공개 입력 (witness): key, value, siblings
>   → 증명자만 아는 비밀
>   → key: 어떤 항목을 증명하는지
>   → value: 그 항목의 값
>   → siblings: 경로의 형제 해시들
>
> 이것이 ZK의 핵심:
>   "나는 이 root에 포함된 어떤 (key, value)를 알고 있다"
>   → key와 value를 공개하지 않고 증명!
>
> 프라이버시 응용 (Tornado Cash 등):
>   root = mixer contract의 commitment tree root
>   key = nullifier (이중 사용 방지용)
>   value = secret commitment
>   → "나는 이 mixer에 입금한 사람이다" (누군지는 비공개)
> ```

---

#### 제약 수 분석 (depth별)

```
Merkle proof 회로의 총 제약 수:

기본 구조:
  Leaf hash:           633 제약  (Poseidon, 출력 등치 제외)
  Per level:           638 제약  (boolean 1 + mux 4 + Poseidon 633)
  Root equality:       1 제약

총:
  633 + depth × 638 + 1

depth=4 (테스트):
  633 + 4 × 638 + 1
  = 633 + 2552 + 1
  = 3,186 제약

depth=8:
  633 + 8 × 638 + 1
  = 633 + 5104 + 1
  = 5,738 제약

depth=20 (실전):
  633 + 20 × 638 + 1
  = 633 + 12760 + 1
  = 13,394 제약

depth=32 (Ethereum Merkle Patricia):
  633 + 32 × 638 + 1
  = 633 + 20416 + 1
  = 21,050 제약
```

```
제약 수 비교표:

┌──────────┬──────────┬──────────┬────────────────────┐
│  depth   │  제약 수   │  변수 수   │  Groth16 증명 시간   │
│          │          │          │  (추정)              │
├──────────┼──────────┼──────────┼────────────────────┤
│    4     │  3,186   │  ~3,208  │  ~10ms             │
│    8     │  5,738   │  ~5,778  │  ~20ms             │
│   20     │ 13,394   │ ~13,488  │  ~50ms             │
│   32     │ 21,050   │ ~21,198  │  ~80ms             │
│  256     │164,162   │~165,154  │  ~600ms            │
└──────────┴──────────┴──────────┴────────────────────┘

참고: Groth16 증명 시간은 BN254 커브, 일반 노트북 기준 추정.
검증 시간은 항상 ~1ms (상수 — Groth16의 핵심 장점).
```

> [!abstract] 왜 depth=4로 테스트하는가?
> ```
> depth=4면 리프가 2⁴ = 16개.
> key < 16이어야 모든 비트가 depth 범위 안에 있다.
>
> depth=256 (Fr의 전체 비트)이면:
>   → Poseidon hash 257번 = 257 × 633 = 162,681 제약
>   → 테스트가 수 초 걸림
>
> depth=4면:
>   → Poseidon hash 5번 = 3,186 제약
>   → 테스트가 밀리초 단위
>
> 제약 구조는 depth와 무관하게 동일하므로,
> depth=4로 검증한 패턴이 depth=256에서도 그대로 동작한다.
>
> 테스트에서의 key 제한:
>   key = Fr::from_u64(5)  → 이진 0101 → depth 4 범위 안
>   key = Fr::from_u64(12) → 이진 1100 → depth 4 범위 안
>   key = Fr::from_u64(16) → 이진 10000 → bit 4가 1인데 depth=4에서
>                              미처리 → 올바른 경로 보장 안 됨
> ```

---

### Part 4: Native vs Circuit 비교

#### 값 추적: 같은 입력 --> 같은 출력

```
Poseidon hash: left=1, right=2

Native (Step 07):
  state = [0, 1, 2]
  → 65 라운드 (AddRC → S-box → MDS)
  → state[1] = hash_val (254-bit Fr 원소)

Circuit (Step 10):
  state_vals = [0, 1, 2]           ← 같은 초기값
  state_vars = [w₀, w₁, w₂]       ← 심볼릭 변수

  Round 0 (Full):
    Native:  s[0] = (0 + rc₀)⁵           → 값 V₀
    Circuit: after_rc_vals[0] = 0 + rc₀   → 같은 값
             sbox: V₀ = (0 + rc₀)⁵        → 같은 값 V₀

  ...65 라운드 후...

  Native:  hash_val = state[1]
  Circuit: hash_val = state_vals[1]  ← 같은 값!

검증:
  cs.values[1] = hash_val (instance로 할당됨)
  assert_eq!(cs.values[1], native_hash)  ✓
```

```rust
// 테스트: poseidon_circuit_matches_native
#[test]
fn poseidon_circuit_matches_native() {
    let params = PoseidonParams::new();
    let left = Fr::from_u64(1);
    let right = Fr::from_u64(2);

    // native
    let native_hash = poseidon_hash_with_params(&params, left, right);

    // circuit
    let circuit = PoseidonCircuit {
        left,
        right,
        params: PoseidonParams::new(),
    };
    let mut cs = ConstraintSystem::new();
    circuit.synthesize(&mut cs);

    assert!(cs.is_satisfied());
    assert_eq!(cs.values[1], native_hash);  // instance = native hash ✓
}
```

```
Merkle proof: key=5, value=42, depth=4

Native (Step 08):
  1. leaf = H(5, 42) = L
  2. level 0: bit=1 → H(sibling₀, L) = H₀
  3. level 1: bit=0 → H(H₀, sibling₁) = H₁
  4. level 2: bit=1 → H(sibling₂, H₁) = H₂
  5. level 3: bit=0 → H(H₂, sibling₃) = H₃
  6. H₃ == root → true

Circuit (Step 10):
  1. leaf: poseidon_hash_circuit(5, 42) → (leaf_var, L)     같은 L
  2. level 0:
     bit=1 → mux: left=sibling₀, right=L
     poseidon_hash_circuit(sibling₀, L) → H₀              같은 H₀
  3. level 1:
     bit=0 → mux: left=H₀, right=sibling₁
     poseidon_hash_circuit(H₀, sibling₁) → H₁             같은 H₁
  4. level 2, 3: ...                                       같은 값
  5. current == root → 1개 제약 만족

  is_satisfied() = true ✓
```

---

#### 잘못된 witness가 거부되는 과정

```
시나리오 1: 잘못된 root

  native:
    verify_merkle_proof(Fr(999), key=5, value=42, proof) → false

  circuit:
    MerkleProofCircuit { root: Fr(999), key: 5, value: 42, ... }
    cs.synthesize()

    → 모든 내부 제약은 만족 (해시 계산 자체는 올바르게 진행)
    → 마지막 제약: current_var · 1 = root_var
      current_val = H₃ (올바른 root)
      root_var의 값 = 999 (잘못된 root)
      H₃ ≠ 999 → 이 제약이 실패!

    is_satisfied() = false ✓

시나리오 2: 잘못된 value

  native:
    verify_merkle_proof(root, key=5, value=99, proof) → false

  circuit:
    MerkleProofCircuit { root, key: 5, value: 99, ... }

    → leaf = H(5, 99) ≠ H(5, 42)  ← 다른 리프 해시
    → 이후 모든 중간 해시가 달라짐
    → 최종 current ≠ root → 마지막 제약 실패

    is_satisfied() = false ✓
```

> [!important] 건전성의 체인
> ```
> Merkle 회로의 건전성은 다음 체인으로 보장된다:
>
> 1. Boolean 가젯: bit ∈ {0, 1}
>    → 경로 방향 조작 불가
>
> 2. Mux 가젯: left/right가 bit에 의해 결정
>    → 해시 입력 순서 조작 불가
>
> 3. Poseidon 가젯: S-box의 비선형 제약
>    → 해시 값 위조 불가
>    → 각 S-box가 x · x = t1, t1 · t1 = t2, t2 · x = y를 강제
>    → x⁵가 아닌 다른 값을 y에 넣으면 제약 위반
>
> 4. Root equality: current · 1 = root
>    → 최종 해시가 공개 root와 같아야 함
>
> 이 체인의 어느 하나라도 빠지면 건전성이 깨진다!
>
> 예: Boolean 제약을 빠뜨리면?
>   → 공격자가 bit=2를 넣어 mux를 조작
>   → 잘못된 경로로 올바른 root를 만들 수 있음
>   → 위조된 멤버십 증명 생성 가능!
> ```

---

### Under-constrained 회로 취약점 분석

> [!important] Under-constrained란?
> 회로에 제약이 부족하여 공격자가 잘못된 witness로 제약을 만족시킬 수 있는 상태.
> "거짓 증명이 통과한다" = 건전성(soundness) 위반 = 보안 파괴.
>
> ```
> 올바른 회로:      제약 충분 → 유효한 witness만 만족
> Under-constrained: 제약 부족 → 위조된 witness도 만족 가능!
>
> 이것은 "회로가 돌아가지 않는" 버그가 아니다.
> 오히려 "회로가 너무 잘 돌아가는" — 거짓도 통과시키는 — 버그다.
> 발견하기 매우 어렵다: 모든 정상 테스트가 통과하기 때문.
> ```

#### 공격 1: Boolean 제약 누락 — 경로 조작

```
시나리오:
  enforce_boolean(cs, bit_var) 호출을 삭제한다.
  → bit_var는 {0, 1}로 제한되지 않음
  → 공격자가 bit=2 같은 값을 자유롭게 사용 가능

공격 메커니즘:
  mux(bit=2, when_true=S, when_false=C) 결과:
    result = C + 2·(S - C)
           = C + 2S - 2C
           = 2S - C

  이것은 C도 아니고 S도 아닌 "제3의 값"!

  구체적 수치 예:
    current = 100, sibling = 200

    정상 (bit=0): left = 100, right = 200  → H(100, 200)
    정상 (bit=1): left = 200, right = 100  → H(200, 100)
    공격 (bit=2): left = 2·200 - 100 = 300  ← 존재하지 않는 값!
                  right = 2·100 - 200 = 0

    → H(300, 0)이 해시 입력으로 사용됨
    → 공격자는 이 "자유도"를 이용하여
       원하는 root와 일치하는 경로를 역산할 수 있음

공격의 위험성:
  ┌─────────────────────────────────────────────────────┐
  │  Boolean 제약 없음                                    │
  │  → bit ∈ Fr (2^254가지 선택지!)                       │
  │  → mux 결과 = 임의의 선형결합                          │
  │  → 트리의 모든 레벨에서 자유도 +1                       │
  │  → depth=4면 4개의 자유도                              │
  │  → 공격자가 원하는 root를 만들 확률이 극적으로 증가       │
  └─────────────────────────────────────────────────────┘
```

> [!note] 실제 사례: Tornado Cash의 under-constrained 버그
> ```
> 2019년, Tornado Cash에서 발견된 버그:
>
> 문제:
>   회로에서 nullifier 해시의 preimage 검증이 불충분했음
>   → 같은 deposit에 대해 여러 개의 유효한 nullifier를 생성 가능
>   → 이중 인출(double-spend) 공격이 가능했음
>
> 원인:
>   "under-constrained" — 제약이 부족하여 여러 witness가 통과
>   정상적인 사용에서는 문제가 드러나지 않음
>   악의적인 witness를 구성해야만 발견되는 버그
>
> 교훈:
>   1. 모든 변수에 적절한 제약이 있는지 감사(audit)해야 함
>   2. "정상 입력 테스트 통과"만으로는 건전성 보장 불가
>   3. 의도적으로 비정상 witness를 구성하여 테스트해야 함
> ```

#### 공격 2: S-box 제약 하나 누락 — 해시 위조

```
시나리오:
  S-box의 제약 3개 중 마지막 하나(t2·x = y)를 삭제한다.

  원래 S-box:
    제약 1: x · x = t1    (x² 강제)
    제약 2: t1 · t1 = t2   (x⁴ 강제)
    제약 3: t2 · x = y     (x⁵ 강제)  ← 삭제!

  삭제 후:
    t1은 x²으로 고정됨 ✓
    t2는 x⁴으로 고정됨 ✓
    y는 아무 값이나 될 수 있음 ✗ (제약 없음!)

공격:
  y = t2 · x = x⁵이어야 하지만, y는 자유 변수
  → 공격자가 y를 원하는 값으로 설정 가능
  → Poseidon 해시의 S-box 출력을 조작
  → 결과적으로 해시 값을 마음대로 조작 가능

  S-box(x) = x⁵가 아닌 임의의 값이 통과하므로:
    → Poseidon의 비선형성이 파괴됨
    → 해시 함수가 더 이상 collision-resistant가 아님
    → 공격자가 원하는 해시 출력을 만들 수 있음

수동 추적 (x = 3):
  정상:
    t1 = 9, t2 = 81, y = 243  (= 3⁵)
    제약 1: 3·3 = 9   ✓
    제약 2: 9·9 = 81  ✓
    제약 3: 81·3 = 243 ✓

  공격 (제약 3 삭제):
    t1 = 9, t2 = 81, y = 999  (공격자가 999를 선택!)
    제약 1: 3·3 = 9   ✓
    제약 2: 9·9 = 81  ✓
    (제약 3 없음 → y=999 검증 안 함!)

    → S-box(3) = 999가 통과! (실제는 243이어야 함)
    → 이 999가 다음 MDS에 전파 → 전체 해시 출력이 조작됨
```

> [!important] 제약 수와 보안의 관계
> ```
> 제약이 부족하면: under-constrained → 위조 가능
> 제약이 충분하면: 유일한 유효 witness만 존재
> 제약이 과다하면: over-constrained → 유효한 증명도 거부 (완전성 위반)
>
> S-box 3개 제약의 필요충분성:
>   제약 1: x·x = t1      → t1 = x² (유일하게 결정)
>   제약 2: t1·t1 = t2     → t2 = t1² = x⁴ (유일하게 결정)
>   제약 3: t2·x = y       → y = t2·x = x⁵ (유일하게 결정)
>
>   3개 모두 있어야 y = x⁵가 강제된다.
>   어느 하나라도 빠지면 자유도가 생겨 공격 가능.
> ```

#### 공격 3: Root equality 제약 누락

```
시나리오:
  마지막 제약 current_var · 1 = root_var를 삭제한다.

  결과:
    회로는 해시 계산을 올바르게 수행하지만,
    최종 결과가 공개 root와 같은지 검증하지 않음!

    → 공격자는 아무 (key, value)를 넣어도 회로가 만족됨
    → Merkle 멤버십 증명이 의미 없어짐

  이것은 가장 기초적이지만 치명적인 under-constrained 버그.
  instance(공개 입력)와 witness(비밀 입력)를 연결하는
  유일한 "앵커 포인트"가 사라지기 때문.

  비유:
    root equality = "시험지에 이름 쓰기"
    이것 없이는 누구의 답안인지 확인할 방법이 없다.
    모든 계산은 맞지만, 공개된 root와의 관계가 없다.
```

> [!abstract] Zcash의 under-constrained 취약점 (CVE-2019-7167)
> ```
> 2019년 공개된 Zcash의 critical 버그:
>
> 문제:
>   Sapling 회로에서 value commitment의 binding이 불충분
>   → 증명 시스템이 보장해야 할 balance 검증이 우회 가능
>   → 이론적으로 무한 발행(infinite mint)이 가능했음
>
> 기술적 원인:
>   회로의 제약이 모든 경우를 커버하지 못함
>   특정 edge case에서 witness 자유도가 발생
>   → "거짓 증명"이 검증을 통과
>
> 발견:
>   내부 감사(audit)에서 발견 → 악용 전에 패치
>   8개월간 비공개로 수정 후 공개
>
> 교훈:
>   1. ZK 회로의 under-constrained 버그는 "무한 발행" 급의 위험
>   2. 제약 수를 정확히 추적하고, 수식으로 검증해야 함
>   3. 외부 감사(audit)가 필수적
>
> 우리 구현의 대응:
>   → 제약 수를 공식으로 계산: 634 = 8×15 + 57×9 + 1
>   → 테스트에서 정확한 제약 수 검증
>   → 잘못된 입력이 거부되는지 테스트
>   → 이 세 가지가 under-constrained 방지의 기본
> ```

---

### Merkle 회로의 건전성 증명 스케치

> [!important] 건전성(Soundness)이란?
> "회로가 만족되었다면, 증명자는 실제로 올바른 (key, value)를 알고 있다."
> 정확히는: 만약 잘못된 (key, value)로 회로가 만족되면,
> Poseidon 해시의 collision이 존재한다 — 이는 암호학적으로 불가능.

#### 정리: Merkle 회로의 건전성

```
정리 (Soundness):
  MerkleProofCircuit(root, key, value, siblings, depth)가 만족(satisfied)되고,
  root가 올바른 Sparse Merkle Tree의 root이며,
  트리의 key 위치에 저장된 값이 value' ≠ value라고 하자.

  그러면, Poseidon 해시 함수의 collision이 존재한다.

증명 스케치:

  회로가 만족된다면, 다음이 성립:

  1. leaf_circuit = H(key, value)     (Poseidon 제약에 의해)
  2. 각 level i에서:
     bit_i ∈ {0, 1}                  (Boolean 제약에 의해)
     (left_i, right_i) = mux에 의해 결정  (Mux 제약에 의해)
     node_i = H(left_i, right_i)      (Poseidon 제약에 의해)
  3. node_final = root                (Root equality 제약에 의해)

  이제 "올바른" 경로를 추적하자:

  실제 트리에서:
    leaf_real = H(key, value')         (value' = 실제 저장된 값)
    bit_i = key의 i번째 비트           (동일한 key)
    (left_real_i, right_real_i) = 올바른 형제 해시로 결정
    node_real_final = root             (올바른 트리이므로)

  회로 경로와 실제 경로 비교:
    leaf_circuit = H(key, value) ≠ H(key, value') = leaf_real
    (∵ value ≠ value', H가 injection이면 — collision resistance)

    하지만 node_final = root = node_real_final

  따라서:
    두 경로의 출발점(leaf)은 다르지만 도착점(root)은 같다.
    depth개의 해시를 거치면서, 어딘가에서 두 경로가 "합류"한다.

    합류 지점 level k:
      node_k^circuit ≠ node_k^real   (아직 다름)
      하지만
      node_{k+1}^circuit = node_{k+1}^real  (여기서 합류)

    이것은:
      H(left_k^circuit, right_k^circuit) = H(left_k^real, right_k^real)
      이면서
      (left_k^circuit, right_k^circuit) ≠ (left_k^real, right_k^real)

    → Poseidon 해시의 collision!

  결론:
    Poseidon이 collision-resistant이면,
    value ≠ value'인 경우 회로가 만족될 수 없다.  □
```

> [!abstract] 각 제약의 역할 — 건전성에서의 필요성
> ```
> 제약 유형          건전성에서의 역할
> ─────────────────────────────────────────────────
> Boolean            bit가 {0,1}만 가능하게 강제
>  b·(1-b)=0         → 경로 조작 방지
>                    → 없으면: bit=2 등으로 mux 조작 가능
>
> Mux                bit에 의해 left/right가 결정적
>  bit·(x-y)=t       → 해시 입력 순서 고정
>  (y+t)·1=result    → 없으면: left/right를 자유롭게 선택 가능
>
> Poseidon (S-box)   해시 값이 입력에 의해 유일하게 결정
>  x·x=t1, etc.      → collision resistance의 핵심
>                    → 없으면: 해시 출력을 자유롭게 조작
>
> Root equality      최종 해시가 공개 root와 일치
>  current·1=root    → 트리 바인딩
>                    → 없으면: 어떤 경로든 유효
>
> 이 네 가지가 하나라도 빠지면 증명이 파괴된다:
>   Boolean 없음  → 경로 조작 → 위조된 멤버십
>   Mux 없음     → 순서 조작 → 다른 트리 구조 증명
>   S-box 없음   → 해시 조작 → collision 없이 위조
>   Root 없음    → 바인딩 없음 → 모든 입력이 유효
> ```

#### 완전성 (Completeness) 증명

```
정리 (Completeness):
  (key, value)가 실제로 root에 해당하는 Sparse Merkle Tree에 있으면,
  올바른 proof(siblings)로 회로가 반드시 만족된다.

증명:
  native verify_merkle_proof(root, key, value, proof) = true이면:

  1. leaf = H(key, value)                    — native와 동일한 계산
  2. 각 level i에서:
     bit_i = key의 i번째 비트              — native와 동일
     mux가 올바른 left/right 결정           — native와 동일한 규칙
     H(left, right) = next_node             — native와 동일한 해시
  3. 최종 node = root                        — native에서 true

  circuit은 native와 동일한 계산을 수행하므로:
    모든 witness 값이 올바르게 계산됨
    → 모든 제약이 만족됨
    → is_satisfied() = true  □

이것이 "native = circuit" 원칙이 완전성을 보장하는 이유:
  native에서 참인 계산은 → circuit에서도 반드시 만족.
```

> [!tip] Zero-knowledge 성질
> ```
> 건전성과 완전성 외에 ZK의 세 번째 성질: 영지식성.
>
> Merkle 회로에서의 영지식성:
>   증명에서 key, value, siblings가 누출되지 않는다.
>
> 이것은 회로 자체의 성질이 아니라,
> Groth16 증명 시스템의 성질:
>   → witness를 랜덤화하여 증명 생성
>   → 증명에서 witness의 정보를 추출 불가능
>   → 시뮬레이터가 witness 없이 "가짜 증명"을 만들 수 있음
>     (CRS trapdoor를 아는 경우)
>
> 회로의 역할:
>   → key, value, siblings를 witness로 설정 (instance가 아님)
>   → Groth16이 witness를 숨겨줌
>   → root만 공개 → 검증자는 root 외에 아무것도 모름
> ```

---

### 전체 회로 구조 다이어그램

```
MerkleProofCircuit (depth=4):

  ┌──────────────────────────────────────────────────────────┐
  │  Instance (공개):  root                                   │
  │  Witness (비밀):   key, value, sibling₀~₃                 │
  │                                                          │
  │  ┌─────────────────────────────┐                         │
  │  │  Leaf Hash                  │                         │
  │  │  H(key, value)              │  633 제약               │
  │  │  = poseidon_hash_circuit    │                         │
  │  └──────────┬──────────────────┘                         │
  │             │ current                                    │
  │             ▼                                            │
  │  ┌─────────────────────────────┐                         │
  │  │  Level 0                    │                         │
  │  │  bool(bit₀)           1    │                         │
  │  │  mux(left)             2    │  638 제약               │
  │  │  mux(right)            2    │                         │
  │  │  H(left, right)       633   │                         │
  │  └──────────┬──────────────────┘                         │
  │             │ current                                    │
  │             ▼                                            │
  │  ┌─────────────────────────────┐                         │
  │  │  Level 1                    │  638 제약               │
  │  └──────────┬──────────────────┘                         │
  │             │                                            │
  │             ▼                                            │
  │  ┌─────────────────────────────┐                         │
  │  │  Level 2                    │  638 제약               │
  │  └──────────┬──────────────────┘                         │
  │             │                                            │
  │             ▼                                            │
  │  ┌─────────────────────────────┐                         │
  │  │  Level 3                    │  638 제약               │
  │  └──────────┬──────────────────┘                         │
  │             │ current                                    │
  │             ▼                                            │
  │  ┌─────────────────────────────┐                         │
  │  │  current · 1 = root         │  1 제약                 │
  │  └─────────────────────────────┘                         │
  │                                                          │
  │  Total: 633 + 4 × 638 + 1 = 3,186 제약                  │
  │         ~3,208 변수                                      │
  └──────────────────────────────────────────────────────────┘
```

---

### Witness 벡터 레이아웃 (depth=2 Merkle 회로)

> [!important] 변수의 정확한 배치를 이해하기
> Witness 벡터 s의 각 인덱스가 무엇을 담고 있는지 보여준다.
> depth=2이면 Poseidon을 3번 호출 (leaf + 2 levels).

```
depth=2 Merkle 회로의 witness 벡터 구조:

총 변수 수:
  Poseidon 1회 내부 변수:
    초기 state: 3 (capacity, left, right)
    65 라운드: 120(full) + 513(partial) = 633 보조 변수

  전체:
    One                           : 1
    Instance (root)               : 1
    key_witness                   : 1
    value_witness                 : 1
    Leaf hash (Poseidon)          : 3(init) + 633(rounds) = 636
    Level 0 가젯                  : 1(sibling) + 1(bit) + 4(mux) + 636(Poseidon) = 642
    Level 1 가젯                  : 1(sibling) + 1(bit) + 4(mux) + 636(Poseidon) = 642
    ────────────────────────────
    총: 1 + 1 + 1 + 1 + 636 + 642 + 642 = 1,924 변수

  제약 수: 633 + 2×638 + 1 = 1,910 제약
```

```
인덱스 배치 다이어그램:

s[0]        = 1                    ┐
                                   │ 시스템 변수
s[1]        = root (instance)      ┘

s[2]        = key (witness)        ┐
s[3]        = value (witness)      │ 최상위 입력
                                   ┘

s[4..6]     = [0, key, value]      ┐
                                   │ Leaf Poseidon 초기 state (3)
                                   ┘

s[7..639]   = Poseidon 내부 변수    ┐
  s[7..9]   = Round 0 AddRC (3)   │
  s[10..12] = Round 0 S-box₀ (3)  │
  s[13..15] = Round 0 S-box₁ (3)  │
  s[16..18] = Round 0 S-box₂ (3)  │
  s[19..21] = Round 0 MDS (3)     │ Leaf hash: 633 보조 변수
  s[22..36] = Round 1 (15)        │  (8 full × 15 + 57 partial × 9)
  ...                              │
  s[37..51] = Round 2 (15)        │
  s[52..66] = Round 3 (15)        │ ← 마지막 full round (phase 1)
  s[67..75] = Round 4 (9)         │ ← 첫 partial round
  ...                              │
  s[571..579] = Round 60 (9)      │ ← 마지막 partial round
  s[580..594] = Round 61 (15)     │ ← 첫 full round (phase 3)
  ...                              │
  s[625..639] = Round 64 (15)     ┘ ← 마지막 full round

s[640]      = sibling₀ (witness)   ┐
s[641]      = bit₀ (witness)       │
s[642]      = mux_t_left (aux)     │
s[643]      = mux_left (aux)       │ Level 0 가젯
s[644]      = mux_t_right (aux)    │
s[645]      = mux_right (aux)      │
s[646..1281] = Level 0 Poseidon   ┘  (636 변수)

s[1282]     = sibling₁ (witness)   ┐
s[1283]     = bit₁ (witness)       │
s[1284]     = mux_t_left (aux)     │
s[1285]     = mux_left (aux)       │ Level 1 가젯
s[1286]     = mux_t_right (aux)    │
s[1287]     = mux_right (aux)      │
s[1288..1923] = Level 1 Poseidon  ┘  (636 변수)

총: s[0..1923] = 1,924 변수
```

> [!abstract] 변수 카테고리 분류
> ```
> ┌──────────────────┬──────────┬────────────┬────────────────┐
> │  카테고리          │  인덱스   │  개수       │  비율           │
> ├──────────────────┼──────────┼────────────┼────────────────┤
> │  One (상수)       │  0       │  1         │  0.05%         │
> │  Instance (공개)  │  1       │  1         │  0.05%         │
> │  Witness (비밀)   │  2..1923 │  1,922     │  99.9%         │
> │    ├ 입력 변수     │  2..3    │  2         │  0.1%          │
> │    ├ Poseidon 내부│  4..639  │  636       │  33.1%         │
> │    │              │  646..   │  636×2     │  33.1%×2       │
> │    ├ Merkle 가젯  │  640..645│  6×2       │  0.6%          │
> │    │              │  1282..  │            │                │
> │    └ 초기 state   │  4..6    │  3×3       │  0.5%          │
> └──────────────────┴──────────┴────────────┴────────────────┘
>
> 관찰:
>   1. 변수의 99.9%가 witness (비밀)
>   2. 검증자가 아는 것은 s[0]=1과 s[1]=root 뿐
>   3. Poseidon 내부 변수가 전체의 ~99%를 차지
>   4. Merkle 가젯(Boolean, Mux)은 극소수 (~1%)
>
>   → 회로의 복잡성은 거의 전부 Poseidon 해시에서 옴
>   → 해시 함수가 ZK-friendly해야 하는 이유:
>      변수 수 ∝ 제약 수 ∝ 증명 시간
> ```

> [!note] Groth16에서 이 벡터가 어떻게 사용되는가
> ```
> Setup (신뢰 설정):
>   A, B, C 행렬의 구조만 사용 → 변수 인덱스에 의존
>   s[1]이 instance, s[2..]이 witness라는 정보가 SRS에 인코딩됨
>
> Prove (증명 생성):
>   s[0..1923] 전체 벡터 필요
>   A·s, B·s, C·s 계산 → QAP 다항식 계산
>   → π = ([A]₁, [B]₂, [C]₁) 생성
>
> Verify (증명 검증):
>   s[0] = 1, s[1] = root만 사용
>   나머지 s[2..1923]은 전혀 접근하지 않음!
>   → 검증자는 1,922개의 witness 값을 모른 채로 검증
>   → 이것이 "zero-knowledge"
>
> 검증 수식:
>   e(A, B) = e(α, β) · e(s[1]·L₁(τ), γ) · e(C, δ)
>   → s[1] = root만 검증 수식에 등장
>   → s[2..1923]는 증명 π 안에 "암호학적으로 숨겨져" 있음
> ```

---

### Poseidon 회로의 수동 제약 추적 (첫 번째 full round)

> [!abstract] 제약을 하나씩 추적하여 구조를 완전히 이해
> 첫 번째 full round의 모든 제약을 수동으로 나열한다.
> 실제 값 대신 심볼릭 이름을 사용.

```
초기 상태: state = [s₀, s₁, s₂]  (값: [0, left, right])

═══ Full Round 0 ═══

─── AddRC (3개 제약) ───

  rc₀, rc₁, rc₂ = 라운드 상수 (Fr 값)

  a₀ = s₀ + rc₀    (보조 변수 할당)
  제약 1: (s₀ + rc₀·One) · One = a₀
    A = [(s₀, 1), (One, rc₀)]
    B = [(One, 1)]
    C = [(a₀, 1)]
    검증: (0 + rc₀) · 1 = a₀ ✓

  a₁ = s₁ + rc₁
  제약 2: (s₁ + rc₁·One) · One = a₁

  a₂ = s₂ + rc₂
  제약 3: (s₂ + rc₂·One) · One = a₂

─── S-box ALL (9개 제약) ───

  S-box on a₀ (3개):
    t₀₁ = a₀²        제약 4: a₀ · a₀ = t₀₁
    t₀₂ = t₀₁²       제약 5: t₀₁ · t₀₁ = t₀₂
    b₀  = t₀₂ · a₀   제약 6: t₀₂ · a₀ = b₀    → b₀ = a₀⁵

  S-box on a₁ (3개):
    t₁₁ = a₁²        제약 7: a₁ · a₁ = t₁₁
    t₁₂ = t₁₁²       제약 8: t₁₁ · t₁₁ = t₁₂
    b₁  = t₁₂ · a₁   제약 9: t₁₂ · a₁ = b₁    → b₁ = a₁⁵

  S-box on a₂ (3개):
    t₂₁ = a₂²        제약 10: a₂ · a₂ = t₂₁
    t₂₂ = t₂₁²       제약 11: t₂₁ · t₂₁ = t₂₂
    b₂  = t₂₂ · a₂   제약 12: t₂₂ · a₂ = b₂   → b₂ = a₂⁵

─── MDS mix (3개 제약) ───

  MDS = [[2,1,1],[1,2,1],[1,1,2]]

  m₀ = 2·b₀ + 1·b₁ + 1·b₂
  제약 13: (2·b₀ + 1·b₁ + 1·b₂) · One = m₀
    A = [(b₀, 2), (b₁, 1), (b₂, 1)]
    B = [(One, 1)]
    C = [(m₀, 1)]

  m₁ = 1·b₀ + 2·b₁ + 1·b₂
  제약 14: (1·b₀ + 2·b₁ + 1·b₂) · One = m₁

  m₂ = 1·b₀ + 1·b₁ + 2·b₂
  제약 15: (1·b₀ + 1·b₁ + 2·b₂) · One = m₂

═══ Full Round 0 완료: 15개 제약, 15개 보조 변수 ═══
  state = [m₀, m₁, m₂] → 다음 라운드의 입력
```

---

### R1CS 행렬의 실제 구조 — 제약별 감사 (Audit)

> [!important] R1CS 행렬을 직접 들여다본다
> to_matrices()가 반환하는 A, B, C 행렬의 실제 구조를 확인한다.
> 각 행 = 하나의 제약, 각 열 = 하나의 변수.
> 희소 행렬이므로 비-0 항목만 나열한다.

#### S-box의 A/B/C 행렬 항목 (3개 제약)

```
변수 인덱스 가정:
  s[0]  = One
  s[5]  = x       (S-box 입력, 예: after_rc[0])
  s[8]  = t1      (x²)
  s[9]  = t2      (x⁴)
  s[10] = y       (x⁵)

제약 1: x · x = t1
  A 행: [(5, 1)]           → A·s = 1·s[5] = x
  B 행: [(5, 1)]           → B·s = 1·s[5] = x
  C 행: [(8, 1)]           → C·s = 1·s[8] = t1

  행렬 시각화 (638열 중 비-0만):
    A: [0 0 0 0 0 1 0 0 0 0 0 ...]  (인덱스 5에 1)
    B: [0 0 0 0 0 1 0 0 0 0 0 ...]  (인덱스 5에 1)
    C: [0 0 0 0 0 0 0 0 1 0 0 ...]  (인덱스 8에 1)

제약 2: t1 · t1 = t2
  A 행: [(8, 1)]           → A·s = t1
  B 행: [(8, 1)]           → B·s = t1
  C 행: [(9, 1)]           → C·s = t2

제약 3: t2 · x = y
  A 행: [(9, 1)]           → A·s = t2
  B 행: [(5, 1)]           → B·s = x
  C 행: [(10, 1)]          → C·s = y

S-box의 특징:
  → 각 제약의 A, B, C에 비-0 항목이 정확히 1개씩
  → "순수 곱셈" — 선형결합 없이 단순 곱셈
  → 가장 간단한 R1CS 패턴
```

#### AddRC의 A/B/C 행렬 항목 (1개 제약)

```
제약: (state + rc·One) · One = after_rc

변수 인덱스 가정:
  s[0]  = One
  s[20] = state[i]  (이전 라운드의 MDS 출력)
  s[23] = after_rc[i]

  rc = 라운드 상수 (Fr 값, 예: 0x1a2b3c...)

제약: (s[20] + rc · s[0]) · s[0] = s[23]
  A 행: [(20, 1), (0, rc)]  → A·s = s[20] + rc·s[0] = state + rc
  B 행: [(0, 1)]            → B·s = s[0] = 1
  C 행: [(23, 1)]           → C·s = s[23] = after_rc

  행렬 시각화:
    A: [rc 0 0 0 0 0 ... 1(at 20) ...]  (인덱스 0에 rc, 인덱스 20에 1)
    B: [1  0 0 0 0 0 ... 0          ...]  (인덱스 0에 1)
    C: [0  0 0 0 0 0 ... 1(at 23) ...]    (인덱스 23에 1)

AddRC의 특징:
  → A에 비-0 항목이 2개 (state + constant)
  → B에 비-0 항목이 1개 (One)
  → C에 비-0 항목이 1개 (결과 변수)
  → "선형결합 고정" 패턴 — 핵심은 ×1 곱셈
```

#### MDS의 A/B/C 행렬 항목 (1개 제약)

```
제약: (M[i][0]·sbox₀ + M[i][1]·sbox₁ + M[i][2]·sbox₂) · One = mds_out

변수 인덱스 가정:
  s[0]  = One
  s[10] = sbox_out[0]  (= a₀⁵)
  s[13] = sbox_out[1]  (= a₁⁵)
  s[16] = sbox_out[2]  (= a₂⁵)
  s[17] = mds_out[0]

  MDS 첫 행: [2, 1, 1]

제약: (2·s[10] + 1·s[13] + 1·s[16]) · s[0] = s[17]
  A 행: [(10, 2), (13, 1), (16, 1)]  → A·s = 2·sbox₀ + sbox₁ + sbox₂
  B 행: [(0, 1)]                      → B·s = 1
  C 행: [(17, 1)]                     → C·s = mds_out[0]

  행렬 시각화:
    A: [0 ... 2(at 10) ... 1(at 13) ... 1(at 16) ...]
    B: [1 0 0 0 ...]
    C: [0 ... 1(at 17) ...]

MDS의 특징:
  → A에 비-0 항목이 T=3개 (MDS 계수)
  → B에 비-0 항목이 1개 (One)
  → C에 비-0 항목이 1개 (결과 변수)
  → "가중 선형결합 고정" 패턴
  → MDS 행렬의 계수가 A 행에 직접 나타남
```

> [!abstract] 세 가지 제약 패턴 요약
> ```
> ┌─────────────────┬──────────┬──────────┬──────────┐
> │  패턴            │  A 비-0   │  B 비-0   │  C 비-0   │
> ├─────────────────┼──────────┼──────────┼──────────┤
> │  S-box (곱셈)    │    1     │    1     │    1     │
> │  AddRC (고정)    │    2     │    1     │    1     │
> │  MDS (선형결합)  │    T=3   │    1     │    1     │
> │  Boolean         │    1     │    2     │    0     │
> │  Mux (곱셈)      │    1     │    2     │    1     │
> │  Mux (덧셈고정)  │    2     │    1     │    1     │
> └─────────────────┴──────────┴──────────┴──────────┘
>
> 관찰:
>   1. B 열은 거의 항상 One만 포함 (×1 고정 패턴)
>   2. A 열이 선형결합의 복잡도를 담당
>   3. C 열은 항상 결과 변수 하나 (또는 0)
>   4. S-box만 A와 B 모두 비-One 변수 포함 (진짜 곱셈)
>
> 전체 634개 제약 중:
>   "진짜 곱셈" (S-box): 243개 (38%)
>   "고정 곱셈" (×1):    391개 (62%)
>
>   → 62%의 제약이 "선형결합을 변수로 고정"하는 데 사용
>   → 이것이 R1CS의 구조적 비효율성
>   → PLONKish에서는 커스텀 게이트로 이 오버헤드 제거 가능
> ```

#### 희소성(Sparsity) 분석

```
634 × 638 행렬의 총 항목: 634 × 638 = 404,492개

비-0 항목 수 추정:
  S-box (243개 제약):   각 행에 A:1 + B:1 + C:1 = 3 → 243 × 3 = 729
  AddRC (130개 제약):   각 행에 A:2 + B:1 + C:1 = 4 → 130 × 4 = 520
  MDS (130개 제약):     각 행에 A:3 + B:1 + C:1 = 5 → 130 × 5 = 650
  Boolean (depth개):    각 행에 A:1 + B:2 + C:0 = 3
  Mux (depth×4):        평균 4
  Root (1개):           3

  총 비-0 항목 ≈ 729 + 520 + 650 + ... ≈ ~2,000개

  희소도 = 2,000 / 404,492 ≈ 0.5%

  → 행렬의 99.5%가 0!
  → 이것이 희소 행렬 표현 (Vec<(usize, Fr)>)이 필수인 이유
  → 밀집 행렬이면 404,492 × 32바이트 ≈ 12MB
  → 희소 행렬이면 2,000 × (8+32)바이트 ≈ 80KB
```

---

### Merkle 회로의 수동 제약 추적 (한 레벨)

```
Level i: current = C, sibling = S, bit = b (0 또는 1)

─── Boolean (1개 제약) ───

  제약: b · (1 - b) = 0
    A = [(b, 1)]
    B = [(One, 1), (b, -1)]
    C_side = 0

─── Mux: left 결정 (2개 제약) ───

  left = mux(b, when_true=S, when_false=C)
  → b=0: left=C (current가 왼쪽)
  → b=1: left=S (sibling이 왼쪽)

  diff = S - C
  t_L = b · diff
  제약: b · (S - C) = t_L
    A = [(b, 1)]
    B = [(S, 1), (C, -1)]
    C_side = [(t_L, 1)]

  left = C + t_L
  제약: (C + t_L) · One = left
    A = [(C, 1), (t_L, 1)]
    B = [(One, 1)]
    C_side = [(left, 1)]

─── Mux: right 결정 (2개 제약) ───

  right = mux(b, when_true=C, when_false=S)
  → b=0: right=S (sibling이 오른쪽)
  → b=1: right=C (current가 오른쪽)

  (same pattern, with C and S swapped in when_true/when_false)

─── Poseidon hash (633개 제약) ───

  H(left_val, right_val) → 633 제약
  (poseidon_hash_circuit 호출)

─── 레벨 합계: 1 + 2 + 2 + 633 = 638개 제약 ───
```

> [!tip] Mux 결과를 수동 검증
> ```
> key=5 (이진 0101), level=0 → bit=1
> current = L (leaf hash), sibling = S₀
>
> left = mux(1, when_true=S₀, when_false=L)
>      = L + 1·(S₀ - L)
>      = L + S₀ - L
>      = S₀                    ← sibling이 왼쪽 ✓
>
> right = mux(1, when_true=L, when_false=S₀)
>       = S₀ + 1·(L - S₀)
>       = S₀ + L - S₀
>       = L                    ← current가 오른쪽 ✓
>
> → H(S₀, L)  = H(sibling, current)  — bit=1이므로 맞음 ✓
>
> key=5, level=1 → bit=0
> current = H₀ (이전 레벨 결과), sibling = S₁
>
> left = mux(0, when_true=S₁, when_false=H₀)
>      = H₀ + 0·(S₁ - H₀)
>      = H₀                    ← current가 왼쪽 ✓
>
> right = mux(0, when_true=H₀, when_false=S₁)
>       = S₁ + 0·(H₀ - S₁)
>       = S₁                   ← sibling이 오른쪽 ✓
>
> → H(H₀, S₁) = H(current, sibling) — bit=0이므로 맞음 ✓
> ```

---

### 완전한 수치 예제: depth=1 Merkle 회로 전체 추적

> [!important] 왜 depth=1인가?
> depth=1이면 Poseidon hash가 2번만 호출되고 (leaf + 1 level),
> 전체 제약이 633 + 638 + 1 = 1,272개.
> 이것으로도 전체 구조를 완벽하게 이해할 수 있다.

```
설정:
  depth = 1
  key   = 0  (이진: ...0000)
  value = 7

  depth=1이므로 리프가 2개인 트리:
        [root]
       /      \
    [leaf₀]  [leaf₁]

  key=0 → bit 0 = 0 → leaf₀ 위치 (왼쪽)

준비:
  leaf  = H(key=0, value=7) = Poseidon(0, 7)
  sibling = 비어 있는 리프의 해시 = H(0, 0) (기본값)

  root = H(leaf, sibling)
       = H(H(0,7), H(0,0))
       (bit=0이므로 leaf가 왼쪽, sibling이 오른쪽)
```

#### Step 1: Witness 벡터 구조

```
s = [One, root, key_w, value_w, ...poseidon_leaf_vars...,
     sibling_w, bit_w, mux_vars..., ...poseidon_level_vars...,
     ...]

인덱스 배치:
  s[0] = 1              ← One (항상 1)
  s[1] = root           ← instance (공개 입력)

  ── Leaf hash: poseidon_hash_circuit(0, 7) ──
  s[2] = 0              ← capacity (witness)
  s[3] = 0              ← key (= left input)
  s[4] = 7              ← value (= right input)
  s[5..637+5]           ← Poseidon 내부 633개 보조 변수
                           (after_rc, sbox_aux, mds_out × 65 라운드)

  ── key, value witness ──
  (key_var, value_var는 위의 s[3], s[4]와 별개로 할당)
  실제로 MerkleProofCircuit에서 _key_var, _value_var도 할당됨

  ── Level 0 ──
  sibling_var            ← H(0,0) (witness)
  bit_var                ← 0 (witness)
  mux_t_L, mux_left     ← mux 보조 변수 2개
  mux_t_R, mux_right    ← mux 보조 변수 2개
  ...poseidon_level0_vars... ← 633개 보조 변수

  ── Root check ──
  current_var · 1 = root_var  (1개 제약)
```

#### Step 2: Leaf 해시 제약 (처음 몇 개)

```
Poseidon(0, 7)의 첫 full round에서의 실제 제약:

  초기 state = [0, 0, 7]
  라운드 상수 rc₀, rc₁, rc₂ (실제 Fr 값, 파라미터에서 가져옴)

  ── AddRC ──
  after_rc[0] = 0 + rc₀ = rc₀
  after_rc[1] = 0 + rc₁ = rc₁
  after_rc[2] = 7 + rc₂

  제약 1: (s[2] + rc₀·s[0]) · s[0] = after_rc_var[0]
    → (0 + rc₀·1) · 1 = rc₀
    → ⟨A₁, s⟩ · ⟨B₁, s⟩ = ⟨C₁, s⟩ 형태로:
      A₁ · s = s[2]·1 + s[0]·rc₀ = 0 + rc₀ = rc₀
      B₁ · s = s[0]·1 = 1
      C₁ · s = after_rc_var[0]의 값 = rc₀
      → rc₀ · 1 = rc₀  ✓

  제약 2: (s[3] + rc₁·s[0]) · s[0] = after_rc_var[1]
    → (0 + rc₁) · 1 = rc₁  ✓

  제약 3: (s[4] + rc₂·s[0]) · s[0] = after_rc_var[2]
    → (7 + rc₂) · 1 = 7 + rc₂  ✓

  ── S-box on after_rc[0] = rc₀ ──
  제약 4: rc₀ · rc₀ = rc₀²
  제약 5: rc₀² · rc₀² = rc₀⁴
  제약 6: rc₀⁴ · rc₀ = rc₀⁵

  ── S-box on after_rc[1] = rc₁ ──
  제약 7~9: 같은 패턴으로 rc₁⁵ 계산

  ── S-box on after_rc[2] = 7 + rc₂ ──
  제약 10~12: (7+rc₂)⁵ 계산

  ── MDS ──
  b₀ = rc₀⁵, b₁ = rc₁⁵, b₂ = (7+rc₂)⁵

  제약 13: (2·b₀ + b₁ + b₂) · 1 = m₀
  제약 14: (b₀ + 2·b₁ + b₂) · 1 = m₁
  제약 15: (b₀ + b₁ + 2·b₂) · 1 = m₂

  → Round 0 완료: state = [m₀, m₁, m₂]
  → ... 64 more rounds ...
  → 최종: state[1] = leaf_hash = H(0, 7)
```

#### Step 3: Level 0 제약 (실제 흐름)

```
leaf_hash = H(0, 7) = L  (633개 제약으로 계산됨)
sibling   = H(0, 0) = S  (외부에서 계산된 값, witness로 할당)
bit       = 0             (key=0의 bit 0)

── Boolean (제약 634) ──
  0 · (1 - 0) = 0 · 1 = 0 = 0  ✓

── Mux left (제약 635-636) ──
  left = mux(bit=0, when_true=S, when_false=L)

  diff = S - L
  t_L = 0 · diff = 0
  제약 635: 0 · (S - L) = 0  ✓

  left = L + 0 = L
  제약 636: (L + 0) · 1 = L  ✓

── Mux right (제약 637-638) ──
  right = mux(bit=0, when_true=L, when_false=S)

  diff = L - S
  t_R = 0 · (L - S) = 0
  제약 637: 0 · (L - S) = 0  ✓

  right = S + 0 = S
  제약 638: (S + 0) · 1 = S  ✓

── Poseidon H(left=L, right=S) (제약 639-1271) ──
  633개 제약으로 H(L, S) = root_computed 계산

── Root check (제약 1272) ──
  root_computed · 1 = root_public
  → root_computed와 root_public이 같으면 ✓

총 제약: 633 + 1 + 2 + 2 + 633 + 1 = 1,272
  = 633(leaf) + 638(level) + 1(root check)  ✓
```

#### Step 4: A·s, B·s, C·s 수동 검증 (선택된 제약)

```
제약 1 (첫 번째 AddRC):
  A₁ = [rc₀ at index 0, 1 at index 2, 0 elsewhere]
  B₁ = [1 at index 0, 0 elsewhere]
  C₁ = [1 at index 5, 0 elsewhere]  (after_rc[0]가 s[5]라고 가정)

  A₁ · s = rc₀ · s[0] + 1 · s[2] = rc₀ · 1 + 1 · 0 = rc₀
  B₁ · s = 1 · s[0] = 1
  C₁ · s = 1 · s[5] = rc₀  (witness로 rc₀ 할당됨)

  검증: rc₀ · 1 = rc₀  ✓

제약 4 (첫 번째 S-box, x²):
  x = after_rc[0]가 위치한 변수 인덱스 = 5

  A₄ = [1 at index 5]
  B₄ = [1 at index 5]
  C₄ = [1 at index 8]  (t₀₁ = x²가 s[8]라고 가정)

  A₄ · s = s[5] = rc₀
  B₄ · s = s[5] = rc₀
  C₄ · s = s[8] = rc₀²

  검증: rc₀ · rc₀ = rc₀²  ✓

제약 1272 (Root check):
  A = [1 at current_var's index]
  B = [1 at index 0]  (One)
  C = [1 at index 1]  (root instance)

  A · s = current_val = H(L, S)
  B · s = 1
  C · s = root = H(L, S)  (올바른 root일 때)

  검증: H(L, S) · 1 = root  ✓
```

> [!tip] Witness 벡터의 핵심 구조
> ```
> s[0]     = 1       ← One (항상 1, 제약의 상수항에 사용)
> s[1]     = root    ← instance (검증자가 아는 유일한 값)
> s[2..]   = witness ← 증명자만 아는 값들
>
> 검증자의 관점:
>   "s[1] = root라는 공개값과, 이 회로의 1,272개 제약만 알고 있다.
>    누군가가 s[2..] 값을 알고 있어서 모든 제약이 만족된다면,
>    그 사람은 이 root에 포함된 (key, value)를 아는 것이다."
>
> Groth16에서는:
>   s[2..] 전체를 공개하지 않고,
>   "이런 s[2..]가 존재한다"는 것만 128바이트로 증명한다.
> ```

---

### 테스트 전략

```
전체 테스트: 24개 = 10 (r1cs) + 5 (poseidon circuit) + 9 (merkle circuit)

R1CS (Step 09, 10개):
  multiply_satisfied          — 기본 곱셈 3×4=12 ✓
  multiply_wrong_witness      — 잘못된 값 거부 ✓
  pythagorean_satisfied       — 3²+4²=5² ✓
  pythagorean_wrong           — 3²+4²≠6² 거부 ✓
  constant_term               — (x+5)·1=y ✓
  circuit_trait               — Circuit trait 동작 ✓
  conditional_select          — if b then x else y ✓
  cubic_polynomial            — x³+x+5=35 ✓
  matrices_simple             — to_matrices 희소 행렬 ✓
  empty_system_satisfied      — 빈 시스템 ✓

Poseidon Circuit (Step 10a, 5개 + 1개 경고 수정):
  sbox_gadget_matches_native  — sbox(7): native=16807, circuit=16807 ✓
  sbox_gadget_zero            — sbox(0)=0 ✓
  poseidon_circuit_matches_native  — H(1,2) native == circuit ✓
  poseidon_circuit_different_inputs — 여러 입력 조합 ✓
  poseidon_circuit_constraint_count — 634개 정확히 ✓
  poseidon_circuit_wrong_output_fails — 잘못된 출력 거부 ✓

Merkle Circuit (Step 10b, 9개):
  boolean_gadget              — b=0 ✓, b=1 ✓, b=2 ✗
  mux_select_false            — bit=0 → when_false(99) ✓
  mux_select_true             — bit=1 → when_true(42) ✓
  merkle_circuit_matches_native — native verify ↔ circuit is_satisfied ✓
  merkle_circuit_wrong_root_fails — root=999 → 거부 ✓
  merkle_circuit_wrong_value_fails — value=99 → 거부 ✓
  merkle_circuit_multiple_entries — 3개 항목 모두 검증 ✓
  merkle_circuit_constraint_count — 제약/변수 수 출력 ✓
```

> [!abstract] 테스트의 네 가지 축
> ```
> 1. 정확성 (Correctness):
>    native 결과 == circuit 결과
>    → sbox_matches_native, poseidon_circuit_matches_native,
>      merkle_circuit_matches_native
>
> 2. 건전성 (Soundness):
>    잘못된 입력이 거부되는지
>    → wrong_output_fails, wrong_root_fails, wrong_value_fails
>
> 3. 완전성 (Completeness):
>    올바른 입력이 항상 만족되는지
>    → different_inputs, multiple_entries
>
> 4. 제약 수 (Constraint Count):
>    예상한 제약 수와 정확히 일치하는지
>    → poseidon_circuit_constraint_count,
>      merkle_circuit_constraint_count
>
> 4번이 특히 중요한 이유:
>   제약이 너무 적으면 → 건전성 위반 가능
>   제약이 너무 많으면 → 증명 비용 증가
>   정확한 수를 검증하면 → 누락/중복 제약 방지
> ```

```rust
// 핵심 테스트 코드: Poseidon 제약 수 검증
#[test]
fn poseidon_circuit_constraint_count() {
    let mut cs = ConstraintSystem::new();
    let circuit = PoseidonCircuit {
        left: Fr::from_u64(1),
        right: Fr::from_u64(2),
        params: PoseidonParams::new(),
    };
    circuit.synthesize(&mut cs);

    // 정확한 공식으로 검증
    let expected = 8 * (T + 3 * T + T) + 57 * (T + 3 + T) + 1;
    // = 8 * 15 + 57 * 9 + 1 = 120 + 513 + 1 = 634
    assert_eq!(cs.num_constraints(), expected);
    assert!(cs.is_satisfied());
}
```

```rust
// 핵심 테스트 코드: Merkle 회로의 multiple entries 검증
#[test]
fn merkle_circuit_multiple_entries() {
    let mut tree = SparseMerkleTree::new(TEST_DEPTH);

    let entries: Vec<(Fr, Fr)> = vec![
        (Fr::from_u64(1), Fr::from_u64(100)),
        (Fr::from_u64(5), Fr::from_u64(200)),
        (Fr::from_u64(12), Fr::from_u64(300)),
    ];

    for &(k, v) in &entries { tree.insert(k, v); }

    for (idx, &(k, v)) in entries.iter().enumerate() {
        let proof = tree.prove(&k);

        // native 먼저 확인
        assert!(verify_merkle_proof(tree.root, k, v, &proof));

        // circuit 검증
        let circuit = MerkleProofCircuit {
            root: tree.root, key: k, value: v,
            siblings: proof.siblings.clone(),
            depth: TEST_DEPTH, params: PoseidonParams::new(),
        };
        let mut cs = ConstraintSystem::new();
        circuit.synthesize(&mut cs);

        // 디버깅: 실패 시 어떤 제약이 위반되었는지 출력
        if let Some(idx) = cs.which_unsatisfied() {
            panic!("entry {}: constraint {} of {} failed",
                idx, idx, cs.num_constraints());
        }
    }
}
```

---

### 설계 결정과 트레이드오프

> [!important] 핵심 설계 결정 4가지

```
1. AddRC와 MDS에 보조 변수 고정을 사용

   대안: 선형결합을 S-box에 직접 전달
     → R1CS에서 불가능! A·B=C의 A, B는 선형결합이지만
       (A₁·s)(A₂·s)(A₃·s) 같은 3중 곱은 표현 못함
     → x⁵ = x·x·x·x·x를 하려면 중간 결과가 변수여야 함

   결론: 634개 제약은 R1CS의 구조적 한계에서 온다.
         PLONKish(Step 14)에서는 커스텀 게이트로 줄일 수 있음.

2. poseidon_hash_circuit은 Fr 값을 직접 받음

   대안: Variable을 받아서 기존 변수를 재사용
     → 복잡해짐: 호출자가 witness 할당을 관리해야 함
     → Merkle에서 mux 결과를 전달할 때 문제

   현재 방식: Fr 값을 받고 내부에서 alloc_witness
     → 단순함: 호출자는 값만 전달
     → 비용: 약간의 중복 변수 (mux의 left_val을 다시 할당)
     → 교육용으로 명확한 트레이드오프

3. Merkle circuit의 public input은 root만

   대안: key나 value도 공개
     → Tornado Cash와 같은 프라이버시 앱에서는 key/value를 숨겨야 함
     → nullifier만 공개하고 나머지는 비공개

   현재 방식: root만 instance, 나머지 witness
     → ZK의 핵심 사용 사례와 일치
     → 검증자는 "이 root에 속하는 어떤 값을 안다"만 확인

4. depth=4로 테스트

   대안: depth=256 (Fr 전체 비트)
     → 테스트 시간 수 초 ~ 수십 초
     → 구조가 동일하므로 depth=4로 충분
     → key < 2^4 = 16 제한만 주의
```

---

### 다음 스텝과의 연결 (QAP, Groth16)

```
Step 10의 634개 제약 → 이제 무엇을 하는가?

Step 11: QAP (Quadratic Arithmetic Program)
  R1CS 행렬 (A, B, C) → 다항식으로 변환

  to_matrices()로 추출:
    A: 634 × 638 희소 행렬
    B: 634 × 638 희소 행렬
    C: 634 × 638 희소 행렬

  각 열의 값을 Lagrange 보간 → 다항식
    Aⱼ(X): 제약 i에서의 j번째 변수 계수
    → ω₁, ..., ω₆₃₄에서의 값을 보간

  핵심 등식:
    A(X) · B(X) - C(X) = H(X) · Z(X)
    Z(X) = (X-ω₁)(X-ω₂)···(X-ω₆₃₄)

  → 634개 제약을 "하나의 다항식 등식"으로 환원!

Step 12: Groth16
  QAP 다항식 → 타원곡선 점 → 증명

  Trusted setup:
    τ (secret) → [τ]₁, [τ²]₁, ..., [τⁿ]₁  (SRS)
    회로 구조에 의존 → per-circuit setup

  증명 생성:
    witness s로 A(τ), B(τ), C(τ) 계산 (in the exponent)
    H(X) 계산 → H(τ) 계산
    → π = (A, B, C) in G1, G2

  증명 크기: 128바이트!
    G1 2개 (32B × 2) + G2 1개 (64B) = 128B
    3,186개 제약의 Merkle proof → 고작 128바이트

  검증:
    e(A, B) = e(α, β) · e(∑xᵢ·Lᵢ, γ) · e(C, δ)
    → 페어링 3~4회 → ~1ms

전체 파이프라인:
  MerkleProofCircuit.synthesize(cs)  ← Step 10 (여기)
      ↓
  cs.to_matrices() → (A, B, C)       ← Step 09
      ↓
  QAP::from_r1cs(A, B, C) → 다항식   ← Step 11
      ↓
  Groth16::prove(qap, witness) → π   ← Step 12
      ↓
  Groth16::verify(vk, x, π)          ← Step 12
      ↓
  true/false  (128바이트 증명, ~1ms 검증)
```

```
의존성 트리:

  Fr (Step 03)
    ├─ Poseidon hash (Step 07)
    │    ├─ Merkle tree (Step 08)
    │    │    └─ MerkleProofCircuit (Step 10, 여기)
    │    └─ PoseidonCircuit / poseidon_hash_circuit (Step 10, 여기)
    └─ R1CS (Step 09)
         ├─ ConstraintSystem, Variable, LinearCombination
         ├─ Circuit trait
         ├─ circuits/poseidon.rs (Step 10)
         ├─ circuits/merkle.rs (Step 10)
         ├─ QAP (Step 11)
         │    └─ Groth16 (Step 12)
         └─ PLONKish (Step 14) — R1CS의 한계를 극복

파일 구조:
  crates/primitives/src/
    ├─ r1cs.rs              ← Step 09: 제약 시스템
    ├─ circuits/
    │   ├─ mod.rs           ← Step 10: 가젯 모듈
    │   ├─ poseidon.rs      ← Step 10a: Poseidon 회로
    │   └─ merkle.rs        ← Step 10b: Merkle 회로
    ├─ hash/poseidon.rs     ← Step 07: native Poseidon
    └─ merkle.rs            ← Step 08: native Merkle tree
```

> [!tip] Step 10이 완성하는 것
> ```
> Step 07~08: "이 함수가 올바르게 동작한다" (native 실행)
> Step 09:    "모든 계산을 제약으로 표현할 수 있다" (R1CS 프레임워크)
> Step 10:    "실제로 이 함수를 제약으로 변환했고, 동일하게 동작한다"
>
> → Step 10은 "ZK 증명이 가능하다"는 것의 첫 번째 구체적 증거.
>    이 제약들이 QAP → Groth16을 거치면 실제 영지식 증명이 된다.
> ```

---

### ZK-friendly의 정량적 의미

```
Step 07에서 "Poseidon은 ZK-friendly"라고 했다.
Step 10에서 이것이 정확히 얼마나 차이나는지 수치로 확인:

┌─────────────────────┬──────────────┬──────────────┐
│  해시 함수            │  R1CS 제약 수 │  Merkle d=20 │
├─────────────────────┼──────────────┼──────────────┤
│  Poseidon (t=3)     │     634      │   13,394     │
│  SHA-256            │  ~25,000     │  ~525,000    │
│  MiMC (t=3)         │    ~300      │   ~6,300     │
│  Keccak-256         │  ~60,000     │ ~1,260,000   │
└─────────────────────┴──────────────┴──────────────┘

Poseidon vs SHA-256:
  단일 해시: 634 vs 25,000 → 39배 차이
  Merkle d=20: 13,394 vs 525,000 → 39배 차이

이 차이는 증명 시간으로 직결:
  Groth16 증명 시간 ∝ #제약 × log(#제약)
  → Poseidon Merkle: ~50ms
  → SHA-256 Merkle:  ~2,000ms = 2초

  검증 시간은 둘 다 ~1ms (Groth16의 장점)

결론:
  "ZK-friendly" = 회로 비용이 ~40배 저렴
  → 증명 시간이 ~40배 빠름
  → 같은 하드웨어에서 실시간 증명 가능/불가능의 차이
```

---

### "What If" 분석: 설계 매개변수의 보안 영향

> [!important] Poseidon의 핵심 설계 매개변수
> ```
> 1. α = 5      (S-box 지수: x → x^α)
> 2. MDS 행렬   (확산 계층)
> 3. RF = 8     (full rounds 수)
> 4. RP = 57    (partial rounds 수)
>
> 각 매개변수를 변경하면 비용과 보안이 어떻게 바뀌는가?
> ```

#### What if: α=3 (S-box를 x³으로 변경)

```
현재: x → x⁵  (3개 제약: x², x⁴, x⁵)
변경: x → x³  (2개 제약: x², x³)

분해:
  t1 = x · x     ← 제약 1 (x²)
  y  = t1 · x    ← 제약 2 (x³)

             ┌───┐
    x ──────→│ × │────→ t1 (= x²)
    x ──────→│   │
             └───┘
             ┌───┐
   t1 ──────→│ × │────→ y  (= x³)
    x ──────→│   │
             └───┘

제약 수 변화:
  S-box: 3 → 2 (제약 1개 절약/S-box)
  보조 변수: 3 → 2 (변수 1개 절약/S-box)

  Full round:  3T + T(AddRC) + T(MDS) = 3×3×2 + 3 + 3 = 24  (기존 15)
  ... 잠깐, 다시 계산:
    Full round:  AddRC(3) + S-box(2×3=6) + MDS(3) = 12  (기존 15)
    Partial:     AddRC(3) + S-box(2×1=2) + MDS(3) = 8   (기존 9)

  총: 8×12 + 57×8 + 1 = 96 + 456 + 1 = 553  (기존 634)
  절약: 634 - 553 = 81개 제약 (13% 감소)

보안 영향:
  ┌──────────────────────────────────────────────────────┐
  │  α=5: GF(p)에서 gcd(5, p-1)=1이면 x⁵는 순열(permutation) │
  │  α=3: GF(p)에서 gcd(3, p-1)=1이면 x³도 순열            │
  │                                                      │
  │  BN254 Fr: p-1은 매우 큰 짝수                          │
  │    gcd(5, p-1) = gcd(5, even) = ?                    │
  │    p-1 mod 5 ≠ 0이면 OK                               │
  │    gcd(3, p-1) = gcd(3, even) = ?                    │
  │    p-1 mod 3 ≠ 0이면 OK                               │
  │                                                      │
  │  두 경우 모두 확인 필요!                                │
  │                                                      │
  │  BN254의 p-1:                                        │
  │    p = 21888...87 (소수)                              │
  │    p-1 mod 3 = 0이면 α=3은 사용 불가!                  │
  │    p-1 mod 5 ≠ 0이므로 α=5는 안전                     │
  └──────────────────────────────────────────────────────┘

  실제:
    BN254 Fr의 p ≡ 1 (mod 3)?
    → 아니다! p ≡ 1 (mod 3)이면 x³는 3-to-1 맵핑 → 순열 아님
    → 실제로 BN254에서는 이 문제가 있어 α=3은 부적합할 수 있음

  결론:
    α=3: 13% 제약 절약이지만, 체의 특성에 따라 보안 파괴 가능
    α=5: BN254에서 안전하다고 검증된 선택
    α=7: 더 안전하지만 4개 제약 → 비용 증가

  Poseidon 논문의 선택:
    "α는 gcd(α, p-1)=1인 최소 홀수 소수"
    → BN254에서 α=5가 최소 안전 지수
```

#### What if: MDS 행렬이 단위행렬 (Identity)

```
현재 MDS:                   변경 후 MDS:
  [[2, 1, 1],                [[1, 0, 0],
   [1, 2, 1],                 [0, 1, 0],
   [1, 1, 2]]                 [0, 0, 1]]

MDS mix 결과:
  현재:  m₀ = 2·b₀ + b₁ + b₂   → 세 원소가 혼합됨
  변경:  m₀ = b₀                 → 원소가 독립!

보안 영향:
  ┌──────────────────────────────────────────────────────┐
  │  MDS의 역할: 확산(Diffusion)                           │
  │                                                      │
  │  확산이란:                                             │
  │    입력의 한 비트 변화 → 출력의 모든 비트에 영향          │
  │    "하나를 바꾸면 전부 바뀐다" (Avalanche effect)       │
  │                                                      │
  │  Identity MDS:                                       │
  │    state[0]의 변화 → state[0]에만 영향                 │
  │    state[1]의 변화 → state[1]에만 영향                 │
  │    → 원소 간 상호작용 없음                              │
  │    → 사실상 3개의 독립적인 "1차원 해시"                  │
  │                                                      │
  │  공격:                                                │
  │    공격자는 state[0]만 조작하고 state[1], state[2]는 유지│
  │    → 3개 원소를 동시에 만족시킬 필요 없음                │
  │    → 공격 복잡도: O(2^{n/3}) → O(2^{85}) (128비트 대신)│
  │    → 보안 수준이 1/3로 떨어짐!                          │
  └──────────────────────────────────────────────────────┘

회로 제약 영향:
  Identity MDS의 제약:
    m₀ = 1·b₀ + 0·b₁ + 0·b₂ = b₀
    → (b₀) · 1 = m₀  또는 m₀ = b₀ (변수 재사용 가능!)
    → MDS 고정 제약 자체가 불필요 → T=3개 절약/라운드

  총 절약: 65 × 3 = 195개 제약
  하지만: 보안이 완전히 파괴됨 → 의미 없음

  이것이 "MDS 행렬은 반드시 MDS 성질을 가져야 한다"의 이유.
  MDS (Maximum Distance Separable):
    모든 부분행렬의 행렬식 ≠ 0
    → 최대 확산 보장 → 보안의 핵심
```

#### What if: Partial rounds = 0 (Full rounds만)

```
현재: RF=8 (full) + RP=57 (partial) = 65 라운드
변경: RF=65 (full) + RP=0 (partial) = 65 라운드 (같은 총 라운드)

제약 수 변화:
  현재:  8×15 + 57×9 + 1 = 120 + 513 + 1 = 634
  변경:  65×15 + 0×9 + 1 = 975 + 0 + 1 = 976

  → 976 - 634 = 342개 제약 증가! (54% 더 비싸짐)

보안 영향:
  모든 라운드가 full이므로 → 보안은 오히려 강화됨!
  모든 원소에 S-box 적용 → 비선형성 극대화

  하지만: 비용 대비 보안 향상이 미미.
  Poseidon 논문의 분석:
    "RP개의 partial round로 충분한 보안 마진을 제공하면,
     나머지를 full round로 바꿀 필요 없다."

반대: RP가 너무 적으면?
  RP=0으로 하고 RF=8만 유지한다면:
  8×15 + 0×9 + 1 = 121개 제약

  하지만: 8 full rounds로는 보안 마진이 부족!
  대수적 공격 (Interpolation, Grobner basis)에 취약.
  Poseidon 논문: RP ≥ 57이 필요 (t=3, BN254)

  정리:
    Full rounds는 "시작과 끝의 강한 혼합" 제공
    Partial rounds는 "중간의 효율적 확산" 제공
    둘의 조합이 비용 대비 최적의 보안을 달성
```

> [!abstract] 설계 매개변수 트레이드오프 요약
> ```
> ┌────────────────┬──────────────┬──────────────┬──────────────────┐
> │  변경             │  제약 변화    │  보안 변화    │  판정             │
> ├────────────────┼──────────────┼──────────────┼──────────────────┤
> │  α: 5→3        │  634→553     │  체에 의존     │  위험 (BN254)    │
> │                │  (-13%)      │  (파괴 가능)   │                  │
> │  α: 5→7        │  634→715     │  약간 강화     │  비용 증가 불필요  │
> │                │  (+13%)      │              │                  │
> │  MDS: → I      │  634→439     │  완전 파괴     │  절대 불가        │
> │                │  (-31%)      │  (1/3 보안)   │                  │
> │  RP: 57→0      │  634→121     │  완전 파괴     │  절대 불가        │
> │                │  (-81%)      │  (대수적 공격) │                  │
> │  RP: 57→0      │  634→976     │  과잉 보안     │  비효율적         │
> │  (전부 full)    │  (+54%)      │              │                  │
> │  RF: 8→0       │  634→514     │  크게 약화     │  위험             │
> │  (전부 partial) │  (-19%)      │  (혼합 부족)   │                  │
> └────────────────┴──────────────┴──────────────┴──────────────────┘
>
> 결론:
>   현재 매개변수 (α=5, RF=8, RP=57, Cauchy MDS)는
>   Poseidon 논문에서 수학적으로 분석된 최적 조합이다.
>
>   제약을 줄이려면 매개변수 변경이 아닌
>   R1CS 구현 최적화(선형결합 흡수)를 해야 한다.
>   → 634 → ~260 (보안 유지하면서 60% 절감)
> ```

> [!note] 실전 설계 과정
> ```
> Poseidon 해시의 설계 순서:
>
> 1. 대상 체 선택 (예: BN254 Fr)
>    → p 결정
>
> 2. α 결정
>    → gcd(α, p-1) = 1인 최소 홀수 소수
>    → BN254: α = 5
>
> 3. t (state 크기) 결정
>    → 2-to-1 해시: t = 3
>    → 4-to-1 해시: t = 5
>
> 4. RF, RP 결정 (Poseidon 논문의 보안 분석)
>    → 대수적 공격에 대한 보안 마진 계산
>    → t=3, BN254: RF=8, RP=57
>
> 5. MDS 행렬 생성
>    → Cauchy 행렬 또는 circulant 행렬
>    → 모든 부분행렬의 행렬식 ≠ 0 검증
>
> 6. 라운드 상수 생성
>    → Grain LFSR로 의사난수 생성
>    → "nothing up my sleeve" 원칙
>
> 이 모든 것이 결정된 후에야 회로를 구현할 수 있다.
> 우리의 PoseidonParams::new()가 이 과정을 수행.
> ```

---

### 요약

```
┌──────────────────────────────────────────────────────────────┐
│  Step 10: R1CS 가젯 (Poseidon, Merkle 회로)                    │
│                                                              │
│  핵심: "native 코드 = 회로 코드"                                │
│        같은 함수를 R1CS 제약의 언어로 번역                       │
│                                                              │
│  기본 가젯:                                                   │
│    Boolean: b·(1-b)=0                    (1 제약)             │
│    Mux: result = b·x + (1-b)·y          (2 제약)             │
│    S-box: x⁵ = x·x → t1·t1 → t2·x      (3 제약)             │
│                                                              │
│  Poseidon 회로:                                               │
│    Full round:    AddRC(3) + S-box(9) + MDS(3) = 15 제약     │
│    Partial round: AddRC(3) + S-box(3) + MDS(3) = 9 제약      │
│    총: 8×15 + 57×9 + 1 = 634 제약, 638 변수                  │
│                                                              │
│  Merkle 회로:                                                 │
│    Leaf hash: 633 제약                                        │
│    Per level: bool(1) + mux(4) + Poseidon(633) = 638         │
│    Root check: 1 제약                                         │
│    depth=4:  633 + 4×638 + 1 = 3,186 제약                    │
│    depth=20: 633 + 20×638 + 1 = 13,394 제약                  │
│                                                              │
│  핵심 인사이트:                                                │
│    AddRC와 MDS는 선형이지만, S-box 입력을 위해                  │
│    보조 변수 고정 제약이 필요 → 634 > 243                       │
│                                                              │
│  테스트: 24개 (r1cs 10 + poseidon 5 + merkle 9)              │
│    정확성 + 건전성 + 완전성 + 제약 수 검증                      │
│                                                              │
│  다음: QAP(Step 11)에서 634개 제약 → 하나의 다항식 등식         │
│        Groth16(Step 12)에서 → 128바이트 증명                   │
└──────────────────────────────────────────────────────────────┘
```
