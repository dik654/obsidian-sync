## Step 07: Poseidon 해시 — ZK-friendly 해시 함수

### 핵심 질문: 왜 SHA-256이 아니라 Poseidon인가?

```
SHA-256:
  비트 연산 (XOR, 회전, 시프트)
  → ZK 회로에서 각 비트를 변수로 표현해야 함
  → 하나의 SHA-256 해시 ≈ 25,000 R1CS 제약

Poseidon:
  체 산술만 사용 (add, mul, pow)
  → ZK 회로에서 필드 원소 단위로 직접 연산
  → 하나의 Poseidon 해시 ≈ 250 R1CS 제약
```

> [!important] ZK-friendly의 의미
> ZK 회로는 **체 원소(Fr)** 위에서 동작한다. SHA-256의 비트 연산을 회로로 번역하면 비트마다 변수가 필요해서 비용 폭증. Poseidon은 처음부터 Fr.mul, Fr.pow만으로 설계되어 **회로 비용이 100배 저렴**하다.
>
> 나중에 R1CS 가젯(Step 10)에서 이 차이가 극적으로 드러남:
> - Poseidon 회로: S-box(x⁵)마다 곱셈 제약 2~3개
> - SHA 회로: XOR 하나마다 비트 분해 + 재조합

---

### SHA-256 vs Poseidon: 구조적 비교

```
┌─────────────────────────────────────────────────────────────────┐
│  SHA-256 (Merkle–Damgård + Davies–Meyer)                        │
│                                                                 │
│  입력: 512-bit 블록 (비트 단위)                                  │
│  상태: 256-bit (8 × 32-bit 워드)                                │
│                                                                 │
│  라운드 연산:                                                    │
│    Ch(e,f,g) = (e AND f) XOR (NOT e AND g)   ← 비트 연산!       │
│    Maj(a,b,c) = (a AND b) XOR (a AND c) XOR (b AND c)          │
│    Σ₀(a) = ROTR²(a) XOR ROTR¹³(a) XOR ROTR²²(a)              │
│                                                                 │
│  ZK 문제: AND, XOR, ROTR은 모두 비트 분해가 필요               │
│    32-bit AND → 32개 비트 변수 + 32개 곱셈 제약                 │
│    64 라운드 × 여러 비트연산 → ~25,000 제약                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Poseidon (SPN: Substitution-Permutation Network)               │
│                                                                 │
│  입력: Fr 원소 (필드 원소 단위)                                  │
│  상태: Fr × T (T개 필드 원소)                                    │
│                                                                 │
│  라운드 연산:                                                    │
│    AddRoundConstants:  state[i] += c_i    ← Fr 덧셈             │
│    S-box:              state[i] = state[i]⁵  ← Fr 곱셈          │
│    MDS mix:            state = M · state  ← Fr 곱셈             │
│                                                                 │
│  ZK 장점: 모든 연산이 체 산술 → R1CS에 직접 매핑                │
│    x⁵ = x · x · x · x · x → 곱셈 제약 3개                     │
│    65 라운드 × (S-box + MDS) → ~250 제약                        │
└─────────────────────────────────────────────────────────────────┘
```

> [!abstract] SPN vs Feistel
> 전통 암호(AES, SHA)도 SPN 구조를 사용하지만, 비트/바이트 단위 S-box.
> Poseidon은 **체 원소 단위** SPN — 같은 철학, 다른 대수 구조.
>
> | 구조 | S-box 단위 | 대표 | ZK 비용 |
> |------|-----------|------|---------|
> | 비트 SPN | 8-bit (S-box 테이블) | AES | 높음 |
> | Feistel | 32-bit 워드 | SHA-256 | 매우 높음 |
> | **체 SPN** | **Fr 원소 (254-bit)** | **Poseidon** | **낮음** |

---

### Poseidon이 Fr 위에서 동작하는 이유

```
페어링 시스템의 세 체:
  Fp  → G1 좌표, 확장체의 기초
  Fp2 → G2 좌표
  Fr  → 스칼라체, ZK witness의 값

Poseidon은 Fr 위에서 동작:
  → ZK 증명의 witness가 Fr 원소이므로
  → 해시 입력도 Fr, 출력도 Fr
  → 회로 안에서 자연스럽게 연결됨
```

> [!note] Fp가 아닌 Fr을 쓰는 이유
> Fp는 타원곡선 좌표용. ZK 회로(R1CS)의 모든 값은 **Fr** 원소로 표현된다.
> Poseidon이 Fr 위에서 동작하면 "native 코드 = 회로 코드"가 가능 — Step 10에서 확인.

---

### Poseidon 구조: Sponge + Permutation

```
┌─────────────────────────────────────────────────┐
│  Sponge Construction                            │
│                                                 │
│  state = [capacity | rate₀ | rate₁]             │
│         = [   0    | left  | right ]             │
│                                                 │
│  ┌───────────────────────────────────┐          │
│  │  Poseidon Permutation             │          │
│  │                                   │          │
│  │  RF/2 full rounds                 │          │
│  │    ↓                              │          │
│  │  RP partial rounds                │          │
│  │    ↓                              │          │
│  │  RF/2 full rounds                 │          │
│  └───────────────────────────────────┘          │
│                                                 │
│  output = state[1]  (첫 번째 rate 원소)           │
└─────────────────────────────────────────────────┘
```

> [!tip] Sponge 구조의 의미
> - **capacity** (1개): 보안 파라미터. 출력에 직접 노출되지 않아서 pre-image 공격 방어
> - **rate** (2개): 입력이 들어가고, 출력이 나오는 위치
> - width = capacity + rate = 1 + 2 = 3 (= T)

---

### Sponge의 보안 보장

```
Sponge 모델에서의 보안 수준:

  보안 비트 = min(capacity × field_bits / 2, capacity × field_bits)
            = min(254/2, 254)
            = 127 bits

capacity = 1이면:
  → pre-image resistance: 2^127 연산 필요
  → collision resistance: 2^127 연산 필요 (birthday bound)
```

> [!important] capacity가 보안을 결정하는 이유
> Sponge에서 capacity 부분은 **외부에 노출되지 않는다**.
> 공격자는 rate 부분만 관측 가능하므로, capacity가 크면 내부 상태를 추측하기 어렵다.
>
> ```
> 공격자 관점:
>   알 수 있는 것: state[1], state[2]  (rate)
>   알 수 없는 것: state[0]            (capacity)
>
>   state[0]은 254-bit Fr 원소 → 2^254 가능한 값
>   pre-image 공격: capacity 비트의 절반 = 127-bit 보안
> ```
>
> capacity를 2로 늘리면 (t=4) 254-bit 보안이 되지만, 비용도 증가.
> ZK에서 128-bit 보안이면 충분하므로 capacity=1이 표준.

---

### 파라미터: BN254 Fr

```
T  = 3     상태 너비 (capacity 1 + rate 2)
α  = 5     S-box 지수 (x → x⁵)
RF = 8     full rounds (4 + 4)
RP = 57    partial rounds
총 라운드  = 65
라운드 상수 = T × 65 = 195개
```

---

### 수학적 기초 I: 왜 α = 5인가 — S-box 순열 조건

#### S-box가 순열이어야 하는 이유

```
해시 함수의 permutation(순열):
  state 공간 Fr^T → Fr^T 의 전단사(bijection)

S-box x → x^α 이 Fr → Fr 전단사가 아니면:
  → 서로 다른 입력이 같은 출력을 만들 수 있음
  → 순열 전체가 전단사가 아니게 됨
  → pre-image resistance 파괴

전단사 조건: x → x^α 가 Fr* → Fr* 전단사
  ⟺ gcd(α, |Fr*|) = gcd(α, r-1) = 1
```

> [!important] 왜 gcd(α, r-1) = 1인가?
> Fr*은 위수 r-1인 **순환군**이다. 생성자를 g라 하면 모든 원소는 g^k 꼴.
>
> ```
> (g^k)^α = g^(kα)
>
> x → x^α 가 전단사
>   ⟺ k → kα (mod r-1) 이 전단사
>   ⟺ gcd(α, r-1) = 1
>
> gcd(α, r-1) > 1이면:
>   α·k₁ ≡ α·k₂ (mod r-1) 인 k₁ ≠ k₂가 존재
>   → g^(k₁) ≠ g^(k₂)이지만 (g^k₁)^α = (g^k₂)^α
>   → 충돌!
> ```

#### BN254 Fr에서의 실제 검증

```
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
r - 1 = 21888242871839275222246405745257275088548364400416034343698204186575808495616

r - 1의 소인수 분해 (부분):
  r - 1 = 2^28 × 3 × ...

따라서:
  gcd(3, r-1) = 3  → α=3은 불가! (r-1이 3의 배수)
  gcd(5, r-1) = ?

  (r-1) mod 5 = ?
  r mod 5 = 21888...617 mod 5 = 2  (끝자리 7, 617 mod 5 = 2)
  (r-1) mod 5 = 1
  → 5 ∤ (r-1)
  → gcd(5, r-1) = 1  ✓

α = 5가 가능한 가장 작은 홀수 지수!
```

> [!tip] α가 작을수록 좋은 이유
>
> | α | Fr 곱셈 수 | R1CS 제약 수/S-box | 순열? |
> |---|-----------|-------------------|-------|
> | 3 | 2 (square + mul) | 2 | **불가** (gcd=3) |
> | 5 | 3 (2 square + mul) | 3 | **가능** ✓ |
> | 7 | 4 | 4 | 가능 |
>
> α=3이 가장 저렴하지만 BN254에서는 불가.
> α=5가 **가능한 최소** → Poseidon의 표준 선택.

#### S-box 역함수의 존재

```
x → x^α 가 전단사이면, 역함수 x → x^(α^(-1)) 가 존재:

  α^(-1) mod (r-1) = ?
  5^(-1) mod (r-1) 계산 (확장 유클리드)

  5 · d ≡ 1 (mod r-1) 인 d가 존재
  (x^5)^d = x^(5d) = x^1 = x

이 역함수는 Poseidon 해시 자체에서는 불필요하지만,
Rescue 해시 (Poseidon의 변형)에서는 역 S-box를 사용한다.
```

---

### 각 라운드의 구조

```
하나의 라운드:
  1. AddRoundConstants — 대칭성 파괴
     state[i] += round_constant[r*T + i]

  2. S-box — 비선형성 (confusion)
     Full round:    state[i] = state[i]⁵  (모든 i)
     Partial round: state[0] = state[0]⁵  (첫 번째만)

  3. MDS mix — 확산 (diffusion)
     state = M · state  (행렬-벡터 곱)
```

> [!abstract] Confusion vs Diffusion (Shannon)
>
> | 성질 | 담당 | Poseidon에서 |
> |---|---|---|
> | **Confusion** (혼란) | 입력-출력 관계를 복잡하게 | S-box (x⁵) |
> | **Diffusion** (확산) | 한 비트 변화 → 전체 변화 | MDS 행렬 곱 |
>
> Full round은 confusion 최대 (모든 원소 S-box), Partial round은 효율성 (하나만 S-box).
> RP가 RF보다 많은 이유: partial round이 훨씬 저렴하지만, MDS mix 덕에 확산은 유지.

---

### S-box: x → x⁵

```rust
// poseidon.rs
/// S-box: x → x⁵
/// Fp 곱셈 횟수: square 2번 + mul 1번 = 3회
pub fn sbox(x: Fr) -> Fr {
    let x2 = x.square();
    let x4 = x2.square();
    x4 * x // x⁴ · x = x⁵
}
```

```
x⁵를 효율적으로 계산 (addition chain):

  x → x² → x⁴ → x⁴·x = x⁵

  square: 2번
  mul:    1번
  합계:   3번의 Fr 곱셈

비교: 나이브하게 x·x·x·x·x = 곱셈 4번
```

> [!tip] 회로에서의 S-box 비용
> R1CS에서 `x⁵`는:
> ```
> t₁ = x · x        ← 제약 1개 (곱셈 게이트)
> t₂ = t₁ · t₁       ← 제약 1개
> y  = t₂ · x        ← 제약 1개
> ```
> 총 **3개 제약**. SHA-256의 비트 연산 하나가 ~32개 제약인 것과 비교하면 극적으로 저렴.
>
> Poseidon 전체 R1CS 비용:
> ```
> Full round:    T=3개 S-box × 3 제약 = 9 제약/라운드
> Partial round: 1개 S-box × 3 제약 = 3 제약/라운드
> MDS mix:       덧셈/곱셈이지만 상수 곱이므로 ≈ 0 추가 제약
>
> 총: 8×9 + 57×3 = 72 + 171 = 243 제약  ← ~250 확인!
> ```

---

### 수학적 기초 II: MDS 행렬 — 왜 "모든 부분행렬의 det ≠ 0"인가

#### MDS의 코딩 이론적 기원

```
MDS = Maximum Distance Separable

코딩 이론에서: [n, k, d] 코드에서 Singleton bound는 d ≤ n-k+1.
MDS 코드는 이 한계를 달성: d = n-k+1 (최대 거리).

이것이 해시와 무슨 관계?

MDS 행렬로 상태를 혼합하면:
  → 입력 T개 중 어떤 t개(1 ≤ t ≤ T)가 바뀌어도
  → 출력이 최소 T-t+1개 바뀜
  → branch number = T+1 = 4 (t=3일 때)

이것이 "최대 확산"의 정확한 의미:
  1개 입력 변화 → 최소 3개 출력 변화 (전부!)
```

> [!important] Branch Number와 보안
> ```
> Branch number B = min(활성 S-box 수)
>
> B가 클수록:
>   → 차분 공격(differential attack)에서 활성 S-box가 많아짐
>   → 각 활성 S-box가 확률을 줄이므로, 공격 확률 기하급수적 감소
>   → 적은 라운드로도 높은 보안
>
> MDS 행렬의 B = T+1 = 4 (t=3일 때)
> 이것이 달성 가능한 최대값!
> ```

#### MDS 행렬 선택과 검증

```rust
/// MDS 행렬: [[2,1,1],[1,2,1],[1,1,2]]
fn generate_mds() -> [[Fr; T]; T] {
    let one = Fr::from_u64(1);
    let two = Fr::from_u64(2);
    [
        [two, one, one],
        [one, two, one],
        [one, one, two],
    ]
}
```

```
MDS 조건 검증: 모든 정방 부분행렬의 det ≠ 0

1×1 부분행렬 (9개):
  {2, 1, 1, 1, 2, 1, 1, 1, 2} → 모두 ≠ 0 ✓

2×2 부분행렬 (C(3,2)² = 9개):
  [[2,1],[1,2]] → det = 4-1 = 3 ✓
  [[2,1],[1,1]] → det = 2-1 = 1 ✓
  [[1,1],[2,1]] → det = 1-2 = -1 ✓
  [[2,1],[1,2]] → det = 3 ✓
  [[1,1],[1,2]] → det = 2-1 = 1 ✓
  [[2,1],[1,1]] → det = 1 ✓
  [[1,2],[1,1]] → det = -1 ✓
  [[1,1],[1,2]] → det = 1 ✓
  [[2,1],[1,2]] → det = 3 ✓
  모두 ≠ 0 ✓ (Fr에서도 당연히 비영 — r은 254-bit 소수)

3×3 (전체):
  det [[2,1,1],[1,2,1],[1,1,2]]
    = 2(4-1) - 1(2-1) + 1(1-2)
    = 6 - 1 - 1 = 4 ≠ 0 ✓
```

> [!note] 왜 이 행렬을 선택했는가?
> 가능한 선택지:
>
> | 방법 | 장점 | 단점 |
> |------|------|------|
> | 랜덤 + 검증 | 유연 | 증명 필요 |
> | **Cauchy matrix** | 이론적 보장 | 역원 계산 필요 |
> | **대칭 circulant** | 단순, 직관적 | 행렬 크기 제한 |
>
> 우리 선택: `I + J` (단위행렬 + 전1행렬)
> ```
> [[2,1,1],[1,2,1],[1,1,2]] = I₃ + J₃
>
> I₃ = [[1,0,0],[0,1,0],[0,0,1]]  (항등)
> J₃ = [[1,1,1],[1,1,1],[1,1,1]]  (전1)
> ```
> 대각 성분이 2, 나머지가 1 → 자신의 가중치가 높으면서도 모든 입력을 혼합.

#### MDS 행렬 곱 연산

```rust
/// MDS 행렬 곱: result = M · state
fn mds_mix(state: &[Fr; T], mds: &[[Fr; T]; T]) -> [Fr; T] {
    let mut result = [Fr::ZERO; T];
    for i in 0..T {
        for j in 0..T {
            result[i] = result[i] + mds[i][j] * state[j];
        }
    }
    result
}
```

```
수동 전개: M · [a, b, c]

  result[0] = 2a + 1b + 1c = 2a + b + c
  result[1] = 1a + 2b + 1c = a + 2b + c
  result[2] = 1a + 1b + 2c = a + b + 2c

각 출력이 모든 입력의 함수 → a만 바뀌어도 세 출력 모두 변경.

Fr 곱셈 횟수: T² = 9 (상수 곱은 최적화 가능하지만 교육용으로 일반 형태 유지)
```

---

### 수학적 기초 III: 라운드 수의 보안 근거

#### Poseidon이 방어하는 공격들

```
1. 대수적 공격 (Algebraic Attack)
   → Poseidon을 다항식으로 표현하여 해를 찾는 공격
   → 방어: S-box의 차수가 높아져서 다항식이 복잡해짐

2. 보간 공격 (Interpolation Attack)
   → 입출력 쌍으로 Poseidon을 다항식으로 복원
   → 방어: full round 수 RF

3. 차분 공격 (Differential Attack)
   → 입력 차분이 출력 차분으로 전파되는 패턴 추적
   → 방어: MDS의 높은 branch number + partial round 수 RP

4. 선형 공격 (Linear Attack)
   → S-box의 선형 근사를 이용
   → 방어: x⁵의 비선형성 + 충분한 라운드 수

5. Gröbner basis 공격
   → 해시를 다항식 시스템으로 모델링하고 Gröbner basis로 풀기
   → 방어: full round 수 RF
```

#### RF = 8의 근거: 보간 공격 방어

```
보간 공격 (Lagrange interpolation):
  R라운드 Poseidon을 하나의 다항식으로 표현하면 차수 = α^R

  t=3, α=5일 때:
  R라운드 후 차수 = 5^R

  공격 성공 조건: 5^R < r (체의 크기)
    → R < log_5(r) ≈ log_5(2^254) ≈ 254/log_2(5) ≈ 254/2.32 ≈ 109

  하지만! full round에서만 전체 S-box가 적용되므로,
  필요한 full round 수: RF ≥ 2 × ceil(log_α(2) × security_bits / T)

  128-bit 보안, T=3, α=5:
    RF ≥ 2 × ceil(128 × log_5(2) / 3)
       ≈ 2 × ceil(128 × 0.431 / 3)
       ≈ 2 × ceil(18.4)
       = 2 × 19...

  실제로는 더 정교한 분석으로 RF=8이면 충분.
  Poseidon 논문의 안전 마진 포함.
```

> [!important] RF의 보수적 선택
> 이론적 최소 RF보다 **안전 마진 +2**를 추가.
> RF = 6이면 이론적으로 충분하지만, RF = 8로 설정하여 안전 마진 확보.
> 이것이 암호학에서의 관례 — "충분하다고 증명된 것보다 약간 더".

#### RP = 57의 근거: 차분 공격 방어

```
차분 공격에서 partial round은 MDS의 확산으로 방어:

한 라운드의 partial S-box가 만드는 비선형성:
  → state[0]만 x⁵ 적용
  → MDS mix로 전체에 확산
  → 다음 라운드에서 state[0]에 다시 비선형성

RP 라운드 후 총 확산:
  MDS의 branch number B = T+1 = 4
  → RP 라운드마다 활성 S-box가 최소 1개
  → 총 활성 S-box ≥ RP/T (대략)

128-bit 보안을 위한 활성 S-box 수:
  5^(-a) < 2^(-128)  (각 S-box의 최대 차분 확률)
  a > 128/log_2(5) ≈ 55

→ RP ≥ 55 필요, 안전 마진 +2로 RP = 57
```

> [!abstract] Full vs Partial: 비용 분석
>
> | 라운드 종류 | S-box 횟수 | 비용 비율 | 역할 |
> |------------|-----------|----------|------|
> | Full round | T=3 | 높음 | 대수적/보간 공격 방어 |
> | Partial round | 1 | **1/T = 33%** | 차분/통계적 공격 방어 |
>
> 전부 full round면: 65 × 3 = 195 S-box
> 실제 (HADES 구조): 8 × 3 + 57 × 1 = 24 + 57 = 81 S-box
>
> **58% 절감** — 보안은 유지하면서 비용을 거의 절반으로!
>
> 이 구조를 **HADES design strategy**라 부른다:
> "외곽은 강하게(full), 내부는 효율적으로(partial)"

---

### 라운드 상수: 결정론적 생성

```rust
/// 시드에서 시작하여 반복적으로 S-box 적용:
///   state_{i+1} = (state_i + i + 1)⁵
fn generate_round_constants() -> Vec<Fr> {
    let count = T * NUM_ROUNDS; // 3 × 65 = 195
    let mut constants = Vec::with_capacity(count);
    let mut state = Fr::from_u64(0);
    for i in 0..count {
        state = state + Fr::from_u64(i as u64 + 1);
        state = sbox(state); // x⁵
        constants.push(state);
    }
    constants
}
```

> [!note] 라운드 상수의 역할
> 상수가 없으면 초기 라운드에서 **대칭적** 입력이 대칭적 출력을 만들 수 있다.
> 각 라운드에 서로 다른 상수를 더해서 대칭성을 파괴.
>
> **Nothing-up-my-sleeve**: 상수 생성이 결정론적이므로 "백도어를 숨겼다"는 의심을 방지.
> 표준 Poseidon은 Grain LFSR로 생성하지만, 우리 구현은 교육용으로 단순한 반복 사용.

> [!question] 표준 구현과의 차이
> Poseidon 논문의 참조 구현은 **Grain LFSR**(선형 피드백 시프트 레지스터)로 상수를 생성.
> 우리 구현은 `(state + counter)⁵`의 반복 — 같은 목적(결정론적, 의사난수적)이지만 다른 방법.
>
> 결과적으로 우리 해시 값은 표준 Poseidon과 **다른 값을 출력**한다.
> 하지만 보안 성질(pre-image resistance, collision resistance)은 동일하게 보장된다 —
> 보안은 **구조**(SPN, 라운드 수, MDS)에서 오지, 특정 상수 값에서 오지 않기 때문.

---

### PoseidonParams 구조체

```rust
pub struct PoseidonParams {
    pub round_constants: Vec<Fr>,   // T × NUM_ROUNDS = 195개
    pub mds: [[Fr; T]; T],          // T × T MDS 행렬
}

impl PoseidonParams {
    /// BN254 Fr 위의 Poseidon 파라미터 생성
    pub fn new() -> Self {
        PoseidonParams {
            round_constants: generate_round_constants(),
            mds: generate_mds(),
        }
    }
}
```

---

### Poseidon 순열 (Permutation)

```rust
pub fn poseidon_permutation(state: &mut [Fr; T], params: &PoseidonParams) {
    let half_rf = RF / 2; // 4

    // Phase 1: RF/2 = 4 full rounds
    for r in 0..half_rf {
        let offset = r * T;
        for i in 0..T {
            state[i] = state[i] + params.round_constants[offset + i];
        }
        // S-box: ALL elements
        for i in 0..T {
            state[i] = sbox(state[i]);
        }
        *state = mds_mix(state, &params.mds);
    }

    // Phase 2: RP = 57 partial rounds
    for r in 0..RP {
        let offset = (half_rf + r) * T;
        for i in 0..T {
            state[i] = state[i] + params.round_constants[offset + i];
        }
        // S-box: ONLY first element
        state[0] = sbox(state[0]);
        *state = mds_mix(state, &params.mds);
    }

    // Phase 3: RF/2 = 4 full rounds
    for r in 0..half_rf {
        let offset = (half_rf + RP + r) * T;
        for i in 0..T {
            state[i] = state[i] + params.round_constants[offset + i];
        }
        // S-box: ALL elements
        for i in 0..T {
            state[i] = sbox(state[i]);
        }
        *state = mds_mix(state, &params.mds);
    }
}
```

```
순열 흐름:

  state ──┬── Full Round ×4 ──┬── Partial Round ×57 ──┬── Full Round ×4 ──┬── output
          │  (S-box 전체)      │  (S-box 첫째만)        │  (S-box 전체)      │
          │                    │                        │                    │
          │  confusion 최대    │  효율성 + 확산 유지     │  confusion 최대    │
```

> [!important] Full vs Partial의 보안 근거
>
> |라운드 종류|S-box 적용|비용|역할|
> |---|---|---|---|
> |Full round|T개 전부|높음|대수적 공격 방어|
> |Partial round|1개만|낮음|통계적 공격 방어 (MDS 확산으로)|
>
> RF/2를 앞뒤에 배치하는 이유: "wide trail" 전략.
> 처음과 끝에서 강한 confusion을 적용하여 차분/선형 공격의 활성 S-box 수를 최대화.

---

### 수동 계산: 첫 번째 라운드 추적

```
입력: state = [0, 1, 2]  (capacity=0, left=1, right=2)

─── Round 0 (Full) ───

1. AddRoundConstants:
   c₀ = sbox(0 + 1) = 1⁵ = 1
   c₁ = sbox(1 + 2) = 3⁵ = 243
   c₂ = sbox(243 + 3) = 246⁵ = (매우 큰 수, Fr mod r)

   state[0] += c₀ → 0 + 1 = 1
   state[1] += c₁ → 1 + 243 = 244
   state[2] += c₂ → 2 + 246⁵ mod r = ...

2. S-box (Full: 모든 원소):
   state[0] = 1⁵ = 1
   state[1] = 244⁵ mod r = ...
   state[2] = ...

3. MDS mix:
   [s₀, s₁, s₂] = M · [state[0], state[1], state[2]]
   s₀ = 2·state[0] + state[1] + state[2]
   s₁ = state[0] + 2·state[1] + state[2]
   s₂ = state[0] + state[1] + 2·state[2]
```

> [!note] 254-bit 필드 위의 연산
> 첫 라운드의 2~3단계에서 이미 값이 254-bit 전체에 퍼진다.
> 이것이 Poseidon의 "체 크기 = 보안 파라미터"인 이유 —
> 비트 연산과 달리, 한 번의 체 곱셈으로 254비트 전체가 뒤섞인다.

---

### 2-to-1 해시 함수

```rust
/// 2-to-1 Poseidon 해시
pub fn poseidon_hash(left: Fr, right: Fr) -> Fr {
    let params = PoseidonParams::new();
    poseidon_hash_with_params(&params, left, right)
}

/// 파라미터를 외부에서 전달하는 버전 (반복 호출 시 효율적)
pub fn poseidon_hash_with_params(params: &PoseidonParams, left: Fr, right: Fr) -> Fr {
    // Sponge: [capacity=0, rate₀=left, rate₁=right]
    let mut state = [Fr::ZERO, left, right];
    poseidon_permutation(&mut state, params);
    state[1] // 첫 번째 rate 원소가 해시 출력
}
```

```
2-to-1 해시 과정:

  입력: left, right ∈ Fr

  1. Sponge 초기화
     state = [0, left, right]
             ↑   ↑      ↑
         capacity rate₀  rate₁

  2. Permutation 적용 (65 라운드)

  3. 출력: state[1] ∈ Fr
```

> [!tip] poseidon_hash vs poseidon_hash_with_params
> `poseidon_hash`는 매번 파라미터(195개 상수 + MDS)를 생성.
> Merkle tree처럼 반복 호출할 때는 `PoseidonParams::new()`를 한 번만 호출하고 `poseidon_hash_with_params`를 사용하면 효율적.
> ```rust
> let params = PoseidonParams::new();
> let h1 = poseidon_hash_with_params(&params, a, b);
> let h2 = poseidon_hash_with_params(&params, h1, c);
> ```

---

### ZK-friendly 해시의 계보

```
ZK-friendly 해시 함수 비교:

MiMC (2016):
  S-box: x³ 또는 x⁷
  구조: 단일 변수 반복
  문제: 높은 곱셈 깊이, 대수적 차수 낮음

Poseidon (2019):   ← 우리 구현
  S-box: x⁵
  구조: SPN (HADES strategy)
  장점: 낮은 R1CS 비용, 성숙한 보안 분석

Rescue (2019):
  S-box: x^α와 x^(α⁻¹) 교대 적용
  구조: Feistel-like
  장점: STARK-friendly (역 S-box로 AIR degree 감소)

Griffin (2022):
  S-box: x^α + 비선형 함수
  구조: 호너(Horner) 방식
  장점: Poseidon보다 약간 빠름

Neptune (2023):
  Poseidon의 t=4,8,12 최적화 버전
  장점: 큰 width에서 효율적
```

> [!tip] 왜 Poseidon인가?
> - **성숙도**: 2019년 이후 가장 많이 분석된 ZK 해시
> - **표준화**: Filecoin, Mina, Zcash 등 주요 프로젝트에서 사용
> - **단순성**: SPN + HADES가 구현과 분석 모두 간단
> - **회로 효율**: t=3에서 ~250 제약으로 가장 균형 잡힌 선택

---

### 다음 스텝과의 연결

```
Step 07: Poseidon 해시 (여기)
    ↓
Step 08: Merkle tree (Poseidon을 해시로 사용)
    ↓
Step 10: R1CS 가젯 (Poseidon을 회로로 재구현)

Merkle tree에서:
  parent = poseidon_hash(left_child, right_child)
  → 2-to-1 해시가 트리의 각 노드 계산에 사용

R1CS 가젯에서:
  native Poseidon과 circuit Poseidon의 결과가 동일해야 함
  → "ZK-friendly"의 의미가 체감되는 순간:
     같은 로직을 회로로 번역할 때 제약이 ~250개만 필요

전체 의존성:
  Fr (Step 03)
    └─ Poseidon (여기, Step 07)
         ├─ Merkle tree (Step 08)
         │    └─ Merkle 회로 (Step 10)
         └─ Poseidon 회로 (Step 10)
              └─ Mixer 회로 (Step 45) — 프라이버시 응용
```

---

### 테스트로 검증되는 성질

```rust
#[test]
fn sbox_basic() {
    assert_eq!(sbox(Fr::from_u64(3)), Fr::from_u64(243)); // 3⁵ = 243
}

#[test]
fn mds_full() {
    // M · [1, 2, 3] = [2+2+3, 1+4+3, 1+2+6] = [7, 8, 9]
    let mds = generate_mds();
    let state = [Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
    let result = mds_mix(&state, &mds);
    assert_eq!(result[0], Fr::from_u64(7));
}

#[test]
fn hash_determinism() {
    let a = Fr::from_u64(1);
    let b = Fr::from_u64(2);
    assert_eq!(poseidon_hash(a, b), poseidon_hash(a, b)); // 결정론적
}

#[test]
fn hash_order_matters() {
    // hash(1, 2) ≠ hash(2, 1) — 순서가 중요
    let h1 = poseidon_hash(Fr::from_u64(1), Fr::from_u64(2));
    let h2 = poseidon_hash(Fr::from_u64(2), Fr::from_u64(1));
    assert_ne!(h1, h2);
}

#[test]
fn hash_collision_resistance_right() {
    let h1 = poseidon_hash(Fr::from_u64(1), Fr::from_u64(2));
    let h2 = poseidon_hash(Fr::from_u64(1), Fr::from_u64(3));
    assert_ne!(h1, h2); // 충돌 저항
}

#[test]
fn hash_sensitivity() {
    // 입력을 1만 바꿔도 출력이 완전히 달라짐 (avalanche)
    let h1 = poseidon_hash(Fr::from_u64(100), Fr::from_u64(200));
    let h2 = poseidon_hash(Fr::from_u64(101), Fr::from_u64(200));
    assert_ne!(h1, h2);
}
```

---

### 테스트 결과

```
running 147 tests
field::fp  ... 35 passed
field::fr  ... 10 passed
field::fp2 ... 17 passed
field::fp6 ... 15 passed
field::fp12 ... 14 passed
curve::g1  ... 15 passed
curve::g2  ... 14 passed
pairing    ... 8 passed
hash::poseidon ... 19 passed  ← Step 07 (S-box 4 + MDS 3 + constants 3 + permutation 3 + hash 6)
```
