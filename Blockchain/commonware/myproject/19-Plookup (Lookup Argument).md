## Step 15: Plookup — 테이블 멤버십을 grand product로 증명하다

### 핵심 질문: "이 값이 테이블에 있는가?"를 어떻게 증명하나?

```
PLONK만으로 range check (0 ≤ x < 256):

  "x가 8비트인가?" → 비트 분해:
    x = b₇·128 + b₆·64 + b₅·32 + b₄·16 + b₃·8 + b₂·4 + b₁·2 + b₀
    각 bᵢ ∈ {0,1} → 8개 boolean gate
    결합 확인     → 추가 gate들

    총 ~16개 제약!

Plookup으로:

  테이블 T = {0, 1, 2, ..., 255}  (256개 엔트리)
  lookup: "x ∈ T?"  → 1개 제약

  16배 효율화!

더 극적인 예 — XOR 연산:
  PLONK만으로: 비트 분해 + 비트별 XOR + 결합 → ~32개 제약 (8비트)
  Plookup으로: XOR 테이블에서 한 번의 lookup → 1개 제약
```

> [!important] Plookup의 핵심 아이디어
> ```
> 문제: f = {f₀, f₁, ..., f_{m-1}} 의 모든 원소가 T에 있음을 증명
>
> 접근: "f ∪ T를 T의 순서로 정렬할 수 있다"는 것으로 환원
>
> 왜 정렬이 멤버십을 의미하는가?
>   T가 [0, 1, 2, 3]이고 f = [1, 2]이면:
>     sorted(f ∪ T) = [0, 1, 1, 2, 2, 3] ← T순서 유지하며 f 삽입 가능
>
>   T가 [0, 1, 2, 3]이고 f = [5]이면:
>     5는 T에 없으므로 T 순서로 정렬 불가능!
>
> 증명 방법: Step 14의 permutation grand product와 동일한 패턴
>   "곱의 텔레스코핑"으로 정렬의 올바름을 증명
> ```

---

### Part 1: Plookup 프로토콜 개요

```
Plookup [Gabizon-Williamson, 2020]:
  "plookup: A simplified polynomial protocol for lookup tables"

입력:
  t = (t₀, t₁, ..., t_{d-1})  — 테이블 (정렬됨, 공개)
  f = (f₀, f₁, ..., f_{n-1})  — 조회값 (witness)

주장: f ⊆ t  (f의 모든 원소가 t에 존재)

프로토콜 단계:
  1. 정렬된 병합: s = sort(f ∪ t)  (t의 순서를 유지)
  2. s를 중첩 분리: h1 = s[..d], h2 = s[d-1..]
     (h1의 마지막 원소 = h2의 첫 원소)
  3. 검증자가 랜덤 챌린지 β, γ를 선택
  4. Grand product Z(x)를 계산
  5. Z가 "닫히는지" (telescope to 1) 확인

시각화:
  ┌──────────────────────────────────────────────────┐
  │                                                    │
  │  Table T:  [0, 1, 2, 3]        (공개, 정렬됨)     │
  │  Lookup f: [1, 2]              (witness)          │
  │                                                    │
  │  ┌──────────────────────────────────────────┐     │
  │  │  sorted(f ∪ T) = [0, 1, 1, 2, 2, 3]     │     │
  │  │                                          │     │
  │  │  h1 = [0, 1, 1, 2]  (|T| = 4개)         │     │
  │  │  h2 = [2, 2, 3]     (|f|+1 = 3개)       │     │
  │  │       ↑ overlap                          │     │
  │  │  h1[last] = h2[0] = 2                   │     │
  │  └──────────────────────────────────────────┘     │
  │                                                    │
  │  Grand Product Z(x):                               │
  │    Z(ω⁰) = 1                                      │
  │    Z(ω^(i+1)) = Z(ωⁱ) · num(i) / den(i)          │
  │    검증: Z(ω^(N-1)) = 1 → f ⊆ T                  │
  │                                                    │
  └──────────────────────────────────────────────────┘
```

---

### Part 2: 정렬된 병합 — 왜 중첩(overlap)이 필요한가?

```
핵심 개념: "연속 쌍(consecutive pair)" 으로 정렬을 인코딩

정렬된 리스트 s의 핵심 성질:
  s의 모든 연속 쌍 (sⱼ, sⱼ₊₁)에 대해:
    sⱼ₊₁ = sⱼ  (같은 값 반복)  또는
    sⱼ₊₁ = t 에서 sⱼ의 다음 원소 (테이블 순서의 다음)

  이 성질이 성립하면 → s는 t의 순서로 올바르게 정렬됨
  이 성질이 성립하면 → f의 모든 원소가 t에 있음
```

#### 중첩이 없으면 왜 안 되는가?

```
s = [0, 1, 1, 2, 2, 3],  |T| = 4

중첩 없는 분리:
  h1 = s[..4] = [0, 1, 1, 2]
  h2 = s[4..] = [2, 3]

  h1의 연속 쌍: (0,1), (1,1), (1,2)
  h2의 연속 쌍: (2,3)

  합집합: {(0,1), (1,1), (1,2), (2,3)} ← 4개

  s의 전체 연속 쌍: (0,1), (1,1), (1,2), (2,2), (2,3) ← 5개
                                          ↑
                                    누락! (h1의 끝과 h2의 시작 사이)

  이 빠진 쌍 때문에 grand product가 닫히지 않음!

중첩 분리:
  h1 = s[..4] = [0, 1, 1, 2]
  h2 = s[3..] = [2, 2, 3]      ← h2[0] = h1[3] = 2 (중첩!)

  h1의 연속 쌍: (0,1), (1,1), (1,2)         ← 3개
  h2의 연속 쌍: (2,2), (2,3)                 ← 2개

  합집합: {(0,1), (1,1), (1,2), (2,2), (2,3)} ← 5개 = s의 전체!

  중첩이 "끊어진 체인"을 연결해줌 → grand product 닫힘 ✓
```

#### 형식적 증명: 중첩 분리 ↔ 연속 쌍 완전 커버

```
정리: h1 = s[0..k], h2 = s[k-1..m] (중첩 at s[k-1])이면
      h1과 h2의 연속 쌍 합집합 = s의 전체 연속 쌍

증명:
  s의 연속 쌍: (s[0],s[1]), (s[1],s[2]), ..., (s[m-2],s[m-1])
    = {(s[j], s[j+1]) : j = 0, ..., m-2}  (총 m-1개)

  h1의 연속 쌍: (s[0],s[1]), ..., (s[k-2],s[k-1])
    = {(s[j], s[j+1]) : j = 0, ..., k-2}  (총 k-1개)

  h2의 연속 쌍: (s[k-1],s[k]), ..., (s[m-2],s[m-1])
    = {(s[j], s[j+1]) : j = k-1, ..., m-2}  (총 m-k개)

  합집합: j = 0, ..., k-2  ∪  j = k-1, ..., m-2
        = j = 0, ..., m-2  ← s의 전체 연속 쌍 ✓

  크기: (k-1) + (m-k) = m-1  ✓ (중복 없음)

  핵심: j = k-1 가 h2에 의해 커버됨 → 체인이 이어짐
```

---

### Part 3: Grand Product 공식 유도

#### 멀티셋 등식에서 Grand Product로

```
우리가 증명하고 싶은 것:
  {f} ∪ {t} = {s}  (멀티셋으로 같음)
  AND s는 t의 순서로 정렬됨

이 두 조건을 하나의 다항식 등식으로 인코딩:

  ∏ᵢ (1+β)(γ+fᵢ) · ∏ⱼ (γ(1+β) + tⱼ + β·tⱼ₊₁)
  ──────────────────────────────────────────────── = 1
  ∏ₖ (γ(1+β) + sₖ + β·sₖ₊₁)

왜 이것이 맞는가? 두 부분으로 나누어 보자:

  (A) 개별 원소의 멀티셋 등식:
      ∏ᵢ(γ+fᵢ) · ∏ⱼ(γ+tⱼ) = ∏ₖ(γ+sₖ)

      이것은 {f} ∪ {t} = {s} 를 인코딩
      (γ가 랜덤이므로 Schwartz-Zippel에 의해 건전)

  (B) 연속 쌍의 멀티셋 등식:
      ∏ⱼ(γ+tⱼ + β·tⱼ₊₁) = ∏ₖ(γ+sₖ + β·sₖ₊₁)

      이것은 s의 연속 쌍이 t의 연속 쌍과 호환됨을 인코딩
      (β, γ가 랜덤이므로 2변수 Schwartz-Zippel에 의해 건전)

  (A)×(B)를 결합하면 위의 공식이 됨:
      (1+β) 인자는 (A)를 (B)의 형태로 변환하기 위함

      (γ+fᵢ) = (γ+fᵢ)·(1+β)/(1+β)
      → 분자에 (1+β)를 곱하면 (γ(1+β) + fᵢ(1+β)) 형태가 되어
        연속 쌍 인코딩 (γ(1+β) + a + β·b)과 같은 구조가 됨
```

#### (1+β) 인자의 의미

```
왜 (1+β)가 필요한가? — 두 인코딩을 호환시키기 위함

개별 원소 인코딩: (γ + fᵢ)
연속 쌍 인코딩:   (γ(1+β) + a + β·b) = (1+β)γ + a + βb

특수한 경우 a = b (같은 값 반복):
  (γ(1+β) + a + β·a) = (1+β)(γ + a)

따라서:
  (1+β)(γ + fᵢ)는 "fᵢ가 자기 자신과 쌍을 이루는" 연속 쌍 인코딩
  → 개별 원소를 연속 쌍 공간으로 끌어올림 (lifting)

이것이 분자에서 f 부분에 (1+β)를 곱하는 이유:
  분자 = ∏(1+β)(γ+fᵢ) · ∏(γ(1+β)+tⱼ+β·tⱼ₊₁)
       = ∏[(1+β)(γ+fᵢ)] · ∏[연속 쌍(tⱼ,tⱼ₊₁)]
```

#### 인자 수 맞추기 (factor count balance)

```
도메인 크기 N에서의 인자 수:

  확장 후: |f_ext| = N-1,  |t_ext| = N
           |s| = (N-1) + N = 2N-1

  h1 = s[..N]    (N개)    →  N-1개 연속 쌍
  h2 = s[N-1..]  (N개)    →  N-1개 연속 쌍

  분자 인자 수: (N-1) [f에서] + (N-1) [t 연속 쌍] = 2(N-1)
  분모 인자 수: (N-1) [h1 연속 쌍] + (N-1) [h2 연속 쌍] = 2(N-1)

  2(N-1) = 2(N-1)  ✓  인자 수 일치!

  이것이 f를 N-1개, t를 N개로 확장하는 이유:
    N-1 + (N-1) = 2(N-1)  (분자)
    (N-1) + (N-1) = 2(N-1)  (분모)
```

---

### Part 4: Grand Product 공식 — 완전한 형태

```
Z(ω⁰) = 1

Z(ω^(i+1)) = Z(ωⁱ) ·
  (1+β) · (γ + fᵢ) · (γ(1+β) + tᵢ + β·tᵢ₊₁)
  ─────────────────────────────────────────────────
  (γ(1+β) + h1ᵢ + β·h1ᵢ₊₁) · (γ(1+β) + h2ᵢ + β·h2ᵢ₊₁)

for i = 0, 1, ..., N-2  (총 N-1 스텝)

검증: Z(ω^(N-1)) = ∏_{i=0}^{N-2} ratio(i) = 1
```

#### 각 항의 의미

```
분자:
  (1+β)(γ+fᵢ)             — f의 i번째 원소를 연속 쌍 공간으로 lifting
  (γ(1+β)+tᵢ+β·tᵢ₊₁)     — t의 i번째 연속 쌍 (tᵢ, tᵢ₊₁) 인코딩

분모:
  (γ(1+β)+h1ᵢ+β·h1ᵢ₊₁)   — h1의 i번째 연속 쌍 인코딩
  (γ(1+β)+h2ᵢ+β·h2ᵢ₊₁)   — h2의 i번째 연속 쌍 인코딩

텔레스코핑 원리:
  분자의 모든 인자를 곱한 멀티셋 = 분모의 모든 인자를 곱한 멀티셋
  → 곱이 1로 돌아옴

  이는 Step 14의 permutation grand product와 정확히 동일한 원리:
    "두 멀티셋이 같으면 곱이 1"
```

---

### Part 5: 수치 트레이스 — T=[0,1,2,3], f=[1,2]

```
── 기본 데이터 ──────────────────────────────────────

T = [0, 1, 2, 3],  f = [1, 2]

N = max(|T|, |f|+1) = max(4, 3) = 4  (이미 2의 거듭제곱)

확장:
  t_ext = [0, 1, 2, 3]     (이미 N=4, 확장 불필요)
  f_ext = [1, 2, 3]         (N-1=3개로 확장, t_last=3 추가)

── sorted list 계산 ─────────────────────────────────

sorted(f_ext ∪ t_ext):
  t[0]=0: push 0                     → [0]
  t[1]=1: push 1, push f의 1         → [0, 1, 1]
  t[2]=2: push 2, push f의 2         → [0, 1, 1, 2, 2]
  t[3]=3: push 3, push f의 3         → [0, 1, 1, 2, 2, 3, 3]

  |sorted| = |f_ext| + |t_ext| = 3 + 4 = 7 = 2N-1  ✓

분리 (중첩):
  h1 = sorted[..4] = [0, 1, 1, 2]    (N = 4개)
  h2 = sorted[3..] = [2, 2, 3, 3]    (N = 4개)

  h1[3] = h2[0] = 2  ← 중첩 확인 ✓

── 연속 쌍 확인 ─────────────────────────────────────

t의 연속 쌍:   (0,1), (1,2), (2,3)     ← 3개
h1의 연속 쌍:  (0,1), (1,1), (1,2)     ← 3개
h2의 연속 쌍:  (2,2), (2,3), (3,3)     ← 3개

h1 ∪ h2 연속 쌍 = (0,1), (1,1), (1,2), (2,2), (2,3), (3,3) ← 6개

sorted 연속 쌍 = (0,1), (1,1), (1,2), (2,2), (2,3), (3,3) ← 6개  ✓

── grand product (β=7, γ=11) ────────────────────────

약속: A = γ(1+β) = 11·8 = 88,  f_ext에 더미 추가: f = [1,2,3,3]

Step i=0:
  num = (1+β)(γ+f[0]) · (A+t[0]+β·t[1])
      = 8·(11+1) · (88+0+7·1)
      = 8·12 · 95
      = 9120

  den = (A+h1[0]+β·h1[1]) · (A+h2[0]+β·h2[1])
      = (88+0+7·1) · (88+2+7·2)
      = 95 · 104
      = 9880

  ratio(0) = 9120/9880

Step i=1:
  num = 8·(11+2) · (88+1+7·2) = 8·13·103 = 10712
  den = (88+1+7·1)·(88+2+7·3) = 96·111 = 10656

  ratio(1) = 10712/10656

Step i=2:
  num = 8·(11+3) · (88+2+7·3) = 8·14·111 = 12432
  den = (88+1+7·2)·(88+3+7·3) = 103·112 = 11536

  ratio(2) = 12432/11536

── 곱이 1이 됨을 기호적으로 증명 ──────────────────────

A = γ(1+β) 으로 놓자.

분자 전체 = 8³ · (γ+1)(γ+2)(γ+3)
           · (A+β)(A+1+2β)(A+2+3β)

분모 전체 = (A+β)(A+1+β)(A+1+2β)
           · (A+2+2β)(A+2+3β)(A+3+3β)

공통 인자 소거: (A+β), (A+1+2β), (A+2+3β) → 양쪽에서 제거

남은 분자 = 8³ · (γ+1)(γ+2)(γ+3)

남은 분모 = (A+1+β) · (A+2+2β) · (A+3+3β)

핵심 관찰: A + k + kβ = γ(1+β) + k(1+β) = (1+β)(γ+k) = 8(γ+k)

따라서:
  (A+1+β) = 8(γ+1)
  (A+2+2β) = 8(γ+2)
  (A+3+3β) = 8(γ+3)

남은 분모 = 8(γ+1)·8(γ+2)·8(γ+3) = 8³(γ+1)(γ+2)(γ+3)

결론: 분자/분모 = 8³(γ+1)(γ+2)(γ+3) / 8³(γ+1)(γ+2)(γ+3) = 1  ✓
```

---

### Part 6: Schwartz-Zippel 건전성 분석

```
Plookup의 건전성: 부정직한 prover가 f ⊄ t 인데 검증을 통과할 확률?

Schwartz-Zippel 보조정리:
  다항식 p(x₁,...,xₖ) ≢ 0 의 차수가 d이고
  x₁,...,xₖ를 유한체 F에서 균일 랜덤으로 선택하면:
    Pr[p(x₁,...,xₖ) = 0] ≤ d/|F|

적용:
  Plookup의 grand product 등식을 다항식으로 보면:

  P(β, γ) = ∏ num(i) - ∏ den(i)

  이것은 β, γ에 대한 다항식으로:
    β의 최대 차수: N-1 (h 연속 쌍에서)
    γ의 최대 차수: N-1 (f 개별 원소에서)
    총 차수: ≤ 2(N-1)

  f ⊄ t 이면 P ≢ 0 이므로:
    Pr[P(β,γ) = 0] ≤ 2(N-1) / |Fr|

  BN254에서 |Fr| ≈ 2²⁵⁴:
    Pr ≤ 2N / 2²⁵⁴ ≈ 0  (무시 가능)

  즉, 부정직한 prover가 성공할 확률은 천문학적으로 작음.

직관:
  β, γ가 랜덤이면 분자와 분모의 "우연한 일치"는 거의 불가능.
  마치 두 다른 다항식이 랜덤 점에서 우연히 같은 값을 가지려면
  차수만큼의 확률이 필요한 것과 같음.
```

---

### Part 7: 확장(extension)이 필요한 이유

```
왜 사후 패딩이 안 되고 사전 확장이 필요한가?

── 잘못된 접근: 사후 패딩 ──────────────────────

1. sorted_list(f_raw, T_raw) 계산  → h1_raw, h2_raw
2. 각각을 domain_size N으로 패딩

문제: 패딩된 원소는 원래 sorted list에 없었음
  → h1, h2의 연속 쌍이 f ∪ T의 정렬을 반영하지 않음
  → 멀티셋 등식이 깨짐 → grand product ≠ 1

── 올바른 접근: 사전 확장 ──────────────────────

1. f를 N-1개로 확장 (더미값 = t_last)
2. T를 N개로 확장 (t_last 반복)
3. sorted_list(f_ext, T_ext) 계산 → h1, h2가 정확히 N개씩

이것이 맞는 이유:
  더미 lookup값 = t_last → 반드시 T에 있음 (유효한 lookup)
  확장된 f_ext도 T_ext의 부분집합 → Plookup 조건 성립
  sorted list가 2N-1개 → 중첩 분리하면 h1, h2 각 N개

수치 예시:
  T = [0,1,2,3], f = [1,2], N = 4

  잘못된 방법:
    sorted(f, T) = [0, 1, 1, 2, 2, 3]  (6개)
    h1 = [0,1,1,2], h2 = [2,2,3]  (3개)
    h2를 [2,2,3,3]으로 패딩 → (2,3)→(3,3) 전이가 실제와 불일치

  올바른 방법:
    f_ext = [1,2,3], t_ext = [0,1,2,3]
    sorted(f_ext, t_ext) = [0,1,1,2,2,3,3]  (7 = 2·4-1)
    h1 = [0,1,1,2], h2 = [2,2,3,3]  (각 4개, 정확!)
```

---

### Part 8: 다중 컬럼 테이블 인코딩

```
XOR 테이블은 3개 컬럼: (a, b, a⊕b)
이것을 어떻게 단일 Fr 값으로 인코딩하는가?

── 랜덤 선형 결합 ─────────────────────────────

encoded = a + α·b + α²·c

여기서 α = 2^bits (비트 수에 따른 구분자)

왜 α = 2^bits 인가?
  각 컬럼 값이 [0, 2^bits - 1] 범위이면:
    a:    0 ~ 2^bits - 1          (α⁰ 자리)
    α·b:  0 ~ α·(2^bits - 1)     (α¹ 자리)
    α²·c: 0 ~ α²·(2^bits - 1)    (α² 자리)

  α = 2^bits이면 각 항의 범위가 겹치지 않음
  → 유일한 디코딩 보장 (충돌 없음)

── 수치 예시: 2비트 XOR ────────────────────────

bits = 2,  α = 2² = 4,  α² = 16
a, b ∈ {0, 1, 2, 3}

(a=2, b=3, c=2⊕3=1):
  encoded = 2 + 4·3 + 16·1 = 2 + 12 + 16 = 30

(a=0, b=0, c=0⊕0=0):
  encoded = 0 + 0 + 0 = 0

(a=3, b=3, c=3⊕3=0):
  encoded = 3 + 4·3 + 16·0 = 3 + 12 = 15

역변환 (α=4):
  30 = 2 + 4·3 + 16·1
  a = 30 mod 4 = 2
  b = (30 / 4) mod 4 = 7 mod 4 = 3
  c = 30 / 16 = 1

전체 2비트 XOR 테이블 (16개 엔트리):
  ┌───┬───┬───────┬─────────────────┐
  │ a │ b │ a⊕b  │ encoded         │
  ├───┼───┼───────┼─────────────────┤
  │ 0 │ 0 │  0   │ 0+0+0 = 0      │
  │ 0 │ 1 │  1   │ 0+4+16 = 20    │
  │ 0 │ 2 │  2   │ 0+8+32 = 40    │
  │ 0 │ 3 │  3   │ 0+12+48 = 60   │
  │ 1 │ 0 │  1   │ 1+0+16 = 17    │
  │ 1 │ 1 │  0   │ 1+4+0 = 5      │
  │ 1 │ 2 │  3   │ 1+8+48 = 57    │
  │ 1 │ 3 │  2   │ 1+12+32 = 45   │
  │ 2 │ 0 │  2   │ 2+0+32 = 34    │
  │ 2 │ 1 │  3   │ 2+4+48 = 54    │
  │ 2 │ 2 │  0   │ 2+8+0 = 10     │
  │ 2 │ 3 │  1   │ 2+12+16 = 30   │
  │ 3 │ 0 │  3   │ 3+0+48 = 51    │
  │ 3 │ 1 │  2   │ 3+4+32 = 39    │
  │ 3 │ 2 │  1   │ 3+8+16 = 27    │
  │ 3 │ 3 │  0   │ 3+12+0 = 15    │
  └───┴───┴───────┴─────────────────┘
```

---

### Part 9: PLONK 제약 시스템과의 통합

```
PlonkConstraintSystem에 추가된 것:

  struct PlonkConstraintSystem {
      // ... 기존 필드 (variables, gates, copies) ...

      lookup_tables: Vec<Vec<Fr>>,               // 등록된 테이블들
      lookup_entries: Vec<(usize, Column, usize)>,  // (row, column, table_id)
  }

── 테이블 등록 ──────────────────────────────────

  register_table(values: Vec<Fr>) -> usize
    → 테이블을 등록하고 ID 반환

── Lookup 제약 추가 ────────────────────────────

  add_lookup(row: usize, column: Column, table_id: usize)
    → "이 행의 이 컬럼 wire 값이 테이블에 있어야 함"

── q_lookup selector ──────────────────────────

  SelectorPolynomials에 q_lookup 추가:
    q_lookup(ωⁱ) = 1   (i번째 행에 lookup 제약이 있으면)
    q_lookup(ωⁱ) = 0   (없으면)

── 사용 예시: 8비트 range check ───────────────

  let mut cs = PlonkConstraintSystem::new();

  // 테이블 등록
  let table = LookupTable::range_table(8);  // {0..255}
  let tid = cs.register_table(table.values);

  // 변수 할당 + 게이트
  let x = cs.alloc_variable(Fr::from_u64(42));
  let dummy = cs.alloc_variable(Fr::ZERO);
  cs.add_gate(PlonkGate::dummy_gate(), x, dummy, dummy);

  // lookup 제약: gate 0의 wire A가 range table에 있어야 함
  cs.add_lookup(0, Column::A, tid);

  assert!(cs.are_lookups_satisfied());  // 42 ∈ {0..255} ✓
```

---

### Part 10: 전체 파이프라인 — compute_plookup

```
compute_plookup(cs, β, γ) → PlookupProof 의 전체 흐름:

  ┌─────────────────────────────────────────────────────┐
  │  1. CS에서 lookup 값 추출                             │
  │     f_raw = [wire 값들]                               │
  │                                                       │
  │  2. 도메인 크기 결정                                   │
  │     N = next_power_of_two(max(|T|, |f|+1))           │
  │                                                       │
  │  3. 확장                                              │
  │     t_ext: |T| → N  (t_last 반복)                    │
  │     f_ext: |f| → N-1  (더미 = t_last)                │
  │                                                       │
  │  4. sorted list (확장된 데이터로!)                      │
  │     (h1, h2) = compute_sorted_list(f_ext, t_ext)     │
  │     |h1| = N, |h2| = N  (정확)                       │
  │                                                       │
  │  5. f를 N개로 확장 (마지막 원소 미사용)                 │
  │     f_ext.push(t_last)                               │
  │                                                       │
  │  6. Lagrange 보간 → 다항식                             │
  │     f_poly, t_poly, h1_poly, h2_poly                 │
  │                                                       │
  │  7. Grand product Z_lookup 계산                       │
  │     N-1 스텝, Z[0]=1, ... , Z[N-1]=1                 │
  │                                                       │
  │  → PlookupProof { f_poly, t_poly, h1_poly, h2_poly,  │
  │                   z_poly }                            │
  └─────────────────────────────────────────────────────┘

PlookupProof의 다항식들은 Step 16에서 KZG로 commit됨.
```

---

### Part 11: permutation grand product와의 비교

```
┌──────────────────┬─────────────────────┬─────────────────────┐
│                  │ Permutation (Step 14) │ Plookup (Step 15)  │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 증명 대상         │ 와이어 값이 일관적     │ 조회값이 테이블에 있음 │
│                  │ (copy constraint)    │ (멤버십)            │
│                  │                     │                     │
│ 핵심 기법         │ grand product Z(x)   │ grand product Z(x)  │
│                  │                     │                     │
│ 분자              │ (wᵢ+β·id(i)+γ)      │ (1+β)(γ+fᵢ)        │
│                  │                     │ ·(A+tᵢ+β·tᵢ₊₁)     │
│                  │                     │                     │
│ 분모              │ (wᵢ+β·σ(i)+γ)       │ (A+h1ᵢ+β·h1ᵢ₊₁)   │
│                  │                     │ ·(A+h2ᵢ+β·h2ᵢ₊₁)   │
│                  │                     │                     │
│ 닫힘 조건         │ Z(ω^N) = Z(ω⁰) = 1 │ Z(ω^(N-1)) = 1     │
│                  │                     │                     │
│ 핵심 데이터       │ σ 순열               │ sorted list h1, h2  │
│                  │                     │                     │
│ 추가 다항식       │ σ_a, σ_b, σ_c       │ f, t, h1, h2       │
│                  │                     │                     │
│ 랜덤 챌린지       │ β, γ                │ β, γ                │
│                  │                     │                     │
│ 건전성           │ Schwartz-Zippel      │ Schwartz-Zippel     │
│                  │ d/|Fr|              │ 2d/|Fr|             │
└──────────────────┴─────────────────────┴─────────────────────┘

공통 패턴:
  1. 두 멀티셋이 같으면 ∏(encoding) / ∏(encoding) = 1
  2. Z(x)를 Lagrange 보간 → 다항식으로 commit
  3. 랜덤 챌린지로 건전성 확보 (Schwartz-Zippel)
```

---

### Part 12: sorted list 구축 알고리즘

```
입력: f = [f₀, ..., f_{m-1}],  T = [t₀, ..., t_{d-1}] (정렬됨)
출력: (h1, h2) — sorted list의 중첩 분리

알고리즘:

  Step 1: 빈도 맵 구축
    freq = {}
    for fᵢ in f:
      if fᵢ ∉ T: return Error
      freq[fᵢ] += 1

  Step 2: T 순서로 sorted list 구축
    sorted = []
    for tⱼ in T:
      sorted.push(tⱼ)              // 테이블 원소
      if tⱼ in freq:
        repeat freq[tⱼ] times:
          sorted.push(tⱼ)          // f에서 같은 값 삽입
        freq[tⱼ] = 0               // consumed

  Step 3: 중첩 분리
    h1 = sorted[..d]
    h2 = sorted[d-1..]

── 트레이스 예시: T=[0,1,2], f=[1,1,1] ──────────

  빈도 맵: {1: 3}

  정렬:
    t[0]=0: push 0                    → [0]
    t[1]=1: push 1, push 1,1,1       → [0, 1, 1, 1, 1]
    t[2]=2: push 2                    → [0, 1, 1, 1, 1, 2]

  분리 (d=3):
    h1 = sorted[..3] = [0, 1, 1]
    h2 = sorted[2..] = [1, 1, 1, 2]

  h1[2] = h2[0] = 1 (중첩) ✓
  |h1| = 3 = |T| ✓
  |h2| = 4 = |f| + 1 ✓
```

---

### Part 13: LookupTable 구현

```
── 데이터 구조 ──────────────────────────────────

pub struct LookupTable {
    pub values: Vec<Fr>,   // 정렬된 순서
}

── 생성자들 ─────────────────────────────────────

range_table(n: u32) → LookupTable
  → {0, 1, 2, ..., 2^n - 1}
  용도: n비트 range check

  예: range_table(4) = {0, 1, ..., 15}
      → "이 값이 4비트인가?" 확인

  예: range_table(8) = {0, 1, ..., 255}
      → "이 값이 1바이트인가?" 확인

xor_table(bits: u32) → LookupTable
  → {a + α·b + α²·(a⊕b) : a,b ∈ [0, 2^bits)}
  α = 2^bits

  예: xor_table(2) → 16개 엔트리 (2²·2² = 16)
  예: xor_table(4) → 256개 엔트리 (4⁴ = 256... 아니, 16·16)
  예: xor_table(8) → 65536개 엔트리

and_table(bits: u32) → LookupTable
  → {a + α·b + α²·(a&b) : a,b ∈ [0, 2^bits)}
  동일한 인코딩, 연산만 AND

── Fr 비교 (fr_cmp) ─────────────────────────────

Fr은 Ord trait을 구현하지 않음 (Montgomery 형태)
→ to_repr()로 canonical representation 변환 후 상위 limb부터 비교

  fn fr_cmp(a: &Fr, b: &Fr) -> Ordering {
      let ar = a.to_repr();  // [u64; 4]
      let br = b.to_repr();
      for i in (0..4).rev() {   // 상위 limb부터
          match ar[i].cmp(&br[i]) {
              Equal => continue,
              other => return other,
          }
      }
      Equal
  }
```

---

### Part 14: Plookup이 PLONK 성능을 혁신적으로 개선하는 실제 사례

```
── 사례 1: SHA-256 해시 ────────────────────────

SHA-256의 핵심 연산: 32비트 XOR, AND, 비트 회전

PLONK만으로:
  각 XOR/AND 연산:
    32비트 분해: 32 boolean gates + 32 결합 gates = ~64 제약
    XOR 계산: 32 비트별 XOR gates = 32 제약
    결과 결합: 32 gates
    총 ~128 제약 per XOR

  SHA-256 블록당 XOR/AND 수: ~700
  총 제약: ~90,000

Plookup으로:
  8비트 XOR 테이블 (256개 엔트리)
  32비트 XOR = 4개 바이트 XOR → 4 lookups
  총 제약: ~2,800  (32배 감소!)

── 사례 2: Merkle proof 검증 ──────────────────

Poseidon 해시를 사용한 Merkle 경로 검증
깊이 20인 트리 → 20번의 해시 호출

각 해시에서 S-box (x^5):
  PLONKish만으로: 2개 곱셈 게이트 (x²=t1, t1²·x=x⁵)
  Plookup 필요 없음 (이미 효율적)

하지만 range check가 필요한 경우:
  각 Merkle 레벨의 left/right 선택에 boolean check 필요
  PLONK: boolean gate (1 제약)
  Plookup: range_table(1) lookup (1 제약) — 비슷

Plookup의 진정한 가치:
  비트 연산이 많은 해시 함수 (SHA, BLAKE)에서 극적 효과
  산술 중심 해시 (Poseidon)에서는 효과가 작음

── 사례 3: 범용 비교 (a < b) ──────────────────

PLONK만으로 a < b (필드 원소 비교):
  1. a - b = d (차이 계산)
  2. d를 비트 분해 → n개 boolean gates
  3. 최상위 비트가 0인지 확인 (양수 판별)
  총: ~2n 제약 (n = 필드 비트 수)

Plookup으로:
  1. a - b = d
  2. d가 range_table(n) 안에 있는지 lookup
  총: 2 제약! (뺄셈 1개 + lookup 1개)
```

---

### Part 15: 테이블 크기 대 효율성 트레이드오프

```
Plookup의 비용:
  1. 테이블 T를 다항식으로 commit → O(|T|) 연산
  2. sorted list 계산 → O(|f| + |T|) 연산
  3. grand product → O(N) 연산 (N = domain size)
  4. KZG commit + open → O(N log N) (FFT 기반)

테이블 크기에 따른 트레이드오프:

  ┌────────────────┬──────────┬──────────┬──────────────────┐
  │ 테이블          │ 크기      │ 도메인 N  │ 용도             │
  ├────────────────┼──────────┼──────────┼──────────────────┤
  │ range_table(4)  │ 16       │ 16       │ 4비트 range      │
  │ range_table(8)  │ 256      │ 256      │ 바이트 range     │
  │ range_table(16) │ 65536    │ 65536    │ 16비트 range     │
  │ xor_table(4)    │ 256      │ 256      │ 4비트 XOR        │
  │ xor_table(8)    │ 65536    │ 65536    │ 바이트 XOR       │
  │ and_table(4)    │ 256      │ 256      │ 4비트 AND        │
  └────────────────┴──────────┴──────────┴──────────────────┘

핵심 원칙:
  - 테이블이 클수록 lookup 1개로 더 많은 비트를 처리
  - 하지만 도메인 크기도 커져서 증명 시간 증가
  - 실용적 최적점: 8~16비트 테이블

프로덕션 시스템의 전략:
  - 8비트 테이블을 사용하고 큰 연산을 바이트 단위로 분해
  - 예: 32비트 XOR = 4개 8비트 XOR lookup
  - 테이블 크기 256 × 4 lookups < 65536 테이블 × 1 lookup
```

---

### Part 16: 코드 구조와 의존성

```
── 파일 구조 ────────────────────────────────────

crates/primitives/src/plonk/
  ├── mod.rs              -- 모듈 루트, Domain, 코셋 상수, re-exports
  ├── arithmetization.rs  -- 게이트 + 제약 시스템 + lookup 필드
  ├── permutation.rs      -- σ 순열 + grand product Z(x)
  └── lookup.rs           -- ★ Plookup (이 단계)

── lookup.rs 내부 구조 ──────────────────────────

  fr_cmp()              — Fr 비교 헬퍼 (to_repr 기반)
  PlookupError          — ValueNotInTable | EmptyTable
  LookupTable           — values, new, range_table, xor_table, and_table
  compute_sorted_list() — f ∪ T 정렬 → (h1, h2) 중첩 분리
  compute_lookup_grand_product() — Z(x) 계산 (N-1 스텝)
  verify_lookup_grand_product()  — ∏ ratio = 1 확인
  PlookupProof          — f_poly, t_poly, h1_poly, h2_poly, z_poly
  compute_plookup()     — 전체 파이프라인 오케스트레이터

── arithmetization.rs 변경 ──────────────────────

  PlonkConstraintSystem에 추가:
    lookup_tables: Vec<Vec<Fr>>
    lookup_entries: Vec<(usize, Column, usize)>
    register_table() → usize
    add_lookup(row, column, table_id)
    are_lookups_satisfied() → bool

  SelectorPolynomials에 추가:
    q_lookup: Polynomial

── 의존성 그래프 ────────────────────────────────

  Fr (스칼라체)
    │
    ├──→ Polynomial (Lagrange 보간, eval)
    │      │
    │      └──→ PLONKish (Step 14)
    │             │
    │             ├──→ ★ Plookup (Step 15)
    │             │      │
    │             │      └──→ PLONK Prover (Step 16)
    │             │
    │             └──→ Permutation (Step 14)
    │                    │
    │                    └──→ PLONK Prover (Step 16)
    │
    └──→ KZG (Step 13) ──→ PLONK Prover (Step 16)
```

---

### Part 17: 테스트 요약

```
15개 테스트 구성:

  LookupTable (5개):
    range_table_values     — range_table(4) → 16개, 0..15 포함
    range_table_8bit       — range_table(8) → 256개, 0..255 포함
    xor_table_2bit         — xor_table(2) → 16개, (2,3,1) 인코딩 확인
    and_table_2bit         — and_table(2) → 16개, (3,2,2) 인코딩 확인
    table_contains         — contains(100)=true, contains(256)=false

  Sorted List (3개):
    sorted_list_simple     — T=[0,1,2,3], f=[1,3] → 중첩 분리 확인
    sorted_list_all_same   — T=[0,1,2], f=[1,1,1] → 중복 처리
    sorted_list_not_in_table — f=[5] ∉ T → Error

  Grand Product (3개):
    grand_product_simple   — T=[0,1,2,3], f=[1,2] → closes ✓
    grand_product_range_8bit — 256개 테이블, f=[0,42,100,255] → closes ✓
    grand_product_violation — f=[5] ∉ T → sorted_list 에러

  CS 통합 (2개):
    cs_register_and_lookup — register + add_lookup, 만족/불만족
    cs_q_lookup_selector   — q_lookup: 4게이트 중 1개만 1

  End-to-End (2개):
    plookup_xor_table      — XOR(1,2)=3 lookup → Z(ω⁰)=1
    plookup_range_check_end_to_end — 여러 값 range check → 전체 검증
```

---

### Part 18: Plookup 변형들과 발전

```
── 원조 Plookup (Gabizon-Williamson 2020) ──────

  이 모듈이 구현한 버전.
  단일 컬럼 테이블 + 멀티 컬럼 인코딩.

── cq (Eagen-Fiore-Gabizon 2022) ───────────────

  "cached quotients for fast amortized lookups"
  테이블 크기가 클 때 (2^20 이상) prover 비용 감소
  테이블을 미리 commit 해놓고 재사용

── Lasso (Setty-Thaler-Goldwasser 2023) ────────

  "Lasso: Lookup argument with sparse tables"
  테이블이 구조화된 경우 (예: 비트 연산)
  테이블 크기에 의존하지 않는 prover 비용
  "decomposable table" 개념

── LogUp (Habock 2022) ─────────────────────────

  logarithmic derivative를 사용한 lookup
  ∑ 1/(X - fᵢ) = ∑ mⱼ/(X - tⱼ)
  (mⱼ = f에서 tⱼ의 출현 횟수)

  장점: 여러 테이블을 동시에 처리 가능
  halo2에서 채택

── 비교 ───────────────────────────────────────

  ┌──────────────┬───────────────┬──────────────────┐
  │ 기법          │ Prover 비용    │ 테이블 크기 제한   │
  ├──────────────┼───────────────┼──────────────────┤
  │ Plookup      │ O(|f|+|T|)    │ 도메인에 맞아야 함 │
  │ LogUp        │ O(|f|+|T|)    │ 유연              │
  │ cq           │ O(|f|)        │ 큰 테이블 가능     │
  │ Lasso        │ O(|f|·c)      │ 구조화 테이블만    │
  └──────────────┴───────────────┴──────────────────┘
  (c = 분해된 소테이블 수)
```

---

### Part 19: 계산 복잡도

```
┌──────────────────────────┬──────────────────────────────┐
│ 연산                      │ 복잡도                        │
├──────────────────────────┼──────────────────────────────┤
│ LookupTable::new          │ O(d log d) — 정렬            │
│ LookupTable::range_table  │ O(2^n) — 생성               │
│ LookupTable::xor_table    │ O(2^(2·bits)) — 전수조합     │
│ LookupTable::contains     │ O(d) — 선형 탐색             │
│ compute_sorted_list       │ O(m·d) — m=|f|, d=|T|       │
│ compute_lookup_grand_prod │ O(N²) — N번 inv + 보간       │
│ verify_lookup_grand_prod  │ O(N) — N번 곱셈              │
│ compute_plookup           │ O(N²) — sorted + 보간 + Z    │
└──────────────────────────┴──────────────────────────────┘

참고: Lagrange 보간이 O(N²)인 이유는 교육용 구현이기 때문.
프로덕션에서는 FFT 기반 O(N log N) 보간 사용.
contains()도 정렬된 배열이므로 이진 탐색 O(log d)로 개선 가능.
```

---

### Part 20: PLONK 전체 파이프라인에서의 위치

```
PLONK 전체 흐름:

  ┌──────────────────────────────────────────────────────┐
  │                                                        │
  │  Step 14: PLONKish Arithmetization + Permutation       │
  │  ┌────────────────────┐  ┌───────────────────────┐    │
  │  │  PlonkGate         │  │  σ 다항식              │    │
  │  │  PlonkConstraintSys│  │  Grand Product Z_perm  │    │
  │  │  Selector polys    │  │                       │    │
  │  │  Wire polys        │  │                       │    │
  │  └────────────────────┘  └───────────────────────┘    │
  │            │                        │                  │
  │            ↓                        ↓                  │
  │  ★ Step 15: Plookup (이 단계)                          │
  │  ┌─────────────────────────────────────────────────┐  │
  │  │  LookupTable (range, XOR, AND)                  │  │
  │  │  compute_sorted_list → h1, h2 (중첩 분리)       │  │
  │  │  Grand Product Z_lookup                         │  │
  │  │  PlookupProof (f, t, h1, h2, z 다항식)          │  │
  │  └─────────────────────────────────────────────────┘  │
  │            │                                           │
  │            ↓                                           │
  │  Step 16: PLONK Prover/Verifier                        │
  │  ┌─────────────────────────────────────────────────┐  │
  │  │  Round 1: Wire commit (KZG)                     │  │
  │  │  Round 2: Permutation Z commit                  │  │
  │  │  Round 3: Quotient polynomial                   │  │
  │  │  Round 4: Opening evaluations                   │  │
  │  │  Round 5: Linearization + batched opening       │  │
  │  │                                                 │  │
  │  │  + Plookup: h1, h2, Z_lookup commit + 검증      │  │
  │  └─────────────────────────────────────────────────┘  │
  │                                                        │
  └──────────────────────────────────────────────────────┘
```

---

> [!summary] Step 15 요약
> ```
> Plookup = sorted list + 중첩 분리 + grand product
>
> 핵심 방정식:
>   sorted = sort(f ∪ t)  (t의 순서로)
>   h1 = sorted[..N], h2 = sorted[N-1..] (중첩)
>   Z(ω⁰) = 1
>   Z(ω^(i+1)) = Z(ωⁱ) · (1+β)(γ+fᵢ)(A+tᵢ+β·tᵢ₊₁)
>                        / ((A+h1ᵢ+β·h1ᵢ₊₁)(A+h2ᵢ+β·h2ᵢ₊₁))
>   검증: Z(ω^(N-1)) = 1  ↔  f ⊆ t
>
> 핵심 포인트:
>   - 중첩 없으면 연속 쌍 누락 → 곱이 안 닫힘
>   - 사후 패딩은 멀티셋 등식 파괴 → 사전 확장 필수
>   - (1+β) 인자 = 개별 원소를 연속 쌍 공간으로 lifting
>   - 건전성: Schwartz-Zippel → 2(N-1)/|Fr| ≈ 0
>
> 코드: plonk/lookup.rs (~770줄, 15 테스트)
> 의존성: Fr, Polynomial, Domain, PlonkConstraintSystem
>
> 다음: Step 16 PLONK Prover/Verifier
> ```
