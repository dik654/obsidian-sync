# Step 06: Optimal Ate Pairing

## 페어링이란?

```
쌍선형 사상 (bilinear map):

e: G1 × G2 → GT

입력: G1의 점 P, G2의 점 Q
출력: GT ⊂ Fp12*의 원소 (유한체 원소)
```

**핵심 성질 — 쌍선형성**:
$$e(aP, bQ) = e(P, Q)^{ab}$$

> [!important] 왜 이것이 혁명적인가?
> 타원곡선에서는 $P$와 $Q = kP$가 주어져도 $k$를 알 수 없다 (ECDLP).
> 하지만 페어링을 사용하면, **두 점 사이의 "곱셈 관계"를 검증**할 수 있다:
>
> ```
> "3P와 Q의 페어링" == "P와 Q의 페어링을 3제곱" 인가?
> → 네! e(3P, Q) = e(P, Q)^3
> → k=3이라는 관계를 비밀 없이 확인 가능
> ```
>
> 이것이 ZK 증명의 **검증 단계**를 가능하게 한다.

---

## 페어링의 세 군

```
G1 ⊂ E(Fp)      — 기저체 위의 점     (좌표: Fp)
G2 ⊂ E'(Fp2)    — 트위스트 위의 점   (좌표: Fp2)
GT ⊂ Fp12*      — Fp12의 곱셈 부분군 (페어링 결과)

세 군 모두 같은 위수 r을 가진다.
r = 21888...5617 (BN254 scalar field order)
```

> [!note] GT가 Fp12*의 **부분군**인 이유
> Fp12*의 전체 곱셈군 위수는 $p^{12}-1$인데, GT는 그 중 위수 $r$인 부분군.
> 정확히는 $r | (p^{12}-1)$ 이고 $r \nmid (p^k-1)$ for $k < 12$.
> 이 12가 바로 **embedding degree** — [[7-Fp12|Step 04]]에서 다룬 이유.

---

## 페어링이 사용되는 곳: Groth16 검증

```
증명자: π = (A, B, C) ∈ G1 × G2 × G1 제출

검증자: 4개의 페어링으로 검증
  e(A, B) = e(α, β) · e(L, γ) · e(C, δ)

  → 비밀(witness)을 모르고도 "계산이 올바른지" 확인
  → 페어링 4회 + GT 곱셈 몇 번이면 끝
```

> [!abstract] 페어링 없이는?
> 검증자가 직접 계산을 재실행해야 한다 — 그러면 ZK가 아니다!
> 페어링 덕분에 "결과의 정당성"만 빠르게 확인 가능.

---

## 전체 구조: Miller Loop + Final Exponentiation

```rust
/// 페어링 함수: e(P, Q) → GT ⊂ Fp12*
/// Jacobian 좌표 → Affine 변환 후 Miller loop + final exp
pub fn pairing(p: &G1, q: &G2) -> Fp12 {
    let p_aff = p.to_affine();   // Jacobian → Affine (inv 1회)
    let q_aff = q.to_affine();
    if p_aff.infinity || q_aff.infinity { return Fp12::ONE; }
    let f = miller_loop(&p_aff, &q_aff);
    final_exponentiation(f)
}
```

```
e(P, Q) = final_exponentiation(miller_loop(P, Q))

Miller Loop:
  → "이산로그의 관계"를 Fp12 원소 f에 인코딩
  → f ∈ Fp12이지만 아직 GT의 원소는 아님

Final Exponentiation:
  → f^((p^12-1)/r) 으로 GT에 투영
  → r-th root of unity 부분군으로 매핑
```

> [!tip] 비유
> Miller loop = "원석 채굴" (정보가 담긴 Fp12 원소를 만듦)
> Final exp = "연마" (GT 부분군에 정확히 떨어지도록 정규화)

---

## Miller Loop 상세

### Miller 알고리즘의 직관

```
f_{n,Q}(P) — "Q에서 시작해서 n번 곱한 것의 흔적을 P에서 평가"

핵심 아이디어:
  타원곡선의 점 덧셈 P+Q에서, P와 Q를 지나는 직선(line)이 존재.
  이 직선의 방정식을 "다른 점"에서 평가하면 Fp12 원소를 얻는다.
  이것을 반복 축적하면 페어링 값이 된다.

루프 파라미터: |6u+2| where u = BN254 parameter
```

### 6u+2는 어디서 오는가?

```
일반 Tate pairing:  루프 길이 = r (≈254비트) → 느림
Ate pairing:        루프 길이 = t-1 (trace, ≈64비트) → 빠름!
Optimal Ate:        루프 길이 = |6u+2| (≈65비트) → 더 빠름!

BN254에서:
  u = 0x44E992B44A6909F1
  p = 36u⁴ + 36u³ + 24u² + 6u + 1
  r = 36u⁴ + 36u³ + 18u² + 6u + 1
  t = 6u² + 1  (Frobenius trace)

6u+2 ≈ 65비트 → r의 254비트 대비 루프 길이가 1/4로 줄어듦!
```

> [!important] 왜 "Optimal"인가?
> Ate pairing은 trace $t-1$을 사용 (약 127비트).
> Optimal Ate는 $t-1$보다 더 짧은 $6u+2$를 사용 (65비트).
> $r$과 $6u+2$의 관계: $6u+2 \equiv t-1 \pmod{r}$ 이런 동치 관계가 성립.

### NAF (Non-Adjacent Form)

```
일반 이진수:  비트 1의 개수 ≈ 절반 → 덧셈이 많다
NAF:         연속된 두 비트가 동시에 비-영이 아님
             → 비-영 비트가 약 1/3로 줄어듦
             → 덧셈(addition step) 횟수 감소
```

```rust
// |6u+2| NAF (LSB first, 65 entries)
// go-ethereum bn256 참조
const ATE_NAF: [i8; 65] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0,
    0, 1, 1, 0, -1, 0, 0, 1, 0, -1, 0, 0, 0, 0, 1, 1,
    1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1,
    1, 0, 0, -1, 0, 0, 0, 1, 1, 0, -1, 0, 0, 1, 0, 1, 1,
];
// 값: {-1, 0, 1} 만 사용. 비-영 비트: 약 22개 (65의 1/3)
```

---

### Line Function: 가장 핵심적인 함수

#### 직선의 방정식

```
타원곡선의 점 덧셈:
  T + T = 2T: 접선이 커브와 세 번째 교점에서 만남
  T + Q:      할선이 커브와 세 번째 교점에서 만남

이 접선/할선의 방정식 = line function ℓ
ℓ을 "다른 점 P"에서 평가 → Fp12 원소
```

```
접선 기울기:  λ = 3x_T² / (2y_T)     ← 미분
할선 기울기:  λ = (y_Q - y_T)/(x_Q - x_T) ← 두 점 기울기

직선 방정식:  ℓ(X, Y) = Y - y_T - λ(X - x_T)
```

#### D-type Twist와 Line 평가

```
문제: P ∈ E(Fp), T ∈ E'(Fp2) — 서로 다른 커브 위의 점!
해결: twist 동형사상 ψ를 통해 같은 커브로 올린다

ψ: E'(Fp2) → E(Fp12)
ψ(X', Y') = (X'·v, Y'·w³)

여기서:
  v³ = ξ = 9+u  (Fp6의 non-residue)
  w² = v         (Fp12의 확장 변수)
  w⁶ = ξ         (6차 twist의 핵심)

ψ⁻¹(P): P = (xP, yP) ∈ E(Fp) → (xP/v, yP/w³) ∈ E'(Fp12)
```

> [!note] ψ⁻¹이 필요한 이유
> line function은 E' 위에서 정의되었으므로, P를 E'로 올려야 평가할 수 있다.
> P의 좌표는 Fp이지만, ψ⁻¹(P)의 좌표는 Fp12 — v와 w가 포함됨.

#### Line 평가 전개

```
ℓ(ψ⁻¹(P)) = yP/w³ - yT - λ·(xP/v - xT)

분모(v, w³)를 제거하기 위해 w³을 곱한다:
(final exponentiation에서 상수 factor는 소거되므로 무관)

w³·ℓ = yP + (-λ·xP)·(w³/v) + (λ·xT - yT)·w³
```

```
w³/v = w·v/v = w  (∵ w³ = w·w² = w·v)

따라서:
w³·ℓ = yP + (-λ·xP)·w + (λ·xT - yT)·w·v
       ~~~   ~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~
       상수    w 성분        w·v 성분
```

#### Fp12 sparse 표현

```
Fp12 = c0 + c1·w  where c0, c1 ∈ Fp6
Fp6 = a0 + a1·v + a2·v²

w³·ℓ = yP + (-λ·xP)·w + (λ·xT - yT)·w·v
     = yP + [(-λ·xP) + (λ·xT - yT)·v]·w

c0 = Fp6(embed(yP), 0, 0)                        ← 상수항
c1 = Fp6(-λ·embed(xP), λ·xT - yT, 0)            ← w의 계수
                                      ↑ v의 계수
```

> [!tip] sparse의 의미
> Fp12는 Fp2 × 12 = 12개의 Fp2 성분을 가진다.
> 하지만 line function은 이 중 **3개만 비영** — 나머지 9개는 0.
> 최적화된 구현에서는 이 sparse 구조를 활용한 특수 곱셈을 사용.
> (우리 구현은 교육 목적으로 full Fp12 multiplication 사용)

```rust
/// Fp를 Fp2에 embed (허수부 = 0)
fn embed(x: Fp) -> Fp2 { Fp2::new(x, Fp::ZERO) }

/// Fp2 실수 상수 생성
fn fp2c(n: u64) -> Fp2 { Fp2::new(Fp::from_u64(n), Fp::ZERO) }

/// line function 결과를 Fp12로 매핑
fn line_eval(lambda: Fp2, xt: Fp2, yt: Fp2, p: &G1Affine) -> Fp12 {
    Fp12::new(
        Fp6::new(embed(p.y), Fp2::ZERO, Fp2::ZERO),    // c0
        Fp6::new(
            -(lambda * embed(p.x)),  // w의 상수 계수: -λ·xP
            lambda * xt - yt,        // w·v의 계수: λ·xT - yT
            Fp2::ZERO,
        ),                                               // c1
    )
}
```

---

### Doubling Step: 접선 계산 + 점 더블링

```rust
/// T에서의 접선을 계산하고, P에서 평가
fn line_double(t: &G2Affine, p: &G1Affine) -> (G2Affine, Fp12) {
    // 접선 기울기: λ = 3x_T² / (2y_T)
    let lambda = (fp2c(3) * t.x.square()) * (fp2c(2) * t.y).inv().unwrap();

    // 새 점 2T (affine 덧셈 공식)
    let x2t = lambda.square() - fp2c(2) * t.x;
    let y2t = lambda * (t.x - x2t) - t.y;

    (G2Affine::new(x2t, y2t), line_eval(lambda, t.x, t.y, p))
}
```

> [!note] Affine 좌표를 사용하는 이유
> Miller loop의 T는 **G2Affine** (Jacobian이 아님).
> 이유: line function의 기울기 λ에 T의 좌표가 직접 사용되므로,
> Jacobian → Affine 변환 비용보다 Affine에서 직접 계산하는 것이 간단.
> (프로덕션에서는 Jacobian + 사영 좌표로 inv를 제거하는 최적화를 사용)

---

### Addition Step: 할선 계산 + 점 덧셈

```rust
/// T와 Q를 지나는 할선을 계산하고, P에서 평가
fn line_add(t: &G2Affine, q: &G2Affine, p: &G1Affine) -> (G2Affine, Fp12) {
    // 할선 기울기: λ = (y_Q - y_T) / (x_Q - x_T)
    let lambda = (q.y - t.y) * (q.x - t.x).inv().unwrap();

    // 새 점 T + Q
    let xr = lambda.square() - t.x - q.x;
    let yr = lambda * (t.x - xr) - t.y;

    (G2Affine::new(xr, yr), line_eval(lambda, t.x, t.y, p))
}
```

> [!tip] line_double과 line_add의 공통점
> 둘 다 `line_eval(lambda, t.x, t.y, p)`를 호출 — 차이는 **λ의 계산**뿐:
>
> | 함수 | λ 계산 | 의미 |
> |------|--------|------|
> | line_double | $3x_T^2 / 2y_T$ | T에서의 접선 기울기 |
> | line_add | $(y_Q - y_T)/(x_Q - x_T)$ | T, Q를 지나는 할선 기울기 |

---

### Miller Loop 코드

```rust
fn miller_loop(p: &G1Affine, q: &G2Affine) -> Fp12 {
    let mut f = Fp12::ONE;  // 누적값
    let mut t = *q;          // 순회 점 (Q에서 시작)
    let neg_q = G2Affine::new(q.x, -q.y);

    // NAF는 LSB first 저장 → index 63→0으로 MSB→LSB 순회
    // index 64 (MSB = 1)는 T = Q 초기화로 대체
    for i in (0..64).rev() {
        // Doubling step: 매 비트마다 반드시 실행
        let (new_t, line) = line_double(&t, p);
        t = new_t;           // T ← 2T
        f = f.square() * line; // f ← f² · ℓ_{T,T}(P)

        // Addition step: NAF 비트가 비-영일 때만
        match ATE_NAF[i] {
            1  => {
                let (new_t, line) = line_add(&t, q, p);
                t = new_t;   // T ← T + Q
                f = f * line;
            }
            -1 => {
                let (new_t, line) = line_add(&t, &neg_q, p);
                t = new_t;   // T ← T + (-Q)
                f = f * line;
            }
            _ => {}  // 0이면 addition step 생략
        }
    }

    // ... BN 보정항 (아래에서 설명)
    f
}
```

> [!abstract] 연산량 분석
>
> | 단계 | 횟수 | Fp12 연산 |
> |------|------|-----------|
> | doubling step | 64회 (매 비트) | square + mul |
> | addition step | ~22회 (NAF 비-영) | mul |
> | BN 보정항 | 2회 | mul |
> | **합계** | | ~64 square + ~88 mul |

---

### BN 보정항: π(Q)와 -π²(Q)

```
Optimal Ate pairing의 이론적 결과에 의하면:
Miller loop 후 T가 정확히 O(무한원점)가 아닐 수 있다.

BN 커브에서는 추가로:
  T ← T + π(Q)     (Frobenius of Q)
  T ← T + (-π²(Q)) (negative Frobenius² of Q)

이 두 step을 추가해야 정확한 페어링 값을 얻는다.
```

#### Frobenius Endomorphism on G2

```
π: (x, y) → (x^p, y^p)  — p-power Frobenius

G2 좌표는 Fp2이므로:
  x^p = conj(x)  (Fp2의 Frobenius = conjugate)
  y^p = conj(y)

하지만! twist 때문에 보정 상수가 필요:
  π(x, y) = (conj(x) · γ₁₁, conj(y) · γ₂₁)

  γ₁₁ = ξ^((p-1)/3)  ← x좌표 보정
  γ₂₁ = ξ^((p-1)/2)  ← y좌표 보정
```

> [!note] 왜 보정 상수가 필요한가?
> Frobenius는 원래 E(Fp12) 위에서 정의되지만, 우리는 twist E'(Fp2)에서 작업.
> E' → E → Frobenius → E → E' 의 왕복에서 twist 파라미터의 p-power가 필요.
>
> 구체적으로: ψ(x', y') = (x'·v, y'·w³)이고 v^p = v·ξ^((p-1)/3) 이므로,
> twist를 통과시킬 때 ξ의 적절한 거듭제곱이 상수로 등장.

```rust
fn frobenius_g2(q: &G2Affine) -> G2Affine {
    let g11 = Fp2::new(                      // γ₁₁ = ξ^((p-1)/3)
        Fp::from_raw([0x99e39557176f553d, 0xb78cc310c2c3330c,
                       0x4c0bec3cf559b143, 0x2fb347984f7911f7]),
        Fp::from_raw([0x1665d51c640fcba2, 0x32ae2a1d0b7c9dce,
                       0x4ba4cc8bd75a0794, 0x16c9e55061ebae20]),
    );
    let g21 = Fp2::new(                      // γ₂₁ = ξ^((p-1)/2)
        Fp::from_raw([0xdc54014671a0135a, 0xdbaae0eda9c95998,
                       0xdc5ec698b6e2f9b9, 0x063cf305489af5dc]),
        Fp::from_raw([0x82d37f632623b0e3, 0x21807dc98fa25bd2,
                       0x0704b5a7ec796f2b, 0x07c03cbcac41049a]),
    );
    G2Affine::new(q.x.conjugate() * g11, q.y.conjugate() * g21)
    //            ^^^^^^^^^^^^^^^^          ^^^^^^^^^^^^^^^^
    //            x^p = conj(x)             y^p = conj(y)
}
```

```rust
fn frobenius_g2_sq(q: &G2Affine) -> G2Affine {
    // π²(x,y) = (x · γ₁₂, -y)
    // γ₁₂ = ξ^((p²-1)/3)  — 실수 (c1=0)
    // γ₂₂ = ξ^((p²-1)/2) = -1
    let g12 = Fp2::new(
        Fp::from_raw([0xe4bd44e5607cfd48, 0xc28f069fbb966e3d,
                       0x5e6dd9e7e0acccb0, 0x30644e72e131a029]),
        Fp::ZERO,     // ← 실수! (p² ≡ 1 mod 2이므로 Fp2의 허수부 소거)
    );
    G2Affine::new(q.x * g12, -q.y)
    //                        ^^^^
    //                        γ₂₂ = -1 이므로 y 부호 반전
}
```

> [!tip] γ₂₂ = -1인 이유
> ξ^((p²-1)/2) = ξ^(p²-1)/2. 그런데 ξ^(p²-1) 은 Fp2의 non-residue의 norm으로,
> p² ≡ 1 (mod 6)이므로 ξ^((p²-1)/2) = -1이 된다.
> 결과적으로 **y좌표는 단순히 부호 반전** — 복잡한 곱셈 불필요!

---

## Final Exponentiation 상세

### 왜 필요한가?

```
Miller loop의 결과 f ∈ Fp12*는 아직 GT의 원소가 아니다.
GT = {x ∈ Fp12* | x^r = 1} (r-th roots of unity)

f를 GT로 매핑하려면:
  f^((p^12-1)/r) 을 계산
  → 이 지수가 r의 배수이므로, 결과의 r승은 항상 1
  → 즉, 결과가 GT에 속함을```

### 지수 분해

```
(p^12 - 1) / r = (p^6 - 1) · (p^2 + 1) · (p^4 - p^2 + 1)/r
                  ~~~~~~~~~   ~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~
                  Easy Part 1  Easy Part 2    Hard Part

Easy parts: Fp12의 conjugate, Frobenius로 빠르게 계산
Hard part:  761비트 지수 → square-and-multiply 필요
```

> [!abstract] 왜 이렇게 분해하는가?
> $p^{12} - 1 = (p^6-1)(p^6+1) = (p^6-1)(p^2+1)(p^4-p^2+1)$
>
> 그리고 $r | (p^4-p^2+1)$ (BN 커브의 성질).
> 따라서 $(p^{12}-1)/r$을 세 인자의 곱으로 표현하면,
> 앞 두 인자는 "easy"하게 계산하고 마지막만 "hard"하게 계산.

---

### Easy Part 1: $f^{p^6-1}$

```
f^(p^6-1) = f^(p^6) · f^(-1)
          = conj(f) · f^(-1)

왜 f^(p^6) = conj(f) 인가?

Fp12 = Fp6[w] / (w² - v)

w^(p^6) = w · (w^2)^((p^6-1)/2)
        = w · v^((p^6-1)/2)
        = w · (-1)           ← v^((p^6-1)/2) = -1 (QR 성질)
        = -w

따라서:
(c0 + c1·w)^(p^6) = c0^(p^6) + c1^(p^6) · w^(p^6)
                   = c0 + c1 · (-w)     ← p^6은 Fp6 위에서 항등
                   = c0 - c1·w
                   = conjugate(f)
```

```rust
// Easy Part 1: 연산 = conjugate 1번 + inv 1번 + mul 1번
let f_inv = f.inv().unwrap();
let r1 = f.conjugate() * f_inv;
```

> [!tip] Conjugate의 비용
> Fp12::conjugate는 **c1의 부호만 반전** — Fp 곱셈 0번!
> 사실상 "공짜" 연산이다. 이것이 타워 구조의 숨은 장점.

---

### Easy Part 2: $\text{result}^{p^2+1}$

```
result^(p^2+1) = result^(p^2) · result
              = φ_2(result) · result

φ_2 = Frobenius² map on Fp12
```

#### Fp12 Frobenius² 구현

```
f^(p²)에서 Fp2 원소 x^(p²) = x (Frobenius 2회 = 항등)
하지만! v와 w는 Fp2의 원소가 아니므로 보정 필요:

v^(p²) = v · ξ^((p²-1)/3)     ← 이 상수가 v 계수에 곱해짐
v^(2p²) = v² · ξ^(2(p²-1)/3)  ← v² 계수에 곱해짐
w^(p²) = w · ξ^((p²-1)/6)     ← w 계수에 곱해짐
```

| 보정 대상 | 상수 | 값 특징 |
|-----------|------|---------|
| v 계수 (Fp6) | $\xi^{(p^2-1)/3}$ | 실수 (c1=0) |
| v² 계수 (Fp6) | $\xi^{2(p^2-1)/3}$ | 실수 (c1=0) |
| w 계수 (Fp12) | $\xi^{(p^2-1)/6}$ | 실수 (c1=0) |

> [!note] 왜 모두 "실수" (허수부 = 0)인가?
> $p^2 \equiv 1 \pmod{2}$ 이므로 Fp2의 Frobenius²는 항등사상.
> 따라서 보정 상수도 Fp 원소 (= 실수 Fp2 원소)가 된다.
> $p^1$의 보정 상수들은 일반적으로 복소수 — 여기서 차이가 있다.

```rust
fn fp6_frob2(a: &Fp6) -> Fp6 {
    let gv1 = Fp2::new(/* ξ^((p²-1)/3) */, Fp::ZERO);
    let gv2 = Fp2::new(/* ξ^(2(p²-1)/3) */, Fp::ZERO);
    Fp6::new(
        a.c0,           // 상수: 변화 없음
        a.c1 * gv1,     // v 계수: 보정 상수 곱
        a.c2 * gv2,     // v² 계수: 보정 상수 곱
    )
}

fn fp12_frob2(f: &Fp12) -> Fp12 {
    let c0 = fp6_frob2(&f.c0);
    let c1 = fp6_frob2(&f.c1);
    let gw = Fp2::new(/* ξ^((p²-1)/6) */, Fp::ZERO);
    // c1의 모든 Fp2 성분에 gw 곱
    Fp12::new(c0, Fp6::new(c1.c0 * gw, c1.c1 * gw, c1.c2 * gw))
}
```

---

### Hard Part: $(p^4 - p^2 + 1)/r$

```
이 지수는 약 761비트.
naive 방법: square-and-multiply → ~761 squarings + ~380 muls

우리 구현: Python으로 지수를 직접 계산하여 [u64; 12]로 하드코딩
```

```python
# 지수 계산 (Python)
p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
hard = (p**4 - p**2 + 1) // r
assert (p**4 - p**2 + 1) % r == 0  # 나누어 떨어짐 확인!
# → 761비트, 12개 u64 limb
```

```rust
fn final_exponentiation(f: Fp12) -> Fp12 {
    // Easy Part 1: f^(p^6-1) = conjugate(f) · f^(-1)
    let f_inv = f.inv().unwrap();
    let r1 = f.conjugate() * f_inv;

    // Easy Part 2: r1^(p^2+1) = φ_2(r1) · r1
    let r2 = fp12_frob2(&r1) * r1;

    // Hard Part: r2^((p^4-p^2+1)/r)
    let hard_exp: [u64; 12] = [
        0xe81bb482ccdf42b1, 0x5abf5cc4f49c36d4,
        0xf1154e7e1da014fd, 0xdcc7b44c87cdbacf,
        0xaaa441e3954bcf8a, 0x6b887d56d5095f23,
        0x79581e16f3fd90c6, 0x3b1b1355d189227d,
        0x4e529a5861876f6b, 0x6c0eb522d5b12278,
        0x331ec15183177faf, 0x01baaa710b0759ad,
    ];
    r2.pow(&hard_exp)
}
```

> [!abstract] Hard Part 최적화 (미구현)
> BN 파라미터 $u$를 사용한 addition chain으로 대폭 최적화 가능:
>
> ```
> a = f^u,  b = a^u = f^(u²),  c = b^u = f^(u³)
> + Frobenius maps (π, π², π³)
> + 약간의 곱셈/제곱
> → 761비트 pow 대신 ~4회의 63비트 pow + Frobenius
> ```
>
> 이 최적화는 프로덕션 구현에서는 필수이지만,
> 교육용에서는 정확성이 우선이므로 naive pow 사용.

---

## 보정 상수 요약

모든 상수는 `ξ = 9+u` (Fp2의 non-residue)의 거듭제곱:

| 상수 | 용도 | 값 |
|------|------|-----|
| $\xi^{(p-1)/3}$ | G2 Frobenius x-보정 | 복소수 Fp2 |
| $\xi^{(p-1)/2}$ | G2 Frobenius y-보정 | 복소수 Fp2 |
| $\xi^{(p^2-1)/3}$ | G2 Frobenius² x-보정 | 실수 (c1=0) |
| $\xi^{(p^2-1)/2}$ | G2 Frobenius² y-보정 | = **-1** |
| $\xi^{(p^2-1)/3}$ | Fp6 Frobenius² v-보정 | 실수 (c1=0) |
| $\xi^{2(p^2-1)/3}$ | Fp6 Frobenius² v²-보정 | 실수 (c1=0) |
| $\xi^{(p^2-1)/6}$ | Fp12 Frobenius² w-보정 | 실수 (c1=0) |

> [!note] 상수 계산 방법
> Python의 `pow(xi, exp, p)`로 Fp2 산술을 수행하여 계산.
> 결과를 `[u64; 4]` 리틀엔디안 리므로 변환하여 Rust에 하드코딩.

---

## 테스트로 검증하는 성질들

### 1. 기본 성질

```rust
#[test] fn pairing_of_identity_is_one()  // e(O, Q) = 1
#[test] fn pairing_identity_g2()          // e(P, O) = 1
#[test] fn pairing_non_degenerate()       // e(G1, G2) ≠ 1
```

> 항등원의 페어링이 1이고, 생성자끼리의 페어링은 1이 아님.

### 2. 쌍선형성 (가장 중요!)

```rust
#[test] fn pairing_bilinearity_lhs()   // e(7P, Q) = e(P, Q)^7
#[test] fn pairing_bilinearity_rhs()   // e(P, 5Q) = e(P, Q)^5
#[test] fn pairing_bilinearity_both()  // e(3P, 5Q) = e(P, Q)^15
```

> [!important] 쌍선형성이 의미하는 것
> ```
> 좌변: 타원곡선에서 스칼라 곱 먼저, 그 다음 페어링
> 우변: 페어링 먼저, 그 다음 GT에서 거듭제곱
>
> e(3P, 5Q) = e(P, Q)^15 ← 두 경로가 같은 결과!
>             ↑                 ↑
>         "곡선 위"에서       "체 위"에서
>         곱셈 관계를         곱셈 관계를
>         인코딩              검증
> ```
> 이것이 ZK 증명에서 "비밀을 모르고도 관계를 검증"할 수 있는 원리.

### 3. 부호 반전

```rust
#[test] fn pairing_negation()  // e(P, -Q) · e(P, Q) = 1
```

> $e(P, -Q) = e(P, Q)^{-1}$ — Q를 부정하면 페어링이 역원이 됨.

### 4. GT 위수 검증

```rust
#[test] fn pairing_result_has_order_r()  // e(P, Q)^r = 1
```

> 페어링 결과가 정확히 위수-r 부분군 GT에 속함을 확인.

---

## Fp12에 추가된 메서드: pow

```rust
// fp12.rs에 추가
pub fn pow(&self, exp: &[u64]) -> Fp12 {
    let mut result = Fp12::ONE;
    let mut base = *self;
    for &limb in exp.iter() {
        for j in 0..64 {
            if (limb >> j) & 1 == 1 {
                result = result * base;
            }
            base = base.square();
        }
    }
    result
}
```

> [!note] 기존 Fp::pow와의 차이
> Fp::pow는 `&[u64; 4]` (고정 크기) — 항상 256비트
> Fp12::pow는 `&[u64]` (가변 크기) — hard part에서 761비트(12 limbs) 필요

---

## 전체 의존성 그래프

```
Fp::pow                    ← Fp::inv (Fermat's little theorem)
  └─ Fp2::inv              ← norm → Fp
      └─ Fp6::inv          ← norm → Fp2
          └─ Fp12::inv     ← norm → Fp6

Fp2::mul_by_nonresidue     ← Fp6::mul 내부 (v³=β 처리)
Fp6::mul_by_nonresidue     ← Fp12::mul 내부 (w²=v 처리)

G1::add, double, scalar_mul ← Fp 산술
G2::add, double, scalar_mul ← Fp2 산술

pairing:
  ├── line_double, line_add  ← Fp2 inv + Fp12 생성
  ├── miller_loop            ← Fp12 square + mul (64회)
  ├── frobenius_g2           ← Fp2 conjugate + 상수 곱
  ├── fp6_frob2, fp12_frob2  ← Fp2 곱셈 (상수)
  ├── Fp12::conjugate        ← c1 부호 반전
  ├── Fp12::inv              ← norm reduction chain
  └── Fp12::pow              ← 761비트 거듭제곱
```

**127 tests passing** (Fp~Fp12: 90개, G1: 15개, G2: 14개, pairing: 8개)

---

---

## 수학적 기초 I: Miller 알고리즘 — 왜 직선의 축적이 페어링이 되는가

### 인수(Divisor)란?

```
타원곡선 위의 유리함수 f에 대해, f의 인수(divisor)는
"f가 어디서 영점이고 어디서 극점인지"를 형식적 합으로 기록한 것.

div(f) = Σ nᵢ(Pᵢ)

  nᵢ > 0 → f는 Pᵢ에서 nᵢ차 영점
  nᵢ < 0 → f는 Pᵢ에서 |nᵢ|차 극점
```

**예시: 직선의 인수**

```
직선 ℓ이 타원곡선과 세 점 P, Q, -(P+Q)에서 만난다면:

div(ℓ) = (P) + (Q) + (-(P+Q)) - 3(O)
                                 ^^^^
                                 사영 좌표에서 무한원점의 극점
```

> [!important] 타원곡선의 인수 정리
> 유리함수 $f$의 인수는 항상 **차수 0**(영점과 극점의 개수가 같음)이고,
> **점들의 합이 O**(군 연산 의미에서)를 만족한다:
> $$\sum n_i = 0, \quad \sum n_i P_i = O$$

---

### Miller 함수 $f_{n,Q}$의 정의

```
Q ∈ E(Fp2)에 대해, f_{n,Q}는 다음 인수를 가지는 유리함수:

div(f_{n,Q}) = n(Q) - ([n]Q) - (n-1)(O)

직관:
  "Q를 n번 더한 결과"의 정보를 인코딩하는 함수.
  영점이 Q에서 n중으로 있고,
  극점이 [n]Q와 무한원점에 있다.
```

> [!note] 왜 이런 인수를 원하는가?
> Tate 페어링의 정의가 바로 이 함수의 "특정 점에서의 값"이기 때문:
> $$\text{Tate}(P, Q) = f_{r,Q}(P) \in \mathbb{F}_{p^{12}}^* / (\mathbb{F}_{p^{12}}^*)^r$$
>
> $r$은 군의 위수이므로 $[r]Q = O$이고, 따라서:
> $$\text{div}(f_{r,Q}) = r(Q) - r(O)$$
> "Q에서 r중 영점, O에서 r중 극점"인 함수.

---

### Miller의 재귀: 왜 line function이 등장하는가

핵심은 $f_{n,Q}$를 직접 구하는 대신, **재귀적으로 쌓아올리는** 것이다.

```
목표: f_{i+j,Q}를 f_{i,Q}와 f_{j,Q}로부터 구하고 싶다.

인수 관계:
  div(f_{i,Q}) = i(Q) - ([i]Q) - (i-1)(O)
  div(f_{j,Q}) = j(Q) - ([j]Q) - (j-1)(O)

두 인수를 더하면:
  (i+j)(Q) - ([i]Q) - ([j]Q) - (i+j-2)(O)

우리가 원하는 것:
  div(f_{i+j,Q}) = (i+j)(Q) - ([i+j]Q) - (i+j-1)(O)

차이:
  ([i+j]Q) - ([i]Q) - ([j]Q) + (O)

이 차이의 인수를 가진 함수가 바로 ℓ/v:
  ℓ_{[i]Q,[j]Q} = [i]Q와 [j]Q를 지나는 직선
  v_{[i+j]Q}     = x = x_{[i+j]Q} 수직선
```

```
직선 ℓ의 인수:   ([i]Q) + ([j]Q) + (-[i+j]Q) - 3(O)
수직선 v의 인수:  ([i+j]Q) + (-[i+j]Q) - 2(O)

ℓ/v의 인수:  ([i]Q) + ([j]Q) - ([i+j]Q) - (O)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             정확히 "차이"와 부호 반대!
```

따라서:

$$f_{i+j,Q} = f_{i,Q} \cdot f_{j,Q} \cdot \frac{\ell_{[i]Q,[j]Q}}{v_{[i+j]Q}}$$

> [!abstract] Miller 재귀의 핵심 공식
>
> **Doubling** ($i = j$):
> $$f_{2i,Q} = f_{i,Q}^2 \cdot \frac{\ell_{[i]Q,[i]Q}}{v_{[2i]Q}}$$
> $\ell_{[i]Q,[i]Q}$ = $[i]Q$에서의 **접선** (tangent)
>
> **Addition** ($j = 1$):
> $$f_{i+1,Q} = f_{i,Q} \cdot f_{1,Q} \cdot \frac{\ell_{[i]Q,Q}}{v_{[i+1]Q}}$$
> $\ell_{[i]Q,Q}$ = $[i]Q$와 $Q$를 지나는 **할선** (secant)
>
> $f_{1,Q} = 1$ (인수가 0)이므로 addition은 $f_{i+1,Q} = f_{i,Q} \cdot \ell / v$

---

### 수직선 $v$는 어디로 가는가?

```
실제 Miller loop에서 수직선 v를 따로 계산하지 않는다!

이유: final exponentiation이 해결해 준다.

f^((p^12-1)/r) 을 계산하면:
  - GT의 원소만 살아남고
  - (Fp12*)^r의 원소는 1로 소거됨

수직선의 평가값은 Fp* ⊂ (Fp12*)^r에 속하므로,
final exponentiation 후 자동으로 소거!
```

> [!tip] 왜 수직선 값이 $(F_{p^{12}}^*)^r$에 속하는가?
> $v_{[i+j]Q}(P)$는 $P$의 $x$좌표만으로 계산되므로 $\mathbb{F}_p^*$의 원소.
> $|\mathbb{F}_{p^{12}}^*| = p^{12}-1$ 이고 $r | (p^{12}-1)$ 이므로:
> $$(\mathbb{F}_p^*)^{(p^{12}-1)/r} = 1$$
> 즉, $\mathbb{F}_p^*$의 모든 원소는 $(p^{12}-1)/r$승하면 1이 된다.
> 따라서 Miller loop에서 $v$를 **무시해도** final exp 후 결과가 동일.

이것이 바로 우리 구현에서 `line_eval`이 수직선을 따로 나누지 않고 직선값만 곱하는 이유:

```rust
// Miller loop 내부
f = f.square() * line;   // ← ℓ만 곱하고, v로 나누지 않는다!
```

---

### Miller Loop 전체 흐름 (인수 관점)

```
n = |6u+2| = Σ bᵢ · 2^i  (NAF 표현)

초기: f ← 1,  T ← Q

MSB부터 LSB까지 반복:
  ① f ← f² · ℓ_{T,T}(P)      ← 인수: f_{2k} = f_k² · (ℓ/v)
     T ← 2T                    ← doubling step

  ② if bᵢ = ±1:
     f ← f · ℓ_{T,±Q}(P)      ← 인수: f_{2k±1} = f_{2k} · (ℓ/v)
     T ← T ± Q                 ← addition step

루프 종료 시:
  div(f) = n(Q) - ([n]Q) - (n-1)(O)  (수직선 인자 제외)
  → f(P) = f_{n,Q}(P) (up to elements in (Fp12*)^r)

BN 보정항 추가:
  f ← f · ℓ_{T,π(Q)}(P)       ← T + π(Q)
  f ← f · ℓ_{T,-π²(Q)}(P)     ← T + (-π²(Q))
  → Optimal Ate가 Tate와 동치임을 보장

Final exponentiation:
  e(P,Q) = f^((p^12-1)/r)     ← GT 부분군으로 투영 + v 소거
```

> [!important] BN 보정항이 필요한 이유
> Optimal Ate 페어링에서 루프 파라미터는 $|6u+2|$이지만,
> 실제로 필요한 것은 다음 관계:
>
> $$f_{6u+2,Q}(P) \cdot \ell_{T,\pi(Q)}(P) \cdot \ell_{T',-\pi^2(Q)}(P)$$
>
> 이것은 Vercauteren의 정리에 의해 Tate 페어링과 동치:
> $$\text{optimal\_ate}(P, Q) = \text{Tate}(P, Q)^c \quad (c \neq 0 \bmod r)$$
>
> $6u+2 \equiv \lambda \pmod{r}$ 인 $\lambda$가 존재하여,
> $Q_1 = \pi(Q)$와 $Q_2 = -\pi^2(Q)$의 보정으로 정확한 페어링 값을 복구한다.

---

## 수학적 기초 II: Final Exponentiation이 GT로 보내는 증명

### GT의 정의

```
GT = μ_r = {x ∈ Fp12* | x^r = 1}

= Fp12*의 r-th roots of unity 부분군

위수: |GT| = r  (r이 소수이므로 GT ≅ Z/rZ)
```

### 정리: $f^{(p^{12}-1)/r} \in \text{GT}$

**증명:**

```
Fp12*는 위수 p^12 - 1인 순환 곱셈군이다.
Fermat의 소정리에 의해:

  ∀ f ∈ Fp12*, f^(p^12 - 1) = 1

e = (p^12 - 1)/r 으로 놓으면:

  (f^e)^r = f^(e·r) = f^(p^12 - 1) = 1

따라서 f^e는 x^r = 1의 해, 즉 μ_r = GT의 원소이다.  □
```

> [!tip] 이것은 "사영(projection)" 연산이다
> $f \mapsto f^{(p^{12}-1)/r}$은 $\mathbb{F}_{p^{12}}^* \twoheadrightarrow \mu_r$인 **군 준동형사상**.
> 핵(kernel)은 $(F_{p^{12}}^*)^r$ — 즉, $r$제곱인 원소들은 모두 1로 보내진다.

### 왜 결과가 trivial하지 않은가? (non-degeneracy)

```
"f^e = 1이면 어쩌지?" → 이것이 degenerate case

f ∈ (Fp12*)^r  ⟺  f^((p^12-1)/r) = 1

Miller loop의 결과가 (Fp12*)^r에 속하지 않으면 비자명한 GT 원소를 얻는다.

비퇴화 정리: P ∈ G1, Q ∈ G2가 모두 생성자이면,
  miller_loop(P, Q) ∉ (Fp12*)^r
  → e(P, Q) ≠ 1

이것은 Tate 페어링의 비퇴화성(non-degeneracy)에서 보장된다.
```

### 왜 $(p^{12}-1)/r$이 정수인가?

```
embedding degree의 정의:
  k = 12는 r | (p^k - 1)을 만족하는 최소 k

따라서 r | (p^12 - 1), 즉 (p^12 - 1)/r은 정수이다.

BN254의 경우 더 강한 성질이 있다:
  r | (p^4 - p^2 + 1)

왜냐하면:
  p^12 - 1 = (p^6-1)(p^6+1)
           = (p^6-1)(p^2+1)(p^4-p^2+1)

  r ∤ (p^6-1): k=12가 최소이므로 r ∤ (p^k-1) for k < 12,
               특히 r ∤ (p^6-1)
  r ∤ (p^2+1): p ≡ 1 (mod r)이면 p^2+1 ≡ 2 (mod r),
               r은 큰 소수이므로 r ∤ 2

  r | (p^12-1)이고 r ∤ (p^6-1)·(p^2+1)이므로:
  → r | (p^4-p^2+1)  ✓
```

### Easy Part가 "공짜"인 이유

```
f^(p^6-1) 후의 결과를 g라 하면:

g^(p^6) = (f^(p^6-1))^(p^6)
        = f^(p^6·(p^6-1))
        = f^(p^12 - p^6)
        = f^(p^12-1) · f^(1-p^6)
        = 1 · g^(-1)    ← Fermat
        = g^(-1)

따라서 g^(p^6) = g^(-1), 즉 g^(p^6+1) = 1.

이것은 g가 Fp12*에서 "특별한 위치"에 있음을 의미한다:
  g ∈ ker(x ↦ x^(p^6+1)) ⊂ Fp12*

이 부분군의 위수는 (p^12-1)/(p^6+1) = p^6-1.
Easy Part 1 후 원소가 살 수 있는 "공간"이 p^12-1 → p^6-1로 줄어든다.

마찬가지로 Easy Part 2 (p^2+1 승) 후:
  공간이 p^6-1 → (p^6-1)/(p^2+1) = p^4-p^2+1로 줄어든다.

Hard Part (÷r 승) 후:
  공간이 p^4-p^2+1 → (p^4-p^2+1)/r = GT의 위수 r 으로 최종 도달.
```

> [!abstract] Easy Part의 수학적 의미
>
> | 단계 | 연산 | 남은 위수 | 의미 |
> |------|------|-----------|------|
> | Miller 후 | $f$ | $p^{12}-1$ | $\mathbb{F}_{p^{12}}^*$ 전체 |
> | Easy 1 | $f^{p^6-1}$ | $p^6+1$ | $\mu_{p^6+1}$ 부분군 |
> | Easy 2 | $\cdot^{p^2+1}$ | $(p^6+1)/(p^2+1)$ | $\mu_{p^4-p^2+1}$ 부분군 |
> | Hard | $\cdot^{(p^4-p^2+1)/r}$ | $r$ | $\mu_r = \text{GT}$ |
>
> 각 단계가 "더 깊은 부분군"으로 사영하는 것이다.
> Easy parts는 Frobenius/conjugate로 거의 공짜이고,
> Hard part만 실제 pow 연산이 필요.

---

## 수학적 기초 III: Twist 동형사상 ψ의 유도

### 출발점: 왜 twist가 필요한가?

```
원래 G2의 정의:
  G2 ⊂ E(Fp12) — 좌표가 Fp12의 원소

Fp12 원소 1개 = Fp 원소 12개
점 하나의 좌표 = Fp12 × 2 = Fp × 24개 — 비실용적!

Twist의 마법:
  E'(Fp2) ≅ G2 인 동형사상을 찾으면
  좌표가 Fp2 × 2 = Fp × 4개로 줄어든다.

  24개 → 4개: 6배 절감 (sextic twist이므로!)
```

### 일반 Twist 이론

```
d-차 twist: 동형사상 ψ: E'(Fp^(k/d)) → E(Fp^k)

가능한 d 값: 2, 3, 4, 6

BN254: k = 12, d = 6 (sextic twist)
  → E'(Fp^(12/6)) = E'(Fp2) 위의 점으로 G2를 표현
```

### Sextic Twist의 구성

```
E: Y² = X³ + b         (원래 커브, b = 3)
E': Y'² = X'³ + b/ξ    (트위스트 커브)

여기서 ξ는 Fp2에서 6th non-residue:
  ξ ∉ (Fp2)^2 ∪ (Fp2)^3
  → ξ의 6제곱근이 Fp2에 존재하지 않음
  → Fp12까지 확장해야 6제곱근이 존재
```

#### 왜 ξ = 9+u 인가?

```
타워 구조에서 이미 ξ를 사용하고 있다:

  Fp6 = Fp2[v] / (v³ - ξ)     ← v³ = ξ = 9+u
  Fp12 = Fp6[w] / (w² - v)    ← w² = v

따라서:
  w⁶ = (w²)³ = v³ = ξ

w는 ξ의 6제곱근이다!  즉, ω = w로 놓으면 ω⁶ = ξ.

이것이 twist와 타워가 자연스럽게 연결되는 이유:
  타워의 non-residue ξ = twist의 파라미터 ξ
  타워의 변수 w = ξ의 6제곱근 = twist에 필요한 ω
```

> [!important] 타워와 twist의 통합
> 이것은 우연이 아니다. BN254의 타워가 $\xi = 9+u$로 설계된 이유 중 하나가
> 바로 sextic twist에서 재활용하기 위함이다.
> 하나의 상수가 두 가지 역할을 겸한다:
>
> 1. **확장체 타워 구성**: $v^3 = \xi$, $w^2 = v$
> 2. **Twist 동형사상**: $\omega^6 = \xi$에서 $\omega = w$

### D-type vs M-type

```
Twist에는 두 종류가 있다:

D-type (Divisor type):
  E': Y'² = X'³ + b/ξ       ← b를 ξ로 "나눈다"
  ψ(X', Y') = (X'·ω², Y'·ω³)

M-type (Multiplier type):
  E'': Y''² = X''³ + b·ξ    ← b에 ξ를 "곱한다"
  ψ(X'', Y'') = (X''/ω², Y''/ω³)

BN254는 D-type을 사용.
```

#### 왜 D-type인가?

```
ψ: E'(Fp2) → E(Fp12)가 올바른 동형사상인지 검증:

E' 위의 점: Y'² = X'³ + b/ξ
ψ(X', Y') = (X'·ω², Y'·ω³) = (X'·v, Y'·w³)

E의 방정식 대입:
  좌변: (Y'·w³)² = Y'² · w⁶ = Y'² · ξ
  우변: (X'·v)³ + b = X'³ · v³ + b = X'³ · ξ + b

Y'² = X'³ + b/ξ 에서 양변에 ξ를 곱하면:
  Y'² · ξ = X'³ · ξ + b

  좌변 = 우변  ✓

따라서 ψ는 E'(Fp2) → E(Fp12)의 올바른 동형사상이다.
```

### ψ의 역사상 ψ⁻¹

```
ψ(X', Y') = (X'·v, Y'·w³)

ψ⁻¹(X, Y) = (X/v, Y/w³) = (X·v⁻¹, Y·w⁻³)

v⁻¹ ∈ Fp6, w⁻³ ∈ Fp12 — 좌표가 Fp12로 "풀어짐"

Miller loop에서의 사용:
  P = (xP, yP) ∈ E(Fp)를 E'(Fp12)로 매핑:
  ψ⁻¹(P) = (xP/v, yP/w³)

  xP/v: xP ∈ Fp이므로 결과는 Fp6의 특정 형태
  yP/w³: yP ∈ Fp이므로 결과는 Fp12의 특정 형태
```

### line function에서 twist가 만드는 sparsity

```
ℓ(ψ⁻¹(P)) = yP/w³ - yT - λ(xP/v - xT)

w³을 곱해서 정리:

w³ · ℓ = yP + (-λ·xP)·(w³/v) + (λxT - yT)·w³

w³ = w·w² = w·v 이므로:
  w³/v = w

따라서:
w³ · ℓ = yP + (-λ·xP)·w + (λxT - yT)·wv

Fp12 = c0 + c1·w  (c0, c1 ∈ Fp6)
Fp6 = a0 + a1·v + a2·v²

yP         → c0의 a0 성분에만 기여  (Fp ⊂ Fp2 ⊂ Fp6)
(-λ·xP)·w  → c1의 a0 성분에 기여    (Fp2 ⊂ Fp6)
(λxT-yT)·wv → c1의 a1 성분에 기여   (Fp2 ⊂ Fp6)

12개 Fp2 슬롯 중 3개만 비영 — "sparse" Fp12!
```

> [!abstract] D-type twist가 만드는 sparsity의 핵심
> $\psi^{-1}(P)$에서 $1/v$와 $1/w^3$이 등장하고,
> $P$의 좌표가 $\mathbb{F}_p$ (기저체)이므로,
> 결과 Fp12 원소의 대부분의 성분이 0이 된다.
>
> M-type twist에서는 $\omega^2$와 $\omega^3$이 등장하여
> **다른 위치**의 슬롯이 비영이 된다 — sparsity 패턴이 다르다.
>
> 어떤 type을 쓰든 sparsity 자체는 유지되지만,
> D-type이 BN 커브에서 더 자연스러운 선택이다.

---

## 수학적 기초 IV: 보정 상수 γ의 유도

### 출발점: 확장체에서 Frobenius는 어떻게 작용하는가?

```
Frobenius 사상: φ(x) = x^p

기저체 Fp에서:     φ(x) = x     (항등!)
확장체 Fp2에서:    φ(a+bu) = a-bu = conjugate  (u² = -1이므로)
확장체 Fp6에서:    φ(a+bv+cv²) = ?  → v에 보정 상수 필요
확장체 Fp12에서:   φ(a+bw) = ?     → w에도 보정 상수 필요
```

### 핵심 보조정리: $\alpha^n = c$이면 $\alpha^p = \alpha \cdot c^{(p-1)/n}$

**증명:**

```
α는 Fp^n \ Fp 위의 원소이고, α^n = c ∈ Fp^(n/d)라 하자.

α^p = α · (α^(p-1))
    = α · (α^n)^((p-1)/n)      ← p-1 = n·((p-1)/n)이므로
    = α · c^((p-1)/n)

단, (p-1)/n이 정수여야 한다.
  → n | (p-1)일 때 성립

n ∤ (p-1)인 경우에도, c^((p^d-1)/n) 형태로 일반화 가능.
```

> [!important] 이것이 모든 보정 상수의 원천이다
> 타워의 각 층에서 확장 변수 $\alpha$의 정의 관계 $\alpha^n = c$로부터,
> Frobenius의 작용이 $\alpha \mapsto \alpha \cdot c^{(p-1)/n}$ 으로 결정된다.

### 각 층에서의 적용

#### Fp2: $u^2 = -1$

```
u^p = u · (-1)^((p-1)/2)

BN254에서 p ≡ 3 (mod 4) → (p-1)/2는 홀수 → (-1)^((p-1)/2) = -1

u^p = u · (-1) = -u

따라서 φ(a+bu) = a + b·(-u) = a - bu = conjugate  ✓

보정 상수: (-1)^((p-1)/2) = -1
→ 이것이 Fp2::conjugate가 부호 반전인 이유
```

#### Fp6: $v^3 = \xi$ (where $\xi = 9+u \in \text{Fp2}$)

```
v^p = v · ξ^((p-1)/3)

(p-1)/3이 정수인지? BN254에서 p ≡ 1 (mod 3) → ✓

v²의 경우:
  (v²)^p = (v^p)² = v² · ξ^(2(p-1)/3)

따라서 Fp6 Frobenius:
  φ(c₀ + c₁v + c₂v²) = φ(c₀) + φ(c₁)·v·ξ^((p-1)/3) + φ(c₂)·v²·ξ^(2(p-1)/3)
```

이것이 Fp6 Frobenius에서 v, v² 계수에 곱해지는 상수:

| 계수 | 보정 상수 | 유도 |
|------|-----------|------|
| $v$ 계수 | $\gamma_{v,1} = \xi^{(p-1)/3}$ | $v^p = v \cdot \xi^{(p-1)/3}$ |
| $v^2$ 계수 | $\gamma_{v,2} = \xi^{2(p-1)/3}$ | $(v^2)^p = v^2 \cdot \xi^{2(p-1)/3}$ |

#### Fp12: $w^2 = v$

```
w^p = w · v^((p-1)/2)

v^((p-1)/2) = (v³)^((p-1)/6) · v^((p-1)/2 - 3(p-1)/6)
            = ξ^((p-1)/6) · v^(...)

더 직접적으로:
  w^p = w · v^((p-1)/2)

  v = w² 이므로:
  v^((p-1)/2) = w^(p-1) = (w^p / w) 의 역... 순환 논리.

올바른 접근:
  w⁶ = ξ 이므로:
  w^p = w · ξ^((p-1)/6)

  (p-1)/6이 정수인지? BN254에서 p ≡ 1 (mod 6) → ✓
```

따라서 **모든 보정 상수는 $\xi$의 적절한 거듭제곱**:

| 대상 | 관계식 | 보정 상수 |
|------|--------|-----------|
| $u$ | $u^2 = -1$ | $(-1)^{(p-1)/2} = -1$ |
| $v$ | $v^3 = \xi$ | $\xi^{(p-1)/3}$ |
| $v^2$ | | $\xi^{2(p-1)/3}$ |
| $w$ | $w^6 = \xi$ | $\xi^{(p-1)/6}$ |

### Frobenius² ($p^2$승)의 보정 상수

```
v^(p²) = v · ξ^((p²-1)/3)
w^(p²) = w · ξ^((p²-1)/6)

이것들이 "실수"(c1=0)인 이유:

ξ^((p²-1)/3) ∈ Fp2이고,
φ²(x) = x for all x ∈ Fp2 이므로,
ξ^((p²-1)/3) = φ²(ξ^((p²-1)/3)) = ξ^(p²·(p²-1)/3) = ξ^((p⁴-p²)/3)

한편 ξ^((p²-1)/3) 자체가 Fp2*의 원소인데:
  Fp2의 norm: N(ξ) = ξ · ξ^p = ξ^(p+1)
  N(ξ^((p²-1)/3)) = ξ^((p+1)(p²-1)/3) = ξ^((p³-1+p²-p)/3)

더 직접적인 증명:
  ξ ∈ Fp2이므로 ξ^(p²) = ξ (Fp2의 Frobenius² = 항등)
  ξ^((p²-1)/3)의 conjugate = (ξ^((p²-1)/3))^p
  = ξ^(p(p²-1)/3)

  p(p²-1)/3 mod (p²-1) 을 계산하면:
  p(p²-1)/3 - (p²-1)/3 = (p-1)(p²-1)/3

  ξ^((p-1)(p²-1)/3) = (ξ^(p²-1))^((p-1)/3) = 1^((p-1)/3) = 1

  따라서 conjugate = ξ^((p²-1)/3) · 1 = 원래 값
  → 허수부 = 0 → 실수!  ✓
```

> [!tip] Frobenius²의 보정 상수가 실수인 이유 (요약)
> $\xi \in \mathbb{F}_{p^2}$이므로 $\xi^{p^2} = \xi$,
> 따라서 $\xi^{(p^2-1)} = 1$.
> $\xi^{(p^2-1)/k}$는 $1$의 $k$-th root of unity in $\mathbb{F}_{p^2}$.
> BN254에서 이 root들은 모두 $\mathbb{F}_p$에 속한다 ($p \equiv 1 \bmod 6$).
> 따라서 **Fp2에서의 허수부가 0** — 곧 "실수"가 된다.

### G2 Frobenius의 보정 상수

```
G2 Frobenius는 Fp12 위의 Frobenius를 twist를 통해 E'(Fp2)에 표현한 것:

π_E: (X, Y) ↦ (X^p, Y^p)   on E(Fp12)

ψ(X', Y') = (X'v, Y'w³) 이므로:
  π_E(X'v, Y'w³) = ((X')^p v^p, (Y')^p (w³)^p)
                  = (conj(X')·v·ξ^((p-1)/3), conj(Y')·w³·ξ^((p-1)/2))

                         ← X' ∈ Fp2이므로 (X')^p = conj(X')
                         ← v^p = v·ξ^((p-1)/3)
                         ← (w³)^p = w³·ξ^(3(p-1)/6) = w³·ξ^((p-1)/2)

이것을 ψ⁻¹로 E'에 되돌리면:
  ψ⁻¹(result) = (result_x / v, result_y / w³)
               = (conj(X')·ξ^((p-1)/3), conj(Y')·ξ^((p-1)/2))

따라서 G2의 Frobenius:
  π': (X', Y') ↦ (conj(X')·γ₁₁, conj(Y')·γ₂₁)

  γ₁₁ = ξ^((p-1)/3)    ← x좌표 보정
  γ₂₁ = ξ^((p-1)/2)    ← y좌표 보정
```

#### G2 Frobenius²

```
π'² = π' ∘ π':

x 좌표:
  conj(conj(X')·γ₁₁) · γ₁₁
  = X' · conj(γ₁₁) · γ₁₁
  = X' · |γ₁₁|²       (if γ₁₁ were Fp2 norm)

더 직접적으로:
  v^(p²) = v·ξ^((p²-1)/3)

  → π'²: (X', Y') ↦ (X'·ξ^((p²-1)/3), Y'·ξ^((p²-1)/2))

  γ₁₂ = ξ^((p²-1)/3)        ← 실수 (위에서 증명)
  γ₂₂ = ξ^((p²-1)/2) = -1   ← 아래에서 증명

  X' 앞에 conj 없음: conj²(X') = X'
  Y' 부호만 반전: · (-1)
```

#### $\xi^{(p^2-1)/2} = -1$의 증명

```
ξ^(p²-1) = 1   (ξ ∈ Fp2*이고 |Fp2*| = p²-1)

따라서 ξ^((p²-1)/2)는 1 또는 -1.

만약 ξ^((p²-1)/2) = 1이라면:
  ξ는 Fp2*에서 quadratic residue
  → ξ^(1/2) ∈ Fp2 존재
  → w² = v, v³ = ξ → w⁶ = ξ의 제곱근이 Fp2에 존재
  → 그러면 Fp2에서 Fp12로의 확장이 6차가 아님! 모순.

따라서 ξ^((p²-1)/2) = -1  ✓

이것이 γ₂₂ = -1, 즉 frobenius_g2_sq에서 y 좌표를 단순히 부호 반전하는 이유.
```

> [!abstract] 모든 γ 상수의 통합 유도
>
> **하나의 원리**: 타워에서 $\alpha^n = c$이면 $\alpha^{p^k} = \alpha \cdot c^{(p^k-1)/n}$
>
> ```
> u² = -1    → u^p   = u · (-1)^((p-1)/2)     = -u
> v³ = ξ     → v^p   = v · ξ^((p-1)/3)         = v · γ_{v,1}
>            → v^(p²) = v · ξ^((p²-1)/3)       = v · γ_{v,1}^{(2)} — 실수
> w⁶ = ξ     → w^p   = w · ξ^((p-1)/6)         = w · γ_w
>            → w^(p²) = w · ξ^((p²-1)/6)       = w · γ_w^{(2)} — 실수
> ```
>
> | 상수 | 식 | 값 | Fp2 타입 | 유도 근거 |
> |------|-----|-----|----------|-----------|
> | $\gamma_{v,1}$ | $\xi^{(p-1)/3}$ | 복소 | $v^3 = \xi$ |
> | $\gamma_{v,2}$ | $\xi^{2(p-1)/3}$ | 복소 | $(v^2)^p$ |
> | $\gamma_{v,1}^{(2)}$ | $\xi^{(p^2-1)/3}$ | 실수 | $v^{p^2}$ |
> | $\gamma_{v,2}^{(2)}$ | $\xi^{2(p^2-1)/3}$ | 실수 | $(v^2)^{p^2}$ |
> | $\gamma_w$ | $\xi^{(p-1)/6}$ | 복소 | $w^6 = \xi$ |
> | $\gamma_w^{(2)}$ | $\xi^{(p^2-1)/6}$ | 실수 | $w^{p^2}$ |
> | $\gamma_{1,1}$ | $\xi^{(p-1)/3}$ | 복소 | G2 x-보정 |
> | $\gamma_{2,1}$ | $\xi^{(p-1)/2}$ | 복소 | G2 y-보정 |
> | $\gamma_{1,2}$ | $\xi^{(p^2-1)/3}$ | 실수 | G2 x-보정 |
> | $\gamma_{2,2}$ | $\xi^{(p^2-1)/2}$ | $= -1$ | G2 y-보정 |
>
> 모든 상수가 **단 하나의 원소** $\xi = 9+u$에서 파생된다.

---

### 상수의 실제 계산 (Python)

```python
# BN254 파라미터
p = 21888242871839275222246405745257275088696311157297823662689037894645226208583

# Fp2 산술: a+bu where u²=-1
def fp2_mul(a, b):
    """(a0+a1·u)(b0+b1·u) = (a0b0-a1b1) + (a0b1+a1b0)u"""
    return ((a[0]*b[0] - a[1]*b[1]) % p,
            (a[0]*b[1] + a[1]*b[0]) % p)

def fp2_pow(base, exp):
    result = (1, 0)  # 1 + 0·u
    b = base
    while exp > 0:
        if exp & 1:
            result = fp2_mul(result, b)
        b = fp2_mul(b, b)
        exp >>= 1
    return result

xi = (9, 1)  # ξ = 9 + u

# γ₁₁ = ξ^((p-1)/3)
g11 = fp2_pow(xi, (p - 1) // 3)

# γ₂₁ = ξ^((p-1)/2)
g21 = fp2_pow(xi, (p - 1) // 2)

# γ₁₂ = ξ^((p²-1)/3)  — 실수 (c1=0 검증!)
g12 = fp2_pow(xi, (p*p - 1) // 3)
assert g12[1] == 0  # 실수 확인

# γ₂₂ = ξ^((p²-1)/2)  — -1 검증!
g22 = fp2_pow(xi, (p*p - 1) // 2)
assert g22 == (p - 1, 0)  # = -1 (mod p)
```

---

## 다음 스텝

→ [[11-Poseidon Hash]] — ZK-friendly 해시 함수 (over Fr)

페어링이 "검증"의 도구라면, Poseidon은 "커밋먼트"의 도구.
ZK 회로 내부에서 효율적으로 동작하는 해시가 필요하다.
