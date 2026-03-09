# Step 71: Brute Force 벡터 검색

Phase 4 (벡터 검색)의 첫 번째 스텝. Step 70에서 학습한 문장 임베딩을 실제로 검색하는 파이프라인 구현.

**핵심 질문**: "두 벡터가 얼마나 비슷한가?"를 어떻게 수량화하는가?

---

## 1. 벡터 유사도와 거리: 왜 여러 메트릭이 필요한가?

고차원 벡터 공간에서 "가깝다"의 정의는 하나가 아니다. 응용에 따라 다른 메트릭이 최적:

| 메트릭 | 측정 대상 | 범위 | 정렬 | 대표 사용처 |
|--------|-----------|------|------|-------------|
| Cosine | 방향 (각도) | $[-1, 1]$ | $\downarrow$ 내림 | 문서 유사도, 문장 검색 |
| Dot Product | 방향 + 크기 | $(-\infty, +\infty)$ | $\downarrow$ 내림 | 추천 시스템 (MIPS) |
| L2 (Euclidean) | 직선 거리 | $[0, +\infty)$ | $\uparrow$ 오름 | k-means, KNN |
| L1 (Manhattan) | 격자 거리 | $[0, +\infty)$ | $\uparrow$ 오름 | 이상치 robust 검색 |

---

## 2. 코사인 유사도: 수식과 기하학적 의미

### 2.1 정의

$$
\mathrm{cos}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|} = \frac{\sum_{i=1}^{D} a_i b_i}{\sqrt{\sum_{i=1}^{D} a_i^2} \cdot \sqrt{\sum_{i=1}^{D} b_i^2}}
$$

### 2.2 기하학적 해석

코사인 유사도는 두 벡터 사이 각도 $\theta$의 코사인:

$$
\mathrm{cos}(\mathbf{a}, \mathbf{b}) = \cos\theta
$$

- $\theta = 0°$: 같은 방향 → $\cos\theta = 1$
- $\theta = 90°$: 직교 → $\cos\theta = 0$
- $\theta = 180°$: 반대 방향 → $\cos\theta = -1$

**스케일 불변성**: $\mathrm{cos}(c\mathbf{a}, \mathbf{b}) = \mathrm{cos}(\mathbf{a}, \mathbf{b})$ (임의의 $c > 0$)

이는 분자와 분모에서 $c$가 상쇄되기 때문:

$$
\frac{c\mathbf{a} \cdot \mathbf{b}}{\|c\mathbf{a}\| \cdot \|\mathbf{b}\|} = \frac{c(\mathbf{a} \cdot \mathbf{b})}{c\|\mathbf{a}\| \cdot \|\mathbf{b}\|} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}
$$

→ 문장 임베딩의 "의미"는 방향에 인코딩되므로, 벡터 크기에 무관한 코사인이 자연스러운 선택.

### 2.3 정규화된 벡터에서의 관계

벡터를 L2 정규화하면 ($\hat{\mathbf{a}} = \mathbf{a}/\|\mathbf{a}\|$, $\|\hat{\mathbf{a}}\| = 1$):

$$
\mathrm{cos}(\hat{\mathbf{a}}, \hat{\mathbf{b}}) = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}} = \mathrm{dot}(\hat{\mathbf{a}}, \hat{\mathbf{b}})
$$

즉 **정규화된 벡터에서 cosine ≡ dot product**. 이 관계가 중요한 이유:
- Step 70의 NT-Xent loss가 L2 정규화 후 내적으로 유사도를 계산한 것과 정확히 일치
- 검색 시 벡터를 미리 정규화해두면 cosine search → dot product search로 변환 가능 (연산량 감소)

---

## 3. 내적 (Dot Product): 유사도 vs MIPS

### 3.1 정의

$$
\mathrm{dot}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{D} a_i b_i
$$

### 3.2 코사인과의 차이

내적은 **방향 + 크기**를 동시에 반영:

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \cdot \|\mathbf{b}\| \cdot \cos\theta
$$

따라서 같은 방향이더라도 크기가 크면 내적이 더 커진다.

### 3.3 MIPS (Maximum Inner Product Search)

추천 시스템에서는 user embedding $\mathbf{u}$와 item embedding $\mathbf{v}$의 내적이 선호도를 나타냄:

$$
\mathrm{score}(u, v) = \mathbf{u} \cdot \mathbf{v}
$$

여기서는 크기 정보가 "인기도"나 "확신도"를 인코딩하므로, 정규화하면 정보 손실. 이것이 MIPS가 cosine search와 구분되는 이유.

### 3.4 MIPS → NNS 변환 (Bachrach et al., 2014)

MIPS를 L2 최근접 탐색으로 변환하는 트릭:

$\mathbf{a}' = [\mathbf{a}; \sqrt{M^2 - \|\mathbf{a}\|^2}]$, $\mathbf{b}' = [\mathbf{b}; 0]$으로 차원을 확장하면:

$$
\|\mathbf{a}' - \mathbf{b}'\|^2 = \|\mathbf{a}\|^2 - 2\mathbf{a}\cdot\mathbf{b} + \|\mathbf{b}\|^2 + M^2 - \|\mathbf{a}\|^2 = M^2 + \|\mathbf{b}\|^2 - 2\mathbf{a}\cdot\mathbf{b}
$$

$\|\mathbf{b}\|$가 상수면 $\arg\max \mathbf{a}\cdot\mathbf{b} = \arg\min \|\mathbf{a}'-\mathbf{b}'\|^2$. 이 변환으로 기존 L2 기반 인덱스(HNSW 등)에서 MIPS 해결 가능.

---

## 4. L2 (유클리드) 거리: 메트릭 공간의 기본

### 4.1 정의

$$
d_2(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{D} (a_i - b_i)^2}
$$

### 4.2 메트릭 공간의 공리

L2 거리는 메트릭의 4가지 공리를 모두 만족:

1. **비음수성**: $d(\mathbf{a}, \mathbf{b}) \geq 0$
2. **동일성**: $d(\mathbf{a}, \mathbf{b}) = 0 \iff \mathbf{a} = \mathbf{b}$
3. **대칭성**: $d(\mathbf{a}, \mathbf{b}) = d(\mathbf{b}, \mathbf{a})$
4. **삼각 부등식**: $d(\mathbf{a}, \mathbf{c}) \leq d(\mathbf{a}, \mathbf{b}) + d(\mathbf{b}, \mathbf{c})$

삼각 부등식이 중요한 이유: 이 성질 덕분에 **탐색 공간을 가지치기(pruning)** 할 수 있다. VP-Tree, Ball-Tree 등의 공간 분할 알고리즘이 이 부등식에 기반.

### 4.3 코사인과의 관계

정규화된 벡터에서:

$$
\|\hat{\mathbf{a}} - \hat{\mathbf{b}}\|_2^2 = \|\hat{\mathbf{a}}\|^2 + \|\hat{\mathbf{b}}\|^2 - 2\hat{\mathbf{a}}\cdot\hat{\mathbf{b}} = 2 - 2\cos\theta = 2(1 - \cos\theta)
$$

따라서:

$$
d_2(\hat{\mathbf{a}}, \hat{\mathbf{b}}) = \sqrt{2(1 - \cos\theta)}
$$

**핵심**: 정규화된 벡터에서 L2 최소화 ≡ cosine 최대화. 세 메트릭이 사실상 동치!

이 관계를 표로 정리:

| $\cos\theta$ | $d_2$ | 의미 |
|---|---|---|
| $1$ | $0$ | 동일 방향 |
| $0$ | $\sqrt{2} \approx 1.414$ | 직교 |
| $-1$ | $2$ | 반대 방향 |

---

## 5. L1 (맨해튼) 거리: Robust 대안

### 5.1 정의

$$
d_1(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_1 = \sum_{i=1}^{D} |a_i - b_i|
$$

격자(grid) 위에서 축 방향으로만 이동할 때의 최단 경로. 뉴욕 맨해튼의 블록 구조에서 유래.

### 5.2 L1 vs L2: 이상치 민감도

차이 벡터 $\mathbf{d} = \mathbf{a} - \mathbf{b}$에 대해:

$$
d_2 = \sqrt{\sum d_i^2}, \quad d_1 = \sum |d_i|
$$

L2는 $d_i^2$으로 큰 차이를 제곱으로 증폭. 반면 L1은 $|d_i|$로 선형적으로만 반영.

**예시**: $\mathbf{d} = [0, 0, 0, 10]$ (한 차원만 크게 다른 경우)
- $d_1 = 10$
- $d_2 = 10$

$\mathbf{d} = [2.5, 2.5, 2.5, 2.5]$ (모든 차원이 고르게 다른 경우)
- $d_1 = 10$
- $d_2 = 5$

L1에서는 두 경우가 동일하지만, L2에서는 이상치가 있는 경우가 2배 더 먼 것으로 평가. 이것이 "L1은 이상치에 robust"한 이유.

### 5.3 Lp-norm의 일반화

$$
d_p(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_p = \left(\sum_{i=1}^{D} |a_i - b_i|^p\right)^{1/p}
$$

- $p = 1$: 맨해튼 (L1)
- $p = 2$: 유클리드 (L2)
- $p \to \infty$: Chebyshev ($\max_i |a_i - b_i|$)

### 5.4 Lp 노름 부등식

Jensen 부등식으로부터 $p \leq q$이면:

$$
\|\mathbf{x}\|_q \leq \|\mathbf{x}\|_p \leq D^{1/p - 1/q} \|\mathbf{x}\|_q
$$

특히 $p=1, q=2$일 때:

$$
\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq \sqrt{D} \cdot \|\mathbf{x}\|_2
$$

→ **L1 ≥ L2** (항상). 차원이 높을수록 gap이 커진다.

---

## 6. 차원의 저주 (Curse of Dimensionality)

### 6.1 고차원에서 거리의 집중 현상

$D$차원 단위 초구(hypersphere) 위의 임의 두 점 사이 L2 거리의 기대값:

$$
\mathbb{E}[d_2] = \sqrt{2} \quad (\text{차원 무관})
$$

하지만 **분산**은 $D$가 커질수록 감소:

$$
\mathrm{Var}[d_2] = O(1/D)
$$

→ 모든 점 쌍의 거리가 $\sqrt{2}$ 근처로 집중. "가장 가까운 것"과 "가장 먼 것"의 차이가 사라짐.

### 6.2 상대적 대비 (Relative Contrast)

$$
\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0 \quad \text{as } D \to \infty
$$

(Beyer et al., 1999)

이것이 **brute force가 고차원에서도 여전히 사용되는 이유**: approximate 알고리즘의 이론적 보장도 약해지므로, exact search가 baseline으로서 가치를 유지.

### 6.3 코사인이 L2보다 고차원에 강한 이유

코사인 유사도는 "방향"만 비교하므로, 크기 정보의 집중에 영향을 받지 않음. 문장 임베딩(768~1024차원)에서 cosine이 사실상 표준 메트릭인 이유.

---

## 7. Brute Force 검색: 복잡도 분석

### 7.1 알고리즘

```
BruteForceSearch(query, index, k, metric):
  scores = []
  for each vector v_i in index:      # O(N)
    s = metric(query, v_i)            # O(D)
    scores.append((i, s))
  sort(scores)                        # O(N log N)
  return top_k(scores, k)             # O(k)
```

### 7.2 복잡도 비교

| | Brute Force | IVF | HNSW | PQ |
|--|--|--|--|--|
| 구축 | $O(N)$ | $O(NK + N)$ | $O(N \log N)$ | $O(NDP')$ |
| 쿼리 | $O(ND)$ | $O(nD + n \log n)$ | $O(D \log N)$ | $O(NP')$ |
| 메모리 | $O(ND)$ | $O(ND + KD)$ | $O(ND + NM)$ | $O(NP' \log K')$ |
| Recall | $1.0$ | $< 1.0$ | $< 1.0$ | $< 1.0$ |

$N$: 벡터 수, $D$: 차원, $n$: probe할 클러스터 수, $M$: HNSW 이웃 수

### 7.3 Brute Force의 장점

1. **정확도 100%**: 모든 approximate 알고리즘의 recall 검증 기준
2. **구축 비용 0**: 인덱스 구조 없이 벡터만 저장
3. **갱신 용이**: 벡터 추가/삭제에 재구축 불필요
4. **구현 단순**: 버그 가능성 최소

### 7.4 Brute Force의 한계

$N = 10^6$, $D = 768$일 때 단일 쿼리: $7.68 \times 10^8$ 부동소수점 연산.
100ms latency 목표라면 ~7.68 GFLOPS 필요 → GPU 없이는 실시간 불가.

이것이 Phase 4의 나머지 스텝(IVF, PQ, HNSW)이 필요한 이유.

---

## 8. 검색 품질 평가: Recall@k

### 8.1 정의

$$
\mathrm{Recall@}k = \frac{|\mathrm{Approx\text{-}top\text{-}}k \cap \mathrm{Exact\text{-}top\text{-}}k|}{k}
$$

Brute force의 결과가 "정답(ground truth)"이 되어 approximate 알고리즘을 평가:

- $\mathrm{Recall@}10 = 1.0$: 10개 결과가 모두 brute force와 동일
- $\mathrm{Recall@}10 = 0.9$: 10개 중 9개 일치

### 8.2 Recall vs Latency 트레이드오프

실용적 벡터 검색의 핵심 트레이드오프:

```
Recall  1.0 ──────●  Brute Force (느림, 정확)
        0.99 ─────●  HNSW (빠름, 거의 정확)
        0.95 ────●   IVF-PQ (매우 빠름, 약간 부정확)
        0.80 ──●     Random Projection (초고속, 부정확)
```

대부분의 실무 시스템은 Recall@10 ≥ 0.95를 목표로 함.

---

## 9. 유사도 함수의 수학적 성질 비교

### 9.1 메트릭 공리 충족 여부

| 성질 | Cosine | Dot | L2 | L1 |
|------|--------|-----|----|----|
| 비음수성 | $\times$ ($[-1,1]$) | $\times$ ($\mathbb{R}$) | $\checkmark$ | $\checkmark$ |
| 동일성 ($d=0 \iff a=b$) | $\times$ (스케일) | $\times$ | $\checkmark$ | $\checkmark$ |
| 대칭성 | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| 삼각 부등식 | $\times$ | $\times$ | $\checkmark$ | $\checkmark$ |

→ L2, L1만 진정한 "메트릭". Cosine은 $d_{\cos} = 1 - \cos\theta$로 변환하면 준메트릭(semimetric)이 됨 (삼각 부등식은 만족하지 않을 수 있음).

→ Angular distance $d_\theta = \arccos(\cos\theta) / \pi \in [0,1]$로 변환하면 진정한 메트릭.

### 9.2 어떤 메트릭을 선택할까?

| 상황 | 추천 메트릭 | 이유 |
|------|-------------|------|
| 문장/문서 검색 | Cosine | 의미는 방향에 인코딩 |
| 추천 시스템 | Dot Product | 인기도/확신도 반영 |
| 클러스터링 | L2 | k-means 등 중심 기반 |
| 이상치 많은 데이터 | L1 | 큰 차이에 덜 민감 |
| 정규화된 벡터 | 아무거나 | 모두 동치 (Section 4.3) |

---

## 10. 구현 코드 발췌

### 10.1 Metric 열거형과 유사도 함수

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    Cosine,      // 유사도 (↑)
    DotProduct,  // 유사도 (↑)
    L2,          // 거리 (↓)
    L1,          // 거리 (↓)
}

pub fn cosine_similarity_vec(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm_a * norm_b + 1e-10) // ε으로 0벡터 보호
}
```

### 10.2 BruteForceIndex 핵심 로직

```rust
pub fn search(&self, query: &[f64], k: usize, metric: Metric)
    -> Vec<(usize, f64, String)>
{
    // 1. 전체 선형 스캔: O(N·D)
    let mut scores: Vec<(usize, f64)> = self.vectors.iter()
        .enumerate()
        .map(|(i, v)| (i, match metric {
            Metric::Cosine => cosine_similarity_vec(query, v),
            Metric::DotProduct => dot_product_vec(query, v),
            Metric::L2 => l2_distance(query, v),
            Metric::L1 => l1_distance(query, v),
        }))
        .collect();

    // 2. 정렬: 유사도는 내림차순, 거리는 오름차순
    match metric {
        Metric::Cosine | Metric::DotProduct =>
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()),
        Metric::L2 | Metric::L1 =>
            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
    }

    // 3. top-k 반환
    scores[..k.min(scores.len())].iter()
        .map(|&(i, s)| (i, s, self.labels[i].clone()))
        .collect()
}
```

### 10.3 Pipeline 테스트 결과

```
=== Pipeline: encode → index → search ===
  #1: A_variant (idx=2, cosine=0.9916)
  #2: A_original (idx=0, cosine=0.9898)
  #3: B_original (idx=1, cosine=-0.7386)
```

같은 그룹(A)의 문장이 top-2, 다른 그룹(B)은 cosine ≈ -0.74로 멀리 배치.

---

## 11. 벡터 검색 시스템의 전체 구조

```
┌─────────────────────────────────────────────────┐
│                Application Layer                 │
│  (Semantic Search, RAG, Recommendation, etc.)    │
├─────────────────────────────────────────────────┤
│               Query Processing                   │
│  text → tokenize → encode → query vector         │
├─────────────────────────────────────────────────┤
│              Similarity Search                   │
│  ┌──────────┬────────┬──────────┬──────────┐    │
│  │  Brute   │  IVF   │   PQ    │  HNSW    │    │
│  │  Force   │        │         │          │    │
│  │ (exact)  │(parti- │(compre- │ (graph)  │    │
│  │          │ tion)  │ ssion)  │          │    │
│  └──────────┴────────┴──────────┴──────────┘    │
├─────────────────────────────────────────────────┤
│              Vector Storage                      │
│  In-memory / mmap / disk-backed                  │
├─────────────────────────────────────────────────┤
│             Embedding Model                      │
│  Word2Vec / BERT / Sentence Embedding            │
└─────────────────────────────────────────────────┘
```

Step 71에서 구현한 Brute Force는 이 스택의 **Similarity Search** 계층에서 가장 기본적인 exact search. Phase 4의 이후 스텝에서 IVF, PQ, HNSW로 확장.

---

## 12. 고차원에서의 Cosine vs L2: 실험적 관찰

### 12.1 BERT 768차원에서의 거리 분포

BERT 임베딩은 anisotropic (Step 70 참조): 벡터들이 좁은 원뿔에 집중.
이 경우:

- **L2 거리 분포**: 평균 근처에 좁게 집중 → 구분력 ↓
- **Cosine 분포**: 1.0 근처에 집중되지만 상대적 순서는 보존 → 구분력 ↑

이것이 문장 검색에서 cosine이 L2보다 선호되는 실용적 이유.

### 12.2 Step 71 테스트 결과 비교

```
Cosine:     #1=close (0.9950), #2=medium (0.7740), #3=far (0.0995)
DotProduct: #1=close (1.0000), #2=medium (0.7700), #3=far (0.1000)
L2:         #1=close (0.1000), #2=medium (0.6708), #3=far (1.3454)
L1:         #1=close (0.1000), #2=medium (0.9000), #3=far (1.9000)
```

4종 메트릭 모두 같은 순위를 반환. 하지만 점수 **분포**가 다름:
- Cosine/Dot: 1위와 2위의 gap이 큼 (0.22)
- L1: 1위와 2위의 gap이 더 큼 (0.8) — L1이 거리 기반에서는 구분력이 더 좋을 수 있음

---

## 13. 정렬 알고리즘과 Top-k 최적화

### 13.1 현재 구현: 전체 정렬

$O(N \log N)$ 비교 정렬 후 상위 $k$개 추출. 단순하지만 최적은 아님.

### 13.2 최적화 가능성: 부분 정렬

$k \ll N$일 때 전체 정렬은 낭비. 대안:

1. **Min/Max-Heap**: $O(N \log k)$ — 크기 $k$의 힙 유지
2. **Quickselect**: $O(N)$ 평균 — k번째 원소만 찾기
3. **Introselect**: $O(N)$ 최악 — 실용적 구현

현재 교육 목적의 구현에서는 전체 정렬로 충분. $N < 10^4$에서 차이 미미.

---

## 14. Phase 연결

```
Step 69 (Word2Vec)          → 단어 벡터
Step 70 (Sentence Embedding) → 문장 벡터 (encode)
Step 71 (Brute Force)        → 벡터 검색 (exact) ← 지금 여기
Step 72 (IVF)               → 공간 분할 (approximate)
Step 73 (PQ)                → 벡터 압축 (메모리 절약)
Step 74 (HNSW)              → 그래프 탐색 (고속)
Step 75 (통합)              → Recall 비교 + 하이브리드 인덱스
```

Brute Force의 결과는 이후 모든 approximate 알고리즘의 **ground truth**. Step 72~75에서 `Recall@k`를 계산할 때 항상 brute force 결과와 비교.
