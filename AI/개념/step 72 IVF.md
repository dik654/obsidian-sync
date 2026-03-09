# Step 72: IVF (Inverted File Index)

Phase 4 (벡터 검색)의 두 번째 스텝. Brute force의 O(N·D)를 O(nprobe·N/K·D)로 줄이는 공간 분할 기반 근사 검색.

**핵심 아이디어**: 벡터 공간을 K개 영역으로 나누고, 쿼리와 가까운 영역만 탐색.

---

## 1. 왜 Brute Force로는 부족한가?

| N (벡터 수) | D (차원) | 쿼리당 연산 | 100ms 기준 필요 처리량 |
|---|---|---|---|
| $10^4$ | 768 | $7.68 \times 10^6$ | ~77 MFLOPS |
| $10^6$ | 768 | $7.68 \times 10^8$ | ~7.68 GFLOPS |
| $10^9$ | 768 | $7.68 \times 10^{11}$ | ~7.68 TFLOPS |

$N$이 커지면 선형 스캔은 실용적이지 않다. **검색 대상을 줄이는** 것이 핵심.

접근법은 크게 3가지:
1. **공간 분할**: 벡터를 클러스터로 나누고 일부만 탐색 (**IVF** — 이번 스텝)
2. **벡터 압축**: 벡터를 짧은 코드로 압축하여 거리 계산 비용 절감 (PQ — Step 73)
3. **그래프 탐색**: 이웃 그래프를 탐색하며 점진적으로 가까운 벡터로 이동 (HNSW — Step 74)

---

## 2. Voronoi 분할과 IVF의 직관

### 2.1 Voronoi 다이어그램

$K$개의 중심점(centroid) $\{c_1, \ldots, c_K\}$가 주어지면, 공간은 $K$개의 **Voronoi cell**로 분할된다:

$$
V_j = \left\{ \mathbf{x} \in \mathbb{R}^D \mid j = \arg\min_{i} \|\mathbf{x} - \mathbf{c}_i\|_2 \right\}
$$

각 점은 가장 가까운 중심에 속하는 영역에 배치된다.

### 2.2 역인덱스 (Inverted Index)

**Forward index**: 벡터 → 소속 클러스터 (assignment)
**Inverted index**: 클러스터 → 소속 벡터 리스트

```
Cluster 0: [vec_3, vec_7, vec_12, ...]
Cluster 1: [vec_0, vec_5, vec_9, ...]
...
Cluster K-1: [vec_2, vec_8, ...]
```

이름의 유래: 정보 검색(IR)에서 "문서 → 단어" 대신 "단어 → 문서 리스트"를 저장하는 inverted index와 동일한 구조.

### 2.3 IVF 검색 흐름

```
Query q
  ↓
1. Coarse search: K개 centroid 중 nprobe개 가장 가까운 것 선택
   cost: O(K·D)
  ↓
2. Candidate collection: nprobe개 cluster의 inverted list 합집합
   평균 후보 수: nprobe · N/K
  ↓
3. Fine search: 후보 벡터들만 정밀 거리 계산 → top-k
   cost: O(nprobe · N/K · D)
  ↓
Result: top-k (index, score, label)
```

---

## 3. K-means 클러스터링: Lloyd's 알고리즘

### 3.1 목적 함수

K-means는 다음 목적 함수를 최소화:

$$
J = \sum_{i=1}^{N} \|\mathbf{x}_i - \mathbf{c}_{a(i)}\|_2^2
$$

여기서 $a(i) = \arg\min_j \|\mathbf{x}_i - \mathbf{c}_j\|_2$는 벡터 $i$의 클러스터 할당.

### 3.2 Lloyd's 알고리즘 (EM 해석)

K-means는 **좌표 하강법(coordinate descent)**으로 볼 수 있다:

**E-step (Assign)**: 중심 고정, 각 벡터를 가장 가까운 중심에 할당

$$
a(i) \leftarrow \arg\min_j \|\mathbf{x}_i - \mathbf{c}_j\|_2^2
$$

**M-step (Update)**: 할당 고정, 각 중심을 멤버의 평균으로 갱신

$$
\mathbf{c}_j \leftarrow \frac{1}{|S_j|} \sum_{i \in S_j} \mathbf{x}_i, \quad S_j = \{i : a(i) = j\}
$$

### 3.3 수렴 증명

**정리**: Lloyd's 알고리즘은 유한 스텝 내에 수렴한다.

**증명 스케치**:
1. E-step에서 $J$는 감소하거나 유지된다 (각 벡터가 더 가까운 중심으로 이동하므로)
2. M-step에서 $J$는 감소하거나 유지된다 (평균은 제곱 거리 합의 최소화자이므로)
3. $J$는 하한(0)이 있고, 가능한 할당의 수는 유한($K^N$)하다
4. 따라서 유한 스텝 후 할당이 변하지 않으면 수렴

**주의**: 전역 최적(global optimum)이 아닌 **지역 최적(local optimum)**으로 수렴. 초기화에 의존.

### 3.4 M-step이 최적인 이유

클러스터 $j$에 속한 점들 $\{x_i : i \in S_j\}$에 대해:

$$
\frac{\partial}{\partial \mathbf{c}_j} \sum_{i \in S_j} \|\mathbf{x}_i - \mathbf{c}_j\|^2 = -2 \sum_{i \in S_j} (\mathbf{x}_i - \mathbf{c}_j) = 0
$$

$$
\implies \mathbf{c}_j = \frac{1}{|S_j|} \sum_{i \in S_j} \mathbf{x}_i
$$

→ 평균이 제곱 거리 합의 유일한 최소화자.

### 3.5 초기화 전략

| 방법 | 설명 | 복잡도 |
|------|------|--------|
| Random | N개 중 K개 무작위 선택 | $O(K)$ |
| K-means++ | 거리 비례 확률로 순차 선택 | $O(NKD)$ |
| Faiss | 여러 랜덤 초기화 중 최소 $J$ 선택 | $O(T \cdot \mathrm{iter} \cdot NKD)$ |

현재 구현은 **Random** (Fisher-Yates shuffle로 K개 선택). 교육 목적에 충분.

### 3.6 K-means++: 왜 더 나은가?

Arthur & Vassilvitskii (2007). 핵심 아이디어: **이미 선택된 중심에서 먼 점일수록 다음 중심으로 선택될 확률이 높다.**

**알고리즘:**
1. 첫 중심 $c_1$을 균등 랜덤으로 선택
2. $i = 2, \ldots, K$에 대해:
   - 각 점 $\mathbf{x}$에 대해 가장 가까운 기존 중심까지의 거리 $d(\mathbf{x})$ 계산
   - $\mathbf{x}$를 확률 $\frac{d(\mathbf{x})^2}{\sum_{\mathbf{x}'} d(\mathbf{x}')^2}$로 다음 중심으로 선택

**이론적 보장:**

$$
\mathbb{E}[J_{\mathrm{kmeans++}}] \leq 8(\ln K + 2) \cdot J_{\mathrm{opt}}
$$

즉 K-means++의 초기화만으로도 최적의 $O(\log K)$배 이내가 보장된다. Random 초기화에는 이런 보장이 없다.

**직관**: 고르게 퍼진 초기 중심 → 각 클러스터가 데이터의 서로 다른 영역을 커버 → 더 균형잡힌 분할 → 더 빠른 수렴.

### 3.7 K-means와 GMM의 관계

K-means는 **Gaussian Mixture Model(GMM)**의 특수 경우:

| | K-means | GMM |
|---|---|---|
| 할당 | Hard ($a(i) \in \{1, \ldots, K\}$) | Soft ($\gamma_{ij} \in [0, 1]$) |
| 클러스터 모양 | 구형 (spherical) | 타원형 (ellipsoidal) |
| 파라미터 | 중심 $\mathbf{c}_j$ | 평균 $\boldsymbol{\mu}_j$, 공분산 $\boldsymbol{\Sigma}_j$, 혼합 가중치 $\pi_j$ |
| 최적화 | Coordinate descent | EM 알고리즘 |

GMM에서 모든 공분산을 $\sigma^2 \mathbf{I}$로 고정하고 $\sigma \to 0$으로 보내면 K-means와 동일:

$$
\gamma_{ij} = \frac{\pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \sigma^2 \mathbf{I})}{\sum_k \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \sigma^2 \mathbf{I})} \xrightarrow{\sigma \to 0} \begin{cases} 1 & \text{if } j = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\| \\ 0 & \text{otherwise} \end{cases}
$$

→ Soft assignment가 hard assignment로 전이. IVF에서는 hard assignment의 단순함이 검색 효율에 유리.

### 3.8 복잡도

| 단계 | 비용 |
|------|------|
| 초기화 | $O(K)$ |
| E-step 1회 | $O(NKD)$ |
| M-step 1회 | $O(ND)$ |
| 전체 (T iterations) | $O(T \cdot NKD)$ |

---

## 4. nprobe와 Recall 트레이드오프

### 4.1 경계 분석

이상적으로 균등한 클러스터($|S_j| = N/K$)에서:

- **nprobe=1**: 후보 $N/K$개 → 탐색 비용 $1/K$, recall은 데이터 분포에 의존
- **nprobe=K**: 후보 $N$개 → brute force, recall = 1.0
- **nprobe=p**: 평균 후보 $pN/K$개 → 탐색 비용 $p/K$

### 4.2 Recall의 확률론적 분석

쿼리 $q$의 진짜 최근접 이웃(true NN)이 클러스터 $j^*$에 속하고, 쿼리가 centroid 거리 순으로 $r$번째로 가까운 클러스터에 속할 확률을 $P(r)$이라 하면:

$$
\mathrm{Recall@1}(\mathrm{nprobe}=p) = \sum_{r=1}^{p} P(r)
$$

true NN이 가장 가까운 centroid의 클러스터에 있을 확률이 높을수록 ($P(1) \to 1$), 작은 nprobe로도 높은 recall.

### 4.3 IVF 경계(boundary) 문제

Voronoi cell의 **경계 근처**에 있는 쿼리는 true NN이 인접 클러스터에 있을 수 있다:

```
        c₁       c₂
    ●───┼────|────┼───●
        |  q ← boundary
        |    근처 쿼리
```

$q$가 $c_1$에 가장 가깝지만, true NN이 $c_2$ 쪽에 있을 수 있다.
→ nprobe > 1이 필요한 근본 원인.

### 4.4 실험 결과 (Step 72 테스트)

```
N=200, D=8, K=5, k=10:
  nprobe=1: recall@10=0.50
  nprobe=2: recall@10=0.80
  nprobe=3: recall@10=1.00
  nprobe=5: recall@10=1.00

N=500, D=16, K=10, k=10:
  nprobe=3:  avg recall@10=0.680
  nprobe=10: avg recall@10=1.000
```

nprobe를 2배로 올리면 recall이 크게 향상. 일반적으로 $\mathrm{nprobe} \approx \sqrt{K}$가 좋은 시작점.

### 4.5 Recall@k의 이론적 하한

균등 분포 가정에서, 쿼리의 true NN이 같은 Voronoi cell에 있을 확률을 분석할 수 있다.

**1차원 단순화**: $[0, 1]$ 구간을 $K$개 균등 구간으로 나누면, 쿼리와 true NN이 같은 구간에 있을 확률:

$$
P(\text{same cell}) \approx 1 - \frac{2\delta}{1/K} = 1 - 2K\delta
$$

여기서 $\delta$는 쿼리와 NN 사이의 거리. $\delta$가 작을수록(NN이 가까울수록) 같은 cell에 있을 확률이 높다.

**고차원으로 확장**: $D$차원에서 Voronoi cell의 "표면적/부피" 비율이 $D$에 따라 증가하므로, 경계 효과가 더 심해진다:

$$
P(\text{boundary miss}) \propto D^{1/2}
$$

→ 차원이 높을수록 같은 cell에 있어도 경계 근처일 확률이 높아 nprobe를 더 키워야 한다. 이것이 고차원 벡터 검색에서 nprobe를 넉넉히 잡는 실용적 이유.

### 4.6 Residual 기반 거리 계산

IVF의 중요한 최적화: **residual vector** 사용.

벡터 $\mathbf{x}$가 centroid $\mathbf{c}_j$에 할당되면, 잔차(residual)를 저장:

$$
\mathbf{r} = \mathbf{x} - \mathbf{c}_j
$$

쿼리 $\mathbf{q}$와의 L2 거리:

$$
\|\mathbf{q} - \mathbf{x}\|^2 = \|\mathbf{q} - \mathbf{c}_j - \mathbf{r}\|^2 = \|(\mathbf{q} - \mathbf{c}_j) - \mathbf{r}\|^2
$$

$(\mathbf{q} - \mathbf{c}_j)$는 해당 클러스터의 모든 벡터에 대해 동일 → 한 번만 계산. 이 최적화는 IVF-PQ(Step 75)에서 핵심이 된다: residual을 PQ로 압축하면 원본보다 정확도가 높다(coarse quantization 오차가 이미 제거되었으므로).

---

## 5. K 선택: 클러스터 수의 영향

### 5.1 이론적 가이드라인

Faiss 권장: $K \approx \sqrt{N}$ ~ $4\sqrt{N}$

| N | $\sqrt{N}$ | $4\sqrt{N}$ | 권장 K 범위 |
|---|---|---|---|
| $10^4$ | 100 | 400 | 100~400 |
| $10^6$ | 1,000 | 4,000 | 1K~4K |
| $10^8$ | 10,000 | 40,000 | 10K~40K |

### 5.2 K의 트레이드오프

**K가 너무 작으면** ($K \ll \sqrt{N}$):
- 각 클러스터가 커서 nprobe=1이어도 많은 벡터 탐색 → 속도 이점 감소
- recall은 높지만 속도가 느림

**K가 너무 크면** ($K \gg 4\sqrt{N}$):
- coarse search 자체가 비용: $O(KD)$
- 일부 클러스터가 비거나 극소 → 비효율적
- K-means 학습 비용 증가

### 5.3 Elbow Method (경험적)

$K$를 증가시키면서 $J$(within-cluster sum of squares)를 관찰. $J$의 감소율이 급격히 둔화되는 "팔꿈치" 지점이 적절한 $K$.

### 5.4 최적 K의 속도 분석

IVF의 총 쿼리 비용을 $K$의 함수로 표현:

$$
T(K) = \underbrace{K \cdot D}_{\text{coarse search}} + \underbrace{\mathrm{nprobe} \cdot \frac{N}{K} \cdot D}_{\text{fine search}}
$$

$\mathrm{nprobe}$를 상수로 고정하고 $T(K)$를 $K$에 대해 미분:

$$
\frac{dT}{dK} = D - \mathrm{nprobe} \cdot \frac{N \cdot D}{K^2} = 0
$$

$$
K^* = \sqrt{\mathrm{nprobe} \cdot N}
$$

**예시**: $N = 10^6$, $\mathrm{nprobe} = 10$ → $K^* = \sqrt{10^7} \approx 3162$. Faiss 권장 범위 $1000 \sim 4000$과 일치!

이 유도는 coarse search와 fine search의 비용이 **균형을 이루는** 지점이 최적이라는 직관을 제공.

### 5.5 K와 Recall의 관계

$K$가 커지면:
- 각 Voronoi cell이 작아져 같은 cell 내 벡터 간 거리가 줄어듦 → cell 내부의 정밀도 ↑
- 하지만 경계 효과도 증가: 인접 cell에 true NN이 있을 확률 ↑
- 결과적으로 같은 recall을 위해 nprobe를 더 키워야 함

최적점: $K \cdot (\mathrm{nprobe}/K)$가 일정할 때, 즉 $\mathrm{nprobe} \propto K$에서 recall이 일정하게 유지.

---

## 6. IVF의 한계와 확장

### 6.1 한계

1. **클러스터 불균형**: 데이터 분포가 비균등하면 일부 클러스터에 벡터 집중 → nprobe 효율 저하
2. **차원의 저주**: 고차원에서 Voronoi cell이 효과적으로 공간을 분할하지 못함
3. **벡터 저장**: 원본 벡터를 모두 메모리에 저장 → $O(ND)$ 메모리

### 6.2 IVF-PQ: 메모리 절약

IVF와 PQ(Product Quantization)의 결합:
- IVF로 후보 축소 → PQ로 벡터 압축
- 메모리: $O(ND) \to O(NM\log K')$ (M: 서브벡터 수, K': 코드북 크기)
- Step 73에서 PQ 단독 구현 후, 통합은 Step 75에서

### 6.3 Multi-probe LSH vs IVF

| | IVF | Multi-probe LSH |
|---|---|---|
| 공간 분할 | K-means (데이터 적응형) | 해시 함수 (데이터 무관) |
| 학습 비용 | $O(TNKD)$ | $O(ND)$ |
| 쿼리 비용 | $O(KD + pN/KD)$ | $O(LD)$ |
| 장점 | 데이터에 맞춤 | 이론적 보장 |
| 단점 | 학습 필요 | 고차원에서 비효율 |

### 6.4 IVF의 갱신 문제

**정적(static) 인덱스**: 한 번 구축 후 변경 없음 → IVF 최적
**동적(dynamic) 인덱스**: 벡터 추가/삭제가 빈번 → 문제 발생

| 연산 | 비용 | 문제 |
|------|------|------|
| 추가 | $O(KD)$ | 기존 centroid 기준으로 할당. 분포 변화 시 centroid가 stale해짐 |
| 삭제 | $O(1)$ | tombstone 마킹. 누적되면 inverted list가 비효율적 |
| 재학습 | $O(TNKD)$ | 전체 데이터로 k-means 재실행. 비용이 큼 |

**실용적 해결책**:
1. **Periodic retrain**: 일정 수의 추가/삭제 후 재학습
2. **Sliding window**: 최근 데이터만 유지, 오래된 데이터 삭제
3. **Merge index**: 작은 버퍼 인덱스 + 큰 메인 인덱스, 주기적 병합 (LSM-tree 패턴)

---

## 7. K-means의 NP-hardness와 근사 비율

### 7.1 K-means는 NP-hard

**정리** (Aloise et al., 2009): 일반적인 K-means 문제 (최적 클러스터링 찾기)는 NP-hard.

심지어 $K = 2$인 경우에도 NP-hard (Dasgupta, 2008). 따라서 Lloyd's 알고리즘은 **휴리스틱**이며, 전역 최적을 보장하지 않는다.

### 7.2 근사 보장

다양한 초기화 전략의 근사 비율:

| 방법 | 근사 비율 | 비고 |
|------|-----------|------|
| Arbitrary init | 보장 없음 | 임의로 나쁠 수 있음 |
| K-means++ | $O(\log K)$ | 기대값 기준 |
| Local search | $9 + \epsilon$ | Kanungo et al. (2004) |
| 최적 | 1.0 | NP-hard로 다항 시간 불가 |

### 7.3 왜 Lloyd's가 실전에서 잘 되는가?

이론적 보장이 약함에도 Lloyd's가 실용적인 이유:

1. **Smoothed analysis** (Arthur & Vassilvitskii, 2006): 입력에 작은 가우시안 노이즈를 추가하면, Lloyd's는 다항 시간에 수렴 ($O(n^{34} k^{34} d^8 \sigma^{-6})$, 이론적이지만 worst-case가 드묾을 의미)
2. **실데이터의 구조**: 랜덤 데이터와 달리 실제 데이터에는 자연스러운 클러스터 구조가 있어, local optimum이 global에 가까움
3. **다중 실행(multi-restart)**: 여러 랜덤 초기화로 실행 후 최소 $J$를 선택하면 실용적으로 충분

---

## 8. 구현 코드 발췌

### 8.1 K-means

```rust
pub fn kmeans(vectors: &[&[f64]], k: usize, max_iter: usize, seed: u64)
    -> (Vec<Vec<f64>>, Vec<usize>)
{
    // Fisher-Yates로 k개 초기 중심 선택
    // Lloyd's: assign(L2) → update(mean) → repeat
    // 할당 변화 없으면 조기 종료
}
```

### 8.2 IVF 핵심: search

```rust
pub fn search(&self, query: &[f64], k: usize, nprobe: usize, metric: Metric)
    -> Vec<(usize, f64, String)>
{
    // 1. Coarse: K개 centroid 중 nprobe개 가까운 것 (항상 L2)
    let mut centroid_dists: Vec<(usize, f64)> = self.centroids.iter()
        .enumerate()
        .map(|(i, c)| (i, l2_distance(query, c)))
        .collect();
    centroid_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // 2. 후보 수집: nprobe개 cluster의 벡터 인덱스
    let mut candidates: Vec<usize> = Vec::new();
    for &(cluster_id, _) in centroid_dists.iter().take(nprobe) {
        candidates.extend_from_slice(&self.inverted_lists[cluster_id]);
    }

    // 3. Fine search: 후보만 user metric으로 정밀 검색
    // 4. Sort + top-k
}
```

### 8.3 Coarse 메트릭이 항상 L2인 이유

K-means의 목적 함수가 L2 기반이므로, Voronoi cell의 경계가 L2 거리로 정의됨. Coarse search에서 다른 메트릭을 쓰면 cell 할당과 불일치 → recall 하락.

Fine search에서는 user가 원하는 메트릭(cosine, dot 등) 사용 가능 — cell 내부에서의 정렬은 cell 할당에 영향 없으므로.

---

## 9. 클러스터 균형과 부하 분산

### 9.1 이상적 vs 현실

이상적 균등 분할: $|S_j| = N/K$ for all $j$.
현실: 데이터 분포에 따라 크기 편차 발생.

Step 72 테스트 결과:
```
IVF train+add: 30 vectors, 3 clusters, sizes=[3, 20, 7]
```

20개가 한 클러스터에 집중 → 해당 클러스터를 probe하면 비용이 높음.

### 9.2 균형 개선 전략

1. **K-means++**: 초기 중심을 더 잘 선택하여 균등 분할 유도
2. **Balanced K-means**: 클러스터 크기에 상한 제약 추가
3. **2-level IVF**: 큰 클러스터를 다시 서브 클러스터로 분할

---

## 10. IVF와 정보 이론

### 10.1 Vector Quantization (VQ) 관점

IVF의 coarse quantizer는 사실 **vector quantizer**:

$$
q(\mathbf{x}) = \mathbf{c}_{j^*}, \quad j^* = \arg\min_j \|\mathbf{x} - \mathbf{c}_j\|_2
$$

입력 벡터 $\mathbf{x} \in \mathbb{R}^D$를 $K$개 코드워드 중 하나로 매핑.

정보량: $\log_2 K$ bits (K=1024이면 10 bits).

### 10.2 Rate-Distortion 관점

K-means의 목적 함수 $J$는 **왜곡(distortion)**:

$$
J = \mathbb{E}\left[\|\mathbf{X} - q(\mathbf{X})\|^2\right]
$$

Shannon의 rate-distortion 이론에 의해, 주어진 비트수 $R = \log_2 K$에서 달성 가능한 최소 왜곡에는 하한이 존재. K-means는 이 하한에 접근하는 실용적 방법.

---

## 11. 실제 시스템에서의 IVF

### 11.1 Faiss (Facebook AI Similarity Search)

Faiss의 `IndexIVFFlat`이 우리 구현과 동일한 구조:
- `train()`: K-means로 centroid 학습
- `add()`: 벡터를 가장 가까운 centroid의 inverted list에 추가
- `search()`: nprobe 클러스터 탐색

Faiss는 추가로:
- GPU 가속 (CUDA K-means, CUDA search)
- `IndexIVFPQ`: IVF + Product Quantization
- `IndexIVFScalarQuantizer`: IVF + 스칼라 양자화
- Multi-GPU 분산 검색

### 11.2 실제 성능 벤치마크

Faiss on SIFT1M (100만 개 128차원):

| 설정 | QPS | Recall@1 |
|------|-----|----------|
| Flat (brute force) | 400 | 1.000 |
| IVF1024, nprobe=1 | 8,000 | 0.385 |
| IVF1024, nprobe=8 | 4,000 | 0.756 |
| IVF1024, nprobe=64 | 1,500 | 0.962 |
| IVF1024, nprobe=256 | 600 | 0.997 |

nprobe 8배 증가 → QPS 절반, recall 크게 향상.

### 11.3 벡터 DB별 IVF 변형

| 시스템 | IVF 변형 | 특징 |
|--------|---------|------|
| Milvus | IVF_FLAT, IVF_SQ8, IVF_PQ | 스칼라/프로덕트 양자화 조합 |
| Qdrant | HNSW 기본, IVF 미사용 | 그래프 기반이 기본 |
| Pinecone | 비공개 | 내부적으로 IVF 변형 사용 추정 |
| Weaviate | HNSW + PQ | IVF 대신 그래프 + 압축 |

최근 추세: **HNSW가 IVF를 대체**하는 경향. 하지만 10억 규모에서는 IVF-PQ가 메모리 효율에서 여전히 우위.

---

## 12. Voronoi 경계의 수학

### 12.1 경계면의 수학적 정의

두 centroid $\mathbf{c}_i$, $\mathbf{c}_j$ 사이의 Voronoi 경계는 **수직 이등분 초평면(perpendicular bisector hyperplane)**:

$$
H_{ij} = \left\{ \mathbf{x} : \|\mathbf{x} - \mathbf{c}_i\| = \|\mathbf{x} - \mathbf{c}_j\| \right\}
$$

이를 전개하면:

$$
\|\mathbf{x}\|^2 - 2\mathbf{x} \cdot \mathbf{c}_i + \|\mathbf{c}_i\|^2 = \|\mathbf{x}\|^2 - 2\mathbf{x} \cdot \mathbf{c}_j + \|\mathbf{c}_j\|^2
$$

$$
2\mathbf{x} \cdot (\mathbf{c}_j - \mathbf{c}_i) = \|\mathbf{c}_j\|^2 - \|\mathbf{c}_i\|^2
$$

$$
\mathbf{x} \cdot (\mathbf{c}_j - \mathbf{c}_i) = \frac{\|\mathbf{c}_j\|^2 - \|\mathbf{c}_i\|^2}{2}
$$

→ 법선 벡터 $\mathbf{n} = \mathbf{c}_j - \mathbf{c}_i$, 오프셋 $b = \frac{\|\mathbf{c}_j\|^2 - \|\mathbf{c}_i\|^2}{2}$인 초평면.

### 12.2 경계 근처 쿼리의 판별

쿼리 $\mathbf{q}$가 centroid $\mathbf{c}_i$에 가장 가깝고, 인접 centroid $\mathbf{c}_j$까지의 거리 차이:

$$
\Delta_{ij} = \|\mathbf{q} - \mathbf{c}_j\| - \|\mathbf{q} - \mathbf{c}_i\|
$$

$\Delta_{ij}$가 작을수록 경계에 가까움. **적응적 nprobe**: $\Delta_{ij} < \theta$인 클러스터도 탐색에 포함하는 전략.

이 아이디어의 구현:
```
sorted_centroids = sort_by_distance(query, centroids)
nprobe_adaptive = count where (dist[i] - dist[0]) < threshold
```

Faiss의 `IndexIVF`에는 이런 적응적 전략이 없지만, 연구 문헌에서 제안됨 (Li et al., 2020).

### 12.3 고차원에서 Voronoi cell의 모양

$D$가 커지면 Voronoi cell의 기하학적 성질이 변한다:

- **2D**: 볼록 다각형, 직관적
- **고차원**: 볼록 다면체(polytope), 대부분의 부피가 "표면" 근처에 집중

구체적으로, $D$차원 단위 구에서 랜덤 점의 중심까지 거리:

$$
\|\mathbf{x}\|_2 \sim \sqrt{D} \cdot \left(1 + O\left(\frac{1}{D}\right)\right)
$$

→ 고차원에서 모든 점이 구의 표면 근처에 집중 (concentration of measure). Voronoi cell의 "내부"가 거의 비어있고 대부분이 경계 근처 → IVF의 경계 문제가 심화.

---

## 13. Fisher-Yates Shuffle: 편향 없는 랜덤 선택

### 13.1 알고리즘

K-means의 초기 centroid 선택에서 사용한 Fisher-Yates shuffle:

```
indices = [0, 1, 2, ..., N-1]
for i in 0..k:
    j = random(i, N-1)    // i ≤ j ≤ N-1
    swap(indices[i], indices[j])
// indices[0..k]가 편향 없는 k개 선택
```

### 13.2 편향 없음의 증명

**정리**: Fisher-Yates shuffle에서 각 원소가 위치 $i < k$에 올 확률은 정확히 $\frac{1}{N}$이다.

**증명**: 원소 $x$가 위치 0에 올 확률:
- 1단계에서 $j = \mathrm{pos}(x)$가 선택될 확률: $\frac{1}{N}$

원소 $x$가 위치 1에 올 확률:
- 1단계에서 선택되지 않을 확률: $\frac{N-1}{N}$
- 2단계에서 선택될 확률: $\frac{1}{N-1}$
- 곱: $\frac{N-1}{N} \cdot \frac{1}{N-1} = \frac{1}{N}$

일반적으로 위치 $i$에 올 확률:

$$
P(\text{위치 } i) = \prod_{t=0}^{i-1} \frac{N-1-t}{N-t} \cdot \frac{1}{N-i} = \frac{(N-1)!/(N-1-i)!}{N!/(N-i)!} \cdot \frac{1}{N-i}
$$

$$
= \frac{(N-i)!}{N!} \cdot \frac{(N-1)!}{(N-1-i)!} \cdot \frac{1}{N-i} = \frac{1}{N}
$$

→ **정확히 균등 분포**. 나이브한 `random(0, N-1)` 반복 선택은 중복이 발생하고 편향될 수 있음.

---

## 14. Phase 연결

```
Step 71 (Brute Force) → O(ND), exact, ground truth
Step 72 (IVF)         → O(pN/K·D), 공간 분할 ← 지금 여기
Step 73 (PQ)          → 벡터 압축, 메모리 N·D → N·M·log(K')
Step 74 (HNSW)        → O(D·log N), 그래프 탐색
Step 75 (통합)        → IVF-PQ, Recall 비교, 하이브리드 인덱스
```

IVF는 **검색 대상을 줄이는** 전략. PQ는 **벡터 크기를 줄이는** 전략. 둘을 결합한 IVF-PQ가 대규모 실용 시스템의 표준.
