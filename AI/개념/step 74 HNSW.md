# Step 74: HNSW (Hierarchical Navigable Small World)

Phase 4 (벡터 검색)의 마지막 스텝. 다층 그래프 기반 근사 최근접 이웃 검색.

## 1. 핵심 아이디어

**문제**: Brute Force는 $O(ND)$, IVF는 $O(N/K \cdot D)$, PQ는 메모리 절약 — 하지만 모두 선형 스캔 기반.

**HNSW 해법**: 다층 navigable small world 그래프를 구축하여 $O(\log N \cdot D)$ 검색.

핵심 통찰:
- **Skip List 구조**: 상위 레이어는 장거리 점프 (highway), 하위 레이어는 세밀한 탐색
- **Navigable Small World**: 각 레이어가 NSW 그래프 — greedy routing이 $O(\log N)$ 홉으로 수렴
- **Train-free**: IVF/PQ와 달리 사전 학습 불필요, 벡터를 삽입하며 그래프가 점진적으로 성장

```
Layer 3:  [EP] ─────────────────── [A]                (희소, 장거리)
Layer 2:  [EP] ──── [B] ──── [A] ──── [C]             (중간)
Layer 1:  [EP] ─ [D] ─ [B] ─ [E] ─ [A] ─ [C] ─ [F]   (밀집)
Layer 0:  모든 노드 연결 (M_max0 = 2M)                 (완전 밀집)
```

## 2. NSW (Navigable Small World)에서 HNSW로

### 2.1 NSW의 한계

단일 NSW 그래프에서 greedy search의 복잡도를 유도한다.

$d$차원 공간에 $N$개 벡터가 균일 분포할 때, 쿼리 $q$에서 시작하여 greedy로 목표 $t$에 도달하는 과정을 생각하자. 각 greedy 홉에서 현재 노드의 이웃 중 $t$에 가장 가까운 노드로 이동한다.

**홉당 진행 거리 분석**:
- 현재 노드 $v$에서 $t$까지 거리 $r = d(v, t)$
- $v$의 이웃 $M$개 중 $t$에 더 가까운 이웃이 존재할 확률은, 반지름 $r$인 구와 $v$ 주변 이웃의 기하학적 배치에 의존
- $d$차원에서 반지름 $r$ 구의 부피 $\propto r^d$ → 한 홉으로 거리가 $r \cdot (1 - 1/M)^{1/d}$ 배 줄어듦

**총 홉 수**: 초기 거리 $R$에서 목표 근처 $\epsilon$까지:

$$\mathrm{hops} \approx \frac{\ln(R/\epsilon)}{\ln\left(\frac{1}{1 - M^{-1/d}}\right)} \approx \frac{d \ln(R/\epsilon)}{\ln M}$$

$R/\epsilon \sim N^{1/d}$ (균일 분포에서 최원점/최근점 비)이므로:

$$T_{\mathrm{search}}^{\mathrm{NSW}} = O\left(\frac{d \cdot N^{1/d}}{\ln M} \cdot M \cdot D\right)$$

고차원에서 $d$가 커지면 $N^{1/d} \to 1$이지만 $d$ 인수 자체가 커져서, 실질적으로 $O(N^{1/d} \cdot D)$에 수렴. 이것이 단일 NSW의 근본적 한계 — 계층 구조 없이는 sublinear를 달성할 수 없다.

### 2.2 HNSW의 해법: 계층 구조

Skip List에서 영감: $L$개 레이어를 쌓아 각 레이어에서 $O(\log_{1/p} N)$ 홉.

노드 $v$의 최대 레이어:
$$l(v) = \lfloor -\ln(\mathrm{uniform}(0,1)) \cdot m_L \rfloor, \quad m_L = \frac{1}{\ln M}$$

레이어 $l$의 기대 노드 수:
$$\mathbb{E}[|V_l|] = N \cdot \left(\frac{1}{M}\right)^l$$

**복잡도 유도**: 총 레이어 수 $L = \log_M N$. 각 레이어에서 greedy search는 그 레이어의 노드 수 $N_l$에 대해 NSW 검색을 수행한다. 핵심은 각 레이어에서 몇 홉이 필요한가:

레이어 $l$에서 $l-1$로 내려갈 때, 레이어 $l$의 "커버리지 반경"은 $N_l^{-1/d}$에 비례. 레이어 $l$에서 greedy로 이 반경 내로 진입하는데 필요한 홉:

$$\mathrm{hops}_l = O\left(\frac{1}{\ln M}\right) \quad \mathrm{(each\ layer:\ constant\ hops)}$$

이유: 레이어 $l$의 노드는 레이어 $l+1$ 대비 $M$배 밀집. Skip List에서 각 레벨당 $O(1/p) = O(M)$ 홉과 동일한 원리. 총:

$$T_{\mathrm{search}}^{\mathrm{HNSW}} = \sum_{l=0}^{L} O(M \cdot D) = O(L \cdot M \cdot D) = O(\log_M N \cdot M \cdot D)$$

$M$이 상수이므로 $O(\log N \cdot D)$. 더 정밀하게 적으면 $O\left(\frac{M}{\ln M} \cdot \ln N \cdot D\right)$.

## 3. 삽입 알고리즘

### 3.1 전체 흐름

새 벡터 $q$를 삽입:

1. **레이어 할당**: $l \leftarrow \lfloor -\ln(u) \cdot m_L \rfloor$
2. **Phase 1 (상위 진입)**: 최상위 레이어 $L$ → $l+1$ — greedy search ($\mathrm{ef}=1$)
3. **Phase 2 (이웃 연결)**: 레이어 $\min(l, L)$ → $0$ — $\mathrm{ef}_{\mathrm{construction}}$ 빔서치, 양방향 연결
4. **Entry point 갱신**: $l > L$이면 $q$가 새 entry point

### 3.2 양방향 연결과 Pruning

레이어 $lc$에서 $q$와 이웃 $n$을 연결할 때:
- $\mathrm{neighbors}[q][lc] \leftarrow \mathrm{neighbors}[q][lc] \cup \{n\}$
- $\mathrm{neighbors}[n][lc] \leftarrow \mathrm{neighbors}[n][lc] \cup \{q\}$

$n$의 이웃 수가 $M_{\max}$를 초과하면:
$$\mathrm{neighbors}[n][lc] \leftarrow \mathrm{SelectNeighbors}(\mathrm{neighbors}[n][lc], M_{\max})$$

Layer 0: $M_{\max 0} = 2M$ (더 많은 연결로 recall 보장)
Layer 1+: $M_{\max} = M$

### 3.3 구현 코드

```rust
pub fn add(&mut self, vector: &[f64], label: &str) {
    let node_id = self.vectors.len();
    let level = self.random_level();
    self.neighbors.push(vec![Vec::new(); level + 1]);

    // 첫 번째 노드: entry point
    if self.entry_point.is_none() {
        self.entry_point = Some(node_id);
        self.max_level = level;
        return;
    }

    // Phase 1: 최상위 → level+1 — greedy (ef=1)
    let mut current_ep = self.entry_point.unwrap();
    for lc in ((level + 1)..=self.max_level).rev() {
        let result = self.search_layer(vector, &[current_ep], 1, lc, Metric::L2);
        if !result.is_empty() { current_ep = result[0].1; }
    }

    // Phase 2: 이웃 연결 (양방향 + pruning)
    for lc in (0..=level.min(self.max_level)).rev() {
        let m_max = if lc == 0 { self.m_max0 } else { self.m };
        let candidates = self.search_layer(vector, &[current_ep], self.ef_construction, lc, metric);
        let selected = Self::select_neighbors(&candidates, m_max);
        // 양방향 연결 + overflow 시 pruning...
    }
}
```

## 4. 검색 알고리즘 (search_layer)

### 4.1 빔서치 핵심

`search_layer`는 HNSW의 핵심. 하나의 레이어에서 $\mathrm{ef}$개 최근접 이웃을 찾는 빔서치:

```
candidates ← min-heap (가장 가까운 것이 top)
results    ← max-heap (가장 먼 것이 top, ef 초과 시 제거)
visited    ← HashSet

초기화: entry_points → candidates, results, visited

while candidates 비어있지 않으면:
    closest ← candidates.pop()         // 가장 가까운 미탐색 노드
    farthest ← results.peek()           // 현재 결과 중 가장 먼 노드

    if dist(closest) > dist(farthest):
        break  // 남은 후보가 모두 현재 결과보다 멀면 종료

    for neighbor in neighbors[closest][layer]:
        if neighbor not in visited:
            visited.insert(neighbor)
            d ← distance(query, vectors[neighbor])

            if |results| < ef OR d < dist(farthest):
                candidates.push(neighbor, d)
                results.push(neighbor, d)
                if |results| > ef:
                    results.pop()  // 가장 먼 것 제거
```

### 4.2 쿼리 흐름

```
search(query, k, ef_search, metric):
    Phase 1: Layer L → 1 — greedy (ef=1), entry point 갱신
    Phase 2: Layer 0 — search_layer(ef=ef_search)
    Return: top-k from candidates
```

### 4.3 BinaryHeap과 FloatOrd

Rust의 `BinaryHeap`은 max-heap. `Reverse`로 min-heap 구현:

```rust
// f64에 Ord 부여 (BinaryHeap 요구사항)
struct FloatOrd(f64);
impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

// candidates: min-heap (Reverse로 가장 가까운 것이 top)
let mut candidates: BinaryHeap<Reverse<(FloatOrd, usize)>>;
// results: max-heap (가장 먼 것이 top → ef 초과 시 pop)
let mut results: BinaryHeap<(FloatOrd, usize)>;
```

## 5. 파라미터 분석

### 5.1 M (최대 이웃 수)

$M$은 그래프 밀도를 제어:

| M | Layer 0 연결 | 메모리 | 검색 속도 | Recall |
|---|---|---|---|---|
| 4 | 8 | 낮음 | 빠름 | 낮음 |
| 16 | 32 | 중간 | 중간 | 높음 |
| 48 | 96 | 높음 | 느림 | 매우 높음 |

**메모리**: 노드당 약 $M \cdot 4$ 바이트 (u32 이웃 ID) × 평균 레이어 수.

### 5.2 $\mathrm{ef}_{\mathrm{construction}}$

삽입 시 빔 폭. 클수록 좋은 이웃을 찾지만 삽입 느려짐:
- $\mathrm{ef}_{\mathrm{construction}} = M$: 최소 (빠르지만 품질↓)
- $\mathrm{ef}_{\mathrm{construction}} = 200$: 고품질 그래프 (권장)

### 5.3 $\mathrm{ef}_{\mathrm{search}}$

검색 시 빔 폭. **런타임에 조절 가능** — 이것이 HNSW의 큰 장점:

$$\mathrm{recall}(\mathrm{ef}_{\mathrm{search}}) \nearrow \quad \mathrm{as} \quad \mathrm{ef}_{\mathrm{search}} \nearrow$$

실험 결과 (N=500, D=16, M=16):
```
ef_search= 10: recall@10=0.985
ef_search= 20: recall@10=1.000
ef_search= 50: recall@10=1.000
ef_search=100: recall@10=1.000
```

### 5.4 $m_L$ (레이어 할당 계수) — 최적값 유도

$$m_L = \frac{1}{\ln M}$$

**Skip List에서의 최적 확률 유도**:

Skip List에서 각 원소가 레벨 $l$에 존재할 확률 $p$를 최적화한다. $N$개 원소, 레벨 $l$의 기대 원소 수 $N \cdot p^l$. 총 레벨 수 $L = \log_{1/p} N$.

검색 비용 = 레벨당 평균 $1/p$ 홉 × $L$ 레벨:

$$T(p) = \frac{L}{p} = \frac{\log_{1/p} N}{p} = \frac{\ln N}{p \cdot \ln(1/p)}$$

$T(p)$를 최소화. $f(p) = p \cdot \ln(1/p) = -p \ln p$로 놓고:

$$f'(p) = -\ln p - 1 = 0 \implies p^* = e^{-1} \approx 0.368$$

하지만 HNSW에서는 레이어당 이웃 수 $M$이 고정이므로, $p = 1/M$이 자연스러운 선택:
- 레이어 $l$에서 $l+1$로 승격 확률 $= 1/M$
- 레이어 $l$의 노드 수 $= N/M^l$ — 각 레이어에서 탐색 공간이 $M$배 축소

**$m_L = 1/\ln M$ 유도**:

레벨 할당: $l = \lfloor -\ln U \cdot m_L \rfloor$. 노드가 레벨 $l$ 이상에 존재할 확률:

$$P(l(v) \geq l) = P(-\ln U \cdot m_L \geq l) = P(U \leq e^{-l/m_L}) = e^{-l/m_L}$$

이것이 $M^{-l}$과 같으려면:

$$e^{-l/m_L} = M^{-l} = e^{-l \ln M}$$

$$\therefore \frac{1}{m_L} = \ln M \implies m_L = \frac{1}{\ln M}$$

**$m_L$ 변동의 영향**:

| $m_L$ | 레이어 수 | 상위 레이어 밀도 | 검색 | 삽입 |
|---|---|---|---|---|
| $< 1/\ln M$ | 적음 | 희소 | 상위 통과 빠르나 L0 부담↑ | 빠름 |
| $= 1/\ln M$ | 최적 | 균형 | **최적** | 균형 |
| $> 1/\ln M$ | 많음 | 밀집 | 상위에서 불필요한 탐색 | 느림 |

## 6. 레이어 분포의 수학적 분석

### 6.1 레이어 확률

노드의 최대 레이어가 정확히 $l$일 확률:

$$P(l(v) = l) = P(l \leq \lfloor -\ln U \cdot m_L \rfloor < l+1)$$

$m_L = 1/\ln M$을 대입하면:

$$P(l(v) \geq l) = P(-\ln U \cdot m_L \geq l) = P(U \leq e^{-l/m_L}) = e^{-l \ln M} = M^{-l}$$

따라서:
$$P(l(v) = l) = M^{-l} - M^{-(l+1)} = M^{-l}\left(1 - \frac{1}{M}\right)$$

### 6.2 최대 레이어 기대값 — Order Statistics 유도

$N$개 독립 노드의 최대 레이어 $L_{\max} = \max_{i=1}^N l(v_i)$.

**CDF 유도**: $P(L_{\max} \leq l)$을 구한다.

$$P(l(v) \leq l) = 1 - P(l(v) > l) = 1 - M^{-(l+1)}$$

$N$개 독립이므로:

$$P(L_{\max} \leq l) = \left(1 - M^{-(l+1)}\right)^N$$

**기대값**: $L_{\max}$이 비음정수이므로:

$$\mathbb{E}[L_{\max}] = \sum_{l=0}^{\infty} P(L_{\max} > l) = \sum_{l=0}^{\infty} \left[1 - \left(1 - M^{-(l+1)}\right)^N\right]$$

$N$이 크면 $1 - (1 - x)^N \approx 1$ when $Nx \gg 1$, 즉 $NM^{-(l+1)} \gg 1 \iff l < \log_M N - 1$.
$NM^{-(l+1)} \ll 1$이면 $1 - (1 - x)^N \approx Nx$. 전환점은 $l^* = \log_M N - 1$에서 발생:

$$\mathbb{E}[L_{\max}] \approx (\log_M N - 1) + \sum_{j=0}^{\infty} NM^{-(\log_M N + j)} = \log_M N - 1 + \sum_{j=0}^{\infty} M^{-j-1}$$

$$= \log_M N - 1 + \frac{1}{M-1} \approx \log_M N$$

더 정밀하게는 Gumbel 분포 근사:

$$L_{\max} \approx \log_M N + \frac{\gamma}{\ln M}$$

여기서 $\gamma \approx 0.5772$는 Euler-Mascheroni 상수. 실용적으로 $\mathbb{E}[L_{\max}] \approx \log_M N$.

**분산**: $\mathrm{Var}[L_{\max}] \approx \frac{\pi^2}{6(\ln M)^2}$ (Gumbel 분포의 분산).

$M=4$이면 $\mathrm{Var} \approx \frac{1.645}{1.92} \approx 0.86$, 즉 $L_{\max}$는 기대값 주변 ±1 정도 변동.

실험 결과 (M=4, N=200):
```
Layer 0: 200 nodes
Layer 1:  45 nodes  (200 × 4⁻¹ ≈ 50)
Layer 2:  19 nodes  (200 × 4⁻² ≈ 12.5)
Layer 3:   5 nodes  (200 × 4⁻³ ≈ 3.1)
Layer 4:   2 nodes  (200 × 4⁻⁴ ≈ 0.8)
```

$\log_4 200 \approx 3.8$ — 실측 max_level=4와 근사. Gumbel 보정: $3.8 + 0.577/1.386 \approx 4.2$.

### 6.3 레이어별 노드 수의 분산

레이어 $l$의 노드 수 $N_l = \sum_{i=1}^N \mathbf{1}[l(v_i) \geq l]$.

각 $\mathbf{1}[l(v_i) \geq l] \sim \mathrm{Bernoulli}(M^{-l})$이고 독립이므로:

$$\mathbb{E}[N_l] = NM^{-l}, \quad \mathrm{Var}[N_l] = NM^{-l}(1 - M^{-l})$$

$M^{-l} \ll 1$인 상위 레이어에서 $N_l \approx \mathrm{Poisson}(NM^{-l})$로 근사.

변동 계수 $\mathrm{CV} = \sqrt{\mathrm{Var}}/\mathbb{E} = \sqrt{(1 - M^{-l})/(NM^{-l})} \approx M^{l/2}/\sqrt{N}$.

$l = \log_M N$이면 $\mathrm{CV} \approx 1$: 최상위 레이어 근처에서 노드 수가 크게 변동.

## 7. Navigable Small World의 이론적 기초

### 7.1 Small World 속성

Watts-Strogatz (1998):
- **Short path length**: 임의 두 노드 사이 평균 경로 $O(\log N)$
- **High clustering**: 이웃의 이웃도 이웃일 확률이 높음

HNSW가 이 속성을 만족하는 이유:
- 삽입 시 greedy search로 "가까운" 이웃 연결 → high clustering
- 상위 레이어의 장거리 간선 → short path length

### 7.2 Greedy Routing 수렴 보장

NSW에서 greedy routing이 수렴하려면:
- 모든 Voronoi 이웃이 그래프 이웃이어야 함 (이상적)
- 실제로는 근사: $\mathrm{ef}_{\mathrm{construction}} \geq M$이면 높은 확률로 수렴

**Monotonic Search Path 속성**: Greedy routing이 정확하려면 다음이 필요:

> 임의의 쿼리 $q$와 현재 노드 $v \neq \mathrm{NN}(q)$에 대해, $v$의 이웃 중 $q$에 더 가까운 노드가 반드시 존재.

형식적으로: $\forall q, \forall v \neq \mathrm{NN}(q)$:
$$\exists w \in \mathrm{neighbors}(v) : d(q, w) < d(q, v)$$

이 속성이 성립하면 greedy search는 유한 홉 후 반드시 $\mathrm{NN}(q)$에 도달. 실패하는 경우 = **local minimum** (dead end).

**HNSW에서 local minimum 확률**: $M$과 $\mathrm{ef}_{\mathrm{construction}}$이 충분히 크면, 삽입 시 양질의 이웃이 연결되어 local minimum이 거의 발생하지 않는다. 직관적으로:

1. 삽입 시 $\mathrm{ef}_{\mathrm{construction}}$개 후보에서 $M$개 선택 → Voronoi 이웃 대부분 커버
2. 양방향 연결로 역방향 경로도 존재
3. 다층 구조에서 상위 레이어가 장거리 "단축키" 역할 → local minimum 탈출

**Theorem (Malkov & Yashunin, 2018)**: HNSW에서 greedy search의 기대 홉 수:
$$\mathbb{E}[\mathrm{hops}] = O\left(\frac{1}{\ln M} \cdot \ln N\right)$$

**증명 스케치**:
1. 레이어 $l$에서 노드 수 $N_l = N/M^l$, 평균 간격 $\sim N_l^{-1/d}$
2. 레이어 $l$에서 $l-1$로 전환 시, 레이어 $l$의 커버리지 반경 내에 도달하는 홉 수 = $O(1)$ (NSW 속성 + 적절한 $M$)
3. 총 레이어 수 $L = \log_M N$ → 총 홉 $= O(L) = O(\log_M N) = O(\ln N / \ln M)$

각 홉에서 $M$개 이웃 거리 계산 → 전체:
$$T = O\left(\frac{M}{\ln M} \cdot \ln N \cdot D\right)$$

### 7.3 search_layer 종료성 증명

`search_layer`의 종료 조건: `c_dist > f_dist` (가장 가까운 미방문 후보가 현재 결과의 가장 먼 것보다 멀면 종료).

**Lemma**: search_layer는 유한 시간에 종료한다.

**증명**:
1. `visited`는 단조증가하므로 각 노드는 최대 한 번 처리
2. `candidates`에 추가되는 노드는 모두 `visited`에 없는 노드뿐
3. 그래프의 노드 수가 유한 ($N$)이므로, `candidates`에 추가 가능한 노드도 유한
4. 매 반복에서 `candidates`에서 하나를 제거 → 최대 $N$번 반복 후 종료

**최악 복잡도**: $O(N \cdot M \cdot D)$ (모든 노드 방문). 하지만 종료 조건 덕분에 실제로는 $O(\mathrm{ef} \cdot M \cdot D)$에 가깝다.

### 7.4 Delaunay Graph와의 관계

이상적인 ANN 그래프: **Delaunay Graph** (모든 Voronoi 이웃 연결).

**Delaunay Graph의 정의**: 점 $p$와 $q$가 Delaunay 이웃 $\iff$ $p$와 $q$의 Voronoi 셀이 공유 경계를 가짐.

**핵심 성질**: Delaunay Graph에서 greedy routing은 항상 정확한 NN을 반환한다.

**증명**: $q$에 대해 현재 노드 $v \neq \mathrm{NN}(q)$라 하자. $q$에서 $v$ 방향으로 이동하면 $v$의 Voronoi 셀을 지나, 반드시 $q$에 더 가까운 Voronoi 셀을 만난다. 해당 셀의 소유 노드 $w$는 $v$의 Delaunay 이웃이므로 $w \in \mathrm{neighbors}(v)$이고 $d(q,w) < d(q,v)$.

**고차원의 문제**: $d$차원에서 Delaunay edge 수:
$$|\mathrm{Edges}| = O(N^{\lceil d/2 \rceil})$$

$d=128$이면 $O(N^{64})$ — 완전히 비실용적. HNSW는 $M$으로 제한된 "근사 Delaunay"를 구축:
- 정확한 Voronoi 이웃 대신 거리순 top-$M$ 이웃
- Local minimum 발생 가능하지만, 다층 구조로 거의 완전히 해소

## 8. 거리 메트릭과 내부 정규화

### 8.1 다중 메트릭 지원

HNSW 그래프는 L2로 구축하되, 검색 시 임의 메트릭 사용 가능:

| 메트릭 | 정의 | 정렬 | 내부 변환 |
|---|---|---|---|
| L2 | $\|\|q - x\|\|$ | 오름차순 | 그대로 |
| L1 | $\sum_i \|q_i - x_i\|$ | 오름차순 | 그대로 |
| Cosine | $\frac{q \cdot x}{\|q\|\|x\|}$ | 내림차순 | $-\mathrm{sim}$ |
| DotProduct | $q \cdot x$ | 내림차순 | $-\mathrm{dot}$ |

**distance_internal**: 모든 메트릭을 "작을수록 가까움"으로 통일:

```rust
fn distance_internal(&self, a: &[f64], b: &[f64], metric: Metric) -> f64 {
    match metric {
        Metric::L2 => l2_distance(a, b),
        Metric::L1 => l1_distance(a, b),
        Metric::Cosine => -cosine_similarity_vec(a, b),
        Metric::DotProduct => -dot_product_vec(a, b),
    }
}
```

반환 시 Cosine/DotProduct는 부호 복원.

### 8.2 L2 구축의 타당성

그래프를 L2로 구축해도 다른 메트릭으로 검색 가능한 이유:
- L2 그래프의 이웃 구조가 다른 메트릭에서도 "good enough" 근사
- 특히 normalized 벡터에서: $\|q - x\|^2 = 2 - 2 \cos(q, x)$ → L2와 Cosine 동치
- DotProduct는 벡터 norm 차이가 크면 불일치 가능 → MIPS 전용 그래프 필요 시 별도 구축 권장

## 9. 이웃 선택 전략

### 9.1 Simple Selection (현재 구현)

후보 중 거리순 상위 $M$개 선택:
$$\mathrm{SelectSimple}(C, M) = \mathrm{argtop}_M\{d(q, c) : c \in C\}$$

장점: 구현 간단, 계산 비용 $O(|C| \log |C|)$
단점: 모든 이웃이 한 방향에 몰릴 수 있음 (diversity 부족)

### 9.2 Heuristic Selection (원 논문)

다양성을 보장하는 탐욕적 선택:

```
SelectHeuristic(q, C, M):
    R ← ∅ (결과), W ← C sorted by distance
    while |R| < M and W ≠ ∅:
        e ← W에서 가장 가까운 원소
        if d(q, e) < min_{r ∈ R} d(e, r):  // e가 기존 이웃보다 q에 더 가까우면
            R ← R ∪ {e}
        W ← W \ {e}
    return R
```

**직관**: 이미 선택된 이웃 $r$과 후보 $e$ 사이 거리가 $d(q, e)$보다 크면 $e$를 추가.
→ 이웃들이 서로 다른 방향을 커버하도록 강제.

**기하학적 해석**: Simple Selection이 실패하는 경우를 보자.

```
Simple: q의 이웃 3개가 모두 한 방향에 집중
                  ← 커버 안 됨
                  •
                 /
    • ─ • ─ • ─ q ─── • • • ← 이웃 3개
                 \
                  •
                  ← 커버 안 됨

Heuristic: 다양한 방향으로 분산
                  •₂ ← 이웃 2
                 /
    •₃ ←────── q ─── •₁ ← 이웃 1
    이웃 3       \
                  •
```

**조건 $d(q, e) < \min_{r \in R} d(e, r)$의 의미**: 후보 $e$가 이미 선택된 이웃 $r$ 보다 $q$에 더 가깝다면, $e$는 $r$이 "커버하지 못하는 방향"에 있는 것. 삼각부등식으로:

$$d(q, e) < d(e, r) \implies \angle(q \to e)\ \mathrm{and}\ \angle(q \to r)\ \mathrm{sufficiently\ different}$$

이것은 RNG (Relative Neighborhood Graph)의 정의와 일치: $d(q, e) < \max(d(q, r), d(e, r))$.

### 9.3 Simple vs Heuristic 비교

| | Simple | Heuristic |
|---|---|---|
| 복잡도 | $O(|C| \log M)$ | $O(|C| \cdot M)$ |
| Recall (low M) | 낮음 | **높음** |
| Recall (high M) | 비슷 | 비슷 |
| 적합한 경우 | $M \geq 32$ | $M \leq 16$ |
| 그래프 구조 | 밀집 클러스터 | 분산 커버리지 |
| Navigability | 약함 (dead-end 가능) | 강함 (다방향 탐색) |

**왜 M이 클 때 차이가 줄어드는가**: $M$이 충분히 크면 Simple에서도 다양한 방향의 이웃이 자연스럽게 포함된다. $M$이 작으면 "가까운" $M$개가 한 방향에 몰릴 확률이 높아 Heuristic의 다양성 보장이 중요해진다.

## 10. Skip List와의 관계

### 10.1 구조적 유사성

| Skip List | HNSW |
|---|---|
| 1D 정렬된 linked list | 고차원 유사도 그래프 |
| 레벨 $l$에 확률 $p^l$로 존재 | 레이어 $l$에 확률 $M^{-l}$로 존재 |
| 포인터 = 다음 원소 | 간선 = 유사한 이웃 |
| 검색: $O(\log N)$ | 검색: $O(\log N)$ |
| 정확한 검색 | 근사 검색 (greedy) |

### 10.2 핵심 차이

- Skip List: **정렬 불변량** 보장 → exact search
- HNSW: **NSW 속성** 보장 → approximate search
- Skip List: 1D → 비교 연산으로 충분
- HNSW: 고차원 → 거리 계산 필요, 정렬 불가

HNSW는 본질적으로 "고차원 Skip List"이며, 정확한 정렬 대신 greedy navigability로 타협.

## 11. 메모리 분석

### 11.1 노드당 메모리

벡터 저장: $D \cdot 8$ 바이트 (f64)
이웃 리스트: 레이어 $l$에서 평균 $M$ 이웃 × 8 바이트 (usize)

**기대 레이어 수 유도**: 노드가 레이어 $l$에 존재할 확률 $M^{-l}$.

$$\mathbb{E}[\mathrm{layers}] = \sum_{l=0}^{\infty} P(l(v) \geq l) = \sum_{l=0}^{\infty} M^{-l} = \frac{1}{1 - 1/M} = \frac{M}{M-1}$$

**노드당 이웃 수 기대값 유도**:

- Layer 0: $M_{\max 0} = 2M$ 이웃 (실제로는 $\leq 2M$, 평균 $\approx 2M$)
- Layer $l \geq 1$: $M$ 이웃, 노드가 존재할 확률 $M^{-l}$

$$\mathbb{E}[\mathrm{neighbors}] = 2M + \sum_{l=1}^{\infty} M^{-l} \cdot M = 2M + M \cdot \frac{1/M}{1 - 1/M} = 2M + \frac{M}{M-1} \approx 2M + 1 + \frac{1}{M-1}$$

$M \gg 1$이면 $\approx 2M + 1$. 실용적으로 $\approx 2M$ (Layer 0 지배적).

$$\mathrm{mem}_{\mathrm{neighbors}} \approx (2M + 1) \cdot 8\ \mathrm{bytes}$$

### 11.2 전체 메모리

$$\mathrm{Total} = N \cdot (D \cdot 8 + 3M \cdot 8) = 8N(D + 3M)$$

$D=128, M=16$: 노드당 $8 \times (128 + 48) = 1408$ 바이트

비교 (N=100만):
| 인덱스 | 메모리 |
|---|---|
| BruteForce | $N \cdot D \cdot 8 = 1.0$ GB |
| HNSW (M=16) | $\approx 1.3$ GB |
| PQ (M=8) | $\approx 8$ MB (코드) + 코드북 |
| IVF-PQ | PQ + 센트로이드 |

HNSW는 메모리 효율은 낮지만 검색 속도가 우수 → **IVF-HNSW-PQ** 조합 사용 (IVF 파티션 내에서 HNSW + PQ).

## 12. Recall 실험 결과

### 12.1 기본 성능

N=500, D=16, k=10:

```
M=16, ef_construction=100, ef_search=50:
  Average recall@10: 1.000
```

### 12.2 M 파라미터 효과

```
M= 4: recall@10=0.980, layers=5, avg_conn_L0= 8.0
M=16: recall@10=1.000, layers=3, avg_conn_L0=32.0
M=32: recall@10=1.000, layers=2, avg_conn_L0=64.0
```

### 12.3 ef_search 트레이드오프

```
ef_search= 10: recall@10=0.985
ef_search= 20: recall@10=1.000
ef_search= 50: recall@10=1.000
ef_search=100: recall@10=1.000
```

## 13. BruteForce, IVF, PQ, HNSW 비교

| | BruteForce | IVF | PQ | HNSW |
|---|---|---|---|---|
| 검색 복잡도 | $O(ND)$ | $O(\frac{N}{K} \cdot n_{\mathrm{probe}} \cdot D)$ | $O(NM + MK'D/M)$ | $O(\log N \cdot M \cdot D)$ |
| 학습 | 불필요 | K-means | K-means × M | 불필요 |
| 점진적 삽입 | O | X (재학습) | X (재학습) | **O** |
| 메모리 | $O(ND)$ | $O(ND + KD)$ | $O(NM)$ | $O(N(D + M))$ |
| 메트릭 | 모두 | 모두 | L2만 | 모두 |
| 튜닝 파라미터 | 없음 | $K, n_{\mathrm{probe}}$ | $M, K'$ | $M, \mathrm{ef}_c, \mathrm{ef}_s$ |
| 강점 | 정확 | 대규모 | 메모리 | **속도 + 유연성** |

## 14. 실전 조합: IVF-HNSW-PQ

대규모 시스템에서의 표준 파이프라인:

```
1. IVF: N개 벡터를 K개 클러스터로 분할 (coarse quantizer)
2. HNSW: 센트로이드 검색용 인덱스 (IVF의 nprobe 결정)
3. PQ: 각 클러스터 내 벡터 압축 (fine quantizer)

검색 흐름:
  Query → HNSW로 top-nprobe 센트로이드 검색 (O(log K))
       → 해당 클러스터들에서 PQ ADC (O(nprobe · N/K · M))
       → Re-rank top candidates with exact distance
```

FAISS의 `IndexIVFPQ` + `IndexHNSWFlat` 조합이 이 패턴.

### 14.2 IVF-HNSW-PQ 정량 분석

$N = 10^9$ (10억) 벡터, $D = 128$, $K = 65536$ 클러스터, PQ $M_{\mathrm{PQ}} = 16$, $K' = 256$:

**메모리**:
- PQ 코드: $N \times M_{\mathrm{PQ}} = 10^9 \times 16 = 16$ GB
- 코드북: $M_{\mathrm{PQ}} \times K' \times (D/M_{\mathrm{PQ}}) \times 4 = 16 \times 256 \times 8 \times 4 = 128$ KB
- HNSW (센트로이드): $K \times (D \times 4 + 2M_{\mathrm{HNSW}} \times 4) \approx 65536 \times 640 = 40$ MB
- **합계: ~16 GB** (원본 $N \times D \times 4 = 512$ GB 대비 32배 압축)

**검색 비용** ($n_{\mathrm{probe}} = 32$):
1. HNSW 센트로이드 검색: $\log_{16} 65536 \approx 4$ 레이어 × $16 \times 128 = 8192$ FLOP
2. PQ ADC ($n_{\mathrm{probe}} \times N/K$): $32 \times 15000 \times 16 = 7.7M$ 룩업
3. Re-rank top-1000: $1000 \times 128 = 128K$ FLOP

**총 latency**: ~1ms (GPU) ~ 5ms (CPU). BruteForce 대비 ~1000배 빠름.

## 15. 삽입 복잡도 상세 분석

단일 벡터 삽입 비용을 상세히 유도한다.

### 15.1 Phase 1: 상위 레이어 통과

새 노드의 레이어 $l$, 현재 최대 레이어 $L$. Phase 1에서 $L - l$개 레이어를 greedy ($\mathrm{ef}=1$)로 통과:

$$T_1 = \sum_{lc=l+1}^{L} O(M \cdot D) = O((L - l) \cdot M \cdot D)$$

기대값: $\mathbb{E}[L - l] = L - \mathbb{E}[l] = \log_M N - \frac{m_L}{1 - M^{-1}} \approx \log_M N - \frac{1}{(\ln M)(1 - 1/M)}$

대부분의 노드는 $l = 0$이므로 $\mathbb{E}[T_1] = O(\log_M N \cdot M \cdot D)$.

### 15.2 Phase 2: 이웃 연결

레이어 $\min(l, L) \to 0$에서 각 레이어당:
1. `search_layer` ($\mathrm{ef}_c$개 후보 탐색): $O(\mathrm{ef}_c \cdot M \cdot D)$
2. 양방향 연결 + overflow pruning: $O(M \cdot M \cdot D) = O(M^2 D)$ (최악: M개 이웃 각각의 이웃 리스트 재정렬)

$$T_2 = \sum_{lc=0}^{\min(l,L)} O((\mathrm{ef}_c \cdot M + M^2) \cdot D) = O((\min(l,L)+1) \cdot \mathrm{ef}_c \cdot M \cdot D)$$

대부분 $l = 0$이므로: $\mathbb{E}[T_2] = O(\mathrm{ef}_c \cdot M \cdot D)$.

### 15.3 전체 삽입 비용

$$T_{\mathrm{insert}} = T_1 + T_2 = O((\log_M N + \mathrm{ef}_c) \cdot M \cdot D)$$

$N$개 벡터 전체 구축:

$$T_{\mathrm{build}} = \sum_{i=1}^{N} T_{\mathrm{insert}}(i) = O\left(N \cdot (\log_M N + \mathrm{ef}_c) \cdot M \cdot D\right)$$

비교:
| 인덱스 | 구축 비용 |
|---|---|
| BruteForce | $O(ND)$ |
| IVF | $O(N K D \cdot \mathrm{iter})$ (K-means) |
| PQ | $O(N M K' D/M \cdot \mathrm{iter})$ |
| **HNSW** | $O(N \cdot \mathrm{ef}_c \cdot M \cdot D \cdot \log N)$ |

HNSW는 구축이 가장 느리지만, 점진적 삽입이 가능하고 학습 단계가 불필요한 것이 장점.

## 16. 차원의 저주와 HNSW

### 16.1 고차원에서의 성능 저하

고차원에서 ANN 검색의 근본적 어려움:

**거리 집중 현상 (Distance Concentration)**:

$d$차원에서 $N$개 균일 분포 벡터의 최근점/최원점 거리 비:

$$\frac{d_{\max}}{d_{\min}} \to 1 \quad \mathrm{as} \quad d \to \infty$$

정밀하게, $d$차원 단위 큐브에서:

$$\mathbb{E}\left[\frac{d_{\max} - d_{\min}}{d_{\min}}\right] = O\left(\frac{1}{\sqrt{d}}\right)$$

모든 벡터가 쿼리에서 비슷한 거리에 있으면, greedy routing의 "가까운 이웃으로 이동" 전략이 무의미해진다.

### 16.2 HNSW가 그래도 작동하는 이유

실제 데이터는 이론적 최악 경우와 다르다:

1. **Intrinsic dimension $\ll$ ambient dimension**: 128차원 벡터도 실제 intrinsic dimension은 ~10~30
2. **클러스터 구조**: 실제 데이터는 매니폴드 위에 분포 → 국소적으로 저차원
3. **HNSW의 적응성**: 그래프 구조가 데이터 분포에 자동 적응

**경험적 법칙**: HNSW가 잘 작동하는 조건:
- Intrinsic dimension $d_{\mathrm{int}} \leq 30$
- $M \geq d_{\mathrm{int}}$ (각 방향을 커버할 이웃 필요)
- $\mathrm{ef}_c \geq 2M$ (충분한 탐색으로 좋은 이웃 확보)

### 16.3 고차원에서의 파라미터 조정

| Intrinsic dim | 권장 M | 권장 $\mathrm{ef}_c$ | 비고 |
|---|---|---|---|
| $\leq 10$ | 8~16 | 64~128 | 저차원, 적은 M으로 충분 |
| 10~30 | 16~32 | 128~256 | 일반적 임베딩 |
| 30~100 | 32~64 | 256~512 | 고차원, M 증가 필요 |
| $> 100$ | 64+ | 512+ | HNSW 비효율, PQ 조합 권장 |

## 17. MIPS (Maximum Inner Product Search)

### 17.1 문제

DotProduct(내적)가 높은 벡터를 찾는 MIPS는 메트릭이 아니다:

$$\langle q, x \rangle \neq d(q, x)$$

삼각부등식 불성립: $\langle q, x \rangle + \langle x, y \rangle \not\geq \langle q, y \rangle$ (일반적으로).

Greedy routing의 수렴 보장이 깨질 수 있다: 내적이 큰 이웃으로 이동해도, 목표에 더 가까워지는 보장이 없음.

### 17.2 MIPS → NNS 변환

**Shrivastava & Li (2014)**: MIPS를 L2 NNS로 변환하는 증강(augmentation):

쿼리 $q \in \mathbb{R}^d$와 데이터 $x \in \mathbb{R}^d$를 변환:

$$\tilde{x} = [x; \|x\|^2; \|x\|^4; \ldots; \|x\|^{2^m}] \in \mathbb{R}^{d+m}$$
$$\tilde{q} = [q; 1/2; 1/2; \ldots; 1/2] \in \mathbb{R}^{d+m}$$

모든 $\|x\| \leq 1$로 정규화한 후:

$$\|\tilde{q} - \tilde{x}\|^2 = \|q\|^2 - 2\langle q, x \rangle + \|x\|^2 + \sum_{j=1}^{m}\left(\frac{1}{2} - \|x\|^{2^j}\right)^2$$

$m$이 충분히 크면 $\|x\|^{2^j} \to 0$ (for $\|x\| < 1$), 따라서:

$$\|\tilde{q} - \tilde{x}\|^2 \approx C - 2\langle q, x \rangle$$

내적 최대화 $\iff$ L2 거리 최소화!

### 17.3 ip-HNSW 접근법

실용적으로는 MIPS 변환 없이 HNSW를 직접 사용:
- 그래프 구축도 내적 기반
- `distance_internal`에서 $-\langle q, x \rangle$ 사용 (부호 반전)
- 이론적 보장은 약하지만, 실측 recall은 L2와 비슷

단, norm 분포가 넓은 데이터에서는 recall 저하 → norm 기준 정규화 또는 MIPS 변환 권장.

## 18. Degree 분포 분석

### 18.1 Out-degree vs In-degree

HNSW에서 각 노드의 연결은 비대칭:
- **Out-degree**: 삽입 시 자신이 선택한 이웃 → 정확히 $\min(M, |C|) \leq M$ (layer 1+), $\leq 2M$ (layer 0)
- **In-degree**: 다른 노드가 자신을 이웃으로 선택 → 제한 없음 (pruning에 의해 간접 제한)

### 18.2 Layer 0의 In-degree 분포

Layer 0에서 노드 $v$의 in-degree $k_v$를 분석한다.

**초기 삽입 노드**: 모든 후속 노드가 $v$를 이웃 후보로 고려 → 높은 in-degree (hub).
**후기 삽입 노드**: 이미 좋은 이웃이 많으므로 선택 확률 낮음 → 낮은 in-degree.

이것은 **preferential attachment** 효과와 유사:

$$P(\mathrm{node}\ v\ \mathrm{selected\ as\ new\ neighbor}) \propto \mathrm{accessibility}(v)$$

초기 노드는 "허브"가 되어 많은 이웃이 연결. 실측에서 layer 0의 degree 분포는 heavy-tail을 보인다.

### 18.3 Pruning의 효과

Pruning은 $M_{\max}$로 degree를 제한:
- In-degree도 간접 제한: $v$가 $w$의 이웃에 추가될 때, $w$의 이웃이 $M_{\max}$ 초과하면 가장 먼 이웃 제거
- 이것은 $v$가 아닌 다른 노드가 제거될 수 있으므로, $v$의 in-degree는 여전히 $> M_{\max}$ 가능

실질적으로: out-degree $\leq M_{\max}$, in-degree는 평균 $\approx M$이지만 분산이 크다.

## 19. HNSW의 한계와 개선

### 19.1 한계

1. **높은 메모리**: 원본 벡터 + 그래프 구조 → PQ 대비 10~100배
2. **삽입 비용**: $O(M \cdot \mathrm{ef}_c \cdot \log N \cdot D)$ — 대량 삽입 시 느림
3. **삭제 어려움**: 그래프에서 노드 제거 시 연결성 유지 어려움 (tombstone 패턴 사용)
4. **메모리 접근 패턴**: 랜덤 접근 → 캐시 미스 빈번

### 19.2 개선 방향

- **HNSW + PQ**: 벡터를 PQ 코드로 저장, 검색 시 ADC로 거리 계산 → 메모리 절감
- **DiskANN**: SSD에 그래프 저장, Vamana 알고리즘으로 디스크 친화적 그래프 구축
- **Filtered HNSW**: 메타데이터 필터링과 결합 (Qdrant, Weaviate)
- **NSG (Navigating Spreading-out Graph)**: HNSW의 개선 — 더 적은 간선으로 동일 recall

## 20. DiskANN과 Vamana 알고리즘

HNSW의 메모리 문제를 해결하는 대표적 접근.

### 20.1 핵심 아이디어

HNSW는 그래프 전체가 메모리에 있어야 한다. 10억 벡터 × 128D × fp32 = 512GB → 불가능.

**DiskANN (Subramanya et al., 2019)**:
- 그래프를 **SSD**에 저장
- 벡터를 **PQ 코드**로 메모리에 압축 보관 (거리 필터용)
- SSD 읽기를 최소화하는 그래프 구조 = **Vamana**

### 20.2 Vamana 그래프

HNSW와의 핵심 차이:

| | HNSW | Vamana |
|---|---|---|
| 레이어 | 다층 ($\log_M N$) | **단층** |
| 이웃 수 | $M$ (layer 1+), $2M$ (layer 0) | $R$ (uniform) |
| Entry point | 최상위 레이어 노드 | **Medoid** (데이터 중심) |
| 구축 | 점진적 삽입 | **배치** (2-pass) |
| 장거리 연결 | 상위 레이어가 제공 | **$\alpha$-pruning**이 제공 |

**$\alpha$-RNG pruning** ($\alpha > 1$):

```
SelectVamana(q, C, R, α):
    R_result ← ∅, W ← C sorted by distance
    while |R_result| < R and W ≠ ∅:
        e ← W에서 가장 가까운 원소
        if d(q, e) < α · min_{r ∈ R_result} d(e, r):  // α > 1로 완화
            R_result ← R_result ∪ {e}
        W ← W \ {e}
    return R_result
```

$\alpha = 1$: 엄격한 RNG (= HNSW heuristic). $\alpha = 1.2$: 장거리 연결 허용.

직관: $\alpha > 1$이면 "이미 있는 이웃보다 $\alpha$배 멀어도 추가" → 다층 없이도 장거리 점프 가능.

### 20.3 SSD I/O 최적화

DiskANN의 검색 시 SSD 접근 패턴:

1. **PQ 코드 (메모리)**: 모든 벡터의 PQ 코드를 메모리에 보관 ($\sim$ 32바이트/벡터)
2. **Beam search with SSD**: 후보 노드의 정확한 벡터 + 이웃 리스트를 SSD에서 읽음
3. **정렬된 SSD 레이아웃**: 노드와 이웃을 같은 4KB 섹터에 배치 → 1회 읽기로 노드+이웃 동시 획득

결과: SSD 읽기 $\sim$ 10~30회로 10억 벡터 검색 가능.

## 21. NSG vs HNSW

### 21.1 NSG (Navigating Spreading-out Graph)

Fu et al. (2019). HNSW의 대안:

- **단층 그래프** (HNSW 다층 불필요)
- **MRNG (Monotonic RNG)**: greedy search가 단조 수렴하는 RNG
- **Navigation node**: 전체 데이터의 중심점을 entry point로 사용

### 21.2 NSG의 핵심 정리

**Theorem (MRNG)**: 그래프 $G$가 MRNG 조건을 만족하면, 임의의 entry point에서 임의의 목표까지 greedy search가 단조 수렴한다.

MRNG 조건: $\forall (u,v) \in G$, $u$와 $v$의 "lune" (두 구의 교집합) 내에 다른 노드가 없음.

고차원에서 MRNG edge 수는 Delaunay보다 적어 실용적.

### 21.3 HNSW vs NSG 비교

| | HNSW | NSG |
|---|---|---|
| 구축 | 점진적 (online) | 배치 (offline) |
| 레이어 | 다층 | 단층 |
| 이론 | Skip List 기반 | RNG 기반 |
| 삽입/삭제 | 지원 | 재구축 필요 |
| Recall/QPS | 비슷 | 비슷 (약간 높은 QPS) |
| 메모리 | 높음 (다층) | 낮음 (단층) |

실용적으로 HNSW의 점진적 삽입이 대부분의 서비스에서 중요하므로 HNSW가 더 널리 사용됨.

## 22. Recall의 이론적 모델

### 22.1 ef_search와 Recall의 관계

$\mathrm{ef}_s$를 증가시키면 recall이 어떻게 변하는가?

**모델**: search_layer에서 방문하는 노드 수 $\approx \mathrm{ef}_s \cdot c$ ($c$는 평균 확장 상수).
true NN $k$개가 이 방문 집합에 포함될 확률:

$$\mathrm{recall}@k(\mathrm{ef}_s) \approx 1 - \binom{N - k}{\mathrm{ef}_s \cdot c} / \binom{N}{\mathrm{ef}_s \cdot c}$$

이것은 비복원 추출의 hypergeometric 근사. $\mathrm{ef}_s \cdot c \ll N$이면:

$$\mathrm{recall}@k \approx 1 - \left(1 - \frac{k}{N}\right)^{\mathrm{ef}_s \cdot c}$$

하지만 HNSW의 탐색은 랜덤이 아닌 **biased** (가까운 영역 집중):

$$\mathrm{recall}@k \approx 1 - \exp\left(-\frac{\mathrm{ef}_s \cdot c'}{k}\right)$$

여기서 $c'$은 그래프 품질에 의존하는 상수. 이 모델에서:
- $\mathrm{ef}_s = k$: recall $\approx 1 - e^{-c'} \approx 0.63$ (for $c' = 1$)
- $\mathrm{ef}_s = 5k$: recall $\approx 1 - e^{-5c'} \approx 0.99$

### 22.2 QPS-Recall Pareto 곡선

실전에서의 성능 평가 기준:

$$\mathrm{QPS}(\mathrm{ef}_s) \propto \frac{1}{\mathrm{ef}_s \cdot M \cdot D}$$

Recall과 QPS를 동시에 그리면 Pareto 곡선 형성:

```
Recall
1.0 ─────────────── •  (ef=500, QPS=100)
                   •   (ef=200, QPS=300)
0.9           •        (ef=50, QPS=1000)
0.8      •             (ef=20, QPS=3000)
    •                  (ef=10, QPS=5000)
0.5
    ├───┬───┬───┬───┬──
    0  1k  2k  3k  5k   QPS
```

운영 목표에 따라 $\mathrm{ef}_s$ 조절. 이것이 HNSW의 큰 장점: **인덱스 재구축 없이 런타임에 recall/latency 트레이드오프 제어**.

## 23. 결정론적 재현성

LCG PRNG으로 레이어 할당 재현:

```rust
fn random_level(&mut self) -> usize {
    self.rng_state = self.rng_state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = (self.rng_state >> 11) as f64 / (1u64 << 53) as f64;
    (-u.max(1e-15).ln() * self.ml).floor().min(16.0) as usize
}
```

동일 seed + 동일 삽입 순서 → 동일 그래프 구조. 테스트에서 검증 완료.

## 24. 벡터 DB에서의 HNSW 활용

주요 벡터 데이터베이스의 HNSW 활용:

| 시스템 | HNSW 역할 | 특수 기능 |
|---|---|---|
| FAISS | `IndexHNSWFlat` | IVF coarse quantizer로 조합 |
| Qdrant | 기본 인덱스 | Filtered HNSW (메타데이터 필터링) |
| Weaviate | 기본 인덱스 | HNSW + PQ 조합, 동적 재인덱싱 |
| Milvus | Knowhere 엔진 | GPU-HNSW, 하이브리드 검색 |
| Pinecone | 내부 구현 | Serverless, 자동 스케일링 |
| pgvector | ivfflat/hnsw | PostgreSQL 확장, WAL 기반 내구성 |

**Filtered HNSW의 전략**:
1. **Pre-filter**: 메타데이터 필터 먼저 적용 → 결과 집합에서 HNSW 검색 (필터가 매우 선택적일 때)
2. **Post-filter**: HNSW 검색 후 필터 적용 → 결과 부족 시 ef 증가 재검색
3. **In-graph filter**: HNSW 탐색 중 필터 조건 만족하는 노드만 결과에 추가 (Qdrant 방식)

## 25. Phase 4 완성 로드맵

| Step | 주제 | 핵심 | 상태 |
|---|---|---|---|
| 71 | BruteForce | 선형 스캔, 4종 메트릭 | 완료 |
| 72 | IVF | K-means 파티셔닝, nprobe 제어 | 완료 |
| 73 | PQ | 서브공간 양자화, ADC, 메모리 압축 | 완료 |
| **74** | **HNSW** | **다층 그래프, O(log N) 검색** | **완료** |

Phase 4 (벡터 검색) 완성!
