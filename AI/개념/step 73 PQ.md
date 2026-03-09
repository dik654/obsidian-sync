# Step 73: Product Quantization (PQ)

Phase 4 (벡터 검색)의 세 번째 스텝. Jegou et al. (2011)의 핵심 알고리즘.

벡터를 **서브공간으로 분할**하고 각 서브공간에서 독립적으로 양자화하여 메모리를 수십 배 절약하면서 근사 최근접 이웃 검색을 수행한다.

---

## 1. PQ의 핵심 아이디어

Brute force와 IVF는 **원본 벡터를 그대로 저장**한다. $N$개의 $D$차원 벡터는 $N \cdot D \cdot 8$ 바이트(f64)를 차지한다. $N = 10^9$, $D = 128$이면 약 **1TB** — 메모리에 올릴 수 없다.

PQ는 벡터를 **M바이트 코드**로 압축:

$$
\underbrace{N \cdot D \cdot 8}_{\mathrm{원본\;(바이트)}} \;\longrightarrow\; \underbrace{N \cdot M}_{\mathrm{코드}} + \underbrace{M \cdot K' \cdot \frac{D}{M} \cdot 8}_{\mathrm{코드북}}
$$

$N$이 충분히 크면 코드북은 무시 가능하고, 압축률은 약 $\frac{D \cdot 8}{M} = \frac{8D}{M}$배.

**예시**: $D = 128$, $M = 8$ → 128배 압축 (코드북 무시 시).

---

## 2. 알고리즘 상세

### 2.1 서브공간 분할

$D$차원 벡터 $\mathbf{x}$를 $M$개 서브벡터로 분할:

$$
\mathbf{x} = [\underbrace{x_1, \ldots, x_{d_s}}_{\mathbf{x}^{(0)}}, \; \underbrace{x_{d_s+1}, \ldots, x_{2d_s}}_{\mathbf{x}^{(1)}}, \; \ldots, \; \underbrace{x_{D-d_s+1}, \ldots, x_D}_{\mathbf{x}^{(M-1)}}]
$$

여기서 $d_s = D / M$은 서브벡터 차원. $D$는 $M$으로 나누어져야 한다.

### 2.2 코드북 학습 (Training)

각 서브공간 $m = 0, \ldots, M-1$에서 독립적으로 K-means 수행:

$$
\mathcal{C}^{(m)} = \{c^{(m)}_0, c^{(m)}_1, \ldots, c^{(m)}_{K'-1}\} \subset \mathbb{R}^{d_s}
$$

총 $M$개 코드북, 각각 $K'$개 중심. 보통 $K' = 256$으로 설정하여 각 코드가 1바이트(u8)에 저장된다.

**Step 72의 kmeans() 재사용**: 동일한 Lloyd's 알고리즘을 서브벡터에 대해 $M$번 호출.

### 2.3 인코딩 (Encoding)

벡터 $\mathbf{x}$를 $M$바이트 코드로 변환:

$$
\mathrm{encode}(\mathbf{x}) = [q_0, q_1, \ldots, q_{M-1}]
$$

$$
q_m = \arg\min_{j \in \{0, \ldots, K'-1\}} \|\mathbf{x}^{(m)} - c^{(m)}_j\|_2
$$

각 $q_m \in \{0, \ldots, 255\}$이므로 u8 한 바이트. **원본 벡터는 폐기**, 코드만 저장.

### 2.4 디코딩 (Reconstruction)

코드에서 근사 벡터 복원:

$$
\hat{\mathbf{x}} = \mathrm{decode}([q_0, \ldots, q_{M-1}]) = [c^{(0)}_{q_0} \| c^{(1)}_{q_1} \| \cdots \| c^{(M-1)}_{q_{M-1}}]
$$

$\|$ 는 벡터 연결(concatenation). 이것이 양자화의 **재구성 벡터**.

### 2.5 ADC (Asymmetric Distance Computation)

PQ의 검색 알고리즘. "비대칭"이란 **쿼리는 양자화하지 않고** DB 벡터만 양자화한다는 뜻.

**핵심 관찰**: L2 거리의 제곱은 서브공간에서 가분(decomposable):

$$
\|\mathbf{q} - \mathbf{x}\|^2 = \sum_{m=0}^{M-1} \|\mathbf{q}^{(m)} - \mathbf{x}^{(m)}\|^2
$$

DB 벡터 $\mathbf{x}$는 코드 $[q_0, \ldots, q_{M-1}]$로 표현되므로:

$$
\|\mathbf{q} - \mathbf{x}\|^2 \approx \sum_{m=0}^{M-1} \|\mathbf{q}^{(m)} - c^{(m)}_{q_m}\|^2
$$

**알고리즘**:

1. **거리 테이블 구축**: 쿼리의 각 서브벡터와 코드북의 모든 중심 사이 squared L2 계산

$$
T[m][j] = \|\mathbf{q}^{(m)} - c^{(m)}_j\|^2, \quad m = 0 \ldots M-1, \; j = 0 \ldots K'-1
$$

비용: $O(M \cdot K' \cdot d_s) = O(K' \cdot D)$

2. **코드 스캔**: 각 DB 벡터의 코드로 테이블 룩업

$$
d(\mathbf{q}, \mathbf{x}_i) \approx \sum_{m=0}^{M-1} T[m][\mathrm{code}_i[m]]
$$

비용: $O(N \cdot M)$

3. **정렬 + top-k 반환**

**총 복잡도**: $O(K' \cdot D + N \cdot M)$ vs brute force $O(N \cdot D)$.

$M \ll D$이므로 스캔 단계에서 큰 속도 향상.

---

## 3. VQ vs PQ: Cartesian Product의 힘

### 3.1 벡터 양자화 (VQ)

일반 벡터 양자화(VQ)는 전체 $D$차원 공간에서 $K$개 중심을 학습:

$$
q(\mathbf{x}) = \arg\min_{j \in \{0,\ldots,K-1\}} \|\mathbf{x} - c_j\|^2
$$

$K = 256$이면 코드 1바이트로 저장 가능하지만, $D = 128$ 공간을 256개 중심으로 양자화하면 **왜곡이 매우 크다**. 양자화 오차를 줄이려면 $K$를 크게 해야 하지만:

- $K = 2^{64}$이면 코드 8바이트 → 코드북에 $2^{64} \times 128 \times 8$ 바이트 필요 → **불가능**
- K-means 학습에 $O(N \cdot K \cdot D)$ → $K$가 크면 학습 불가

### 3.2 PQ의 Cartesian Product 구조

PQ의 핵심 통찰: **M개 독립 코드북의 직적(Cartesian product)**으로 거대한 양자화 공간을 구성.

각 서브공간의 코드북 크기가 $K'$이면, PQ의 **실효 코드북 크기**:

$$
|\mathcal{C}_{\mathrm{PQ}}| = \underbrace{K'}_{m=0} \times \underbrace{K'}_{m=1} \times \cdots \times \underbrace{K'}_{m=M-1} = {K'}^M
$$

$K' = 256$, $M = 8$이면 $256^8 = 2^{64}$ — VQ로는 절대 도달할 수 없는 코드북 크기를 PQ는 **$M \cdot K'$ 개 중심만 저장**하면서 달성.

### 3.3 저장 비교

| 방법 | 코드 크기 | 실효 코드북 | 코드북 저장 | 학습 비용 |
|---|---|---|---|---|
| VQ ($K = 2^{64}$) | 8 bytes | $2^{64}$ | 불가능 | 불가능 |
| PQ ($M = 8, K' = 256$) | 8 bytes | $256^8 = 2^{64}$ | $8 \times 256 \times d_s \times 8$ | $O(M \cdot N \cdot K' \cdot d_s)$ |

**동일한 코드 크기(8바이트)**에서 PQ는 VQ와 같은 실효 해상도를 가지지만, 코드북 저장과 학습이 실현 가능하다. 대가는 서브공간 간 **독립성 가정** — 교차 서브공간 상관관계를 무시한다.

### 3.4 K-means ↔ VQ ↔ PQ 관계

$$
\underset{\mathrm{K\text{-}means}}{\mathrm{clustering}} \;\xrightarrow{\mathrm{encode/decode}}\; \underset{\mathrm{VQ}}{\mathrm{Vector\;Quantization}} \;\xrightarrow{\mathrm{product\;structure}}\; \underset{\mathrm{PQ}}{\mathrm{Product\;Quantization}}
$$

K-means = VQ의 학습 알고리즘. PQ = 구조화된 VQ (structured VQ). K-means를 서브공간에서 독립 수행하는 것이 PQ의 전부.

---

## 4. ADC vs SDC

| | ADC (Asymmetric) | SDC (Symmetric) |
|---|---|---|
| **쿼리 처리** | 양자화하지 않음 (원본 사용) | 쿼리도 양자화 |
| **거리 테이블** | $T[m][j] = \|\mathbf{q}^{(m)} - c^{(m)}_j\|^2$ | $T[m][i][j] = \|c^{(m)}_i - c^{(m)}_j\|^2$ (사전 계산) |
| **정확도** | 높음 (쿼리 오차 없음) | 낮음 (양쪽 양자화 오차) |
| **메모리** | 테이블: $M \cdot K'$ per query | 사전 테이블: $M \cdot K'^2$ (고정) |
| **사용** | 표준 PQ 검색 | 대량 배치 검색 (테이블 재사용) |

ADC가 거의 항상 우세하여 실무에서 기본.

---

## 5. 양자화 오차 분석

### 5.1 Distortion (왜곡)

양자화 오차의 기대값:

$$
\mathrm{MSE} = \mathbb{E}\left[\|\mathbf{x} - \hat{\mathbf{x}}\|^2\right] = \sum_{m=0}^{M-1} \mathbb{E}\left[\|\mathbf{x}^{(m)} - c^{(m)}_{q_m}\|^2\right]
$$

각 서브공간의 양자화 오차가 독립적으로 합산. 이것이 PQ의 **가분성(decomposability)** 원리.

### 5.2 Lloyd-Max 양자화와의 관계

PQ의 각 서브공간에서의 K-means는 **Lloyd-Max 최적 양자화기**의 다차원 버전이다.

1차원 Lloyd-Max 양자화기는 입력 분포 $p(x)$에 대해 MSE를 최소화하는 $K'$개 재현 레벨과 결정 경계를 찾는다:

$$
\min_{\{c_j\}, \{b_j\}} \sum_{j=0}^{K'-1} \int_{b_j}^{b_{j+1}} (x - c_j)^2 \, p(x) \, dx
$$

최적 조건:
- **재현 레벨** (centroid condition): $c_j = \frac{\int_{b_j}^{b_{j+1}} x \, p(x) \, dx}{\int_{b_j}^{b_{j+1}} p(x) \, dx}$ (영역 평균)
- **결정 경계** (nearest neighbor condition): $b_j = \frac{c_{j-1} + c_j}{2}$ (중점)

이것이 정확히 K-means의 assign-update와 동일:

$$
\mathrm{K\text{-}means} = \mathrm{다차원\;Lloyd\text{-}Max\;양자화기}
$$

### 5.3 ADC 근사 오차의 엄밀한 바운드

쿼리 $\mathbf{q}$와 DB 벡터 $\mathbf{x}$에 대해 ADC의 근사 거리와 실제 거리의 차이:

$$
\tilde{d}(\mathbf{q}, \mathbf{x}) = \|\mathbf{q} - \hat{\mathbf{x}}\|^2, \quad d(\mathbf{q}, \mathbf{x}) = \|\mathbf{q} - \mathbf{x}\|^2
$$

삼각 부등식을 squared 형태로 전개:

$$
\|\mathbf{q} - \hat{\mathbf{x}}\|^2 = \|\mathbf{q} - \mathbf{x} + \mathbf{x} - \hat{\mathbf{x}}\|^2 = \|\mathbf{q} - \mathbf{x}\|^2 + \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + 2\langle \mathbf{q} - \mathbf{x}, \mathbf{x} - \hat{\mathbf{x}} \rangle
$$

따라서 오차:

$$
|\tilde{d} - d| = \left|\|\mathbf{x} - \hat{\mathbf{x}}\|^2 + 2\langle \mathbf{q} - \mathbf{x}, \mathbf{x} - \hat{\mathbf{x}} \rangle\right| \leq \epsilon^2 + 2r\epsilon
$$

여기서 $\epsilon = \|\mathbf{x} - \hat{\mathbf{x}}\|$ (양자화 오차), $r = \|\mathbf{q} - \mathbf{x}\|$ (실제 거리).

**의미**: 양자화 오차 $\epsilon$이 이웃 간 거리 차이 $\Delta r$보다 작으면 순위가 보존된다. $r$이 클수록 절대 오차는 크지만 어차피 관심 밖 (nearest neighbor는 $r$이 작은 벡터).

### 5.4 Rate-Distortion 트레이드오프

PQ 코드는 $B = M \cdot \log_2 K'$ 비트를 사용. 정보이론에서 가우시안 소스 $X \sim \mathcal{N}(0, \sigma^2)$의 rate-distortion 함수:

$$
R(D) = \frac{1}{2} \log_2 \frac{\sigma^2}{D}, \quad D \leq \sigma^2
$$

$D$차원 i.i.d. 가우시안에서 왜곡 $D_1$ per 차원을 달성하는 총 rate:

$$
R_{\mathrm{total}} = \frac{D}{2} \log_2 \frac{\sigma^2}{D_1}
$$

PQ는 각 서브공간을 독립 양자화하므로 실제 rate:

$$
R_{\mathrm{PQ}} = M \cdot \log_2 K' \quad (\mathrm{bits})
$$

PQ가 rate-distortion 하한에 근접하려면 서브공간 내 차원들이 **독립이고 동일 분산**이어야 한다. 상관관계가 있으면 OPQ의 회전으로 decorrelation이 필요.

### 5.5 M과 K'의 최적 배분

총 코드 비트: $B = M \cdot \log_2 K'$

동일 비트 예산 $B$에서:
- **M 크고 K' 작음**: 서브공간 차원 $d_s = D/M$ 감소 → K-means가 저차원에서 동작 → 효과적, 하지만 $K'$가 작으면 해상도↓
- **M 작고 K' 큼**: 서브공간 차원↑ → K-means가 고차원에서 동작 → 차원의 저주 영향

**Water-filling**: $B$ 비트를 $M$개 서브공간에 균등 배분 ($\log_2 K' = B/M$)이 i.i.d. 가우시안에서 최적. 분산이 불균등하면 분산이 큰 서브공간에 더 많은 비트를 할당해야 하지만, PQ는 균등 배분만 지원 (모든 서브공간에 동일한 $K'$).

실무에서 $K' = 256$ (8비트)이 **구현 효율** (u8 타입, SIMD 정렬)과 **충분한 해상도**의 균형점.

---

## 6. L2 거리의 서브공간 분해 증명

PQ의 이론적 기반은 L2² 거리의 **가분성(additivity)**:

$$
\|\mathbf{q} - \mathbf{x}\|^2 = \sum_{d=1}^{D} (q_d - x_d)^2 = \sum_{m=0}^{M-1} \sum_{d=1}^{d_s} (q^{(m)}_d - x^{(m)}_d)^2 = \sum_{m=0}^{M-1} \|\mathbf{q}^{(m)} - \mathbf{x}^{(m)}\|^2
$$

이 등식은 **서브공간들이 차원을 분할**하기 때문에 정확히 성립한다. 교차항이 없다.

### 다른 메트릭은?

**내적(dot product)**:

$$
\langle \mathbf{q}, \mathbf{x} \rangle = \sum_{m=0}^{M-1} \langle \mathbf{q}^{(m)}, \mathbf{x}^{(m)} \rangle
$$

내적도 가분! 따라서 PQ로 내적 검색도 가능하다. 단, 코드북 학습 시 L2 K-means 대신 **내적 최대화 기준**이 필요 (OPQ).

**코사인 유사도**: 벡터를 L2 정규화하면 $\|\mathbf{a} - \mathbf{b}\|^2 = 2 - 2\cos(\mathbf{a}, \mathbf{b})$이므로 L2 PQ로 코사인 검색 가능.

---

## 7. 메모리 분석

### 7.1 저장 구조

| 항목 | 크기 | 비고 |
|---|---|---|
| 코드 | $N \cdot M$ 바이트 | $N$에 비례 (지배항) |
| 코드북 | $M \cdot K' \cdot d_s \cdot 8$ 바이트 | $N$과 무관 (상수) |
| 라벨 | 가변 | 문자열 저장 |

### 7.2 실험 결과 (step73 테스트)

$D = 128$, $M = 8$, $K' = 256$, $N = 1000$:
- 원본: $1{,}024{,}000$ 바이트 (1MB)
- 코드: $8{,}000$ 바이트
- 코드북: $262{,}144$ 바이트
- 합계: $270{,}144$ 바이트 → **3.8배 압축**

$N = 10^6$이면:
- 원본: $1{,}024{,}000{,}000$ 바이트 (~1GB)
- 코드: $8{,}000{,}000$ 바이트
- 코드북: $262{,}144$ 바이트 (무시 가능)
- 합계: ~$8{,}262{,}144$ 바이트 (~8MB) → **124배 압축**

### 7.3 압축률 공식

$$
\rho = \frac{N \cdot M + M \cdot K' \cdot d_s \cdot 8}{N \cdot D \cdot 8} = \frac{1}{8 \cdot d_s} + \frac{K'}{N}
$$

$N \gg K'$이면:

$$
\rho \approx \frac{1}{8 \cdot d_s} = \frac{M}{8D}
$$

압축 배율: $1/\rho \approx \frac{8D}{M}$

---

## 8. Recall 실험 결과

### 8.1 M 트레이드오프 (D=16, K'=16, N=300)

| M | recall@10 | 압축 배율 | 코드 바이트 |
|---|---|---|---|
| 2 | 0.300 | 14.5x | 600 |
| 4 | 0.450 | 11.8x | 1200 |
| 8 | 0.770 | 8.6x | 2400 |
| 16 | 0.870 | 5.6x | 4800 |

**핵심 관찰**: M이 증가하면 recall이 단조 증가하지만, 압축률은 감소한다. $M = D$일 때 양자화 오차가 최소 (각 서브공간이 1차원)이지만 압축 효과도 최소.

### 8.2 Recall 하한 분석

주어진 양자화 오차 $\epsilon = \mathbb{E}[\|\mathbf{x} - \hat{\mathbf{x}}\|]$에서, ADC 검색의 recall 하한:

$$
\mathrm{Recall@k} \geq 1 - k \cdot P\left[\left|\|\mathbf{q} - \hat{\mathbf{x}}\| - \|\mathbf{q} - \mathbf{x}\|\right| > \delta\right]
$$

삼각 부등식에 의해 $\left|\|\mathbf{q} - \hat{\mathbf{x}}\| - \|\mathbf{q} - \mathbf{x}\|\right| \leq \|\mathbf{x} - \hat{\mathbf{x}}\| = \epsilon$이므로, $\delta > \epsilon$이면 위 확률이 0. 즉 양자화 오차보다 이웃 간 거리 차이가 충분히 크면 recall = 1.

---

## 9. OPQ (Optimized Product Quantization)

PQ는 차원을 순서대로 분할하지만, 이것이 최적은 아니다. 차원 간 상관관계가 있으면 분할 전 **회전(rotation)**을 적용하여 서브공간 간 독립성을 높일 수 있다.

### 9.1 문제 정의

PQ의 총 양자화 오차:

$$
E_{\mathrm{PQ}} = \sum_{i=1}^{N} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 = \sum_{i=1}^{N} \sum_{m=0}^{M-1} \|\mathbf{x}_i^{(m)} - c^{(m)}_{q_m(i)}\|^2
$$

차원 순서를 바꾸거나 회전을 적용하면 서브공간 분할이 달라지고 오차도 변한다. 최적 회전 $R^*$:

$$
R^* = \arg\min_{R \in O(D)} \sum_{i=1}^{N} \|R\mathbf{x}_i - \hat{R\mathbf{x}_i}\|^2
$$

여기서 $\hat{R\mathbf{x}_i}$는 회전된 벡터의 PQ 재구성.

### 9.2 교대 최적화 (Alternating Optimization)

$R$과 코드북을 동시에 최적화하는 것은 어렵다. 대신 교대로:

**Step A** ($R$ 고정): 회전된 벡터 $\mathbf{y}_i = R\mathbf{x}_i$에 대해 표준 PQ 학습 → 코드북 + 코드 갱신

**Step B** (코드북 고정): 재구성 벡터 $\hat{\mathbf{y}}_i$ 고정. 최적 $R$은:

$$
R^* = \arg\min_{R \in O(D)} \sum_i \|R\mathbf{x}_i - \hat{\mathbf{y}}_i\|^2 = \arg\min_{R} \|RX - \hat{Y}\|_F^2
$$

여기서 $X, \hat{Y} \in \mathbb{R}^{D \times N}$.

### 9.3 Procrustes 풀이

위 문제는 **직교 Procrustes 문제**. 해:

$$
X\hat{Y}^T = U\Sigma V^T \quad (\mathrm{SVD})
$$

$$
R^* = VU^T
$$

**증명**: $\|RX - \hat{Y}\|_F^2 = \|X\|_F^2 + \|\hat{Y}\|_F^2 - 2\mathrm{tr}(R X \hat{Y}^T)$. $\mathrm{tr}(R X \hat{Y}^T)$를 최대화하려면 Von Neumann의 trace 부등식에 의해 $R = VU^T$가 최적.

### 9.4 수렴 성질

- 각 단계에서 목적 함수(총 양자화 오차)가 감소
- 오차는 0 이상으로 하한 → **단조 수렴 보장**
- 전역 최적은 보장하지 않음 (non-convex)
- 실무에서 3-5회 반복이면 충분

OPQ는 PQ 대비 **5-15% recall 향상**. Faiss에서 `OPQ` 전처리로 제공.

---

## 10. IVF-PQ: IVF + PQ 결합

실무에서 가장 많이 사용하는 조합:

```
쿼리 → IVF coarse search (nprobe개 클러스터)
     → 해당 클러스터의 PQ 코드만 ADC로 검색
     → top-k 반환
```

**Residual PQ**: IVF 중심으로부터의 **잔차(residual)**를 PQ로 양자화:

$$
\mathbf{r}_i = \mathbf{x}_i - c_{\mathrm{assign}(i)}
$$

잔차는 원본 벡터보다 분산이 작으므로 동일 비트 수에서 양자화 오차가 감소.

### 10.1 잔차 기반 거리 계산

쿼리 $\mathbf{q}$와 DB 벡터 $\mathbf{x}_i$ (클러스터 $k$에 할당) 사이의 거리:

$$
\|\mathbf{q} - \mathbf{x}_i\|^2 = \|\mathbf{q} - c_k - \mathbf{r}_i\|^2 = \|(\mathbf{q} - c_k) - \mathbf{r}_i\|^2
$$

$\mathbf{q} - c_k$를 **쿼리 잔차**라 하면, 잔차 공간에서의 거리가 된다. 이 잔차 벡터에 PQ의 ADC를 적용:

$$
\|(\mathbf{q} - c_k) - \mathbf{r}_i\|^2 \approx \sum_{m=0}^{M-1} T_k[m][\mathrm{code}_i[m]]
$$

여기서 $T_k[m][j] = \|(\mathbf{q} - c_k)^{(m)} - c^{(m)}_j\|^2$. 클러스터마다 거리 테이블이 다르다 (쿼리 잔차가 다르므로).

### 10.2 왜 잔차가 더 좋은가?

원본 벡터의 분산: $\mathrm{Var}[\mathbf{x}] = \sigma^2_{\mathrm{total}}$

잔차의 분산: $\mathrm{Var}[\mathbf{r}] = \sigma^2_{\mathrm{total}} - \sigma^2_{\mathrm{between}} = \sigma^2_{\mathrm{within}}$

K-means의 목적 함수가 within-cluster 분산을 최소화하므로:

$$
\sigma^2_{\mathrm{within}} < \sigma^2_{\mathrm{total}}
$$

분산이 작으면 동일한 $K'$개 중심으로 더 정밀한 양자화가 가능. 이것이 **IVF-PQ > 순수 PQ**인 이유.

**복잡도**: $O(\mathrm{nprobe} \cdot \frac{N}{K} \cdot M)$ — IVF와 PQ의 장점 결합.

| 항목 | Brute Force | IVF | PQ | IVF-PQ |
|---|---|---|---|---|
| 메모리 | $O(ND)$ | $O(ND)$ | $O(NM)$ | $O(NM)$ |
| 쿼리 속도 | $O(ND)$ | $O(\frac{\mathrm{nprobe} \cdot N}{K} \cdot D)$ | $O(NM)$ | $O(\frac{\mathrm{nprobe} \cdot N}{K} \cdot M)$ |
| 정확도 | Exact | nprobe↑ → exact | M↑ → 높음 | nprobe↑, M↑ → 높음 |

---

## 11. 구현 코드 발췌

### 11.1 PQIndex 구조체

```rust
pub struct PQIndex {
    dim: usize,                        // 전체 벡터 차원 D
    n_sub: usize,                      // 서브벡터 개수 M
    n_sub_clusters: usize,             // 서브공간별 클러스터 수 K' (≤256)
    sub_dim: usize,                    // ds = D / M
    codebooks: Vec<Vec<Vec<f64>>>,     // [M][K'][ds]
    codes: Vec<Vec<u8>>,               // [N][M] — 원본 벡터 미저장!
    labels: Vec<String>,
    is_trained: bool,
}
```

핵심: `codes`만 저장하고 원본 벡터는 폐기. BruteForce/IVF의 `vectors: Vec<Vec<f64>>`와 대조적.

### 11.2 코드북 학습

```rust
pub fn train(&mut self, vectors: &[&[f64]], max_iter: usize, seed: u64) {
    for m in 0..self.n_sub {
        let start = m * self.sub_dim;
        let end = start + self.sub_dim;

        // 서브벡터 추출
        let sub_vecs: Vec<Vec<f64>> = vectors.iter()
            .map(|v| v[start..end].to_vec()).collect();
        let sub_refs: Vec<&[f64]> = sub_vecs.iter().map(|v| v.as_slice()).collect();

        // Step 72의 kmeans() 재사용
        let seed_m = seed.wrapping_add(m as u64);
        let (centroids, _) = kmeans(&sub_refs, self.n_sub_clusters, max_iter, seed_m);
        codebooks.push(centroids);
    }
}
```

### 11.3 ADC 검색

```rust
pub fn search(&self, query: &[f64], k: usize) -> Vec<(usize, f64, String)> {
    // 1. 거리 테이블: dist_table[m][j] = ||q_m - c_m_j||²
    let mut dist_table: Vec<Vec<f64>> = Vec::with_capacity(self.n_sub);
    for m in 0..self.n_sub {
        let q_sub = &query[m * self.sub_dim..(m+1) * self.sub_dim];
        let table: Vec<f64> = self.codebooks[m].iter().map(|centroid| {
            q_sub.iter().zip(centroid.iter())
                .map(|(a, b)| { let d = a - b; d * d }).sum()
        }).collect();
        dist_table.push(table);
    }

    // 2. 코드 스캔: M번 테이블 룩업으로 근사 거리
    let mut scores: Vec<(usize, f64)> = self.codes.iter().enumerate().map(|(i, code)| {
        let dist: f64 = (0..self.n_sub)
            .map(|m| dist_table[m][code[m] as usize]).sum();
        (i, dist)
    }).collect();

    // 3. 정렬 → top-k
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    // ...
}
```

---

## 12. Squared L2 vs L2

PQ의 ADC는 **squared L2** ($\|\mathbf{q} - \mathbf{x}\|^2$)를 반환한다. sqrt를 생략하는 이유:

1. **순서 보존**: $\|a\|^2 < \|b\|^2 \iff \|a\| < \|b\|$ (단조 함수)
2. **성능**: $M \cdot K'$번의 sqrt 연산 제거
3. **정확도**: 부동소수점 오차 감소

따라서 search()의 반환값 score는 **실제 거리가 아닌 squared 거리**. 실제 거리가 필요하면 `score.sqrt()`을 호출.

---

## 13. 고차원에서의 PQ 성능

### 13.1 차원의 저주와 PQ

고차원에서 모든 점 쌍의 거리가 비슷해지는 **concentration of measure**:

$$
\frac{\max_i d(q, x_i) - \min_i d(q, x_i)}{\min_i d(q, x_i)} \to 0 \quad \text{as } D \to \infty
$$

PQ는 서브공간으로 분할하므로, 각 서브공간은 $d_s = D/M$ 차원. $d_s$가 작으면 concentration이 덜 심해 양자화가 더 효과적이다. 이것이 $M$을 키울 때 recall이 향상되는 이유.

### 13.2 최적 서브공간 차원

경험적으로 $d_s \in [4, 16]$이 좋은 성능을 보인다:
- $d_s < 4$: K-means가 유의미한 클러스터를 찾기 어려움
- $d_s > 16$: concentration 효과로 양자화 정밀도 감소

$D = 128$이면 $M \in [8, 32]$이 실용적 범위.

---

## 14. Faiss의 PQ 구현과 비교

### 14.1 SIMD 최적화

Faiss는 ADC의 코드 스캔을 **SIMD (SSE/AVX)**로 병렬화:
- 4/8/16개 코드를 동시에 룩업하여 거리 합산
- 고정 $K' = 256$으로 테이블이 L1 캐시에 적합 ($M \cdot 256 \cdot 4$ 바이트, float32)

### 14.2 Polysemous Codes

코드 자체를 **해밍 거리로 사전 필터링**:
1. 코드북 학습 시 코드의 해밍 거리와 실제 L2 거리가 상관하도록 최적화
2. 검색 시 해밍 거리로 빠르게 후보 축소 → ADC 정밀 검색

이는 PQ의 코드 스캔 단계를 추가 가속.

### 14.3 PQ4 vs PQ8

Faiss의 4비트 PQ ($K' = 16$):
- 코드 2개를 1바이트에 저장 → 추가 2배 압축
- 하지만 양자화 오차 증가
- 교육적 구현에서는 $K' = 256$ (8비트) 기본

---

## 15. PQ의 일반화: AQ와 RQ

### 15.1 Additive Quantization (AQ)

PQ는 서브공간을 **차원 분할**로 구성하지만, AQ는 이를 일반화:

$$
\hat{\mathbf{x}} = \sum_{m=0}^{M-1} c^{(m)}_{q_m}
$$

여기서 각 코드북 $\mathcal{C}^{(m)}$은 **전체 $D$차원**의 중심을 가진다. PQ와 달리 서브공간 분할이 아니라 **중심 벡터의 합**으로 재구성.

$$
\mathrm{PQ}: \hat{\mathbf{x}} = [c^{(0)}_{q_0} \| c^{(1)}_{q_1} \| \cdots] \quad (\mathrm{concatenation})
$$
$$
\mathrm{AQ}: \hat{\mathbf{x}} = c^{(0)}_{q_0} + c^{(1)}_{q_1} + \cdots \quad (\mathrm{addition})
$$

AQ는 PQ보다 표현력이 높지만 (교차 서브공간 상관관계 포착), 인코딩이 NP-hard (beam search 필요). ADC 테이블도 $M \cdot K'$로 동일하게 구성 가능.

### 15.2 Residual Quantization (RQ)

순차적으로 잔차를 양자화하는 계층적 접근:

1. 1단계 양자화: $\hat{\mathbf{x}}_1 = c^{(0)}_{q_0}$, 잔차 $\mathbf{r}_1 = \mathbf{x} - \hat{\mathbf{x}}_1$
2. 2단계 양자화: $\hat{\mathbf{r}}_1 = c^{(1)}_{q_1}$, 잔차 $\mathbf{r}_2 = \mathbf{r}_1 - \hat{\mathbf{r}}_1$
3. 반복...

최종 재구성:

$$
\hat{\mathbf{x}} = \sum_{m=0}^{M-1} c^{(m)}_{q_m}
$$

RQ는 AQ의 특수한 경우 (greedy 인코딩). 각 단계의 잔차가 이전 단계보다 분산이 작으므로 **동일 비트에서 PQ보다 정확**. 하지만 순차 의존성으로 ADC 최적화가 어려움.

### 15.3 PQ ↔ AQ ↔ RQ 관계

$$
\mathrm{PQ} \subset \mathrm{RQ} \subset \mathrm{AQ}
$$

| 방법 | 코드북 구조 | 인코딩 | ADC | 정확도 |
|---|---|---|---|---|
| PQ | 서브공간 분할 | $O(M \cdot K' \cdot d_s)$ | $O(N \cdot M)$ | 기본 |
| RQ | 전체차원, 순차 | $O(M \cdot K' \cdot D)$ | $O(N \cdot M)$ | PQ < RQ |
| AQ | 전체차원, 최적 | NP-hard (beam) | $O(N \cdot M)$ | RQ ≤ AQ |

PQ가 실무에서 지배적인 이유: 인코딩 속도 + ADC 효율 + SIMD 친화성.

---

## 16. 실전 튜닝 가이드

### 16.1 파라미터 선택

| 파라미터 | 권장 범위 | 가이드라인 |
|---|---|---|
| $K'$ | 256 (거의 항상) | u8, SIMD, L1 캐시 최적 |
| $M$ | $D/16$ ~ $D/4$ | $d_s \in [4, 16]$ 유지 |
| 학습 벡터 수 | $\geq 30 \cdot K'$ | K-means 수렴을 위한 최소 |
| max_iter | 20~50 | 수렴 확인 후 조정 |

### 16.2 일반적인 설정 예시

| 유즈케이스 | D | M | K' | 코드 크기 | 압축률 |
|---|---|---|---|---|---|
| SIFT-1M | 128 | 8 | 256 | 8 bytes | 128x |
| OpenAI ada-002 | 1536 | 48 | 256 | 48 bytes | 256x |
| BERT-base | 768 | 32 | 256 | 32 bytes | 192x |
| Cohere v3 | 1024 | 64 | 256 | 64 bytes | 128x |

### 16.3 성능 진단

- **Recall이 낮으면**: M 증가 (서브공간 해상도↑) 또는 OPQ 적용
- **메모리가 부족하면**: M 감소 또는 PQ4 ($K' = 16$) 사용
- **학습이 느리면**: 샘플링으로 학습 데이터 축소 (전체 데이터 불필요)
- **쿼리가 느리면**: IVF-PQ 결합 (nprobe로 스캔 범위 축소)

---

## 17. PQ의 기하학적 직관

### 17.1 서브공간 Voronoi의 곱

VQ는 $D$차원 공간에 $K$개 Voronoi cell을 만든다. PQ는 각 $d_s$차원 서브공간에 $K'$개 Voronoi cell을 만들고, 전체 공간의 양자화 셀은 이들의 **직적(Cartesian product)**:

$$
V_{(j_0, j_1, \ldots, j_{M-1})} = V^{(0)}_{j_0} \times V^{(1)}_{j_1} \times \cdots \times V^{(M-1)}_{j_{M-1}}
$$

2차원 예시 ($D = 2$, $M = 2$, $d_s = 1$):
- 서브공간 0 (x축): $K' = 3$개 구간으로 분할 → 3개 수직 띠
- 서브공간 1 (y축): $K' = 3$개 구간으로 분할 → 3개 수평 띠
- 전체 공간: $3 \times 3 = 9$개 직사각형 셀

```
y  ┌───┬───┬───┐
   │0,2│1,2│2,2│   ← 서브공간 1의 Voronoi
   ├───┼───┼───┤
   │0,1│1,1│2,1│   코드 (j₀, j₁)로 셀 식별
   ├───┼───┼───┤
   │0,0│1,0│2,0│
   └───┴───┴───┘ x
        ↑ 서브공간 0의 Voronoi
```

VQ의 셀은 **임의 형태** (Voronoi polytope)지만, PQ의 셀은 **축 정렬 직사각형**. 이것이 PQ의 한계 — 대각선 방향의 구조를 포착하지 못한다. OPQ의 회전은 이 직사각형을 데이터에 맞게 기울이는 것.

### 17.2 왜 작동하는가: 차원 독립의 근사

실제 임베딩 벡터는 완전히 독립적이지 않지만, 고차원에서 **대부분의 분산이 소수 차원에 집중**된다 (PCA 관점).

PQ의 각 서브공간이 분산의 일부를 담당하면, 서브공간 내 양자화가 전체 양자화의 좋은 근사가 된다. 차원 간 상관관계가 강하면 (예: 인접 차원이 비슷한 값을 가짐) OPQ로 decorrelation 후 PQ를 적용하는 것이 효과적.

### 17.3 ADC의 기하학적 의미

ADC의 거리 테이블 $T[m][j]$는 **쿼리에서 각 Voronoi cell 중심까지의 부분 거리**. 코드 스캔은 이 부분 거리들을 합산하여 전체 거리를 근사한다:

$$
d_{\mathrm{ADC}}(\mathbf{q}, \mathbf{x}) = \sum_m T[m][\mathrm{code}[m]] = \sum_m \|\mathbf{q}^{(m)} - c^{(m)}_{\mathrm{code}[m]}\|^2
$$

기하학적으로, 쿼리에서 **각 서브공간의 가장 가까운 코드북 중심을 거치는 경로**의 길이의 제곱합. 실제 벡터 대신 중심으로 "스냅"한 후의 거리.

---

## 18. Re-ranking: 2단계 검색 패턴

실무에서 PQ는 단독이 아닌 **1단계 필터링**으로 사용하고, 2단계에서 정밀 re-ranking:

```
1단계: PQ/IVF-PQ로 top-k' 후보 추출 (k' >> k, 예: k'=1000)
2단계: 원본 벡터로 top-k' 후보만 정확한 거리 계산 → top-k 반환
```

### 18.1 왜 필요한가

PQ의 ADC는 근사 거리이므로 순위가 뒤집힐 수 있다. 특히 경계에 있는 벡터들:

$$
\mathrm{Recall@k'} > \mathrm{Recall@k} \quad (k' > k)
$$

충분히 큰 $k'$를 잡으면 true top-k가 후보에 포함될 확률이 높아진다.

### 18.2 최적 k' 선택

Oversampling ratio $\alpha = k'/k$. 일반적으로:
- $\alpha = 5\text{-}10$: 대부분의 경우 충분
- Recall@k'가 0.99 이상이면 re-ranking 후 거의 exact

**비용**: 원본 벡터 $k'$개 로드 + $k' \cdot D$ 거리 계산. $k' \ll N$이므로 총 비용 미미.

### 18.3 원본 벡터 저장 전략

| 전략 | 메모리 | 장점 |
|---|---|---|
| 전체 저장 (메모리) | $N \cdot D \cdot 8$ | 즉시 접근 |
| 디스크 저장 + mmap | 디스크 | 메모리 절약, 랜덤 I/O |
| 별도 스토리지 (S3 등) | 없음 | 무제한 확장, 지연 증가 |

Faiss의 `IndexIVFPQR` (PQ + Refine): PQ로 검색 후 원본 벡터로 re-rank.

---

## 19. ScaNN: Anisotropic Quantization

Google Research (2020)의 **ScaNN** (Scalable Nearest Neighbors)은 PQ의 근본적 한계를 지적하고 **비등방적(anisotropic) 양자화**를 제안.

### 19.1 PQ의 등방적 문제

표준 PQ는 양자화 오차를 **등방적(isotropic)**으로 최소화:

$$
\min \mathbb{E}[\|\mathbf{x} - \hat{\mathbf{x}}\|^2]
$$

하지만 검색에서 중요한 것은 **쿼리 방향**의 오차. 쿼리와 평행한 방향의 양자화 오차는 순위를 뒤집지만, 수직 방향의 오차는 영향이 적다.

### 19.2 비등방적 손실 함수

내적 검색 ($\langle \mathbf{q}, \mathbf{x} \rangle$)에서, ScaNN은 양자화 오차를 방향에 따라 가중:

$$
\min \mathbb{E}\left[\left(\langle \mathbf{q}, \mathbf{x} \rangle - \langle \mathbf{q}, \hat{\mathbf{x}} \rangle\right)^2\right] = \min \mathbb{E}\left[\langle \mathbf{q}, \mathbf{x} - \hat{\mathbf{x}} \rangle^2\right]
$$

이를 전개하면:

$$
= \mathbb{E}_{\mathbf{q}}\left[(\mathbf{x} - \hat{\mathbf{x}})^T \mathbf{q}\mathbf{q}^T (\mathbf{x} - \hat{\mathbf{x}})\right] = (\mathbf{x} - \hat{\mathbf{x}})^T \underbrace{\mathbb{E}[\mathbf{q}\mathbf{q}^T]}_{\Sigma_q} (\mathbf{x} - \hat{\mathbf{x}})
$$

$\Sigma_q$가 단위 행렬이면 등방적 (표준 PQ), 아니면 비등방적. ScaNN은 $\hat{\mathbf{x}} = \eta \cdot \tilde{\mathbf{x}}$로 스케일링하여 bias-variance 트레이드오프를 제어 ($\eta < 1$이면 중심 쪽으로 축소).

### 19.3 성능

SIFT-1M, GloVe 등에서 PQ 대비 **10-20% recall 향상** (동일 메모리). 특히 내적/코사인 메트릭에서 효과가 큼. Google 내부에서 대규모 배포.

---

## 20. Quantization-Aware Training

임베딩 모델 학습 시 PQ의 양자화 오차를 고려하는 접근.

### 20.1 문제

일반 임베딩 모델은 연속 벡터 공간에서 학습되지만, 서빙 시 PQ로 양자화된다. 학습과 서빙의 불일치:

$$
\mathcal{L}_{\mathrm{train}} = f(\mathbf{x}_i, \mathbf{x}_j), \quad \mathcal{L}_{\mathrm{serve}} = f(\hat{\mathbf{x}}_i, \hat{\mathbf{x}}_j)
$$

### 20.2 접근법

**End-to-end 학습**: 코드북을 미분 가능하게 만들어 역전파:

$$
\hat{\mathbf{x}} = \sum_{m} \mathrm{softmax}(-\|\mathbf{x}^{(m)} - \mathcal{C}^{(m)}\|^2 / \tau) \cdot \mathcal{C}^{(m)}
$$

temperature $\tau \to 0$이면 hard assignment (실제 PQ)에 수렴. 학습 중에는 soft assignment로 그래디언트 전파, 서빙 시에는 hard assignment.

**Straight-through estimator (STE)**: forward에서 hard assignment, backward에서 identity gradient:

$$
\frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{x}} \approx I
$$

### 20.3 효과

양자화 오차를 학습 목표에 포함하면, 모델이 **PQ-친화적 임베딩**을 학습한다:
- 서브공간 내 차원 간 독립성 증가
- 코드북 중심 근처에 벡터가 분포
- 동일 코드에서 recall 5-10% 향상

---

## 21. Multi-Index ADC (SIMD 내부 구현)

Faiss의 핵심 성능 최적화인 **PQ Fast Scan** 내부 동작.

### 21.1 기본 ADC의 병목

기본 ADC 코드 스캔:
```
for each vector i:
    dist = 0
    for m = 0..M:
        dist += table[m][code[i][m]]   // 랜덤 메모리 접근
```

$M$번의 **랜덤 테이블 룩업**이 병목. 특히 $K' = 256$이면 테이블이 L1 캐시보다 클 수 있다.

### 21.2 PQ4 + SIMD 패킹

Faiss의 PQ Fast Scan ($K' = 16$, 4비트 코드):

1. **니블 패킹**: 코드 2개를 1바이트에 저장 (상위/하위 4비트)
2. **SIMD 셔플 룩업**: AVX2의 `vpshufb`로 16개 엔트리 테이블을 **한 번에 룩업**
3. **벡터 누적**: 16개 벡터의 부분 거리를 동시에 합산

```
// 의사코드: 16개 벡터 동시 처리
simd_table = load_16_entries(table[m])  // 16개 중심 거리
codes_16 = load_16_codes(codes[i..i+16])
partial_dist = vpshufb(simd_table, codes_16)  // SIMD 룩업
accum += partial_dist
```

### 21.3 성능 비교

| 방법 | 처리량 | 메모리 대역폭 |
|---|---|---|
| 기본 ADC (PQ8) | ~10M vec/s | 테이블 캐시 미스 |
| SIMD ADC (PQ4) | ~500M vec/s | L1 캐시 내 |
| GPU ADC | ~1B vec/s | HBM 대역폭 |

PQ4의 양자화 오차 증가를 PQ Fast Scan의 속도로 보상: **더 많은 후보를 탐색 → re-ranking으로 정확도 회복**.

---

## 22. 벡터 DB에서의 PQ 활용

| DB | PQ 사용 | 특징 |
|---|---|---|
| **Faiss** | IVF-PQ, OPQ, Polysemous | 참조 구현, CPU/GPU |
| **Milvus** | IVF-PQ, HNSW-PQ | 분산 아키텍처, Faiss 기반 |
| **Qdrant** | Scalar/PQ | 이진 PQ + oversampling |
| **Pinecone** | 내부 PQ | SaaS, 자동 튜닝 |
| **Weaviate** | PQ | Compressed vectors |

Faiss의 `IndexIVFPQ`가 사실상 표준. 10억 벡터 규모에서 밀리초 단위 검색.

---

## 23. Phase 4 로드맵

| Step | 주제 | 핵심 | 상태 |
|---|---|---|---|
| 71 | Brute Force | 정확한 기준선, O(ND) | ✅ |
| 72 | IVF | K-means 분할, nprobe 트레이드오프 | ✅ |
| **73** | **PQ** | **서브공간 양자화, ADC, 메모리 압축** | **✅** |
| 74 | HNSW | 그래프 기반 검색, O(log N) | 예정 |
