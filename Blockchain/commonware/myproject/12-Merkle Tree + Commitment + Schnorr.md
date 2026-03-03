## Step 08: Merkle Tree + Commitment + Schnorr 서명

### 세 가지 원시 도구 — 왜 이 조합인가?

```
블록체인/ZK의 모든 응용은 세 가지 원시 도구의 조합이다:

  1. Merkle Tree    → "이 데이터가 집합에 포함되어 있음"을 증명
  2. Commitment     → "값을 숨기되, 나중에 열 수 있음"을 보장
  3. Digital Signature → "이 메시지는 내가 보냈음"을 인증

응용 예시:
  ┌──────────────────┬──────────┬────────────┬───────────┐
  │ 응용              │ Merkle   │ Commitment │ Signature │
  ├──────────────────┼──────────┼────────────┼───────────┤
  │ 블록체인 상태      │ state tree│            │ tx 서명    │
  │ Mixer (Tornado)  │ 멤버십 증명│ note 은닉   │           │
  │ Rollup           │ state root│            │ batch 서명 │
  │ 익명 투표         │ 그룹 멤버십│ 투표 은닉   │           │
  │ Private Transfer │ UTXO 증명 │ value 은닉  │           │
  └──────────────────┴──────────┴────────────┴───────────┘
```

> [!important] "원시 도구 세트"의 의미
> Step 07까지는 **기반 암호학**(체, 커브, 페어링, 해시)을 만들었다.
> Step 08의 세 가지는 그 기반 위에 세워진 **최초의 응용 계층** —
> 이후의 모든 것(chain, privacy, rollup)이 이것들을 직접 사용한다.
>
> ```
> 기반 (Step 01-07):
>   Fr → Fp → Fp2 → ... → Fp12 → G1 → G2 → Pairing → Poseidon
>
> 응용 도구 (Step 08, 여기):
>   Poseidon → Merkle Tree
>   Fr       → Commitment (Poseidon 기반)
>   G1 + Fr  → Schnorr Signature
>
> 프로토콜 (Step 09+):
>   도구들의 조합 → R1CS → Groth16 → Chain → Privacy → ...
> ```

---

## Part 1: Sparse Merkle Tree

### 머클 트리란?

```
"대량의 데이터를 하나의 해시값(root)으로 요약하고,
 특정 데이터의 포함 여부를 효율적으로 증명하는 자료구조"

핵심 아이디어:
  N개의 데이터를 이진 트리로 조직하면,
  하나의 데이터에 대한 증명은 log₂(N)개의 해시값만 필요하다.
```

#### 기본 구조: 이진 해시 트리

```
데이터: [A, B, C, D]  (4개 리프)

        root = H(H₁₂, H₃₄)
       /                    \
  H₁₂ = H(H₁, H₂)      H₃₄ = H(H₃, H₄)
    /        \              /        \
  H₁=H(A)  H₂=H(B)    H₃=H(C)  H₄=H(D)
    |         |           |         |
    A         B           C         D

여기서 H = poseidon_hash (2-to-1 해시)
```

> [!abstract] 머클 트리의 세 가지 핵심 성질
> 1. **요약(Digest)**: root 하나로 전체 데이터 집합을 표현
> 2. **멤버십 증명(Membership Proof)**: 특정 데이터가 트리에 있음을 증명
> 3. **변조 감지(Tamper Detection)**: 데이터 하나라도 바뀌면 root가 변경
>
> ```
> 복잡도:
>   저장:  O(N) 노드
>   증명:  O(log N) 해시값
>   검증:  O(log N) 해시 연산
>   갱신:  O(log N) 해시 재계산
> ```

---

### Sparse vs Dense Merkle Tree

```
Dense Merkle Tree (밀집):
  모든 리프가 실제 데이터
  리프 수 = 데이터 수
  예: [A, B, C, D] → 4개 리프, 높이 2

Sparse Merkle Tree (희소):
  리프 공간이 고정 크기 (예: 2^256개)
  대부분의 리프는 "비어 있음" (기본값 = 0)
  소수의 리프만 실제 값을 가짐
  키(key) → 인덱스 매핑으로 접근

  예: 높이 256, key = Fr 원소
      2^256개 가능한 위치 중 실제로 사용되는 건 수십~수백 개
```

#### 왜 Sparse Merkle Tree를 쓰는가?

```
Dense Tree의 한계:
  ✗ 데이터 추가 시 리프 위치가 변경될 수 있음
  ✗ "이 데이터가 트리에 없음"을 증명하기 어려움
  ✗ 임의의 키-값 매핑을 지원하지 않음

Sparse Tree의 장점:
  ✓ 키 → 인덱스가 결정론적 (해시로 매핑)
  ✓ non-membership proof가 자연스러움 (기본값인지 확인)
  ✓ 키-값 저장소처럼 사용 가능 (계정 상태 등)
  ✓ 트리 구조가 입력 순서에 무관 (같은 집합 → 같은 root)

블록체인에서의 사용:
  key = 계정 주소 (Fr 원소)
  value = 계정 상태 (balance, nonce 등의 해시)
  root = "전체 상태의 요약" → 블록 헤더에 포함
```

> [!tip] 이더리움의 선택
> 이더리움은 Merkle Patricia Trie (MPT)를 사용한다.
> MPT는 Sparse Merkle Tree + Patricia Trie(경로 압축)의 조합.
> 우리는 교육용으로 순수 Sparse Merkle Tree를 구현 —
> 개념은 동일하지만 구조가 더 단순하다.

---

### Sparse Merkle Tree의 구조

```
높이 depth = 256 (Fr 원소가 254-bit이므로 충분)
  → 2^256개의 가능한 리프 위치

키(key)의 비트 표현이 트리 경로를 결정:
  key = b₂₅₅ b₂₅₄ b₂₅₃ ... b₁ b₀  (256비트)

  root에서 시작:
    b₂₅₅ = 0 → 왼쪽, 1 → 오른쪽
    b₂₅₄ = 0 → 왼쪽, 1 → 오른쪽
    ...
    b₀에서 리프 도달

                      root
                    /      \
            b₂₅₅=0          b₂₅₅=1
              /  \              /  \
        b₂₅₄=0  b₂₅₄=1  b₂₅₄=0  b₂₅₄=1
          ...     ...      ...     ...
                          (256 레벨)
```

#### 핵심 최적화: 기본 해시 (Default Hash)

```
2^256개 리프를 실제로 저장하는 것은 불가능!
→ "비어 있는 서브트리"는 미리 계산된 기본 해시로 대체

기본 해시 배열 default[0..256]:
  default[0]   = Fr::ZERO           (빈 리프의 값)
  default[1]   = H(default[0], default[0])   (빈 높이-1 서브트리)
  default[2]   = H(default[1], default[1])   (빈 높이-2 서브트리)
  ...
  default[256] = H(default[255], default[255]) (완전히 빈 트리의 root)

이 기본 해시들은 트리 생성 시 한 번만 계산하면 된다.
실제로 값이 있는 경로만 저장 → O(N × depth) 공간 (N = 실제 데이터 수)
```

> [!important] 기본 해시가 핵심인 이유
> Sparse Merkle Tree의 효율성은 전적으로 이 최적화에 달려 있다.
>
> ```
> 저장하지 않는 것:
>   2^256 - N개의 빈 리프와 그 조상 노드들
>   → 기본 해시로 대체 가능하므로 저장 불필요
>
> 실제 저장하는 것:
>   N개의 실제 데이터 + 경로상의 O(N × 256) 노드
>   → HashMap으로 관리
>
> 1억 개의 계정이 있어도:
>   저장: ~256 × 10^8 해시값 ≈ 25.6 × 10^9 (관리 가능)
>   빈 공간: 2^256 - 10^8 ≈ 2^256 (저장 불필요!)
> ```

---

### 머클 증명 (Merkle Proof)

#### 증명의 구조

```
"key K에 value V가 저장되어 있음"을 증명하려면:

MerkleProof {
    siblings: [Fr; depth]    // 경로상의 형제 노드 해시값들
    path_bits: [bool; depth] // 각 레벨에서 좌/우 방향
}

증명 크기: depth개의 Fr 원소 = 256 × 32바이트 = 8KB
  → 트리에 2^256개의 리프가 있어도 증명은 항상 8KB!
```

#### 증명 생성 과정

```
예시: depth=4, key의 비트 = [1, 0, 1, 0] (MSB → LSB)

                root
              /      \
            N₃        S₃ ← sibling[3] (root의 오른쪽 자식)
          /    \
        S₂     N₂ ← sibling[2]를 거쳐
              /    \
            N₁      S₁ ← sibling[1]
          /    \
        S₀     leaf(V) ← 증명 대상
               ↑
          sibling[0]

siblings = [S₀, S₁, S₂, S₃]  (리프부터 root까지)
path_bits = [0, 1, 0, 1]      (key의 비트, LSB부터)

각 S는 "경로상에서 반대편 형제의 해시값"
```

#### 증명 검증 과정

```
검증: root를 재구성하여 알려진 root와 비교

입력:
  root     : 알려진 머클 루트
  key      : 검증할 키
  value    : 검증할 값
  siblings : [S₀, S₁, ..., S_{d-1}]

알고리즘:
  current = H(key, value)     // 리프 해시

  for i in 0..depth:
    bit = key의 i번째 비트
    if bit == 0:
      current = H(current, siblings[i])   // current가 왼쪽
    else:
      current = H(siblings[i], current)   // current가 오른쪽

  return current == root      // 재구성한 root와 비교
```

> [!abstract] 왜 이것이 안전한가?
> **Poseidon 해시의 collision resistance**에 의존한다.
>
> ```
> 공격 시나리오: 공격자가 V' ≠ V를 포함하면서 같은 root를 만들려 함
>
> H(key, V') ≠ H(key, V)  (collision resistance)
>   → 리프 해시가 다름
>   → 경로를 따라 올라가면 모든 조상 노드가 달라짐
>   → root가 다름
>
> 결론: 유효한 proof를 위조하려면 Poseidon의 collision을 찾아야 함
>       → 2^127 연산 필요 (사실상 불가능)
> ```

---

### 머클 증명의 수학적 기초

#### 정리: 머클 증명의 완전성과 건전성

```
완전성 (Completeness):
  "실제로 트리에 있는 값은 항상 유효한 증명을 생성할 수 있다"

  증명:
    key K에 value V가 저장되어 있으면,
    insert 과정에서 경로상의 모든 노드가 올바르게 계산됨.
    → siblings를 수집하면 유효한 증명이 됨.
    → verify(root, K, V, proof) = true  ✓

건전성 (Soundness):
  "트리에 없는 값에 대해 유효한 증명을 만드는 것은
   해시 함수의 collision을 찾는 것만큼 어렵다"

  증명 (귀류법):
    V' ≠ V이면서 verify(root, K, V', proof') = true인
    proof'가 존재한다고 가정.

    그러면 리프 해시 H(K, V') ≠ H(K, V) (collision resistance)
    경로를 따라 올라가면 어느 한 레벨 i에서:
      H(a', b') = H(a, b) 이면서 (a', b') ≠ (a, b)
    이것은 Poseidon의 collision
    → 발생 확률 < 2^(-127) □
```

#### Non-membership Proof

```
Sparse Merkle Tree의 고유한 장점:
  "key K에 아무 값도 저장되어 있지 않음"도 증명 가능!

방법:
  1. key K의 경로를 따라 리프까지 내려감
  2. 리프 값이 기본값(Fr::ZERO)임을 확인
  3. 같은 siblings로 증명 생성
  4. verify(root, K, ZERO, proof) = true
     → "K 위치에 아무것도 없다"

이것이 Dense Merkle Tree로는 불가능한 이유:
  Dense Tree에서 "K가 없다" → 어디에 있어야 하는지조차 불분명
  Sparse Tree에서 "K가 없다" → K 위치의 리프가 ZERO
```

---

### 머클 트리: 실제 값으로 수동 계산 추적

> [!important] 왜 수동 추적이 필요한가?
> 알고리즘을 이해하는 가장 확실한 방법은 **실제 값을 넣고 한 단계씩 따라가는 것**이다.
> 여기서는 depth=4인 작은 트리에 key=5, value=42를 삽입하고,
> 증명을 생성하고, 검증하는 전체 과정을 추적한다.
>
> 실제 Poseidon 해시값은 254-bit 수이므로 표기가 불가능하다.
> 대신 구조를 명확히 보여주기 위해 `H₁, H₂, ...` 같은 기호를 사용하되,
> 해시의 **입력이 무엇인지**를 정확히 표기한다.

#### Step 1: 트리 초기화 — 기본 해시 계산

```
depth = 4인 빈 트리 생성:

  default[0] = Fr::ZERO = 0                              (빈 리프)
  default[1] = H(default[0], default[0]) = H(0, 0)       = D₁
  default[2] = H(default[1], default[1]) = H(D₁, D₁)    = D₂
  default[3] = H(default[2], default[2]) = H(D₂, D₂)    = D₃
  default[4] = H(default[3], default[3]) = H(D₃, D₃)    = D₄

  root = default[4] = D₄  (완전히 빈 트리의 루트)

  시각화 (모든 노드가 기본 해시):

          D₄ (root)
         /         \
       D₃           D₃
      /   \        /   \
    D₂     D₂   D₂     D₂
   / \    / \   / \    / \
  D₁  D₁ D₁ D₁ D₁ D₁ D₁ D₁
  각 D₁ 아래에 2^(depth-4) 크기의 빈 서브트리가 있지만
  depth=4이므로 D₁의 자식이 바로 리프 (default[0] = 0)
```

#### Step 2: Insert(key=5, value=42) — 비트 분석

```
key = Fr::from_u64(5)
key의 비트 표현:  5 = 0b...0000_0101

  key_repr = [5, 0, 0, 0]  (u64 4개의 limb 표현)

  비트 추출 (get_bit):
    bit 0 = (5 >> 0) & 1 = 1    ← level 0에서의 방향
    bit 1 = (5 >> 1) & 1 = 0    ← level 1에서의 방향
    bit 2 = (5 >> 2) & 1 = 1    ← level 2에서의 방향
    bit 3 = (5 >> 3) & 1 = 0    ← level 3에서의 방향

  경로 해석 (리프 → 루트):
    level 0: bit=1 → current는 오른쪽 자식 → sibling은 왼쪽
    level 1: bit=0 → current는 왼쪽 자식  → sibling은 오른쪽
    level 2: bit=1 → current는 오른쪽 자식 → sibling은 왼쪽
    level 3: bit=0 → current는 왼쪽 자식  → sibling은 오른쪽
```

#### Step 3: Insert — 리프 해시 및 경로 갱신

```
leaf_hash = H(key, value) = H(5, 42) = L₅

이제 리프에서 루트까지 올라가며 각 레벨의 부모를 재계산:

Level 0 (리프 레벨):
  node_idx = key_repr >> 0 = [5, 0, 0, 0]     (= ...0101)
  sibling_idx = flip_bit0([5,0,0,0]) = [4, 0, 0, 0]  (= ...0100)
  sibling = nodes.get((0, [4,0,0,0])) → 없음 → default[0] = 0

  bit 0 = 1 → sibling이 왼쪽, current가 오른쪽
  parent = H(sibling, current) = H(0, L₅) = N₁

  parent_idx = key_repr >> 1 = [2, 0, 0, 0]   (= ...0010)
  nodes.insert((1, [2,0,0,0]), N₁)

Level 1:
  node_idx = key_repr >> 1 = [2, 0, 0, 0]     (= ...0010)
  sibling_idx = flip_bit0([2,0,0,0]) = [3, 0, 0, 0]  (= ...0011)
  sibling = nodes.get((1, [3,0,0,0])) → 없음 → default[1] = D₁

  bit 1 = 0 → current가 왼쪽, sibling이 오른쪽
  parent = H(current, sibling) = H(N₁, D₁) = N₂

  parent_idx = key_repr >> 2 = [1, 0, 0, 0]   (= ...0001)
  nodes.insert((2, [1,0,0,0]), N₂)

Level 2:
  node_idx = key_repr >> 2 = [1, 0, 0, 0]     (= ...0001)
  sibling_idx = flip_bit0([1,0,0,0]) = [0, 0, 0, 0]  (= ...0000)
  sibling = nodes.get((2, [0,0,0,0])) → 없음 → default[2] = D₂

  bit 2 = 1 → sibling이 왼쪽, current가 오른쪽
  parent = H(sibling, current) = H(D₂, N₂) = N₃

  parent_idx = key_repr >> 3 = [0, 0, 0, 0]   (= ...0000)
  nodes.insert((3, [0,0,0,0]), N₃)

Level 3:
  node_idx = key_repr >> 3 = [0, 0, 0, 0]     (= ...0000)
  sibling_idx = flip_bit0([0,0,0,0]) = [1, 0, 0, 0]  (= ...0001)
  sibling = nodes.get((3, [1,0,0,0])) → 없음 → default[3] = D₃

  bit 3 = 0 → current가 왼쪽, sibling이 오른쪽
  parent = H(current, sibling) = H(N₃, D₃) = N₄

  parent_idx = key_repr >> 4 = [0, 0, 0, 0]
  nodes.insert((4, [0,0,0,0]), N₄)

root = N₄ = H(H(D₂, H(H(0, L₅), D₁)), D₃)
```

#### Step 4: 트리 상태 시각화

```
Insert(5, 42) 후의 트리 (depth=4):

               N₄ (root) ← 새로운 루트!
              /           \
           N₃              D₃ ← 기본값 (빈 서브트리)
          /    \
        D₂     N₂ ← 변경된 노드
              /    \
            N₁      D₁ ← 기본값
           /   \
         0     L₅ ← leaf_hash = H(5, 42)
         ↑      ↑
      default  key=5의 위치
       [0]     (bit 경로: 1,0,1,0)

  변경된 노드: L₅, N₁, N₂, N₃, N₄ (경로상 5개)
  나머지: 모두 기본 해시 (default[i])로 대체

  경로 해석 (루트에서 리프로):
    N₄ → bit3=0 → 왼쪽 → N₃
    N₃ → bit2=1 → 오른쪽 → N₂
    N₂ → bit1=0 → 왼쪽 → N₁
    N₁ → bit0=1 → 오른쪽 → L₅  ← 도착!
```

#### Step 5: Prove(key=5) — 증명 생성

```
증명 = 각 레벨에서 형제 노드의 해시값을 수집

Level 0:
  node_idx = [5,0,0,0], sibling_idx = [4,0,0,0]
  sibling = nodes.get((0, [4,0,0,0])) → 없음 → default[0] = 0
  siblings[0] = 0

Level 1:
  node_idx = [2,0,0,0], sibling_idx = [3,0,0,0]
  sibling = nodes.get((1, [3,0,0,0])) → 없음 → default[1] = D₁
  siblings[1] = D₁

Level 2:
  node_idx = [1,0,0,0], sibling_idx = [0,0,0,0]
  sibling = nodes.get((2, [0,0,0,0])) → 없음 → default[2] = D₂
  siblings[2] = D₂

Level 3:
  node_idx = [0,0,0,0], sibling_idx = [1,0,0,0]
  sibling = nodes.get((3, [1,0,0,0])) → 없음 → default[3] = D₃
  siblings[3] = D₃

MerkleProof {
    siblings: [0, D₁, D₂, D₃]
}

관찰: key=5 하나만 삽입했으므로 모든 sibling이 기본값이다!
     여러 키를 삽입하면 경로가 겹치는 부분에서 실제 값이 나타남.
```

#### Step 6: Verify(root=N₄, key=5, value=42, proof) — 검증

```
검증자는 트리를 갖고 있지 않다. root, key, value, proof만으로 검증.

1. 리프 해시 재계산:
   current = H(key, value) = H(5, 42) = L₅

2. 레벨별 부모 재계산:

   Level 0: bit=1 → H(siblings[0], current) = H(0, L₅) = N₁
   Level 1: bit=0 → H(current, siblings[1]) = H(N₁, D₁) = N₂
   Level 2: bit=1 → H(siblings[2], current) = H(D₂, N₂) = N₃
   Level 3: bit=0 → H(current, siblings[3]) = H(N₃, D₃) = N₄

3. 비교: current = N₄ == root(= N₄)  ✓ 증명 유효!

핵심 관찰:
  검증 과정은 insert 과정과 정확히 동일한 계산을 수행한다.
  → insert가 올바르게 수행되었으면, 검증은 반드시 통과 (완전성)
  → 값이 하나라도 다르면, 해시 충돌을 찾지 않는 한 root가 달라짐 (건전성)
```

> [!abstract] 해시 전개식으로 보는 전체 구조
> ```
> root = N₄
>      = H(N₃, D₃)
>      = H(H(D₂, N₂), D₃)
>      = H(H(D₂, H(N₁, D₁)), D₃)
>      = H(H(D₂, H(H(0, H(5, 42)), D₁)), D₃)
>
> 여기서:
>   H(5, 42)     = leaf_hash (리프: key와 value의 해시)
>   0            = default[0] (빈 리프, key=4의 위치)
>   D₁ = H(0,0)  = default[1] (빈 서브트리)
>   D₂ = H(D₁,D₁) = default[2] (빈 서브트리)
>   D₃ = H(D₂,D₂) = default[3] (빈 서브트리)
>
> 이 전개식에서 볼 수 있듯이:
>   - H(5, 42)의 값을 바꾸면 → N₁ → N₂ → N₃ → N₄가 모두 변경
>   - 어떤 중간 노드를 변조해도 → 해시 충돌을 찾아야 같은 root 유지
>   - collision resistance → 위조 불가
> ```

#### 두 번째 삽입: Insert(key=3, value=99)

```
key=3의 비트: 3 = 0b0011
  bit 0 = 1, bit 1 = 1, bit 2 = 0, bit 3 = 0

leaf_hash = H(3, 99) = L₃

Level 0:
  node_idx = [3,0,0,0], sibling_idx = [2,0,0,0]
  sibling = default[0] = 0
  bit 0 = 1 → H(0, L₃) = M₁
  → nodes((1, [1,0,0,0])) = M₁

Level 1:
  node_idx = [1,0,0,0], sibling_idx = [0,0,0,0]
  sibling = default[1] = D₁
  bit 1 = 1 → H(D₁, M₁) = M₂
  → nodes((2, [0,0,0,0])) = M₂

Level 2:
  node_idx = [0,0,0,0], sibling_idx = [1,0,0,0]
  sibling = nodes.get((2, [1,0,0,0])) → 있음! = N₂  ← key=5의 경로!
  bit 2 = 0 → H(M₂, N₂) = M₃
  → nodes((3, [0,0,0,0])) = M₃

Level 3:
  node_idx = [0,0,0,0], sibling_idx = [1,0,0,0]
  sibling = default[3] = D₃
  bit 3 = 0 → H(M₃, D₃) = M₄
  → root = M₄

이제 key=5의 증명이 달라진다:
  prove(key=5):
    siblings[0] = default[0] = 0      (변화 없음)
    siblings[1] = default[1] = D₁     (변화 없음)
    siblings[2] = M₂  ← key=3이 삽입되면서 이 노드가 생김!
    siblings[3] = default[3] = D₃     (변화 없음)

핵심: 다른 키의 삽입이 기존 키의 siblings를 변경할 수 있다.
      하지만 검증 로직은 동일 — siblings만 업데이트되면 여전히 유효.
```

---

### Insert 연산

```
insert(key, value):

  1. key의 비트 표현으로 리프 위치 결정

  2. 리프 해시 계산:
     leaf_hash = H(key, value)

  3. 리프에서 root까지 경로 갱신:
     current = leaf_hash
     for i in 0..depth:
       sibling = 해당 레벨의 형제 노드 (없으면 default[i])
       bit = key의 i번째 비트
       if bit == 0:
         current = H(current, sibling)
       else:
         current = H(sibling, current)
       store(level=i+1, index, current)  // 갱신된 노드 저장

  4. root = current

비용: depth회의 Poseidon 해시 = 256 × ~250 제약 (회로에서)
```

> [!note] 리프 해시에 key를 포함하는 이유
> ```
> leaf_hash = H(key, value)  (우리 구현)
>
> vs
>
> leaf_hash = H(value)       (단순 버전)
> ```
>
> key를 포함하지 않으면 **second preimage attack** 가능:
> ```
> 공격: key₁ ≠ key₂이지만 같은 value를 가진 경우
>   H(value) = H(value) → 두 리프의 해시가 동일
>   → 한 키의 증명을 다른 키의 증명으로 사용 가능
>
> key를 포함하면:
>   H(key₁, value) ≠ H(key₂, value) → 각 리프가 고유
> ```

---

### Poseidon이 머클 트리에 적합한 이유

```
머클 트리의 핵심 연산 = 2-to-1 해시를 반복 적용

parent = H(left_child, right_child)

이 연산이 Poseidon과 완벽하게 맞는 이유:
  1. Poseidon은 2-to-1 해시로 설계됨 (rate=2)
  2. 입출력이 모두 Fr → 트리 전체가 Fr 위에서 동작
  3. ZK 회로에서 머클 증명 검증이 효율적:
     depth=256이면 256 × ~250 = ~64,000 제약
     SHA-256이었다면: 256 × ~25,000 = ~6,400,000 제약 (100배!)

Step 10에서 구현할 "Merkle proof 회로 가젯"이 바로 이것:
  poseidon_hash를 depth번 반복하는 회로.
```

> [!tip] 실제 ZK 프로토콜에서의 머클 트리 깊이
>
> | 프로토콜 | 트리 깊이 | 최대 리프 수 | 머클 증명 제약 수 |
> |---------|----------|------------|-----------------|
> | Tornado Cash | 20 | ~10^6 | ~5,000 |
> | Semaphore | 20 | ~10^6 | ~5,000 |
> | Zcash (Sapling) | 32 | ~4×10^9 | ~8,000 |
> | 우리 구현 | 256 | 2^256 | ~64,000 |
>
> 실용적으로는 깊이 20~32면 충분하지만,
> 우리는 교육 목적으로 Fr의 전체 비트 수(256)를 사용.

---

### Merkle Tree와 ZK의 연결

```
ZK에서 머클 트리의 역할:

  "나는 집합의 원소를 알고 있다" — 비밀은 유지하면서

증명 방식:
  공개 입력: root (누구나 알 수 있음)
  비공개 입력 (witness): key, value, siblings

  회로 (R1CS):
    1. leaf = poseidon_hash_circuit(key, value)
    2. for each level:
         current = poseidon_hash_circuit(current, sibling)
         (방향은 key 비트로 결정)
    3. assert current == root

  검증자: root만 알면 증명을 검증 가능
  → key, value, siblings는 전혀 노출되지 않음!

응용:
  Mixer:     "내 commitment가 이 pool에 있다" (어떤 것인지는 비밀)
  Semaphore: "나는 이 그룹의 멤버다" (누구인지는 비밀)
  Rollup:    "이 상태 전이가 올바르다" (상태 자체는 비밀 아님, 효율성)
```

> [!important] root = "집합의 지문"
> root 하나로 2^256개의 가능한 값이 "있는지 없는지"를 모두 결정한다.
> root가 같으면 트리의 내용이 동일함이 보장된다 (collision resistance).
>
> 이것이 블록체인에서 state_root가 블록 헤더에 포함되는 이유:
> ```
> Block Header:
>   parent_hash: ...
>   state_root:  Merkle root of all accounts    ← 여기!
>   tx_root:     Merkle root of all transactions
>   ...
>
> state_root 하나가 "전체 계정 상태의 요약"
> → Step 25에서 이 구조를 직접 사용
> ```

---

### 구현 코드 분석: Merkle Tree (merkle.rs)

> [!important] 코드와 수학의 대응
> 아래는 `merkle.rs`의 핵심 코드를 수학적 구조와 대응시킨 분석이다.
> 각 함수가 **왜 그렇게 구현되었는지** 수학적 근거를 설명한다.

#### 노드 인덱싱 체계

```rust
// 핵심: 노드 인덱싱
// (level, index) → 해시값
// level 0 = 리프, level depth = 루트
// index = key_repr >> level (키를 level만큼 오른쪽 시프트)
//
// 왜 이 인덱싱인가?
//   트리에서 같은 서브트리에 속하는 키들은
//   상위 비트가 동일하다.
//
//   예: depth=8에서
//     key=5 (0b00000101)과 key=7 (0b00000111)은
//     bit 2부터 같음 → level 2 이상에서 같은 경로
//     key_repr >> 2 = 1 (둘 다) → level 2에서 같은 부모
//
// 시프트 연산의 의미:
//   key_repr >> level = key의 상위 비트들
//   → level에서의 노드 위치를 유일하게 결정
```

```rust
// shr_bits: [u64; 4]를 n비트 오른쪽 시프트
//
// Fr의 내부 표현은 [u64; 4] = 256-bit 정수
// 이것을 n비트 시프트하는 것은:
//   "하위 n비트를 버리고 상위 비트만 남기기"
//   = "level n에서의 노드 인덱스"
//
// word_shift = n / 64: 건너뛸 limb 수
// bit_shift = n % 64: limb 내에서의 시프트
//
// 예: n=65이면
//   word_shift=1 (limb[0]을 건너뜀)
//   bit_shift=1  (limb[1]을 1비트 시프트)
//   result[0] = val[1] >> 1 | val[2] << 63
```

#### 형제 노드 찾기

```rust
// flip_bit0: 형제 노드의 인덱스를 구하는 핵심 연산
//
// 같은 레벨에서 형제(sibling)는 인덱스의 최하위 비트만 다르다
// → XOR 1로 형제 인덱스 획득
//
// 수학적 근거:
//   부모 인덱스 = 자식 인덱스 >> 1
//   왼쪽 자식: 부모 << 1 | 0 = 짝수 인덱스
//   오른쪽 자식: 부모 << 1 | 1 = 홀수 인덱스
//   → 형제 = 자기 인덱스 XOR 1
//
// 예: node_idx = [5, 0, 0, 0]  (= ...0101)
//     sibling  = [4, 0, 0, 0]  (= ...0100)
//     부모     = [5, 0, 0, 0] >> 1 = [2, 0, 0, 0]
//
// 이 연산이 O(1)인 이유:
//   XOR 1은 첫 번째 u64 limb에만 적용
//   → val[0] ^ 1로 충분 (나머지 limb 불변)
```

#### Insert 연산의 핵심 루프

```rust
// insert의 핵심 루프 (merkle.rs lines 138-157):
//
// for level in 0..self.depth {
//     let node_idx = shr_bits(&key_repr, level);
//     let sibling_idx = flip_bit0(&node_idx);
//     let sibling = self.nodes.get(&(level, sibling_idx))
//                       .unwrap_or(self.default_hashes[level]);
//
//     current = if !get_bit(&key_repr, level) {
//         H(current, sibling)    // current가 왼쪽
//     } else {
//         H(sibling, current)    // current가 오른쪽
//     };
//
//     let parent_idx = shr_bits(&key_repr, level + 1);
//     self.nodes.insert((level + 1, parent_idx), current);
// }
//
// 불변식 (Loop Invariant):
//   루프 시작 시 current는 level에서의 현재 노드의 해시값
//   루프 종료 시 current는 level+1에서의 부모 노드의 해시값
//
// 왜 get_bit으로 방향을 결정하는가:
//   bit=0 → key가 왼쪽 서브트리에 있음 → current가 왼쪽 자식
//   bit=1 → key가 오른쪽 서브트리에 있음 → current가 오른쪽 자식
//   Poseidon 해시는 비가환적: H(a,b) ≠ H(b,a)
//   → 왼쪽/오른쪽 순서가 중요!
//
// 왜 unwrap_or(default_hashes[level])인가:
//   대부분의 형제 노드는 빈 서브트리
//   → HashMap에 없으면 기본 해시로 대체
//   → 2^256개 노드를 저장하지 않고도 올바른 해시 계산
```

#### 검증 함수의 독립성

```rust
// verify_merkle_proof는 트리 구조체 없이 동작:
//   root, key, value, proof만으로 검증
//
// pub fn verify_merkle_proof(root, key, value, proof) -> bool
//
// 이것이 중요한 이유:
//   1. ZK 검증자는 트리를 가지고 있지 않다
//      → root(공개 입력)과 proof(증명)만으로 검증
//   2. 온체인 검증: 블록체인에 저장된 state_root와 비교
//      → 전체 상태를 다운로드할 필요 없이 특정 값을 검증
//   3. 경량 클라이언트: 블록 헤더의 root만 신뢰하면
//      → 임의의 상태 쿼리를 merkle proof로 검증 가능
//
// PoseidonParams::new()를 매번 생성하는 것에 대해:
//   new()는 MDS 행렬과 round constant를 계산
//   → 테스트/검증용으로는 OK
//   → 프로덕션에서는 전역 상수로 최적화해야 함
```

---

## Part 2: Commitment (커밋먼트)

### 커밋먼트란?

```
커밋먼트 = "봉투에 값을 넣고 봉인하는 것"

1단계 (Commit):  값 v를 봉투에 넣고 봉인 → commitment c 공개
2단계 (Open):    봉투를 열어서 v를 공개 → 검증자가 c와 대조

핵심: 봉인 후에는 값을 바꿀 수 없고(binding),
      열기 전에는 값을 알 수 없다(hiding).
```

#### 두 가지 보안 성질

```
1. Hiding (은닉성):
   "commitment c를 봐도 원래 값 v를 알 수 없다"

   c = Commit(v, r)에서 r이 랜덤이면,
   같은 v에 대해 매번 다른 c가 나옴
   → c로부터 v를 역산하는 것은 불가능

2. Binding (구속성):
   "한 번 commit한 후에는 다른 값으로 열 수 없다"

   Commit(v₁, r₁) = Commit(v₂, r₂)이면
   반드시 v₁ = v₂ (and r₁ = r₂)
   → 커밋한 값을 나중에 바꿀 수 없음
```

> [!abstract] Hiding vs Binding의 딜레마
>
> | 성질 | 수학적 보장 | 공격자 능력 |
> |------|-----------|-----------|
> | **Computationally hiding** | 계산적으로 v를 알아내기 어려움 | PPT 공격자 |
> | **Perfectly hiding** | 정보 이론적으로 v가 노출 불가 | 무한 계산 능력 |
> | **Computationally binding** | 계산적으로 다른 값으로 열기 어려움 | PPT 공격자 |
> | **Perfectly binding** | 정보 이론적으로 다른 값으로 열기 불가 | 무한 계산 능력 |
>
> **불가능 정리**: Perfectly hiding + Perfectly binding은 동시에 불가능.
> → 실용적 선택: 둘 다 Computationally 보장 (충분!)
>
> PPT = Probabilistic Polynomial Time (현실적 공격자)

---

### 해시 기반 커밋먼트 (Poseidon)

```
가장 단순한 커밋먼트: 해시 함수를 사용

  Commit(value, randomness) = H(value, randomness)

  여기서 H = poseidon_hash (Step 07)

왜 이것이 커밋먼트인가?

  Hiding:
    randomness가 균일 랜덤 Fr 원소 (254-bit)
    → 같은 value에 대해 2^254가지 다른 commitment
    → commitment에서 value를 역산하려면 Poseidon의 pre-image를 찾아야 함
    → 2^127 연산 필요 (computationally hiding)

  Binding:
    Commit(v₁, r₁) = Commit(v₂, r₂) ⟺ H(v₁, r₁) = H(v₂, r₂)
    → v₁ ≠ v₂이면 collision을 찾아야 함
    → 2^127 연산 필요 (computationally binding)
```

#### 커밋먼트의 흐름

```
┌─────────┐                           ┌─────────┐
│ Committer│                           │ Verifier│
└────┬────┘                           └────┬────┘
     │                                     │
     │  1. value v, randomness r 선택       │
     │     c = H(v, r)                     │
     │                                     │
     │  ──── c 전송 (commit) ────────────→  │
     │                                     │
     │     (시간 경과... c는 공개되어 있음)   │
     │     (하지만 v는 아직 비밀)            │
     │                                     │
     │  ──── (v, r) 전송 (open) ─────────→  │
     │                                     │
     │                    H(v, r) == c ?    │
     │                    → 성공: v가 원래 값 │
     └─────────────────────────────────────┘
```

---

### Pedersen Commitment (타원곡선 기반)

```
더 강력한 커밋먼트: 타원곡선의 이산로그 문제에 기반

  두 개의 독립적인 생성자 g, h ∈ G1 선택
  (h = sG1이되 s는 아무도 모르는 값)

  Commit(v, r) = v·g + r·h    (타원곡선 점)

  여기서:
    v ∈ Fr — 커밋할 값 (스칼라)
    r ∈ Fr — 랜덤 블라인딩 팩터
    g, h ∈ G1 — 공개 파라미터
```

#### 왜 Pedersen이 특별한가: 동형(Homomorphic) 성질

```
Pedersen commitment의 핵심 장점 = 덧셈 동형성

  Commit(v₁, r₁) + Commit(v₂, r₂)
  = (v₁·g + r₁·h) + (v₂·g + r₂·h)
  = (v₁+v₂)·g + (r₁+r₂)·h
  = Commit(v₁+v₂, r₁+r₂)

의미:
  커밋먼트를 열지 않고도 "커밋된 값들의 합"에 대한 커밋먼트를 얻을 수 있다!

응용 (Private Transfer):
  입력 커밋먼트: C_in₁ + C_in₂ = Commit(v₁+v₂, r₁+r₂)
  출력 커밋먼트: C_out₁ + C_out₂ = Commit(v₃+v₄, r₃+r₄)

  v₁+v₂ = v₃+v₄ (보존 법칙) ⟺ C_in₁+C_in₂ - C_out₁-C_out₂ = (r₁+r₂-r₃-r₄)·h
  → 값은 숨긴 채로 "보존 법칙"을 검증 가능!
```

> [!important] 해시 커밋먼트 vs Pedersen 커밋먼트
>
> | 성질 | 해시 기반 H(v, r) | Pedersen v·g + r·h |
> |------|------------------|-------------------|
> | Hiding | Computational | **Perfect** (정보 이론적) |
> | Binding | Computational | Computational (DLP) |
> | 동형성 | **없음** | **덧셈 동형** ✓ |
> | 출력 크기 | Fr 원소 (32B) | G1 점 (64B) |
> | ZK 회로 비용 | ~250 제약 | ~수천 제약 (scalar mul) |
>
> 우리 구현: **해시 기반** (Poseidon)
> - 이유 1: ZK 회로에서 훨씬 저렴 (scalar mul 불필요)
> - 이유 2: Mixer/Semaphore에서는 동형성이 필요 없음
> - 이유 3: Step 07의 Poseidon을 직접 재사용
>
> Pedersen은 Step 47 (Private Transfer)에서 value conservation 증명에 필요할 때 도입.

---

### Pedersen Commitment의 보안 증명

```
Perfectly Hiding:

  임의의 value v에 대해, r이 Fr 위 균일 랜덤이면:
    C = v·g + r·h

  r이 Fr 전체를 균일하게 순회하므로,
  r·h도 G1 전체를 균일하게 순회
  → C = v·g + (균일 랜덤 G1 점) = 균일 랜덤 G1 점
  → v에 대한 정보가 전혀 없음!

  이것이 "perfectly hiding"의 증명:
  무한한 계산 능력이 있어도 C만으로 v를 알 수 없다.
  (어떤 v'에 대해서도 C = v'·g + r'·h인 r'이 존재하므로)

Computationally Binding:

  Commit(v₁, r₁) = Commit(v₂, r₂) 이면:
    v₁·g + r₁·h = v₂·g + r₂·h
    (v₁-v₂)·g = (r₂-r₁)·h

  v₁ ≠ v₂이면:
    g = ((r₂-r₁)/(v₁-v₂))·h
    → g와 h 사이의 이산로그를 알게 됨
    → DLP를 푼 것!

  따라서 binding을 깨려면 DLP를 풀어야 함.
  BN254에서 DLP: ~2^128 연산 → computationally binding ✓
```

> [!note] "h의 이산로그를 아무도 모른다"는 가정
> h = s·g이고 s를 아는 사람이 있으면:
> ```
> 그 사람은 binding을 깰 수 있다:
>   Commit(v, r) = v·g + r·h = v·g + r·s·g = (v + rs)·g
>   → (v₁ + r₁·s)·g = (v₂ + r₂·s)·g
>   → v₁ + r₁·s = v₂ + r₂·s (mod r)
>   → 풀 수 있음!
> ```
>
> 이것이 Pedersen의 **trusted setup** 요소:
> h를 생성할 때 s를 즉시 폐기해야 한다 ("toxic waste").
> 또는 "nothing-up-my-sleeve" 방법으로 h를 결정론적으로 선택:
> ```
> h = hash_to_curve("Pedersen generator h")
> → s를 아는 사람이 없음이 보장됨
> ```

---

### 해시 커밋먼트의 형식적 보안 증명

> [!important] Pedersen과의 차이
> Pedersen은 **perfectly hiding** + **computationally binding**이다.
> 해시 기반 커밋먼트 H(v,r)은 **둘 다 computational**이지만,
> ZK 회로에서 훨씬 저렴하므로 우리 구현에서 채택했다.
> 아래는 해시 기반 커밋먼트의 보안을 **해시 함수의 성질로 리덕션**하는 형식적 증명이다.

#### Computationally Hiding (Pre-image Resistance로부터)

```
정리: Poseidon이 pre-image resistant이면, H(v,r)은 computationally hiding이다.

증명 (리덕션):
  공격자 A가 commitment c에서 value v를 구별할 수 있다고 가정.

  리덕션 B가 A를 사용하여 Poseidon pre-image를 찾는다:
    1. B는 pre-image 챌린지 c*를 받음
       (즉, c* = H(x₁, x₂)인 (x₁, x₂)를 찾아야 함)
    2. B가 A에게 c*를 commitment으로 제공
    3. A가 v를 구별 → B가 (v, r)을 찾아 c* = H(v, r)

  실제로는 r이 균일 랜덤이므로 (v, r) 쌍에서 v에 대한 정보가 c에 남지 않음.
  → A의 advantage ≤ Pr[pre-image] < 2^(-127)

  직관적 설명:
    r이 254-bit 랜덤이면, Poseidon의 출력 공간도 ~254-bit.
    각 value v에 대해, r을 바꾸면 H(v, r)은 출력 공간 전체에 걸쳐
    "의사 랜덤"하게 분포한다 (Poseidon이 좋은 해시 함수라면).

    따라서 commitment c를 관찰해도:
      "c = H(42, r₁)"인지 "c = H(99, r₂)"인지 구별하려면
      r₁ 또는 r₂를 추측해야 하는데 → 2^254가지 가능성
      → brute force: 2^254 해시 계산 필요 (비현실적)

  형식적으로:
    ∀ PPT 공격자 A, ∀ v₀, v₁ ∈ Fr:
      |Pr[A(H(v₀, r)) = 0] - Pr[A(H(v₁, r)) = 0]| < negl(λ)
      where r ← Fr 균일 랜덤, λ = 보안 파라미터 (127-bit)

  비교: Pedersen은 정보 이론적으로 hiding (통계적으로 구별 불가)
        해시 기반은 계산적으로 hiding (충분히 빠른 컴퓨터가 있으면 이론상 가능)
        실용적 차이: 2^127 연산이 필요하므로 실질적으로 동등 □
```

#### Computationally Binding (Collision Resistance로부터)

```
정리: Poseidon이 collision resistant이면, H(v,r)은 computationally binding이다.

증명 (리덕션):
  공격자 A가 binding을 깨는 것은 다음을 찾는 것이다:
    (v₁, r₁) ≠ (v₂, r₂)  이면서  H(v₁, r₁) = H(v₂, r₂)

  Case 1: v₁ ≠ v₂인 경우
    → (v₁, r₁) ≠ (v₂, r₂)이고 H(v₁, r₁) = H(v₂, r₂)
    → 이것은 정확히 Poseidon의 collision!
    → collision resistance에 의해 발생 확률 < 2^(-127)

  Case 2: v₁ = v₂이지만 r₁ ≠ r₂인 경우
    → (v₁, r₁) ≠ (v₁, r₂)이고 H(v₁, r₁) = H(v₁, r₂)
    → 이것도 Poseidon의 collision!
    → 마찬가지로 발생 확률 < 2^(-127)

  두 경우 모두:
    Binding 공격 성공 확률 ≤ Collision 공격 성공 확률 < 2^(-127)

  리덕션의 방향을 명확히:
    A가 binding을 깬다 → B가 A의 출력으로 collision을 구성
    B: (v₁, r₁, v₂, r₂) ← A()
       collision = ((v₁, r₁), (v₂, r₂))  — 서로 다른 입력, 같은 출력
    → B의 성공 확률 = A의 성공 확률

  결론:
    ∀ PPT 공격자 A:
      Pr[A가 (v₁,r₁) ≠ (v₂,r₂) s.t. H(v₁,r₁) = H(v₂,r₂)를 찾음] < 2^(-127) □
```

> [!note] Perfectly Binding이 아닌 이유
> 해시 기반 커밋먼트는 **perfectly binding이 아니다**.
> ```
> 이론적으로 collision이 반드시 존재한다:
>   H: Fr × Fr → Fr  (입력 공간 ~2^508, 출력 공간 ~2^254)
>   비둘기집 원리: 각 출력에 평균 ~2^254개의 서로 다른 입력이 매핑
>   → collision이 존재한다 (정보 이론적으로 binding이 아님)
>
> 하지만 이런 collision을 찾는 것은 2^127 연산이 필요:
>   Birthday attack: √(2^254) = 2^127 해시 계산
>   → "계산적으로 binding" (computationally binding)
>
> 대조: Pedersen commitment은 DLP가 깨져야 binding이 깨지므로,
>        마찬가지로 computationally binding이지만 기반 가정이 다르다.
>        (해시 collision vs 이산로그)
> ```

---

### 우리 구현: Poseidon 기반 커밋먼트

```
우리의 선택: 단순한 해시 커밋먼트

  commit(value, randomness) = poseidon_hash(value, randomness)
  open(commitment, value, randomness) = (commitment == poseidon_hash(value, randomness))

왜 이것으로 충분한가:
  Step 43-46 (Mixer)에서의 사용:
    commitment = H(nullifier, secret)
    → nullifier와 secret이 value와 randomness 역할
    → 동형성 불필요 (합산 증명 없음)
    → 회로 비용이 핵심 → Poseidon이 최적

구조:
  pub fn commit(value: Fr, randomness: Fr) -> Fr {
      poseidon_hash(value, randomness)
  }

  pub fn verify_commitment(commitment: Fr, value: Fr, randomness: Fr) -> bool {
      commit(value, randomness) == commitment
  }
```

---

### Commitment의 블록체인 응용

```
1. Mixer (Tornado Cash 패턴):
   ┌──────────────────────────────────────────────┐
   │ Deposit:                                      │
   │   note = (nullifier, secret)                  │
   │   commitment = H(nullifier, secret)           │
   │   → commitment을 Merkle tree에 삽입           │
   │                                               │
   │ Withdraw:                                     │
   │   공개: root, nullifier_hash, recipient       │
   │   비공개: nullifier, secret, merkle_proof     │
   │   ZK 증명:                                    │
   │     1. commitment = H(nullifier, secret)      │
   │     2. commitment이 root에 포함됨 (Merkle)    │
   │     3. nullifier_hash = H(nullifier)          │
   │   → 어떤 deposit인지 모르게 withdraw!          │
   └──────────────────────────────────────────────┘

2. 경매 (Sealed-bid Auction):
   1단계: 입찰가를 commit → C = H(bid, r)
   2단계: 모든 입찰 후 → (bid, r) 공개
   → 다른 사람의 입찰가를 보고 바꿀 수 없음 (binding)
   → 공개 전까지 입찰가를 알 수 없음 (hiding)

3. 랜덤 비콘 (Randomness Beacon):
   각 참여자가 랜덤 값을 commit → 모든 commit 수집 후 open
   → 결과: 모든 값의 XOR = 조작 불가능한 공유 랜덤
```

---

## Part 3: Schnorr 서명

### 전자서명이란?

```
"메시지 m에 대한 서명 σ를 생성하면,
 공개키 pk를 아는 누구나 서명을 검증할 수 있지만,
 비밀키 sk 없이는 유효한 서명을 생성할 수 없다"

세 가지 알고리즘:
  KeyGen()      → (sk, pk)         비밀키/공개키 쌍 생성
  Sign(sk, m)   → σ                비밀키로 서명
  Verify(pk, m, σ) → bool          공개키로 검증
```

#### 왜 EdDSA가 아닌 Schnorr인가?

```
EdDSA (Ed25519):
  별도의 커브 (Curve25519, 또는 BN254 위의 ed-on-bn254)가 필요
  → 새로운 커브를 구현해야 함
  → Step 05의 G1을 재사용할 수 없음

Schnorr:
  임의의 그룹 위에서 동작
  → Step 05의 G1을 그대로 사용!
  → 추가 커브 구현 불필요

비교:
  ┌────────────┬──────────────────┬──────────────────────┐
  │            │ Schnorr (우리)    │ EdDSA (Ed25519)      │
  ├────────────┼──────────────────┼──────────────────────┤
  │ 그룹       │ G1 (BN254)       │ Ed25519 또는          │
  │            │ (기존 Step 05)   │ ed-on-bn254 (새로 필요)│
  │ 추가 구현   │ 없음             │ 새 커브 전체          │
  │ 보안 가정   │ ECDLP on BN254   │ ECDLP on Ed25519     │
  │ 선형성      │ ✓ (다중서명 쉬움) │ ✗                    │
  │ 비고       │ Taproot (BTC)    │ Solana, Zcash        │
  └────────────┴──────────────────┴──────────────────────┘
```

> [!tip] 비트코인의 Taproot 업그레이드
> 비트코인은 2021년 Taproot 업그레이드로 ECDSA → **Schnorr**로 전환했다.
> 이유: Schnorr의 **선형성** 덕분에 다중서명(multisig)이 단일 서명처럼 보임 → 프라이버시 향상.
>
> 이 "선형성"이란:
> ```
> σ₁ = Sign(sk₁, m), σ₂ = Sign(sk₂, m)
> σ₁ + σ₂ = 유효한 서명 for pk₁ + pk₂
> → N명의 서명을 합치면 1개의 서명처럼 보임!
> → ECDSA에서는 이것이 불가능
> ```

---

### 이산로그 문제 (DLP)와 Schnorr의 보안 기반

```
이산로그 문제 (Discrete Logarithm Problem):

  주어진: G (생성자), P = k·G (공개점)
  찾기: k (비밀 스칼라)

  BN254 G1에서:
    G = (1, 2)  ← Step 05의 생성자
    P = k·G    ← double-and-add로 계산 가능
    k를 P에서 역산하는 것은 ~2^128 연산 필요

  이것이 Schnorr 서명의 보안 기반:
    sk (비밀키) = k ∈ Fr    ← 랜덤 스칼라
    pk (공개키) = k·G ∈ G1  ← 타원곡선 점

    pk에서 sk를 알아내는 것 = DLP를 푸는 것 = 사실상 불가능
```

> [!important] ECDLP의 보안 수준
> ```
> 최선의 알려진 공격: Pollard's rho algorithm
>   복잡도: O(√r) ≈ O(2^127) for BN254
>
> 양자 컴퓨터: Shor's algorithm
>   복잡도: O(log³ r) — 다항 시간!
>   → 충분히 큰 양자 컴퓨터가 있으면 DLP가 깨짐
>   → 현재(2024): 아직 암호학적 크기의 DLP를 깰 수 있는 양자 컴퓨터는 없음
>   → Post-quantum cryptography 연구가 활발한 이유
> ```

---

### Schnorr 서명 스킴: 상세 설명

#### 키 생성 (Key Generation)

```
1. sk ← Fr에서 랜덤 스칼라 선택
     sk ∈ {1, 2, ..., r-1}  (0이 아닌 값)
     (r = BN254의 Fr 모듈러스 ≈ 2^254)

2. pk = sk · G
     G = G1의 생성자 (1, 2)
     pk ∈ G1 (타원곡선 점)

보안: pk에서 sk를 알아내는 것 = ECDLP ≈ 2^128 연산

예시 (교육용 작은 수):
  sk = 42
  pk = 42·G = 42번의 point addition (실제로는 double-and-add)
```

#### 서명 (Signing)

```
입력: sk (비밀키), m (메시지 해시, Fr 원소)
출력: σ = (R, s) ∈ G1 × Fr

알고리즘:

  1. 난스(nonce) 선택:
     k ← Fr에서 랜덤 스칼라
     R = k · G                ← 난스의 "공개 부분"

  2. 챌린지(challenge) 계산:
     e = H(R || m || pk)      ← Fiat-Shamir 해싱

     구체적으로: e = poseidon_hash(
       poseidon_hash(R.x, R.y),    // R을 Fr로 압축
       poseidon_hash(m, pk.x)       // 메시지와 공개키
     )

  3. 응답(response) 계산:
     s = k + e · sk    (mod r)  ← Fr 산술

  4. 서명 출력:
     σ = (R, s)

직관:
  R은 "난스 k의 커밋먼트" (k·G를 공개, k는 비밀)
  e는 "예측 불가능한 챌린지" (R, m, pk의 해시)
  s는 "챌린지에 대한 응답" (k와 sk를 혼합)
  → sk를 모르면 s를 만들 수 없음
```

#### 검증 (Verification)

```
입력: pk (공개키), m (메시지 해시), σ = (R, s) (서명)
출력: bool (유효/무효)

알고리즘:

  1. 챌린지 재계산:
     e = H(R || m || pk)      ← 서명자와 동일한 해시

  2. 등식 검증:
     s · G == R + e · pk      ← 핵심 검증 등식

정당성 증명:
  s · G = (k + e·sk) · G
        = k·G + e·sk·G
        = R + e·pk         ✓

  좌변 = s·G
  우변 = R + e·pk = k·G + e·sk·G = (k + e·sk)·G = s·G  ✓
```

---

### Schnorr 서명의 수학적 기초

#### 정리: 완전성 (Correctness)

```
정직한 서명자가 만든 서명은 항상 검증을 통과한다.

증명:
  서명: k 선택, R = k·G, e = H(R||m||pk), s = k + e·sk
  검증: s·G ?= R + e·pk

  s·G = (k + e·sk)·G          (s의 정의)
      = k·G + e·(sk·G)        (스칼라 곱의 분배법칙)
      = R + e·pk               (R과 pk의 정의)

  → 좌변 = 우변 ✓ □
```

#### 정리: 비위조성 (Unforgeability) — 직관적 설명

```
비밀키 sk를 모르는 공격자가 유효한 서명을 만들 수 없다.

보안 모델: EU-CMA (Existential Unforgeability under Chosen Message Attack)
  "공격자가 여러 메시지의 서명을 받아본 후에도,
   새로운 메시지에 대한 서명을 위조할 수 없다"

핵심 아이디어 (forking lemma에 기반한 증명 스케치):

  가정: 공격자 A가 확률 ε > 0으로 서명을 위조한다.

  리덕션:
    1. DLP solver B가 A를 서브루틴으로 사용
    2. B는 DLP 인스턴스 (G, pk=?·G)를 받음
    3. B가 A에게 pk를 공개키로 제공
    4. A가 위조 서명 (R, s)를 생성
    5. B가 A를 "같은 랜덤 테이프, 다른 해시"로 다시 실행 (forking)
    6. 두 번의 실행에서:
       s₁ = k + e₁·sk    (첫 번째 실행)
       s₂ = k + e₂·sk    (두 번째 실행, 같은 R이지만 다른 e)
    7. s₁ - s₂ = (e₁ - e₂)·sk
       → sk = (s₁ - s₂) / (e₁ - e₂)
       → DLP 해결!

  결론: A의 존재 → DLP를 효율적으로 풀 수 있음
        DLP가 어려움 → A는 존재하지 않음 (모순)
        → Schnorr는 EU-CMA 안전 (Random Oracle Model에서)
```

> [!abstract] Forking Lemma의 직관
> ```
> 첫 번째 실행: 공격자가 (R, s₁)을 만들고 e₁ = H(R||m||pk)
> 두 번째 실행: 같은 R이지만 H를 다른 값 e₂로 프로그래밍
>              공격자가 (R, s₂)를 만듦
>
> s₁ = k + e₁·sk
> s₂ = k + e₂·sk
> ─────────────── 빼기
> s₁ - s₂ = (e₁ - e₂)·sk
> sk = (s₁ - s₂) · (e₁ - e₂)⁻¹
>
> → 비밀키 sk를 복원! → DLP 해결
> ```
>
> 이 증명이 "Random Oracle Model"에서 성립하는 이유:
> 해시 함수 H를 자유롭게 프로그래밍할 수 있어야 두 번째 실행에서
> 다른 e₂를 줄 수 있기 때문. 실제 Poseidon은 Random Oracle이 아니지만,
> 충분히 좋은 해시 함수로 실용적 보안은 보장.

---

### Schnorr 서명의 영지식 성질 (Zero-Knowledge)

> [!important] 서명이 왜 영지식과 관련되는가?
> Schnorr 서명의 근본은 **Σ-프로토콜** (대화식 영지식 증명)이다.
> "나는 pk = sk·G에서 sk를 알고 있다"를 sk 노출 없이 증명하는 것이 핵심.
> 서명은 이 Σ-프로토콜의 Fiat-Shamir 변환이므로,
> 서명 과정이 왜 비밀키 정보를 노출하지 않는지를 이해하려면
> 먼저 Σ-프로토콜의 영지식 성질을 증명해야 한다.

#### 정리: Honest-Verifier Zero-Knowledge (HVZK)

```
정리: Schnorr Σ-프로토콜은 Honest-Verifier Zero-Knowledge (HVZK)이다.

정의: HVZK란
  "정직한(honest) 검증자와의 대화 기록 (R, e, s)을
   비밀키 sk 없이도 통계적으로 동일한 분포로 시뮬레이션할 수 있다"

증명 (시뮬레이터 구성):

  시뮬레이터 S는 sk를 모르고도 (R, e, s) 대화 기록을 만든다:

  1. s ← Fr에서 균일 랜덤 선택
  2. e ← Fr에서 균일 랜덤 선택
  3. R = s·G - e·pk    (역산으로 R을 구성!)

  핵심: R을 먼저 계산하고 e를 나중에 받는 게 아니라,
        s와 e를 먼저 정하고 R을 역산한다.

  검증: s·G = R + e·pk ?
    R + e·pk = (s·G - e·pk) + e·pk
             = s·G  ✓

  → 시뮬레이터가 만든 (R, e, s)는 검증 등식을 만족한다!
```

#### 분포 동일성 증명

```
실제 대화 (Real Transcript):
  k ← Fr 균일 랜덤
  R = k·G
  e ← Fr 균일 랜덤 (honest verifier가 선택)
  s = k + e·sk

시뮬레이션 대화 (Simulated Transcript):
  s ← Fr 균일 랜덤
  e ← Fr 균일 랜덤
  R = s·G - e·pk

두 분포가 동일함을 보인다:

  실제에서:
    (1) k가 Fr에서 균일 랜덤
        → R = k·G는 G1에서 균일 랜덤 (생성자 G의 스칼라 곱)
    (2) e가 Fr에서 균일 랜덤
    (3) s = k + e·sk
        k가 균일 랜덤이고 e·sk는 상수 (주어진 e, sk에 대해)
        → s = (균일 랜덤) + 상수 = 균일 랜덤
    → (R, e, s)에서 R은 균일, e는 균일, s는 균일

  시뮬레이션에서:
    (1) s가 Fr에서 균일 랜덤
    (2) e가 Fr에서 균일 랜덤
    (3) R = s·G - e·pk
        s가 균일 랜덤 → s·G는 G1에서 균일 랜덤
        e·pk는 상수 (주어진 e, pk에 대해)
        → R = (균일 랜덤 G1 점) - 상수 = 균일 랜덤 G1 점
    → (R, e, s)에서 R은 균일, e는 균일, s는 균일

  두 분포가 모두 (균일, 균일, 균일)이고,
  각 변수가 같은 공간(G1, Fr, Fr)에서 균일 분포
  + 검증 등식 s·G = R + e·pk를 만족하는 조건부 분포도 동일

  → 두 분포가 통계적으로 동일! □
```

#### 의미와 한계

```
HVZK의 의미:
  검증자가 대화를 관찰해도 sk에 대해 아무것도 배우지 못한다.
  왜냐하면 대화 기록을 sk 없이도 완벽하게 시뮬레이션할 수 있으므로.
  → 대화 기록에 sk에 대한 정보가 전혀 담겨 있지 않다!

직관적 이해:
  서명 σ = (R, s)에서:
    R = k·G   — k는 랜덤이므로 R은 G1에서 랜덤한 점
    s = k + e·sk — k가 sk의 정보를 "마스킹"

  만약 k를 모르면 (관찰자 입장):
    s에서 sk를 추출하려면: sk = (s - k)/e
    하지만 k를 모르므로 → sk를 알 수 없음
    (one-time pad와 유사: 랜덤 k가 sk를 완벽히 은닉)

한계: "Honest-Verifier" 조건
  위 증명은 검증자가 e를 정직하게 (균일 랜덤으로) 선택한다고 가정.
  악의적 검증자가 e를 조작하면?
    → 일반 ZK (not just HVZK)를 위해서는 추가 기법 필요
    → 하지만 Fiat-Shamir 변환에서는 e = H(R||m||pk)이므로
      검증자가 e를 선택하지 않음 → HVZK로 충분!

Fiat-Shamir에서의 영지식:
  비대화식 서명 σ = (R, s)를 관찰해도:
    R은 G1에서 (의사) 랜덤, s는 Fr에서 (의사) 랜덤
    → Random Oracle Model에서 시뮬레이션 가능
    → 서명으로부터 sk에 대한 정보 유출 없음
```

---

### Schnorr EU-CMA 보안: 상세 리덕션 증명

> [!abstract] 이 절의 목표
> 앞의 직관적 설명을 **형식적 리덕션**으로 확장한다.
> Pointcheval-Stern (2000)의 Forking Lemma를 사용하여
> "Schnorr 위조 → ECDLP 해결"의 리덕션을 단계별로 구성한다.

#### 정리 (Pointcheval-Stern, 2000)

```
정리 (Pointcheval-Stern, 2000):
  Random Oracle Model에서, ECDLP가 어려우면 Schnorr은 EU-CMA 안전하다.

형식적 서술:
  ∀ PPT 위조자 A:
    Adv^{EU-CMA}_{Schnorr}(A) ≤ √(q_H · Adv^{ECDLP}(B)) + negl(λ)

  여기서:
    q_H = A의 해시 쿼리 수
    B = A를 사용하여 구성한 ECDLP solver
    λ = 보안 파라미터

직관: 위조자의 성공 확률이 높으면 → ECDLP solver의 성공 확률도 높다
      ECDLP가 어렵다 → 위조자의 성공 확률이 낮다 □
```

#### 증명: 리덕션 B의 구성

```
ECDLP 인스턴스: (G, Q = x·G)가 주어졌을 때, x를 찾아라.

리덕션 B의 구성:
  B는 위조자 A를 서브루틴으로 사용하여 x를 찾는다.

  ┌─────────────────────────────────────────────────┐
  │  ECDLP Challenger  ──(G, Q=x·G)──→  리덕션 B    │
  │                                       │          │
  │                     pk=Q를 제공       ↓          │
  │                              ┌──────────────┐   │
  │                              │  위조자 A     │   │
  │                              │ (signing oracle│  │
  │                              │  + hash oracle)│  │
  │                              └───────┬──────┘   │
  │                                      │          │
  │                     (R*, s*) 위조 서명  │          │
  │                              ←───────┘          │
  │                                                 │
  │  B가 Forking Lemma로 x 추출                      │
  └─────────────────────────────────────────────────┘
```

#### Phase 1: Setup

```
Setup:
  B는 A에게 pk = Q를 공개키로 제공
  (B는 x를 모르지만, Q는 유효한 공개키)
  → A의 관점에서는 정상적인 Schnorr 키 쌍과 구분 불가
```

#### Phase 2: Signing Oracle 시뮬레이션

```
서명 쿼리 (Signing Oracle):
  A가 메시지 mᵢ에 대한 서명을 요청할 때,
  B는 x를 모르므로 진짜 서명을 만들 수 없다!

  → B는 HVZK 시뮬레이터와 동일한 트릭을 사용:

  1. sᵢ ← Fr에서 균일 랜덤 선택
  2. eᵢ ← Fr에서 균일 랜덤 선택
  3. Rᵢ = sᵢ·G - eᵢ·Q    (시뮬레이션! x를 모르고도 가능)
  4. H(Rᵢ || mᵢ || pk) = eᵢ로 프로그래밍 (Random Oracle 제어)
     → B가 해시 함수를 "제어"하여 원하는 출력을 강제
  5. 서명 (Rᵢ, sᵢ)를 A에게 반환

  검증 (A가 서명을 의심할까?):
    eᵢ = H(Rᵢ || mᵢ || pk)  ← B가 프로그래밍해서 성립
    sᵢ·G = Rᵢ + eᵢ·pk ?
    Rᵢ + eᵢ·Q = (sᵢ·G - eᵢ·Q) + eᵢ·Q = sᵢ·G  ✓
    → A는 진짜 서명과 구분 불가!

  왜 Random Oracle Model이 필요한가:
    B가 H의 출력을 프로그래밍하려면
    H가 "블랙박스"여야 한다 (A가 H의 내부 구조를 알면 조작을 감지)
    → Random Oracle = 완전 랜덤 함수 (내부 구조 없음)
    → 실제 Poseidon은 구조가 있지만, 충분히 랜덤해서 실용적으로 안전
```

#### Phase 3: 위조 추출 (Forking)

```
A가 새 메시지 m*에 대한 위조 서명 (R*, s*)를 출력했다고 가정.

  e* = H(R* || m* || pk)

1차 실행에서 A의 출력:
  서명: (R*, s*)
  등식: s* = k* + e*·x    (k*는 A가 내부적으로 선택한 난스)

B가 Forking Lemma를 적용:
  같은 랜덤 시드로 A를 다시 실행
  단, H(R* || m* || pk)의 값만 다르게 프로그래밍:
    e*' ≠ e*    (다른 해시 값)

2차 실행에서 A의 출력:
  서명: (R*, s*')    (같은 R*이지만 다른 e에 대한 응답)
  등식: s*' = k* + e*'·x    (같은 k* — 같은 랜덤 시드!)

두 등식으로부터 x 추출:
  s*  = k* + e*  · x    ... (1)
  s*' = k* + e*' · x    ... (2)

  (1) - (2):
  s* - s*' = (e* - e*') · x

  e* ≠ e*'이므로 (e* - e*') ≠ 0  (Fr에서 역원 존재)

  x = (s* - s*') · (e* - e*')⁻¹    ← ECDLP 해결!
```

#### 확률 분석 (Forking Lemma)

```
Forking Lemma (Pointcheval-Stern):
  A의 위조 확률이 ε이면,
  두 번째 실행에서 같은 R*에서 다른 e*'에 대해 위조할 확률은?

  분석:
    A가 q_H번의 해시 쿼리를 하고, 그 중 하나에서 위조에 성공한다.
    1차 실행: 확률 ε로 위조 성공
    2차 실행: 같은 R*에서 다른 e*'에 대해 위조 성공 확률

    핵심 관찰:
      A의 위조가 특정 해시 쿼리 j에서 발생했다면 (q_H 중 하나),
      2차 실행에서 그 쿼리까지 같은 경로를 따라감 (같은 랜덤 시드)
      → 같은 R*에 도달
      → 다른 e*'에 대해서도 위조 확률 ≈ ε

    하지만 어떤 쿼리 j에서 위조가 발생하는지 모름
    → q_H 중 하나 → 확률이 q_H로 나누어짐

  결과:
    B의 성공 확률 ≥ ε²/q_H - negl(λ)

    ε가 non-negligible이면 (예: ε ≥ 1/poly(λ)):
      ε²/q_H ≥ 1/poly(λ)²/poly(λ) = 1/poly(λ)³
      → non-negligible!
      → B가 ECDLP를 non-negligible 확률로 해결
      → ECDLP 가정과 모순!

  따라서:
    ECDLP가 어렵다 → ε < negl(λ)
    → Schnorr는 EU-CMA 안전 (Random Oracle Model에서) □
```

> [!note] "Random Oracle Model"의 의미와 한계
> ```
> Random Oracle Model (ROM):
>   해시 함수 H를 "완전 랜덤 함수"로 모델링
>   → B가 H의 출력을 자유롭게 프로그래밍 가능
>   → 실제 해시 함수 (Poseidon, SHA-256 등)와 완전히 같지는 않음
>
> 실용적 의미:
>   ROM에서의 증명 = "해시가 충분히 좋으면 안전"
>   → 모든 실용적 Schnorr 구현은 이 가정 아래 안전
>   → 30년 이상 실제 공격이 발견되지 않음
>
> Standard Model에서의 증명:
>   Schnorr의 Standard Model 보안 증명은 오래 열린 문제였으며,
>   최근(2023) Fuchsbauer et al.이 일부 조건 하에서 증명
>   → 대부분의 실용에서는 ROM 증명으로 충분
>
> 우리 구현에서:
>   H = Poseidon hash (Step 07)
>   Poseidon은 algebraic hash로 Random Oracle은 아니지만,
>   알려진 대수적 공격에 대한 저항성이 검증됨
>   → 실용적 보안: ≈ 2^127 수준
> ```

---

### Fiat-Shamir 변환: 대화식 → 비대화식

```
Schnorr 서명의 기원: Σ-프로토콜 (대화식 증명)

대화식 Schnorr 증명:
  ┌─────────┐                      ┌─────────┐
  │ Prover P │                      │Verifier V│
  │(sk 알고 있음)                    │(pk 알고 있음)│
  └────┬────┘                      └────┬────┘
       │                                │
       │  1. k ← random Fr              │
       │     R = k·G                    │
       │  ──── R 전송 (commitment) ───→  │
       │                                │
       │                   2. e ← random│
       │  ←── e 전송 (challenge) ──────  │
       │                                │
       │  3. s = k + e·sk               │
       │  ──── s 전송 (response) ─────→  │
       │                                │
       │                   4. s·G == R + e·pk ?
       └────────────────────────────────┘

이것은 "sk를 알고 있음"의 영지식 증명 (ZK proof of knowledge)!
  Zero-Knowledge: 검증자는 sk에 대해 아무것도 배우지 못함
  Soundness: sk를 모르면 e에 대한 올바른 s를 만들 수 없음
```

#### Fiat-Shamir 해싱

```
문제: 대화식 프로토콜은 검증자가 온라인이어야 함
해법: 검증자의 챌린지를 해시로 대체! (Fiat-Shamir)

  대화식:  V가 랜덤 e를 보냄
  비대화식: e = H(R || m || pk)  ← "랜덤 오라클"로 대체

왜 안전한가:
  H가 Random Oracle이면:
    e는 R이 결정된 후에야 알 수 있음
    → P가 e를 미리 알 수 없으므로, 대화식과 동일한 보안

  서명자의 전체 과정:
    k → R = k·G → e = H(R||m||pk) → s = k + e·sk
    → σ = (R, s) 출력
    → 검증자에게 보냄 (한 번의 통신으로 끝!)
```

> [!important] Σ-프로토콜 → 서명 스킴 변환
> Schnorr 서명은 **Σ-프로토콜의 Fiat-Shamir 변환**이다.
>
> ```
> Σ-프로토콜:
>   P → V: R (commitment)
>   V → P: e (challenge)
>   P → V: s (response)
>   검증: s·G == R + e·pk
>
> Fiat-Shamir 서명:
>   서명: R = k·G, e = H(R||m||pk), s = k + e·sk, σ=(R,s)
>   검증: e = H(R||m||pk), s·G == R + e·pk
>
> 차이점: 오직 "e를 V가 보내는가, H로 계산하는가" 뿐
> ```
>
> 이 패턴은 ZK 전체에서 반복:
> - Groth16의 증명도 Fiat-Shamir로 비대화식화
> - PLONK의 증명도 마찬가지
> - **모든 ZK-SNARK는 대화식 프로토콜의 비대화식 변환**

---

### 난스(nonce) k의 치명적 중요성

```
⚠️ 경고: k를 재사용하면 비밀키가 노출된다!

시나리오: 같은 k로 두 메시지 m₁, m₂에 서명

  서명 1: R = k·G, e₁ = H(R||m₁||pk), s₁ = k + e₁·sk
  서명 2: R = k·G, e₂ = H(R||m₂||pk), s₂ = k + e₂·sk

  공격자가 (R, s₁, e₁)과 (R, s₂, e₂)를 관찰:
    s₁ - s₂ = (e₁ - e₂)·sk
    sk = (s₁ - s₂) · (e₁ - e₂)⁻¹   ← 비밀키 복원!

  → 두 서명의 R이 같으면 (=k가 같으면) 즉시 비밀키 유출
```

> [!warning] 실제 사건: PlayStation 3 해킹 (2010)
> Sony는 PS3의 ECDSA 서명에서 **랜덤 k 대신 고정 값을 사용**.
> 해커 fail0verflow 팀이 두 서명에서 k가 동일함을 발견 → 비밀키 복원.
> Sony의 코드 서명 키가 유출되어 임의의 코드 실행이 가능해짐.
>
> 교훈:
> ```
> ✗ k를 고정값으로 사용
> ✗ k를 약한 난수 생성기로 생성
> ✓ k를 RFC 6979 (결정론적 k 생성)로 생성
>   → k = HMAC(sk, m) — sk와 m이 다르면 k도 다름
>   → 같은 메시지에 같은 sk면 같은 k → 같은 서명 (멱등)
>   → 난수 생성기의 품질에 의존하지 않음
> ```
>
> 우리 구현에서는 교육 목적으로 랜덤 k를 사용하지만,
> 프로덕션에서는 반드시 결정론적 k 생성을 사용해야 한다.

---

### 구현 코드 분석: Schnorr 서명 (signature.rs)

> [!important] 코드와 수학의 대응
> `signature.rs`의 각 함수가 수학적 구조를 어떻게 구현하는지 분석한다.
> 특히 **challenge_hash**의 Fp → Fr 변환이 왜 안전한지를 상세히 다룬다.

#### challenge_hash: Fiat-Shamir 챌린지 구성

```rust
// challenge_hash(r_point, message, pk) → Fr
//
// e = H(R || m || pk)의 구현
//
// 입력: R ∈ G1Affine (Fp × Fp), m ∈ Fr, pk ∈ G1Affine (Fp × Fp)
// 출력: e ∈ Fr
//
// 문제: Poseidon은 Fr 위에서 동작하는데, R과 pk의 좌표는 Fp
// 해법: Fp → Fr 변환 후 Poseidon에 입력

// 코드 구조:
//   let rx = Fr::from_raw(r_point.x.0);    // Fp → Fr
//   let ry = Fr::from_raw(r_point.y.0);    // Fp → Fr
//   let r_hash = poseidon_hash(rx, ry);    // R의 해시
//
//   let pkx = Fr::from_raw(pk.x.0);
//   let pky = Fr::from_raw(pk.y.0);
//   let pk_hash = poseidon_hash(pkx, pky); // pk의 해시
//
//   let msg_hash = poseidon_hash(message, pk_hash);
//   poseidon_hash(r_hash, msg_hash)        // 최종 챌린지
```

#### Fp → Fr 변환의 안전성 분석

```
Fp → Fr 변환: Fr::from_raw(fp.0)

이것이 안전한 이유를 형식적으로 분석:

1. 결정론적 (Deterministic):
   같은 Fp 값 → 항상 같은 Fr 값
   → Fiat-Shamir의 핵심 요구: 서명자와 검증자가 같은 e를 계산

   증명: from_raw는 단순히 [u64; 4] 비트 패턴을 재해석
         Fp 값이 같으면 내부 [u64; 4]이 같으므로 Fr 값도 같다 □

2. 충돌 방지 (Collision-Free):
   다른 Fp 값 → 다른 Fr 값 (단사 함수)

   분석:
     Fp의 Montgomery 표현: val ∈ [0, p-1]을 [u64; 4]로 인코딩
     Fr::from_raw는 이 [u64; 4]를 Fr 원소로 해석

     Fp 모듈러스 p > Fr 모듈러스 r이므로:
       일부 Fp 값의 Montgomery 표현이 r 이상일 수 있음
       → Fr에서 mod r로 reduction됨
       → 이론적으로 두 Fp 값이 같은 Fr 값에 매핑될 수 있음

     하지만 실제로는:
       p와 r의 차이가 ~2^127 (254-bit 중 상위 비트 차이)
       → 충돌이 발생하는 Fp 값 쌍은 극히 드묾
       → 해싱 입력으로 사용하므로 보안에 영향 없음

3. Fiat-Shamir의 실제 요구사항:
   "챌린지가 commitment R에 대해 의사 랜덤"이면 충분
   → from_raw가 전사(surjective)일 필요 없음
   → 결정론성 + 다른 입력에 대해 다른 출력 = 안전

4. 대안과의 비교:
   더 엄밀한 방법: Fp 값을 바이트로 직렬화 → SHA-256 → Fr
   우리 방법: from_raw로 직접 재해석
   → 더 효율적이고, ZK 회로에서도 같은 방식 사용 가능
   → 실용적 보안 수준은 동등
```

#### sign 함수 구조

```rust
// pub fn sign(sk, message, nonce) -> Signature
//
// 수학: σ = (R, s) where R = k·G, s = k + e·sk
//
// 코드 흐름:
//   1. assert!(!nonce.is_zero())
//      → nonce = 0이면 R = O (항등원)
//      → e = H(O || m || pk)
//      → s = 0 + e·sk = e·sk
//      → 공격자가 s와 e를 알면: sk = s/e
//      → 비밀키 즉시 노출!
//
//   2. R = G1::generator().scalar_mul(&nonce.to_repr())
//      → projective 좌표에서 scalar multiplication
//      → to_affine()으로 (x, y) 형태로 변환
//      → R의 x, y가 challenge_hash의 입력
//
//   3. pk = sk.public_key()
//      → sign 함수 내에서 pk를 재계산
//      → 이유: sk만으로 서명이 가능해야 함 (API 단순화)
//      → 최적화: pk를 캐싱하면 scalar_mul 1회 절약
//
//   4. s = nonce + e * sk.0
//      → Fr 산술: + 와 * 는 mod r에서 수행
//      → 오버플로 없음 (Fr::add, Fr::mul이 자동 reduction)
//
// 출력: Signature { r: r_affine, s }
//   r은 G1Affine (Fp × Fp 좌표)
//   s는 Fr 스칼라
//   → 서명 크기: 2 × 32 + 32 = 96 바이트
```

#### verify 함수 구조

```rust
// pub fn verify(pk, message, sig) -> bool
//
// 수학: s·G == R + e·pk ?
//
// 코드 흐름:
//   1. e = challenge_hash(&sig.r, message, &pk_affine)
//      → 서명자와 동일한 해시 계산 (결정론적)
//
//   2. 좌변: s_g = G1::generator().scalar_mul(&sig.s.to_repr())
//      → s·G: 스칼라 곱 1회
//
//   3. 우변:
//      e_pk = pk.0.scalar_mul(&e.to_repr())   → e·pk: 스칼라 곱 1회
//      r_plus_e_pk = sig.r.to_projective().add(&e_pk)  → R + e·pk: 점 덧셈
//
//   4. s_g == r_plus_e_pk
//      → G1의 PartialEq: projective 좌표 비교
//      → (X₁·Z₂² == X₂·Z₁², Y₁·Z₂³ == Y₂·Z₁³)
//      → affine 변환 없이 비교 가능 (효율적)
//
// 비용: scalar_mul 2회 + point_add 1회 + poseidon 3회
//   scalar_mul이 지배적: ~254 × (double + add) ≈ 수천 Fp 연산
//   → verify는 sign보다 비쌈 (sign: scalar_mul 1회)
```

---

### Schnorr의 선형성

```
Schnorr 서명의 가장 강력한 성질: 서명이 선형적으로 합성 가능

다중서명 (Multi-signature) — MuSig 프로토콜:

  N명의 서명자: (sk₁, pk₁), (sk₂, pk₂), ..., (skₙ, pkₙ)
  집합 공개키: pk_agg = pk₁ + pk₂ + ... + pkₙ

  서명 과정:
    1. 각자 k_i 선택, R_i = k_i·G 공유
    2. R_agg = R₁ + R₂ + ... + Rₙ
    3. e = H(R_agg || m || pk_agg)
    4. 각자 s_i = k_i + e·sk_i 계산
    5. s_agg = s₁ + s₂ + ... + sₙ

  검증:
    s_agg·G ?= R_agg + e·pk_agg

    증명:
    s_agg·G = Σ(k_i + e·sk_i)·G
            = Σk_i·G + e·Σsk_i·G
            = R_agg + e·pk_agg  ✓

  결과: (R_agg, s_agg)는 pk_agg에 대한 유효한 단일 서명!
  → 외부에서 보면 1명이 서명한 것과 구분 불가
  → 비트코인 Taproot: N-of-N 다중서명이 단일 서명과 동일한 크기/비용
```

> [!abstract] ECDSA에서는 왜 불가능한가?
> ECDSA 서명: `s = k⁻¹(hash + sk·R.x)  (mod r)`
>
> 비선형 구조 (k⁻¹과의 곱):
> ```
> s₁ + s₂ = k₁⁻¹(hash + sk₁·R₁.x) + k₂⁻¹(hash + sk₂·R₂.x)
>          ≠ (k₁+k₂)⁻¹(hash + (sk₁+sk₂)·(R₁+R₂).x)
> ```
> → 서명의 합이 집합키에 대한 유효한 서명이 아님!
> → ECDSA 다중서명: 서명을 개별로 검증해야 함 → N배 비용

---

### 서명 과정 수동 추적

```
예시 (실제 Fr 값이 아닌 교육용 작은 수):

키 생성:
  sk = 7
  G = (1, 2)  ← G1 생성자
  pk = 7·G = P₇  (7번 point addition)

서명 (메시지 m = Fr::from_u64(42)):
  1. k = 13 (랜덤 난스)
     R = 13·G = P₁₃

  2. e = H(R.x || R.y || m || pk.x)
     = poseidon_hash(poseidon_hash(R.x, R.y), poseidon_hash(42, pk.x))
     = 어떤 Fr 값... = E (254-bit)

  3. s = k + e·sk = 13 + E·7  (mod r)
     = S (Fr 원소)

  서명: σ = (P₁₃, S)

검증:
  1. e = H(R.x || R.y || m || pk.x) = E (동일한 값)
  2. S·G ?= P₁₃ + E·P₇
     좌변: (13 + E·7)·G = 13·G + E·7·G = P₁₃ + E·P₇
     우변: P₁₃ + E·P₇
     → 동일! ✓
```

---

### 실제 테스트 값으로 추적

> [!important] 구현 코드의 테스트와 1:1 대응
> 아래는 `signature.rs`의 `test_keypair()`와 `sign_and_verify()` 테스트에서
> 사용하는 실제 값을 따라가는 추적이다. 교육용 작은 수가 아니라
> **실제 구현이 어떤 계산을 수행하는지** 보여준다.

#### 키 생성 (실제 테스트)

```
실제 구현의 테스트에서:

  sk = Fr::from_u64(42)
  → sk의 내부 표현: Montgomery 형태의 42
  → sk.to_repr() = [42, 0, 0, 0]  (u64 4-limb)

  pk = sk.public_key()
     = G1::generator().scalar_mul(&[42, 0, 0, 0])
     = 42·G  (G = BN254의 G1 생성자)

  G = (1, 2) ∈ Fp × Fp (affine 좌표)
  pk = 42·G = double-and-add:
    42 = 32 + 8 + 2 = 2⁵ + 2³ + 2¹
    → G, 2G, 4G, 8G, 16G, 32G를 계산
    → 2G + 8G + 32G = 42G
    → 결과: (pk.x, pk.y) ∈ Fp × Fp  (254-bit 좌표)
```

#### 서명 과정 (signature.rs의 sign 함수)

```
입력:
  sk = Fr::from_u64(42)
  message = Fr::from_u64(123)
  nonce = Fr::from_u64(7777)

Step 1: R = nonce · G = 7777·G
  R_projective = G1::generator().scalar_mul(&[7777, 0, 0, 0])
  R_affine = R_projective.to_affine()
  → R_affine = (R.x, R.y) ∈ Fp × Fp

Step 2: 챌린지 해시 (challenge_hash 함수 추적)

  Fp → Fr 변환:
    Fr::from_raw(fp.0)
    Fp와 Fr은 같은 254-bit 크기이지만 다른 모듈러스.
    from_raw는 Fp의 Montgomery 표현 [u64; 4]를
    그대로 Fr의 값으로 재해석한다.
    (결정론적이고 단사이므로 Fiat-Shamir에 안전)

  2a. R의 좌표를 Fr로 변환:
    rx = Fr::from_raw(R_affine.x.0)   // Fp → Fr
    ry = Fr::from_raw(R_affine.y.0)   // Fp → Fr
    r_hash = poseidon_hash(rx, ry)    // R의 해시

  2b. pk의 좌표를 Fr로 변환:
    pkx = Fr::from_raw(pk_affine.x.0)
    pky = Fr::from_raw(pk_affine.y.0)
    pk_hash = poseidon_hash(pkx, pky)  // pk의 해시

  2c. 메시지와 결합:
    msg_hash = poseidon_hash(Fr::from_u64(123), pk_hash)

  2d. 최종 챌린지:
    e = poseidon_hash(r_hash, msg_hash)

  해시 구조도:
    e = H(H(rx, ry), H(123, H(pkx, pky)))
              ↑                    ↑
           R의 정보         메시지 + pk의 정보

Step 3: 응답 계산
  s = nonce + e · sk
    = 7777 + e · 42    (mod r, Fr 산술)
    = Fr에서의 덧셈과 곱셈

Step 4: 서명 출력
  σ = Signature { r: R_affine, s: s }
```

#### 검증 과정 (signature.rs의 verify 함수)

```
입력:
  pk = PublicKey(42·G)
  message = Fr::from_u64(123)
  sig = (R_affine, s)

Step 1: 챌린지 재계산
  e' = challenge_hash(&sig.r, message, &pk_affine)
  → sign에서와 정확히 같은 계산
  → e' = e  (결정론적이므로)

Step 2: 좌변 계산
  s_g = G1::generator().scalar_mul(&sig.s.to_repr())
      = s · G
      = (7777 + e·42) · G

Step 3: 우변 계산
  e_pk = pk.0.scalar_mul(&e.to_repr())
       = e · pk
       = e · (42·G)
       = (e·42) · G
  r_plus_e_pk = sig.r.to_projective().add(&e_pk)
              = R + e·pk
              = 7777·G + (e·42)·G

Step 4: 비교
  s_g == r_plus_e_pk ?
  (7777 + e·42)·G == 7777·G + (e·42)·G

  스칼라 곱의 분배법칙:
    (a + b)·G = a·G + b·G  (G1은 아벨 군)

  (7777 + e·42)·G = 7777·G + e·42·G = 7777·G + (e·42)·G  ✓

  → 검증 통과!
```

> [!note] Fp → Fr 변환의 미묘함
> ```
> 코드에서 challenge_hash가 하는 Fp → Fr 변환:
>   Fr::from_raw(fp_value.0)
>
> 이것이 안전한 이유:
>   1. 결정론적: 같은 Fp 값 → 항상 같은 Fr 값
>      → Fiat-Shamir에서 중요: 서명자와 검증자가 같은 e를 얻어야 함
>
>   2. 단사(injective): 다른 Fp 값 → 다른 Fr 값
>      Fp 모듈러스 (p) > Fr 모듈러스 (r)이므로:
>        일부 Fp 값은 Fr에서 reduction될 수 있지만,
>        Montgomery 표현을 재해석하는 것이므로
>        실질적으로 다른 비트 패턴 → 다른 Fr 값
>
>   3. Fiat-Shamir의 요구사항:
>      "챌린지가 결정론적이고 조작 불가"이면 충분
>      → from_raw가 전사(surjective)일 필요는 없음
>      → 결정론성만 보장하면 안전
>
>   4. 실제로는 두 모듈러스 차이가 미미:
>      p ≈ 2^254, r ≈ 2^254
>      차이: p - r ≈ 2^127 수준
>      → "같은 254-bit 수를 다른 모듈러스로 해석"
> ```

---

### Schnorr vs ECDSA vs EdDSA 비교

```
┌────────────┬──────────────┬──────────────────┬───────────────┐
│            │ Schnorr      │ ECDSA            │ EdDSA         │
├────────────┼──────────────┼──────────────────┼───────────────┤
│ 수식       │ s = k+e·sk   │ s = k⁻¹(h+sk·r) │ s = k+e·sk    │
│ 보안 증명   │ ROM에서 증명  │ 간접 증명         │ ROM에서 증명   │
│ 선형성      │ ✓            │ ✗                │ ✓              │
│ 다중서명    │ MuSig        │ 비효율적          │ 가능           │
│ 서명 크기   │ (G1점, Fr)   │ (Fr, Fr)         │ (G1점, Fr)    │
│ 검증 비용   │ 2 scalar mul │ 2 scalar mul     │ 2 scalar mul  │
│ 역사       │ 1989 (특허)  │ 1992 (특허 회피)  │ 2011          │
│ 사용처     │ BTC Taproot  │ 기존 BTC, ETH    │ Solana, Zcash │
│ 커브       │ 아무거나     │ 아무거나          │ 트위스트 필요   │
└────────────┴──────────────┴──────────────────┴───────────────┘
```

> [!note] 역사적 맥락
> Schnorr (1989)가 먼저 발명되었지만 **특허**로 보호됨.
> ECDSA는 Schnorr 특허를 피하기 위해 설계된 변형 — 수학적으로 더 복잡하지만 특허 자유.
> 비트코인(2009)은 Schnorr 특허가 아직 유효했으므로 ECDSA를 채택.
> 2008년 Schnorr 특허 만료 → 2021년 비트코인 Taproot에서 Schnorr 도입.
>
> EdDSA는 Edwards 곡선 + Schnorr 서명의 조합.
> 수학적으로 Schnorr과 거의 동일하지만, 특정 커브 형태에 최적화.

---

## Part 4: 세 가지의 연결

### 의존성 다이어그램

```
Step 03: Fr ─────────────────────────────────────┐
         │                                       │
Step 05: G1 ──────────────────┐                  │
         │                    │                  │
Step 07: Poseidon ──┐         │                  │
                    │         │                  │
Step 08:            ↓         ↓                  ↓
         ┌──────────────┐ ┌───────────┐ ┌─────────────┐
         │ Merkle Tree  │ │  Schnorr  │ │ Commitment  │
         │ (Poseidon    │ │  (G1 + Fr)│ │ (Poseidon   │
         │  기반 해시)   │ │           │ │  기반 해시)  │
         └──────┬───────┘ └─────┬─────┘ └──────┬──────┘
                │               │              │
    ┌───────────┼───────────────┼──────────────┤
    │           │               │              │
    ↓           ↓               ↓              ↓
Step 10    Step 25         Step 26         Step 43-45
R1CS 가젯   State Tree      Transaction     Privacy
(회로화)    (블록체인 상태)   (서명된 tx)     (Mixer)
```

### 전체 프리미티브 의존성 맵

```
Step 01-02: Fp (기저체)
     │
     ├─→ Step 03: Fr (스칼라체)
     │        │
     │        ├─→ Step 07: Poseidon (해시)
     │        │        │
     │        │        ├─→ Step 08a: Merkle Tree
     │        │        │
     │        │        └─→ Step 08b: Commitment
     │        │
     │        └─→ Step 08c: Schnorr (서명)
     │                  ↑
     ├─→ Step 03-04: Fp2 → Fp6 → Fp12
     │        │
     ├─→ Step 05: G1 ──┘ (Schnorr에 사용)
     │        │
     ├─→ Step 05: G2
     │        │
     └─→ Step 06: Pairing (G1 × G2 → GT)
                │
                └─→ Step 12: Groth16
                         │
                         └─→ Step 45: Mixer 회로
                                  ↑
                    Merkle + Commitment이 여기서 결합!
```

---

### 다음 스텝과의 연결

```
Step 08 (여기) → 세 가지 원시 도구
    │
    ├─→ Step 09: R1CS 제약 시스템
    │     "모든 계산을 a·b=c 형태로"
    │
    ├─→ Step 10: R1CS 가젯
    │     Poseidon 가젯 + Merkle 가젯
    │     → "native Merkle proof == circuit Merkle proof"
    │     → ZK-friendly의 의미가 체감되는 순간
    │
    ├─→ Step 25: Chain State Tree
    │     SparseMerkleTree로 계정 상태 관리
    │     state_root = "전체 상태의 지문"
    │
    ├─→ Step 26: Transactions
    │     Schnorr 서명으로 트랜잭션 인증
    │     "이 송금 요청은 내가 보냈다"
    │
    └─→ Step 43-45: Privacy (Mixer)
          Commitment = note 은닉
          Merkle Tree = membership 증명
          → "내가 입금했음을 증명하되, 어떤 입금인지는 비밀"
```

---

### 테스트 전략

```
Merkle Tree 테스트:
  1. insert 후 root 변경 확인
  2. get으로 삽입한 값 조회
  3. proof 생성 → 검증 성공
  4. 값 변조 → 검증 실패
  5. non-membership proof (빈 키)
  6. 여러 값 삽입 후 각각 proof 유효

Commitment 테스트:
  1. commit → verify 성공
  2. 잘못된 value → verify 실패
  3. 잘못된 randomness → verify 실패
  4. 결정론적: 같은 입력 → 같은 commitment
  5. 은닉성: 같은 value, 다른 randomness → 다른 commitment

Schnorr 테스트:
  1. sign → verify 성공
  2. 메시지 변조 → verify 실패
  3. 다른 공개키 → verify 실패
  4. 서명 변조 (R 변조) → verify 실패
  5. 서명 변조 (s 변조) → verify 실패
  6. 여러 메시지에 대한 서명/검증
```

---

### 요약: Step 08에서 배우는 것

```
┌─────────────────────────────────────────────────────────────┐
│  Merkle Tree                                                 │
│    "대량의 데이터를 하나의 해시(root)로 요약"                  │
│    "특정 데이터의 포함을 O(log N) 크기의 증명으로"             │
│    핵심: Poseidon 2-to-1 해시의 반복 적용                     │
│    ZK에서: "비밀을 숨기면서 멤버십을 증명"                     │
├─────────────────────────────────────────────────────────────┤
│  Commitment                                                  │
│    "값을 숨기되(hiding), 나중에 바꿀 수 없게(binding)"         │
│    핵심: H(value, randomness)                                │
│    ZK에서: "note를 은닉하고, 나중에 ZK로 열기"                 │
├─────────────────────────────────────────────────────────────┤
│  Schnorr Signature                                           │
│    "비밀키 없이는 위조 불가능한 인증"                          │
│    핵심: s = k + e·sk, 검증 s·G = R + e·pk                  │
│    보안 기반: ECDLP on BN254 G1                              │
│    블록체인에서: 트랜잭션 인증, 블록 서명                      │
└─────────────────────────────────────────────────────────────┘

이 세 가지가 "원시 도구 세트":
  이후 50개 Step의 모든 응용이 이것들의 조합이다.
```
