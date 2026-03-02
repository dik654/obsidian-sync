# Step 50: DataLoader와 Accuracy

## step49 → step50: 무엇이 바뀌었나

||step49|step50|
|---|---|---|
|배치 순회|수동 셔플 + 인덱스 슬라이싱 + `get()` 반복|`for (x, t) in &mut loader`|
|평가 지표|loss만|loss + **accuracy**|
|평가 대상|train만|train + **test** (`no_grad`)|
|헬퍼 코드|`SimpleRng`, `shuffle()` 직접 구현|DataLoader가 내장|

학습 결과는 동일하지만, **학습 루프의 보일러플레이트가 사라진다**.

---

## 1. Accuracy (정확도)

### 왜 필요한가

loss만으로는 모델이 얼마나 잘하는지 직관적으로 파악하기 어렵다:

- loss = 0.15 → 이게 좋은 건가? 나쁜 건가?
- accuracy = 95.7% → 300개 중 287개 맞춤 → 바로 이해됨

### 공식

$$\mathrm{accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\arg\max_j y_{ij} = t_i]$$

각 기호의 의미:

- $y_{ij}$: 모델이 샘플 $i$에 대해 클래스 $j$에 부여한 점수 (logit)
- $\arg\max_j y_{ij}$: 가장 높은 점수를 받은 클래스 = **예측 클래스**
- $t_i$: 샘플 $i$의 **정답 클래스**
- $\mathbf{1}[\cdot]$: 조건이 참이면 1, 거짓이면 0 (지시 함수)

> **유도/의도**: 분류에서 가장 자연스러운 질문은 "몇 개 맞혔나?"이다. 맞은 횟수를 전체 수로 나누면 비율이 되고, 이것이 accuracy.
> 
> 예시: 3클래스, 5개 샘플
> 
> ```
> 예측: [1, 0, 2, 1, 0]
> 정답: [1, 2, 2, 1, 0]
>        ✓  ✗  ✓  ✓  ✓  → 4/5 = 0.8
> ```

### 왜 backward가 필요 없는가

accuracy는 **미분 불가능**하다.

$$\mathbf{1}[\arg\max(y) = t]$$

이 함수는 $y$가 아주 조금 변해도 argmax가 바뀌지 않으면 값이 같고, 바뀌는 순간 0 ↔ 1로 불연속 점프한다. 따라서:

- 미분값이 거의 모든 곳에서 0 (변화 없음)
- 불연속점에서는 미분 불가

그래서 **학습에는 cross-entropy를 쓰고** (미분 가능한 대리 손실), **평가에는 accuracy를 쓴다** (인간이 이해하기 쉬운 지표).

|역할|함수|미분 가능|용도|
|---|---|---|---|
|학습|cross-entropy|✓|gradient로 파라미터 업데이트|
|평가|accuracy|✗|모델 성능을 인간이 파악|

### Rust 구현

```rust
pub fn accuracy(y: &Variable, t: &[usize]) -> f64 {
    let y_data = y.data();
    let n = y_data.shape()[0];
    let c = y_data.shape()[1];

    let mut correct = 0;
    for i in 0..n {
        // argmax: 각 행에서 가장 큰 값의 인덱스
        let mut max_j = 0;
        let mut max_val = f64::NEG_INFINITY;
        for j in 0..c {
            if y_data[[i, j]] > max_val {
                max_val = y_data[[i, j]];
                max_j = j;
            }
        }
        if max_j == t[i] {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}
```

Variable이 아닌 **f64를 반환**한다. gradient가 흐를 필요가 없으므로 계산 그래프에 포함시킬 이유가 없다.

---

## 2. DataLoader

### 왜 필요한가

step49의 학습 루프에는 배치 처리를 위한 보일러플레이트가 많았다:

```rust
// step49: 매 테스트 파일마다 이 코드를 반복 작성
struct SimpleRng { state: u64 }
impl SimpleRng { fn next_f64(&mut self) -> f64 { ... } }
fn shuffle(indices: &mut [usize], rng: &mut SimpleRng) { ... }

let mut index: Vec<usize> = (0..data_size).collect();
shuffle(&mut index, &mut rng);
for i in 0..max_iter {
    let start = i * batch_size;
    let end = ((i + 1) * batch_size).min(data_size);
    let batch_index = &index[start..end];
    let mut batch_x_data = Vec::new();
    let mut batch_t = Vec::new();
    for &idx in batch_index {
        let (x, t) = train_set.get(idx);
        batch_x_data.extend_from_slice(&x);
        batch_t.push(t);
    }
    let batch_x = Variable::new(ArrayD::from_shape_vec(...));
    // ... 드디어 학습 코드 시작
}
```

DataLoader가 이 모든 것을 캡슐화:

```rust
// step50: 한 줄로 배치 순회
for (x, t) in &mut train_loader {
    // 바로 학습 코드
}
```

### DataLoader의 책임

|기능|설명|
|---|---|
|인덱스 관리|`[0, 1, ..., N-1]` 생성 및 관리|
|셔플|`shuffle=true`면 매 `reset()`마다 Fisher-Yates 셔플|
|배치 분할|현재 위치에서 `batch_size`개씩 잘라서 반환|
|샘플 조립|`Dataset.get(i)`로 개별 샘플을 꺼내 `(Variable, Vec<usize>)` 배치로 조립|
|소진 감지|모든 배치를 다 반환하면 `None` → for 루프 종료|

### Iterator 트레잇

Rust의 `Iterator` 트레잇을 구현하면 `for ... in` 구문을 자연스럽게 사용할 수 있다.

```rust
impl<'a> Iterator for DataLoader<'a> {
    type Item = (Variable, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.len() {
            return None;  // 소진 → for 루프 종료
        }
        // batch_size만큼 잘라서 반환
        let start = self.current;
        let end = (start + self.batch_size).min(data_size);
        self.current = end;
        // Dataset.get()으로 개별 샘플을 모아 배치 조립
        Some((batch_x, batch_t))
    }
}
```

> **`for (x, t) in &mut loader`에서 `&mut`인 이유**: `Iterator::next(&mut self)`는 `&mut self`를 받는다 — 내부 상태(`current` 포인터)를 변경해야 하므로. `for ... in loader`로 쓰면 소유권이 이동되어 `reset()` 후 재사용이 불가능. `&mut loader`로 빌려서 사용하면 루프 후에도 `loader`가 살아있어 `reset()` 가능.

### reset() — 에폭 재시작

Python의 DataLoader는 매 `for x, t in loader`마다 자동으로 처음부터 이터레이션한다. Rust의 Iterator는 한 번 소진되면 끝이므로, `reset()` 메서드를 명시적으로 호출:

```rust
pub fn reset(&mut self) {
    self.current = 0;           // 포인터를 처음으로
    if self.shuffle {
        self.shuffle_indices(); // 인덱스 재셔플
    }
}
```

**에폭 간 흐름**:

```
epoch 0: [셔플] → 배치0 → 배치1 → ... → 배치9 → 소진(None)
         reset()
epoch 1: [재셔플] → 배치0 → 배치1 → ... → 배치9 → 소진(None)
         reset()
...
```

### 전체 구조체

```rust
pub struct DataLoader<'a> {
    dataset: &'a dyn Dataset,  // 데이터셋 참조
    batch_size: usize,
    shuffle: bool,
    rng_state: u64,            // LCG 셔플용 내장 RNG
    indices: Vec<usize>,       // [0, 1, ..., N-1] (셔플 대상)
    current: usize,            // 현재 위치 포인터
}
```

---

## 3. Train/Test 분리

### 왜 분리하는가

학습 데이터로만 평가하면 **암기(overfitting)**도 좋은 성적이 된다:

- 학생이 시험 문제를 미리 알고 외워서 100점 → 실력이 아님
- 처음 보는 문제에서도 잘 풀어야 진짜 실력

마찬가지로:

- **train accuracy 99%**: 학습 데이터를 거의 외운 것일 수도 있음
- **test accuracy 95%**: 처음 보는 데이터에서도 잘 맞춤 → 진짜 학습됨

### Spiral 데이터셋의 Train/Test

```rust
let train_set = Spiral::new(true);   // seed=1984 → 학습용 300개
let test_set = Spiral::new(false);   // seed=2020 → 평가용 300개
```

같은 나선 구조지만 **다른 시드**로 생성되어 다른 점들이 만들어진다. 모델이 "나선 패턴"을 학습했다면 test에서도 잘 분류할 수 있고, 단순히 점 위치를 외웠다면 test에서 정확도가 떨어진다.

### no_grad 사용

```rust
{
    let _guard = no_grad();  // 역전파 그래프 생성 OFF
    for (x, t) in &mut test_loader {
        let y = model.forward(&x);
        let loss = softmax_cross_entropy_simple(&y, &t);
        let acc = accuracy(&y, &t);
        // ... 기록만, backward 안 함
    }
}  // _guard 소멸 → 자동으로 역전파 그래프 생성 복원
```

테스트 시에는 파라미터를 업데이트하지 않으므로:

- `backward()`를 호출하지 않음
- 따라서 계산 그래프를 만들 필요가 없음
- `no_grad()`로 비활성화하면 **메모리 절약 + 속도 향상**

> **RAII 패턴**: `no_grad()`가 반환하는 `NoGradGuard`가 스코프를 벗어나면 자동으로 이전 상태를 복원. Python의 `with dezero.no_grad():`와 동일한 역할을 Rust의 소유권 시스템으로 구현.

---

## 4. 학습 결과 분석

```
epoch 1:   train acc 33.7%, test acc 33.3%  ← 랜덤 (1/3 = 33.3%)
epoch 91:  train acc 84.3%, test acc 86.0%
epoch 300: train acc 98.3%, test acc 95.7%
```

### 초기 상태 (epoch 1)

accuracy 33.3% = 3클래스 중 무작위로 하나 고르는 수준. 모델이 아직 아무것도 학습하지 않았다.

> **왜 정확히 1/3인가?** 초기 가중치가 작은 랜덤값이므로 모든 클래스에 비슷한 점수를 부여. softmax로 변환하면 거의 $[0.33, 0.33, 0.33]$에 가까움. 300개 × 3클래스에서 균등 분배 → $\approx 33.3%$

### 수렴 후 (epoch 300)

- **train acc 98.3%**: 학습 데이터 300개 중 295개 정답
- **test acc 95.7%**: 처음 보는 데이터 300개 중 287개 정답
- **train > test**: 약간의 overfitting이 있지만, 차이가 작으므로(2.6%p) 양호

### loss vs accuracy의 관계

|epoch|train loss|train acc|
|---|---|---|
|1|1.158|33.7%|
|91|0.388|84.3%|
|300|0.085|98.3%|

둘 다 개선 방향은 같지만(loss↓, acc↑), loss는 연속적으로 부드럽게 감소하는 반면 accuracy는 이산적으로 변한다(맞거나 틀리거나). 학습에는 미분 가능한 loss를, 보고에는 직관적인 accuracy를 사용하는 이유.

---

## 5. 전체 학습 루프 구조

```rust
for epoch in 0..max_epoch {
    // ── 학습 ────────────────────
    for (x, t) in &mut train_loader {    // DataLoader가 배치 제공
        let y = model.forward(&x);       // 순전파
        let loss = softmax_cross_entropy_simple(&y, &t);  // 손실
        let acc = accuracy(&y, &t);      // 정확도 (평가용)
        model.cleargrads();              // 기울기 초기화
        loss.backward(false, false);     // 역전파
        optimizer.update();              // SGD 업데이트
    }

    // ── 평가 (no_grad) ─────────
    let _guard = no_grad();
    for (x, t) in &mut test_loader {
        let y = model.forward(&x);
        let loss = softmax_cross_entropy_simple(&y, &t);
        let acc = accuracy(&y, &t);
        // backward 없음, update 없음
    }

    train_loader.reset();   // 다음 에폭 준비 (재셔플)
    test_loader.reset();
}
```