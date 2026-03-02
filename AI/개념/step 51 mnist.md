# Step 51: MNIST 손글씨 숫자 분류

## step50 → step51: 무엇이 바뀌었나

||step50 (Spiral)|step51 (MNIST)|
|---|---|---|
|데이터셋|나선형 2D 좌표|손글씨 숫자 이미지|
|샘플 수|train 300, test 300|train **60,000**, test **10,000**|
|입력 차원|2|**784** (28×28 픽셀)|
|클래스 수|3|**10** (숫자 0~9)|
|은닉 크기|10|**1000**|
|에폭 수|300|**5** (데이터 200배 많으므로)|
|학습률|1.0|**0.01**|
|데이터 출처|코드 내 생성|**인터넷 다운로드**|

학습 루프의 코드는 step50과 **완전히 동일**하다 — `Spiral::new(true)`를 `MNIST::new(true)`로 바꾸는 것만으로 동작. 이것이 step49~50에서 만든 추상화(Dataset, DataLoader)의 효과.

---

## 1. MNIST 데이터셋이란

### 개요

MNIST (Modified National Institute of Standards and Technology)는 머신러닝의 **"Hello World"** 데이터셋. 1998년 Yann LeCun이 공개한 이래 가장 많이 사용된 벤치마크 중 하나.

- **28×28** 그레이스케일 이미지 (손으로 쓴 숫자 0~9)
- **train**: 60,000개, **test**: 10,000개
- 각 픽셀은 0(흑)~255(백) 정수

### 전처리

1. **Flatten**: 28×28 2D 이미지 → 784차원 1D 벡터 $$[28 \times 28] \rightarrow [784]$$
    
2. **정규화**: 픽셀값을 $[0, 255] \rightarrow [0, 1]$로 스케일링 $$x_{\mathrm{norm}} = \frac{x_{\mathrm{raw}}}{255}$$
    

> **왜 정규화하는가?** 입력이 0~255 범위이면 가중치와의 곱이 매우 커져서:
> 
> - sigmoid가 포화 영역에 빠짐 ($\sigma(100) \approx 1$, 기울기 $\approx 0$)
> - 학습이 극도로 느려지거나 발산
> 
> $[0, 1]$ 범위로 줄이면 초기 가중치(Xavier 초기화)와 스케일이 맞아서 학습이 안정적.

---

## 2. IDX 파일 형식

MNIST는 자체 바이너리 형식인 **IDX**로 저장된다. gzip으로 압축되어 `.gz` 파일로 배포.

### 이미지 파일 구조

```
오프셋  크기   내용
0000    4B    매직 넘버: 0x00000803 (2051)
0004    4B    이미지 수: 60000 (big-endian)
0008    4B    행 수: 28
0012    4B    열 수: 28
0016    784B  첫 번째 이미지 픽셀 데이터
0800    784B  두 번째 이미지 픽셀 데이터
...
```

> **매직 넘버 0x00000803의 의미**:
> 
> - 처음 2바이트 `0x0000`: 항상 0
> - 3번째 바이트 `0x08`: 데이터 타입 (unsigned byte)
> - 4번째 바이트 `0x03`: 차원 수 (3차원 = count × rows × cols)

### 라벨 파일 구조

```
오프셋  크기   내용
0000    4B    매직 넘버: 0x00000801 (2049)
0004    4B    라벨 수: 60000
0008    1B    첫 번째 라벨 (0~9)
0009    1B    두 번째 라벨
...
```

### Rust 파싱 구현

```rust
fn load_mnist_images(gz_path: &str) -> Vec<Vec<f64>> {
    // 1. gzip 해제
    let file = std::fs::File::open(gz_path)?;
    let mut decoder = GzDecoder::new(file);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf)?;

    // 2. 헤더 파싱 (big-endian 4바이트 정수)
    let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let count = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let rows = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let cols = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    let pixels = rows * cols;  // 784

    // 3. 픽셀 데이터 → f64로 변환 + 정규화
    for i in 0..count {
        let offset = 16 + i * pixels;
        let img: Vec<f64> = buf[offset..offset + pixels]
            .iter()
            .map(|&b| b as f64 / 255.0)  // [0,255] → [0,1]
            .collect();
        images.push(img);
    }
}
```

---

## 3. 다운로드와 캐싱

### 흐름

```
MNIST::new(true)
  ├─ 캐시 확인: ~/.dezero/mnist/train-images-idx3-ubyte.gz 있나?
  │   ├─ 있음 → 스킵
  │   └─ 없음 → ureq로 HTTP GET 다운로드 → 파일 저장
  ├─ flate2로 gzip 해제
  ├─ IDX 바이너리 파싱
  └─ Vec<Vec<f64>> + Vec<usize> 반환
```

### 다운로드 파일 목록

|파일|크기 (압축)|내용|
|---|---|---|
|`train-images-idx3-ubyte.gz`|~9.9 MB|train 이미지 60,000개|
|`train-labels-idx1-ubyte.gz`|~29 KB|train 라벨 60,000개|
|`t10k-images-idx3-ubyte.gz`|~1.6 MB|test 이미지 10,000개|
|`t10k-labels-idx1-ubyte.gz`|~4.5 KB|test 라벨 10,000개|

### 새 의존성

|크레이트|용도|
|---|---|
|`ureq`|동기 HTTP 클라이언트 (다운로드)|
|`flate2`|gzip 압축 해제|

```rust
fn download_if_missing(url: &str, path: &str) {
    if std::path::Path::new(path).exists() {
        return;  // 캐시에 있으면 스킵
    }
    let resp = ureq::get(url).call()?;
    let mut bytes = Vec::new();
    resp.into_body().as_reader().read_to_end(&mut bytes)?;
    std::fs::write(path, &bytes)?;
}
```

---

## 4. MNIST가 Dataset 트레잇에 들어맞는 방식

```rust
pub struct MNIST {
    data: Vec<Vec<f64>>,  // N개 샘플, 각 784차원
    label: Vec<usize>,    // N개 라벨 (0~9)
}

impl Dataset for MNIST {
    fn len(&self) -> usize { self.data.len() }

    fn get(&self, index: usize) -> (Vec<f64>, usize) {
        (self.data[index].clone(), self.label[index])
    }
}
```

Spiral과 **동일한 패턴**: 전체 데이터를 메모리에 올리고, `get()`으로 개별 샘플 반환.

```rust
// step50 (Spiral)
let train_set = Spiral::new(true);     // 300개, 2D
let mut loader = DataLoader::new(&train_set, 30, true);

// step51 (MNIST) — 이 한 줄만 변경
let train_set = MNIST::new(true);      // 60000개, 784D
let mut loader = DataLoader::new(&train_set, 100, true);

// 학습 루프는 완전히 동일
for (x, t) in &mut loader { ... }
```

---

## 5. 모델 구조

```
입력 (784) → Linear(1000) → sigmoid → Linear(10) → softmax → 예측
```

|레이어|파라미터 수|
|---|---|
|Linear(784 → 1000)|W: 784,000 + b: 1,000 = **785,000**|
|Linear(1000 → 10)|W: 10,000 + b: 10 = **10,010**|
|**합계**|**795,010**|

> Spiral(2→10→3)의 파라미터는 63개였다. MNIST는 약 **12,600배** 많은 파라미터. 이래서 release 모드가 필요하고 (debug: 수 시간 vs release: 34초), 데이터가 많아도 5 에폭만으로 학습이 되는 것.

### Lazy Initialization의 역할

```rust
let model = MLP::new(&[1000, 10]);
// 이 시점에서 W1의 shape는 아직 모름 (입력 크기 = ?)

let y = model.forward(&x);  // x.shape = (100, 784)
// 첫 forward 시 W1: (784, 1000)이 자동 생성 (Xavier 초기화)
```

step44에서 도입한 lazy init 덕분에 입력 크기(784)를 명시할 필요 없이 `&[1000, 10]`만 지정하면 된다.

---

## 6. 학습 결과

```
epoch 1: train loss 1.9233, acc 54.3%  |  test loss 1.5457, acc 74.5%
epoch 2: train loss 1.2935, acc 76.5%  |  test loss 1.0484, acc 81.6%
epoch 3: train loss 0.9312, acc 81.5%  |  test loss 0.7968, acc 84.3%
epoch 4: train loss 0.7447, acc 83.8%  |  test loss 0.6608, acc 85.6%
epoch 5: train loss 0.6391, acc 85.3%  |  test loss 0.5795, acc 86.9%
```

### 분석

**초기 (epoch 1)**:

- train acc 54.3%: 10클래스 랜덤(10%)보다는 높지만 아직 학습 초기
- test acc 74.5%가 train보다 높은 이유: 1에폭 동안 모델이 급격히 개선되면서, 에폭 **초반** 배치들의 낮은 정확도가 평균을 끌어내림. test는 에폭 **끝**의 최신 모델로 평가.

**최종 (epoch 5)**:

- test acc 86.9%: sigmoid + SGD(lr=0.01)로는 양호한 결과
- ReLU + Adam으로 바꾸면 97%+ 달성 가능 (Python 코드의 주석에 언급)

### 학습률이 0.01인 이유

Spiral에서는 `lr = 1.0`이었지만 MNIST에서는 `lr = 0.01`.

> 입력 차원이 784로 커지면서 gradient의 크기도 커진다. $\nabla_W L$의 각 원소는 입력 $x$에 비례하므로, 입력 차원이 크면 gradient의 L2 norm이 커짐. 큰 gradient에 큰 학습률을 곱하면 파라미터가 한 번에 너무 크게 바뀌어 발산. 학습률을 낮춰서 안정적인 업데이트를 보장.
