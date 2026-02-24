## Step 2-6: 연산자 오버로딩 + Display

### 핵심 질문

> `a.mont_mul(&b)` 대신 `a * b`로 쓸 수는 없나?

Rust에서는 `std::ops` 트레잇을 구현하면 연산자를 커스텀 타입에 붙일 수 있다.

### Rust의 연산자 = 트레잇 메서드

Rust에서 `a + b`를 쓰면 컴파일러가 실제로는 이렇게 호출한다:

```rust
a + b       →  <Fp as Add<Fp>>::add(a, b)
a - b       →  <Fp as Sub<Fp>>::sub(a, b)
a * b       →  <Fp as Mul<Fp>>::mul(a, b)
-a          →  <Fp as Neg>::neg(a)
```

연산자는 문법적 설탕(syntactic sugar)일 뿐, 실체는 트레잇의 메서드다.

### Add 구현

```rust
impl Add for Fp {
    type Output = Fp;
    fn add(self, rhs: Fp) -> Fp { Fp::add(&self, &rhs) }
}
```

|요소|의미|
|---|---|
|`impl Add for Fp`|`Fp + Fp`를 가능하게 함|
|`type Output = Fp`|결과 타입도 Fp|
|`fn add(self, rhs: Fp)`|값으로 받음 (Copy이므로 OK)|
|`Fp::add(&self, &rhs)`|Step 2-3에서 만든 모듈러 덧셈 호출|

> [!note] `self` vs `&self` 트레잇의 `add(self, rhs)`는 값을 소비하는데, `Fp`는 `Copy`이므로 문제없다. 32바이트(u64 × 4)는 복사 비용이 거의 없다.

### 참조 버전: `Add<&Fp>`

```rust
impl Add<&Fp> for Fp {
    type Output = Fp;
    fn add(self, rhs: &Fp) -> Fp { Fp::add(&self, rhs) }
}
```

이걸 구현하면:

```rust
let c = a + b;   // Fp + Fp  (Add<Fp>)
let c = a + &b;  // Fp + &Fp (Add<&Fp>)
```

루프 안에서 `&b`로 빌려쓰면 불필요한 복사를 피할 수 있다. 지금은 Copy라 성능 차이가 없지만, 관용적 패턴.

### Sub, Mul, Neg

패턴은 전부 동일하다:

```rust
impl Sub for Fp {
    type Output = Fp;
    fn sub(self, rhs: Fp) -> Fp { Fp::sub(&self, &rhs) }
}

impl Mul for Fp {
    type Output = Fp;
    fn mul(self, rhs: Fp) -> Fp { self.mont_mul(&rhs) }
    //                            ↑ 곱셈만 mont_mul을 직접 호출
}

impl Neg for Fp {
    type Output = Fp;
    fn neg(self) -> Fp { Fp::neg(&self) }
    //                    ↑ 단항 연산자라 인자가 self 하나
}
```

> [!note] Mul이 `self.mont_mul(&rhs)`를 호출하는 이유 유한체 곱셈은 Montgomery 곱셈이다. `Fp::mul`이라는 메서드는 없고, `mont_mul`이 곱셈의 실체.

### Display

```rust
impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.to_repr();
        write!(f, "Fp(0x{:016x}{:016x}{:016x}{:016x})", r[3], r[2], r[1], r[0])
    }
}
```

Montgomery form이 아닌 **실제 값**을 출력한다:

```rust
let a = Fp::from_u64(42);
println!("{}", a);
// Fp(0x0000000000000000000000000000000000000000000000000000000000000002a)
//                                                                    ↑ 42 = 0x2a
```

|포맷|의미|
|---|---|
|`{:016x}`|16진수, 16자리, 앞을 0으로 채움|
|`r[3], r[2], r[1], r[0]`|big-endian 순서로 출력 (사람이 읽기 편하게)|

> [!important] to_repr()를 쓰는 이유 내부 저장값은 Montgomery form (`a * R mod p`)이라 그대로 출력하면 의미 없는 숫자가 나온다. `to_repr()`로 normal form으로 변환한 뒤 출력해야 사람이 읽을 수 있는 값이 된다.

### 체(Field) 공리 테스트

연산자가 생겼으니 수학적 성질을 검증할 수 있다:

```rust
// 항등원: a + 0 = a, a * 1 = a
assert_eq!(a + Fp::ZERO, a);
assert_eq!(a * Fp::ONE, a);

// 교환법칙: a + b = b + a, a * b = b * a
assert_eq!(a + b, b + a);
assert_eq!(a * b, b * a);

// 결합법칙: (a + b) + c = a + (b + c)
assert_eq!((a + b) + c, a + (b + c));
assert_eq!((a * b) * c, a * (b * c));

// 분배법칙: a * (b + c) = a*b + a*c
assert_eq!(a * (b + c), a * b + a * c);
```

이 공리들이 **전부 통과해야** $\mathbb{F}_p$가 올바른 유한체라고 할 수 있다. 하나라도 실패하면 구현에 버그가 있는 것.

> [!tip] 왜 공리 테스트가 중요한가? Montgomery 구현은 상수 하나만 틀려도 모든 곱셈이 조용히 틀린 값을 낸다. 공리 테스트는 "구현이 수학적으로 올바른 체인가"를 한꺼번에 검증하는 통합 테스트 역할을 한다.