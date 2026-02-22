# step33: 이중 역전파 (Double Backpropagation)

  

## 핵심 변경: grad의 타입을 ArrayD → Variable로

  

### 왜 Variable로 바꿔야 하는가?

  

기울기(grad)도 cos, mul, pow 같은 **연산을 거쳐 만들어진 값**이다.

그 연산 이력(creator 체인)을 보존해야 다시 미분할 수 있다.

  

```

예) f(x) = x⁴ - 2x², x = 2.0

  

ArrayD일 때: x.grad = 24.0 ← 숫자일 뿐, 끝

Variable일 때: x.grad = Variable {

data: 24.0,

creator: SubFn ← MulFn ← PowFn ← x

}

→ "24.0은 4x³-4x를 계산해서 나온 값"이라는 정보가 남아있음

→ grad.backward() 하면 이 체인을 따라 f''(x) = 12x²-4 자동 계산

```

  

`backward()`는 Variable의 메서드이다. ArrayD에는 `backward()`가 없다.

그래서 `grad.backward()`로 2차 미분을 자동 계산하려면 grad가 반드시 Variable이어야 한다.

  

```rust

// grad가 ArrayD일 때

let gx: ArrayD<f64> = x.grad();

gx.backward(); // ❌ 컴파일 에러! ArrayD에는 backward()가 없음

  

// grad가 Variable일 때

let gx: Variable = x.grad_var();

gx.backward(); // ✅ creator 체인을 따라 f''(x) 자동 계산

```

  

## 이중 역전파란?

  

역전파(backward)를 **두 번** 하는 것이다.

  

```

순전파: x → f(x) → y (값 계산)

1차 역전파: y.backward() → f'(x) (1차 미분)

2차 역전파: f'(x).backward() → f''(x) (2차 미분)

```

  

step29에서는 f''(x) = 12x²-4를 사람이 직접 수식으로 써야 했지만,

이중 역전파를 쓰면 backward 두 번 호출만으로 자동 계산된다.

  

## grad()와 grad_var()

  

내부에 저장된 건 동일한 `grad: Option<Variable>` 하나이다. 꺼내는 방식만 다르다.

  

```

저장된 상태: grad = Some(Variable { data: 24.0, creator: SubFn←... })

```

  

**grad()** — Variable에서 data(숫자)만 벗겨서 반환:

```rust

pub fn grad(&self) -> Option<ArrayD<f64>> {

self.inner.borrow().grad.as_ref().map(|g| g.data())

// ^^^^^^^^

// Variable에서 data만 추출 → 24.0

}

```

  

**grad_var()** — Variable 통째로 반환:

```rust

pub fn grad_var(&self) -> Option<Variable> {

self.inner.borrow().grad.clone()

// ^^^^^^^^

// Variable 자체를 반환 → Variable { data: 24.0, creator: ... }

}

```

  

`grad()`를 `Option<Variable>`로 바꾸면 기존 step 파일들이 전부 깨진다:

```rust

// step29 등에서 이렇게 쓰고 있음

let grad = x.grad().unwrap(); // ArrayD를 기대

x.set_data(x.data() - &grad / &gx2); // ArrayD 연산

```

그래서 기존 호환용 `grad()`는 그대로 두고, Variable이 필요한 step33부터 `grad_var()`를 사용한다.

  

참고: Variable이 Rc(참조 카운트 포인터)라서 clone해도 내부 데이터가 복사되는 게 아니라

같은 데이터를 가리키는 포인터가 하나 더 생기는 것이다.

  

## backward(create_graph) 매개변수

  

```rust

pub fn backward(&self, retain_grad: bool, create_graph: bool)

```

  

- `create_graph=false`: 기존과 동일. 역전파 계산 시 그래프 생성 비활성화.

- `create_graph=true`: 역전파 계산 자체도 그래프에 기록. 이중 역전파 가능.

  

내부적으로 Python의 `using_config('enable_backprop', create_graph)`에 해당하는

`using_backprop(create_graph)` 가드를 사용한다.

  

## step29 vs step33 비교

  

```rust

// step29: f''(x)를 사람이 직접 계산

fn gx2(x: &ArrayD<f64>) -> ArrayD<f64> {

12.0 * x.mapv(|v| v * v) - 4.0 // 수동으로 유도한 공식

}

y.backward(false, false);

let grad = x.grad().unwrap();

x.set_data(x.data() - &grad / &gx2(&x.data()));

  

// step33: backward 두 번으로 자동 계산

y.backward(false, true); // 1차 역전파 + 그래프 기록

let gx = x.grad_var().unwrap(); // f'(x) as Variable

x.cleargrad();

gx.backward(false, false); // 2차 역전파 → f''(x) 자동 계산

let gx2 = x.grad().unwrap(); // f''(x) as ArrayD

x.set_data(x.data() - &gx.data() / &gx2);

```