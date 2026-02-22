# Variable::backward 동작 원리

  

## 데이터 구조

  

Variable과 FuncState가 서로를 참조하며 계산 그래프를 형성한다.

  

```

+----------------------------------+ +------------------------------------+

|              VarInner            | |               FuncState            |

+----------------------------------+ +------------------------------------+

| data: ArrayD<f64>                | | func: Box<dyn Function> |

| grad: Option<Variable>           | | inputs: Vec<Variable> |

| creator: Option<FuncStateRef>  --+>| outputs: Vec<Weak<VarInner>> |

| generation: u32                  | | generation: u32 |

+----------------------------------+ +------------------------------------+

Variable가 FuncState를 가리킴 (creator)

FuncState가 Variable을 가리킴 (inputs, outputs)

```

  

## 순전파: Func::call에서 그래프가 만들어지는 과정

  

```rust

// sin(x)를 예로 들면

pub fn sin(x: &Variable) -> Variable {

Func::new(SinFn).call(&[x])

}

```

  

### call 내부 (387~415행)

  

```rust

pub fn call(&self, inputs: &[&Variable]) -> Variable {

// 1. input의 data를 꺼내서 forward 실행

let xs = inputs의 data들; // 388행

let ys = self.state.func.forward(&xs); // 390행

let outputs = ys를 Variable로 감싸기; // 391행

  

// 2. ENABLE_BACKPROP이 true일 때만 그래프 연결

if ENABLE_BACKPROP { // 393행

// FuncState에 inputs 저장

state.inputs = inputs.clone(); // 402행

  

// output.creator = 이 FuncState

for output in &outputs {

output.set_creator(&self.state); // 406행

}

// => output.creator = Some(FuncState) // 161행

// output.generation = func_gen + 1 // 162행

  

// FuncState에 outputs 저장 (Weak 참조)

state.outputs = outputs의 Weak 참조들; // 408행

}

  

return output

}

```

  

### sin(&x) 실행 후 상태

  

```

x (VarInner) FuncState(SinFn) y (VarInner)

+----------------+ +---------------------+ +----------------+

| data: 1.0 | inputs | func: SinFn |outputs| data: 0.841 |

| grad: None |<--------| inputs: [x] |------>| grad: None |

| creator: None | | outputs: [Weak->y] | | creator: ------+-->(이 FuncState)

| gen: 0 | | generation: 0 | | gen: 1 |

+----------------+ +---------------------+ +----------------+

```

  

- **x.creator = None**: x는 사용자가 만든 값이므로 creator 없음

- **y.creator = FuncState(SinFn)**: y는 SinFn이 만들었으므로 creator 있음

- **FuncState.inputs = [x]**: 역전파 때 input을 찾기 위해 저장

- **FuncState.outputs = [Weak->y]**: 역전파 때 output의 grad를 가져오기 위해 저장

  

## 역전파: backward에서 그래프를 거슬러 올라가는 과정

  

### backward 코드 구조 (169~237행)

  

```rust

pub fn backward(&self, retain_grad: bool, create_graph: bool) {

// 1. 자기 자신의 grad를 ones로 초기화 (dy/dy = 1)

if inner.grad.is_none() {

inner.grad = Some(Variable::new(ArrayD::ones(...))); // 173행

}

  

// 2. 자신의 creator를 funcs 큐에 추가

if let Some(creator) = self.creator { // 191행

funcs.push(creator);

}

  

// 3. 큐에서 함수를 꺼내며 역전파 반복

while let Some(state_ref) = funcs.pop() { // 195행

  

// a. ENABLE_BACKPROP을 create_graph 값으로 설정

let _guard = using_backprop(create_graph); // 200행

  

// b. FuncState.outputs에서 grad를 꺼내 gys로 만듦

let gys: Vec<Variable> = state.outputs // 204행

.iter()

.map(|o| o.upgrade() // Weak -> Rc로 복원

.borrow().grad.clone().unwrap()) // 그 Variable의 grad를 꺼냄

.collect();

  

// c. FuncState.inputs를 xs로 꺼냄

let xs: Vec<Variable> = state.inputs.clone(); // 209행

  

// d. backward 호출 -> 입력에 대한 기울기 계산

let gxs = state.func.backward(&xs, &gys); // 211행

  

// e. 계산된 기울기를 input의 grad에 저장

for (input, gx) in inputs.zip(gxs) { // 215행

input.grad = Some(gx); // 218행

  

// f. input에 creator가 있으면 funcs에 추가 (더 거슬러 올라감)

if let Some(creator) = input.creator { // 224행

funcs.push(creator);

}

}

}

}

```

  

## 단순한 경우: y = sin(x)

  

### 순전파 후 상태

  

```rust

let x = Variable::new(arr0(1.0));

let y = sin(&x); // Func::new(SinFn).call(&[x])

```

  

```
x --> [SinFn] --> y

x:      data=1.0,   grad=None,  creator=None,   gen=0
SinFn:  inputs=[x], outputs=[Weak->y],          gen=0
y:      data=0.841, grad=None,  creator=SinFn,  gen=1

```

  

### y.backward(false, false) 실행

  

```
[173행] y.grad = Some(Variable::new(ones))     <-- dy/dy = 1

  x:      grad=None,    creator=None
  SinFn:  inputs=[x],   outputs=[Weak->y]
  y:      grad=ones,    creator=SinFn

[191행] y.creator = Some(SinFn) -> funcs에 추가
        funcs = [FuncState(SinFn)]

[195행] funcs.pop() -> FuncState(SinFn) 꺼냄

[204행] state.outputs = [Weak->y]
        -> y.grad를 꺼냄
        -> gys = [ones]

[209행] state.inputs = [x]
        -> xs = [x]

[211행] SinFn.backward(xs=[x], gys=[ones])
        -> cos(x) * ones = cos(1.0)
        -> gxs = [cos(1.0)]

[218행] x.grad = Some(cos(1.0))

  x:      grad=cos(1),  creator=None
  SinFn:  inputs=[x],   outputs=[Weak->y]
  y:      grad=ones,    creator=SinFn

[224행] x.creator = None -> funcs에 추가 안 함

[195행] funcs 비었음 -> 종료
```

  

## 합성 함수: y = sin(x^2)

  

gys가 ones가 아닌 경우.

  

### 순전파 후 상태

  

```
x                          t (= x^2)                    y
+----------------+         +----------------+           +----------------+
| data: 2.0      |         | data: 4.0      |           | data: -0.756   |
| creator: None  |         | creator: PowFn |           | creator: SinFn |
| gen: 0         |         | gen: 1         |           | gen: 2         |
+----------------+         +----------------+           +----------------+
        ^                       ^   ^                       ^
        |                       |   |                       |
    [PowFn]                     |   [SinFn]                 |
    inputs: [x]                 |   inputs: [t]             |
    outputs: [Weak->t] ---------+   outputs: [Weak->y] -----+


```

  

### y.backward(false, false) 실행

  

```

[173행] y.grad = ones

  

[191행] y.creator = SinFn -> funcs = [SinFn]

  

--- SinFn 처리 ---

[204행] SinFn.outputs = [Weak->y] -> gys = [y.grad] = [ones] <-- 첫 함수만 ones

[209행] SinFn.inputs = [t] -> xs = [t]

[211행] SinFn.backward -> cos(t) * ones = cos(x^2)

[218행] t.grad = cos(x^2)

[224행] t.creator = PowFn -> funcs = [PowFn]

  

--- PowFn 처리 ---

[204행] PowFn.outputs = [Weak->t] -> gys = [t.grad] = [cos(x^2)] <-- ones가 아님!

[209행] PowFn.inputs = [x] -> xs = [x]

[211행] PowFn.backward -> 2x * cos(x^2) <-- 연쇄법칙

[218행] x.grad = 2x*cos(x^2)

[224행] x.creator = None -> 종료

```

  

**핵심**: gys[0]이 ones인 건 맨 첫 함수뿐. 이후는 앞에서 계산된 기울기가 gys로 전달된다.

  

## 이중 역전파: create_graph=true

  

### 순전파

  

```rust

let y = sin(&x);

```

```

x -> SinFn -> y

```

  

### 1차 역전파: y.backward(false, true)

  

create_graph=true이므로 `using_backprop(true)` -> ENABLE_BACKPROP=true.

backward 안에서 호출되는 cos, mul 등도 **Func::call을 통해 새 그래프 노드를 만든다**.

  

```

[200행] using_backprop(true) -> ENABLE_BACKPROP = true

[211행] SinFn.backward(xs=[x], gys=[ones]) 호출:

  fn backward(&self, xs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
      vec![&cos(&xs[0]) * &gys[0]]
  }

  (1) cos(&x) 실행 -> Func::new(CosFn).call(&[x])
      call 내부 (393행): ENABLE_BACKPROP=true이므로 그래프 연결!
      -> FuncState(CosFn).inputs = [x]
      -> cos_var.creator = FuncState(CosFn)     <-- 새 노드!

  (2) &cos_var * &ones 실행 -> Func::new(MulFn).call(&[cos_var, ones])
      call 내부 (393행): ENABLE_BACKPROP=true이므로 그래프 연결!
      -> FuncState(MulFn).inputs = [cos_var, ones]
      -> gx.creator = FuncState(MulFn)           <-- 새 노드!

[218행] x.grad = Some(gx)

```

  

1차 역전파 후 새로 만들어진 그래프:

```
x -> FuncState(CosFn) -> cos_var --+
     inputs: [x]                   +--> FuncState(MulFn) -> gx (= cos(x))
     outputs: [cos_var]            |    inputs: [cos_var, ones]
                                   |    outputs: [gx]
                         ones -----+
                         (creator: None)
```

  

### 2차 역전파: gx.backward(false, true)

  

gx의 creator(MulFn)부터 시작해서 거슬러 올라간다.

  

```

[173행] gx.grad = ones_new

  

[191행] gx.creator = FuncState(MulFn) -> funcs = [MulFn]

  

--- MulFn 처리 ---

[204행] MulFn.outputs = [Weak->gx] -> gys = [gx.grad] = [ones_new]

[209행] MulFn.inputs = [cos_var, ones] -> xs = [cos_var, ones]

[211행] MulFn.backward(xs, gys):

gx0 = xs[1] * gys[0] = ones * ones_new = ones_result -> cos_var용

gx1 = xs[0] * gys[0] = cos_var * ones_new -> ones용

[218행] cos_var.grad = ones_result

[224행] cos_var.creator = FuncState(CosFn) 있음 -> funcs = [CosFn]

ones.creator = None -> 무시

  

--- CosFn 처리 ---

[204행] CosFn.outputs = [Weak->cos_var] -> gys = [cos_var.grad] = [ones_result]

[209행] CosFn.inputs = [x] -> xs = [x]

[211행] CosFn.backward(xs, gys):

-sin(x) * ones_result = -sin(x)

[218행] x.grad = Some(-sin(x)) <-- 2차 미분 완료!

[224행] x.creator = None -> 종료

```

  

### 정리

  

```

y.backward(true) -> SinFn::backward 호출 -> x.grad = cos(x) (1차 미분)

gx.backward(true) -> MulFn -> CosFn 호출 -> x.grad = -sin(x) (2차 미분)

gx2.backward(true) -> 또 새 그래프 따라감 -> x.grad = -cos(x) (3차 미분)

```

  

매번 backward할 때마다 create_graph=true이면 새 그래프가 만들어지고,

그 그래프에 대해 다시 backward하면 한 차수 높은 미분이 자동으로 계산된다.

  

## create_graph=true vs false

  

```rust

let _guard = using_backprop(create_graph); // 200행

let gxs = state.func.backward(&xs, &gys); // 211행

```

  

- `create_graph=true`: ENABLE_BACKPROP=true

-> backward 안의 cos, mul 등이 Func::call에서 **그래프 노드를 만듦** (393행 조건 통과)

-> 결과 Variable에 creator가 설정됨 -> 다시 backward 가능

  

- `create_graph=false`: ENABLE_BACKPROP=false

-> Func::call에서 **그래프 연결을 건너뜀** (393행 조건 실패)

-> 결과 Variable에 creator 없음 -> 숫자만 있고 다시 backward 불가능

  

이중 역전파가 필요 없으면 false로 하는 게 효율적이다.

  

## grad와 gys의 관계

  

grad는 Variable에 저장된 기울기이고, gys는 backward 루프에서 꺼내 쓰는 것이다.

  

```

1. y.grad = ones <-- backward() 시작 시 초기화 (173행)

2. gys[0] = y.grad <-- FuncState.outputs에서 grad를 꺼냄 (204~208행)

3. gxs = func.backward(xs, gys) <-- 함수에 전달하여 기울기 계산 (211행)

4. input.grad = gx <-- 결과를 FuncState.inputs의 grad에 저장 (218행)

5. 다음 함수에서 gys[0] = input.grad <-- 저장된 grad가 다음 함수의 gys가 됨

```

  

이렇게 grad -> gys -> backward -> gx -> 다음 grad로 연쇄적으로 전파되는 것이 역전파이다.