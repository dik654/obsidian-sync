## Sigmoid 미분 유도

### 정의

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### 몫의 미분법 적용

$$f = 1, \quad g = 1 + e^{-x}$$

$$\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$$

$$f' = 0, \quad g' = -e^{-x}$$

$$\sigma'(x) = \frac{0 - 1 \cdot (-e^{-x})}{(1 + e^{-x})^2} = \frac{e^{-x}}{(1 + e^{-x})^2}$$

### σ(x)(1 - σ(x))와 동일함을 증명

$$\sigma(x)(1 - \sigma(x)) = \frac{1}{1+e^{-x}} \cdot \left(1 - \frac{1}{1+e^{-x}}\right)$$

$$= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}}$$

$$= \frac{e^{-x}}{(1+e^{-x})^2}$$

### 결론

$$\boxed{\sigma'(x) = \sigma(x)(1 - \sigma(x))}$$
