# nn & nn.function

> 필요 패키지
>
> ```python
> import torch.nn as nn
> import torch.nn.functional as F
> ```



![파이토치(hqdefault-1593611043261.jpg) 3. nn & nn.functional - YouTube](https://i.ytimg.com/vi/SMOuUKrhTro/hqdefault.jpg)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

input = torch.ones(1, 1, 3, 3)
filter = torch.ones(1, 1, 3, 3)

input = Variable(input, requires_grad=True)
print(input)
"""
Variable containing:
(0, 0, ...) =
1 1 1
1 1 1
1 1 1
"""

filter = Variable(filter)
out = F.conv2d(input, filter)
print(out)
"""
Variable containing:
(0, 0, ...) =
 9
"""

out.backward()
print(out.grad_f)
"""
<ConvNdBackward object at ...>
"""

print(input.grad)
"""
Variable containing:
(0, 0, ...) =
1 1 1
1 1 1
1 1 1
"""
```



### 3x3 Conv2d

```python
input = Variable(torch.ones(1, 1, 3, 3), requires_grad=True)
filter = filter + 1
out = F.conv2d(input, filter)
print(out)
"""
Variable containing:
(0, 0, ...) =
18
"""
out.backward()
print(out.grad_fn)
"""
<ConvNdBackward object at ...>
"""

print(input_grad)
"""
Variable containing:
(0, 0, ...) =
2 2 2
2 2 2
2 2 2
"""
```



### nn.Conv2d

```python
import torch
import torch.nn as nn
import torch.autograd import Variable

input = torch.ones(1,1,3,3)
input = Variable(input, requires_grad=True)

func = nn.Conv2d(1,1,3)

func.weight
"""
Parameter containing:
(0, 0, ...) =
-0.0399 -0.1940 -0.1945
 0.0039  0.3010 -0.2322
 0.0276  0.1099  0.2031
"""

out = func(input)
print(out)
"""
Variable containing:
(0, 0, ..) =
-0.3172
"""

print(input.grad)
# None

out.backward()

print(input.grad)
"""
Variable containing:
(0, 0, ...) =
-0.0399 -0.1940 -0.1945
 0.0039  0.3010 -0.2322
 0.0276  0.1099  0.2031
"""
```







