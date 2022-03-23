<!-- This file contains data for `control.txt`. It is NOT called by `main.cpp`. -->

`control.txt` training format:
```
[number of inputs]
[number of hiddens1]
[number of hiddens2]
[number of outputs]
train
[train file name]
[weights file name]
[load weights from file]
[random weight minimum]
[random weight maximum]
[max iterations]
[error threshold]
[learning factor]
```

`control.txt` example training configuration:
```
3
5
20
3
train
train.txt
weights.txt
0
-1
1.5
100000
0.0007
.3
```

`control.txt` running format:
```
[number of inputs]
[number of hiddens]
[number of outputs]
run
[inputs file name]
[weights file name]
```

`control.txt` example running configuration
```
2
5
20
3
run
inputs.txt
weights.txt

```
  3 {1.000000, 1.000000} expects {1.000000, 1.000000, 0.000000}, outputs {0.991829, 0.999354, 0.011525} with error {0.000100}