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
3
3
train
train.txt
weights.txt
.1
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
3
3
3
run
inputs.txt
weights.txt
```