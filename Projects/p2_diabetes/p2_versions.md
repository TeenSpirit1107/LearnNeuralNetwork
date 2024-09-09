# Neural Network, Practice 3: More Models
## Basics
- Date: 2024/9/9
- a continuation of practice 1: setup pyTorch and first model in pyTorch
## Resource 
- [2024PyTorch基础入门](https://blog.csdn.net/ccaoshangfei/article/details/126074300)
# Project 1: Diabetes Classification
## Resource 
- Code Source: [2024PyTorch基础入门](https://blog.csdn.net/ccaoshangfei/article/details/126074300)
    - gitHub:
- Data Source: [diabetes糖尿病数据集](https://download.csdn.net/download/ccaoshangfei/88711388?spm=1001.2101.3001.9500)
## Version 1: Original Version
- Loss-Epoch: ![loss-epoch](image/nn_p3_moreModels/1725879921098.png)
- Time: About 10 mins.
- Terminal:
```
Epoch[1/100],loss:0.657018
Epoch[2/100],loss:0.654177
Epoch[3/100],loss:0.653923
Epoch[4/100],loss:0.653442
...
Epoch[97/100],loss:0.646175
Epoch[98/100],loss:0.645299
Epoch[99/100],loss:0.646160
Epoch[100/100],loss:0.647016
```
## Next Step
### Problem
1. My graph is weird. Even in the very beginning, it doesn't have the same loss rate as described in the article, which is around 0.8; instead, it starts at bout 0.66, and ends at about 0.64. Not very significant. Could it just be a coincidence from randomly generating the initial parameters?
    - My Graph: ![loss-epoch](image/nn_p3_moreModels/1725879921098.png)
    - Author's Graph: ![author's loss-epoch](image/nn_p3_moreModels/1725880120796.png)
2. It seems that the epoch is too many. According to the author's graph, the loss rate converge at about 20 epochs.
### Solution
- Solution: Change the epoch number to 40; run the code again. 
- Expectation: start at similar loss as the author's; ends at similar loss rate, but with fewer epochs.

## Version 2: Reduce epoch to 40
- Result as expected.
- Terminal: 
```
Epoch[1/40],loss:0.795328
Epoch[2/40],loss:0.771633
Epoch[3/40],loss:0.752378
Epoch[4/40],loss:0.734586
...
Epoch[37/40],loss:0.643974
Epoch[38/40],loss:0.649514
Epoch[39/40],loss:0.647105
Epoch[40/40],loss:0.644685
```
- Graph
    - My Graph:! ![my loss-epoch, 40 epochs](image/nn_p3_moreModels/1725880736729.png)
    - Author's Graph: ![author's loss-epoch, 100 epochs](image/nn_p3_moreModels/1725880120796.png)
### Next Step
- Find out whether my Version 1's outcome is a coincidence due to the random initialization of the param.
    - print the random initialized params
    - print the random initial error in multiple random initializations

## Version 3: Test Random Initialization
- refer to gitHub
- As expected, parameters are randomly initialized; random initial errors vary greatly.
### Next Step
- Change sigmoid to ReLU. I'm not sure whether directly changing the function will work. 
- It's said that ReLU works better than sigmoid. Reason: refer to my machine learning note 1.

## Version 4: sigmoid change to ReLU, 40 Epochs
- Graph: 
    - My Version 4 ![v4](image/nn_p3_moreModels/1725885810064.png)
    - Author's Version ![author's loss-epoch, 100 epochs](image/nn_p3_moreModels/1725880120796.png)
- Terminal:
```
Epoch[1/40],loss:0.723948
Epoch[2/40],loss:0.652448
Epoch[3/40],loss:0.644904
Epoch[4/40],loss:0.644003
...
Epoch[26/40],loss:0.638964
Epoch[27/40],loss:0.640198
Epoch[28/40],loss:0.639180
Epoch[29/40],loss:0.638767
Epoch[30/40],loss:0.635981
Epoch[31/40],loss:0.633920
Epoch[32/40],loss:0.629841
Epoch[33/40],loss:0.624751
Epoch[34/40],loss:0.614766
Epoch[35/40],loss:0.614761
Epoch[36/40],loss:0.609119
Epoch[37/40],loss:0.602843
Epoch[38/40],loss:0.601925
Epoch[39/40],loss:0.596841
Epoch[40/40],loss:0.591384
```
### Conclusions
1. The first step converges so much! Perhaps this implies that ReLU is much better than sigmoid; but it may also implies that the learning rate is too large for ReLU.
2. Apparently, there's the overfitting problem after epoch 25.
### Next Step
- don't reduce epoch, but reduce learning rate?
- Problem: Sometimes the initial loss is too large. I believe it's due to some problems in the **optimizer**: the optimizer is designed for sigmoid, so I need to change it into another function corresponding to ReLU.
    - Sigmoid optimizer:`optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`
    - Terminal Output
```
Epoch[1/40],loss:65.353261
Epoch[2/40],loss:65.081522
Epoch[3/40],loss:65.081522
Epoch[4/40],loss:65.896739
Epoch[5/40],loss:65.217391
Epoch[6/40],loss:65.625000
···
```

## Version 4: ReLU, epoch 40, reduced learning rate

### Step 1: Change the Optimizer.

#### Step 1.1: Problem: which optimizer correspond to ReLU?
- I went to the definition of torch.optim, but found none of the following functions have a name directly related to ReLU, only one directly related to sigmoid (SGD).
```python
from . import 
lr_scheduler, swa_utils
from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .adamax import Adamax
from .adamw import AdamW
from .asgd import ASGD
from .lbfgs import LBFGS
from .nadam import NAdam
from .optimizer import Optimizer
from .radam import RAdam
from .rmsprop import RMSprop
from .rprop import Rprop
from .sgd import SGD
from .sparse_adam import SparseAdam
```
#### Step 1.2: Solution: Ask GPT For Help.
##### My Problem
In pyTorch package torch.optim, which one of these corresponds to ReLU, given that torch.optim.SGD corresponds to sigmoid: from . import lr_scheduler, swa_utils
from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .adamax import Adamax
from .adamw import AdamW
from .asgd import ASGD
from .lbfgs import LBFGS
from .nadam import NAdam
from .optimizer import Optimizer
from .radam import RAdam
from .rmsprop import RMSprop
from .rprop import Rprop
from .sgd import SGD
from .sparse_adam import SparseAdam
##### GPT's Reponse
> In PyTorch, the optimizers in the `torch.optim` package are not directly tied to specific activation functions like ReLU or sigmoid. Instead, they are general-purpose optimization algorithms that can be used with any model, regardless of the activation functions used in that model. 
>
> However, if you are asking which optimizer might typically be more suitable when using ReLU activation functions, there isn't a strict correspondence. That said, ReLU is often used with adaptive optimizers like `Adam`, `AdamW`, `RMSprop`, etc., because these optimizers can handle the sparse gradients that ReLU activation functions can produce. 
>
>To summarize:
>- **torch.optim.SGD** is a stochastic gradient descent optimizer and can be used with any activation function, including sigmoid.
>- **ReLU** does not correspond to a specific optimizer, but adaptive optimizers like `Adam`, `AdamW`, and `RMSprop` are popular choices due to their ability to handle sparse gradients and dynamic learning rates.
>
>In pyTorch package torch.optim, which one of these corresponds to ReLU, given that torch.optim.SGD corresponds to sigmoid: 
```python
from . import lr_scheduler, swa_utils
from .adadelta import Adadelta
from .adagrad import Adagrad
from .adam import Adam
from .adamax import Adamax
from .adamw import AdamW
from .asgd import ASGD
from .lbfgs import LBFGS
from .nadam import NAdam
from .optimizer import Optimizer
from .radam import RAdam
from .rmsprop import RMSprop
from .rprop import Rprop
from .sgd import SGD
from .sparse_adam import SparseAdam
```
##### Comment on the response
#### Step 1.3: Change optimizer.