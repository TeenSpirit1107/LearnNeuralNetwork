# Project 2: Diabetes
## Source
- [Source code and training data](https://blog.csdn.net/ccaoshangfei/article/details/126074300)
## Modification: classifyDiabetes.py
- I modified the epoch from 100 to 40.
- The original author's loss-epoch graph: Loss converges at epoch 20.
  ![original loss-epoch](https://github.com/user-attachments/assets/335c23e1-6c6f-4d0e-a2e6-a551928ca0d9)
- My graph, with 40 epochs.  
  ![40 loss-epoch](https://github.com/user-attachments/assets/2c8a0e51-0340-4882-9654-c9437ab74313)
## Modification: diabetes_randomInit.ph
- This modified version is not for training model, but for showing the random initial loss caused by the random initialization of parameters. In this model, sometimes the randomly initialized model gives a loss similar to the trained model.
### A test result:
- Graph: ![random initial error](https://github.com/user-attachments/assets/a8bbd1ce-a88d-4596-8dde-7f9a6535629c)
- Terminal:
```
Show Parameter's Random Initialization
Round 1
Parameter containing:
tensor([[ 0.2009,  0.0345,  0.2937, -0.1641,  0.0706, -0.2745,  0.0318, -0.3441],
        [-0.3221, -0.1946, -0.2920, -0.2506,  0.2422, -0.3274, -0.0965, -0.1650],
        [-0.1536, -0.0487, -0.1558, -0.0983, -0.0164, -0.0664, -0.2347, -0.0478],
        [-0.1312,  0.0359, -0.2100, -0.1494,  0.2755,  0.0282, -0.0289, -0.3229],
        [ 0.2766,  0.0021,  0.0578,  0.1799, -0.0700,  0.2410,  0.3473, -0.0070],
        [ 0.0169,  0.2802,  0.3388,  0.1279, -0.2068, -0.1251,  0.0887, -0.1595]],
       requires_grad=True)
Parameter containing:
tensor([[ 0.3620, -0.1583,  0.1995,  0.2050,  0.2203,  0.2955],
        [ 0.2364, -0.0526,  0.0586, -0.0217, -0.3272, -0.2841],
        [-0.1753,  0.1883,  0.1719, -0.0334, -0.0789, -0.0345],
        [ 0.3003,  0.3036,  0.0903, -0.0608,  0.3363,  0.2814]],
       requires_grad=True)
Parameter containing:
tensor([[ 0.2693,  0.3661, -0.0678, -0.0048],
        [-0.3960,  0.4461,  0.2683,  0.4406]], requires_grad=True)
Parameter containing:
tensor([[0.1758, 0.0466]], requires_grad=True)

Round 2
Parameter containing:
tensor([[ 0.1900, -0.1456,  0.1419, -0.2514,  0.3240, -0.3357,  0.1760, -0.3347],
        [ 0.2859,  0.0710,  0.2709, -0.2377,  0.1569,  0.2660,  0.0719, -0.2282],
        [-0.2539,  0.2624, -0.3011,  0.2172,  0.2196,  0.0332, -0.2903, -0.1164],
        [ 0.1511,  0.3351, -0.1761, -0.2346, -0.2426, -0.1748,  0.1261,  0.1778],
        [ 0.1741, -0.3194, -0.3190,  0.1840,  0.0178,  0.0253, -0.2159,  0.0885],
        [-0.2436,  0.1642, -0.3324,  0.0587, -0.2971, -0.0019,  0.2228,  0.2319]],
       requires_grad=True)
Parameter containing:
tensor([[ 0.3848, -0.0484,  0.0971,  0.3062,  0.2466,  0.0770],
        [-0.1845, -0.3553,  0.2696, -0.2811,  0.2133, -0.4051],
        [-0.0793,  0.3303, -0.0892, -0.1325, -0.3616,  0.4081],
        [-0.1729,  0.3449, -0.4002,  0.3561,  0.0405,  0.3655]],
       requires_grad=True)
Parameter containing:
tensor([[-0.0922,  0.1920,  0.2785, -0.1468],
        [ 0.1213, -0.0567, -0.2072,  0.4977]], requires_grad=True)
Parameter containing:
tensor([[0.6235, 0.3612]], requires_grad=True)

Test[1],loss:0.647814
Test[2],loss:0.649400
Test[3],loss:0.654445
Test[4],loss:0.648070
Test[5],loss:0.685041
Test[6],loss:0.650617
Test[7],loss:0.967742
Test[8],loss:0.643529
Test[9],loss:0.755603
Test[10],loss:0.818530
```
