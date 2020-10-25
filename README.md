# Pytorch Implementation of Stable Dynamic Flows (ImitationFlows)
This library provides the models and the learning algorithms for learning deep stochastic stable dynamics by Invertible Flows.
The model is composed by a latent stochastic stable dynamic system and an invertible flow. See [1]

The models and the learning algorithms are implemented in PyTorch.

## Installation
Inside the repository,

```
pip install -r requirements.txt
```

```
pip install -e .
```

## Examples
Examples are placed in the [`examples`](./examples) directory.

<img width="250" align="middle" src="https://github.com/TheCamusean/iflow/blob/main/.assets/rshape.gif">

You can run examples in

#### Toy Dataset
Limit Cycle with  IROS dataset [1] (results are saved in examples/experiments)

```
python examples/train_iros.py 
```

Goto Motions with LASA dataset [2]
```
python examples/train_lasa.py 
```

#### Real Robot Dataset
Goto Motions with pouring dataset
```
python examples/train_pouring.py 
```
Limit Cycle Motions with drumming dataset
```
python examples/train_pouring.py 
```


## Basics

Stable Dynamic Flows, named ImitationFlows in [1], represents a family of neural network architectures, 
which combines a latent stable dynamic system and an invertible neural network(Normalizing Flows).

You can find the set of stable dynamic system models in  [`dynamics`](./iflow/model/dynamics).

For the invertible networks, we have used RealNVP layer [3], Neural Spline Flows [4] and 
Neural ODE [5]. You can find our models in [`flows`](./iflow/model/flows).


### References
[1] Julen Urain, Michele Ginesi, Davide Tateo, Jan Peters. "ImitationFlows: Learning Deep Stable Stochastic 
Dynamic Systems by Normalizing Flows" *IEEE/RSJ International Conference on
Intelligent Robots and Systems.* 2020.[ARXIV]

[2] Khansari-Zadeh, S. Mohammad, and Aude Billard. "Learning stable nonlinear dynamical 
systems with gaussian mixture models." *IEEE Transactions on Robotics* 2011.

[3] Dinh, Laurent, Jascha Sohl-Dickstein, and Samy Bengio. 
"Density estimation using real nvp." *International Conference on Learning Representations* 2016.

[4] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. 
"Neural spline flows." *Advances in Neural Information Processing Systems* 2019.

[5] Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. 
"Neural ordinary differential equations". *In Advances in neural information processing systems* 2018.

---

If you found this library useful in your research, please consider citing
```
@article{urain2020imitationflows,
  title={ImitationFlows: Learning Deep Stable Stochastic Dynamic Systems by Normalizing Flows},
  author={Urain, Julen and Ginesi, Michele and Tateo, Davide and Peters, Jan},
  journal={IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year={2020}
}
```

_Our Flows library has been highly influenced by the amazing repositories_
 
 https://github.com/bayesiains/nsf
 https://github.com/rtqichen/torchdiffeq
