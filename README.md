# Adversarial Attacks on MNIST and CIFAR-10

This project demonstrates both **targeted** and **untargeted** adversarial attacks (FGSM & PGD) on MNIST and CIFAR-10 datasets using PyTorch.  
Problems 1, 2, and 3 have been implemented independently for each dataset (MNIST and CIFAR-10).
## How to Run
```bash
pip install -r requirements.txt
```
Make sure all dependencies are installed (see `requirements.txt`), then run:


###  Run on CIFAR-10

```bash
python test.py --dataset "cifar"
```

###  Run on MNIST
```bash
python test.py --dataset "mnist"
```
⚠️ _If targeted attacks(FGSM) do not change the prediction, try increasing eps slightly in config.py. This sometimes happens._
##  Structure
```bash
.
├── model.py              # CNN models
├── attacks.py            # FGSM, PGD, gradient_attack
├── test.py               # Main training & attack runner
├── config.py             # Device, hyperparameters, class lists
.
```
## Environments information
```bash
python==3.9.21
torch==2.6.0
torchvision==0.21.0
```