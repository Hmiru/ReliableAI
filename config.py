# config.py
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 데이터셋 별 공격 시나리오
mnist_attacks = [
    dict(name="Targeted FGSM", sample_idx=1, k=1, eps=0.35, target=6, targeted=True),
    dict(name="Untargeted FGSM", sample_idx=14, k=1, eps=0.3, targeted=False),
    dict(name="Targeted PGD", sample_idx=1, k=10, eps=0.35, eps_step=0.025, target=6, targeted=True),
    dict(name="Untargeted PGD", sample_idx=14, k=10, eps=0.3, eps_step=0.025, targeted=False)
]

cifar_attacks = [
    dict(name="Targeted FGSM", sample_idx=1, k=1, eps=0.35, target=0, targeted=True),
    dict(name="Untargeted FGSM", sample_idx=14, k=1, eps=0.25, targeted=False),
    dict(name="Targeted PGD", sample_idx=1, k=10, eps=0.25, eps_step=0.025, target=0, targeted=True),
    dict(name="Untargeted PGD", sample_idx=14, k=10, eps=0.25, eps_step=0.025, targeted=False)
]

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

model_hyperparameter ={
    "num_classes" : 10,
    "learning_rate" : 1e-3,
    "num_epochs" :5,
    "batch_size" : 32
}