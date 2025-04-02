import argparse
import torch
from model import MnistConvNet, CifarConvNet, train, test
from attacks import gradient_attack
import torchvision
import torchvision.transforms as transforms
from config import device, CIFAR10_CLASSES, model_hyperparameter, mnist_attacks, cifar_attacks
import torch.nn.functional as F

def get_mnist_loaders(batch_size):
    """
    MNIST 데이터셋을 위한 DataLoader 생성
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def get_cifar10_loaders(batch_size):
    """
    cifar-10 데이터셋을 위한 DataLoader 생성
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def run_attack(test_loader, model, sample_idx, eps,
                eps_step=None, k=1, target=None, targeted=False, class_names=None):
    """
    공격을 수행하는 함수
    Args:
        test_loader: 테스트 데이터 로더
        model: 학습된 모델
        sample_idx: 공격할 샘플의 인덱스
        eps: 전체 허용 perturbation 크기
        eps_step: 한 스텝당 perturbation 크기 (없으면 eps와 동일)
        k: 공격 반복 횟수 (k=1이면 FGSM, 그 이상이면 PGD)
        target: 목표 레이블 (targeted attack일 때)
        targeted: True면 targeted attack, False면 untargeted attack
        class_names: 클래스 이름 리스트 (None이면 인덱스 사용)
    """
    sample_image, sample_label = next(iter(test_loader))
    sample_image = sample_image[sample_idx].unsqueeze(0).to(device)
    label = sample_label[sample_idx].item()

    y = torch.tensor([target]).to(device) if targeted else torch.tensor([label]).to(device)
    x_adv = gradient_attack(model=model, x=sample_image, label_or_target=y,
                      eps=eps, eps_step=eps_step, k=k, targeted=targeted)

    pred = model(x_adv).argmax(1).item()
    prob = F.softmax(model(x_adv), dim=1)
    topk = prob[0].topk(3)
    
    print_attack_result(label, pred, target, topk,
                        class_names=class_names, targeted=targeted, k=k)

def print_attack_result(label, pred, target, topk, class_names=None, targeted=False, k=1):
    """
    공격 결과를 출력하는 함수
    Args:
        label: 원본 레이블
        pred: 공격 후 예측 레이블
        target: 목표 레이블 (targeted attack일 때)
        topk: top-k 예측 결과
        class_names: 클래스 이름 리스트 (None이면 인덱스 사용)
        targeted: True면 targeted attack, False면 untargeted attack
        k: 공격 반복 횟수 (k=1이면 FGSM, 그 이상이면 PGD)
    """
    values, indices = topk.values, topk.indices
    attack_type = "targeted" if targeted else "untargeted"
    attack_name = "FGSM" if (k is None or k == 1) else "PGD"

    true_label_str = class_names[label] if class_names else str(label)
    pred_str = class_names[pred] if class_names else str(pred)
    target_str = class_names[target] if class_names and target is not None else str(target)

    print(f"\n{attack_type.upper()} {attack_name} Attack")
    if targeted:
        print(f"The target is {target_str}")
    print(f"Original label: {true_label_str}, attack → Prediction: {pred_str}")
    print("Top-3 Predictions:")
    for i in range(3):
        idx = indices[i].item()
        class_str = class_names[idx] if class_names else str(idx)
        print(f"  {i+1}) Class {class_str} with probability {values[i].item():.4f}")
    print("=" * 40)

def main(dataset):
    num_classes = model_hyperparameter["num_classes"]
    batch_size = model_hyperparameter["batch_size"]
    lr = model_hyperparameter["learning_rate"]
    num_epochs = model_hyperparameter["num_epochs"]
    criterion = torch.nn.CrossEntropyLoss()

    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size)
        class_names = [str(i) for i in range(num_classes)]
        model = MnistConvNet(num_classes).to(device)

    elif dataset == 'cifar':
        train_loader, test_loader = get_cifar10_loaders(batch_size)
        class_names = CIFAR10_CLASSES
        model = CifarConvNet(num_classes).to(device)  

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("\n Training the model...")
    train(model, train_loader, criterion, optimizer, num_epochs)
    test(model, test_loader)

    attack_scenarios = mnist_attacks if dataset == 'mnist' else cifar_attacks
    for attack in attack_scenarios:
        run_attack(
            test_loader=test_loader,
            model=model,
            sample_idx=attack['sample_idx'],
            eps=attack['eps'],
            eps_step=attack.get('eps_step'),
            k=attack.get('k'),
            target=attack.get('target'),
            targeted=attack['targeted'],
            class_names=class_names
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='select dataset')
    parser.add_argument('--dataset', type=str, default='mnist', help='mnist or cifar')
    args = parser.parse_args()
    main(args.dataset)
