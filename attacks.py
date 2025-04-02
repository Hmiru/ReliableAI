import torch
import torch.nn.functional as F
from config import device

def gradient_attack(model, x, label_or_target, eps, eps_step=None, k=1, targeted=False):
    """
    FGSM / PGD 공격 함수 (k=1이면 FGSM, k>1이면 PGD)

    Args:
        model: 학습된 모델
        x: 입력 이미지 (1 x C x H x W)
        label_or_target: 정답 레이블 (untargeted) 또는 목표 레이블 (targeted)
        eps: 전체 허용 perturbation 크기
        eps_step: 한 스텝당 perturbation 크기 (없으면 eps와 동일)
        k: 공격 반복 횟수 (k=1이면 FGSM, 그 이상이면 PGD)
        targeted: True면 targeted attack, False면 untargeted attack

    Returns:
        x_adv: 생성된 적대적 샘플
    """
    # 원본 이미지 보존
    x_orig = x.clone().detach().to(device)
    x_adv = x_orig.clone().detach()
    label_or_target = label_or_target.to(x.device)

    # 스텝 크기 지정 (FGSM이면 한 번에 eps만큼 진행)
    if eps_step is None:
        eps_step = eps

    for _ in range(k):
        x_adv.requires_grad = True
        # 모델 예측 및 손실 계산
        output = model(x_adv)
        loss = F.cross_entropy(output, label_or_target)
        # 기울기 계산
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data

        # 입력 업데이트 (targeted: gradient 반대 방향, untargeted: 같은 방향)
        if targeted:
            x_adv = x_adv - eps_step * grad.sign()
        else:
            x_adv = x_adv + eps_step * grad.sign()

        # 원본 기준으로 epsilon 안에서 clip
        eta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = x_orig + eta

        # 이미지 값은 항상 [0, 1] 사이로 clip
        x_adv = torch.clamp(x_adv, 0, 1).detach()

    return x_adv
