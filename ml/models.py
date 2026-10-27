"""Modele: baseline CNN od zera oraz MobileNetV3 z transfer learningiem."""
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class BaselineCNN(nn.Module):
    """Mała sieć splotowa uczona od zera - punkt odniesienia dla transfer learningu."""

    def __init__(self, num_classes):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256), block(256, 256))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_mobilenet(num_classes, pretrained=True):
    """MobileNetV3-Large z wagami ImageNet; ostatnia warstwa podmieniona na nasze klasy."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    # classifier = [Linear(960,1280), Hardswish, Dropout, Linear(1280,1000)]
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def build_model(name, num_classes, pretrained=True):
    if name == "baseline":
        return BaselineCNN(num_classes)
    if name.startswith("mobilenet"):
        return build_mobilenet(num_classes, pretrained)
    raise ValueError(f"Nieznany model: {name}")


def set_backbone_trainable(model, unfreeze_from=None):
    """None = cały backbone zamrożony; k = trenowalne bloki features[k:]."""
    for i, layer in enumerate(model.features):
        trainable = unfreeze_from is not None and i >= unfreeze_from
        for p in layer.parameters():
            p.requires_grad = trainable


def freeze_batchnorm(model):
    """BatchNorm w trybie eval: zachowuje statystyki z ImageNet przy małym zbiorze."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
