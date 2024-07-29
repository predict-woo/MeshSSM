import torch
from torch import nn
import torch.nn.functional as F


def conv3(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(
            576, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 128, 3)
        self.layer2 = self._make_layer(block, 192, 4)
        self.layer3 = self._make_layer(block, 256, 6)
        self.layer4 = self._make_layer(block, 384, 3)
        self.fc = nn.Linear(384, 1152)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks):
        downsample = None
        if self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.permute(x, (0, 2, 1))
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas


class MeshDecoder(nn.Module):
    def __init__(self):
        super(MeshDecoder, self).__init__()
        self.resnet = ResNet(
            block=BasicBlock,
        )

    def forward(self, x):
        return self.resnet(x)
