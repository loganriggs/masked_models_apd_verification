# models/__init__.py
from .resnet import ResNet20
from .masked_models import BinaryMaskedModel

__all__ = ['ResNet20', 'BinaryMaskedModel']
