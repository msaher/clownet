"""Model definition."""

from torch import nn
import torchvision
import torch
import numpy as np
from mobileformer.mobile_former import mobile_former_26m
import torchvision
from pycoviar.transforms import GroupRandomHorizontalFlip
from pycoviar.transforms import GroupMultiScaleCrop

# loading the mobile former pretrained net for embedding images
mobile = mobile_former_26m(pretrained=False)

state = torch.load("./mobile-former-26m.pth.tar", map_location=torch.device('cuda'))['state_dict']

mobile.load_state_dict(state)
mobile.to(torch.device('cuda'))


# freezing the params
# for param in mobile.parameters():
#     param.requires_grad = False

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Clownet(nn.Module):
    def __init__(self, num_class, num_segments, representation, base_model='mobile'):
        super(Clownet).__init__()
        self._representation = representation
        self.num_segments = num_segments

#         print(("""
# Initializing model:
#     base model:         {}.
#     input_representation:     {}.
#     num_class:          {}.
#     num_segments:       {}.
#         """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_tsn(self, num_class):

        feature_dim: int = self.base_model.classifier.classifier[1].in_features
        self.fc_final = nn.Linear(feature_dim, num_class)

        if self._representation == 'mv':
            self.data_bn = nn.BatchNorm2d(2) # changed this 2 --> 3
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)

    def unfreeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def _prepare_base_model(self, base_model):

        if 'mobile' in base_model:
            # the mobileformer is pretrained, so its okay if all the Clownet
            # objects share the same instance becuase its immutable
            self.base_model = mobile

            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):
        # saher: this merges the batch_size with segment number
        # the result is
        # (batch_size*num_of_segments, channel, width, height)
        input = input.view((-1, ) + input.size()[-3:])

        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        base_out = self.base_model(input)
        out = self.fc_final(base_out)

        return out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    # this is kinda silly, but lets keep it
    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        # print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))])
