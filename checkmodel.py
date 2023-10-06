# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 15:31:57 2022

@author: tuann
"""
from torchsummary import summary
from models.TickNet import *
#from models.mobilenet_with_FR import *

model = build_TickNet(1000, typesize='large_new',cifar=False)
#model = build_TickNet(100, typesize='small',cifar=True)
#model = build_TickNet(100, typesize='large_large_new',cifar=True)
#model = build_TickNet(1000, typesize='large_new',cifar=False)

#model = build_mobilenet_v2(100, width_multiplier=1.0, cifar=True)
#model = build_mobilenet_v3(120, version = "large", width_multiplier=1.0, cifar=True,use_lightweight_head=False)


model = model.cuda()
print ("model")
print (model)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
#print(model)
#model.cuda()
summary(model, (3, 224, 224))
#summary(model, (3, 32, 32))