# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches, split_meta_train_test, l2_between_dicts
from domainbed.optimizers import get_optimizer

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)

from domainbed.gam import GAM, ProportionScheduler, smooth_crossentropy

from domainbed.sagm import SAGM, LinearScheduler, perturb_gap

from domainbed.iada import IADA, LinearScheduler, perturb_gap, IADA_FINAL

from backpack import extend, backpack
from backpack.extensions import Variance, BatchGrad
# from domainbed.iada_final import , LinearScheduler, perturb_gap

import torch.distributed as dist
import torch.multiprocessing as mp

def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches

def flat_grad(grad_tuple):
    return torch.cat([p.view(-1) for p in grad_tuple]).unsqueeze(0)

class Algorithm(torch.nn.Module):
    """C
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams
        self.args = args

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],)
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ITTA(Algorithm):
    """
    Improved Test-Time Adaptation (ITTA)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(ITTA, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.featurizer = networks.ResNet_ITTA(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.test_mapping = networks.MappingNetwork()  # specialized for resnet18
        self.test_optimizer = torch.optim.Adam(self.test_mapping.parameters(), lr=self.hparams["lr"] * 0.1)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            [{'params': self.featurizer.parameters()},{'params': self.classifier.parameters()}],
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.MSEloss = nn.MSELoss()
        self.adaparams = networks.Adaparams()  # specialized for resnet18
        self.adaparams_optimizer = torch.optim.Adam(self.adaparams.parameters(), lr=self.hparams["lr"] * 0.1)

    def _get_grads(self, loss):
        self.optimizer.zero_grad()
        loss.backward(inputs=list(self.featurizer.parameters()),
                      retain_graph=True, create_graph=True)
        dict = OrderedDict(
            [
                (name, weights.grad.clone().view(weights.grad.size(0), -1))
                for name, weights in self.featurizer.named_parameters()
            ]
        )
        return dict

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        ############################# this is for network update
        #############################
        z_ori, z_aug = self.featurizer(all_x)
        z_ori, z_aug = self.featurizer.fea2(z_ori, z_aug)
        z_ori, z_aug = self.featurizer.fea_forward(z_ori), self.featurizer.fea_forward(z_aug)
        loss_reg = self.MSEloss(self.adaparams(z_aug - z_ori), torch.zeros_like(z_aug))
        loss_cla = F.cross_entropy(self.classifier(z_ori), all_y) + \
                   F.cross_entropy(self.classifier(z_aug), all_y)
        loss = loss_reg + loss_cla
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ############################# this is for adaparams update
        #############################
        z_ori, z_aug = self.featurizer(all_x)
        z_ori, z_aug = self.featurizer.fea2(z_ori, z_aug)
        z_ori, z_aug = self.featurizer.fea_forward(z_ori), self.featurizer.fea_forward(z_aug)
        loss_reg = self.MSEloss(self.adaparams(z_aug - z_ori), torch.zeros_like(z_aug))
        loss_cla = F.cross_entropy(self.classifier(z_ori), all_y) + \
                   F.cross_entropy(self.classifier(z_aug), all_y)
        dict_reg = self._get_grads(loss_reg)
        dict_cla = self._get_grads(loss_cla)
        penalty = l2_between_dicts(dict_reg, dict_cla, normalize=True) * 0.1
        self.adaparams_optimizer.zero_grad()
        penalty.backward(inputs=list(self.adaparams.parameters()))
        self.adaparams_optimizer.step()

        return {'loss': loss_cla.item(), 'reg': loss_reg.item()}

    def test_adapt(self, x):
        z_ori, z_aug = self.featurizer(x)
        z_ori, z_aug = self.test_mapping.fea1(z_ori), self.test_mapping.fea1(z_aug)
        z_ori, z_aug = self.featurizer.fea2(z_ori, z_aug)
        z_ori, z_aug = self.test_mapping.fea2(z_ori), self.test_mapping.fea2(z_aug)
        z_ori, z_aug = self.featurizer.fea3(z_ori), self.featurizer.fea3(z_aug)
        z_ori, z_aug = self.test_mapping.fea3(z_ori), self.test_mapping.fea3(z_aug)
        z_ori, z_aug = self.featurizer.fea4(z_ori), self.featurizer.fea4(z_aug)
        z_ori, z_aug = self.test_mapping.fea4(z_ori), self.test_mapping.fea4(z_aug)
        z_ori, z_aug = self.featurizer.flat(z_ori), self.featurizer.flat(z_aug)
        ########## small lr for large datasets
        loss_reg = self.MSEloss(self.adaparams(z_aug - z_ori), torch.zeros_like(z_ori)) * self.hparams['ada_lr']
        self.test_optimizer.zero_grad()
        loss_reg.backward(inputs=list(self.test_mapping.parameters()))
        self.test_optimizer.step()

    def predict(self, x):
        z_ori, z_aug = self.featurizer(x)
        z_ori = self.test_mapping.fea1(z_ori)
        z_ori, z_aug = self.featurizer.fea2(z_ori, z_aug)
        z_ori = self.test_mapping.fea2(z_ori)
        z_ori = self.featurizer.fea3(z_ori)
        z_ori = self.test_mapping.fea3(z_ori)
        z_ori = self.featurizer.fea4(z_ori)
        z_ori = self.test_mapping.fea4(z_ori)
        z_ori = self.featurizer.flat(z_ori)
        return self.classifier(z_ori)


class RIDG(Algorithm):
    """
    Rational Invariance for Domain Generalization (RIDG)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(RIDG, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.num_classes = num_classes
        self.rational_bank = torch.zeros(num_classes, num_classes, self.featurizer.n_outputs, device='cuda')
        self.init = torch.ones(num_classes, device='cuda')
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x,y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        features = self.featurizer(all_x)
        logits = self.predict(all_x)
        rational = torch.zeros(self.num_classes, all_x.shape[0], self.featurizer.n_outputs, device='cuda')
        for i in range(self.num_classes):
            rational[i] = (self.classifier.weight[i] * features)

        classes = torch.unique(all_y)
        loss_rational = 0
        for i in range(classes.shape[0]):
            rational_mean = rational[:, all_y==classes[i]].mean(dim=1)
            if self.init[classes[i]]:
                self.rational_bank[classes[i]] = rational_mean
                self.init[classes[i]] = False
            else:
                self.rational_bank[classes[i]] = (1 - self.hparams['momentum']) * self.rational_bank[classes[i]] + \
                            self.hparams['momentum'] * rational_mean
            loss_rational += ((rational[:, all_y==classes[i]] - (self.rational_bank[classes[i]].unsqueeze(1)).detach())**2).sum(dim=2).mean()
        loss = F.cross_entropy(logits, all_y)+ self.hparams['ridg_reg'] * loss_rational

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class IADA_DG_FINAL(Algorithm):
    """
    Inconsistency-Aware Domain Augmentation
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super().__init__(input_shape, num_classes, num_domains, hparams,args)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(nn.Linear(self.featurizer.n_outputs, num_classes))
        self.network = (nn.Sequential(self.featurizer, self.classifier))
        self.network = nn.DataParallel(self.network)

        if args.use_dpp:
            self.network = nn.parallel.DistributedDataParallel(self.network)


        self.args = args
        self.args.num_domains = num_domains
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        print('Model is loaded')
        checkpoint = torch.load(args.out_dir / 'eval_best.pt')

        new_checkpoint = OrderedDict()
        for k,v in checkpoint['model_state_dict'].items():
            name = 'module.'+k
            new_checkpoint[name]=v

        self.network.load_state_dict(new_checkpoint)
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.lr_scheduler = LinearScheduler(T_max=5000, max_value=self.hparams["lr"],min_value=self.hparams["lr"], optimizer=self.optimizer)
        self.rho_scheduler = LinearScheduler(T_max=5000, max_value=0.05,
                                         min_value=0.05)
        self.syn_lr = 0.005

        self.IADA_optimizer = IADA_FINAL(params=self.network.parameters(), base_optimizer=self.optimizer, model=self.network, rho_scheduler=self.rho_scheduler,hparams=self.hparams,args=self.args,classifier=self.classifier, adaptive=False)
        self.perturb_gap = perturb_gap(params=self.network.parameters(), base_optimizer=self.optimizer, model=self.network,
                                   alpha=self.hparams["alpha"], rho_scheduler=self.rho_scheduler, adaptive=False)

    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def sam_update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        loss = F.cross_entropy(self.predict(all_x), all_y)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        '''Epsilon reflected gradient'''
        eps = [g * scale for g in grad_w]

        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)
        with open("sam_0.txt", "a") as f:
            f.write(str(loss.item()))
            f.write("\n")

        self.optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}

    def iada_update(self, x,y, cur_step):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        syn_x = all_x.clone().detach()
        syn_x.requires_grad_(True)

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.IADA_optimizer.set_closure(loss_fn, all_x, all_y, syn_x)
        predictions, loss, inconsistency_loss = self.IADA_optimizer.step(cur_step)
        self.lr_scheduler.step()
        self.IADA_optimizer.update_rho_t()
        return {"loss": loss.item(), "inconsistency_loss": inconsistency_loss.item()}


    def update(self, x, y, **kwargs):

        if kwargs['step'] <= kwargs['n_steps'] // self.args.sam_warm_up_ratio:
            return self.sam_update(x,y)
        else:
            return self.iada_update(x,y, kwargs['step'])

    def predict(self, x):
        return self.network(x)

    def feat(self, x):
        return self.featurizer(x)

    def classify(self,x):
        return self.classifier(x)

    def get_perturbed_loss(self, x, y, **kwargs):
        all_x = x
        all_y = y

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.perturb_gap.set_closure(loss_fn, all_x, all_y)
        original_loss, perturbed_loss_list = self.perturb_gap.step()


        return {"origin_loss": original_loss,
                "loss_0.01": perturbed_loss_list[0],
                "loss_0.02": perturbed_loss_list[1],
                "loss_0.03": perturbed_loss_list[2],
                "loss_0.04": perturbed_loss_list[3],
                "loss_0.05": perturbed_loss_list[4]
                }

class SAGM_DG(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    # def __init__(self, input_shape, num_classes, num_domains, hparams):
    #     assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
    #     super().__init__(input_shape, num_classes, num_domains, hparams)
    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super().__init__(input_shape, num_classes, num_domains, hparams,args)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.lr_scheduler = LinearScheduler(T_max=5000, max_value=self.hparams["lr"],
                                    min_value=self.hparams["lr"], optimizer=self.optimizer)

        self.rho_scheduler = LinearScheduler(T_max=5000, max_value=0.05,
                                         min_value=0.05)

        self.SAGM_optimizer = SAGM(params=self.network.parameters(), base_optimizer=self.optimizer, model=self.network,
                               alpha=self.hparams["alpha"], rho_scheduler=self.rho_scheduler, adaptive=False)

        self.perturb_gap = perturb_gap(params=self.network.parameters(), base_optimizer=self.optimizer, model=self.network,
                                   alpha=self.hparams["alpha"], rho_scheduler=self.rho_scheduler, adaptive=False)

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.SAGM_optimizer.set_closure(loss_fn, all_x, all_y)
        predictions, loss = self.SAGM_optimizer.step()
        self.lr_scheduler.step()
        self.SAGM_optimizer.update_rho_t()
        return {"loss": loss.item()}


    def get_perturbed_loss(self, x, y, **kwargs):
        # all_x = torch.cat(x)
        # all_y = torch.cat(y)
        all_x = x
        all_y = y

        def loss_fn(predictions, targets):
            return F.cross_entropy(predictions, targets)

        self.perturb_gap.set_closure(loss_fn, all_x, all_y)
        original_loss, perturbed_loss_list = self.perturb_gap.step()

        return {"origin_loss": original_loss,
                "loss_0.01": perturbed_loss_list[0],
                "loss_0.02": perturbed_loss_list[1],
                "loss_0.03": perturbed_loss_list[2],
                "loss_0.04": perturbed_loss_list[3],
                "loss_0.05": perturbed_loss_list[4]
                }


    def predict(self, x):
        return self.network(x)



class Mixstyle(Algorithm):
    """MixStyle w/o domain label (random shuffle)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams,args)
        if hparams["resnet18"]:
            network = resnet18_mixstyle_L234_p0d5_a0d1()
        else:
            # network = resnet50_mixstyle_L234_p0d5_a0d1(postion=["conv2_x", "conv3_x", "conv4_x"])
            network = resnet50_mixstyle_L234_p0d5_a0d1(postion=[])

        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle2(Algorithm):
    """MixStyle w/ domain label"""

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains,hparams, args)
        if hparams["resnet18"]:
            network = resnet18_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle2_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def pair_batches(self, xs, ys):
        xs = [x.chunk(2) for x in xs]
        ys = [y.chunk(2) for y in ys]
        N = len(xs)
        pairs = []
        for i in range(N):
            j = i + 1 if i < (N - 1) else 0
            xi, yi = xs[i][0], ys[i][0]
            xj, yj = xs[j][1], ys[j][1]

            pairs.append(((xi, yi), (xj, yj)))

        return pairs

    def update(self, x, y, **kwargs):
        pairs = self.pair_batches(x, y)
        loss = 0.0

        for (xi, yi), (xj, yj) in pairs:
            #  Mixstyle2:
            #  For the input x, the first half comes from one domain,
            #  while the second half comes from the other domain.
            x2 = torch.cat([xi, xj])
            y2 = torch.cat([yi, yj])
            loss += F.cross_entropy(self.predict(x2), y2)

        loss /= len(pairs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class GAM_DG(Algorithm):
    """
    Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization (GAM)
    Official github: https://github.com/xxgege/GAM/tree/main
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(GAM_DG, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        base_optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=5000)
        grad_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=self.hparams["lr"], min_lr=0.0,
                                                      max_value=0.05, min_value=0.02)
        grad_norm_rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=lr_scheduler, max_lr=self.hparams["lr"],
                                                            min_lr=0.0, max_value=0.05, min_value=0.02)
        self.gam_optimizer = GAM(params=self.network.parameters(), base_optimizer=base_optimizer,
                                 lr_scheduler=lr_scheduler, grad_rho_scheduler=grad_rho_scheduler,
                                 grad_norm_rho_scheduler=grad_norm_rho_scheduler, model=self.network)

    def loss_fn(self, predictions, targets):
        return smooth_crossentropy(predictions, targets, smoothing=0.1).mean()

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        self.gam_optimizer.set_closure(self.loss_fn, all_x, all_y)
        _, loss = self.gam_optimizer.step()
        self.gam_optimizer.update_rho_t()

        # loss = F.cross_entropy(self.predict(all_x), all_y)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)

class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class SAM(ERM):
    """Sharpness-Aware Minimization
    """
    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)
        with open("sam_0.txt", "a") as f:
            f.write(str(loss.item()))
            f.write("\n")

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance,args):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams,args)

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, x, y, **kwargs):
        self.update_count += 1
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        minibatches = to_minibatch(x, y)
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda") for i, (x, y) in enumerate(minibatches)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,args=args
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,args=args
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        penalty_weight = (
            self.hparams["irm_lambda"]
            if self.update_count >= self.hparams["irm_penalty_anneal_iters"]
            else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams,args)

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class OrgMixup(ERM):
    """
    Original Mixup independent with domains
    """

    def update(self, x, y, **kwargs):
        x = torch.cat(x)
        y = torch.cat(y)

        indices = torch.randperm(x.size(0))
        x2 = x[indices]
        y2 = y[indices]

        lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

        x = lam * x + (1 - lam) * x2
        predictions = self.predict(x)

        objective = lam * F.cross_entropy(predictions, y)
        objective += (1 - lam) * F.cross_entropy(predictions, y2)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class CutMix(ERM):
    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def update(self, x, y, **kwargs):
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        x = torch.cat(x)
        y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(x.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            output = self.predict(x)
            objective = F.cross_entropy(output, target_a) * lam + F.cross_entropy(
                output, target_b
            ) * (1.0 - lam)
        else:
            output = self.predict(x)
            objective = F.cross_entropy(output, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        self.register_buffer("q", torch.Tensor())

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams,args)

    def update(self, x, y, **kwargs):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        minibatches = to_minibatch(x, y)
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        # for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches):

            inner_net = copy.deepcopy(self.network)

            inner_opt = get_optimizer(
                self.hparams["optimizer"],
                #  "SGD",
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # 1. Compute supervised loss for meta-train set
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # 2. Compute meta loss for meta-val set
            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}


#  class SOMLDG(MLDG):
#      """Second-order MLDG"""
#      # This commented "update" method back-propagates through the gradients of
#      # the inner update, as suggested in the original MAML paper.  However, this
#      # is twice as expensive as the uncommented "update" method, which does not
#      # compute second-order derivatives, implementing the First-Order MAML
#      # method (FOMAML) described in the original MAML paper.

#      def update(self, x, y, **kwargs):
#          minibatches = to_minibatch(x, y)
#          objective = 0
#          beta = self.hparams["mldg_beta"]
#          inner_iterations = self.hparams.get("inner_iterations", 1)

#          self.optimizer.zero_grad()

#          with higher.innerloop_ctx(
#              self.network, self.optimizer, copy_initial_weights=False
#          ) as (inner_network, inner_optimizer):
#              for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
#                  for inner_iteration in range(inner_iterations):
#                      li = F.cross_entropy(inner_network(xi), yi)
#                      inner_optimizer.step(li)

#                  objective += F.cross_entropy(self.network(xi), yi)
#                  objective += beta * F.cross_entropy(inner_network(xj), yj)

#              objective /= len(minibatches)
#              objective.backward()

#          self.optimizer.step()

#          return {"loss": objective.item()}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian,args):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(
            x1_norm
        )
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True,args=args)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False,args=args)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, num_classes)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer("embeddings", torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams["mtl_ema"]

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = nn.Linear(self.network_f.n_outputs, num_classes)
        # style network
        self.network_s = nn.Linear(self.network_f.n_outputs, num_classes)

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return get_optimizer(
                hparams["optimizer"], p, lr=hparams["lr"], weight_decay=hparams["weight_decay"]
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).cuda()

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, x, y, **kwargs):
        # inputs
        all_x = torch.cat([xi for xi in x])
        # labels
        all_y = torch.cat([yi for yi in y])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

class LTD(Algorithm): # learning to diversify
    """
    Official code:https://github.com/BUserName/Learning_to_diversify
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(LTD, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        if hparams["resnet18"]:
            self.featurizer = networks.resnet18(n_class=num_classes)
        else:
            self.featurizer = networks.resnet50(n_class=num_classes)

        self.convertor = networks.AugNet().cuda()
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.featurizer.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.convertor_opt = torch.optim.SGD(self.convertor.parameters(), lr=0.005)
        self.criterion = nn.CrossEntropyLoss()
        self.con = SupConLoss()
        self.num_classes = num_classes
        self.transform = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def loglikeli(self, mu, logvar, y_samples):
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).mean()  # .sum(dim=1).mean(dim=0)

    def club(self, mu, logvar, y_samples):

        sample_size = y_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # /len(kernel_val)

    def mmd_rbf(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = 0

        if ver == 1:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                loss += kernels[s1, s2] + kernels[t1, t2]
                loss -= kernels[s1, t2] + kernels[s2, t1]
            loss = loss.abs_() / float(batch_size)
        elif ver == 2:
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            loss = torch.mean(XX + YY - XY - YX)
        else:
            raise ValueError('ver == 1 or 2')

        return loss

    def conditional_mmd_rbf(self, source, target, label, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
        loss = 0
        for i in range(num_class):
            source_i = source[label == i]
            target_i = target[label == i]
            loss += self.mmd_rbf(source_i, target_i)
        return loss / num_class

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # Stage 1
        self.optimizer.zero_grad()
        # Aug
        inputs_max = self.transform(torch.sigmoid(self.convertor(all_x)))
        inputs_max = inputs_max * 0.6 + all_x * 0.4
        aug_x = torch.cat([inputs_max, all_x])
        labels = torch.cat([all_y, all_y])
        # forward
        logits, tuple = self.featurizer(aug_x)
        # Maximize MI between z and z_hat
        emb_src = F.normalize(tuple['Embedding'][:all_y.size(0)]).unsqueeze(1)
        emb_aug = F.normalize(tuple['Embedding'][all_y.size(0):]).unsqueeze(1)
        # Likelihood
        mu = tuple['mu'][all_y.size(0):]
        logvar = tuple['logvar'][all_y.size(0):]
        y_samples = tuple['Embedding'][:all_y.size(0)]
        # Total loss & backward
        loss = self.criterion(logits, labels)-self.loglikeli(mu, logvar, y_samples)+self.con(torch.cat([emb_src, emb_aug], dim=1), all_y)
        loss.backward()
        self.optimizer.step()

        # STAGE 2
        inputs_max = self.transform(torch.sigmoid(self.convertor(all_x, estimation=True)))
        inputs_max = inputs_max * 0.6 + all_x * 0.4
        aug_x = torch.cat([inputs_max, all_x])
        # forward with the adapted parameters
        _, tuples = self.featurizer(x=aug_x)
        # Upper bound MI
        mu = tuples['mu'][all_y.size(0):]
        logvar = tuples['logvar'][all_y.size(0):]
        y_samples = tuples['Embedding'][:all_y.size(0)]
        div = self.club(mu, logvar, y_samples)
        # Semantic consistency
        e = tuples['Embedding']
        e1 = e[:all_y.size(0)]
        e2 = e[all_y.size(0):]
        dist = self.conditional_mmd_rbf(e1, e2, all_y, num_class=self.num_classes)
        # Total loss and backward
        self.convertor_opt.zero_grad()
        (dist + 0.1 * div).backward()
        self.convertor_opt.step()

        return {"loss": loss.item()}

    def predict(self, x):
        self.featurizer.eval()
        return self.featurizer(x, train=False)[0]


class UMGUD(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams,args):
        super(UMGUD, self).__init__(input_shape, num_classes, num_domains, hparams,args)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.mse_loss = nn.MSELoss().cuda()

        if hparams["resnet18"]:
            self.network = networks.resnet18_umgud(n_class=num_classes)
        else:
            self.network = networks.resnet50_umgud(n_class=num_classes)

        parameter_theta, parameter_phi = [], []
        for name, param in self.network.named_parameters():
            if 'p_' in name:
                parameter_phi.append(param)
            else:
                parameter_theta.append(param)

        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.optimizer_theta = torch.optim.Adam(parameter_theta,lr=0.0001)
        self.aug_optimizer = torch.optim.SGD(parameter_phi, 1)

    def update(self, x, y, **kwargs):

        all_x = torch.cat(x)
        all_y = torch.cat(y)
        counter_k=0

        if kwargs['step'] > 100:
            for _ in range(30):
                feature, _, _ = self.network(x=all_x, noise=False)
                feature_noisy, out, tuples = self.network(x=all_x, noise=True)
                ce_loss = self.criterion(out, all_y)
                constraint_loss = self.mse_loss(feature, feature_noisy)
                aug_loss = -ce_loss + constraint_loss
                self.aug_optimizer.zero_grad()
                aug_loss.backward()
                self.aug_optimizer.step()
            counter_k=1

        _, out, _ = self.network(x=all_x, noise=False)
        ce_loss = self.criterion(out, all_y)

        if counter_k==0:
            loss = ce_loss
            self.optimizer_theta.zero_grad()
            loss.backward()
            self.optimizer_theta.step()
        else:
            grads = torch.autograd.grad(ce_loss, self.network.parameters())
            scale = self.hparams["lr"]
            eps = [g * scale for g in grads]
            with torch.no_grad():
                for p, v in zip(self.network.parameters(), eps):
                    p.add_(v)

            _, out, _ = self.network(x=all_x, noise=True)
            loss = ce_loss+self.criterion(out,all_y)

            self.optimizer.zero_grad()
            loss.backward(create_graph=True)
            self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)[1]