# coding: UTF-8
import torch
import torch.nn.functional as F

from commons.utils import clamp, upper_limit, lower_limit


class AdvCore(object):
    def __init__(self, config, model, optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        pass

    def trn_baseline(self, labels, trains):
        outputs = self.model(trains)
        self.model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss, outputs

    def trn_pgd(self, labels, trains):
        delta = torch.zeros_like(trains)
        if self.config.delta_init == 'random':
            delta.uniform_(-self.config.epsilon, self.config.epsilon)
            delta.data = clamp(delta, lower_limit - trains, upper_limit - trains)
        delta.requires_grad = True
        for _ in range(self.config.attack_iters):
            output = self.model(trains + delta)
            loss = F.cross_entropy(output, labels)
            loss.backward(retain_graph=True)
            grad = delta.grad
            delta.data = clamp(delta + self.config.alpha * torch.sign(grad), torch.tensor(-self.config.epsilon),
                               torch.tensor(self.config.epsilon))
            delta.data = clamp(delta, lower_limit - trains, upper_limit - trains)
            delta.grad.zero_()
        delta = delta.detach()
        outputs = self.model(trains + delta)
        loss = F.cross_entropy(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, outputs

    def trn_free(self, labels, trains):
        delta = torch.zeros_like(trains)
        delta.requires_grad = True
        for _ in range(self.config.minibatch_replays):
            output = self.model(trains + delta[:trains.size(0)])
            loss = F.cross_entropy(output, labels)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grad = delta.grad.detach()
            delta.data = clamp(delta + self.config.epsilon * torch.sign(grad), torch.tensor(-self.config.epsilon),
                               torch.tensor(self.config.epsilon))
            delta.data[:trains.size(0)] = clamp(delta[:trains.size(0)], lower_limit - trains, upper_limit - trains)
            self.optimizer.step()
            delta.grad.zero_()
        return loss, output

    def trn_fgsm(self, labels, trains):
        if self.config.delta_init == 'zero':
            delta = torch.zeros_like(trains)
        if self.config.delta_init == 'random':
            delta.uniform_(-self.config.epsilon, self.config.epsilon)
            delta.data = clamp(delta, lower_limit - trains, upper_limit - trains)
        delta.requires_grad = True

        output = self.model(trains + delta[:trains.size(0)])
        loss = F.cross_entropy(output, labels)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        grad = delta.grad.detach()
        delta.data = clamp(delta + self.config.alpha * torch.sign(grad), torch.tensor(-self.config.epsilon),
                           torch.tensor(self.config.epsilon))
        delta.data[:trains.size(0)] = clamp(delta[:trains.size(0)], lower_limit - trains, upper_limit - trains)
        delta = delta.detach()
        output = self.model(trains + delta[:trains.size(0)])
        loss = F.cross_entropy(output, labels)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss, output

    def trn_router(self, labels, trains):
        trns = self.model.embedding(trains[0])
        if self.config.sgdflag == "fgsm":
            loss, outputs = self.trn_fgsm(labels, trns)
        elif self.config.sgdflag == "pgd":
            loss, outputs = self.trn_pgd(labels, trns)
        elif self.config.sgdflag == "free":
            loss, outputs = self.trn_free(labels, trns)
        else:
            loss, outputs = self.trn_baseline(labels, trns)
        return loss, outputs
