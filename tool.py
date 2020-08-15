import models
import math
import numpy as np


# todo new sample method by YW
class Sampler(object):
    def __init__(self, data: list, num: int, prototypes: models.Prototypes, net: models.DenseNet, is_novel: bool, soft: bool, use_log: bool):
        self.prototypes = prototypes
        self.data = data
        self.num = num
        self.net = net
        self.selected = []
        self.is_novel = is_novel
        self.soft = soft
        self.use_log = use_log

    def return_data(self):
        return self.selected

    def sampling(self):
        self.selected = []
        scores = []
        metric = self.noval_metric if self.is_novel else self.original_metric

        for i, d in enumerate(self.data):
            scores.append(metric(d))

        for i in range(self.num):
            m = max(scores)
            j = scores.index(m)
            self.selected.append(self.data[j])
            scores.pop(j)
            self.data.pop(j)

    def original_metric(self, d):
        feature, label = d
        feature, label = feature.to(self.net.device).unsqueeze(0), label.item()
        feature, _ = self.net(feature)
        prototype, distance = self.prototypes.closest(feature, label)
        distance = max(0.001, distance)
        score = prototype.weight / distance
        if self.use_log:
            score = max(score * 1000, 1.0001)
            score = math.log2(score)
        return score

    def noval_metric(self, d):
        feature, _ = d
        feature = feature.to(self.net.device).unsqueeze(0)
        feature, _ = self.net(feature)
        prototype, score = self.prototypes.closest(feature)
        if self.use_log:
            score = max(score * 1000, 1.0001)
            score = math.log2(score)
        return score

    def soft_sampling(self):
        self.selected = []
        scores = []
        metric = self.noval_metric if self.is_novel else self.original_metric

        for i, d in enumerate(self.data):
            scores.append(metric(d))

        temp_list = range(len(self.data))
        temp_list = np.array(temp_list)
        scores = np.array(scores)
        scores = scores / scores.sum()
        temp_list = np.random.choice(temp_list, size=self.num, p=scores, replace=False)

        for x in temp_list:
            self.selected.append(self.data[x])

        return
