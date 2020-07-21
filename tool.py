import models

# todo new sample method by YW
class Sampler(object):
    def __init__(self, data: list, num: int, prototypes: models.Prototypes, net: models.DenseNet, is_novel: bool):
        self.prototypes = prototypes
        self.data = data
        self.num = num
        self.net = net
        self.selected = []
        self.is_novel = is_novel

    def return_data(self):
        return self.selected

    def sampling(self):
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
        return prototype.weight / distance

    def noval_metric(self, d):
        feature, _ = d
        feature = feature.to(self.net.device).unsqueeze(0)
        feature, _ = self.net(feature)
        prototype, score = self.prototypes.closest(feature)
        return score
