import torch
import torch.nn as nn
import math
import torch.nn.init as init


class RetentionFault():
    def __init__(self, weights: torch.tensor(0) = None, delta1: int = None, delta2: int = None):
        self.delta1 = delta1
        self.delta2 = delta2
        self.weights = weights

    # Create the switching probability of the array with _shape_ for a given time _t_ and the thermal stability _delta_
    # equation 2 in paper
    @staticmethod
    def prepare_retention_probabilities(shape, t, delta):
        result = torch.full(shape, -delta, requires_grad=False, dtype=torch.float64, device="cuda")
        result.exp_()
        result.mul_(-t / 0.000000001)
        result.exp_()
        result.mul_(-1)
        result.add_(1)
        return result

    # Use an array with the switching probability of each cell to determine the cells that actually switch
    def prepare_delta_rnd_mask(self, retention_time_mask):
        """
        distributions.uniform.Uniform Generates uniformly distributed random samples from the half-open
        interval [low, high).
        """
        randgen = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        mask = randgen.sample(retention_time_mask.shape).cuda().to(
            torch.float64)  # Full fault mask for one distribution
        mask = torch.reshape(mask,
                             (retention_time_mask.shape[0], -1))  # reshape to the same row size as retentionTimeMask
        """
        torch.le(input, other, out=None) → Tensor
        Computes input <= other element wise
        torch.le_ is the in-plece version of the torch.le
        """
        mask = mask.le(retention_time_mask).to(torch.float64).add(0.5)
        # mask = mask + 0.5
        # m1 = mask
        # mask.le_(retention_time_mask).to(torch.float64).add_(0.5)
        # print('Diff le and le_ ------------ ', m.sum()-m1.sum())
        # compares uniformly distributed mask with retention_time_prob matrix then add 0.5

        return mask

    def inject_fault_with_delta_mask(self, model, retention_time_mask, layer, delta):
        """
        get NN model and retention probability matrix, return model with faulty weights
        """

        weights1 = model.classifier[layer].weight
        retention_time_mask = self.prepare_retention_probabilities(weights1.shape, retention_time_mask, delta)

        mask1 = self.prepare_delta_rnd_mask(retention_time_mask).to(torch.float32)

        with torch.no_grad():
            weights1 = weights1.sub(mask1).sign()  # .sub(.5).sign()
        model.classifier[layer].weight = nn.Parameter(weights1)  ############
        return model

    def inject_fault_with_delta_mask_all_layer_mlp(self, model, t, d1):
        """
        get NN model and retention probability matrix, return model with faulty weights
        """

        weights0 = model.classifier[0].weight
        weights1 = model.classifier[3].weight
        weights2 = model.classifier[6].weight
        weights3 = model.classifier[9].weight
        weights4 = model.classifier[12].weight
        weights5 = model.classifier[15].weight


        retention_time_mask0 = self.prepare_retention_probabilities(weights0.shape, t, 60)
        retention_time_mask1 = self.prepare_retention_probabilities(weights1.shape, t, d1)
        retention_time_mask2 = self.prepare_retention_probabilities(weights2.shape, t, d1)
        retention_time_mask3 = self.prepare_retention_probabilities(weights3.shape, t, d1)
        retention_time_mask4 = self.prepare_retention_probabilities(weights4.shape, t, d1)
        retention_time_mask5 = self.prepare_retention_probabilities(weights5.shape, t, 60)

        mask0 = self.prepare_delta_rnd_mask(retention_time_mask0).to(torch.float32)
        mask1 = self.prepare_delta_rnd_mask(retention_time_mask1).to(torch.float32)
        mask2 = self.prepare_delta_rnd_mask(retention_time_mask2).to(torch.float32)
        mask3 = self.prepare_delta_rnd_mask(retention_time_mask3).to(torch.float32)
        mask4 = self.prepare_delta_rnd_mask(retention_time_mask4).to(torch.float32)
        mask5 = self.prepare_delta_rnd_mask(retention_time_mask5).to(torch.float32)

        with torch.no_grad():
            weights0 = weights0.sub(mask0).sign()  # .sub(.5).sign()
            weights1 = weights1.sub(mask1).sign()  # .sub(.5).sign()
            weights2 = weights2.sub(mask2).sign()  # .sub(.5).sign()
            weights3 = weights3.sub(mask3).sign()  # .sub(.5).sign()
            weights4 = weights4.sub(mask4).sign()  # .sub(.5).sign()
            weights5 = weights5.sub(mask5).sign()  # .sub(.5).sign()

        model.classifier[0].weight = nn.Parameter(weights0)  ############
        model.classifier[3].weight = nn.Parameter(weights1)  ############
        model.classifier[6].weight = nn.Parameter(weights2)  ############
        model.classifier[9].weight = nn.Parameter(weights3)  ############
        model.classifier[12].weight = nn.Parameter(weights4)  ############
        model.classifier[15].weight = nn.Parameter(weights5)  ############
        return model

    @staticmethod
    def scrub_index(rows, cols, raise_):
        row = []
        col = []
        for i in range(cols + 1):
            for j in range(raise_ * i):
                if j < rows:
                    row.append(j)
                    col.append(i - 1)
                else:
                    return [row, col]
        return [row, col]

    @staticmethod
    def scrub_fully_connected(model, layer, diag):
        weight_m = model.classifier[layer].weight.data

        with torch.no_grad():
            weight_m = weight_m.tril(diag).add(0.5).sign()

        model.classifier[layer].weight = nn.Parameter(weight_m)

        return model

    @staticmethod
    def scrub_hidden_fully_connected(model, diag):
        weight_m1 = model.classifier[3].weight.data
        weight_m2 = model.classifier[6].weight.data
        weight_m3 = model.classifier[9].weight.data
        weight_m4 = model.classifier[12].weight.data


        with torch.no_grad():
            weight_m1 = weight_m1.tril(diag).add(0.5).sign()
            weight_m2 = weight_m2.tril(diag).add(0.5).sign()
            weight_m3 = weight_m3.tril(diag).add(0.5).sign()
            weight_m4 = weight_m4.tril(diag).add(0.5).sign()

        model.classifier[3].weight = nn.Parameter(weight_m1)
        model.classifier[6].weight = nn.Parameter(weight_m2)
        model.classifier[9].weight = nn.Parameter(weight_m3)
        model.classifier[12].weight = nn.Parameter(weight_m4)

        return model

    def scrub_conv(self, model, layer, d1):
        weight_m = model.features[layer].weight.data
        conv_shape = weight_m.shape

        scrub_row, scrub_col = self.scrub_index(rows=conv_shape[1] * conv_shape[2] * conv_shape[3], cols=conv_shape[0], raise_=d1)

        with torch.no_grad():
            weight_m[scrub_row, scrub_col] = 1.

        model.fc1.weight = nn.Parameter(weight_m)

        return model
