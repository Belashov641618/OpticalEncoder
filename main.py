import torch

if __name__ == '__main__':
    from utilities.losses import Normalizable, Normalization, LossLinearCombination
    mse = Normalizable.MeanSquareError(Normalization.Minmax())
    ce = Normalizable.CrossEntropy(Normalization.Softmax())
    loss = LossLinearCombination(mse, ce)
    loss.proportions(0.8, None)
    value = loss(torch.tensor([[0., 1., 0., 0.]]), torch.tensor([[0., 1., 0., 0.]]))
    print(mse(torch.tensor([[0., 1., 0., 0.]]), torch.tensor([[0., 1., 0., 0.]])))
    print(ce(torch.tensor([[0., 1., 0., 0.]]), torch.tensor([[0., 1., 0., 0.]])))
    print(loss.coefficients, value)






