import numpy as np
import torch
from torch import nn
import pytorch_utils as ptu
from collections import OrderedDict

from torch.distributions import Distribution as TorchDistribution
from torch.distributions import Normal as TorchNormal
from torch.distributions import Independent as TorchIndependent
from torch.distributions import Bernoulli as TorchBernoulli
from torch.distributions import Categorical, OneHotCategorical, kl_divergence
from torch.nn import functional as F

import abc

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 0.001

class Distribution(TorchDistribution):
    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}

class TorchDistributionWrapper(Distribution):
    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return 'Wrapped ' + self.distribution.__repr__()

class Independent(Distribution, TorchIndependent):
    def get_diagnostics(self):
        return self.base_dist.get_diagnostics()

class MultivariateDiagonalNormal(TorchDistributionWrapper):
    from torch.distributions import constraints
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(TorchNormal(loc, scale_diag),
                           reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        super().__init__(dist)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(ptu.create_stats_ordered_dict(
            'mean',
            ptu.get_numpy(self.mean),
            # exclude_max_min=True,
        ))
        stats.update(ptu.create_stats_ordered_dict(
            'std',
            ptu.get_numpy(self.distribution.stddev),
        ))
        return stats

    def __repr__(self):
        return self.distribution.base_dist.__repr__()

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = MultivariateDiagonalNormal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73

        This formula is mathematically equivalent to log(1 - tanh(x)^2).

        Derivation:

        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = - 2. * (
            ptu.from_numpy(np.log([2.]))
            - pre_tanh_value
            - torch.nn.functional.softplus(-2. * pre_tanh_value)
        ).sum(dim=1)
        return log_prob + correction

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            # errors or instability at values near 1
            value = torch.clamp(value, -0.999999, 0.999999)
            pre_tanh_value = torch.log(1+value) / 2 - torch.log(1-value) / 2
        return self._log_prob_from_pre_tanh(pre_tanh_value)

    def rsample_with_pretanh(self):
        z = (
                self.normal_mean +
                self.normal_std *
                MultivariateDiagonalNormal(
                    ptu.zeros(self.normal_mean.size()),
                    ptu.ones(self.normal_std.size())
                ).sample()
        )
        return torch.tanh(z), z

    def sample(self):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value.detach()

    def rsample(self):
        """
        Sampling in the reparameterization case.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        return value

    def sample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        value, pre_tanh_value = value.detach(), pre_tanh_value.detach()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_logprob_and_pretanh(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p, pre_tanh_value

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    @property
    def stddev(self):
        return self.normal_std

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(ptu.create_stats_ordered_dict(
            'mean',
            ptu.get_numpy(self.mean),
        ))
        stats.update(ptu.create_stats_ordered_dict(
            'normal/std',
            ptu.get_numpy(self.normal_std)
        ))
        stats.update(ptu.create_stats_ordered_dict(
            'normal/log_std',
            ptu.get_numpy(torch.log(self.normal_std)),
        ))
        return stats

def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)
    else:
        return np_array_or_other

def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other

def elem_or_tuple_to_numpy(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(np_ify(x) for x in elem_or_tuple)
    else:
        return np_ify(elem_or_tuple)

class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output

class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=ptu.identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

class DistributionGenerator(nn.Module, metaclass=abc.ABCMeta):
    def forward(self, *input, **kwarg) -> Distribution:
        raise NotImplementedError

class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input

class ModuleToDistributionGenerator(
    MultiInputSequential,
    DistributionGenerator,
    metaclass=abc.ABCMeta
):
    pass


class Beta(ModuleToDistributionGenerator):
    def forward(self, *input):
        alpha, beta = super().forward(*input)
        return Beta(alpha, beta)


class Gaussian(ModuleToDistributionGenerator):
    def __init__(self, module, std=None, reinterpreted_batch_ndims=1):
        super().__init__(module)
        self.std = std
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, *input):
        if self.std:
            mean = super().forward(*input)
            std = self.std
        else:
            mean, log_std = super().forward(*input)
            std = log_std.exp()
        return MultivariateDiagonalNormal(
            mean, std, reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)

class Bernoulli(Distribution, TorchBernoulli):
    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(ptu.create_stats_ordered_dict(
            'probability',
            ptu.get_numpy(self.probs),
        ))
        return stats

class BernoulliGenerator(ModuleToDistributionGenerator):
    def forward(self, *input):
        probs = super().forward(*input)
        return Bernoulli(probs)


class IndependentGenerator(ModuleToDistributionGenerator):
    def __init__(self, *args, reinterpreted_batch_ndims=1):
        super().__init__(*args)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, *input):
        distribution = super().forward(*input)
        return Independent(
            distribution,
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
        )

class GaussianMixtureDistribution(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = MultivariateDiagonalNormal(normal_means, normal_stds)
        self.normals = [MultivariateDiagonalNormal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = weights
        self.categorical = OneHotCategorical(self.weights[:, :, 0])

    def log_prob(self, value, ):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]
        log_p = torch.stack(log_p, -1)
        log_p = log_p.sum(dim=1)
        log_weights = torch.log(self.weights[:, :, 0])
        lp = log_weights + log_p
        m = lp.max(dim=1)[0]  # log-sum-exp numerical stability trick
        log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=1))
        return log_p_mixture

    def sample(self):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def rsample(self):
        z = (
                self.normal_means +
                self.normal_stds *
                MultivariateDiagonalNormal(
                    ptu.zeros(self.normal_means.size()),
                    ptu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.matmul(z, c)
        return torch.squeeze(s, 2)

    def mle_estimate(self):
        """Return the mean of the most likely component.

        This often computes the mode of the distribution, but not always.
        """
        c = ptu.zeros(self.weights.shape[:2])
        ind = torch.argmax(self.weights, dim=1) # [:, 0]
        c.scatter_(1, ind, 1)
        s = torch.matmul(self.normal_means, c[:, :, None])
        return torch.squeeze(s, 2)

    def __repr__(self):
        s = "GaussianMixture(normal_means=%s, normal_stds=%s, weights=%s)"
        return s % (self.normal_means, self.normal_stds, self.weights)

class GaussianMixtureFullDistribution(Distribution):
    def __init__(self, normal_means, normal_stds, weights):
        self.num_gaussians = weights.shape[-1]
        self.normal_means = normal_means
        self.normal_stds = normal_stds
        self.normal = MultivariateDiagonalNormal(normal_means, normal_stds)
        self.normals = [MultivariateDiagonalNormal(normal_means[:, :, i], normal_stds[:, :, i]) for i in range(self.num_gaussians)]
        self.weights = (weights + epsilon) / (1 + epsilon * self.num_gaussians)
        assert (self.weights > 0).all()
        self.categorical = Categorical(self.weights)

    def log_prob(self, value, ):
        log_p = [self.normals[i].log_prob(value) for i in range(self.num_gaussians)]
        log_p = torch.stack(log_p, -1)
        log_weights = torch.log(self.weights)
        lp = log_weights + log_p
        m = lp.max(dim=2, keepdim=True)[0]  # log-sum-exp numerical stability trick
        log_p_mixture = m + torch.log(torch.exp(lp - m).sum(dim=2, keepdim=True))
        raise NotImplementedError("from Vitchyr: idk what the point is of "
                                  "this class, so I didn't both updating "
                                  "this, but log_prob should return something "
                                  "of shape [batch_size] and not [batch_size, "
                                  "1] to be in accordance with the "
                                  "torch.distributions.Distribution "
                                  "interface.")
        return torch.squeeze(log_p_mixture, 2)

    def sample(self):
        z = self.normal.sample().detach()
        c = self.categorical.sample()[:, :, None]
        s = torch.gather(z, dim=2, index=c)
        return s[:, :, 0]

    def rsample(self):
        z = (
                self.normal_means +
                self.normal_stds *
                MultivariateDiagonalNormal(
                    ptu.zeros(self.normal_means.size()),
                    ptu.ones(self.normal_stds.size())
                ).sample()
        )
        z.requires_grad_()
        c = self.categorical.sample()[:, :, None]
        s = torch.gather(z, dim=2, index=c)
        return s[:, :, 0]

    def mle_estimate(self):
        """Return the mean of the most likely component.

        This often computes the mode of the distribution, but not always.
        """
        ind = torch.argmax(self.weights, dim=2)[:, :, None]
        means = torch.gather(self.normal_means, dim=2, index=ind)
        return torch.squeeze(means, 2)

    def __repr__(self):
        s = "GaussianMixture(normal_means=%s, normal_stds=%s, weights=%s)"
        return s % (self.normal_means, self.normal_stds, self.weights)


class GaussianMixture(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixtureDistribution(mixture_means, mixture_stds, weights)


class GaussianMixtureFull(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixtureFullDistribution(mixture_means, mixture_stds, weights)


class TanhGaussian(ModuleToDistributionGenerator):
    def forward(self, *input):
        mean, log_std = super().forward(*input)
        std = log_std.exp()
        return TanhNormal(mean, std)

class Policy(object, metaclass=abc.ABCMeta):
    """
    General policy interface.
    """
    @abc.abstractmethod
    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        pass

    def reset(self):
        pass

class ExplorationPolicy(Policy, metaclass=abc.ABCMeta):
    def set_num_steps_total(self, t):
        pass

class TorchStochasticPolicy(
    DistributionGenerator,
    ExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action(self, obs_np, ):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np, ):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist

class Delta(Distribution):
    """A deterministic distribution"""
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value.detach()

    def rsample(self):
        return self.value

    @property
    def mean(self):
        return self.value

    @property
    def variance(self):
        return 0

    @property
    def entropy(self):
        return 0

class TanhGaussianPolicy(Mlp, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

    def get_action(self, obs_np, ):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np, ):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist

class MakeDeterministic(TorchStochasticPolicy):
    def __init__(
            self,
            action_distribution_generator: DistributionGenerator,
    ):
        super().__init__()
        self._action_distribution_generator = action_distribution_generator

    def forward(self, *args, **kwargs):
        dist = self._action_distribution_generator.forward(*args, **kwargs)
        return Delta(dist.mle_estimate())