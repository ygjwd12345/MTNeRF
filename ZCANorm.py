from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from mpmath import *
import numpy as np


eps=1e-10

class BatchNorm(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-2, momentum=0.1, affine=True):
        super(BatchNorm, self).__init__()
        print('BatchNorm Group Num {}'.format(groups))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(groups, int(num_features/groups), 1))
        self.bias = Parameter(torch.Tensor(groups, int(num_features/groups), 1))
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        self.register_buffer("running_subspace", torch.eye(length, length).view(1,length,length).repeat(self.groups,1,1))

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))


    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            a, HW = x.size()
            N=1024
            C=64
            if a <N*C:
                return x
            x=x.view(N,C,HW)
            x = x.transpose(0,1).contiguous().view(C, -1)
            mean = x.mean(1, keepdim=True)
            var = x.var(1,keepdim=True)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum)  * mean + self.momentum * self.running_mean
                self.running_var = (1 - self.momentum)  * var + self.momentum * self.running_mean
            G = self.groups
            n_mem = C // G
            x = x.view(G, n_mem, -1)
            mu = x.mean(2, keepdim=True)
            xx = torch.bmm((x-mu), (x-mu).transpose(1, 2)) / (N * HW) + torch.eye(n_mem, out=torch.empty_like(x)).unsqueeze(
                0) * self.eps
            x = (x-mean)/torch.sqrt(var+1e-10) * self.weight + self.bias
            x = x.view(C, N, HW).transpose(0, 1)
            return x.reshape(-1,HW)
        else:
            N, C, H, W = x.size()
            x = x.transpose(0,1).contiguous().view(C, -1)
            x = (x - self.running_mean) / torch.sqrt(self.running_var+eps)
            x = x * self.weight + self.bias
            x = x.view(C, N, HW).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, '.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class ZCANormBatch(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-2, momentum=0.1, affine=True):
        super(ZCANormBatch, self).__init__()
        print('ZCANormBatch Group Num {}'.format(groups))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(groups, int(num_features/groups), 1))
        self.bias = Parameter(torch.Tensor(groups, int(num_features/groups), 1))
        #Matrix Square Root or Inverse Square Root layer
        self.svdlayer = MPA_Lya.apply
        self.register_buffer('running_mean', torch.zeros(groups, int(num_features/groups), 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        self.register_buffer("running_subspace", torch.eye(length, length).view(1,length,length).repeat(self.groups,1,1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            n_mem = C // G
            x = x.transpose(0, 1).contiguous().view(G, n_mem, -1)
            mu = x.mean(2, keepdim=True)
            x = x - mu
            xxt = torch.bmm(x, x.transpose(1,2)) / (N * H * W) + torch.eye(n_mem, out=torch.empty_like(x)).unsqueeze(0) * self.eps
            assert C % G == 0
            subspace = torch.inverse(self.svdlayer(xxt))
            xr = torch.bmm(subspace, x)
            with torch.no_grad():
                running_subspace = self.__getattr__('running_subspace')
                running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr

        else:
            N, C, H, W = x.size()
            G = self.groups
            n_mem = C // G
            x = x.transpose(0, 1).contiguous().view(G, n_mem, -1)
            x = (x - self.running_mean)
            subspace = self.__getattr__('running_subspace')
            x= torch.bmm(subspace, x)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormBatch, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
mp.dps = 32
one = mpf(1)
mp.pretty = True

def f(x):
    return sqrt(one-x)

# Derive the taylor and pade' coefficients for MTP, MPA
a = taylor(f, 0, 10)
pade_p, pade_q = pade(a, 5, 5)
a = torch.from_numpy(np.array(a).astype(float))
pade_p = torch.from_numpy(np.array(pade_p).astype(float))
pade_q = torch.from_numpy(np.array(pade_q).astype(float))

def matrix_taylor_polynomial(p, I):
    p_sqrt= I
    p_app = I - p
    p_hat = p_app
    for i in range(10):
      p_sqrt += a[i+1]*p_hat
      p_hat = p_hat.bmm(p_app)
    return p_sqrt

def matrix_pade_approximant(p,I):
    p_sqrt = pade_p[0]*I
    q_sqrt = pade_q[0]*I
    p_app = I - p
    p_hat = p_app
    for i in range(5):
        p_sqrt += pade_p[i+1]*p_hat
        q_sqrt += pade_q[i+1]*p_hat
        p_hat = p_hat.bmm(p_app)
    #There are 4 options to compute the MPA: comput Matrix Inverse or Matrix Linear System on CPU/GPU;
    #It seems that single matrix is faster on CPU and batched matrices are faster on GPU
    #Please check which one is faster before running the code;
    return torch.linalg.solve(q_sqrt, p_sqrt)
    #return torch.linalg.solve(q_sqrt.cpu(), p_sqrt.cpu()).cuda()
    #return torch.linalg.inv(q_sqrt).mm(p_sqrt)
    #return torch.linalg.inv(q_sqrt.cpu()).cuda().bmm(p_sqrt)

def matrix_pade_approximant_inverse(p,I):
    p_sqrt = pade_p[0]*I
    q_sqrt = pade_q[0]*I
    p_app = I - p
    p_hat = p_app
    for i in range(5):
        p_sqrt += pade_p[i+1]*p_hat
        q_sqrt += pade_q[i+1]*p_hat
        p_hat = p_hat.bmm(p_app)
    #There are 4 options to compute the MPA_inverse: comput Matrix Inverse or Matrix Linear System on CPU/GPU;
    #It seems that single matrix is faster on CPU and batched matrices are faster on GPU
    #Please check which one is faster before running the code;
    return torch.linalg.solve(p_sqrt, q_sqrt)
    #return torch.linalg.solve(p_sqrt.cpu(), q_sqrt.cpu()).cuda()
    #return torch.linalg.inv(p_sqrt).mm(q_sqrt)
    #return torch.linalg.inv(p_sqrt.cpu()).cuda().bmm(q_sqrt)

#Differentiable Matrix Square Root by MPA_Lya
class MPA_Lya(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        normM = torch.norm(M,dim=[1,2]).reshape(M.size(0),1,1)
        I = torch.eye(M.size(1), requires_grad=False, device=M.device).reshape(1,M.size(1),M.size(1)).repeat(M.size(0),1,1)
        #This is for MTP calculation
        #M_sqrt = matrix_taylor_polynomial(M/normM,I)
        M_sqrt = matrix_pade_approximant(M / normM, I)
        M_sqrt = M_sqrt * torch.sqrt(normM)
        ctx.save_for_backward(M, M_sqrt, normM,  I)
        return M_sqrt

    @staticmethod
    def backward(ctx, grad_output):
        M, M_sqrt, normM,  I = ctx.saved_tensors
        b = M_sqrt / torch.sqrt(normM)
        c = grad_output / torch.sqrt(normM)
        for i in range(8):
            #In case you might terminate the iteration by checking convergence
            #if th.norm(b-I)<1e-4:
            #    break
            b_2 = b.bmm(b)
            c = 0.5 * (c.bmm(3.0*I-b_2)-b_2.bmm(c)+b.bmm(c).bmm(b))
            b = 0.5 * b.bmm(3.0 * I - b_2)
        grad_input = 0.5 * c
        return grad_input

#Differentiable Inverse Square Root by MPA_Lya_Inv
class MPA_Lya_Inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        normM = torch.norm(M,dim=[1,2]).reshape(M.size(0),1,1)
        I = torch.eye(M.size(1), requires_grad=False, device=M.device).reshape(1,M.size(1),M.size(1)).repeat(M.size(0),1,1)
        #M_sqrt = matrix_taylor_polynomial(M/normM,I)
        M_sqrt_inv = matrix_pade_approximant_inverse(M / normM, I)
        M_sqrt_inv = M_sqrt_inv / torch.sqrt(normM)
        ctx.save_for_backward(M, M_sqrt_inv,  I)
        return M_sqrt_inv

    @staticmethod
    def backward(ctx, grad_output):
        M, M_sqrt_inv,  I = ctx.saved_tensors
        M_inv = M_sqrt_inv.bmm(M_sqrt_inv)
        grad_lya = - M_inv.bmm(grad_output).bmm(M_inv)
        norm_sqrt_inv = torch.norm(M_sqrt_inv)
        b = M_sqrt_inv / norm_sqrt_inv
        c = grad_lya / norm_sqrt_inv
        for i in range(8):
            #In case you might terminate the iteration by checking convergence
            #if th.norm(b-I)<1e-4:
            #    break
            b_2 = b.bmm(b)
            c = 0.5 * (c.bmm(3.0 * I - b_2) - b_2.bmm(c) + b.bmm(c).bmm(b))
            b = 0.5 * b.bmm(3.0 * I - b_2)
        grad_input = 0.5 * c
        return grad_input