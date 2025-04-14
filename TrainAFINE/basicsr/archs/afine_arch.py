import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class AFINEQhead(nn.Module):
    def __init__(self, chns = (3, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768), feature_out_channel = 1,
                       input_dim = 768, hidden_dim = 128,
                       mean = (0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711)):
        super(AFINEQhead, self).__init__()

        self.chns = chns
        self.feature_out_channel = feature_out_channel
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.register_buffer("mean", torch.tensor(mean).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,-1,1,1))

        self.proj_feat = nn.Linear(input_dim * 2, hidden_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(self.chns[0] * 2 + hidden_dim * (len(self.chns) - 1), hidden_dim * 6),
            nn.GELU(),
            nn.Linear(hidden_dim * 6, self.feature_out_channel)
        )


    def forward(self, x, h_list_x):
        x = x * self.std + self.mean

        img_feature_x = x.flatten(2).permute(0, 2, 1)

        feature_list_x = []

        feature_list_x.append(img_feature_x)
        for h_x in h_list_x:
            feature_list_x.append(F.relu(h_x))

        final_feature_list_x = []

        for k in range(len(self.chns)):
            x_mean = feature_list_x[k].mean(1, keepdim=True)

            x_var = ((feature_list_x[k]-x_mean)**2).mean(1, keepdim=True)

            concat_x_feature = torch.cat((x_mean.flatten(1), x_var.flatten(1)), dim=1)

            if k != 0:
                concat_x_feature = self.proj_feat(concat_x_feature)

            final_feature_list_x.append(concat_x_feature)

        concat_final_feature_lixt_x = torch.cat(final_feature_list_x, dim = 1)

        n_x = self.proj_head(concat_final_feature_lixt_x)

        return n_x



@ARCH_REGISTRY.register()
class AFINEDhead(nn.Module):
    def __init__(self, chns = (3, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768),
                 mean = (0.48145466, 0.4578275, 0.40821073), std = (0.26862954, 0.26130258, 0.27577711)):
        super(AFINEDhead, self).__init__()

        self.chns = chns

        self.register_parameter("alpha", nn.Parameter(torch.randn(1, 1, sum(self.chns)), requires_grad=True))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, 1, sum(self.chns)), requires_grad=True))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)

        self.softplus = nn.Softplus()

        self.register_buffer("mean", torch.tensor(mean).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,-1,1,1))

    def forward(self, x, y, h_list_x, h_list_y):
        ### the input image should be generalized back to its original values
        x = x * self.std + self.mean
        y = y * self.std + self.mean

        # print(f"mean is {self.mean}, std is {self.std}")

        img_feature_x = x.flatten(2).permute(0, 2, 1)
        img_feature_y = y.flatten(2).permute(0, 2, 1)

        feature_list_x = []
        feature_list_y = []

        feature_list_x.append(img_feature_x)
        for h_x in h_list_x:
            feature_list_x.append(F.relu(h_x))

        feature_list_y.append(img_feature_y)
        for h_y in h_list_y:
            feature_list_y.append(F.relu(h_y))

        dist1 = 0
        dist2 = 0
        c1 = 1e-10
        c2 = 1e-10

        alpha_ = self.softplus(self.alpha)
        beta_ = self.softplus(self.beta)

        w_sum = alpha_.sum() + beta_.sum() + 1e-10
        alpha = torch.split(alpha_/w_sum, self.chns, dim=2)
        beta = torch.split(beta_/w_sum, self.chns, dim=2)

        for k in range(len(self.chns)):
            x_mean = feature_list_x[k].mean(1, keepdim=True)
            y_mean = feature_list_y[k].mean(1, keepdim=True)

            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            # print(f"feature_list_x{[k]} shape is {feature_list_x[k].shape}, feature_list_y{[k]} shape is {feature_list_y[k].shape}, alpha[{k}] shape is {alpha[k].shape}, S1 shape is {S1.shape}")
            dist1 = dist1+(alpha[k]*S1).sum(2,keepdim=True)

            x_var = ((feature_list_x[k]-x_mean)**2).mean(1, keepdim=True)
            y_var = ((feature_list_y[k]-y_mean)**2).mean(1, keepdim=True)
            xy_cov = (feature_list_x[k]*feature_list_y[k]).mean(1,keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(2,keepdim=True)

        score = 1 - (dist1+dist2).squeeze(2)
        # print(f"score shape is {score.shape}")

        return score


@ARCH_REGISTRY.register()
class AFINELearnLambda(nn.Module):
    def __init__(self, k = 5):
        super(AFINELearnLambda, self).__init__()

        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32), requires_grad = True)


    def forward(self, x_nr, ref_nr, xref_fr):
        k_ = F.softplus(self.k)
        # print(f"self.k is {self.k}, k_ is {k_}")
        u = torch.exp(k_*(ref_nr - x_nr)) * x_nr + xref_fr

        return u



@ARCH_REGISTRY.register()
### Non-linear mapping to generalize NR scores to a fixed limitation
class AFINENLM_NR_Fit(nn.Module):
    def __init__(self, yita1 = 2, yita2 = -2, yita3 = 3.7833, yita4 = 7.5676):
        super(AFINENLM_NR_Fit, self).__init__()
        self.yita3 = nn.Parameter(torch.tensor(yita3, dtype=torch.float32), requires_grad = True)
        self.yita4 = nn.Parameter(torch.tensor(yita4, dtype=torch.float32), requires_grad = True)
        self.yita1 = yita1
        self.yita2 = yita2

    def forward(self, x):
        # print(f"For NR, self.yita3 is {self.yita3}, self.yita4 is {self.yita4}")
        # d_hat = (self.yita1 - self.yita2) / (1 + torch.exp(-1 * (x - self.yita3) / (torch.abs(self.yita4) + 1e-10))) + self.yita2

        exp_pow = -1 * (x - self.yita3) / (torch.abs(self.yita4) + 1e-10)

        if exp_pow >=10:
            d_hat = (self.yita1 - self.yita2) * torch.exp(-1 * exp_pow) / (1 + torch.exp(-1 * exp_pow)) + self.yita2
        else:
            d_hat = (self.yita1 - self.yita2) / (1 + torch.exp(exp_pow)) + self.yita2

        return d_hat



@ARCH_REGISTRY.register()
### Non-linear mapping to generalize FR scores to a fixed limitation
class AFINENLM_FR_Fit_with_limit(nn.Module):
    def __init__(self, yita1 = 2, yita2 = -2, yita3 = -24.1335, yita4 = 8.1093, yita3_upper = -21, yita3_lower = -27, yita4_upper = 9, yita4_lower = 7):
        super(AFINENLM_FR_Fit_with_limit, self).__init__()
        self.yita3 = nn.Parameter(torch.tensor(yita3, dtype=torch.float32), requires_grad = True)
        self.yita4 = nn.Parameter(torch.tensor(yita4, dtype=torch.float32), requires_grad = True)
        self.yita1 = yita1
        self.yita2 = yita2
        self.yita3_upper = yita3_upper
        self.yita3_lower = yita3_lower
        self.yita4_upper = yita4_upper
        self.yita4_lower = yita4_lower

    def forward(self, x):
        yita3_ = torch.clamp(self.yita3, self.yita3_lower, self.yita3_upper)
        yita4_ = torch.clamp(self.yita4, self.yita4_lower, self.yita4_upper)
        # print(f"For FR, self.yita3 is {self.yita3}, yita3 is {yita3_}, self.yita4 is {self.yita4}, yita4 is {yita4_}")
        # d_hat = (self.yita1 - self.yita2) / (1 + torch.exp(-1 * (x - yita3_) / (torch.abs(yita4_) + 1e-10))) + self.yita2

        exp_pow = -1 * (x - yita3_) / (torch.abs(yita4_) + 1e-10)

        if exp_pow >=10:
            d_hat = (self.yita1 - self.yita2) * torch.exp(-1 * exp_pow) / (1 + torch.exp(-1 * exp_pow)) + self.yita2
        else:
            d_hat = (self.yita1 - self.yita2) / (1 + torch.exp(exp_pow)) + self.yita2

        return d_hat