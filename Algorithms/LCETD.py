from Algorithms.BaseTD import BaseTD
import numpy as np


class LCETD(BaseTD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.F = 1
        self.old_rho = 0
        self.step = 0  # time step
        self.beta = kwargs.get('beta')
        self.type = 'diag'
        if self.task.num_policies > 1:
            self.F = np.ones(self.task.num_policies)
            self.old_rho = np.zeros(self.task.num_policies)

    @staticmethod
    def related_parameters():
        return ['alpha', 'lmbda', 'beta']

    def get_m(self, beta_vec=None):
        if self.task.num_policies == 1:
            if self.type == 'diag':  # connecting full IS TD and off-policy TD
                coef1 = self.lmbda * (1-self.beta)**self.beta * self.step**(-self.beta)
                coef2 = self.lmbda * (1-self.beta) * self.step**(-self.beta)
            elif self.type == 'bottom':  # connecting AETD and off-policy TD
                coef1 = self.lmbda * self.step**(- self.beta)
                coef2 = coef1
            elif self.type == 'left':  # ETDLB2
                coef1 = self.lmbda
                coef2 = self.lmbda * (1 - self.beta)
            elif self.type == 'right':  # connecting AETD and full IS TD
                coef1 = self.lmbda * (1-self.beta) * self.step**(-1)
                coef2 = coef1
            elif self.type == 'diag2':  # connecting full IS TD and AETD
                coef1 = self.lmbda * (1-self.beta)**(1-self.beta) * self.step**(self.beta-1)
                coef2 = self.lmbda * (1-self.beta) * self.step**(self.beta-1)
            else:
                raise NotImplementedError
            m = (1 - coef1) * self.F + coef2
        else:
            if self.type == 'diag':  # connecting full IS TD and off-policy TD
                coef1_vec = self.lmbda * (1-beta_vec)**beta_vec * self.step**(-beta_vec)
                coef2_vec = self.lmbda * (1-beta_vec) * self.step**(-beta_vec)
            elif self.type == 'bottom':  # connecting AETD and off-policy TD
                coef1_vec = self.lmbda * self.step**(-beta_vec)
                coef2_vec = coef1_vec
            elif self.type == 'left':  # ETDLB2
                coef1_vec = self.lmbda * np.ones(self.task.num_policies)
                coef2_vec = self.lmbda * (1 - beta_vec)
            elif self.type == 'right':  # connecting AETD and full IS TD
                coef1_vec = self.lmbda * (1-beta_vec) * self.step**(-1)
                coef2_vec = coef1_vec
            elif self.type == 'diag2':  # connecting full IS TD and AETD
                coef1_vec = self.lmbda * (1-beta_vec)**(1-beta_vec) * self.step**(beta_vec-1)
                coef2_vec = self.lmbda * (1-beta_vec) * self.step**(beta_vec-1)
            else:
                raise NotImplementedError
            m = (1 - coef1_vec) * self.F + coef2_vec
        return m

    def learn_single_policy(self, s, s_p, r, is_terminal):
        x, x_p = self.get_features(s, s_p, is_terminal)
        delta = self.get_delta(r, x, x_p)
        # self.F = self.beta * self.old_rho * self.F + 1
        self.step += 1
        if self.step == 1:
            # FIXME: this is a hack to get correct values for the first step
            coef = self.F
        elif self.type == 'diag':  # connecting full IS TD and off-policy TD
            coef = (1 - self.beta) * self.step**(-self.beta)
        elif self.type == 'bottom':  # connecting AETD and off-policy TD
            coef = self.step**(-self.beta)
        elif self.type == 'left':  # ETDLB2
            coef = 1 - self.beta
        elif self.type == 'right':  # connecting AETD and full IS TD
            coef = (1 - self.beta) * self.step**(-1)
        elif self.type == 'diag2':  # connecting full IS TD and AETD
            coef = (1 - self.beta) * self.step**(self.beta-1)
        else:
            raise NotImplementedError
        self.F = (1 - coef) * self.old_rho * self.F + coef
        m = self.get_m()
        rho = self.get_isr(s)
        self.z = rho * (x * m + self.gamma * self.lmbda * self.z)
        self.w += self.compute_step_size() * delta * self.z
        self.old_rho = rho

    def learn_multiple_policies(self, s, s_p, r, is_terminal):
        delta, alpha_vec, *_, rho, _ = super().learn_multiple_policies(s, s_p, r, is_terminal)
        stacked_x = self.task.stacked_feature_rep[:, :, s]
        beta_vec = self.beta * self.gamma_vec_t / self.gamma
        # self.F = beta_vec * self.old_rho * self.F + np.ones(self.task.num_policies)
        self.step += 1
        if self.step == 1:
            # FIXME: this is a hack to get correct values for the first step
            coef_vec = self.F * np.ones(self.task.num_policies)
        elif self.type == 'diag':  # connecting full IS TD and off-policy TD
            coef_vec = (1 - beta_vec) * self.step**(-beta_vec)
        elif self.type == 'bottom':  # connecting AETD and off-policy TD
            coef_vec = self.step**(-beta_vec)
        elif self.type == 'left':  # ETDLB2
            coef_vec = 1 - beta_vec
        elif self.type == 'right':  # connecting AETD and full IS TD
            coef_vec = (1 - beta_vec) * self.step**(-1)
        elif self.type == 'diag2':  # connecting full IS TD and AETD
            coef_vec = (1 - beta_vec) * self.step**(beta_vec-1)
        else:
            raise NotImplementedError
        self.F = (1 - coef_vec) * self.old_rho * self.F + coef_vec
        m = self.get_m(beta_vec)
        self.z = rho[:, None] * (self.lmbda * self.z * self.gamma_vec_t[:, None] + stacked_x * m[:, None])
        self.w += (alpha_vec * delta)[:, None] * self.z
        self.old_rho = rho
        self.gamma_vec_t = self.gamma_vec_tp

    def reset(self):
        super().reset()
        self.F = 1
        self.old_rho = 0
        self.step = 0
        if self.task.num_policies > 1:
            self.old_rho = np.zeros(self.task.num_policies)
            self.F = np.ones(self.task.num_policies)


class LCETD1(LCETD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.type = 'diag'


class LCETD2(LCETD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.type = 'bottom'


class LCETD3(LCETD):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.type = 'right'
