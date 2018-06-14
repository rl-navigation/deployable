from __future__ import print_function, division
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random, time, sys, collections

sys.dont_write_bytecode = True
from profile import *

#==============================================================================
# HELPERS
#==============================================================================

def print_opt_parameters(opt, named_parameters):
    # print optimized parameters
    print("----------------")
    print("Optimizing over:")
    names = {p:n for n,p in named_parameters}
    num_params = 0
    for k,v in [(k,v) for k,v in opt.param_groups[0].items() if k == "params"]:
        for p in v:
            name = names[p]
            if "diagnostic" not in name:
                num_params += p.numel()
                print(names[p].ljust(30), str(tuple(p.shape)).ljust(15), p.numel())
    print("Total parameters: {}".format(num_params))
    print("----------------")

#==============================================================================
# DATA STRUCTURES
#==============================================================================

Rollout    = collections.namedtuple("Rollout",    ["obs", "goal", "pact", "act", "ret", "done", "d_l", "d_g", "v_f"])
Losses     = collections.namedtuple("Losses",     ["loss_a", "loss_c", "loss_e", "loss_dl", "loss_dg"])
Telemetry  = collections.namedtuple("Telemetry",  ["ret", "v_f", "acc_dl", "acc_dg"])
Transition = collections.namedtuple("Transition", ["obs", "goals", "pacts", "acts", "rews", "nobs", "dones", "diag_locs", "diag_goals"])

#==============================================================================
# ACTOR-CRITIC
#==============================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, num_locations, gamma, ent_weight, lr):
        super(self.__class__,self).__init__()
        self.obs_size      = obs_space if type(obs_space) is int else obs_space.shape
        self.size_act      = act_space if type(act_space) is int else act_space.n
        self.gamma         = gamma
        self.ent_weight    = ent_weight
        self.encsize       = 256
        self.recsize       = 256

        # observation and goal
        csize = self.obs_size[0]*2

        # policy components
        self.encoder = nn.Sequential(nn.Linear(csize, self.encsize), nn.ReLU()).cuda()
        self.lstm    = nn.LSTMCell(self.encsize+self.size_act, self.recsize).cuda()
        self.actor   = nn.Sequential(nn.Linear(self.recsize, self.size_act)).cuda()
        self.critic  = nn.Sequential(nn.Linear(self.recsize, 1)).cuda()

        # auxiliary losses for diagnostics (gradients are not propagated into the agent)
        self.diagnostic_position = nn.Linear(self.recsize,num_locations).cuda()
        self.diagnostic_goal     = nn.Linear(self.recsize,num_locations).cuda()

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

        print_opt_parameters(self.opt, self.named_parameters())

    #--------------------------------------------------------------------------

    def rec(self, batch_size):
        return (self.lstm.weight_hh.data.new(batch_size, self.recsize),
                self.lstm.weight_hh.data.new(batch_size, self.recsize))

    #--------------------------------------------------------------------------

    def rec_detach(self, rstate):
        return (rstate[0].clone().detach(), rstate[1].clone().detach())

    #--------------------------------------------------------------------------

    def rec_mask(self, rstate, done):
        W = len(done)
        if isinstance(done, np.ndarray): done = torch.from_numpy(done).cuda()
        mask = (1-done).view(W,1).expand(W,self.recsize)
        return (rstate[0]*mask, rstate[1]*mask)

    #--------------------------------------------------------------------------

    def encode(self, x, g):
        return self.encoder(torch.cat([x,g],dim=1))

    #--------------------------------------------------------------------------

    def forward(self, obs, goals, act, r):
        x   = torch.from_numpy(obs  ).cuda().float()
        g   = torch.from_numpy(goals).cuda().float()
        a   = torch.from_numpy(act  ).cuda().float()
        e   = self.encode(x, g)
        e   = torch.cat([e, a], dim=1)
        h,c = self.lstm(e, r)
        return h,c

    #--------------------------------------------------------------------------

    def act(self, obs, goals, pact, r, deterministic=False):
        with torch.no_grad():
            h,c   = self.forward(obs, goals, pact, r)
            lgt   = self.actor(h)
            pi    = torch.distributions.Categorical(logits=lgt)
            act   = torch.max(lgt, dim=1)[1] if deterministic else pi.sample()
            all_p = F.softmax(lgt, dim=1)
            return act.data.squeeze().cpu().numpy(), all_p.data.cpu().numpy(), (h,c)

    #--------------------------------------------------------------------------

    def loc(self, r):
        with torch.no_grad():
            h,c = r
            d_l = self.diagnostic_position(h)
            d_g = self.diagnostic_goal    (h)
            d_l = F.softmax(d_l, dim=1)
            d_g = F.softmax(d_g, dim=1)
            return d_l.data.squeeze().cpu().numpy(), d_g.data.squeeze().cpu().numpy()

    #--------------------------------------------------------------------------

    def value(self, obs, goals, pact, r):
        with torch.no_grad():
            h,c = self.forward(obs, goals, pact, r)
            return self.critic(h).data.cpu().numpy()

    #--------------------------------------------------------------------------

    def _record_gradients(self, metrics):
        with torch.no_grad():
            for n,p in self.named_parameters():
                if p.grad is not None:
                    metrics["gradient/{}".format(n)] = torch.mean(p.grad).data.cpu().item()

    #--------------------------------------------------------------------------

    def train(self, rollout, rstate, v_final):
        metrics = {}

        losses, telemetry = self._loss(rollout, rstate=rstate, v_final=v_final)
        loss = losses.loss_a + losses.loss_c + losses.loss_e + losses.loss_dl + losses.loss_dg

        self.opt.zero_grad()
        loss.backward()
        self._record_gradients(metrics)
        self.opt.step()

        metrics.update({"training/v_f_min"   : np.min (telemetry.v_f),
                        "training/v_f_max"   : np.max (telemetry.v_f),
                        "training/v_f_mean"  : np.mean(telemetry.v_f),
                        "training/v_f_std"   : np.std (telemetry.v_f),
                        "training/loss_a"    : losses.loss_a              .data.cpu().item(),
                        "training/loss_c"    : losses.loss_c              .data.cpu().item(),
                        "training/loss_e"    : losses.loss_e              .data.cpu().item(),
                        "training/loss"      : loss                       .data.cpu().item(),
                        "training/diag_loc"  : telemetry.acc_dl           .data.cpu().item(),
                        "training/diag_goal" : telemetry.acc_dg           .data.cpu().item(),
                        "training/ret_min"   : torch.min (telemetry.ret  ).data.cpu().item(),
                        "training/ret_max"   : torch.max (telemetry.ret  ).data.cpu().item(),
                        "training/ret_mean"  : torch.mean(telemetry.ret  ).data.cpu().item(),
                        "training/ret_std"   : torch.std (telemetry.ret  ).data.cpu().item()})

        return metrics

    #--------------------------------------------------------------------------

    def _loss(self, rollout, rstate, v_final):
        rstate = self.rec_detach(rstate)

        R = self.rollout2var(rollout, v_final)

        T = len(rollout)
        W = len(rollout[0].obs)

        # agent losses
        losses_c  = []
        losses_a  = []
        losses_e  = []

        # diagnostics
        losses_dl = []
        losses_dg = []
        acc_dl    = []
        acc_dg    = []

        for t in range(T):
            # feedforward encoder
            f_enc = self.encode(R.obs[t],R.goal[t]).view(W,self.encsize)
            f_enc = torch.cat([f_enc, R.pact[t]], dim=1)

            # recurrent update
            rstate = self.lstm(f_enc, rstate)
            f_enc  = rstate[0]
            rstate = self.rec_mask(rstate, R.done[t])

            # actor
            logits = self.actor(f_enc).view(W,self.size_act)
            pi_t   = torch.distributions.Categorical(logits=logits)
            lp_t   = pi_t.log_prob(R.act[t].view(W)).view(W,1)

            # critic
            v_t = self.critic(f_enc).view(W,1)
            adv = R.ret[t] - v_t

            # accumulate results
            losses_c.append(0.5*adv**2)
            losses_a.append(-lp_t * adv.detach())
            losses_e.append(self.ent_weight*torch.sum(F.softmax(logits,dim=1)*F.log_softmax(logits,dim=1),dim=1))

            # diagnostics (gradients do not propagate into agent)
            d_f  = f_enc.detach()
            d_l  = self.diagnostic_position(d_f)
            d_g  = self.diagnostic_goal    (d_f)
            losses_dl.append(0.01*F.cross_entropy(d_l, R.d_l[t]))
            losses_dg.append(0.01*F.cross_entropy(d_g, R.d_g[t]))
            # diagnostic prediction accuracy
            d_pl = torch.max(d_l, dim=1)[1]
            d_pg = torch.max(d_g, dim=1)[1]
            acc_dl.append(torch.mean(torch.eq(d_pl,R.d_l[t]).float()))
            acc_dg.append(torch.mean(torch.eq(d_pg,R.d_g[t]).float()))

        # mean losses
        loss_a = torch.stack(losses_a).view(T,W,1)
        loss_c = torch.stack(losses_c).view(T,W,1)
        loss_e = torch.stack(losses_e).view(T,W,1)

        # total loss
        loss_a  = torch.mean(loss_a)
        loss_c  = torch.mean(loss_c)
        loss_e  = torch.mean(loss_e)
        loss_dl = torch.mean(torch.stack(losses_dl))
        loss_dg = torch.mean(torch.stack(losses_dg))

        # diagnostic accuracy
        acc_dl = torch.mean(torch.stack(acc_dl))
        acc_dg = torch.mean(torch.stack(acc_dg))

        return (Losses(loss_a=loss_a, loss_c=loss_c, loss_e=loss_e, loss_dl=loss_dl, loss_dg=loss_dg),
                Telemetry(ret=torch.stack(R.ret), v_f=R.v_f, acc_dl=acc_dl, acc_dg=acc_dg))

    #-------------------------------------------------------------------------

    def rollout2var(self, rollout, v_final):
        T = len(rollout)
        W = len(rollout[0].obs)

        os   = self.obs_size
        obs  = [torch.from_numpy(t.obs       ).cuda().float().view(W,*os) for t in rollout]
        goal = [torch.from_numpy(t.goals     ).cuda().float().view(W,*os) for t in rollout]
        pact = [torch.from_numpy(t.pacts     ).cuda().float().view(W, -1) for t in rollout]
        act  = [torch.from_numpy(t.acts      ).cuda(). long().view(W,  1) for t in rollout]
        done = [torch.from_numpy(t.dones     ).cuda().float().view(W,  1) for t in rollout]
        d_l  = [torch.from_numpy(t.diag_locs ).cuda(). long().view(W    ) for t in rollout]
        d_g  = [torch.from_numpy(t.diag_goals).cuda(). long().view(W    ) for t in rollout]

        ret = [np.zeros((W,1),np.float32) for t in range(T)] + [v_final.reshape(W,1)]
        for t in reversed(range(T)):
            r = rollout[t].rews .reshape(W,1)
            d = rollout[t].dones.reshape(W,1)
            ret[t] = r + (1-d) * self.gamma * ret[t+1]
        ret = [torch.from_numpy(r).cuda().float().view(W,1) for r in ret[:-1]]

        return Rollout(obs=obs, goal=goal, pact=pact, act=act, ret=ret, done=done, d_l=d_l, d_g=d_g, v_f=v_final)

