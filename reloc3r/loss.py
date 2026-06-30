# references: DUSt3R: https://github.com/naver/dust3r


from copy import copy, deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LLoss (nn.Module):
    """ L-norm loss
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim-1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, LLoss), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = 'none'  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)

        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


# class GradientBlendingEstimator:
#     def __init__(self, model_class, model_kwargs):
#         """
#         Args:
#             model_class: Class of the model to instantiate
#             model_kwargs: Keyword arguments for model initialization
#         """
#         self.model_class = model_class
#         self.model_kwargs = model_kwargs

#     def gb_estimate(self, ckpt_N):
#         weights = {}
#         for modality in ["visual", "audio", "joint"]:
#             model = self.model_class(**self.model_kwargs)
#             model.load_state_dict(ckpt_N)

#             model_loss

            

#         return weights
        


class RelativeCameraPoseRegression(Criterion, MultiLoss): 
    def __init__(self, criterion):
        super().__init__(criterion)
        self.PoseLoss = Reloc3rPoseLoss() 

    # def reset_weights(self, weights):
    #     """ Reset weights for the loss function """
    #     if isinstance(self.PoseLoss, GradientBlendingLoss):
    #         self.PoseLoss.visual_weight = weights[0]
    #         self.PoseLoss.audio_weight = weights[1]
    #         self.PoseLoss.joint_weight = weights[2]
    #     else:
    #         pass

    def get_poses(self, gt1, gt2, pose1, pose2, rot_only=False):
        # if size of camera_pose is one item, dont do this. otherwise keep operation the same
        # if rot_only:
        #     # Compute raw differences
        #     gt_pose2to1 = gt1['relative_angle']  
        #     gt_pose1to2 = gt2['relative_angle']

        #     # # Wrap angles into [-180, 180)
        #     # gt_pose2to1 = (gt_pose2to1 + 180) % 360 - 180
        #     # gt_pose1to2 = (gt_pose1to2 + 180) % 360 - 180

        #     pr_pose2to1 = pose2['pose']
        #     pr_pose1to2 = pose1['pose']
        #     return gt_pose2to1, pr_pose2to1, gt_pose1to2, pr_pose1to2, {}
        # else:
        gt_pose2to1 = torch.inverse(gt1['camera_pose']) @ gt2['camera_pose']
        gt_pose1to2 = torch.inverse(gt2['camera_pose']) @ gt1['camera_pose']
        pr_pose2to1 = pose2['pose'] 
        pr_pose1to2 = pose1['pose']
        return gt_pose2to1, pr_pose2to1, gt_pose1to2, pr_pose1to2, {}

    # def get_poses(self, gt1, gt2, full_pose_info)f:
    #     gt_pose2to1 = torch.inverse(gt1['camera_pose']) @ gt2['camera_pose']
    #     gt_pose1to2 = torch.inverse(gt2['camera_pose']) @ gt1['camera_pose']
     
    #     return gt_pose2to1, None, gt_pose1to2, None, {}

    def compute_loss(self, gt1, gt2, pose1, pose2, rot_only=False, **kw):

        gt_pose2to1, pr_pose2to1, gt_pose1to2, pr_pose1to2, monitoring = self.get_poses(gt1, gt2, pose1, pose2, rot_only)
        # else:
        #     gt_pose2to1, pr_pose2to1, gt_pose1to2, pr_pose1to2, monitoring = self.get_poses(gt1, gt2, full_poses)   
        #     pr_pose2to1 = {"joint": full_poses["joint"][1],
        #                     "visual": full_poses["visual"][1],
        #                     "audio":  full_poses["audio"][1]}
        #     pr_pose1to2 = {"joint": full_poses["joint"][0],
        #                     "visual": full_poses["visual"][0],
        #                     "audio": full_poses["audio"][0]}

        # compute loss
        #if not rot_only:
        loss_pose2, loss_Terr2, loss_Rerr2 = self.PoseLoss(pr_pose2to1, gt_pose2to1)   
        loss_pose1, loss_Terr1, loss_Rerr1 = self.PoseLoss(pr_pose1to2, gt_pose1to2)


        # record and return details
        self_name = type(self).__name__
        details = {
                self_name+'_terr1': float(loss_Terr1*180 / math.pi),
                self_name+'_rerr1': float(loss_Rerr1*180 / math.pi),
                self_name+'_terr2': float(loss_Terr2*180 / math.pi),
                self_name+'_rerr2': float(loss_Rerr2*180 / math.pi),
                #    self_name+'_visual_terr1': float(visual_loss_Terr1*180 / math.pi),
                #    self_name+'_visual_rerr1': float(visual_loss_Rerr1*180 / math.pi),
                #    self_name+'_visual_terr2': float(visual_loss_Terr2*180 / math.pi),
                #    self_name+'_visual_rerr2': float(visual_loss_Rerr2*180 / math.pi),
                #    self_name+'_audio_terr1': float(audio_loss_Terr1*180 / math.pi),
                #    self_name+'_audio_rerr1': float(audio_loss_Rerr1*180 / math.pi),
                #    self_name+'_audio_terr2': float(audio_loss_Terr2*180 / math.pi),
                #    self_name+'_audio_rerr2': float(audio_loss_Rerr2*180 / math.pi),
                }
        return loss_pose1 + loss_pose2, dict(pose_loss = float(loss_pose1 + loss_pose2), **(details | monitoring))

        # else:
        #     # loss1 = nn.functional.l1_loss(pr_pose1to2, gt_pose1to2, reduction='mean')
        #     # loss2 = nn.functional.l1_loss(pr_pose2to1, gt_pose2to1, reduction='mean')

        #     # convert values to rotation matrices
        #     # TODO: let pr be cos value predicted, and take acos of pr
        #     gt_R1 = gt_pose1to2
        #     gt_R2 = gt_pose2to1
        #     pr_R1 = pr_pose1to2 
        #     pr_R2 = pr_pose2to1 

        #     # TODO: let pr be cos value predicted, and take acos of pr
        #     # pr_R1 = torch.clamp(pr_R1, -1.0, 1.0)
        #     # pr_R2 = torch.clamp(pr_R2, -1.0, 1.0)
        #     # pr_R1 = torch.acos(pr_R1)
        #     # pr_R2 = torch.acos(pr_R2)

        #     pr_R1[pr_R1 <= -90] = -180 - pr_R1[pr_R1 <= -90]
        #     pr_R1[pr_R1 >= 90] = 180 - pr_R1[pr_R1 >= 90]

        #     pr_R2[pr_R2 <= -90] = -180 - pr_R2[pr_R2 <= -90]
        #     pr_R2[pr_R2 >= 90] = 180 - pr_R2[pr
        # _R2 >= 90]

        #     loss1 = torch.mean(abs(pr_R1 - gt_R1)) 
        #     loss2 = torch.mean(abs(pr_R2 - gt_R2))

        #     loss = loss1 + loss2
        #     self_name = type(self).__name__
        #     details = {
        #         self_name+'_rot_only_loss1': float(loss1),
        #         self_name+'_rot_only_loss2': float(loss2),
        #     }
        #     return loss, dict(pose_loss = float(loss), **(details | monitoring))

class GradientBlendingLoss (nn.Module):
    def __init__(self, visual_weight=0.85, audio_weight=0.05, joint_weight=0.1):
        "Expects weights for each model, the combined model, and an overall scale"
        super(GradientBlendingLoss, self).__init__()
        self.audio_weight = audio_weight
        self.visual_weight = visual_weight
        self.joint_weight = joint_weight

        self.criterion = Reloc3rPoseLoss()  

    # def reset_weights(self, visual_weight=0.3, audio_weight=0.3, joint_weight=0.4):
    #     self.visual_weight = visual_weight
    #     self.audio_weight = audio_weight
    #     self.joint_weight = joint_weight

    def forward(self, pose_pred, pose_gt, weights=None):
        if weights is not None:
            self.visual_weight = weights["visual"]
            self.audio_weight = weights["audio"]
            self.joint_weight = weights["joint"]

        visual_pose_pred = pose_pred["visual"]["pose"]
        audio_pose_pred = pose_pred["audio"]["pose"]
        joint_pose_pred = pose_pred["joint"]["pose"]

        visual_loss, visual_tloss, visual_rloss = self.criterion(visual_pose_pred, pose_gt)
        audio_loss, audio_tloss, audio_rloss = self.criterion(audio_pose_pred, pose_gt)
        joint_loss, joint_tloss, joint_rloss = self.criterion(joint_pose_pred, pose_gt)

        loss = self.visual_weight * visual_loss + self.audio_weight * audio_loss + self.joint_weight * joint_loss
        tloss = self.visual_weight * visual_tloss + self.audio_weight * audio_tloss + self.joint_weight * joint_tloss
        rloss = self.visual_weight * visual_rloss + self.audio_weight * audio_rloss + self.joint_weight * joint_rloss
        return (visual_loss, visual_tloss, visual_rloss), (audio_loss, audio_tloss, audio_rloss), (loss, tloss, rloss)


class Reloc3rPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pose_pred, pose_gt):
        t = pose_pred[:,0:3,-1]
        tgt = pose_gt[:,0:3,-1]
        R = pose_pred[:, :3, :3]
        Rgt = pose_gt[:, :3, :3]

        trans_loss = self.transl_ang_loss(t, tgt)
        rot_loss = self.rot_ang_loss(R, Rgt)
        loss = rot_loss + trans_loss #trans_loss + rot_loss
        return loss, trans_loss, rot_loss

    def transl_ang_loss(self, t, tgt, eps=1e-6):
        """
        Args: 
            t: estimated translation vector [B, 3]
            tgt: ground-truth translation vector [B, 3]
        Returns: 
            T_err: translation direction angular error 
        """
        t_norm = torch.norm(t, dim=1, keepdim=True)
        t_normed = t / (t_norm + eps)
        tgt_norm = torch.norm(tgt, dim=1, keepdim=True)
        tgt_normed = tgt / (tgt_norm + eps)
        cosine = torch.sum(t_normed * tgt_normed, dim=1)
        T_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return T_err.mean()

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()


class MSELoss(nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        self.criterion = criterion if criterion is not None else nn.MSELoss()

    def forward(self, view, doas):
        """ Computes the loss for a single view and its DOAs
        Args:
            view: dict with keys 'img', 'camera_intrinsics', 'camera_pose'
            doas: predicted DOAs
        Returns:
            loss: computed loss value
        """
        assert "doas" in view, "view must contain 'doas' key"
        # print(view['doas'].shape)
        # print(doas.shape)
        return self.criterion(doas, view['doas'])

class PolicyBCELoss(MultiLoss):
    """ MultiLoss for policy decision
    """
    def __init__(self, criterion=None):
        super().__init__()
        self.energy_savings_factor = 0.8 #* TODO: figure out what constant can enable me to minimize vision usage while maximizing performance
        if self.energy_savings_factor == 1:
            self.criterion = nn.BCEWithLogitsLoss() 
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([self.energy_savings_factor])) #if criterion is None else criterion


    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'


    def compute_loss(self, view, policy_decision, **kw):
        policy_loss = self.criterion(policy_decision, view['policy_decision'].unsqueeze(1).float())
        audio_rot_error = view['audio_rot_error']
        audio_trans_error = view['audio_trans_error']
        visual_rot_error = view['vision_rot_error']
        visual_trans_error = view['vision_trans_error']

           # decision_mask = policy_decision > 0  # logits > 0 => prob > 0.5
        prob = torch.sigmoid(policy_decision)  # keep differentiable w.r.t. policy_decision

        # ensure dtype match for safe operations
        audio_rot_error = audio_rot_error.to(prob.dtype)
        audio_trans_error = audio_trans_error.to(prob.dtype)
        visual_rot_error = visual_rot_error.to(prob.dtype)
        visual_trans_error = visual_trans_error.to(prob.dtype)

        # Expected error under the policy probability (differentiable)
        selected_rot_error = prob * audio_rot_error + (1.0 - prob) * visual_rot_error
        selected_trans_error = prob * audio_trans_error + (1.0 - prob) * visual_trans_error

        selected_rot_error = prob*audio_rot_error+ (1.0 - prob)*visual_rot_error
        selected_trans_error = prob*audio_trans_error + (1.0 - prob)*visual_trans_error

        #Optionally log the expected errors
        policy_rot_err = selected_rot_error.mean()
        policy_trans_err = selected_trans_error.mean()

        loss = policy_loss + 0.5 * (audio_rot_error.mean() + audio_trans_error.mean()) + 0.05 * (visual_rot_error.mean() + visual_trans_error.mean()) #*10 + policy_rot_err + policy_trans_err #3 + 10*self.contrastive_regularizer(embedding, view['policy_decision']) + torch.log(policy_rot_err + policy_trans_err) # + torch.log(policy_rot_err)
        return loss, dict(policy_loss=float(loss))
    
        # if not torch.is_tensor(policy_loss):
        #     policy_loss = torch.tensor(policy_loss, device=policy_decision.device, dtype=policy_decision.dtype)
        # return policy_loss, dict(loss=float(policy_loss))

class DOALoss(MultiLoss):
    """ MultiLoss for DOA estimation
    """
    def __init__(self, criterion=None):
        super().__init__()
        self.criterion = MSELoss(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def compute_loss(self, view, doas, **kw):
        loss = self.criterion(view, doas)
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=doas.device, dtype=doas.dtype)

        device = doas.device
        view_doas = view["doas"].to(device)
        nfft = 1024


        pred_p = torch.nn.functional.pad(doas, (0, nfft - doas.shape[-1]))
        targ_p = torch.nn.functional.pad(view_doas, (0, nfft - view_doas.shape[-1]))


       # Compute FFT along last dimension
        S_pred = torch.fft.fft(pred_p, n=nfft, dim=-1)
        S_targ = torch.fft.fft(targ_p, n=nfft, dim=-1)


        # Magnitude loss
        mag_loss = torch.mean((torch.abs(S_pred) - torch.abs(S_targ))**2)
        comp_loss = torch.mean(torch.abs(S_pred - S_targ)**2)

        loss = loss + mag_loss + comp_loss

        return loss, dict(doa_loss=float(loss))

    # def compute_loss(self, view, doas, **kw):
    #     """ Computes the loss for a single view and its DOAs
    #     Args:
    #         view: dict with keys 'img', 'camera_intrinsics', 'camera_pose'
    #         doas: predicted DOAs
    #     Returns:
    #         loss: computed loss value
    #     """
    #     loss = self.criterion(view, doas)
    #     device = doas.device
    #     doas = doas.to(device)
    #     view_doas = view["doas"].to(device)
    #     nfft = 1024

    #     pred_p = torch.nn.functional.pad(doas, (0, nfft - doas.shape[-1]))
    #     targ_p = torch.nn.functional.pad(view_doas, (0, nfft - view_doas.shape[-1]))

    #     S_pred = torch.fft.rfft(pred_p, n=nfft, dim=-1)  # complex
    #     S_targ = torch.fft.rfft(targ_p, n=nfft, dim=-1)

    #     # S_pred_abs = torch.abs(S_pred)
    #     # S_targ_abs = torch.abs(S_targ)
    #     S_pred_abs = torch.sqrt(S_pred.real**2 + S_pred.imag**2)
    #     S_targ_abs = torch.sqrt(S_targ.real**2 + S_targ.imag**2)
    #     S_error = S_pred - S_targ
    #     S_error_abs_sq = S_error.real**2 + S_error.imag**2
    #     mag_loss = torch.mean((S_pred_abs - S_targ_abs)**2)
    #     comp_loss = torch.mean(S_error_abs_sq)

    #     loss += mag_loss + comp_loss
    #     return loss, dict(doa_loss=float(loss))

# TODO: add weights to this function that can be added
def loss_of_one_batch(batch, model, criterion, device, use_amp=False, ret=None):
    view1, view2 = batch

    for view in batch:
        for name in 'img camera_intrinsics camera_pose doas audio_spec yamnet_embed policy_decision audio_rot_error audio_trans_error vision_rot_error vision_trans_error'.split(): 
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if "doas" in view1:
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            doas, emb = model(view1)
            with torch.cuda.amp.autocast(enabled=False):
                loss = criterion(view1, doas) if criterion is not None else None
            result = dict(view1=view1, doas=doas, loss=loss)
            return result[ret] if ret else result
    elif "policy_decision" in view1:
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            policy_decision = model(view1, view2)
            with torch.cuda.amp.autocast(enabled=False):
                loss = criterion(view1, policy_decision) if criterion is not None else None
            result = dict(view1=view1, view2=view2, policy_decision=policy_decision, loss=loss)
            return result[ret] if ret else result
    elif "waveform" in view1:
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            inputs = {'img_1': view1['img'], 
                      'img_2': view2['img'],
                      'audio_1': view1['waveform'],
                      'audio_2': view2['waveform'],}

            with torch.cuda.amp.autocast(enabled=False):
                loss = model(inputs, loss=True)
            loss = loss.mean()
            loss_details= {"spec_loss": float(loss)}
            result = dict(view1=view1, view2=view2, loss=(loss, loss_details))
            return result[ret] if ret else result
    else:
        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            pose1, pose2 = model(view1, view2) #full_poses["joint"]

            # loss is supposed to be symmetric
            with torch.cuda.amp.autocast(enabled=False):
                if "slfm" in view1["dataset"][0].lower():   
                    loss = criterion(view1, view2, pose1, pose2, rot_only=True)
                else:
                    loss = criterion(view1, view2, pose1, pose2) if criterion is not None else None

            result = dict(view1=view1, view2=view2, pose1=pose1, pose2=pose2, loss=loss)
            return result[ret] if ret else result

