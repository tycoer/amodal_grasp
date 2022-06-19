from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_loss
from mmcv.runner import BaseModule
from mmdet.models.detectors.base import BaseDetector
from torchvision.models.resnet import resnet50, resnet18, resnet34
from torchvision.models.detection.faster_rcnn import resnet_fpn_backbone
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models._utils import IntermediateLayerGetter
from mmdet.models.dense_heads.yolact_head import YOLACTProtonet


class Protonet(YOLACTProtonet):
    def forward(self, x):
        return self.protonet(x)




import matplotlib.pyplot as plt
def resnet50_without_fc(pretrained=False):
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    model = IntermediateLayerGetter(resnet50(pretrained), return_layers=return_layers)
    return model

def resnet34_without_fc(pretrained=False):
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    model = IntermediateLayerGetter(resnet34(pretrained), return_layers=return_layers)
    return model

def resnet18_without_fc(pretrained=False):
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    model = IntermediateLayerGetter(resnet18(pretrained), return_layers=return_layers)
    return model



class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx




def batch_index_vu_value(tensor: torch.Tensor,
                   vu: torch.Tensor):
    '''
    example:
    N, C, H, W = 2, 3, 3, 3
    tensor = torch.randn(N, C, H, W)
    vu = torch.randint(H, (N, C, 2), dtype=torch.long)
    vu_value = batch_index_vu_value(tensor, vu)
    '''

    assert tensor.dim() == 4
    assert vu.dim() == 3 and vu.size(-1) == 2
    N, C, H, W = tensor.shape
    u =  vu[:, :, 1]
    v =  vu[:, :, 0]

    vu_flatten = v * W + u
    vu_flatten = vu_flatten.view(N, C, 1)
    tensor = tensor.view(N, C, H * W)
    vu_value =  torch.gather(tensor, -1 , vu_flatten)
    return vu_value



class FeatsFusion(nn.Module):
    def __init__(self):
        super(FeatsFusion, self).__init__()

        self.conv3 = nn.ConvTranspose2d(2048, 1024, 2, 2)
        self.conv2 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv0 = nn.Sequential(
            nn.Conv2d(256, 512, 1,),
            # nn.Conv2d(512, 1024, 1,),
            # nn.Conv2d(1024, 512, 1,),
            nn.Conv2d(512, 256, 1,),
            nn.Conv2d(256, 128, 1,),
            nn.Conv2d(128, 64, 1),
            nn.Conv2d(64, 32, 1),
        )



    def forward(self,
                feats):
        feat0, feat1, feat2, feat3 = feats
        x3 = self.conv3(feat3)
        x2 = self.conv2(x3 + feat2)
        x1 = self.conv1(x2 + feat1)
        x0 = x1 + feat0

        out = self.conv0(F.relu(x0))
        return out

class GraspHead(nn.Module):
    def __init__(self):
        super(GraspHead, self).__init__()
        self.conv = nn.Conv2d(256, 1, 1)
    def forward_train(self,
                      x,
                      gt_vu,
                      gt_qual,
                      ):
        x = self.conv(x)
        losses = {}
        B, C, H, W = x.shape
        batch_index = torch.arange(B)
        gt_vu = (gt_vu * H).int().long()
        pred_qual = x[batch_index, :, gt_vu[:, 0], gt_vu[:, 1]]
        pred_qual = pred_qual.squeeze().sigmoid()
        loss_qual = self.loss_qual(gt_qual=gt_qual,
                                   pred_qual=pred_qual)
        losses['loss_qual'] = loss_qual
        # print(float(loss_qual.detach().cpu()))
        return losses


    def loss_qual(self, pred_qual, gt_qual):
        loss_qual = F.binary_cross_entropy(pred_qual, gt_qual, reduction='mean')
        return loss_qual


    def simple_test(self, x, gt_vu):
        bz, c, w, h = x.shape
        # gt_vu = gt_vu.reshape(-1, 1, 2)
        # assert w == h
        # gt_vu_vox = (gt_vu * w).long()
        # gt_vu_vox = gt_vu_vox.repeat(1, c, 1)
        # x_sampled = batch_index_vu_value(x, gt_vu_vox)

        gt_vox = gt_vu * 2 - 1
        x_sampled = F.grid_sample(x, gt_vox.reshape(bz, 1, 1, -1), align_corners=True)
        x_sampled = x_sampled.squeeze(-1)
        x_sampled = x_sampled.transpose(1, 2) #  (bz , 1, 32)
        vu_feat = self.fc_p(gt_vu).unsqueeze(1)
        for i in range(5):
            out = vu_feat + self.fc_c[i](x_sampled)
            out = self.blocks[i](out)
        pred_qual =  self.fc_out(F.relu(out)).flatten().sigmoid()
        return pred_qual




@DETECTORS.register_module()
class AmodalGraspOnlyGraspNOMMDet(BaseDetector):
    def __init__(self,
                 **kwargs):
        super().__init__(None)
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)       # self.neck = FeatsFusion()
        # self.backbone = resnet50_without_fc()
        # self.head = GraspHead()
        self.conv = nn.Conv2d(256, 1, 1)
        # self.neck = Protonet(9)
        # self.conv = nn.Conv2d(512, 1, 1)
        loss_mask = dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        self.loss = build_loss(loss_mask)
    def forward_train(self,
                      img,
                      img_metas,
                      **kwargs):
        feats = self.backbone(img)
        feats = (feats['0'], feats['1'], feats['2'], feats['3'])
        # x = self.neck(feats[2])
        x = feats[2] # (bz , 512, 32, 32)
        # x = self.conv(F.relu(x)).squeeze(1).sigmoid()
        print(x[:, 0].sigmoid().max())
        loss_hm = F.binary_cross_entropy(x[:, 0].sigmoid(), kwargs['gt_hm'])
        losses = dict(loss_hm=loss_hm)
        return losses

    def simple_test(self,
                    img,
                    **kwargs):
        feats = self.backbone(img)
        feats = (feats['0'], feats['1'], feats['2'], feats['3'])
        # x = self.neck(feats)
        x = feats[2] # (bz , 512, 32, 32)
        # x = self.conv(x).squeeze(1).sigmoid()
        return x[:, 0].sigmoid()

        # grasp_losses = self.head.forward_train(
        #     x=x,
        #     gt_vu=kwargs['gt_grasp_vu_on_obj'],
        #     gt_qual=kwargs['gt_grasp_qual'],
        #     # gt_quat=kwargs['gt_grasp_quat'],
        #     # gt_width=kwargs['gt_grasp_width']
        #     )
        # return grasp_losses



    def aug_test(self, imgs, img_metas, **kwargs):
        pass
    def extract_feat(self, imgs):
        pass
    # def simple_test(self,
    #                   img,
    #                   img_metas,
    #                   **kwargs):
    #     feats = self.backbone(img)
    #     feats = (feats['0'], feats['1'], feats['2'], feats['3'])
    #     x = self.neck(feats)
    #     pred_qual = self.head.simple_test(
    #         x=x,
    #         gt_vu=kwargs['gt_grasp_vu_on_obj'],
    #         )
    #     return pred_qual