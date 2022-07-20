import torch
from torch import nn
import torch.nn.functional as F
import timm
from torchvision.models import resnet50, vgg16

class DINO(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True).to(device).eval()
        self.layer_dims = [768] * 4
        self.layers = [11, 8, 5, 2]
        self.spatial_sizes = [14] * 4
        self.images_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.images_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        print("Loaded DINO model")

    def get_intermediate_layers(self, inputs, cls=False, attn=False):
        ## after augmentation. between -1 to 1.
        inputs = inputs*0.5 + 0.5 # normalize to [0, 1]
        inputs = F.interpolate(inputs, size=(224, 224), mode='area') 
        centered_inputs = (inputs - self.images_mean) / self.images_std

        if not attn:
            image_features = self.model.get_intermediate_layers(centered_inputs, n=12)
            if not cls:
                image_features = [image_features[l][:, 1:].view(-1, 14, 14, 768).permute(0, 3, 1, 2) for l in self.layers]
            else:
                image_features = [image_features[l][:, 0].view(-1, 768, 1, 1) for l in self.layers]        
        else:
            image_features = self.model.get_last_selfattention(centered_inputs)[:, :, 1:,1:]
            image_features = torch.max(image_features, dim=1)[0]
            image_features = [None, None, None, image_features]
        return image_features
    

class ResNet50Both(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        if 'dino' in model_name.lower():
            model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=True).to(device).eval()
            print("Loaded ResNet50-DINO Model")
        else:
            model = resnet50(pretrained=True).to(device).eval()
            print("Loaded ResNet50 Model")

        self.model = nn.Module()
        all_layers = list(model.children())
        self.model.layer0 = nn.Sequential(*all_layers[:5])
        self.model.layer1 = nn.Sequential(*all_layers[5])
        self.model.layer2 = nn.Sequential(*all_layers[6])
        self.model.layer3 = nn.Sequential(*all_layers[7])
        
        self.layer_dims = [2048, 1024, 512, 256]
        self.spatial_sizes = [7, 14, 28, 56]
        
        self.images_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.images_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def get_intermediate_layers(self, inputs, layers=None, cls=None, attn=None):
        ## after augmentation. between -1 to 1.
        inputs = inputs*0.5 + 0.5 # normalize to [0, 1]
        inputs = F.interpolate(inputs, size=(224, 224), mode='area') 
        centered_inputs = (inputs - self.images_mean) / self.images_std

        out0 = self.model.layer0(centered_inputs)
        out1 = self.model.layer1(out0)
        out2 = self.model.layer2(out1)
        out3 = self.model.layer3(out2)
        image_features = [out3, out2, out1, out0]
        return image_features
    
class VGG16(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features.to(device).eval()
        self.layer0 = torch.nn.Sequential()
        self.layer1 = torch.nn.Sequential()
        self.layer2 = torch.nn.Sequential()
        self.layer3 = torch.nn.Sequential()

        self.layer_dims = [512, 512, 256, 128]
        self.spatial_sizes = [7, 14, 28, 56]
        
        self.images_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.images_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        N_slices = 4
        for x in range(10):
            self.layer0.add_module(str(x), vgg_pretrained_features[x])
        for x in range(10, 17):
            self.layer1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(17, 24):
            self.layer2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(24, 31):
            self.layer3.add_module(str(x), vgg_pretrained_features[x])

    def get_intermediate_layers(self, inputs, layers=None):
        ## after augmentation. between -1 to 1.
        inputs = inputs*0.5 + 0.5 # normalize to [0, 1]
        inputs = F.interpolate(inputs, size=(224, 224), mode='area') 
        centered_inputs = (inputs - self.images_mean) / self.images_std

        out0 = self.layer0(centered_inputs)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        image_features = [out3, out2, out1, out0]
        return image_features

class DeiT(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).to(device).eval()
        self.layer_dims = [768] * 4
        self.layers = [11, 8, 5, 2]
        self.spatial_sizes = [14] * 4
        self.images_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.images_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        print("Loaded DeiT Model")

    def get_intermediate_layers(self, inputs, n=12, cls=False, attn=None):
        ## after augmentation. between -1 to 1.
        inputs = inputs*0.5 + 0.5 # normalize to [0, 1]
        inputs = F.interpolate(inputs, size=(224, 224), mode='area') 
        centered_inputs = (inputs - self.images_mean) / self.images_std

        x = self.model.patch_embed(centered_inputs)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.model.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        image_features = []
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            image_features.append(x)
        if not cls:
            image_features = [image_features[l][:, 1:].view(-1, 14, 14, 768).permute(0, 3, 1, 2) for l in self.layers]
        else:
            image_features = [image_features[l][:, 0] for l in self.layers]
        return image_features

        
class PretrainedFE(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.model = DINO(device) if model_name == 'DINO' else (
            ResNet50DINO(model_name, device) if 'resnet50' in model_name.lower() else (
            DeiT(device) if model_name == 'DeiT' else None
            )
        )

        self.layer_dims = self.model.layer_dims
        self.spatial_sizes = self.model.spatial_sizes


    def get_intermediate_layers(self, inputs, layers=None, cls=None, attn=None):
        return self.model.get_intermediate_layers(inputs, cls=cls, attn=attn)
