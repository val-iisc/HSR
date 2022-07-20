import torch
from torch.nn import functional as F

from training.load_fe import PretrainedFE

@torch.no_grad()
def get_moments(training_set, model_name, device):
	model = PretrainedFE(model_name, device)
	mu = [torch.zeros(l).to(device) for l in model.layer_dims]
	sqmu = [torch.zeros(l).to(device) for l in model.layer_dims]
	
	for i, (img, _) in enumerate(training_set):
		#img between -1 and 1
		img = 2 * torch.Tensor((img / 255) if img.max() > 1 else img).to(device) - 1
		image_features = model.get_intermediate_layers(img.unsqueeze(0))
		mu = [(m * i + imf.mean([0,2,3])) / (i+1) 
				for m, imf in zip(mu, image_features)]
		sqmu = [(sm * i + imf.square().mean([0,2,3])) / (i+1) \
				for sm, imf in zip(sqmu, image_features)]
		
		del img, image_features

	var = [sm - m for (sm, m) in zip(sqmu, mu)]

	return mu, var
