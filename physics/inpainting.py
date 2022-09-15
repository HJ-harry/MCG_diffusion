import os
import torch

class Inpainting():
    def __init__(self, img_heigth=512, img_width=512, mode='random', mask_rate=0.3, resize=False, device='cuda:0'):
        mask_path = './physics/mask_random{}.pt'.format(mask_rate)
        if os.path.exists(mask_path):
            self.mask = torch.load(mask_path).to(device)
        else:
            self.mask = torch.ones(img_heigth, img_width, device=device)
            self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0
            torch.save(self.mask, mask_path)

    def A(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)

    def A_dagger(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)
