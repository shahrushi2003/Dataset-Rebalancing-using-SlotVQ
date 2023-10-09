from itertools import product
import torch
from tqdm.auto import tqdm
import numpy as np

class Slot_Collector():
    def __init__(self, num_slots, codebook_size):
        super().__init__()
        self.num_slots = num_slots
        self.codebook_size = codebook_size
        self.mult_array = torch.tensor([self.codebook_size**i for i in range(self.num_slots)][::-1]).reshape(-1, 1)
        self.codebooks = []
        self.slot_collector = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(self.codebook_size**self.num_slots):
            self.slot_collector[i] = list([])

    def collect(self, final_codes, image_paths, targets):
        slot_indices = torch.matmul(final_codes, self.mult_array).reshape(-1).tolist()
        for i in range(len(slot_indices)):
            self.slot_collector[slot_indices[i]].append((image_paths[i], targets[i]))
    
    def get_slots(self, loader, model):
        model.to(self.device)
        train_loop = tqdm(loader)
        for idx, (image, paths, target) in enumerate(train_loop):
            image = image.to(self.device)
            target = target.to(self.device)
            recon_combined, recons, masks, slots, vq_loss, engagement, final_codes = model.model(image)
            self.collect(final_codes.cpu(), paths, target)
            del recon_combined, recons, masks
        return self.slot_collector
    
def apply_temp(temp, probs):
    denom = np.sum(np.power(probs, 1/temp))
    new_probs = (np.power(probs, 1/temp))/denom
    return new_probs