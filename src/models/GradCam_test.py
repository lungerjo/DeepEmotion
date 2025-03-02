import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
import numpy as np
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# from engine_animals.baselines1 import ERMClassifier
# from engine_miccai.base_classifier import ERMClassifier
# from engine_miccai import utils

class GradCamWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Instead of running the whole CNN, we use the custom method
        # that returns the target tensor for GradCAM.
        return self.model.get_gradcam_target(x)

# Alternatively, define a small module that acts as a target layer.
class GradCamTarget(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.get_gradcam_target(x)

class CacheDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.cache = {}

    def __len__(self):
        return len(self.cache)

    def update(self,idx,data):
        self.cache[idx] = data

    def setup_keys(self):
        self.keys = list(self.cache.keys())

    def __getitem__(self,idx):
        key = self.keys[idx]
        return self.cache[key]


class MaskTuneClassifier(ERMClassifier):
    def __init__(self, dataset, algo, seed=10, bias_balanced=False, train_only_on_unbiased=False,
                 num_classes=4, num_bias_classes=4, masking_threshold=2, gradcam_layer_depth = 2, args=None
                 ):
        super().__init__(dataset=dataset, algo=algo, seed=seed, args=args)
                        # bias_balanced=bias_balanced,
                        #  train_only_on_unbiased=train_only_on_unbiased, num_classes=num_classes,
                        #  num_bias_classes=num_bias_classes)

        self.masking_threshold = masking_threshold
        self.gradcam_layer_depth = gradcam_layer_depth
    
    def get_path(self, item):
        save_path = '/cluster/home/t130016uhn/SilverLiningResults/%s/%s' % (self.name, item)
        utils.create_folder(save_path)
        return save_path


    def setup_gradcam(self):
        
        # layers = [
        #     self.model.feature_extractor.resnet.layer1[-1].conv2,
        #     self.model.feature_extractor.resnet.layer2[-1].conv2,
        #     self.model.feature_extractor.resnet.layer3[-1].conv2,
        #     self.model.feature_extractor.resnet.layer4[-1].conv2,
        # ]

        # if self.gradcam_layer_depth > len(layers):
        #     raise ValueError('Invalid GradCAM Layer Depth')
        # elif 0 < self.gradcam_layer_depth < len(layers):
        #     target_layers = [layers[self.gradcam_layer_depth-1]]
        # elif self.gradcam_layer_depth == -1:
        #     target_layers = layers
        
        target_layers = [GradCamTarget(self.model)]
        self.heat_map_generator = XGradCAM(
            model=GradCamWrapper(self.model),
            target_layers=target_layers,
            # use_cuda=torch.cuda.is_available()
        )

    def setup_masked_train_dl(self,load_prev=False):
        path = self.get_path('cache_dataset.pth')

        if load_prev:
            cache_dataset = torch.load(path)
            self.masked_dl = torch.utils.data.DataLoader(cache_dataset, batch_size=self.dls['test'].batch_size, shuffle=True)
            return

        else:
            self.model.eval()
            cache_dataset = CacheDataset()
            batch_size = None
            for batch in tqdm(self.dls['train'], desc='Masking'):
                if batch_size is None:
                    batch_size = batch['img'].shape[0]

                img = batch['img'].to(self.device)
                y_r, y_b, idxes = batch['label'], batch['bias_label'], batch['idx']

                targets = [ClassifierOutputTarget(cl.item()) for cl in y_r]
                heat_map = self.heat_map_generator(img,targets=targets)

                heat_map =  torch.from_numpy(heat_map)
                heat_map = heat_map.unsqueeze(1).float()
                mean, std = heat_map.mean(), heat_map.std()
                thresh = mean + self.masking_threshold * std
                mask = (heat_map < thresh)*1.0
                img = img.cpu()
                masked_imgs = img * mask

                # ub_idx = y_b == 0
                # masked_imgs[ub_idx] = img[ub_idx]

                for idx, masked_img, label, bias_label in zip(idxes, masked_imgs, y_r, y_b):
                    cache_dataset.update(idx, {'img': masked_img, 'label': label, 'bias_label': bias_label})

            cache_dataset.setup_keys()
            torch.save(cache_dataset, path)
            tqdm.write('Masked Dataset Saved at %s'%path)
            self.masked_dl = torch.utils.data.DataLoader(cache_dataset, batch_size=batch_size, shuffle=True)

    def train_erm(self, num_epochs=10):
        self.fit(num_epochs=num_epochs, resume='none')
        self.setup(resume='best')
        self.save('best_erm')

    def finetune(self, num_epochs=1, finetune_lr=1e-5):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=finetune_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

        for epoch in range(num_epochs):
            train_res = self.train_one_epoch(self.masked_dl)
            valid_res = self.validate(self.dls['valid'])
            log_str = f'Finetune Epoch {epoch} ' + ' '.join([f'{k} {v:.4f}' for k, v in train_res.items()])
            log_str += ' Valid ' + ' '.join([f'{k} {v:.4f}' for k, v in valid_res.items()])
            tqdm.write(log_str)
            self.scheduler.step(valid_res['auc_ub'])

        self.save('best_finetune')

    def masktune(self, finetune_epochs=1,finetune_lr=1e-5, load_prev=False):
        self.setup(resume='best_erm')

        if not load_prev:
            self.setup_gradcam()
            self.setup_masked_train_dl()  # Generate masked data
        else:
            self.setup_masked_train_dl(load_prev=True)

        self.finetune(finetune_epochs,finetune_lr=finetune_lr)  # Finetune on masked data





def main():
    datasets = ['95_line','95_brightness','95_both']
    # datasets = ['95_brightness','95_both']
    # datasets = ['95_line',]
    # datasets = ['95_brightness',]
    # datasets = ['95_brightness',]
    # datasets = ['95_both',]

    # for seed in [20,30,40]:
    # for seed in [20,30]:
    # for seed in [30,]:
    for seed in [10,20,30]:
        for dataset in datasets:
            clf = MaskTuneClassifier(dataset=dataset, algo='masktune', seed=seed, num_classes=4, num_bias_classes=4,
                                    #  masking_threshold=2, gradcam_layer_depth=-1) #
                                    #  masking_threshold=2, gradcam_layer_depth=4) #
                                    masking_threshold=2, gradcam_layer_depth=2) #
                                    #  masking_threshold=3, gradcam_layer_depth=3) #

            # clf.train_erm(num_epochs=25)
            # clf.train_erm(num_epochs=15)
            # clf.test(resume='best_erm')
            # clf.masktune(finetune_epochs=1,finetune_lr=1e-5,load_prev=False)
            # clf.masktune(finetune_epochs=1,finetune_lr=1e-5,load_prev=True)
            # clf.masktune(finetune_epochs=1,load_prev=True)
            # clf.test_and_save(resume='best_finetune')
            # clf.test(resume='best_finetune')
            clf.test_and_save(resume='best_finetune',dl_name='test_counter')
            del(clf)