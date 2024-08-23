from inference_VQ_Diffusion import VQ_Diffusion
import os
import torch

import sys
sys.path.append(os.getcwd()+'/../Vqvae/')
from utils.midi_transfer import pianoroll2midi


class MidiGenerate(VQ_Diffusion):
    def __int__(self):
        super().__int__()

    def inference_generate_sample_with_class(self, text, truncation_rate, save_root, batch_size=2, cnt=1, fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['label'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top" + str(truncation_rate) + 'r',
                # sample_type='normal',
            )  # B x C x H x W

        # save results
        content = model_out['content']
        midi = pianoroll2midi(content[1])

        save_base_name = '{}'.format(str(cnt).zfill(6))
        save_path = os.path.join(save_root_, save_base_name + '.midi')
        midi.write(save_path)


if __name__ == "__main__":
    cfg_path = '/root/autodl-tmp/VqDiffusion/config/maestro-20230721.yaml'
    model_path = '/root/autodl-tmp/VqDiffusion/OUTPUT/maestro-20230721/checkpoint/'
    save_root = 'truncation_rate0.86_samples'

    cls, num = 3, 1000
    midi_generate = MidiGenerate(config=cfg_path, path=model_path)
    for i in range(cls):
        for j in range(num):
            midi_generate.inference_generate_sample_with_class(i, truncation_rate=0.86, save_root=save_root, cnt=j)