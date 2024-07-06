# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and extract embeddings from input audio. 
Please pre-install "modelscope".
Usage:
    1. extract the embedding from the wav file.
        `python infer_sv.py --model_id $model_id --wavs $wav_path `
    2. extract embeddings from two wav files and compute the similarity score.
        `python infer_sv.py --model_id $model_id --wavs $wav_path1 $wav_path2 `
    3. extract embeddings from the wav list.
        `python infer_sv.py --model_id $model_id --wavs $wav_list `
"""
import json
import os
import sys
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..' % os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

parser = argparse.ArgumentParser(description='Extract speaker embeddings.')
parser.add_argument('--model_id', default='', type=str, help='Model id in modelscope')
parser.add_argument('--wavs', nargs='+', type=str, help='Wavs')
parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')
parser.add_argument('--pre_model', default=None, type=str, help='Local model dir')

with open("../../config/model.json", "r") as f:
    json_data = json.loads(f.read())
    (
        CAMPPLUS_VOX,
        CAMPPLUS_COMMON,
        ERes2Net_VOX,
        ERes2NetV2_COMMON,
        ERes2Net_COMMON,
        ERes2Net_base_COMMON,
        ERes2Net_Base_3D_Speaker,
        ERes2Net_Large_3D_Speaker,
        EPACA_CNCeleb,
        supports
    ) = json_data.values()


def main():
    args = parser.parse_args()
    # 定义参数  根据model id自动到modelscope上在线下载模型权重，然后用于推理
    args.model_id = "iic/speech_eres2net_sv_zh-cn_16k-common"
    args.wavs = [r"D:\asr_demo\dataset\data\data\data_thchs30\test\D4_750.wav",
                 r"D:\asr_demo\dataset\data\data\data_thchs30\test\D4_751.wav"]  # 长度是0或者2
    args.pre_model = r"D:\shuohuaren\3D_Speaker\model\speech_eres2net_sv_zh-cn_16k-common\pretrained_eres2net_aug.ckpt"
    #args.pre_model = None

    conf = None
    if args.pre_model is None:
        path = "../../model"
        args.local_model_dir = path
        assert isinstance(args.model_id, str) and \
               is_official_hub_path(args.model_id), "Invalid modelscope model id."
        if args.model_id.startswith('damo/'):
            args.model_id = args.model_id.replace('damo/', 'iic/', 1)
        assert args.model_id in supports, "Model id not currently supported."
        save_dir = os.path.join(args.local_model_dir, args.model_id.split('/')[1])
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        conf = supports[args.model_id]
        # download models from modelscope according to model_id
        cache_dir = snapshot_download(
            args.model_id,
            revision=conf['revision'],
        )
        cache_dir = pathlib.Path(cache_dir)

        embedding_dir = save_dir / 'embeddings'
        embedding_dir.mkdir(exist_ok=True, parents=True)

        # link
        download_files = ['examples', conf['model_pt']]
        for src in cache_dir.glob('*'):
            if re.search('|'.join(download_files), src.name):
                dst = save_dir / src.name
                try:
                    dst.unlink()
                except FileNotFoundError:
                    pass
                dst.symlink_to(src)

        pretrained_model = save_dir / conf['model_pt']

    else:
        conf = supports[args.model_id]  # 定义模型配置
        save_dir = "../../model"
        embedding_dir = os.path.split(args.pre_model)[0] + '/embeddings'
        # embedding_dir.mkdir(exist_ok=True, parents=True)
        pretrained_model = args.pre_model
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    if torch.cuda.is_available():
        msg = 'Using gpu for inference.'
        print(f'[INFO]: {msg}')
        device = torch.device('cuda')
    else:
        msg = 'No cuda device is detected. Using cpu.'
        print(f'[INFO]: {msg}')
        device = torch.device('cpu')

    # load model
    model = eval(conf['model'])  # 修改 添加eval 获取变量值
    embedding_model = dynamic_import(model['obj'])(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.to(device)
    embedding_model.eval()

    def load_wav(wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[['rate', str(obj_fs)]]
            )
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def compute_embedding(wav_file, save=True):
        # load wav
        wav = load_wav(wav_file)
        # compute feat
        feat = feature_extractor(wav).unsqueeze(0).to(device)
        # compute embedding
        with torch.no_grad():
            embedding = embedding_model(feat).detach().cpu().numpy()

        if save:
            if args.pre_model is None:

                save_path = embedding_dir / (
                        '%s.npy' % (os.path.basename(wav_file).rsplit('.', 1)[0]))
            else:
                save_path = embedding_dir + "/" + (
                        '%s.npy' % (os.path.basename(wav_file).rsplit('.', 1)[0]))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 不存在就会创建
            np.save(save_path, embedding)
            print(f'[INFO]: The extracted embedding from {wav_file} is saved to {save_path}.')

        return embedding

    # extract embeddings
    print(f'[INFO]: Extracting embeddings...')

    if args.wavs is None or len(args.wavs) == 2:
        if args.wavs is None:  # 远程路径add方式
            try:
                # use example wavs
                examples_dir = save_dir / 'examples'
                wav_path1, wav_path2 = list(examples_dir.glob('*.wav'))[0:2]
                print(f'[INFO]: No wavs input, use example wavs instead.')
            except:
                assert Exception('Invalid input wav.')
        else:
            # use input wavs
            wav_path1, wav_path2 = args.wavs

        embedding1 = compute_embedding(wav_path1)
        embedding2 = compute_embedding(wav_path2)

        # compute similarity score
        print('[INFO]: Computing the similarity score...')
        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        scores = similarity(torch.from_numpy(embedding1), torch.from_numpy(embedding2)).item()
        print('[INFO]: The similarity score between two input wavs is %.4f' % scores)
    elif len(args.wavs) == 1:
        # input one wav file
        if args.wavs[0].endswith('.wav'):
            # input is wav path
            wav_path = args.wavs[0]
            embedding = compute_embedding(wav_path)
        else:
            try:
                # input is wav list
                wav_list_file = args.wavs[0]
                with open(wav_list_file, 'r') as f:
                    wav_list = f.readlines()
            except:
                raise Exception('[ERROR]: Input should be wav file or wav list.')
            for wav_path in wav_list:
                wav_path = wav_path.strip()
                embedding = compute_embedding(wav_path)
    else:
        raise Exception('[ERROR]: Supports up to two input files')


if __name__ == '__main__':
    main()
