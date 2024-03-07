# -*- coding: utf-8 -*-
#
# @Authors: Lukas Burget, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com

import argparse
import logging
import os
import time
import tqdm

import kaldi_io
import numpy as np
import onnxruntime
import soundfile as sf

import features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            logger.info('Start: {}: '.format(self.name))

    def __exit__(self, type, value, traceback):
        if self.name:
            logger.info('End:   {}: Elapsed: {} seconds'.format(self.name, time.time() - self.tstart))
        else:
            logger.info('End:   {}: '.format(self.name))


def get_embedding(fea, model, label_name=None, input_name=None, backend='pytorch'):
    if backend == 'onnx':
        return model.run([label_name],
                  {input_name: fea.astype(np.float32).transpose()
                  [np.newaxis, :, :]})[0].squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='', help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--weights', required=True, type=str, default=None, help='path to pretrained model weights')
    parser.add_argument('--ndim', required=False, type=int, default=64, help='dimensionality of features')
    parser.add_argument('--embed-dim', required=False, type=int, default=256, help='dimensionality of the emb')
    parser.add_argument('--in-wav-scp', required=True, type=str, help='input list of files')
    parser.add_argument('--out-ark-fn', required=True, type=str, help='output embedding file')
    parser.add_argument('--backend', required=False, default='onnx', choices=['onnx'],
                        help='backend that is used for x-vector extraction')

    args = parser.parse_args()

    print(args.gpus)

    device = ''

    model, label_name, input_name = '', None, None

    if args.backend == 'onnx':
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        model = onnxruntime.InferenceSession(args.weights, sess_options)
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
    else:
        raise ValueError('Wrong combination of --model/--weights/--model_file '
                         'parameters provided (or not provided at all)')

    with open(args.out_ark_fn, 'wb') as ark_file:
        for one_line in open(args.in_wav_scp, "rt"):
            utt_id, wav_path = one_line.strip().split(" ")
            with Timer(f'Processing file {utt_id}'):
                signal, samplerate = sf.read(wav_path)
                if samplerate == 8000:
                    noverlap = 120
                    winlen = 200
                    window = features.povey_window(winlen)
                    fbank_mx = features.mel_fbank_mx(
                        winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=3700, htk_bug=False)
                elif samplerate == 16000:
                    noverlap = 240
                    winlen = 400
                    window = features.povey_window(winlen)
                    fbank_mx = features.mel_fbank_mx(
                        winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
                else:
                    raise ValueError(f'Only 8kHz and 16kHz are supported. Got {samplerate} instead.')

                LC = 150
                RC = 149

                np.random.seed(3)  # for reproducibility
                signal = features.add_dither((signal*2**15).astype(int))

                print(f"Finished the feature extracting {signal.shape}")

                seg = signal
                if seg.shape[0] > 0.01*samplerate:  # process segment only if longer than 0.01s
                    # Mirror noverlap//2 initial and final samples
                    seg = np.r_[seg[noverlap // 2 - 1::-1],
                                seg, seg[-1:-winlen // 2 - 1:-1]]
                    fea = features.fbank_htk(seg, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
                    fea = features.cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)

                    data = fea
                    xvector = get_embedding(
                        data, model, label_name=label_name, input_name=input_name, backend=args.backend)

                    key = utt_id
                    if np.isnan(xvector).any():
                        logger.warning('NaN found, not processing: {}'.format(key))
                    else:
                        kaldi_io.write_vec_flt(ark_file, xvector, key=key)

