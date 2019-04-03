
import os
import json
import argparse

import numpy as np
import tensorflow as tf
from tensorpack import Inferencer, logger
from tensorpack.callbacks import (DataParallelInferenceRunner, ModelSaver,
                                  ScheduledHyperParamSetter)
from tensorpack.tfutils import get_model_loader, SaverRestore
from tensorpack.train import (SyncMultiGPUTrainerParameterServer, TrainConfig, launch_train_with_config)

import loader.loader as loader
from config import Config
from misc.utils import get_files
from model.graph import Model_NP_XY, Model_NP_DIST


####
class StatCollector(Inferencer, Config):
    """
    Accumulate output of inference during training.
    After the inference finishes, calculate the statistics
    """
    def __init__(self, prefix='valid'):
        super(StatCollector, self).__init__()
        self.prefix = prefix

    def _get_fetches(self):
        return self.train_inf_output_tensor_names

    def _before_inference(self):
        self.true_list = []
        self.pred_list = []

    def _on_fetches(self, outputs):
        pred, true = outputs
        self.true_list.extend(true)
        self.pred_list.extend(pred)
 
    def _after_inference(self):
        stat_dict = {}
        pred = np.array(self.pred_list)
        true = np.array(self.true_list)

        # have to get total number pixels for mean per pixel
        nr_pixels = np.ones(true[...,:1].shape).sum()

        # classification statistic
        pred_blb = pred[...,:1]
        true_blb = true[...,:1]
        pred_blb[pred_blb >  0.5] = 1.0
        pred_blb[pred_blb <= 0.5] = 0.0

        accuracy = (pred_blb == true_blb).sum() / nr_pixels
        inter = (pred_blb * true_blb).sum()
        total = (pred_blb + true_blb).sum()
        dice = 2 * inter / total

        stat_dict[self.prefix + '_acc' ] = accuracy
        stat_dict[self.prefix + '_dice'] = dice

        # regression statistic
        pred_xy = pred[...,-2:]
        true_xy = true[...,-2:]
        error = pred_xy - true_xy
        mae = np.sum(np.abs(error)) / nr_pixels
        mse = np.sum(error * error) / nr_pixels

        stat_dict[self.prefix + '_mae'] = mae
        stat_dict[self.prefix + '_mse'] = mse
        return stat_dict

###########################################
class Trainer(Config):   
    ####
    def get_datagen(self, mode='train', view=False):
        if mode == 'train':
            batch_size = self.train_batch_size
            augmentors = self.get_train_augmentors(view)
            data_files = get_files(self.train_dir, self.data_ext)
            data_generator = loader.train_generator
            nr_procs = self.nr_procs_train
        else:
            batch_size = self.infer_batch_size
            augmentors = self.get_valid_augmentors(view)
            data_files = get_files(self.valid_dir, self.data_ext)
            data_generator = loader.valid_generator
            nr_procs = self.nr_procs_valid

        # set nr_proc=1 for viewing to ensure clean ctrl-z
        nr_procs = 1 if view else nr_procs
        dataset = loader.DatasetSerial(data_files)
        datagen = data_generator(dataset,
                        shape_aug=augmentors[0],
                        input_aug=augmentors[1],
                        label_aug=augmentors[2],
                        batch_size=batch_size,
                        nr_procs=nr_procs)
        
        return datagen        
    ####
    def view_dataset(self, mode='train'):
        assert mode == 'train' or mode == 'valid', "Invalid view mode"
        datagen = self.get_datagen(mode=mode, view=True)
        loader.visualize(datagen, 4)
        return
    ####
    def run_once(self, nr_gpus, freeze, sess_init=None, save_dir=None):
        ####
        train_datagen = self.get_datagen(mode='train')
        valid_datagen = self.get_datagen(mode='valid')

        ###### must be called before ModelSaver
        if save_dir is None:
            logger.set_logger_dir(self.save_dir)
        else:
            logger.set_logger_dir(save_dir)
            
        callbacks=[
                ModelSaver(max_to_keep=200),
                ScheduledHyperParamSetter('learning_rate', self.lr_sched),
                ]
        ######

        # multi-GPU inference (with mandatory queue prefetch)
        infs = [StatCollector()]
        callbacks.append(DataParallelInferenceRunner(
                                valid_datagen, infs, list(range(nr_gpus))))

        ######
        steps_per_epoch = train_datagen.size() // nr_gpus

        MODEL_MAKER = Model_NP_XY if self.model_mode == 'np+xy' else Model_NP_DIST

        config = TrainConfig(
                    model           = MODEL_MAKER(freeze),
                    callbacks       = callbacks      ,
                    dataflow        = train_datagen  ,
                    steps_per_epoch = steps_per_epoch,
                    max_epoch       = self.nr_epochs ,
                )
        config.session_init = sess_init

        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpus))
        tf.reset_default_graph() # remove the entire graph in case of multiple runs
        return
    ####
    def run(self, nr_gpus):
        """
        Do the 2 phase training, both phases have the same nr_epochs and lr_sched
        Phase 1: finetune decoder portion (together with the first 7x7 in the encoder),
                 using the pretrained preact-resnet50 on ImageNet 
        Phase 2: unfreeze all layer, training whole, weigths taken from last epoch of
                 Phase 1
        """
        def get_last_chkpt_path(phase1_dir):
            stat_file_path = phase1_dir + '/stats.json'
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            chkpt_list = [epoch_stat['global_step'] for epoch_stat in info]
            last_chkpts_path = "%smodel-%d.index" % (phase1_dir, max(chkpt_list))
            return last_chkpts_path

        save_dir = self.save_dir + '/base/'
        self.train_batch_size = self.train_phase1_batch_size
        init_weights = get_model_loader(self.pretrained_preact_resnet50_path)
        self.run_once(nr_gpus, True, sess_init=init_weights, save_dir=save_dir)

        save_dir = self.save_dir + '/tune/'
        self.train_batch_size = self.train_phase2_batch_size
        phase1_last_model_path = get_last_chkpt_path(self.save_dir + '/base/')
        init_weights = SaverRestore(phase1_last_model_path, ignore=['learning_rate'])
        self.run_once(nr_gpus, False, sess_init=init_weights, save_dir=save_dir)
        return
    ####
####

###########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help="comma separated list of GPU(s) to use.")
    parser.add_argument('--view', help="view dataset, received either 'train' or 'valid' as input")
    args = parser.parse_args()

    trainer = Trainer()
    if args.view:
        trainer.view_dataset(args.view)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        nr_gpus = len(args.gpu.split(','))
        trainer.run(nr_gpus)
