"""
Load an agent trained with train_agent.py and
"""

import time

import tensorflow as tf
import numpy as np
from coinrun import setup_utils
import coinrun.main_utils as utils
from coinrun.config import Config
from coinrun import config
from coinrun import policies, wrappers
import imageio
import sys


# Import for saliency:
import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
slim=tf.contrib.slim
import saliency


titlesize = 30
# Boilerplate methods.
def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    # im = ((im + 1) * 127.5).astype(np.uint8)
    P.imshow(im)
    P.title(title, fontsize=titlesize)


def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')

    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title, fontsize=titlesize)


def ShowDivergingImage(grad, title='', percentile=99, ax=None):
    if ax is None:
        fig, ax = P.subplots()
    else:
        fig = ax.figure

    P.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
    fig.colorbar(im, cax=cax, orientation='vertical')
    P.title(title)


def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = np.asarray(im)
    return im / 127.5 - 1.0


# First one is None for the original image
models = [None, '0322_plain', None,
          '0401_l2a_l2w_uda', '0401_l2w_uda', '0401_l2w',
          None, '0327_l2a1e4', '0401_l2a1e4_noUda',
          ]

names = {
    '0322_plain': 'No Regularization',
    # '0322_plain_all': 'L2W + UDA + BN',
    '0401_l2w_uda': 'L2W + UDA',
    '0401_l2w': 'L2W',
    '0401_l2a_l2w_uda': 'L2W + L2A + UDA',
    '0327_l2a1e4': 'L2A + UDA',
    '0401_l2a1e4_noUda': 'L2A',
    # '0405__vib_l2w_uda_nn': 'VIB(modified) + L2W + UDA'
}
ROWS = 3
COLS = 3
UPSCALE_FACTOR = 10
# regular code:

mpi_print = utils.mpi_print

def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    act = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)

    return act

def enjoy_env_sess():
    # utils.setup_mpi_gpus()
    # setup_utils.setup_and_load({'restore_id': collecting_model})



    directory = './images/'
    directory_saliency = "./images_saliency"

    def create_saliency(model_idx, sess):
        graph = tf.get_default_graph()
        env = utils.make_general_env(1)
        env = wrappers.add_final_wrappers(env)
        agent = create_act_model(sess, env, 1)
        action_selector = tf.placeholder(tf.int32)
        gradient_saliency = saliency.GradientSaliency(graph, sess, agent.pd.logits[0][action_selector], agent.X)
        sess.run(tf.global_variables_initializer())

        # setup_utils.restore_file(models[model_idx])
        try:
            loaded_params = utils.load_params_for_scope(sess, 'model')
            if not loaded_params:
                print('NO SAVED PARAMS LOADED')
        except AssertionError as e:
            models[model_idx] = None
            return [None]*3
        return agent, gradient_saliency, action_selector

    orig_images_low = []
    orig_images_high = []
    filenames = []

    print("Loading files...")
    for idx, filename in enumerate(os.listdir(directory)):
        if len(filename) > 15 or os.path.isdir(os.path.join(directory, filename)):
            continue
        print('.', end='')
        img = imageio.imread(os.path.join(directory, filename))
        img = img.astype(np.float32)
        if filename.startswith('img_') and len(filename) < 15:
            filenames.append(filename)
            list_to_append = orig_images_low
        if filename.startswith('imgL_') and len(filename) < 15:
            list_to_append = orig_images_high
        list_to_append.append(img)

    list_of_images_lists = [] # First one for 0
    list_of_vmax_lists = []

    for idx, model_name in enumerate(models):
        if model_name is None:
            list_of_images_lists.append(None)
            list_of_vmax_lists.append(None)
            continue

        model_images = []
        vmaxs = []
        config.Config = config.ConfigSingle()
        setup_utils.setup_and_load(use_cmd_line_args=False, restore_id=model_name, replay=True)
        print("\nComputing saliency for Model {}\{}: {}...".format(idx, len(models)-1, names[model_name]))

        with tf.Session() as sess:
            agent, gradient_saliency, action_selector = create_saliency(idx, sess)
            for img in orig_images_low:
                print('.', end=''); sys.stdout.flush()
                action, values, state, _ = agent.step(np.expand_dims(img, 0), agent.initial_state, False)
                s_vanilla_mask_3d = gradient_saliency.GetSmoothedMask(img, feed_dict={'model/is_training_:0': False, action_selector: action[0]})
                s_vanilla_mask_grayscale, vmax = saliency.VisualizeImageGrayscale(s_vanilla_mask_3d)
                model_images.append(s_vanilla_mask_grayscale)
                vmaxs.append(vmax)

            list_of_images_lists.append(model_images)
            list_of_vmax_lists.append(vmaxs)

    print("\nMaking pretty images..")
    for idx, filename in enumerate(filenames):
        print('.', end=''); sys.stdout.flush()
        P.figure(figsize=(COLS * UPSCALE_FACTOR, ROWS * UPSCALE_FACTOR))
        ShowImage(orig_images_high[idx]/255, title="Original", ax=P.subplot(ROWS, COLS, 1))
        for row in range(ROWS):
            for col in range(COLS):
                model_idx = col + row * COLS
                if models[model_idx] is None:
                    continue
                ShowGrayscaleImage(
                    list_of_images_lists[model_idx][idx],
                    title=names[models[model_idx]] + "     Vmax: {:.2E}".format(list_of_vmax_lists[model_idx][idx]),
                    ax=P.subplot(ROWS, COLS, model_idx+1))
        P.savefig(os.path.join(directory_saliency, filename[:-4]+"_saliency.png"))
        P.close()
    print("\nDone")


def main():
    enjoy_env_sess()

if __name__ == '__main__':
    main()
