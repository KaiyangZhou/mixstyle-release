import sys
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from baselines.common.distributions import make_pdtype, _matching_fc
from baselines.common.input import observation_input
ds = tf.contrib.distributions

from coinrun.config import Config

def mixstyle_layer(x, alpha=0.3, eps=1e-6):
    # x: (batch, height, width, channel)
    mu, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    sig = tf.math.sqrt(var + eps)
    mu = tf.stop_gradient(mu)
    sig = tf.stop_gradient(sig)
    x_normed = (x - mu) / sig

    idxs = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_idxs = tf.random.shuffle(idxs)
    mu2 = tf.gather(mu, shuffled_idxs)
    sig2 = tf.gather(sig, shuffled_idxs)

    lmda = tf.distributions.Beta(alpha, alpha).sample([tf.shape(x)[0], 1, 1, 1])
    mu_mix = mu * lmda + mu2 * (1 - lmda)
    sig_mix = sig * lmda + sig2 * (1 - lmda)
    x_out = x_normed * sig_mix + mu_mix

    return x_out

def impala_cnn(images, depths=[16, 32, 32]):
    use_batch_norm = Config.USE_BATCH_NORM == 1

    slow_dropout_assign_ops = []
    fast_dropout_assign_ops = []

    def dropout_openai(out, rate, name):
        out_shape = out.get_shape().as_list()
        var_name = 'mask_{}'.format(name)
        batch_seed_shape = out_shape[1:]
        batch_seed = tf.get_variable(var_name, shape=batch_seed_shape, initializer=tf.random_uniform_initializer(minval=0, maxval=1), trainable=False)
        batch_seed_assign = tf.assign(batch_seed, tf.random_uniform(batch_seed_shape, minval=0, maxval=1))
        dout_assign_ops = [batch_seed_assign]
        curr_mask = tf.sign(tf.nn.relu(batch_seed[None,...] - rate))
        curr_mask = curr_mask * (1.0 / (1.0 - rate))
        out = out * curr_mask
        return out, dout_assign_ops

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same')
        if use_batch_norm:
            out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=True)
        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value
        out = tf.nn.relu(inputs)
        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    if Config.MIXSTYLE:
        print('Do MixStyle')

    out = images
    for nr, depth in enumerate(depths):
        out = conv_sequence(out, depth)
        if nr in [0, 1] and Config.MIXSTYLE:
            #prob = tf.random_uniform([], minval=0, maxval=1)
            #do_mixstyle = tf.less(prob, 0.5) # do mixstyle with a probability
            #out = tf.cond(do_mixstyle, lambda: mixstyle_layer(out, alpha=Config.ALPHA), lambda: out)
            out = mixstyle_layer(out, alpha=Config.ALPHA)
    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)

    if Config.BETA >= 0:
        print("Creating VIB layer")
        params = tf.layers.dense(out, 256*2)
        mu, rho = params[:, :256], params[:, 256:]
        encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)

        with tf.variable_scope("info_loss"):
            prior = ds.Normal(0.0, 1.0)
            info_loss = tf.reduce_sum(tf.reduce_mean(
                ds.kl_divergence(encoding, prior), 0)) / np.log(2)

            # info_loss = tf.identity(info_loss, name="info_loss")
            tf.add_to_collection("INFO_LOSS", info_loss)
            # info_loss = tf.Print(info_loss, [info_loss])

        with tf.control_dependencies([info_loss]):
            batch_size = tf.shape(out)[0]
            # batch_size = tf.Print(batch_size, [tf.shape(out)])
            out = tf.reshape(
                    encoding.sample(Config.NR_SAMPLES),
                    shape=(batch_size * Config.NR_SAMPLES, 256))
            out_mean = mu

    elif Config.BETA_L2A >= 0:
        print("Creating L2A regularized layer")
        out = tf.layers.dense(out, 256)
        with tf.variable_scope("info_loss"):
            info_loss = tf.reduce_sum(tf.reduce_mean(tf.square(out), 0))
        tf.add_to_collection("INFO_LOSS_L2A", info_loss)

        with tf.control_dependencies([info_loss]):
            out = tf.identity(out)
        out_mean = out
    # elif Config.DROPOUT > 0:
    #     print("Creating Dropout layer")
    #     out_mean = tf.layers.dense(out, 256)
    #     out = tf.nn.dropout(out_mean, rate=Config.DROPOUT)
    elif Config.DROPOUT > 0:
        print("Creating Dropout layer")
        latent = tf.layers.dense(out, 256)
        out, fast_dropout_assign_ops = dropout_openai(latent, rate=Config.DROPOUT, name='fast')
        out_mean, slow_dropout_assign_ops = dropout_openai(latent, rate=Config.DROPOUT, name='slow')
    else:
        out = tf.layers.dense(out, 256)
        out_mean = out
    out = tf.identity(out, name="bottleneck_layer")
    tf.add_to_collection("BOTTLENECK_LAYER", out)
    out = tf.nn.relu(out)
    out_mean = tf.nn.relu(out_mean)
    return out, out_mean, slow_dropout_assign_ops, fast_dropout_assign_ops

def choose_cnn(images):
    arch = Config.ARCHITECTURE
    scaled_images = tf.cast(images, tf.float32) / 255.

    if arch == 'nature':
        raise NotImplementedError()
        out = nature_cnn(scaled_images)
    elif arch == 'impala':
        return impala_cnn(scaled_images)
    elif arch == 'impalalarge':
        return impala_cnn(scaled_images, depths=[32, 64, 64, 64, 64])
    else:
        assert(False)


class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)

        # So that I can compute the saliency map
        if Config.REPLAY:
            X = tf.placeholder(shape=(nbatch,) + ob_space.shape, dtype=np.float32, name='Ob')
            processed_x = X
        else:
            X, processed_x = observation_input(ob_space, nbatch)


        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            h, h_vf, slow_dropout_assign_ops, fast_dropout_assigned_ops = choose_cnn(processed_x)
            self.train_dropout_assign_ops = fast_dropout_assigned_ops
            self.run_dropout_assign_ops = slow_dropout_assign_ops

            # Noisy policy and value function for train
            if Config.BETA >= 0:
                pdparam = _matching_fc(h, 'pi', ac_space.n, init_scale=1.0, init_bias=0)
                pdparam = tf.reshape(pdparam, shape=(Config.NR_SAMPLES, -1, ac_space.n))
                pdparam = tf.transpose(pdparam, perm=[1,0,2])

                dists = ds.Categorical(logits=pdparam)
                self.pd_train = ds.MixtureSameFamily(
                    mixture_distribution=ds.Categorical(probs=[1./Config.NR_SAMPLES]*Config.NR_SAMPLES),
                    components_distribution=dists)
                self.pd_train.neglogp = lambda a: - self.pd_train.log_prob(a)
                self.vf_train = tf.reduce_mean(tf.reshape(fc(h, 'v', 1), shape=(Config.NR_SAMPLES, -1, 1)), 0)[:, 0]
            else:
                self.pd_train, _ = self.pdtype.pdfromlatent(h, init_scale=0.01)
                self.vf_train = fc(h, 'v', 1)[:, 0]

            if Config.SNI:
                assert Config.DROPOUT == 0
                assert not Config.OPENAI
                # Used with VIB: Noiseless pd_run and _both_ value functions
                print("Activating SNI (includes VF)")

                # Use deterministic value function for both as VIB for regression seems like a bad idea
                self.vf_run = self.vf_train = fc(h_vf, 'v', 1)[:, 0]

                # Have a deterministic run policy based on the mean
                self.pd_run, _ = self.pdtype.pdfromlatent(h_vf, init_scale=0.01)
            elif Config.SNI2:
                assert not Config.OPENAI
                # Used with Dropout instead of OPENAI modifier
                # 'RUN' versions are updated slowly, train versions updated faster, gradients are mixed
                print("Activating SNI2")

                # Deterministic bootstrap value... doesn't really matter but this is more consistent
                self.vf_run = fc(h_vf, 'v', 1)[:, 0]

                # Run policy based on slow changing latent
                self.pd_run, _ = self.pdtype.pdfromlatent(h_vf, init_scale=0.01)
                # Train is updated for each gradient update, slow is only updated once per batch
            elif Config.OPENAI:
                # Completely overwrite train versions as everything changes slowly
                # Train version is same as run version, both of which are slow
                self.pd_run, _ = self.pdtype.pdfromlatent(h_vf, init_scale=0.01)
                self.pd_train = self.pd_run
                self.vf_run = self.vf_train = fc(h_vf, 'v', 1)[:, 0]

                # Stochastic version is never used, so can set to ignore
                self.train_dropout_assign_ops = []
            else:
                # Plain Dropout version: Only fast updates / stochastic latent for VIB
                self.pd_run = self.pd_train
                self.vf_run = self.vf_train

                # For Dropout: Always change layer, so slow layer is never used
                self.run_dropout_assign_ops = []


        # Used in step
        a0_run = self.pd_run.sample()
        neglogp0_run = self.pd_run.neglogp(a0_run)
        self.initial_state = None

        def step(ob, update_frac, *_args, **_kwargs):
            if Config.REPLAY:
                ob = ob.astype(np.float32)
            a, v, neglogp = sess.run([a0_run, self.vf_run, neglogp0_run], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, update_frac, *_args, **_kwargs):
            return sess.run(self.vf_run, {X: ob})

        self.X = X
        self.processed_x = processed_x
        self.step = step
        self.value = value


def get_policy():
    use_lstm = Config.USE_LSTM
    
    if use_lstm == 1:
        raise NotImplementedError()
        policy = LstmPolicy
    elif use_lstm == 0:
        policy = CnnPolicy
    else:
        assert(False)

    return policy
