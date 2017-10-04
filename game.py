"""
Visual Genome reference game.
"""

from argparse import ArgumentParser
from collections import defaultdict
import itertools
import json
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from reinforce import reinforce_episodic_gradients


class RSAModel(object):

    def __init__(self, vocab, vocab_weights, all_objects,
                 temperature=1.0, max_scene_size=10):
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.vocab_weights = vocab_weights
        self.all_objects = all_objects
        self.max_scene_size = max_scene_size

        self.temperature = temperature
        self._build_model()

        # Prepare outputs.
        self._speaker_distrs = self._infer_pragmatic_speaker()
        self._listener_distrs = self._infer_pragmatic_listener()

        self._speaker_samples = tf.multinomial(self._speaker_distrs, 1)[:, 0]
        self._listener_samples = tf.multinomial(self._listener_distrs, 1)[:, 0]

    @property
    def _session(self):
        return tf.get_default_session()

    def _build_model(self):
        ## Input placeholders.
        self._scene_size = tf.placeholder(tf.int32, shape=(None,),
                                          name="scene_size")
        self._scene_masks = tf.placeholder(tf.bool, shape=(None, len(self.all_objects)),
                                           name="scene_masks")
        # Referent observed by speaker.
        self._object_idxs = tf.placeholder(tf.int32, shape=(None,),
                                           name="object_idxs")
        # Utterance observed by listener.
        self._utterance_idxs = tf.placeholder(tf.int32, shape=(None,),
                                              name="utterance_idxs")

        self._embeddings = tf.get_variable("embeddings",
                                           shape=(len(self.all_objects), self.vocab_size),
                                           initializer=tf.random_normal_initializer(),
                                           dtype=tf.float32)
        self._listener_embeddings = tf.get_variable("listener_embeddings",
                shape=(self.vocab_size, len(self.all_objects)),
                initializer=tf.random_normal_initializer(),
                dtype=tf.float32)

        self._scene_masks_ = tf.expand_dims(tf.to_float(self._scene_masks), 2)

    def _infer_literal_speaker(self):
        weights = tf.nn.embedding_lookup(self._embeddings, self._object_idxs)
        return tf.nn.softmax(weights / self.temperature)

    def _infer_pragmatic_listener(self, utterances=None):
        return tf.nn.embedding_lookup(self._listener_embeddings, utterances if utterances is not None else self._utterance_idxs)
        if utterances is None:
            utterances = self._utterance_idxs

        embeddings = tf.expand_dims(self._embeddings, 0) # 1 * O * V
        # Mask embeddings: only consider objects on scene
        weights = tf.nn.softmax(embeddings * self._scene_masks_ / self.temperature) # B * 1 * 0
        weights = tf.transpose(weights, (0, 2, 1)) # => B * V * O
        weights /= tf.reduce_sum(weights, axis=2, keep_dims=True)

        # For each example, get the `utterance`-th row of the corresponding matrix
        flat = tf.reshape(weights, (-1, len(self.all_objects)))
        batch_size = tf.shape(self._scene_masks)[0]
        idxs = tf.range(batch_size) * self.vocab_size + utterances
        return tf.gather(flat, idxs)

    def _infer_pragmatic_speaker(self):
        return tf.nn.embedding_lookup(self._embeddings, self._object_idxs)
        embeddings = tf.expand_dims(self._embeddings, 0) # 1 * O * V
        # Mask embeddings: only consider objects on scene
        weights = tf.nn.softmax(embeddings * self._scene_masks_ / self.temperature) # B * 1 * 0
        weights = tf.transpose(weights, (0, 2, 1)) # => B * V * O
        weights /= tf.reduce_sum(weights, axis=2, keep_dims=True)
        weights = tf.transpose(weights, (0, 2, 1)) # => B * O * V
        weights /= tf.reduce_sum(weights, axis=2, keep_dims=True)

        # For each example, get the `object_idx`-th row of the corresponding matrix
        flat = tf.reshape(weights, (-1, self.vocab_size))
        batch_size = tf.shape(self._scene_masks)[0]
        idxs = tf.range(batch_size) * len(self.all_objects) + self._object_idxs
        return tf.gather(flat, idxs)

    def _make_scene_masks(self, scenes):
        mask = np.zeros((len(scenes), len(self.all_objects)))
        for i, scene in enumerate(scenes):
            for obj_idx in scene:
                mask[i, obj_idx] = 1
        return mask

    def _make_speaker_feed(self, scenes, referent_idxs):
        return {
            self._scene_masks: self._make_scene_masks(scenes),
            self._object_idxs: [scene[referent_idx] for scene, referent_idx
                                in zip(scenes, referent_idxs)]
        }

    def _make_listener_feed(self, scenes, utterances):
        return {
            self._scene_masks: self._make_scene_masks(scenes),
            self._utterance_idx: utterances
        }

    def infer_speaker(self, scenes, referent_idxs):
        """
        Predict a speaker distribution p(utterance | referent).

        Args:
            scenes: List of scene lists, each consisting of object IDs
            referent_idxs: Indexes into corresponding `scene`s denoting referents

        Returns:
            A `batch_size * vocab_size`-dimensional matrix over vocabulary
            items, which should be valid probability distributions.
        """
        return self._session.run(self._speaker_distrs,
                                 self._make_speaker_feed(scenes, referent_idxs))

    def infer_listener(self, scenes, utterances):
        """
        Predict a listener distribution p(referent | utterance).

        Args:
            scene:
            utterance: List of vocabulary idxs, one per utterance

        Returns:
            A `batch_size * len(scene)`-dimensional vector over vocabulary
            items, which should be valid probability distributions.
        """
        return self._session.run(self._listener_distrs,
                                 self._make_listener_feed(scenes, utterances))


def build_reinforce_model(rsa_model, vocabulary_weights):
    # Get articulation weight for each utterance.
    weights = tf.gather(vocabulary_weights, rsa_model._speaker_samples)
    l_distrs = rsa_model._infer_pragmatic_listener(utterances=tf.to_int32(rsa_model._speaker_samples))
    l_samples = tf.to_int32(tf.multinomial(l_distrs, 1)[:, 0])
    matches = tf.equal(rsa_model._object_idxs, l_samples)
    unscaled_rewards = tf.to_float(matches)
    rewards = unscaled_rewards * weights

    s_gradients = reinforce_episodic_gradients([rsa_model._speaker_distrs],
                                               [rsa_model._speaker_samples],
                                               rewards)
    l_gradients = reinforce_episodic_gradients([l_distrs], [l_samples],
                                               rewards)

    return (unscaled_rewards, rewards), s_gradients, l_gradients


def main(args):
    vocabulary = np.arange(args.vocab_size)
    vocab_weights = np.random.uniform(size=len(vocabulary)).astype(np.float32)#np.ones(len(vocabulary), dtype=np.float32)#

    with open(args.data_path, "r") as data_f:
        scenes = json.load(data_f)
    all_objects = sorted(set(itertools.chain.from_iterable(scenes)))
    print("%i objects" % len(all_objects))
    obj2idx = {obj: idx for idx, obj in enumerate(all_objects)}
    scenes = [[obj2idx[obj] for obj in scene] for scene in scenes]

    temperature = tf.placeholder(tf.float32, name="temperature", shape=())
    rsa_model = RSAModel(vocabulary, vocab_weights, all_objects,
                         temperature=temperature)
    (unscaled_rewards, rewards), s_gradients, l_gradients = \
            build_reinforce_model(rsa_model, vocab_weights)

    opt = tf.train.MomentumOptimizer(args.learning_rate, 0.9)
    train_op = opt.apply_gradients(list(s_gradients) + list(l_gradients))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    with sess.as_default():
        for i in range(25000):
            # Sample a batch of scenes.
            scene_idxs = np.random.choice(len(scenes), size=args.batch_size,
                                          replace=False)
            b_scenes = [scenes[idx] for idx in scene_idxs]
            b_referents = [np.random.choice(len(b_scene)) for b_scene in b_scenes]
            feed = rsa_model._make_speaker_feed(b_scenes, b_referents)

            decay_rate = 0.99
            decay_steps = 500
            feed[temperature] = 1. * decay_rate ** ((i + 1) / decay_steps)

            _, b_rewards, b_unscaled_rewards = \
                    sess.run((train_op, rewards, unscaled_rewards), feed)
            print("%05i\t%.05f\t%.05f" % (i, np.mean(b_rewards), np.mean(b_unscaled_rewards)))

        embs = sess.run(rsa_model._embeddings)

    inferred_labels = embs.argmax(axis=1)
    obj2label = dict(zip(all_objects, inferred_labels))
    label2objs = defaultdict(list)
    for obj, label in obj2label.items():
        label2objs[label].append(obj)
    pprint({idx: (weight, label2objs[idx]) for idx, weight in enumerate(vocab_weights)})


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument("--data_path", required=True)

    p.add_argument("--vocab_size", default=100, type=int)

    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--learning_rate", default=0.1, type=float)

    args = p.parse_args()
    main(args)
