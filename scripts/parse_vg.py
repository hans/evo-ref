"""
Parse the Visual Genome objects JSON file into a simple usable format for our
purposes.
"""


from argparse import ArgumentParser
from collections import Counter
import itertools
import json
import sys


def iter_scenes(corpus_path):
    with open(corpus_path, "r") as corpus_f:
        vg = json.load(corpus_f)
        for scene in vg:
            objects = []
            for s_object in scene["objects"]:
                synsets = s_object["synsets"]
                if len(synsets) > 0:
                    objects.append(synsets[0])

            yield set(objects)


def main(args):
    scenes = list(iter_scenes(args.corpus_path))

    if args.freq_threshold > 0:
        c = Counter()
        for scene in scenes:
            c.update(scene)

        filtered_objs = set(k for k in c if c[k] >= args.freq_threshold)
        sys.stderr.write("Retaining %i of %i objects.\n"
                         % (len(filtered_objs), len(c)))
        for obj, count in c.most_common(25):
            sys.stderr.write("\t%s\t%i\n" % (obj, count))

        # Filter out low-freq items.
        filtered_scenes = []
        for scene in scenes:
            filtered_scene = scene & filtered_objs
            if len(filtered_scene) > 1:
                filtered_scenes.append(filtered_scene)
        scenes = filtered_scenes

    scenes = list(map(list, scenes))
    json.dump(scenes, sys.stdout)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("corpus_path")

    p.add_argument("--freq_threshold", type=int, default=0)

    args = p.parse_args()
    main(args)
