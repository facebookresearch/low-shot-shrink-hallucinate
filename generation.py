# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import analogy_generation


init_generator = dict(analogies=analogy_generation.init_generator)
do_generate = dict(analogies=analogy_generation.do_generate)

def get_generator(generator_name, generator_file):
    return do_generate[generator_name], init_generator[generator_name](generator_file)

