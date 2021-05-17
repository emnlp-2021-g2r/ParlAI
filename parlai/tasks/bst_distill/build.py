import os
from parlai.core import build_data


def build(opt):
    version = 'v0.1'
    build_data.mark_done(
        os.path.join(opt['datapath'], 'bst_distill'),
        version_string=version,
    )
