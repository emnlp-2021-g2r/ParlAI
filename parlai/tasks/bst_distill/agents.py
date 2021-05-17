#!/usr/bin/env python3
import copy
import os
from typing import Optional

from parlai.core.opt import Opt
from parlai.core.teachers import (
    ParlAIDialogTeacher,
)
from parlai.core.params import ParlaiParser
from .build import build


##################################################
#### Teacher for the BlendedSkillTalk Dataset ####
##################################################

def _processed_data_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    postfix = opt["train_data_postfix"]
    if dt == "train" and len(postfix) > 0:
        dt = f"{dt}-{postfix}"
    return os.path.join(opt['datapath'], 'bst_distill', dt + '.txt')


class BSTDistillTeacher(ParlAIDialogTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group('BSTDistillFlatTeacher arguments')
        agent.add_argument(
            '--train-data-postfix',
            type=str,
            default="",
        )
        return parser

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _processed_data_path(opt)
        super().__init__(opt, shared)

    def _setup_data(self, path):
        super()._setup_data(path)
        for episode in self.episodes:
            for msg in episode:
                label_candidates_scores = msg.get("label_candidates_scores")
                if label_candidates_scores is not None:
                    scores = [float(x) for x in label_candidates_scores.split("|")]
                    assert len(scores) == len(msg["label_candidates"])
                    msg.force_set("label_candidates_scores", scores)

                label_candidates_masks = msg.get("label_candidates_masks")
                if label_candidates_masks is not None:
                    masks = [float(x) for x in label_candidates_masks.split("|")]
                    assert len(masks) == len(msg["label_candidates"])
                    msg.force_set("label_candidates_masks", masks)


class DefaultTeacher(BSTDistillTeacher):
    pass
