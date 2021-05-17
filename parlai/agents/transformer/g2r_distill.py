import random
from parlai.core.params import ParlaiParser
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from parlai.core.torch_agent import Output
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
from parlai.core.metrics import AverageMetric
from parlai.utils.fp16 import FP16SafeCrossEntropy
import parlai.utils.logging as logging


class G2RDistilLoss(torch.nn.Module):
    def __init__(self, distill_coef=0.5, fp16=True):
        super().__init__()
        self.distill_coef = distill_coef
        if fp16:
            self.main_loss = FP16SafeCrossEntropy(reduction='none')
        else:
            self.main_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target, mask, cand_loss_masks, cand_inds, cand_batch_inds):
        # Distill loss
        cand_logits = torch.nn.functional.log_softmax(input + (mask - 1.) * 1e5)
        distill_losses = -(target * cand_logits)
        distill_loss = (distill_losses * mask).sum(dim=-1)

        # Main loss
        cand_inds = torch.cat(cand_inds).cuda()
        cand_batch_inds = cand_batch_inds.cuda()
        cand_loss_masks = cand_loss_masks.cuda()

        num_flat_pos = cand_batch_inds.size(0)
        flat_pos_input = input[cand_batch_inds, cand_inds].unsqueeze(-1)
        flat_neg_inputs = input[cand_batch_inds][:, num_flat_pos:]
        flat_inputs = torch.cat([flat_pos_input, flat_neg_inputs], dim=-1)
        main_losses = self.main_loss(flat_inputs, torch.zeros_like(cand_batch_inds))

        main_loss = (main_losses * cand_loss_masks).sum() / (cand_loss_masks.sum() + 1e-6)
        # HACK for matching dimension
        main_loss = main_loss * torch.ones_like(distill_loss)

        loss = main_loss * (1. - self.distill_coef) + distill_loss * self.distill_coef
        return loss


class G2RDistillMixin:
    """Mixin for using G2R Distill."""
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('G2R Distill Arguments')
        agent.add_argument(
            '--distill-temperature',
            type=float,
            default=1.0,
            help='temperature for distill',
        )
        agent.add_argument(
            '--distill-loss-coef',
            type=float,
            default=0.1,
            help="coef for distill loss",
        )
        agent.add_argument(
            '--response-path',
            type=str,
            help='path or response set'
        )
        agent.add_argument(
            '--num-random-negatives',
            type=int,
            default=512,
        )
        agent.add_argument(
            '--max-candidate-per-instance',
            type=int,
            default=10,
        )
        agent.add_argument(
            '--use-cand-loss-mask',
            action='store_true',
            help='If given, use candidate loss mask'
        )
        return parser

    def __init__(self, opt, shared=None):
        # Override opt: candidates -> "batch-all-cands"
        opt["candidates"] = "batch-all-cands"
        super().__init__(opt, shared)
        self.train_criterion = G2RDistilLoss(opt['distill_loss_coef'])

        if shared:
            self.candidates = shared["candidates"]
            self.candidate_vecs = shared["candidate_vecs"]
        else:
            with open(opt["response_path"]) as f:
                cands = [line.strip() for line in f]
            self.candidates = cands
            self.candidate_vecs = self._make_candidate_vecs(cands)

    def share(self):
        shared = super().share()
        shared["candidates"] = self.candidates
        shared["candidate_vecs"] = self.candidate_vecs
        return shared

    def _build_candidates(self, batch, source, mode):
        if mode == "train":
            # Ignore source and just use candidate
            warn_once(
                '[ Executing {} mode with candidate distill mode ]'
                ''.format(mode)
            )
            cands = []
            flat_cands_vecs = []
            cands_to_id = {}
            cands_idxs = []
            cand_loss_masks = []
            cand_batch_idxs = []
            scores = []

            use_cand_loss_mask = self.opt["use_cand_loss_mask"]

            for i, candidates in enumerate(batch.candidates):
                enum_candidates = list(enumerate(candidates))
                if len(enum_candidates) > self.opt["max_candidate_per_instance"]:
                    enum_candidates = random.sample(
                        enum_candidates, self.opt["max_candidate_per_instance"])

                _cands_idxs = []
                _scores = []
                for j, cand in enum_candidates:
                    if cand not in cands_to_id:
                        cands.append(cand)
                        cands_to_id[cand] = len(cands_to_id)
                        flat_cands_vecs.append(batch.candidate_vecs[i][j])
                    _cands_idxs.append(cands_to_id[cand])
                    _scores.append(batch.candidate_scores[i][j])
                    cand_batch_idxs.append(i)

                    candidate_masks = batch.candidate_masks
                    if candidate_masks is None or (not use_cand_loss_mask):
                        mask = 1.
                    else:
                        mask = candidate_masks[i][j]
                    cand_loss_masks.append(mask)
                cands_idxs.append(torch.LongTensor(_cands_idxs))
                scores.append(torch.FloatTensor(_scores))

            neg_idxs = []
            random_negative_idxs = np.random.choice(
                len(self.candidate_vecs),
                self.opt["num_random_negatives"],
                replace=False,
            )
            for global_negative_idx in random_negative_idxs:
                cand = self.candidates[global_negative_idx]
                if cand not in cands_to_id:
                    cands.append(cand)
                    cands_to_id[cand] = len(cands_to_id)
                    flat_cands_vecs.append(self.candidate_vecs[global_negative_idx])
                    neg_idxs.append(len(cands_to_id) - 1)

            flat_cands_vecs, _ = self._pad_tensor(flat_cands_vecs)
            cand_batch_idxs = torch.LongTensor(cand_batch_idxs)
            cand_loss_masks = torch.FloatTensor(cand_loss_masks)
            return cands, flat_cands_vecs, cands_idxs, cand_batch_idxs, cand_loss_masks, neg_idxs, scores
        else:
            return super()._build_candidates(batch, source, mode)

    def _build_target(self, batch, num_cands, cands_idxs, neg_inds, candidate_scores):
        bsz = len(batch.candidate_scores)
        target = torch.zeros((bsz, num_cands), dtype=torch.float)
        mask = torch.zeros((bsz, num_cands), dtype=torch.float)
        mask[:, neg_inds] = 1.

        for batch_idx, scores in enumerate(candidate_scores):
            logits = scores / self.opt["distill_temperature"]
            probs = torch.softmax(logits, dim=-1)
            target[batch_idx, cands_idxs[batch_idx]] = torch.FloatTensor(probs)
            mask[batch_idx, cands_idxs[batch_idx]] = 1.

        if self.use_cuda:
            target = target.cuda()
            mask = mask.cuda()
        return target, mask

    def _get_batch_train_metrics(self, scores, targets):
        """
        Get fast metrics calculations if we train with batch candidates.
        Specifically, calculate accuracy ('train_accuracy'), average rank, and mean
        reciprocal rank.
        """
        # TODO: Currently using top-score example as label.
        bsz = scores.size(0)
        targets = targets.max(dim=1)[1]
        nb_ok = (scores.max(dim=1)[1] == targets).float()
        self.record_local_metric('train_accuracy', AverageMetric.many(nb_ok))
        # calculate mean_rank
        target_scores = scores[
            torch.arange(bsz, device=scores.device, dtype=torch.long),
            targets,
        ]
        above_dot_prods = scores - target_scores.view(-1, 1)
        ranks = (above_dot_prods > 0).float().sum(dim=1) + 1
        mrr = 1.0 / (ranks + 0.00001)
        self.record_local_metric('rank', AverageMetric.many(ranks))
        self.record_local_metric('mrr', AverageMetric.many(mrr))

    def train_step(self, batch):
        self._maybe_invalidate_fixed_encs_cache()
        if batch.text_vec is None and batch.image is None:
            return
        self.model.train()
        self.zero_grad()

        cands, cand_vecs, cand_inds, cand_batch_inds, cand_loss_masks, neg_inds, cand_scores = self._build_candidates(
            batch, source=self.candidates, mode='train'
        )
        try:
            scores = self.score_candidates(batch, cand_vecs)
            target, mask = self._build_target(batch, len(cand_vecs), cand_inds, neg_inds, cand_scores)
            loss = self.train_criterion(scores, target, mask, cand_loss_masks, cand_inds, cand_batch_inds)
            self.record_local_metric('mean_loss', AverageMetric.many(loss))
            loss = loss.mean()
            self.backward(loss)
            self.update_params()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                logging.error(
                    'Ran out of memory, skipping batch. '
                    'if this happens frequently, decrease batchsize or '
                    'truncate the inputs to the model.'
                )
                return Output()
            else:
                raise e

        # Get train predictions
        self._get_batch_train_metrics(scores, target)
        return Output()
