from torch import Tensor, nn
from os.path import join as pjoin


class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, **kwargs) -> None:
        super().__init__()

        njoints = datamodule.njoints

        data_name = datamodule.name
        print("in BaseMetrics, data_name:", data_name)

        # None = legacy configs without TYPE (load full stack). [] = inference-only (no T2M/MM deps).
        metric_types = cfg.METRIC.get("TYPE")
        skip_evaluators = metric_types is not None and len(metric_types) == 0

        if data_name in ["humanml3d", "kit", "motionx", "tomato"] and not skip_evaluators:
            if metric_types is None or "TM2TMetrics" in metric_types:
                from .t2m import TM2TMetrics

                self.TM2TMetrics = TM2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                    njoints=njoints,
                )
            if (metric_types and "M2TMetrics" in metric_types) or cfg.model.params.task == "m2t":
                from .m2t import M2TMetrics

                self.M2TMetrics = M2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    w_vectorizer=datamodule.hparams.w_vectorizer,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            # Always loaded for training/eval when any metrics run (loads T2M evaluators).
            from .mm import MMMetrics

            self.MMMetrics = MMMetrics(
                cfg=cfg,
                dataname=data_name,
                mm_num_times=cfg.METRIC.MM_NUM_TIMES,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                njoints=njoints,
            )
            if metric_types and "TemosMetric" in metric_types:
                from .compute import ComputeMetrics

                self.TemosMetric = ComputeMetrics(
                    njoints=njoints,
                    jointstype=cfg.DATASET.JOINT_TYPE,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            if metric_types and "TMRMetrics" in metric_types:
                from .tmr import TMRMetrics

                self.TMRMetrics = TMRMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                    threshold_selfsim_metrics=0.95,
                )

        if not skip_evaluators and metric_types and "MRMetrics" in metric_types:
            from .mr import MRMetrics

            self.MRMetrics = MRMetrics(
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            )
        if not skip_evaluators and metric_types and "PredMetrics" in metric_types:
            from .m2m import PredMetrics

            self.PredMetrics = PredMetrics(
                cfg=cfg,
                njoints=njoints,
                jointstype=cfg.DATASET.JOINT_TYPE,
                dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                task=cfg.model.params.task,
            )
