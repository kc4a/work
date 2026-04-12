#!/usr/bin/env python3
"""
IoT 设备识别 — 多粒度行为表示比较框架
主入口

用法:
    python main.py --stage all          # 运行全部 5 个阶段
    python main.py --stage 1            # 仅生成 packet-level CSV
    python main.py --stage 2            # 仅运行五组主实验
    python main.py --stage 3            # 仅运行参数敏感性分析
    python main.py --stage 4            # 仅运行特征重要性分析
    python main.py --stage 5            # 仅汇总结果

    python main.py --group a --model rf
    python main.py --group b --model xgb --window 300 --step 300
    python main.py --group c --model rf  --seq_len 20 --seq_step 20
    python main.py --group d --model lstm --seq_len 20 --seq_step 20
    python main.py --group e --model rf  --sample_step 10 --hist_window 30
"""
import argparse
import logging
import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import (
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_STEP,
    DEFAULT_SEQ_LENGTH, DEFAULT_SEQ_STEP,
    DEFAULT_GROUP_E_SAMPLE_STEP, DEFAULT_GROUP_E_HIST_WINDOW,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="IoT 设备识别多粒度实验框架")

    parser.add_argument("--stage", type=str, default=None,
                        help="运行阶段: 1/2/3/4/5/all")
    parser.add_argument("--group", type=str, default=None,
                        help="单独运行实验组: a/b/c/d")
    parser.add_argument("--model", type=str, default="rf",
                        help="模型: rf/xgb/lstm/gru")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE,
                        help="窗口大小 (秒)")
    parser.add_argument("--step", type=int, default=None,
                        help="窗口步长 (秒) 或序列步长 (包数)")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LENGTH,
                        help="序列长度 (包数)")
    parser.add_argument("--seq_step", type=int, default=None,
                        help="序列步长 (包数)")
    parser.add_argument("--sample_step", type=int, default=DEFAULT_GROUP_E_SAMPLE_STEP,
                        help="Group E 采样步长 (秒)")
    parser.add_argument("--hist_window", type=int, default=DEFAULT_GROUP_E_HIST_WINDOW,
                        help="Group E 历史窗口大小 (秒)")
    parser.add_argument("--cross_dataset", action="store_true",
                        help="运行跨数据集泛化评估")
    parser.add_argument("--force", action="store_true",
                        help="强制重新解析 PCAP")

    args = parser.parse_args()

    # 延迟导入，避免无参调用时 import 慢
    from src.preprocessing.pcap_loader import parse_pcap_to_packet_csv
    from experiments.run_experiments import (
        run_group_a, run_group_b, run_group_c, run_group_d,
        run_group_e, run_group_e1,
        run_parameter_sensitivity_window,
        run_parameter_sensitivity_sequence,
        run_parameter_sensitivity_group_e,
        run_parameter_sensitivity_group_e1,
        run_feature_importance,
        aggregate_results,
        run_all,
    )

    # ── 跨数据集评估 ─────────────────────────
    if args.cross_dataset:
        from experiments.cross_dataset_eval import run_cross_dataset_all
        run_cross_dataset_all()
        return

    # ── 按阶段运行 ───────────────────────────
    if args.stage:
        stage = args.stage.lower()
        if stage == "all":
            run_all()
        elif stage == "1":
            parse_pcap_to_packet_csv(force=args.force)
        elif stage == "2":
            for mt in ("rf", "xgb"):
                run_group_a(mt)
            for mt in ("rf", "xgb"):
                run_group_b(mt)
            for mt in ("rf", "xgb"):
                run_group_c(mt)
            for rt in ("lstm", "gru"):
                run_group_d(rt)
            for mt in ("rf", "xgb"):
                run_group_e(mt)
            for mt in ("rf", "xgb"):
                run_group_e1(mt)
        elif stage == "3":
            run_parameter_sensitivity_window()
            run_parameter_sensitivity_sequence()
            run_parameter_sensitivity_group_e()
            run_parameter_sensitivity_group_e1()
        elif stage == "4":
            run_feature_importance()
        elif stage == "5":
            aggregate_results()
        else:
            parser.error(f"未知阶段: {stage}")
        return

    # ── 按单组运行 ───────────────────────────
    if args.group:
        group = args.group.lower()
        if group == "a":
            run_group_a(args.model)
        elif group == "b":
            w_step = args.step if args.step else args.window
            run_group_b(args.model, window_size=args.window, step=w_step)
        elif group == "c":
            s_step = args.seq_step if args.seq_step else (args.step if args.step else args.seq_len)
            run_group_c(args.model, seq_length=args.seq_len, step=s_step)
        elif group == "d":
            s_step = args.seq_step if args.seq_step else (args.step if args.step else args.seq_len)
            run_group_d(args.model, seq_length=args.seq_len, step=s_step)
        elif group == "e" or group == "e0":
            run_group_e(args.model,
                        sample_step=args.sample_step,
                        hist_window=args.hist_window)
        elif group == "e1":
            run_group_e1(args.model,
                         sample_step=args.sample_step,
                         hist_window=args.hist_window)
        else:
            parser.error(f"未知实验组: {group}")
        return

    # ── 无参数时显示帮助 ─────────────────────
    parser.print_help()


if __name__ == "__main__":
    main()
