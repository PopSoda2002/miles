import miles.utils.external_utils.command_utils as U
import os

MODEL_NAME = "Qwen3-4B"
CUDA_VISIBLE_DEVICES = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
# SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE=True
# os.environ["SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE"] = SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE
WANDB_API_KEY = "a37f4796e6205800c4212556a38e1319b5f144b7"

def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    wandb_args = (
        "--use-wandb "
        "--wandb-project miles-lora "
        "--wandb-group lora1-chunk16_True "
        f"--wandb-key {WANDB_API_KEY} "
    )

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {3000 if U.get_env_enable_infinite_run() else 60} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 2 " "--sglang-decode-log-interval 1000 " "--sglang-enable-metrics "

    fsdp_args = (
        # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
        # "--fsdp-full-params "  # Uncomment this line to enable full params mode
        # Set the bucket size for weight update
        "--update-weight-buffer-size 536870912 "  # 512MB
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.71 "  # loose threshold at 60 step
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
        "--colocate "
        "--train-backend fsdp "
        "--loss-type sft_loss "
        "--disable-compute-advantages-and-returns "
        "--debug-train-only "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{sglang_args} "
        f"{wandb_args} "
        f"{fsdp_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=2,
        megatron_model_type=None,
    )


if __name__ == "__main__":
    prepare()
    execute()
