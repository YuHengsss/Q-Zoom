# qwen-vl-finetune scripts

| File | Purpose |
|---|---|
| `zero2.json`, `zero3.json`, `zero3_offload.json` | DeepSpeed configurations for various memory budgets. |
| `sft.sh`, `sft_7b.sh`, `sft_32b.sh` | Stock supervised fine-tuning launchers from the upstream Qwen-VL repo (kept as a starting point for non-Q-Zoom SFT). |

For Q-Zoom training pipelines, see `examples/stage{1,2,3}_train_eval/`
in the repository root, which set the additional `--enable_twig`,
`--twig_K`, `--twig_T`, `--roi_*` flags expected by `train_qwen.py`.
