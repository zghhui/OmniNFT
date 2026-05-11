from generator import Generator
import json
import os
import torch
import gc
from utils.pipelines import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="生成图片")
    parser.add_argument(
        "--json_path",
        type=str,
        help="json路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="输出目录",
    )
    parser.add_argument("--num_devices", type=int, default=8, help="设备数量")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--num_machine", type=int, default=1, help="机器数量")
    parser.add_argument("--machine_id", type=int, default=0, help="机器id")
    parser.add_argument(
        "--pipeline_name", type=str, nargs="+", default=None, help="pipeline名称"
    )
    parser.add_argument("--enable_availabel_check", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    num_devices = args.num_devices
    pipeline_params = [globals()[f"{name}_pipe"] for name in args.pipeline_name]

    if args.reverse:
        pipeline_params = pipeline_params[::-1]

    # first check all pipeline
    if args.enable_availabel_check:
        print(f"Checking {len(pipeline_params)} pipelines")
        for pipeline_param in pipeline_params:
            generator = Generator(
                pipe_name=pipeline_param.pipeline_name,
                pipe_type=pipeline_param.pipeline_type,
                pipe_init_kwargs=pipeline_param.pipe_init_kwargs,
                num_devices=num_devices,
            )

            with open(args.json_path, "r") as f:
                entries = json.load(f)
            info_dict = entries[: args.batch_size]
            generator.generate(
                info_dict,
                os.path.join(args.out_dir, pipeline_param.generation_path),
                batch_size=args.batch_size,
                num_processes=num_devices,
                seed=42,
                weight_dtype=pipeline_param.pipe_init_kwargs["torch_dtype"],
                generation_kwargs=pipeline_param.generation_kwargs,
                base_resolution=pipeline_param.base_resolution,
                force_aspect_ratio=pipeline_param.force_aspect_ratio,
            )
            del generator
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Finished Checking {pipeline_param.pipeline_name}")

    for pipeline_param in pipeline_params:
        generator = Generator(
            pipe_name=pipeline_param.pipeline_name,
            pipe_type=pipeline_param.pipeline_type,
            pipe_init_kwargs=pipeline_param.pipe_init_kwargs,
            num_devices=num_devices,
        )

        with open(args.json_path, "r") as f:
            entries = json.load(f)

        for i in range(args.num_machine):
            start_idx = i * len(entries) // args.num_machine
            end_idx = (
                (i + 1) * len(entries) // args.num_machine
                if i != args.num_machine - 1
                else len(entries)
            )
            if i == args.machine_id:
                info_dict = entries[start_idx:end_idx]

        info_dict = sorted(info_dict, key=lambda x: x["aspect_ratio"])

        print(f"Generating {len(info_dict)} images")
        generator.generate(
            info_dict,
            os.path.join(args.out_dir, pipeline_param.generation_path),
            batch_size=args.batch_size,
            num_processes=num_devices,
            seed=42,
            weight_dtype=pipeline_param.pipe_init_kwargs["torch_dtype"],
            generation_kwargs=pipeline_param.generation_kwargs,
            base_resolution=pipeline_param.base_resolution,
            force_aspect_ratio=pipeline_param.force_aspect_ratio,
        )

        print(f"Finished generating {pipeline_param.pipeline_name}")

        for pipeline in generator.pipelines:
            pipeline.to("cpu")
        del generator
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
