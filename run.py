import argparse
import os
import torch
import numpy as np

from diffusers import StableDiffusion3Pipeline, FluxPipeline, AuraFlowPipeline
from diffusers import StochasticRFOvershotDiscreteScheduler
# from pipelines import FluxPipeline


def run(args):
    with open(args.prompt_file, 'r') as file:
        prompts = file.readlines()
    
    if args.model_type == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)
        guidance_scale = 7.0
    elif args.model_type == "flux":
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        guidance_scale = 3.5
    elif args.model_type == "auraflow": 
        pipe = AuraFlowPipeline.from_pretrained("fal/AuraFlow", torch_dtype=torch.float16)
        guidance_scale = 3.5
    pipe.enable_model_cpu_offload()
    
    if args.scheduler == 'overshoot':
        scheduler_config = pipe.scheduler.config
        scheduler = StochasticRFOvershotDiscreteScheduler.from_config(scheduler_config)
        overshot_func = lambda t, dt: t+dt
        exp_prefix = f"{args.scheduler}_c={str(args.c).zfill(4)}_use_att={args.use_att}"
        
        pipe.scheduler = scheduler
        pipe.scheduler.set_c(args.c)
        pipe.scheduler.set_overshot_func(overshot_func)
    elif args.scheduler == "euler":
        exp_prefix = f"{args.scheduler}"
    
    for i in range(len(prompts)):
        file_save_dir = os.path.join(args.exp_dir, "generated_image", f"num_steps={str(args.num_inference_steps).zfill(4)}", exp_prefix)
        os.makedirs(file_save_dir, exist_ok=True)
        img_save_path = os.path.join(file_save_dir, f"sample_{str(i).zfill(4)}.png")
        
        generator = torch.Generator(device='cuda')
        generator.manual_seed(args.seed)
        
        output = pipe(
            prompt=prompts[i],
            num_inference_steps=args.num_inference_steps,
            height=args.img_size,
            width=args.img_size,
            guidance_scale=guidance_scale,
            generator=generator,
            use_att=args.use_att, 
        )
        image = output.images[0]
        image.save(img_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", type=str, default="euler", 
                        help="scheduler to use")
    parser.add_argument("--c", type=float, default=2.0, help="c value for overshooting scheduler")
    parser.add_argument("--prompt_file", type=str, default="prompts.txt", 
                        help="file with prompts")
    parser.add_argument("--num_inference_steps", type=int, default=28, 
                        help="number of steps")
    parser.add_argument("--exp_dir", type=str, default="exps/flux", 
                        help="experiment directory")
    parser.add_argument("--model_type", type=str, default="flux", 
                        choices=["sd3", "flux", "auraflow"], help="model type")
    parser.add_argument("--use_att", action="store_true", help="use attention")
    parser.add_argument("--seed", type=int, default=10, help="seed")
    parser.add_argument("--img_size", type=int, default=1024, help="image size")
    args = parser.parse_args()
    
    run(args)
    