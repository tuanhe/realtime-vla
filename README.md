# Running VLA in Real Time

This project provides accelerated inference kernels of the Pi0 model from [OpenPI](https://github.com/Physical-Intelligence/openpi) project.

The inference time for one set of observations (10 flow steps, empty prompt) on RTX 4090 (max boosted clock 2.79GHz) is as follows:

| 1 view | 2 views | 3 views |
|---|---|---|
| 20.0ms | 27.3ms | 36.8ms |

## How to Use

The intended usage is to directly copy the pi0_infer.py file into your project. The usage is:
```python
converted_checkpoint = pickle.load(open('converted_checkpoint.pkl', 'rb'))

from pi0_infer import Pi0Inference

infer = Pi0Inference(converted_checkpoint, number_of_images, length_of_trajectory)
infer.forward(
   normalized_observation_image_bfloat16, # (number_of_images, 3, 224, 224)
   observation_state_bfloat16, # (32,)
   diffusion_input_noise_bfloat16, # (length_of_trajectory, 32)
)
```

where you should first convert your JAX checkpoint to the required weight buffers by
```bash
python3 convert_from_jax.py \
   --jax_path /path/to/checkpoint/folder\
   --output converted_checkpoint.pkl\
   --prompt "your task prompt"
```

The code is specifically tuned on RTX 4090, CUDA 12.6, but it should work on similar platforms so long as torch and triton themselves work.

## Checking Performance

You can check the inference time on you local machine by
```bash
python3 benchmark.py --num_views 2 --prompt_len 0 --chunk_size 63
```


## Acknowledgements

This project is developed based on Physical Intelligence's [OpenPI](https://github.com/Physical-Intelligence/openpi) project.


## Citation
If you want, you can cite this work with: