# Image Generation Module

This module is designed for generating images from text prompts using various pretrained diffusion models. It supports parallel generation across multiple GPUs and can be extended to include new models easily.

## File Structure

-   `gen_images_from_prompt.py`: The main script for running the image generation process. It reads prompts from a JSON file and handles command-line arguments.
-   `generator.py`: Contains the core `Generator` class, which manages the model pipelines and distributes the generation tasks across different devices.
-   `utils/pipelines.py`: Defines the configurations for all supported pretrained models. This is where you can add or modify model parameters.
-   `utils/utils.py`: Contains helper functions for initializing `diffusers` pipelines and interacting with model APIs.

## How to Use

To generate images, run the main script with the required arguments.

### Basic Command

```bash
python gen_images_from_prompt.py \
    --json_path /path/to/your/prompts.json \
    --out_dir /path/to/your/output_directory \
    --pipeline_name sd_xl_pipe flux_schnell_pipe
```

### Command-Line Arguments

-   `--json_path` (required): Path to a JSON file containing a list of prompts. Each item in the list should be an object with a `"caption"` key. 

    - For generating images according to real images, you should specify `"image_file"` which is the original image path, and `"aspect_ratio"` of this image. The specific height and width will be adjusted according to model's best practice resolution.

    - For generating images from prompt only, you should specify `"save_name"`, `"height"` and `"width"`
    **Example `prompts.json` format:**
    ```json
    [
      {
        "image_file": "1.jpg",
        "caption": "A beautiful landscape painting of a mountain range at sunset.",
        "aspect_ratio": 0.5,
      },
      {
        "image_file": "2.jpg",
        "caption": "A close-up photo of a red rose with water droplets.",
        "aspect_ratio": 1.0,
      },
      {
        "image_file": "3.jpg",
        "caption": "An astronaut riding a horse on Mars, digital art.",
        "aspect_ratio": 1.77,
      }
    ]
    ```
-   `--out_dir` (required): The root directory where generated images will be saved. A subdirectory will be created for each pipeline.
-   `--pipeline_name` (required): One or more pipeline configuration names to use for generation. These names must correspond to the `PipelineParam` variable names defined in `utils/pipelines.py`.
-   `--num_devices`: The number of GPU devices to use for generation. Defaults to `8`.
-   `--batch_size`: The batch size per device. Defaults to `1`.
-   `--num_machine`: The total number of machines used in a distributed setup. Defaults to `1`.
-   `--machine_id`: The ID of the current machine in a distributed setup. Defaults to `0`.
-   `--enable_availabel_check`: If set, the script will first run a quick check on a small batch to ensure each pipeline can be loaded and run without errors.
-   `--reverse`: If set, the order of the specified pipelines will be reversed.

## How to Add a New Model

You can easily add a new text-to-image model by configuring it in the `utils/pipelines.py` file.

1.  **Open `utils/pipelines.py`**.
2.  **Import `PipelineParam`** if it's not already imported.
3.  **Create a new `PipelineParam` instance** for your model. Define the following parameters:
    -   `pipeline_name`: The model's path on the Hugging Face Hub or a local directory.
    -   `generation_path`: The name of the subdirectory where the output images will be saved.
    -   `pipeline_type`: The type of pipeline, e.g., `'t2i'` (text-to-image) or `'t2v'` (text-to-video). Defaults to `'t2i'`.
    -   `pipe_init_kwargs`: A dictionary of arguments required for initializing the model pipeline (e.g., `{"torch_dtype": torch.float16}`).
    -   `generation_kwargs`: A dictionary of arguments for the generation process (e.g., `{"guidance_scale": 7.0, "num_inference_steps": 28}`).
    -   `base_resolution`: The base resolution the model was trained on (e.g., `1024`).
    -   `force_aspect_ratio`: Optionally force a specific aspect ratio (e.g., `1` for square images).

    **Example:**

    ```python
    from pydantic import BaseModel, Field
    import torch

    class PipelineParam(BaseModel):
        # ... (class definition)

    # Add your new model configuration
    my_new_model_pipe = PipelineParam(
            pipeline_name='organization/my-cool-model',
            generation_path=f'generation/my_cool_model',
            pipe_init_kwargs={
                "torch_dtype": torch.float16,
            },
            base_resolution=1024,
            generation_kwargs={
                "guidance_scale": 5.0,
                "num_inference_steps": 30,
            }
        )
    ```

4.  **Run the generation script** using the name of your new `PipelineParam` variable in the `--pipeline_name` argument.

```bash
python gen_images_from_prompt.py --pipeline_name my_new_model_pipe ...
```
