## Model Performance Evaluation (`evaluate.py`)

This script is used to evaluate the model's performance on a test set. It can operate in two modes:

-   **`pair`**: Calculates pairwise accuracy. 
-   **`ranking`**: Calculates ranking accuracy. 

**Pair-wise Sample**

We set path1's image is better than path2's image for simplicity.

```json
[
    {
        "prompt": ".....",
        "path1": ".....",
        "path2": "....."
    },
    {
        "prompt": ".....",
        "path1": ".....",
        "path2": "....."
    },
  ...
]
```

**Rank-wise Sample**

```json
[
    {
        "id": "005658-0040",
        "prompt": ".....",
        "generations": [
            "path to image1",
            "path to image2",
            "path to image3",
            "path to image4"
        ],
        "ranking": [
            1,
            2,
            5,
            3
        ]
    },
  ...
]
```

### Usage

```bash
python evaluate/evaluate.py \
  --test_json /path/to/your/test_data.json \
  --config_path config/HPSv3_7B.yaml \
  --checkpoint_path checkpoints/HPSv3_7B/model.pth \
  --mode pair \
  --batch_size 8 \
  --num_processes 8
```

**Arguments:**

-   `--test_json`: (Required) Path to the JSON file containing evaluation data.
-   `--config_path`: (Required) Path to the model's configuration file.
-   `--checkpoint_path`: (Required) Path to the model checkpoint.
-   `--mode`: The evaluation mode. Can be `pair` or `ranking`. (Default: `pair`)
-   `--batch_size`: Batch size for inference. (Default: 8)
-   `--num_processes`: Number of parallel processes to use. (Default: 8)

---

## Reward Benchmarking (`benchmark.py`)

This script is used to run inference with a reward model over one or more folders of images. It calculates a reward score for each image based on its corresponding text prompt (expected in a `.txt` file with the same name). The script then outputs statistics (mean, std, min, max) for each folder and saves the detailed results to a JSON file.

It supports multiple reward models through the `--model_type` argument.

### Usage

The script is run using `argparse`. Below is a command-line example:

```bash
python evaluate/benchmark.py \
  --config_path config/HPSv3_7B.yaml \
  --checkpoint_path checkpoints/HPSv3_7B/model.pth \
  --model_type hpsv3 \
  --image_folders /path/to/images/folder1 /path/to/images/folder2 \
  --output_path ./benchmark_results.json \
  --batch_size 16 \
  --num_processes 8
```

**Arguments:**

-   `--config_path`: (Required) Path to the model's configuration file.
-   `--checkpoint_path`: (Required) Path to the model checkpoint.
-   `--model_type`: The reward model to use. Choices: `hpsv3`, `hpsv2`, `imagereward`. (Default: `hpsv3`)
-   `--image_folders`: (Required) One or more paths to folders containing the images to benchmark.
-   `--output_path`: (Required) Path to save the output JSON file with results.
-   `--batch_size`: Batch size for processing. (Default: 16)
-   `--num_processes`: Number of parallel processes to use. (Default: 8)
-   `--num_machines`: For distributed inference, the total number of machines. (Default: 1)
-   `--machine_id`: For distributed inference, the ID of the current machine. (Default: 0)

