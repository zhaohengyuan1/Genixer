1. Clone this repository and navigate to FERRET folder
```bash
git clone https://github.com/apple/ml-ferret
cd ml-ferret
```

2. Install Package
```Shell
conda create -n ferret python=3.10 -y
conda activate ferret
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install pycocotools
pip install protobuf==3.20.0
```

3. Install additional packages for training cases
```
pip install ninja
pip install flash-attn --no-build-isolation
```


## Train

FERRET is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`.

### Hyperparameters
We use a similar set of hyperparameters as LLaVA(Vicuna) in finetuning.  

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| FERRET-7B | 128 | 2e-5 | 3 | 2048 | 0 |
| FERRET-13B | 128 | 2e-5 | 3 | 2048 | 0 |

### Prepare Vicuna checkpoint and LLaVA's projector

Before you start, prepare our base model Vicuna, which is an instruction-tuned chatbot. Please download its weights following the instructions [here](https://github.com/lm-sys/FastChat#model-weights). Vicuna v1.3 is used in FERRET.

Then download LLaVA's first-stage pre-trained projector weight ([7B](https://huggingface.co/liuhaotian/llava-336px-pretrain-vicuna-7b-v1.3), [13B](https://huggingface.co/liuhaotian/llava-336px-pretrain-vicuna-13b-v1.3)).


### FERRET Training

The scripts are provided ([7B](experiments/ferret_7b_train.sh), [13B](experiments/ferret_13b_train.sh)).


## Evaluation

Please see this [doc](EVAL.md) for the details.

## Checkpoints
We extracted the `delta` between our pre-trained model and Vicuna. Please first download weights of Vicuna following the [previous instruction](#prepare-vicuna-checkpoint-and-llavas-projector). Then download our prepared offsets of weights: [7B](https://docs-assets.developer.apple.com/ml-research/models/ferret/ferret-7b/ferret-7b-delta.zip), [13B](https://docs-assets.developer.apple.com/ml-research/models/ferret/ferret-13b/ferret-13b-delta.zip) using `wget` or `curl`, and unzip the downloaded offsets. Lastly, apply the offset to the Vicuna's weight by running the following script:
```Shell
# 7B
python3 -m ferret.model.apply_delta \
    --base ./model/vicuna-7b-v1-3 \
    --target ./model/ferret-7b-v1-3 \
    --delta path/to/ferret-7b-delta
# 13B
python3 -m ferret.model.apply_delta \
    --base ./model/vicuna-13b-v1-3 \
    --target ./model/ferret-13b-v1-3 \
    --delta path/to/ferret-13b-delta
```

**Notices**: Apple's rights in the attached weight differentials are hereby licensed under the CC-BY-NC license. Apple makes no representations with regards to LLaMa or any other third party software, which are subject to their own terms.

Please refer to the next section about how to set up a local demo with pre-trained weight.

## Demo

To run our demo, you need to train FERRET and use the checkpoints locally. Gradio web UI is used. Please run the following commands one by one. 

#### Launch a controller
```Shell
python -m ferret.serve.controller --host 0.0.0.0 --port 10000
```

#### Launch a gradio web server.
```Shell
python -m ferret.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --add_region_feature
```

#### Launch a model worker

This is the worker that load the ckpt and do the inference on the GPU.  Each worker is responsible for a single model specified in `--model-path`.

```Shell
CUDA_VISIBLE_DEVICES=0 python -m ferret.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/FERRET-13B-v0 --add_region_feature
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...".  Now, refresh your Gradio web UI, and you will see the model you just launched in the model list.


<p align="center">
    <img src="figs/ferret_demo.png" width="105%"></a> <br>
    Example of Ferret Interactive Demo.
</p>
