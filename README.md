# cog-dolly-2

This repository is an implementation of [dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog). 

# Prerequisites 

* LLaMA weights. The weights for LLaMA have not yet been released publicly. To apply for access, fill out this Meta Research form.

* GPU machine. You'll need a Linux machine with an NVIDIA GPU attached and the NVIDIA Container Toolkit installed. If you don't already have access to a machine with a GPU, check out our guide to getting a GPU machine. This codebase currently assumes a single device with sufficient VRAM (>24GB) is available. If, instead, you have access to a multi-device environment, you can modify the code to distribute your model across devices. 

* Docker. You'll be using the Cog command-line tool to build and push a model. Cog uses Docker to create containers for models.

## Step 0: Install Cog

First, [install Cog](https://github.com/replicate/cog#install):

```
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

## Step 1: Set up weights

Run `cog build` and then use `./scripts/download_weights.py` to download the model weights:

```
chmod +x ./scripts/download_weights.py
cog run ./scripts/download_weights.py
```

This script will download the model weights to `./pretrained_weights/` and it will copy the tokenizer files to `./tokenizer`.

To use these weights, you need to comment out `pretrained_weights` in your `.dockerignore` file. When this is commented out,`pretrained_weights` aren't built into the resulting cog image.

However, we recommend converting the model to a `tensorizer` object rather than working with the raw torch weights. To do that, you can use the `scripts/tensorize_model.py` script:

```
chmod +x ./scripts/tensorize_model.py
cog run ./scripts/tensorize_model.py
```

This will load the model in fp16 and then use `tensorizer` to export it to `./tensorized_models/dolly-v2-12b-fp16.tensors`. 

To use these weights, comment out `tensorized_models` in `.dockerignore` and set `PATH_TO_TENSORIZER_WEIGHTS` in `predict.py` to `"./tensorized_models/dolly-v2-12b-fp16.tensors"`.

## Step 2: Run the model


You can run the model locally to test it:

```
cog predict -i prompt="Who was Dolly the sheep?" -i temperature=0.75 -i repetition_penalty=1.2
```

## Step 3: Create a model on Replicate

Go to [replicate.com/create](https://replicate.com/create) to create a Replicate model.

Make sure to specify "private" to keep the model private.

## Step 4: Configure the model to run on A100 GPUs

Replicate supports running models on a variety of GPUs. The default GPU type is a T4, but for best performance you'll want to configure your model to run on an A100.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

## Step 5: Push the model to Replicate

Log in to Replicate:

```
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 3:

```
cog push r8.im/username/modelname
```

[Learn more about pushing models to Replicate.](https://replicate.com/docs/guides/push-a-model)


## Step 6: Run the model on Replicate

Now that you've pushed the model to Replicate, you can run it from the website or with an API.

To use your model in the browser, go to your model page.

To use your model with an API, click on the "API" tab on your model page. You'll see commands to run the model with cURL, Python, etc.

To learn more about how to use Replicate, [check out our documentation](https://replicate.com/docs).