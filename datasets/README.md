# Dataset Setup
Use `dataset_setup.py` to download datasets, for example:
```
python3 datasets/dataset_setup.py \
  --data_dir=~/data \
  --ogbg
```

This will require the same pip dependencies as `submission_runner.py`.

Some datasets require signing a form before downloading:

FastMRI:
Fill out form on https://fastmri.med.nyu.edu/ and run this script with the
links that are emailed to you for "knee_singlecoil_train" and
"knee_singlecoil_val".

ImageNet:
Register on https://image-net.org/ and run this script with the links to the
ILSVRC2012 train and validation images. Also, for the ImagenetV2 test set, the link was moved to huggingface. Since the script downloads using tfds, then tensorflow==4.9.2 must be downloaded to include this [update](https://github.com/tensorflow/datasets/pull/4848).

Note for tfds ImageNet, you may have to increase the max number of files allowed
open at once using `ulimit -n 8192`.

Note that in order to avoid potential accidental deletion, this script does NOT
delete any intermediate temporary files (such as zip archives) without a user
confirmation. Deleting temp files is particularly important for Criteo 1TB, as
there can be multiple copies of the dataset on disk during preprocessing if
files are not cleaned up. If you do not want any temp files to be deleted, you
can pass --interactive_deletion=false and then all files will be downloaded to
the provided --temp_dir, and the user can manually delete these after
downloading has finished.

Note that some functions use subprocess.Popen(..., shell=True), which can be
dangerous if the user injects code into the --data_dir or --temp_dir flags. We
do some basic sanitization in main(), but submitters should not let untrusted
users run this script on their systems.

## Librispeech
Librispeech is downloaded into the temporary folder /tmp/librispeech by default unless overwritten using the --temp_dir flag. To process the downloaded raw data, first unzip the following files in /tmp/librispeech:
```
train-clean-100.tar.gz, train-clean-360.tar.gz,
train-other-500.tar.gz, dev-clean.tar.gz,
dev-other.tar.gz, test-clean.tar.gz, test-other.tar.gz
```
The generated directories will be nested in /tmp/librispeech/LibriSpeech. Move those folders one level up to have /tmp/librispeech/train-clean-100, /tmp/librispeech/train-clean-360, etc.

Now, the tokenizer can be built and passed to the pre-processor.
### Training SPM Tokenizer
This step trains a simple sentence piece tokenizer over librispeech training data.
This tokenizer is then used in later preprocessing step to tokenize transcripts.
This command will generate `spm_model.vocab` file in `$DATA_DIR/librispeech`:
```bash
python3 librispeech_tokenizer.py --train --data_dir=$DATA_DIR/librispeech
```

The trained tokenizer can be loaded back to do sanity check by tokenizing + de-tokenizing a constant string:
```bash
python3 librispeech_tokenizer.py --data_dir=$DATA_DIR/librispeech
```

### Preprocessing Script
The preprocessing script will generate `.npy` files for audio data, `features.csv` which has paths to saved audio `.npy`, and `trans.csv` which has paths to `features.csv` and transcription data.

```bash
python3 librispeech_preprocess.py --raw_input_dir=/tmp/librispeech --output_dir={YOUR OUTPUT DIRECTORY} --tokenizer_vocab_path={WHERE THE TOKENIZER FILE IS SAVED}
```
## FastMRI
Download the links manually onto the experimentation server using curl. Make sure knee_singlecoil_train.tar, knee_singlecoil_val.tar and knee_singlecoil_test.tar are placed in the {datadir}/fastmri. Then, extract those compressed files to have
```
{datadir}/fastmri/knee_singlecoil_train
{datadir}/fastmri/knee_singlecoil_val
{datadir}/fastmri/knee_singlecoil_test
```
with each folder containing the h5 files.

## Criteo
Criteo is currently hosted on a WeTransfer server [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/). Go to the download link, click download, pause the download, copy the download link from downloads and then do the following:
```
export TEMP_DIR={ENTER YOUR DATA DIR}
mkdir $TEMP_DIR/criteo
cd $TEMP_DIR/criteo
curl -L -o criteo.zip "{ENTER YOUR LINK}" 
unzip criteo.zip
```
Please note that the temp_dir use above should be the same passed below.
```
cd {algorithmic-efficiency repo}
python datasets/dataset_setup.py --criteo \
    --criteo \
    --data_dir={your data directory} \
    --temp_dir={your temp directory as above}
```
This will unpack the day_*.gz files from temp directory to your data directory.

## Test Scripts
Below are scripts used to test the dataset access. Please note that these scripts are run inside a docker image inherited from fadyrezk/jax_learned_optimization docker image. Inside this image, the following was run:
```
conda install ffmpeg
pip install gputil psutil clu tensorflow-text tensorflow_addons jaxopt tensorflow_probability absl-py==1.0.0 pandas==1.3.5 protobuf==3.20.* six==1.16.0 scikit-learn==1.0.1 h5py==3.7.0 scikit_image==0.19.3 jraph==0.0.6.dev0 sentencepiece==0.1.97 sacrebleu==1.3.1 pydub==0.25.1
git clone https://github.com/FadyRezkGhattas/algorithmic-efficiency
cd algorithmic-efficiency
pip3 install -e '.[full]'
```
### OGBG (tested and working)
```
python3 submission_runner.py \
    --framework=jax \
    --workload=ogbg \
    --experiment_dir=~/ogbg_test/ \
    --experiment_name=ogbg_test \
    --submission_path=reference_algorithms/development_algorithms/ogbg/ogbg_jax/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/ogbg/tuning_search_space.json \
    --data_dir=/data0/

```
### LibriSpeech (tested and working)
```
python3 submission_runner.py \
    --framework=jax \
    --workload=librispeech_conformer \
    --experiment_dir=~/librispeech_conformer_test/ \
    --experiment_name=librispeech_conformer_test \
    --submission_path=reference_algorithms/development_algorithms/librispeech_conformer/librispeech_jax/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/librispeech_conformer/tuning_search_space.json \
    --data_dir=/data0/librispeech

```
### FastMRI (tested and working)
```
python3 submission_runner.py \
    --framework=jax \
    --workload=fastmri \
    --experiment_dir=~/fastmri_test/ \
    --experiment_name=fastmri_test \
    --submission_path=reference_algorithms/development_algorithms/fastmri/fastmri_jax/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/fastmri/tuning_search_space.json \
    --data_dir=/data0/fastmri
```
### ImageNet (tested and working)
```
python3 submission_runner.py \
    --framework=jax \
    --workload=imagenet_resnet \
    --experiment_dir=~/imagenet_test/ \
    --experiment_name=imagenet_test \
    --submission_path=reference_algorithms/development_algorithms/imagenet_resnet/imagenet_jax/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/imagenet_resnet/tuning_search_space.json \
    --data_dir=/data0/imagenet \
    --imagenet_v2_data_dir=/data0/imagenet
```

### Criteo (tested and working)
```
python3 submission_runner.py \
    --framework=jax \
    --workload=criteo1tb \
    --experiment_dir=~/criteo_test/ \
    --experiment_name=criteo_test \
    --submission_path=reference_algorithms/development_algorithms/criteo1tb/criteo1tb_jax/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/criteo1tb/tuning_search_space.json \
    --data_dir=/data0/criteo
```
### WMT

