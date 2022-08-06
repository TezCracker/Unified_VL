# Unified_VL

Authors: Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal

File name - COCO Vision to Text.ipynb. 
Google colab link for the Coco Captioning  - https://colab.research.google.com/drive/1dog1qC5LEsl3IWMBUuok_ztpHEfDnSBI#scrollTo=U8l4RJ0XRPEm

#
File name - VQA.ipynb. 
Google colab link for the VQA - https://colab.research.google.com/drive/1EtjFWl_Mi2G865RKRK6evgwQmnFZBhqU#scrollTo=hlDZ0LkPMf48

#
File name - VQA_Results.ipynb. 
Google colab link for the VQA Result - https://colab.research.google.com/drive/1IgFcG6c4zgVMbQo9k8ZsCJEBhJLDslRP#scrollTo=GXB4QTHAoE57

We evaluate VL-adapter in a unified multi-task
For the image-text tasks, we use four diverse V&L datasets: VQAv2, GQA, NLVR2, and MSCOCO image captioning.

-- Instal process
--Install all python dependencies
pip install -r requirements.txt

--Download T5/BART checkpoint
python download_backbones.py

--For MSCOCO captioning evaluation 
python -c "import language_evaluation; language_evaluation.download('coco')"
     

--Train VL-T5 with adapters
./VL-T5/
 src/
modeling_t5.py modeling_bart.py                       <= VL-T5/VL-BART model classes
pretrain.py, pretrain_data.py, pretrain_model.py      <= pretraining
vqa.py, vqa_data.py vqa_model.py                      <= fine-tuning on downstream tasks 
multitask.py, multitask_data.py multiask_model.py     <= multitask learning on 7 downstream tasks
param.py                                              <= configuration
tokenization.py                                       <= custom tokenizer
utils.py, dist_utils.py                               <= utility functions
snap/                                                     <= store weight checkpoints
scripts/                                                  <= bash scripts for pretraining and finetuning


Image-text dataset
checkout the link [link](https://drive.google.com/file/d/1O_RU1iFh_sbItZCTkOHUrbVIQQ_89Djj/view?usp=sharing) to download the processed CLIP features.Better to use [gdrive](https://github.com/prasmussen/gdrive) to download it. Unzip the downloaded file and arrange the folders following the format which is shown in the "Code Structure."

when downloading the data from dgrive, Use the following commands.

gdrive download 1O_RU1iFh_sbItZCTkOHUrbVIQQ_89Djj


Extract the CLIP features
Please refer to feature_extraction for more details.

--Video-text dataset
Please go to [VALUE](https://github.com/VALUE-Leaderboard/DataRelease) to download the ViT processed data.

--Run different approaches
The following scripts can run every approach with the best hyper-parameters.

--Image dataset


Full fine-tuning
cd VL-T5/
 scripts/image/full_finetuning.sh 1

Single Adapter
cd VL-T5/
 scripts/image/single_adapter.sh 1

Multiple Adapters
cd VL-T5/
 scripts/image/multiple_adapters.sh 1

Hyperformer
cd VL-T5/
 scripts/image/hyperformer.sh 1

Single Compacter
cd VL-T5/
 scripts/image/single_compacter.sh 1

Multiple Compacters
cd VL-T5/
 scripts/image/multiple_compacters.sh 1

Single LoRA
cd VL-T5/
 scripts/image/single_lora.sh 1

Multiple LoRA
cd VL-T5/
 scripts/image/multiple_lora.sh 1

Single Prompt
cd VL-T5/
 scripts/image/single_prompt.sh 1

Multiple Prompts
cd VL-T5/
 scripts/image/multiple_prompts.sh 1


--Download Pre-trained models / Pre-extracted features
We host model checkpoints and features via google drive.
We recommend using [gdrive](https://github.com/prasmussen/gdrive) to download them.


--Pretrained Models
- Download snap/ from [Google Drive](https://drive.google.com/drive/folders/1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph?usp=sharing)

gdrive download 1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph --recursive


 COCO+VG pretraining (default)
* VL-T5/snap/pretrain/VLT5/Epoch30.pth: VL-T5 pretrained for 30 epochs on COCO+VG
* VL-T5/snap/pretrain/VLBart/Epoch30.pth: VL-BART pretrained for 30 epochs on COCO+VG

VCR pretraining (2nd stage)
* VL-T5/snap/vcr_pretrain/VLT5/Epoch20.pth: VL-T5 further pretrained for 20 epochs on VCR
* VL-T5/snap/vcr_pretrain/VLBart/Epoch20.pth: VL-BART further pretrained for 20 epochs on VCR

---Dataset Preparation / Feature extraction
- Download datasets/ from [Google Drive](https://drive.google.com/drive/folders/1MBBhlkP83VMKS2Qe0SmFfzkHhMpIG5wf?usp=sharing)

gdrive download 1MBBhlkP83VMKS2Qe0SmFfzkHhMpIG5wf --recursive


--Pretraining on COCO+VG

Pretraining with 4 gpus
cd VL-T5/
bash scripts/COCOVG_pretrain_VLT5.sh 4
bash scripts/COCOVG_pretrain_VLBart.sh 4

