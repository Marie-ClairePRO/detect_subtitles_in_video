# Generate srt file from hard subtitled video

## Description

This projects uses [**EasyOCR**] (https://github.com/JaidedAI/EasyOCR) library that detects texts in images. It takes as input a video and generates its png frames in a side folder with ffmpeg. The EasyOCR method detects boxes containing text in frames, and this code will merge intelligently the information given by each frame to generate a video_name.srt file.

## Installation

```bash
conda create -n subtitles_detector python=3.10
conda activate subtitles_detector
pip install -r requirements.txt
```

This code requires [PyTorch](https://pytorch.org/get-started/previous-versions/). Make sure to select the right cuda version, currently set to 12.8 in the requirements.txt. To run on CPU only, remove cuda extension in torch installation.

## Usage

To use this code, please run

```bash
python to_srt_v2.py --video video_name.mp4
```

The options :
- video : is required, all formats handled by ffmpeg are fine
- frames_folder : name of folder to put frames of your video. Default is "frames" but you can modify it.
- lang : language(s) of subtitles. You can give many. Default is english, french and german.
- cut_side : 4 sides you can cut your video to find text. By default, we cut the top half of the image. If there is unwanted text elsewhere you can use one of the options to cut it.
- difficulty_level : estimation of difficulty of subtitles. It is always better to set to 1 (default) but if ever it is very tricky, you can try bigger values up to 5. 