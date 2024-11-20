# ASFWB
Pan-Tilt-Zoom Camera-Based Active Search Framework for Wetland Bird Monitoring

## Install
1.Create Anaconda environment  
```html<div style="background-color: #f0f0f0; padding: 10px;">
conda create -n asfwb python=3.8.5 pip
conda activate asfwb
```  
2.Install the dependencies  
```html<div style="background-color: #f0f0f0; padding: 10px;">
pip install -r requirements.txt
```

## Training
The [datasets](https://drive.google.com/drive/folders/1m_bLniSAtury3YLBxzs5jFjEpCZPNAuL?usp=sharing) used in the training and the associated pre-trained [models](https://drive.google.com/drive/folders/1gv8ZFmCTcii84svOpTLhJXvrJKUnpq2x?usp=sharing) are available.  
All the parameters needed to reproduce the results can be found in the config folder. 
You can run the experiment using the following command.
```html<div style="background-color: #f0f0f0; padding: 10px;">
python main.py --config configs/automatic_search_asfwb.yaml
```

## Credits
Some of our code come from the following repository. We appreciate these authors to share their valuable codes.  
* SCQ [SCQ](https://github.com/purewater0901/SCQ)
* SAM2 [SAM2](https://github.com/facebookresearch/sam2)
