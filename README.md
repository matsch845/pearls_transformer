# Transformer Models for Outcome-Oriented Predictive Process Monitoring

This fork adds the outcome-oriented perspective to the other approaches already implemented.
It incorporates a transformer architecture and data pre-processing as suggested by the original repository.
Additionally, a slightly different transformer model with an different pre-processing (bag-of-words model with 2 step lead) is implemented (`outcome_oriented_new.py`).

## How to run

0. `pip install -r requirements.txt`
1. Download the datasets cited in this repository (Section Datasets).
2. Place the `.xes` files in the corresponding folders.
3. Make sure that the files in the dataset folders have exactly the same naming as the folders where they are placed in (e.g., `BPIC2015M1.xes``).
4. `cd datasets`
5. `python convert_to_csv.py`
6. `cd ..`
7. `./run_data_processing.sh`
8. `./run_training.sh`
9. `./run_data_processing_new.sh`
10. `./run_training_new.sh`

## Datasets

Those datasets are used to benchmark transformer models to predict the outcome of a case.

- BPIC 2011: https://data.4tu.nl/articles/dataset/Real-life_event_logs_-_Hospital_log/12716513/1

- BPIC 2012: https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204/1

- BPIC 2015_1: https://data.4tu.nl/articles/dataset/BPI_Challenge_2015_Municipality_1/12709154/1

- BPIC 2015_5: https://data.4tu.nl/articles/dataset/BPI_Challenge_2015_Municipality_5/12713285/1

- BPIC 2015_2: https://data.4tu.nl/articles/dataset/BPI_Challenge_2015_Municipality_2/12697349/1

- BPIC 2015_3: https://data.4tu.nl/articles/dataset/BPI_Challenge_2015_Municipality_3/12718370/1

- BPIC 2015_4 https://data.4tu.nl/articles/dataset/BPI_Challenge_2015_Municipality_4/12697898/1

- BPIC 2017: https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884/1


## ProcessTransformer: Predictive Business Process Monitoring with Transformer Network

![header image](https://github.com/Zaharah/processtransformer/blob/main/pt.JPG)

<details><summary>Abstract (click to expand)</summary>
<p>

Predictive business process monitoring focuses on predicting future characteristics of a running process using event logs. The foresight into process execution promises great potentials for efficient operations, better resource management, and effective customer services. Deep learning-based approaches have been widely adopted in process mining to address the limitations of classical algorithms for solving multiple problems, especially the next event and remaining-time prediction tasks. Nevertheless, designing a deep neural architecture that performs competitively across various tasks is challenging as existing methods fail to capture long-range dependencies in the input sequences and perform poorly for lengthy process traces. In this paper, we propose ProcessTransformer, an approach for learning high-level representations from event logs with an attention-based network. Our model incorporates long-range memory and relies on a self-attention mechanism to establish dependencies between a multitude of event sequences and corresponding outputs. We evaluate the applicability of our technique on nine real event logs. We demonstrate that the transformer-based model outperforms several baselines of prior techniques by obtaining on average above 80% accuracy for the task of predicting the next activity. Our method also perform competitively, compared to baselines, for the tasks of predicting event time and remaining time of a running case.

</p>
</details>


#### Tasks
- Next Activity Prediction
- Time Prediction of Next Activity
- Remaining Time Prediction

### Install 
```
pip install processtransformer
```


### Usage  
We provide the necessary code to use ProcessTransformer with the event logs of your choice. We illustrate the examples using the helpdesk dataset. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tiOh2VS8yzOVON26CbmWn0oUn-dWAFhN?usp=sharing)

For the data preprocessing,  run:

```python
python data_processing.py --dataset=helpdesk --task=next_activity
python data_processing.py --dataset=helpdesk --task=next_time
python data_processing.py --dataset=helpdesk --task=remaining_time
```
To train and evaluate the model, run:

```python
python next_activity.py --dataset=helpdesk --epochs=100
python next_time.py --dataset=helpdesk --epochs=100
python remaining_time.py --dataset=helpdesk --epochs=100
```


### Tools
- <a href="http://tensorflow.org/">Tensorflow >=2.4</a>

## Data 
The events log for the predictive busienss process monitoring can be found at [4TU Research Data](https://data.4tu.nl/categories/_/13500?categories=13503)

## How to cite 

Please consider citing our paper if you use code or ideas from this project:

Zaharah A. Bukhsh, Aaqib Saeed, & Remco M. Dijkman. (2021). ["ProcessTransformer: Predictive Business Process Monitoring with Transformer Network"](https://arxiv.org/abs/2104.00721). arXiv preprint arXiv:2104.00721 


```
@misc{bukhsh2021processtransformer,
      title={ProcessTransformer: Predictive Business Process Monitoring with Transformer Network}, 
      author={Zaharah A. Bukhsh and Aaqib Saeed and Remco M. Dijkman},
      year={2021},
      eprint={2104.00721},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```