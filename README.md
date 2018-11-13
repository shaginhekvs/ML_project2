Data from : https://www.crowdai.org/challenges/epfl-ml-recommender-system/dataset_files  
copy to data folder, rename to train.csv, test.csv  


Change to DeepRecommender folder   
To preprocess data:  
python data_convert.py   

To train:   
python run.py --gpu_ids 0 --path_to_train_data ./data_processed/TRAIN/ --path_to_eval_data ./data_processed/VALID --hidden_layers 512,512,512,512,512,1024 --non_linearity_type selu --batch_size 128 --logdir model_save --drop_prob 0.8 --optimizer momentum --lr 0.005 --weight_decay 0 --aug_step 1 --noise_prob 0 --num_epochs 500 --summary_frequency 4000  

To visualize:  
tensorboard --logdir=model_save  

To make submission:  
python infer.py --path_to_train_data ./data_processed/TRAIN --path_to_eval_data ./data_processed/TEST --hidden_layers 512,512,512,512,512,1024 --non_linearity_type selu --save_path model_save/model.last --drop_prob 0.8 --predictions_path preds.txt  

python ml_data.py