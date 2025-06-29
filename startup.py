from run.run import run
import torch
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = r"C:\\Users\\hp\\.cache\\huggingface"



# model_name = 'HMCAN'
# args_dict = {
#     'train_path': 'F:\\VScode\\Project\\third_year\\QNLP\\fakenewsnet_dataset\\fakenewsnet_dataset\\Test1\\HMCAN\\data\\politifact_train.json',
#     'batch_size': 8,
#     'num_epochs': 15,
#     'test_path' : 'F:\\VScode\\Project\\third_year\QNLP\\fakenewsnet_dataset\\fakenewsnet_dataset\\Test1\\HMCAN\\data\\politifact_test.json'
# }

# model_name = 'mcan'
# args_dict = {
#     'train_path': 'dataset/example/MCAN/politifact_train.json',
#     'test_path': 'dataset/example/MCAN/politifact_test.json',
#     'batch_size': 32,
#     'num_epochs': 10
# }


model_name = 'cafe'
args_dict = {
    'dataset_dir': 'C:\\Users\\hp\\Desktop\\CAFE\\data',
    #'dataset_dir': 'D:\\编程软件\\VScode\\Project\\third_year\\QNLP\\fakenewsnet_dataset\\fakenewsnet_dataset\\Test1\\data',
    #'dataset_dir': 'D:\编程软件\VScode\Project\third_year\QNLP\FaKnow-master\FaKnow-master\dataset\example\Twitter_Rumor_Detection',
    'batch_size': 64,
    'lr': 1e-3,
    'epoch_num' : 30
}

run(model_name, **args_dict)


# for i in range(1,30):
#     model_name = 'cafe'
#     args_dict = {
#         'dataset_dir': 'F:\\VScode\\Project\\third_year\\QNLP\\fakenewsnet_dataset\\fakenewsnet_dataset\\Test1\\data',
#         #'dataset_dir': 'D:\\编程软件\\VScode\\Project\\third_year\\QNLP\\fakenewsnet_dataset\\fakenewsnet_dataset\\Test1\\data',
#         #'dataset_dir': 'D:\编程软件\VScode\Project\third_year\QNLP\FaKnow-master\FaKnow-master\dataset\example\Twitter_Rumor_Detection',
#         'batch_size': 64,
#         'epoch_num' : i
#     }


#     run(model_name, **args_dict)

# test_data = np.load("D:\\编程软件\\VScode\\Project\\third_year\\QNLP\\fakenewsnet_dataset\\fakenewsnet_dataset\\Test1\\data\\politifact_test_text_with_label.npz")
# train_data = np.load("D:\\编程软件\\VScode\\Project\\third_year\\QNLP\\fakenewsnet_dataset\\fakenewsnet_dataset\\Test1\\data\\politifact_train_text_with_label.npz")

# print(f"Train data shape: {train_data['data'].shape}, dtype: {train_data['data'].dtype}")
# print(f"Test data shape: {test_data['data'].shape}, dtype: {test_data['data'].dtype}")

