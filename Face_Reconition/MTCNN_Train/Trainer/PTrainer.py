"""
此文件用于P网络的训练
"""
from MTCNN_Train.Trainer.TrainModule import trainer
from MTCNN_Train.DetectorNet import PNet

if __name__ == '__main__':
    # --------------------------实例化训练器-----------------------------
    otrainer = trainer(net=PNet(),
                       dataset_path=r'F:\Datasets\MTCNN\12',
                       save_path=r'F:\咕泡学院项目实战\Face_Reconition\ModuleFiles\PNet.pt',
                       isCuda=True
                       )
    # ---------------------------开始训练---------------------------
    otrainer.train()