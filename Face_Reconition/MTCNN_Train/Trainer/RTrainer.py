"""
此文件用于R网络的训练
"""
from MTCNN_Train.Trainer.TrainModule import trainer
from MTCNN_Train.DetectorNet import RNet

if __name__ == '__main__':
    # --------------------------实例化训练器-----------------------------
    otrainer = trainer(net=RNet(),
                       dataset_path=r'F:\Datasets\MTCNN\24',
                       save_path=r'F:\咕泡学院项目实战\Face_Reconition\ModuleFiles\RNet.pt',
                       isCuda=True)
    # ---------------------------开始训练---------------------------
    otrainer.train()