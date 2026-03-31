import torch
import numpy as np

# MotherNet / TICL imports
from ticl.prediction import MotherNetClassifier



class mothernet(torch.nn.Module):
    def __init__(self, params, tasktype, input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="mothernet"):
        
        super(mothernet, self).__init__()
        self.tasktype = tasktype
        self.model = MotherNetClassifier(device=device)
    
    def fit(self, X_train, y_train):
        labeled_flag = torch.unique(torch.where(~torch.isnan(y_train))[0])
        label_X_train = X_train[labeled_flag]
        label_y_train = y_train[labeled_flag]
        
        if self.tasktype == "multiclass":
            label_y_train = torch.argmax(label_y_train, dim=1)
        try:
            self.model.fit(label_X_train.cpu().numpy(), label_y_train.cpu().numpy())
            self.exception = False
        except ValueError as e:
            self.exception = True
            print(f"Error occurred while fitting the model: {e}")
            
    def predict(self, X_test):
        if self.exception:
            print("Model fitting failed. Cannot make predictions.")
            return None
        else:
            preds = self.model.predict(X_test.cpu().numpy())
            return preds
        
    def predict_proba(self, X_test, logit=False):
        if self.exception:
            return None
        else:
            return self.model.predict_proba(X_test.cpu().numpy())