import DataGenerate as DG
import DataPreprocess as DP

# DataGeModel = DG.DataGeneratorModel()
# X, labels = DataGeModel.fit()
# DataGeModel.show_data(X=X, labels=labels)

# DataPreProModel = DP.DataPreprocessorModel(X=X, labels=labels)
# X_train_pca, labels_train, X_val_pca, labels_val, X_test_pca, labels_test = DataPreProModel.fit()

# DataGeModel.show_data(X=X_train_pca, labels=labels_train)

# import numpy as np
# import ActivationFunction as AF
# data1 = [-1,-2,0,1,2,3]
# data2 = [-1.2, -2.4,0.0, 0.3, 1.43, 2.324]
# #data = tanh(data)
# AFModel = AF.NNActivator()
# data1 = AFModel.fit(data=data1, function_name="sigmoid")
# data2 = AFModel.softmax_fit(data=data2)
# print(data1)
# print(data2)


data1 = [-1,-2,0,1,2,3]
data2 = [-1.2, -2.4,0.0, 0.3, 1.43, 2.324]

print(len(data1))