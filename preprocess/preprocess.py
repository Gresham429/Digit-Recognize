import pandas as pd

class Preprocesser:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.labels = []
        self.features = self.data

    # 去重
    def remove_duplicates(self):
        self.data = self.data.drop_duplicates()
        return self.data
   
    # 去空值 
    def remove_null_values(self):
        self.data = self.data.dropna()
        return self.data
    
    # 输出处理后的数据
    def save(self, output_path):
        self.data.to_csv(output_path, index=False)
    
    # 划分标签和特征
    def split_features_labels(self, label_names):
        for label_name in label_names:
            self.labels.append(self.data[label_name])
            self.features = self.features.drop(label_name, axis=1)

        return self.labels, self.features