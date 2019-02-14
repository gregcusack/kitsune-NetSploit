import numpy as np
import os

class GetFeatures:
    def __init__(self):
        self.features = None
        self.train_files = None
        self.train_y = []
        self.train_x = []


    def runner(self):
        self.train_files = self.get_feature_files('monday','../Mohammad-Kitsune/115_features_new/')
        print(self.train_files)
        self.get_files()
        self.train_x = np.concatenate(self.train_x, axis=0)

        # exit()
        # self.train_files = self.get_feature_files('monday', './115_features/')
        # self.train_x = np.concatenate(self.train_x, axis=0)

        # clean up
        # self.train_y = None
        self.train_files = None

    def compare_str(self, s1, s2):
        st1 = s1.split('_')[-1]
        st1 = st1[:st1.index('.')]
        st2 = s2.split('_')[-1]
        st2 = st2[:st2.index('.')]
        i1 = int(st1)
        i2 = int(st2)
        if i1 > i2:
            return 1
        elif i1 == i2:
            return 0
        else:
            return -1

    def get_feature_files(self, day, prefix='your_directory/packet_labels_only_v4'):
        all_files = []
        print("sup")
        for file in os.listdir(prefix):
            if file.endswith(".npy") and file.startswith(day):
                #         print(os.path.join("./SmallFiles/", file))
                all_files.append(os.path.join(prefix, file))
        all_files = sorted(all_files)
        #     for f in all_files:
        #         print(f)
        return all_files

    def get_files(self):
        for f in self.train_files:
            print(f)
            self.train_x.append(np.load(f))

    # def get_feature_files(self, day, prefix='your_directory/packet_labels_only_v4'):
    #     all_files = []
    #     for file in os.listdir(prefix):
    #         if file.endswith(".npy") and file.startswith(day):
    #             #         print(os.path.join("./SmallFiles/", file))
    #             all_files.append(os.path.join(prefix, file))
    #     all_files = sorted(all_files)
    #     #     for f in all_files:
    #     #         print(f)
    #     return all_files
    #
    # def get_train_data(self):
    #     for f in self.train_files:
    #         print(f)
    #         self.train_x.append(np.load(f))

