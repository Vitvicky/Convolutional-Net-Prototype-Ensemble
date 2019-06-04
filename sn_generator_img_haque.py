import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from randomcolor import RandomColor


def generate(args, files):
    for file_name in files:
        path = os.path.join(args.file_dir, file_name)
        init_size =args.init_size
        time_delay = args.time_delay
        existing_density = args.density

        # read data
        data = pd.read_csv(path).values

        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]

        # separate features and labels
        features = data[:, :-1]
        labels = data[:, -1]

        class_assignment = {}

        # find class distribution
        for i, y in enumerate(labels):
            if y not in class_assignment:
                class_assignment[y] = []
            class_assignment[y].append(i)

        # shuffle
        for c in class_assignment:
            np.random.shuffle(class_assignment[c])

        # print out class distribution
        total_size = 0
        for label in class_assignment:
            print("label: {} -- size: {}".format(label, len(class_assignment[label])))
            total_size += len(class_assignment[label])
        
        # number of emerging classes
        num_exist = int(len(class_assignment) * existing_density)

        # find two major class
        class_size = sorted(class_assignment.items(), key=lambda x: x[1], reverse=True)
        existing_class = [cls for cls, size in class_size[:num_exist]]
        # existing_class = [0, 3, 5, 8, 6]

        # create a stream
        stream = []

        count = 0
        insert = True
        print("total size", total_size)
        while len(stream) < total_size:
            print("stream length: ", len(stream))
            print("existing class: ", existing_class)
            print("existing class size: ", [len(class_assignment[c]) for c in existing_class])
            if len(stream) < init_size:
                candidate_cls = existing_class
                # randomly pick a class
                while True:
                    cls = np.random.choice(candidate_cls, size=1)[0]
                    if len(class_assignment[cls]) > 0:
                        break
                # randomly pick some points
                stream.append(class_assignment[cls].pop(0))
            else:
                # determine if we need to add a new class
                r = np.random.random()
                if r > 0.5 and insert:
                    # add a class
                    while True:
                        # print("1")
                        n_cls = np.random.choice(list(class_assignment.keys()), size=1)[0]
                        if n_cls not in existing_class:
                            n_cls = [n_cls]
                            break
                        if set(existing_class) == set(class_assignment.keys()):
                            n_cls = []
                            break
                    candidate_cls = existing_class + n_cls
                    # randomly pick a class
                    while True:
                        # print("2")
                        cls = np.random.choice(candidate_cls, size=1)[0]
                        if len(class_assignment[cls]) > 0:
                            break
                        if all([len(class_assignment[cls]) == 0 for cls in candidate_cls]):
                            insert = True
                            count = 0
                            cls = None
                            break
                    # print("cls", cls, )
                    if cls is None:
                        continue
                    # randomly pick a few points
                    for i in range(args.add_num):
                        if len(class_assignment[cls]) == 0:
                            break
                        stream.append(class_assignment[cls].pop(0))
                    if cls in n_cls:
                        existing_class += n_cls
                        insert = False
                else:
                    candidate_cls = existing_class
                    tmp = [each for each in candidate_cls]
                    while True:
                        # print("3")
                        cls = np.random.choice(tmp, size=1)[0]
                        if len(class_assignment[cls]) > 0:
                            break
                        else:
                            tmp.remove(cls)
                        if all([len(class_assignment[c]) == 0 for c in candidate_cls]):
                            insert = True
                            count = 0
                            cls = None
                            break
                    if cls is None:
                        continue
                    
                    # add one existing 
                    stream.append(class_assignment[cls].pop(0))
                if not insert:
                    count += 1
                if count == time_delay:
                    insert = True
                    count = 0
                if not insert:
                    # check if all existing class are empty
                    if all([len(class_assignment[c]) == 0 for c in existing_class]):
                        insert = True
                        count = 0

        # check stream
        assert(len(stream) == len(set(stream)))  # no duplicates
        assert(len(stream) == len(features))   # no missing data
        # take points
        features = features[stream]
        labels = labels[stream]

        # plot stream
        class_x = {}
        for i, y1 in enumerate(labels):
            if y1 not in class_x:
                class_x[y1] = []
            class_x[y1].append(i)

        # fig, ax = plt.subplots()
        # r = RandomColor()
        # colors = r.generate(count=len(class_x))
        # for i, c in enumerate(class_x):
            # ax.plot(class_x[c], [c]*len(class_x[c]), 'o', markersize=1, color=colors[i])
        # plt.show()

        features = np.array(features)
        labels = np.array(labels).reshape(-1, 1)
        print(features.shape, labels.shape)
        data = np.concatenate((features, labels), axis=1)
        print(data.shape)
        print(data)
        # save d
        np.savetxt(file_name[:file_name.index('.')]+"_"+str(args.density)+"_simulated.csv", data, delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-dir", type=str, default="/home/wzy/Coding/Data/fashion-mnist/ori/"
                                                        "original data", metavar="-fd", help="directory to data files")
    parser.add_argument("--file", type=str, metavar='-F', help='file name')
    parser.add_argument("--density", type=float, default=0.5, metavar="-D", help="initial exisiting class density")
    parser.add_argument("--init-size",type=int, default=5000, help="initial window size")
    parser.add_argument("--time-delay", type=int, default=2000, help="time delay")

    parser.add_argument("--add-num", type=int, default=300, help="number of instances to add at a time")
    args = parser.parse_args()
    
    # args.file_dir = '/home/wzy/Coding/Data/cifar-10'
    args.add_argument = 200
    # files = ["cifar-10-ori.csv"]
    generate(args, files)




