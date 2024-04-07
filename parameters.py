import os

DataSetsPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
if not os.path.exists(DataSetsPath):
    os.mkdir(DataSetsPath)

if __name__ == '__main__':
    print(DataSetsPath)