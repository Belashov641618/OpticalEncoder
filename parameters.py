import os

import belashovplot

# Папка с датасетами, по умолчанию: ProjectFolder/datasets
DataSetsPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
if not os.path.exists(DataSetsPath):
    os.mkdir(DataSetsPath)

# Дефолтные параметры графиков
FigureWidthHeight = (8.3, 11.7)
FontLibrary = belashovplot.FontLibraryClass()

if __name__ == '__main__':
    print(DataSetsPath)