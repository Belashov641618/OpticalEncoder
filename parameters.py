import os

import belashovplot

# Папка с датасетами, по умолчанию: ProjectFolder/datasets
DataSetsPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
if not os.path.exists(DataSetsPath):
    os.mkdir(DataSetsPath)

# Дефолтные параметры графиков
FigureWidthHeight = (16.5, 11.7)
FontLibrary = belashovplot.FontLibraryClass()
FontLibrary.MultiplyFontSize(0.7)
FontLibrary.SynchronizeFont('DejaVu Sans')
FontLibrary.Fonts.SmallCaption.FontSize *= 0.7

if __name__ == '__main__':
    print(DataSetsPath)