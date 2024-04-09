from utilities.basics import ChangeableParam, XYParams, InOutParams

def changeable_params():
    print('Test changeable params')
    def change_function():
        print('Change function triggered')

    a = ChangeableParam[int](change_function)
    a = 3
    print(a)

def xy_params():
    print('Test xy params')
    def change_x_function():
        print('Change x function triggered')
    def change_y_function():
        print('Change y function triggered')

    a = XYParams[int](change_x_function, change_y_function)

    a.x = 3
    a.y = 4
    print(a.x, a.y)