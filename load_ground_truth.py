import  os;
import numpy as np
def ground_loader(ground_file):
    points = []
    file_path = os.path.join(os.getcwd(), ground_file)
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split())
            points.append((x,y))

    return np.array(points)