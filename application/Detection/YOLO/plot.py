import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

import pandas as pd
import seaborn as sns

from scipy import interpolate

filenames = (
    '/home/mooziisp/Documents/latex/doc/run-Nov26_170742_YOLO_SpineDetection_P61.529B_F49.746B-tag-train_loss.csv',
    '/home/mooziisp/Documents/latex/doc/run-Nov26_152851_YOLOv_SpineDetection_P48.987B_F49.727B-tag-train_loss.csv'
)

# filenames = (
#     '/home/mooziisp/Documents/latex/doc/run-Nov27_052454_YOLO_SpineDetection_P61.529B_F49.746B-tag-train_loss.csv',
#     '/home/mooziisp/Documents/latex/doc/run-Nov27_121732_YOLOv_SpineDetection_P48.987B_F49.727B-tag-train_loss.csv'
# )

fig = plt.figure()
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim([0, 2])
styles = ['.', '*']
for style, filename, label in zip(*(styles[:len(filenames)],
                                    filenames,
                                    ['YOLO', 'ours'])):
    csv = pd.read_csv(filename)
    columns = csv.columns
    print(columns)

    x = csv.get(columns[1])
    y = csv.get(columns[2])

    # if label == 'ours':
    #     x = x.to_numpy()[:496]
    #     y = y.to_numpy()[:496]

    spl = interpolate.splrep(x, y)
    x2 = np.arange(x.min(), x.max(), 220)
    y2 = interpolate.splev(x2, spl)

    plt.plot(x2, y2, style, label=label, markersize=10)
    plt.plot(x2, y2, color='gray', linewidth=0.5)
plt.legend()
plt.savefig('loss-nomulti.svg', format='svg')
