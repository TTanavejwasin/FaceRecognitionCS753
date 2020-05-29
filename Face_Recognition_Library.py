import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import pandas as pd

files_image = sorted(glob.glob('/Users/mewz/Desktop/OpenCV/testimage/*.jpg'))
clf = cv2.face.EigenFaceRecognizer_create()
clf.read('EIGENFACEOPENCV.xml')

id = 0

for image in files_image:
    id += 1
    image_in_file = cv2.imread(image)
    image_resize = cv2.resize(image_in_file, (200, 250))
    image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
    clf_predict, _ = clf.predict(image_gray)
    print(clf_predict, _)

    if clf_predict == 1:
        cv2.putText(image_in_file, 'Mew :'+str(_), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    elif clf_predict == 2:
        cv2.putText(image_in_file, 'Parn :'+str(_), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    elif clf_predict == 3:
        cv2.putText(image_in_file, 'Mind :'+str(_), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    elif clf_predict == 4:
        cv2.putText(image_in_file, 'Nine :'+str(_), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    elif clf_predict == 5:
        cv2.putText(image_in_file, 'P-Tee :'+str(_), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Image' + str(id), image_in_file)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        continue


confusion_matrix = [[10, 0, 0, 0, 0], [0, 10, 0, 0, 0], [0, 0, 10, 0, 0], [3, 0, 0, 7, 0], [0, 0, 0, 0, 10]]
df_cm = pd.DataFrame(confusion_matrix, index=[i for i in 'MTPMN'],
                     columns=[i for i in 'MTPMN'])
plt.figure(figsize=(10, 7))
plt.title('CONFUSION MATRIX')
sns.heatmap(df_cm, annot=True)
plt.show()

cv2.waitKey(1)
cv2.destroyAllWindows()