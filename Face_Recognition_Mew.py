import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import pandas as pd

files_image = sorted(glob.glob('/Users/mewz/Desktop/OpenCV/EIGENFACEOPENCV/*.jpg'))
name = ['Mind', 'P-Tee', 'Parn', 'Mew', 'Nine']

raw_data = []
id = 0

for image in files_image:
    id += 1
    image_in_file = cv2.imread(image)
    image_resize = cv2.resize(image_in_file, (200, 250))
    image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY).flatten()
    # cv2.imshow('raw data' + str(id), image_resize)
    raw_data.append(image_gray)
    cv2.waitKey(1)
    print(image_gray.shape)

raw_data_array = np.asarray(raw_data)
raw_data_trans = np.transpose(raw_data)
print('IMAGE DATA : ', raw_data_array.shape)

# for i in range(len(raw_data_array)):
#     img = raw_data_array[i].reshape(250, 200)
#     print(name[i])
#     plt.subplot(1, 5, 1+i)
#     plt.suptitle('FACE TRAIN')
#     plt.xlabel(name[i], color='blue')
#     plt.imshow(img, cmap='gray')
#     plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off',right='off',left='off', which='both')
# plt.show()

mean_face = np.mean(raw_data_array, axis=0)
print('MEAN', mean_face.shape)

plt.suptitle('MEAN FACE TRAIN')
plt.imshow(mean_face.reshape(250, 200), cmap='gray')
plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
plt.show()

normalized = []
for i in range(len(raw_data_array)):
    dif_image = np.subtract(raw_data_array[i],  mean_face)
    normalized.append(dif_image)
normalized_array = np.asarray(normalized)
normalized_array_trans = np.transpose(normalized_array)

for i in range(len(normalized_array)):
    img = normalized_array[i].reshape(250, 200)
    plt.subplot(10, 5, 1+i)
    plt.suptitle('NORMALIZED FACE TRAIN')
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
plt.show()

cov = np.cov(normalized_array)
cov_matrix = np.divide(cov, len(normalized_array))
print('COVARIANCE MATRIX :', cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print('EIGENVALUES : {} // EIGENVECTORS : {}// : '.format(eigenvalues, eigenvectors))

k = 50
k_eigen_vectors = eigenvectors[0:k, :]
# print('K EIGEN VECTOR', k_eigen_vectors)

eigen_face = []
for j in range(len(k_eigen_vectors)):
    face_space = np.dot(normalized_array_trans, k_eigen_vectors[j])
    eigen_face.append(face_space)
eigen_face_array = np.array(eigen_face)
print('EIGEN FACE : ', eigen_face_array)

for i in range(len(eigen_face_array)):
    img = eigen_face_array[i].reshape(250, 200)
    plt.subplot(10, 5, 1+i)
    plt.suptitle('EIGEN FACE TRAIN')
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
plt.show()

weight = np.dot(normalized_array, np.transpose(eigen_face_array))
print('WEIGHT : ', weight)
print('SHAPE WEIGHT : ', weight.shape)

#TEST------------------------------------------------------
files_image_test = sorted(glob.glob('/Users/mewz/Desktop/OpenCV/testimage/*.jpg'))

test_data = []
test_id = 0

for image in files_image_test:
    test_id += 1
    image_in_file = cv2.imread(image)
    image_resize = cv2.resize(image_in_file, (200, 250))
    image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY).flatten()
    # cv2.imshow('raw data' + str(id), image_resize)
    test_data.append(image_gray)
    cv2.waitKey(1)

test_array = np.asarray(test_data)
print(test_array.shape)

for i in range(len(test_array)):
    img = test_array[i].reshape(250, 200)
    plt.subplot(10, 5, 1+i)
    plt.suptitle('FACE TRAIN')
    plt.title('unknown', color='red')
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off',right='off',left='off', which='both')
plt.show()

normalized_test = []
for i in range(len(test_array)):
    dif_image = np.subtract(test_array[i],  mean_face)
    normalized_test.append(dif_image)
normalized_test_array = np.asarray(normalized_test)
normalized_test_array_trans = np.transpose(normalized_array)

for i in range(len(normalized_test_array)):
    img = normalized_test_array[i].reshape(250, 200)
    plt.subplot(10, 5, 1+i)
    plt.suptitle('NORMALIZED FACE TEST')
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
plt.show()

# plt.imshow(normalized_test.reshape(250, 200), cmap='gray')
# plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')
# plt.show()

predict = {'0': 'Predict : Mind', '1': 'Predict : P-Tee', '2': 'Predict : Parn', '3': 'Predict : Mew',
           '4': 'Predict : Nine'}
for test in range(len(normalized_test_array)):
    w_unknown = np.dot(normalized_test_array[test], np.transpose(eigen_face_array))
    diff = w_unknown - weight
    index = np.argmin(np.linalg.norm(diff, axis=1))
    print('DIFF : ', diff)
    print('LINALG NORM : ', np.linalg.norm(diff, axis=1))
    print('INDEX : ', index)

#unknown = 9.53512032e+07
#Mind  = 1.28035686 e-08
#Nine  = 1.76706034 e-08
#P'Tee = 5.21623789 e-08
#Parn  = 6.66400187 e-08
#Mew   = 8.06118734 e-08
    #
    # img_test = test_array[test].reshape(250, 200)
    # plt.suptitle('PREDICTION TEST', color='pink')
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_test, cmap='gray')
    # plt.xlabel('Test Image', color='blue')
    # plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
    #                 which='both')
    if 20 <= index <= 29:
        img_test = test_array[test].reshape(250, 200)
        plt.suptitle('PREDICTION TEST', color='pink')
        plt.subplot(1, 2, 1)
        plt.imshow(img_test, cmap='gray')
        plt.xlabel('Test Image', color='blue')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
        img_true = cv2.resize(cv2.imread('/Users/mewz/Desktop/OpenCV/testmind/mind (1).jpg'), (200, 250))
        print(predict['0'])
        plt.subplot(1, 2, 2)
        plt.xlabel(predict['0'], color='green')
        plt.imshow(img_true, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
    elif 40 <= index <= 49:
        img_test = test_array[test].reshape(250, 200)
        plt.suptitle('PREDICTION TEST', color='pink')
        plt.subplot(1, 2, 1)
        plt.imshow(img_test, cmap='gray')
        plt.xlabel('Test Image', color='blue')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
        img_true = cv2.resize(cv2.imread('/Users/mewz/Desktop/OpenCV/testptee/p-tee (1).jpg'), (200, 250))
        print(predict['1'])
        plt.subplot(1, 2, 2)
        plt.xlabel(predict['1'], color='green')
        plt.imshow(img_true, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')

    elif 10 <= index <= 19:
        img_test = test_array[test].reshape(250, 200)
        plt.suptitle('PREDICTION TEST', color='pink')
        plt.subplot(1, 2, 1)
        plt.imshow(img_test, cmap='gray')
        plt.xlabel('Test Image', color='blue')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
        img_true = cv2.resize(cv2.imread('/Users/mewz/Desktop/OpenCV/testparn/parn1.jpg'), (200, 250))
        print(predict['2'])
        plt.subplot(1, 2, 2)
        plt.xlabel(predict['2'], color='green')
        plt.imshow(img_true, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off', which='both')

    elif 0 <= index <= 9:
        img_test = test_array[test].reshape(250, 200)
        plt.suptitle('PREDICTION TEST', color='pink')
        plt.subplot(1, 2, 1)
        plt.imshow(img_test, cmap='gray')
        plt.xlabel('Test Image', color='blue')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
        img_true = cv2.resize(cv2.imread('/Users/mewz/Desktop/OpenCV/testmew/mew1.jpg'), (200, 250))
        print(predict['3'])
        plt.subplot(1, 2, 2)
        plt.xlabel(predict['3'], color='green')
        plt.imshow(img_true, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
    elif 30 <= index <= 39:
        img_test = test_array[test].reshape(250, 200)
        plt.suptitle('PREDICTION TEST', color='pink')
        plt.subplot(1, 2, 1)
        plt.imshow(img_test, cmap='gray')
        plt.xlabel('Test Image', color='blue')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
        img_true = cv2.resize(cv2.imread('/Users/mewz/Desktop/OpenCV/testnine/nine (1).jpg'), (200, 250))
        print(predict['4'])
        plt.subplot(1, 2, 2)
        plt.xlabel(predict['4'], color='green')
        plt.imshow(img_true, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
    else:
        img_test = test_array[test].reshape(250, 200)
        plt.title('PREDICTION TEST', color='pink')
        plt.subplot(1, 2, 1)
        plt.imshow(img_test, cmap='gray')
        plt.xlabel('Test Image', color='blue')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
        plt.subplot(1, 2, 2)
        plt.xlabel('UNKNOWN FACE', color='red')
        plt.imshow(mean_face, cmap='gray')
        plt.tick_params(labelleft='off', labelbottom='off', bottom='off', top='off', right='off', left='off',
                        which='both')
    plt.show()

confusion_matrix = [[10, 0, 0, 0, 0], [0, 10, 0, 0, 0], [0, 0, 10, 0, 0], [2, 0, 1, 7, 0], [0, 0, 0, 0, 10]]
df_cm = pd.DataFrame(confusion_matrix, index=[i for i in 'MTPMN'],
                     columns=[i for i in 'MTPMN'])
plt.figure(figsize=(10, 7))
plt.title('CONFUSION MATRIX')
sns.heatmap(df_cm, annot=True)
plt.show()

cv2.waitKey(1)
cv2.destroyAllWindows()