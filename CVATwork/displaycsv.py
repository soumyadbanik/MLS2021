import cv2
import pandas as pd

path_to_csv = input("Enter full path to the csv: ")
#print(path_to_csv)
path_to_dir = input("Enter full path to the extracted frames directory: ")
path_to_keyfr = input("Enter full path to the directory where you want to save the annotated frames: ")
df = pd.read_csv(path_to_csv, index_col=None)

for i, row in df.iterrows():
        if i>=1:
            frno  = int(row['frameno'])
            left  = int(row['left']-1.5)
            top   = int(row['top']-1.5)
            right = int(row['right']-1.5)
            bottom= int(row['bottom']-1.5)
            #print(frno)
            #read frameno from allframes directory
            im = cv2.imread(path_to_dir+'/{:03d}.jpg'.format(frno))
            img= cv2.UMat(im)
            color=(0,0,255)
            box = cv2.rectangle(img,(left,top), (right,bottom), color, 1)
            cv2.imwrite(path_to_keyfr +'/{:03d}.jpg'.format(frno), box)
print('Last frame:{}'.format(frno))
print('Done')