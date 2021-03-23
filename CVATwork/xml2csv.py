import csv
import xml.etree.ElementTree as ET

tree = ET.parse('track.xml')
root = tree.getroot()
headers=['frameno','left', 'top', 'right', 'bottom']
filename="track.csv"
boxpos=[]
for box in root.iter('box'):
    #frame.append(int(box.get('frame')))
    frno = int(box.get('frame'))
    xmax = float(box.get('xbr'))
    ymax = float(box.get('ybr'))
    xmin = float(box.get('xtl'))
    ymin = float(box.get('ytl'))
    
    boxpos.append({'frameno': frno,
                'left': xmin,
                'top': ymin,
                'right': xmax,
                'bottom': ymax})
    
with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames= headers)
    writer.writeheader()
    writer.writerows(boxpos)