import cv2
vidcap = cv2.VideoCapture('5.mp4')

success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print 'Read a new frame: ', success
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
  
'''import subprocess as sp
cmd='ffmpeg -i 5.mp4 -r 1 outputFrames_%04d.jpeg'
sp.call(cmd,shell=True)'''  

#ffmpeg -i input.mov -r 0.25 output_%04d.png