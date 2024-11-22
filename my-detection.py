#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils
import keyboard

net=jetson.inference.detectNet("ssd-mobilenet-v2",threshold=0.5) #load model

camera=jetson.utils.videoSource("/dev/video0") #opening the camera stream

display=jetson.utils.videoOutput("display://0") #display loop

while display.IsStreaming(): #main llop will go here
  img = camera.Capture()
  if img is None:
    continue
  detections=net.Detect(img)
  for detection in detections:
    if keyboard.is_pressed('q'):
      print(net.GetClassDesc(detection.ClassID))
      print(detection)
      jetson.utils.saveImageRGBA("1.jpg",img)
    if keyboard.is_pressed('w'):
      print(net.GetClassDesc(detection.ClassID))
      print(detection)
      jetson.utils.saveImageRGBA("2.jpg",img)
    if keyboard.is_pressed('e'):
      print(net.GetClassDesc(detection.ClassID))
      print(detection)
      jetson.utils.saveImageRGBA("3.jpg",img)
    if keyboard.is_pressed('r'):
      print(net.GetClassDesc(detection.ClassID))
      print(detection)
      jetson.utils.saveImageRGBA("4.jpg",img)
  
      
  display.Render(img)
  display.SetStatus("Object Detection | Network {:.0f}FPS".format(net.GetNetworkFPS()))
