# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to classify objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import picamera

import RPi.GPIO as GPIO

from PIL import Image
from tflite_runtime.interpreter import Interpreter


#라벨을 생성한다
def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():

  isMask=False
 
  GPIO.setmode(GPIO.BCM)
  GPIO.setup(12,GPIO.OUT)
  GPIO.output(12,False)
  
  #마스크 모델 위치 및 라벨
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--fmodel', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--flabels', help='File path of labels file.', required=True)

  args = parser.parse_args()

  labels = load_labels(args.labels)
  flabels= load_labels(args.flabels)

  #비밀번호
  password =[2,2,0];
  pw_input=[];      #입력하는 비밀번호
  pw_tmp=["*","*","*"]
  label_id_buf=None;
  i=0

  interpreter = Interpreter(args.model)
  finterpreter = Interpreter(args.fmodel)
  
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  finterpreter.allocate_tensors()
  _, height, width, _ = finterpreter.get_input_details()[0]['shape']


  with picamera.PiCamera(resolution=(480, 800), framerate=15) as camera:
    
    camera.start_preview()
    #path2 = r'/home/pi/ai/frame/frame5.png'
    #img = Image.open(path2)
    #pad = Image.new('RGBA', (
    #    ((img.size[0] + 31) // 32) * 32,
    #    ((img.size[1] + 15) // 16) * 16,
    #    ))
    #pad.paste(img, (0,0)) # (x,y) means pos
    
    #o = camera.add_overlay(pad.tobytes(),size=img.size)
    #o.alpha = 255 # overlay image opacity doesn't work on RGBA
    #o.layer = 3

    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize((width, height),
                                                         Image.ANTIALIAS)
        start_time = time.time()
            
        camera.annotate_foreground = picamera.Color('black')
        camera.annotate_background = picamera.Color('white')
        camera.annotate_text_size = 50
        
        
        if isMask==False:
            results = classify_image(interpreter, image)
            elapsed_ms = (time.time() - start_time) * 1000
            label_id, prob = results[0]
            
            stream.seek(0)
            stream.truncate()
            camera.annotate_text = '%s %.2f\n%.1fms' % (labels[label_id], prob,
                                                    elapsed_ms)
            if label_id==0:
                  isMask=True
                  print("isMask ==1")
                  camera.annotate_text = 'Mask Detected'
                  camera.annotate_text = 'ready to input pw...'
                  time.sleep(4)
            time.sleep(0.5)
        if (isMask==True and len(pw_input)<1) or (len(pw_input)==len(password) and pw_input!=password):
            pw_input=[];      #입력하는 비밀번호
            pw_tmp=["*","*","*"]
            label_id_buf=None
            print("hellworld!")
            i=0
            time.sleep(0.5)
        

        if len(pw_input)<=len(password) and isMask:
            
            results = classify_image(finterpreter, image)
            #elapsed_ms = (time.time() - start_time) * 1000
            label_id, prob = results[0]
            print("stacking pw")
            label_id_buf=label_id
            time.sleep(1)
            pw_input.append(label_id_buf)
            pw_tmp[i]=pw_input[i]
            i=i+1
            time.sleep(0.1)
            
            if len(pw_input)==3:
                print(pw_input)

            stream.seek(0)
            stream.truncate()
            camera.annotate_text = '%s %s %s' %(pw_tmp[0],pw_tmp[1],pw_tmp[2])
            #camera.annotate_text = '%s %.2f\n%.1fms' % (flabels[label_id_buf], prob,
            #                                        elapsed_ms)
            time.sleep(0.5)
            if pw_input==password:
                  #불키고
                  print("your password is correct")
                  GPIO.output(12, True)
                  time.sleep(2)
                  GPIO.output(12, False)
                  time.sleep(1)
                  GPIO.output(12, True)
                  time.sleep(2)
                  GPIO.output(12, False)
                  time.sleep(1)
                  break
            elif pw_input!=password and len(pw_input)>=len(password):
                  #비번틀림
                  pw_input=[]
                  isMask=False
                  print("your password is incorrect")

        
              
              

    finally:
      camera.stop_preview()


if __name__ == '__main__':
  main()