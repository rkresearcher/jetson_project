import cv2
import time
import os
import datetime
from threading import Thread
from queue import Queue
import object_detection as cv
from object_detection import draw_bbox
import cv2
import time
import numpy as np
from vidgear.gears import CamGear


class Camera:
    def __init__(self, mirror=False):
        self.data = None
        
        self.cam =CamGear(source='https://www.youtube.com/watch?v=zGVITGPj7Mk', stream_mode = True, logging=True).start()  #ass the name of the video here with correct directory name in my case the directory is data

        self.WIDTH = 512 # only for testing update according to the video
        self.HEIGHT = 1024

        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2
        self.touched_zoom = False

        self.image_queue = Queue()
        self.video_queue = Queue()

        self.scale = 1
        #self.__setup()

        self.recording = False

        self.mirror = mirror

    def __setup(self):
        #self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        #self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        time.sleep(2)

    def get_location(self, x, y):
        self.center_x = x
        self.center_y = y
        self.touched_zoom = True

    def stream(self):
        
        def streaming():
            
            self.ret = True
            while self.ret:
                np_image = self.cam.read()
               # np_image = cv2.resize(np_image,(self.WIDTH,self.HEIGHT))
                if np_image is None:
                    print("None") 
                    continue
                if self.mirror:
                    
                    np_image = cv2.flip(np_image, 1)
                if self.touched_zoom:
                    np_image = self.__zoom(np_image, (self.center_x, self.center_y))
                else:
                    if not self.scale == 1:
                        np_image = self.__zoom(np_image)
                
                ###########
                age = 0
                newb = 0
                frame = np_image
                bbox, label, conf,p_id = cv.detect_common_objects(frame)
                print (label)
                print (bbox)
                # calculating the average of label[j]
                newb=time.time()
                act = newb-age
                age = newb 
                j = 0
                if len(bbox) >=0:
                   
                 for i in bbox:
                    if label[j] == 'person':
                       if 0<i[3]<=700: # at point i am taking 250 ad the thereshold. i will upload the code on github
                         isDrowning = True
                         np_image = out = draw_bbox(frame, i, label[j], conf[j],isDrowning,act)
                       else:
                            print ("No Drowning")
                            isDrowning = False
                            np_image = out = draw_bbox(frame, i, label[j], conf[j],isDrowning,act)
                    j  = j+1
                ##########
                 
                self.data = np_image
                k = cv2.waitKey(1)
                if k == ord('q'):
                    self.release()
                    break

        Thread(target=streaming).start()

    def __zoom(self, img, center=None):
        
        height, width = img.shape[:2]
        if center is None:
            
            center_x = int(width / 2)
            center_y = int(height / 2)
            radius_x, radius_y = int(width / 2), int(height / 2)
        else:
            
            rate = height / width
            center_x, center_y = center

            
            if center_x < width * (1-rate):
                center_x = width * (1-rate)
            elif center_x > width * rate:
                center_x = width * rate
            if center_y < height * (1-rate):
                center_y = height * (1-rate)
            elif center_y > height * rate:
                center_y = height * rate

            center_x, center_y = int(center_x), int(center_y)
            left_x, right_x = center_x, int(width - center_x)
            up_y, down_y = int(height - center_y), center_y
            radius_x = min(left_x, right_x)
            radius_y = min(up_y, down_y)        
        radius_x, radius_y = int(self.scale * radius_x), int(self.scale * radius_y)        
        min_x, max_x = center_x - radius_x, center_x + radius_x
        min_y, max_y = center_y - radius_y, center_y + radius_y        
        cropped = img[min_y:max_y, min_x:max_x]        
        new_cropped = cv2.resize(cropped, (width, height))
        return new_cropped

    def touch_init(self):
        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2
        self.touched_zoom = False
        self.scale = 1

    def zoom_out(self):
        
        if self.scale < 1:
            self.scale += 0.1
        if self.scale == 1:
            self.center_x = self.WIDTH
            self.center_y = self.HEIGHT
            self.touched_zoom = False

    def zoom_in(self):
        
        if self.scale > 0.2:
            self.scale -= 0.1

    def zoom(self, num):
        if num == 0:
            self.zoom_in()
        elif num == 1:
            self.zoom_out()
        elif num == 2:
            self.touch_init()

    def save_picture(self):
        
        ret, img = self.cam.read()
        if ret:
            now = datetime.datetime.now()
            date = now.strftime('%Y%m%d')
            hour = now.strftime('%H%M%S')
            user_id = '00001'
            filename = './images/cvui_{}_{}_{}.png'.format(date, hour, user_id)
            cv2.imwrite(filename, img)
            self.image_queue.put_nowait(filename)

    def record_video(self):
        
        fc = 20.0
        record_start_time = time.time()
        now = datetime.datetime.now()
        date = now.strftime('%Y%m%d')
        t = now.strftime('%H')
        num = 1
        filename = 'videos/cvui_{}_{}_{}.avi'.format(date, t, num)
        while os.path.exists(filename):
            num += 1
            filename = 'videos/cvui_{}_{}_{}.avi'.format(date, t, num)
        codec = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter(filename, codec, fc, (int(self.cam.get(3)), int(self.cam.get(4))))
        while self.recording:
            if time.time() - record_start_time >= 600:
                self.record_video()
                break
            ret, frame = self.cam.read()
            if ret:
                if len(os.listdir('./videos')) >= 100:
                    name = self.video_queue.get()
                    if os.path.exists(name):
                        os.remove(name)
                out.write(frame)
                self.video_queue.put_nowait(filename)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

    def show(self):
        while True:
            frame = self.data
            
            if frame is not None:
                cv2.imshow('SMS', frame)
                cv2.setMouseCallback('SMS', self.mouse_callback)
                
            cv2.waitKey(0)
        self.release()
        cv2.destroyAllWindows()
              #   break

            #elif key == ord('z'):
                # z : zoom - in
             #   self.zoom_in()

#            elif key == ord('x'):
 #               # x : zoom - out
  #              self.zoom_out()

#            elif key == ord('p'):
 #               # p : take picture and save image (image folder)
  #              self.save_picture()

   #         elif key == ord('v'):
    #            self.touch_init()

     #       elif key == ord('r'):
      #          self.recording = not self.recording
       #         if self.recording:
        #            t = Thread(target=cam.record_video)
         #           t.start()
            

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.get_location(x, y)
            self.zoom_in()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.zoom_out()
        elif event == None:
            break


if __name__ == '__main__':
    cam = Camera(mirror=True)
    cam.stream()
    cam.show()
