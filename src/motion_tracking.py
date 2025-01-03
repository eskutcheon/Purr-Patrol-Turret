try:
    import cv2
except Exception as e:
    print("Warning: OpenCV not installed. To use motion detection, make sure you've properly configured OpenCV.")
import time
import torch
import torchvision.transforms as TT
import torch.nn.functional as F

# TODO: gonna need heavy modification for our libraries, use case, and general updates to Python3
class MotionTracker(object):
    """
    Helper functions for video utilities.
    """
    @staticmethod
    def live_video(camera_port=0):
        """
        Opens a window with live video.
        :param camera:
        :return:
        """
        video_capture = cv2.VideoCapture(camera_port)
        while True:
            # TODO: probably want to add some time.sleep(n) statements here to take a frame every `n` seconds
            # Capture frame-by-frame
            # NOTE: probably want a bigger delay like every 30 frames, assuming 30 fps - will depend on speed of object detection
            time.sleep(0.333) #sleeping for 333 milliseconds, checking roughly every 10 frames.
            ret, frame = video_capture.read()
            # Display the resulting frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): #checking for a q key press every millisecond to break the video loop.
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    # TODO: decompose this into more functions for better flexibility
    def find_motion(callback, camera_port=0, show_video=False):
        camera = cv2.VideoCapture(camera_port)
        time.sleep(0.25)
        # initialize the first frame in the video stream
        firstFrame = None
        tempFrame = None
        count = 0
        # loop over the frames of the video
        while True:
            # grab the current frame and initialize the occupied/unoccupied text
            (grabbed, frame) = camera.read()
            # if the frame could not be grabbed, then we have reached the end of the video
            if not grabbed:
                break
            # TODO: may want to move most of code below to pytorch to do more on CUDA - almost definitely the transforms at least
                # will depend on whether all this is implemented somewhere since openCV has some unique functions
            ###############################################################################################################
            # resize the frame, convert it to grayscale, and blur it
            # removing dependency on imutils with explicit changes:
            H, W = frame.shape[:2]
            # ratio of new width to old width
            r = 500/float(W)
            frame = cv2.resize(frame, dsize=(500, int(H*r)), interpolation=cv2.INTER_LINEAR)
            #frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            # if the first frame is None, initialize it
            if firstFrame is None:
                print("Waiting for video to adjust...")
                if tempFrame is None:
                    tempFrame = gray
                    continue
                else:
                    delta = cv2.absdiff(tempFrame, gray)
                    tempFrame = gray
                    tst = cv2.threshold(delta, 5, 255, cv2.THRESH_BINARY)[1]
                    tst = cv2.dilate(tst, None, iterations=2)
                    if count > 30:
                        print("Done.\n Waiting for motion.")
                        if not cv2.countNonZero(tst) > 0:
                            firstFrame = gray
                        else:
                            continue
                    else:
                        count += 1
                        continue
            # compute the absolute difference between the current frame and first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            # dilate the thresholded image to fill in holes, then find contours on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            c = MotionTracker.get_best_contour(thresh.copy(), 5000)
            # TODO: may create a new utils file and add my draw_bounding box functions to replace this
            if c is not None:
                # compute the bounding box for the contour, draw it on the frame, and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                callback(c, frame)
            # TODO: end stuff to edit to pytorch
            #######################################################################################################################
            # show the frame and record if the user presses a key
            if show_video:
                cv2.imshow("Security Feed", frame)
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                    break
        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_best_contour(imgmask, threshold):
        im, contours, hierarchy = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = threshold
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_cnt