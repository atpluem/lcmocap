import os
import sys
import cv2
import time
from sys import platform

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    try:
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            # sys.path.append(dir_path + '/../../python/openpose/Release');
            # os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            # import pyopenpose as op
            print('Error: win32')
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            # sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            import pyopenpose as op
    
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e


    params = dict()
    params["model_folder"] = "../openpose/models/"
    params["image_dir"] = "input/images/"

    imagePaths = op.get_images_on_directory("input/images/")
    start = time.time()

    # Process and display images
    for imagePath in imagePaths:
        # Startting Openpose
        opWrapper = op.WrapperPython()

        # Capture body and hands keypoints
        params["face"] = False
        params["hand"] = True
        opWrapper.configure(params)
        opWrapper.start()

        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        poseKeypoint = str(datum.poseKeypoints)
        leftHandKeypoint = str(datum.handKeypoints[0])
        rightHandKeypoint = str(datum.handKeypoints[1])

        # Capture body and face keypoint
        params["face"] = True
        params["hand"] = False
        opWrapper.configure(params)
        opWrapper.start()

        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        faceKeypoint = str(datum.faceKeypoints)

        print("Body keypoints: \n" + poseKeypoint)
        print("Face keypoints: \n" + faceKeypoint)
        print("Left hand keypoints: \n" + leftHandKeypoint)
        print("Right hand keypoints: \n" + rightHandKeypoint)
    
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        key = cv2.waitKey(0)
        if key == 27: 
            break

except Exception as e:
    print(e)
    sys.exit(-1)