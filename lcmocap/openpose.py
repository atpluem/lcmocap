import os
import sys
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
            sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

except Exception as e:
    print(e)
    sys.exit(-1)