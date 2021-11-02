import sys
import openpose as op

def main():
    keypoints = op.openposeAPI()
    print(keypoints)

if __name__ == "__main__":
    main()