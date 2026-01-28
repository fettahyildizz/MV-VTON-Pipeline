import cv2
import matplotlib.pyplot as plt
import copy
import argparse

from src import util
from src.body import Body
from src.hand import Hand

def main(args):
    body_estimation = Body('model/body_pose_model.pth')

    test_image = args.image
    oriImg = cv2.imread(test_image)  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    keypoints = util.get_keypoints(candidate)
    print(f"Candidate candidate:\n{candidate}")
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Body Pose Estimation')
    parser.add_argument('-i', '--image',
                        type=str,
                        default='images/dogan_frontal_pose1.jpeg',
                        help='Path to the input image'
                        )
    args = parser.parse_args()
    main(args)