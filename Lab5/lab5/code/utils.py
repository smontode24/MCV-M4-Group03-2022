import itertools
import pathlib
from argparse import Namespace
from operator import itemgetter

import cv2
# render 2d/3d plots
import matplotlib.pyplot as plt
import numpy as np
import random

import fundamental as fundamental
import image_matches
import track as tk
import vps as vp
from common_misc import projective2img
import common_misc

# TODO: Whoever made this "logging" scheme should be sentenced to prison
# Keep global variables at minimum
# Used for debugging
# -1: QUIET, don't print anything
#  0: NORMAL, show steps performed
#  1: INFO, show values for different methods
#  2: VERBOSE, show relevant matrices of pipeline
#  3: INSANE, show all values of data structures
debug = 1

debug_display = True
normalise = True  # activate coordinate normalisation
opencv = True  # whether use opencv or matplot to display images

# Find path to dataset in student computer
dataset_path = ["../castle_dataset_L5"] #sorted(pathlib.Path("../..").glob("**/*5/castle_dataset_L5"))
assert len(dataset_path) == 1, f"Not sure where to look for the images, help me: \n {dataset_path}"
dataset_path = pathlib.Path(dataset_path[0])


def plot_images(imgs, titles=None):
    if not titles:
        titles = [f"Img {i} {img.shape}" for i, img in enumerate(imgs)]

    fig, axs = plt.subplots(ncols=len(imgs), nrows=1)
    for i, img, ax, title in zip(range(len(imgs)), imgs, axs, titles):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def load_n_images(num_images):
    """
    Loads the lab sequence of images in order
    :param num_images:
    :return: rgb_images, gray_images
    """
    assert 0 < num_images <= 5, "Only 5 images available"
    img_paths = sorted(dataset_path.glob("*.png"))

    gray_imgs = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in img_paths]
    rgb_imgs = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., ::-1] for path in img_paths]
    return rgb_imgs[:num_images], gray_imgs[:num_images]


def process_images(rgb_imgs, gray_imgs):
    """
    Utility function to process individual and pairs of images.
    :param rgb_imgs:
    :param gray_imgs:
    :return:
        - img_pairs_features: (Namespace) Class-like/NamedTuple-like holding a nested dictionary of image pairs
            features. That is for every img1, and img2 the features returned are:
                x1_e: euclidean coordinates of matches on img 1
                x2_e: euclidean coordinates of matches on img 2
                xr1_e: Corrected euclidean coordinates of matches on img 1
                xr2_e: Corrected euclidean coordinates of matches on img 2
                x1_h: Homogenous/Projective coordinates of matches on img 1
                x2_h: Homogenous/Projective coordinates of matches on img 2
                F: Fundamental matrix between cameras of img1 and img2
                matches: Matches are calculated through Brute-force with Hamming Norm.
                inlier_idx: Inliers Matches of the Fundamental matrix.
        - orb_features: (dict) Dictionary containing ORB features for each img
        - tracks: A list of `Tracks`
        - hs_vs: TODO
        - vanish_pts: Estimated vanishing points per image
    """
    assert len(rgb_imgs) == len(gray_imgs)
    n_imgs = len(rgb_imgs)
    # ORB features (Keypoints, Descriptors)
    orb_features = [image_matches.find_features_orb(img) for img in gray_imgs]
    vanish_pts = [vp.estimate_vps(img) for img in gray_imgs]

    tracks = []   # list of tracking views
    hs_vs = {}    # dictionary as hash table of views-tracks

    img_pairs_features = {}
    imgs_pairs_ids = sorted(set(itertools.combinations(range(n_imgs), 2)))

    for img1_id, img2_id in imgs_pairs_ids:
        print("\t Matching images", img1_id, "and", img2_id, " for obtaining tracks")
        # Can you do better ?
        matches = image_matches.match_features_hamming(orb_features[img1_id][1], orb_features[img2_id][1],
                                                         img1_id, img2_id)
        # Get match points projective coordinates in each image
        x1_h, x2_h = [], []
        for m in matches:
            x1_h.append([orb_features[img1_id][0][m.queryIdx].pt[0], orb_features[img1_id][0][m.queryIdx].pt[1], 1])
            x2_h.append([orb_features[img2_id][0][m.trainIdx].pt[0], orb_features[img2_id][0][m.trainIdx].pt[1], 1])
        x1_h = np.asarray(x1_h).T  # Cam 1 projective coordinates
        x2_h = np.asarray(x2_h).T  # Cam 2 projective coordinates

        # Estimate fundamental matrix, and get inliers
        F, indices_inlier_matches = fundamental.fundamental_matrix_ransac(points1=x1_h, points2=x2_h,
                                                                          threshold=1, max_iterations=5000)
        inlier_matches = itemgetter(*indices_inlier_matches)(matches)
        
        print("Before cleaning:", x1_h.shape, x2_h.shape)
        x1_h, x2_h = x1_h[:, indices_inlier_matches], x2_h[:, indices_inlier_matches]
        print("After cleaning:", x1_h.shape, x2_h.shape)
        
        # Get points image/euclidean coordinates of matches
        x1_e = projective2img(x1_h)
        x2_e = projective2img(x2_h)
        # Refine the coordinates using Optimal Triangulation Method
        xr1_e, xr2_e = fundamental.refine_matches(x1_e.T, x2_e.T, F)
        # Changed: Transpose elements
        tk.add_tracks(x1_e.T, x2_e.T, xr1_e.T, xr2_e.T, img1_id, img2_id, tracks, hs_vs)
        # h.draw_matches_cv(imgs[prev], imgs[i], x1, x2)

        print("  Tracks added after matching", img1_id, "and", img2_id)
        print("    Size of tracks:", len(tracks))
        print("    Size of hash table of views:", len(hs_vs))

        # Save the image pair features
        features = {"x1_e": x1_e, "x2_e": x2_e,
                    "xr1_e": xr1_e, "xr2_e": xr2_e,
                    "x1_h": x1_h, "x2_h": x2_h, "F": F, "vanish_pts": vanish_pts,
                    "matches": matches, "inliers_idx": indices_inlier_matches}
        img_pairs_features[(img1_id, img2_id)] = Namespace(**features)

    return img_pairs_features, orb_features, vanish_pts, tracks, hs_vs


def process_images_sift(rgb_imgs, gray_imgs):
    """
    Utility function to process individual and pairs of images.
    :param rgb_imgs:
    :param gray_imgs:
    :return:
        - img_pairs_features: (Namespace) Class-like/NamedTuple-like holding a nested dictionary of image pairs
            features. That is for every img1, and img2 the features returned are:
                x1_e: euclidean coordinates of matches on img 1
                x2_e: euclidean coordinates of matches on img 2
                xr1_e: Corrected euclidean coordinates of matches on img 1
                xr2_e: Corrected euclidean coordinates of matches on img 2
                x1_h: Homogenous/Projective coordinates of matches on img 1
                x2_h: Homogenous/Projective coordinates of matches on img 2
                F: Fundamental matrix between cameras of img1 and img2
                matches: Matches are calculated through Brute-force with Hamming Norm.
                inlier_idx: Inliers Matches of the Fundamental matrix.
        - orb_features: (dict) Dictionary containing ORB features for each img
        - tracks: A list of `Tracks`
        - hs_vs: TODO
        - vanish_pts: Estimated vanishing points per image
    """
    random.seed(999)
    assert len(rgb_imgs) == len(gray_imgs)
    n_imgs = len(rgb_imgs)
    # SIFT features (Keypoints, Descriptors)
    sift_features = [image_matches.find_features_sift(img, 0) for i, img in enumerate(gray_imgs)]
    vanish_pts = [vp.estimate_vps(img, num_img=i) for i, img in enumerate(gray_imgs)]

    tracks = []   # list of tracking views
    hs_vs = {}    # dictionary as hash table of views-tracks

    img_pairs_features = {}
    imgs_pairs_ids = sorted(set(itertools.combinations(range(n_imgs), 2)))

    for img1_id, img2_id in imgs_pairs_ids:
        print("\t Matching images", img1_id, "and", img2_id, " for obtaining tracks")
        # Can you do better ?
        matches = image_matches.match_features_sift(sift_features[img1_id][1], sift_features[img2_id][1])
        # Get match points projective coordinates in each image
        x1_h, x2_h = [], []
        for m in matches:
            x1_h.append([sift_features[img1_id][0][m[0].queryIdx].pt[0], sift_features[img1_id][0][m[0].queryIdx].pt[1], 1])
            x2_h.append([sift_features[img2_id][0][m[0].trainIdx].pt[0], sift_features[img2_id][0][m[0].trainIdx].pt[1], 1])
    
        x1_h = np.asarray(x1_h).T  # Cam 1 projective coordinates
        x2_h = np.asarray(x2_h).T  # Cam 2 projective coordinates

        # Estimate fundamental matrix, and get inliers
        F, indices_inlier_matches = fundamental.fundamental_matrix_ransac(points1=x1_h, points2=x2_h,
                                                                          threshold=1, max_iterations=100000)
        inlier_matches = itemgetter(*indices_inlier_matches)(matches)
        
        print("Before cleaning:", x1_h.shape, x2_h.shape)
        x1_h, x2_h = x1_h[:, indices_inlier_matches], x2_h[:, indices_inlier_matches]
        print("After cleaning:", x1_h.shape, x2_h.shape)
        
        # Get points image/euclidean coordinates of matches
        x1_e = projective2img(x1_h)
        x2_e = projective2img(x2_h)
        # Refine the coordinates using Optimal Triangulation Method
        xr1_e, xr2_e = fundamental.refine_matches(x1_e.T, x2_e.T, F)
        # Changed: Transpose elements
        tk.add_tracks(x1_e.T, x2_e.T, xr1_e.T, xr2_e.T, img1_id, img2_id, tracks, hs_vs)
        # h.draw_matches_cv(imgs[prev], imgs[i], x1, x2)

        print("  Tracks added after matching", img1_id, "and", img2_id)
        print("    Size of tracks:", len(tracks))
        print("    Size of hash table of views:", len(hs_vs))

        # Save the image pair features
        features = {"x1_e": x1_e, "x2_e": x2_e,
                    "xr1_e": xr1_e, "xr2_e": xr2_e,
                    "x1_h": x1_h, "x2_h": x2_h, "F": F, "vanish_pts": vanish_pts,
                    "matches": matches, "inliers_idx": indices_inlier_matches}
        img_pairs_features[(img1_id, img2_id)] = Namespace(**features)

    return img_pairs_features, sift_features, vanish_pts, tracks, hs_vs

def drawlines(img1, img2, lines, x1, x2, colors=None):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    if not colors:
        colors = np.random.randint(0, 255, size=(x1.shape[0], 3)).tolist()
    h, w = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2, color in zip(lines, x1, x2, colors):
        if np.isnan(np.sum(x1)) or np.isnan(np.sum(x2)):
            continue

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), tuple(color), thickness=1)
        img1 = cv2.circle(img1, tuple(pt1), 6, tuple(color), -1)
        img2 = cv2.circle(img2, tuple(pt2), 6, tuple(color), -1)
    return img1, img2


def draw_matches(img1, img2, x1, x2, colors=None):
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    if not colors:
        colors = np.random.randint(0, 255, size=(x1.shape[0], 3))

    for pt1, pt2, color in zip(x1, x2, colors):
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)
    return img1, img2


def draw_matches_cv(img1, img2, x1, x2):
    kp1 = [cv2.KeyPoint(p[0], p[1], 1) for p in x1]
    kp2 = [cv2.KeyPoint(p[0], p[1], 1) for p in x2]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, np.random.choice(matches, 100), None, flags=2)
    plt.imshow(img3), plt.show()


def display_epilines(img1, img2, x1, x2, F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(x2, 2, F)
    lines1 = lines1.reshape(-1, 3)
    colors = np.random.randint(0, 255, size=(x1.shape[0], 3)).tolist()
    img1l, _ = drawlines(img1, img2, lines1, x1.astype(np.int), x2.astype(np.int), colors=colors)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(x1, 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2l, _ = drawlines(img2, img1, lines2, x2.astype(np.int), x1.astype(np.int), colors=colors)

    return img1l, img2l


def show_matches(img1, img2, x1, x2):
    # Draw matches between two images
    cv2.namedWindow('matches at img1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('matches at img2', cv2.WINDOW_NORMAL)

    img3, img4 = draw_matches(img1, img2, x1, x2)

    if opencv:
        cv2.imshow('matches at img1', img3)
        cv2.imshow('matches at img2', img4)
        # ASCII(q) = 113, ASCII(esc) = 27, ASCII(space) = 32
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 113 or key == 27:
                cv2.destroyWindow('matches at img1')
                cv2.destroyWindow('matches at img2')
                break
    else:
        plt.subplot(121), plt.imshow(img3)
        plt.subplot(122), plt.imshow(img4)
        plt.show()

def display_3d_points(X, x, img, cameras=None):
    x_img = x.astype(int)
    rgb_txt = (img[x_img[:,1], x_img[:,0]])/255
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if not cameras is None:
        for P, w, h, scale in cameras:
            points = common_misc.get_camera_frame_points(P, w, h, scale)
            for l in points:
                ax.plot(xs=[l[0][0], l[1][0]], ys=[l[0][1], l[1][1]], zs=[l[0][2], l[1][2]])
                
    ax.scatter3D(X[:,0], X[:,1], X[:,2], c=rgb_txt)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
        
