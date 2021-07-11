import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

def get_sift_stereo_image_by_cv(img1, img2):
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    FLANN_INDEX_KDTREE = 0 
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

    retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
    mask = mask.ravel()

    pts1 = src_pts[mask == 1]
    pts2 = dst_pts[mask == 1]

    return pts1.T, pts2.T

def get_skew_transform(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def get_P_from_E_mat(E):

    U, S, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T, np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T, np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s

def get_corresp_mat(p1, p2):
    p1x, p1y = p1[:2]
    p2x, p2y = p2[:2]

    return np.array([p1x * p2x, p1x * p2y, p1x,
        p1y * p2x, p1y * p2y, p1y, p2x, p2y, np.ones(len(p1x))
    ]).T

    return np.array([p2x * p1x, p2x * p1y, p2x,
        p2y * p1x, p2y * p1y, p2y, p1x, p1y, np.ones(len(p1x))
    ]).T

def get_image_to_image_mat(x1, x2, compute_essential=False):
    A = get_corresp_mat(x1, x2)
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    if compute_essential:
        S = [1, 1, 0] 
    F = np.dot(U, np.dot(np.diag(S), V))

    return F

def get_scale_and_trans_pt(points):

    x = points[0]
    y = points[1]
    center = points.mean(axis=1) 
    cx = x - center[0] 
    cy = y - center[1]
    distance = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / distance.mean()
    norm_3d = np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]],[0, 0, 1]
    ])

    return np.dot(norm_3d, points), norm_3d

def get_normal_img_to_img_mat(p1, p2, compute_essential=True):
    n = p1.shape[1]

    p1_n, T1 = get_scale_and_trans_pt(p1)
    p2_n, T2 = get_scale_and_trans_pt(p2)

    F = get_image_to_image_mat(p1_n, p2_n, compute_essential)
    F = np.dot(T1.T, np.dot(F, T2))

    return F / F[2, 2]

def get_homo_XY(arr):
    
    if arr.ndim == 1:
        return np.hstack([arr, 1])
    return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))

def get_linear_triang(p1, p2, m1, m2):

    num_points = p1.shape[1]
    res = np.ones((4, num_points))

    for i in range(num_points):
        A = np.asarray([
            (p1[0, i] * m1[2, :] - m1[0, :]), (p1[1, i] * m1[2, :] - m1[1, :]), (p2[0, i] * m2[2, :] - m2[0, :]), (p2[1, i] * m2[2, :] - m2[1, :])
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1, :4]
        res[:, i] = X / X[3]

    return res

def get_reconst_1pt(pt1, pt2, m1, m2):

    A = np.vstack([np.dot(get_skew_transform(pt1), m1),np.dot(get_skew_transform(pt2), m2)])
    U, S, V = np.linalg.svd(A)
    P = np.ravel(V[-1, :4])

    return P / P[3]

def get_P_from_E_mat(E):

    U, S, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T, np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2s

def main():

    input_src1 = cv2.imread('input1.jpeg')
    input_src2 = cv2.imread('input2.jpeg')

    pts1, pts2 = get_sift_stereo_image_by_cv(input_src1, input_src2)
    point_s1 = get_homo_XY(pts1)
    point_s2 = get_homo_XY(pts2)

    height, width, ch = input_src1.shape
    intrinsic_mat = np.array([[2360, 0, width / 2], [0, 2360, height / 2], [0, 0, 1]])

    return point_s1, point_s2, intrinsic_mat

pts1, pts2, intrinsic_mat = main()

pt_1n = np.dot(np.linalg.inv(intrinsic_mat), pts1)
pt_2n = np.dot(np.linalg.inv(intrinsic_mat), pts2)

E = get_normal_img_to_img_mat(pt_1n, pt_2n)
P1 = np.array([ [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]])
P2s = get_P_from_E_mat(E)

ind = -1
for i, P2 in enumerate(P2s):

    d1 = get_reconst_1pt(pt_1n[:, 0], pt_2n[:, 0], P1, P2)
    homo_p2 = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
    d2 = np.dot(homo_p2[:3, :4], d1)

    if d1[2] > 0 and d2[2] > 0:
        ind = i

P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
p3d = get_linear_triang(pt_1n, pt_2n, P1, P2)

fig = plt.figure()
fig.suptitle('Model', fontsize=12)
pax = fig.gca(projection='3d')
pax.plot(p3d[0], p3d[1], p3d[2], 'b.')

pax.set_xlabel('X-axis')
pax.set_ylabel('Y-axis')
pax.set_zlabel('Z-axis')

pax.view_init(elev=180, azim=150)
plt.show()
