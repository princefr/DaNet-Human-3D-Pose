




def convert(op_kpts):
    """
    convert openpose keypoints(25) format to coco keypoints(17) format

    0-16 map to 0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11
    :param op_kpts:
    :return:
    """
    coco_keypoints = []
    for i, j in enumerate([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]):
        score = op_kpts[j][-1]
        if score < 0.2 and j in [15, 16, 17, 18]:
            coco_keypoints.append(op_kpts[0])
        else:
            coco_keypoints.append(op_kpts[j])

    return coco_keypoints







def convert_18(op_keypoints):
    """

    convert openpose keypoints(25) format to keypoints (18)

    :param op_keypoints:
    :return:
    """

    coco_keypoints = []
    for i, j in enumerate(range(0,18)):
        if i<8:
            coco_keypoints.append(op_keypoints[j])
        else:
            coco_keypoints.append(op_keypoints[j +1])
    return coco_keypoints