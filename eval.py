#!/usr/bin/env python
import numpy as np
from ResultMerge import mergebypoly, mergebyrec
import os, sys
import cv2
from distutils.util import strtobool

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_colormap(n_colors):
    """ Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color extracted by the HSV colormap """

    import matplotlib.cm as cmx
    from matplotlib import colors

    color_norm = colors.Normalize(vmin=0, vmax=n_colors-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb(index):
        """ closure mapping index to color - range [0 255] for OpenCV """
        return 255 * np.array(scalar_map.to_rgba(index))

    return map_index_to_rgb

def draw_detection(im_name, detections):
    """ draw bounding boxes in the form (class, box, score) """
    
    im = cv2.imread(im_name)
    for j, name in enumerate(detections):
        if name == '__background__':
            continue
        #color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        color = cmap(j)
        dets = detections[name]
        for det in dets:
            bbox = det[:-1]
            score = det[-1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            #cv2.putText(im, '%s %.2f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
            #            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

def draw_quadrangle_detection(im_name, detections, cmap):
    """
    visualize all detections in one image
    """
    import random
    im = cv2.imread(im_name)
    #color_white = (255, 255, 255)
    for j, name in enumerate(detections):
        if name == '__background__':
            continue
        #color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        color = cmap(j)
        dets = detections[name]
        for det in dets:
            bbox = det[:-1]
            score = det[-1]
            bbox = map(int, bbox)
            for i in range(3):
                cv2.line(im, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=color, thickness=2)
            cv2.line(im, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2)
            #cv2.putText(im, '%s %.2f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
            #            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im

def draw_merged(img_result, img_dir, out_dir, task=1):
    cmap = get_colormap(15)
    for imgname, dets in img_result.items():
        im_name = img_dir + '/' + imgname + '.png'
        if task == 1: im = draw_quadrangle_detection(im_name, dets, cmap)
        else: im = draw_detection(im_name, dets, cmap)
        out_name = out_dir + '/' + imgname + '.png'
        cv2.imwrite(out_name, im)

def eval_merged_result(model_output_path, dota_home=r'../DOTA', task=1):
    if task == 1:
        from dota_evaluation_task1 import voc_eval
        detpath = model_output_path + '/merge_results/Task1_{:s}.txt'
    else:
        from dota_evaluation_task2 import voc_eval
        detpath = model_output_path + '/merge_results/Task2_{:s}.txt'

    annopath = dota_home + '/val/labelTxt/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = dota_home + '/valset.txt'
    
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    if 'c6' in model_output_path:
        classnames = ['plane','ship','storage-tank','harbor','bridge','large-vehicle']
    
    classaps = []
    aps = []
    map = 0
    for classname in classnames:
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        #print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        #print '%s: %.2f'%(classname,ap*100)
        classaps.append(ap)
        aps.append('%.2f'%(ap*100))
    
        # show p-r curve of each category
        show_pr = False
        if show_pr:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,4))
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.plot(rec, prec, lw=2, label='Precision-recall curve of class {} (area = {:.4f})'.format(cls, ap)) # plt.plot(rec, prec)
            # plt.show()
            plt.savefig('PR-curv-%s.png'%classname)

    map = map/len(classnames)
    print('Task{} mAP = \x1b[0;32;40m{}\x1b[0m'.format(task, '%.2f'%(map * 100)))
    #classaps = 100 * np.array(classaps)
    #print('classaps: ', classaps)
    print '\t'.join(classnames)
    print '\t\t'.join(aps)

def start_eval(model_output_path, task=1, draw=False, dota_home=r'../DOTA'):
    '''
    merge and eval, the detection result files by class locate in model_output_path
    model_output_path = r'../DCR/output/rcnn/DOTA_quadrangle/DOTA_quadrangle/val'
    '''
    print 'start merge result...'
    img_dir =  dota_home + '/val/images'
    dstpath = model_output_path + '/merge_results'
    out_img_dir = model_output_path + '/vis'
    mkdir(dstpath)
    mkdir(out_img_dir)
    if task == 1:
        srcpath = model_output_path + '/test_results_by_class_task1'
        img_result = mergebypoly(srcpath, dstpath)
    else:
        srcpath = model_output_path + '/test_results_by_class_task2'
        img_result = mergebyrec(srcpath, dstpath)

    if draw:
        print 'start draw...'
        draw_merged(img_result, img_dir, out_img_dir, task=task)
    print 'start eval...'
    # the dota_home include /val/labelTxt/ & valset.txt(which include big image id)
    eval_merged_result(model_output_path, dota_home=dota_home, task=task)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: eval model_output_path [task 1/2, default 1] [draw, default false]'
        sys.exit(0)

    model_output_path = sys.argv[1]
    if len(sys.argv) > 2: task = int(sys.argv[2])
    else: task = 1
    if len(sys.argv) > 3: draw = strtobool(sys.argv[3])
    else: draw = False

    start_eval(model_output_path, task=task, draw=draw)

