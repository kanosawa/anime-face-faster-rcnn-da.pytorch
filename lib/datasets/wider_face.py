# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import PIL
from datasets.imdb import imdb
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import uuid
from model.utils.config import cfg


class wider_face(imdb):
    def __init__(self, image_set):
        """
        WIDER Face data loader
        """
        name = 'wider_face_' + image_set
        imdb.__init__(self, name)
        self._devkit_path = self._get_default_path()  # ./data/WIDER2015
        # ./data/WIDER2015/WIDER_train/images
        self._data_path = os.path.join(self._devkit_path, 'WIDER_' + image_set, 'images')
        # Example path to image set file:
        image_set_file = os.path.join(self._devkit_path, 'wider_face_split', 'wider_face_' + image_set + '.mat')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        self._wider_image_set = sio.loadmat(image_set_file, squeeze_me=True)
        self._classes = ('__background__',  # always index 0
                         'face')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index, self._face_bbx = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'matlab_eval': False,
                       'rpn_file': None}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        event_list = self._wider_image_set['event_list']
        file_list = self._wider_image_set['file_list']
        face_bbx_list = self._wider_image_set['face_bbx_list']
        image_index = []
        face_bbx = []
        for i in range(len(event_list)):
            for j in range(len(file_list[i])):
                image_index.append(str(event_list[i]) + '/' + str(file_list[i][j]))
                face_bbx.append(face_bbx_list[i][j].reshape(-1, 4))
        # _wider_image_set = np.concatenate(_wider_image_set['file_list']).ravel().tolist()
        # image_index = map(str, _wider_image_set)
        return image_index, face_bbx

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'WIDER2015')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_wider_annotation(index)
                    for index in range(len(self.image_index))]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_wider_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        imw, imh = PIL.Image.open(self.image_path_at(index)).size
        num_objs = self._face_bbx[index].shape[0]

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix in range(num_objs):
            assert not np.any(np.isnan(self._face_bbx[index][ix]))
            x1 = min(max(0, self._face_bbx[index][ix][0]), imw - 1)
            y1 = min(max(0, self._face_bbx[index][ix][1]), imh - 1)
            w = abs(self._face_bbx[index][ix][2])
            h = abs(self._face_bbx[index][ix][3])
            x2 = min(max(x1 + w, 0), imw - 1)
            y2 = min(max(y1 + h, 0), imh - 1)
            cls = 1
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (w + 1) * (h + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id
