from __future__ import division
import numpy as np
import onnxruntime
import os
import cv2
import logging
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger(__name__)

def softmax(z):
    """Improved softmax with better numerical stability"""
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    """Enhanced bbox calculation with better boundary handling"""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Enhanced keypoints calculation with improved precision"""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
            
        preds.append(px)
        preds.append(py)
    
    return np.stack(preds, axis=-1)

class ResnetFace:
    def __init__(self, model_file=None, session=None, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        """Enhanced initialization with better error handling"""
        try:
            self.model_file = model_file
            self.session = session
            self.taskname = 'detection'
            
            if self.session is None:
                assert self.model_file is not None, "Model file path is None."
                assert os.path.exists(self.model_file), "RetinaFace weights not found."
                self.session = onnxruntime.InferenceSession(self.model_file, providers=providers)
            
            self.center_cache = {}
            self.nms_thresh = 0.4
            self.det_thresh = 0.5
            self._init_vars()
            
        except Exception as e:
            logger.error(f"Error initializing RetinaFace: {str(e)}")
            raise

    def _init_vars(self):
        """Enhanced initialization of model variables"""
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
            
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.input_mean = 127.5
        self.input_std = 128.0
      
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        
        # Enhanced model configuration based on outputs
        if len(outputs)==6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs)==9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs)==10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs)==15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        """Enhanced preparation with better parameter handling"""
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
            
        # Update parameters if provided
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
            
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
            
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                logger.warning('det_size is already set in detection model, ignore')
            else:
                self.input_size = input_size

    def forward(self, img, threshold):
        """Enhanced forward pass with improved feature handling"""
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        # Prepare input
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 
            1.0/self.input_std, 
            input_size, 
            (self.input_mean, self.input_mean, self.input_mean), 
            swapRB=True
        )
        
        # Run inference
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        
        # Process each feature stride
        for idx, stride in enumerate(self._feat_stride_fpn):
            # Get predictions
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc] * stride
            
            if self.use_kps:
                kps_preds = net_outs[idx+fmc*2] * stride
                
            # Calculate grid
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            
            # Get or create anchor centers
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # Enhanced anchor center calculation
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape((-1, 2))
                    
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            # Filter predictions
            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
                
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=0, metric='default'):
        """Enhanced detection with improved face filtering"""
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        # Calculate image ratios and sizes
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
            
        det_scale = float(new_height) / img.shape[0]
        
        # Resize and pad image
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        # Forward pass
        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        # Process detections
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
            
        # Apply NMS
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        
        if self.use_kps:
            kpss = kpss[order,:,:]
            kpss = kpss[keep,:,:]
        else:
            kpss = None

        # Filter by max_num if needed
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area if metric == 'max' else area - offset_dist_squared * 2.0
            
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex, :]
            
            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss

    def nms(self, dets):
        """Enhanced NMS with improved overlap calculation"""
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate overlaps
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            # Enhanced IoU calculation
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep