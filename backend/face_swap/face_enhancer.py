# import numpy as np
# import cv2
# import logging
# import onnxruntime
# import os
# from face_swap.utils.common import Face
# from config import FACE_ENHANCER_MODEL_PATH
# logger = logging.getLogger(__name__)

# _global_enhancer = None

# def get_global_enhancer():
#     global _global_enhancer
#     if _global_enhancer is None:
#         try:
#             session_options = onnxruntime.SessionOptions()
#             session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
#             session_options.enable_mem_pattern = False
#             session_options.enable_cpu_mem_arena = False
#             session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            
#             providers_to_try = [
#                 [
#                     ('CUDAExecutionProvider', {
#                         'device_id': 0,
#                         'arena_extend_strategy': 'kSameAsRequested',
#                         'gpu_mem_limit': 512 * 1024 * 1024,
#                         'cudnn_conv_algo_search': 'HEURISTIC',
#                         'do_copy_in_default_stream': True,
#                         'cudnn_conv_use_max_workspace': '0',
#                     }),
#                     'CPUExecutionProvider'
#                 ],
#                 [
#                     ('CUDAExecutionProvider', {
#                         'device_id': 0,
#                         'arena_extend_strategy': 'kSameAsRequested',
#                         'gpu_mem_limit': 256 * 1024 * 1024,
#                         'cudnn_conv_algo_search': 'HEURISTIC',
#                     }),
#                     'CPUExecutionProvider'
#                 ],
#                 ['CPUExecutionProvider']
#             ]
            
#             for providers in providers_to_try:
#                 try:
#                     _global_enhancer = onnxruntime.InferenceSession(
#                         FACE_ENHANCER_MODEL_PATH, 
#                         sess_options=session_options, 
#                         providers=providers
#                     )
#                     provider_info = _global_enhancer.get_providers()[0]
#                     logger.info(f"Global face enhancer initialized with {provider_info}")
#                     break
#                 except Exception as provider_error:
#                     logger.warning(f"Failed to initialize with {providers[0] if isinstance(providers[0], tuple) else providers[0]}: {str(provider_error)}")
#                     continue
#             else:
#                 raise Exception("Failed to initialize enhancer with any provider")
                
#         except Exception as e:
#             logger.error(f"Error initializing global enhancer: {str(e)}")
#             raise
#     return _global_enhancer

# class GFPGAN:
#     def __init__(self, model_path=None, providers=None):
#         if model_path is None:
#             model_path = os.environ.get('GFPGAN_MODEL_PATH', 'models/GFPGANv1.4.onnx')
            
#         if providers is None:
#             providers = [
#                 ('CUDAExecutionProvider', {
#                     'device_id': 0,
#                     'arena_extend_strategy': 'kSameAsRequested',
#                     'gpu_mem_limit': 256 * 1024 * 1024,
#                     'cudnn_conv_algo_search': 'HEURISTIC',
#                     'cudnn_conv_use_max_workspace': '0',
#                 }),
#                 'CPUExecutionProvider'
#             ]
            
#         try:
#             session_options = onnxruntime.SessionOptions()
#             session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
#             session_options.enable_mem_pattern = False
#             session_options.enable_cpu_mem_arena = False
#             session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            
#             try:
#                 self.session = onnxruntime.InferenceSession(
#                     path_or_bytes=model_path, 
#                     sess_options=session_options, 
#                     providers=providers
#                 )
#                 logger.info("GFPGAN initialized with GPU")
#             except:
#                 self.session = onnxruntime.InferenceSession(
#                     path_or_bytes=model_path, 
#                     sess_options=session_options, 
#                     providers=['CPUExecutionProvider']
#                 )
#                 logger.info("GFPGAN initialized with CPU (fallback)")
            
#             model_inputs = self.session.get_inputs()
#             self.input_name = model_inputs[0].name
#             self.input_shape = model_inputs[0].shape
#             self.resolution = self.input_shape[-2:] if len(self.input_shape) >= 3 else (512, 512)
            
#             logger.info(f"GFPGAN model loaded with resolution {self.resolution}")
            
#         except Exception as e:
#             logger.error(f"Error initializing GFPGAN model: {str(e)}")
#             raise
    
#     def enhance(self, face_image):
#         try:
#             if face_image.shape[:2] != self.resolution:
#                 face_image_resized = cv2.resize(
#                     face_image, 
#                     (self.resolution[1], self.resolution[0]), 
#                     interpolation=cv2.INTER_LINEAR
#                 )
#             else:
#                 face_image_resized = face_image
            
#             face_image_resized = face_image_resized.astype(np.float32, copy=False)
#             face_image_resized *= (1.0 / 255.0)
#             face_image_resized = (face_image_resized - 0.5) * 2.0
            
#             face_image_resized = face_image_resized[:, :, ::-1]
#             face_image_resized = np.transpose(face_image_resized, (2, 0, 1))
#             face_image_resized = np.expand_dims(face_image_resized, axis=0)
            
#             output = self.session.run(None, {self.input_name: face_image_resized})[0]
            
#             output = np.squeeze(output, axis=0)
#             output = np.clip(output, -1, 1, out=output)
#             output = (output + 1) * 127.5
#             output = output.transpose(1, 2, 0)
#             output = output[:, :, ::-1]
#             output = output.astype(np.uint8)
            
#             if face_image.shape[:2] != output.shape[:2]:
#                 output = cv2.resize(
#                     output, 
#                     (face_image.shape[1], face_image.shape[0]),
#                     interpolation=cv2.INTER_LINEAR
#                 )
                
#             return output
            
#         except Exception as e:
#             logger.error(f"Error in GFPGAN enhancement: {str(e)}")
#             return face_image

# def blend_frame_fast(temp_frame, paste_frame, blend_factor=0.5):
#     cv2.addWeighted(temp_frame, blend_factor, paste_frame, 1 - blend_factor, 0, temp_frame)
#     return temp_frame

# def paste_back_fast(temp_frame, crop_frame, affine_matrix):
#     try:
#         inverse_affine_matrix = cv2.invertAffineTransform(affine_matrix)
#         temp_frame_height, temp_frame_width = temp_frame.shape[:2]
#         crop_frame_height, crop_frame_width = crop_frame.shape[:2]

#         inverse_crop_frame = cv2.warpAffine(
#             crop_frame, 
#             inverse_affine_matrix, 
#             (temp_frame_width, temp_frame_height),
#             flags=cv2.INTER_LINEAR,
#             borderMode=cv2.BORDER_TRANSPARENT
#         )
        
#         mask = np.ones((crop_frame_height, crop_frame_width), dtype=np.float32)
#         inverse_mask = cv2.warpAffine(
#             mask,
#             inverse_affine_matrix,
#             (temp_frame_width, temp_frame_height),
#             flags=cv2.INTER_LINEAR
#         )
        
#         inverse_mask = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)
        
#         inverse_mask = cv2.GaussianBlur(inverse_mask, (5, 5), 0)
        
#         temp_frame = inverse_mask * inverse_crop_frame + (1 - inverse_mask) * temp_frame
#         return temp_frame.astype(np.uint8)

#     except Exception as e:
#         logger.error(f"Error in paste_back_fast: {str(e)}")
#         return temp_frame

# def normalize_crop_frame_fast(crop_frame):
#     try:
#         crop_frame = np.clip(crop_frame, -1, 1, out=crop_frame)
#         crop_frame = (crop_frame + 1) * 127.5
#         crop_frame = crop_frame.transpose(1, 2, 0)
#         crop_frame = crop_frame.astype(np.uint8)[:, :, ::-1]
#         return crop_frame
#     except Exception as e:
#         logger.error(f"Error in normalize_crop_frame_fast: {str(e)}")
#         return None

# def prepare_crop_frame_fast(crop_frame):
#     try:
#         crop_frame = crop_frame.astype(np.float32, copy=False)
#         crop_frame = crop_frame[:, :, ::-1] * (1.0 / 255.0)
#         crop_frame = (crop_frame - 0.5) * 2.0
#         crop_frame = np.transpose(crop_frame, (2, 0, 1))
#         crop_frame = np.expand_dims(crop_frame, axis=0)
#         return crop_frame
#     except Exception as e:
#         logger.error(f"Error in prepare_crop_frame_fast: {str(e)}")
#         return None

# def warp_face_fast(target_face: Face, temp_frame):
#     try:
#         template = np.array([
#             [192.98138, 239.94708],
#             [318.90277, 240.1936],
#             [256.63416, 314.01935],
#             [201.26117, 371.41043],
#             [313.08905, 371.15118]
#         ], dtype=np.float32)

#         affine_matrix = cv2.estimateAffinePartial2D(
#             target_face['kps'].astype(np.float32),
#             template,
#             method=cv2.LMEDS
#         )[0]

#         crop_frame = cv2.warpAffine(
#             temp_frame, 
#             affine_matrix, 
#             (512, 512),
#             flags=cv2.INTER_LINEAR
#         )
#         return crop_frame, affine_matrix

#     except Exception as e:
#         logger.error(f"Error in warp_face_fast: {str(e)}")
#         return None, None

# _gfpgan_instance = None

# def enhance_face_ultra_fast(target_face: Face, temp_frame, face_enhancer_model=None, enhancement_settings=None, use_gfpgan=False):
#     global _gfpgan_instance
    
#     if face_enhancer_model is None:
#         try:
#             face_enhancer_model = get_global_enhancer()
#         except Exception as e:
#             logger.error(f"Failed to get global enhancer: {str(e)}")
#             return temp_frame
    
#     crop_frame, affine_matrix = warp_face_fast(target_face, temp_frame)
    
#     if crop_frame is None or affine_matrix is None:
#         return temp_frame
    
#     original_crop_frame = crop_frame.copy() if use_gfpgan else None
    
#     try:
#         crop_frame_prepared = prepare_crop_frame_fast(crop_frame)
#         if crop_frame_prepared is None:
#             return temp_frame
        
#         if enhancement_settings:
#             smoothness = enhancement_settings.smoothness / 100.0
            
#             if smoothness < 0.3:
#                 kernel_size = int((1 - smoothness) * 3) * 2 + 1
#                 if kernel_size > 1:
#                     crop_frame_prepared = cv2.GaussianBlur(crop_frame_prepared, (kernel_size, kernel_size), 0)
        
#         try:
#             crop_frame = face_enhancer_model.run(None, {'input': crop_frame_prepared})[0][0]
#             crop_frame = normalize_crop_frame_fast(crop_frame)
            
#             if crop_frame is None:
#                 logger.warning("Enhancement normalization failed, using original")
#                 return temp_frame
                
#         except Exception as enhance_error:
#             logger.error(f"Base enhancement failed: {str(enhance_error)}")
#             try:
#                 import gc
#                 gc.collect()
#                 if 'CUDAExecutionProvider' in [p for p in face_enhancer_model.get_providers()]:
#                     logger.info("Attempting GPU memory cleanup and retry")
#                 crop_frame = face_enhancer_model.run(None, {'input': crop_frame_prepared})[0][0]
#                 crop_frame = normalize_crop_frame_fast(crop_frame)
#             except Exception as retry_error:
#                 logger.error(f"Enhancement retry failed: {str(retry_error)}")
#                 return temp_frame
        
#         if use_gfpgan and _gfpgan_instance is not None and original_crop_frame is not None:
#             try:
#                 gfpgan_enhanced = _gfpgan_instance.enhance(original_crop_frame)
#                 crop_frame = cv2.addWeighted(crop_frame, 0.7, gfpgan_enhanced, 0.3, 0)
#             except Exception as gfpgan_error:
#                 logger.error(f"GFPGAN enhancement failed: {str(gfpgan_error)}")
        
#         paste_frame = paste_back_fast(temp_frame, crop_frame, affine_matrix)
        
#         temp_frame = blend_frame_fast(temp_frame, paste_frame, 0.5)
        
#         return temp_frame
        
#     except Exception as outer_error:
#         logger.error(f"Complete enhancement failed: {str(outer_error)}")
#         return temp_frame

# def enhance_face(target_face: Face, temp_frame, face_enhancer_model, enhancement_settings=None, use_gfpgan=False):
#     return enhance_face_ultra_fast(target_face, temp_frame, face_enhancer_model, enhancement_settings, use_gfpgan)

# def blend_frame(temp_frame, paste_frame):
#     return blend_frame_fast(temp_frame, paste_frame)

# def paste_back(temp_frame, crop_frame, affine_matrix):
#     return paste_back_fast(temp_frame, crop_frame, affine_matrix)

# def normalize_crop_frame(crop_frame):
#     return normalize_crop_frame_fast(crop_frame)

# def prepare_crop_frame(crop_frame):
#     return prepare_crop_frame_fast(crop_frame)

# def warp_face(target_face: Face, temp_frame):
#     return warp_face_fast(target_face, temp_frame)

import numpy as np
import cv2
import logging
from face_swap.utils.common import Face

logger = logging.getLogger(__name__)

def blend_frame(temp_frame, paste_frame):
    """Blend frames with fixed blend factor"""
    face_enhancer_blend = 0.5
    temp_frame = cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)
    return temp_frame

def paste_back(temp_frame, crop_frame, affine_matrix):
    """Paste the enhanced face back to the original frame"""
    try:
        inverse_affine_matrix = cv2.invertAffineTransform(affine_matrix)
        temp_frame_height, temp_frame_width = temp_frame.shape[0:2]
        crop_frame_height, crop_frame_width = crop_frame.shape[0:2]

        # Create masks
        inverse_mask = np.ones((crop_frame_height, crop_frame_width, 3), dtype=np.float32)
        
        # Warp crop frame and mask
        inverse_crop_frame = cv2.warpAffine(
            crop_frame, 
            inverse_affine_matrix, 
            (temp_frame_width, temp_frame_height)
        )
        inverse_mask_frame = cv2.warpAffine(
            inverse_mask,
            inverse_affine_matrix,
            (temp_frame_width, temp_frame_height)
        )

        # Process mask
        inverse_mask_frame = cv2.erode(inverse_mask_frame, np.ones((2, 2)))
        inverse_mask_border = inverse_mask_frame * inverse_crop_frame
        inverse_mask_area = np.sum(inverse_mask_frame) // 3
        inverse_mask_edge = int(inverse_mask_area ** 0.5) // 20
        inverse_mask_radius = inverse_mask_edge * 2

        # Create and process center mask
        inverse_mask_center = cv2.erode(inverse_mask_frame, np.ones((inverse_mask_radius, inverse_mask_radius)))
        inverse_mask_blur_size = inverse_mask_edge * 2 + 1
        inverse_mask_blur_area = cv2.GaussianBlur(inverse_mask_center, (inverse_mask_blur_size, inverse_mask_blur_size), 0)

        # Blend frames
        temp_frame = inverse_mask_blur_area * inverse_mask_border + (1 - inverse_mask_blur_area) * temp_frame
        temp_frame = temp_frame.clip(0, 255).astype(np.uint8)
        
        return temp_frame

    except Exception as e:
        logger.error(f"Error in paste_back: {str(e)}")
        return temp_frame

def normalize_crop_frame(crop_frame):
    """Normalize the crop frame for processing"""
    try:
        crop_frame = np.clip(crop_frame, -1, 1)
        crop_frame = (crop_frame + 1) / 2
        crop_frame = crop_frame.transpose(1, 2, 0)
        crop_frame = (crop_frame * 255.0).round()
        crop_frame = crop_frame.astype(np.uint8)[:, :, ::-1]
        return crop_frame
    except Exception as e:
        logger.error(f"Error in normalize_crop_frame: {str(e)}")
        return None

def prepare_crop_frame(crop_frame):
    """Prepare the crop frame for the model"""
    try:
        # Convert to float32 for better precision
        crop_frame = crop_frame.astype(np.float32)
        
        # Normalize and prepare for model input
        crop_frame = crop_frame[:, :, ::-1] / 255.0
        crop_frame = (crop_frame - 0.5) / 0.5
        crop_frame = np.expand_dims(crop_frame.transpose(2, 0, 1), axis=0)
        return crop_frame
    except Exception as e:
        logger.error(f"Error in prepare_crop_frame: {str(e)}")
        return None

def warp_face(target_face: Face, temp_frame):
    """Warp the face for enhancement"""
    try:
        template = np.array([
            [192.98138, 239.94708],
            [318.90277, 240.1936],
            [256.63416, 314.01935],
            [201.26117, 371.41043],
            [313.08905, 371.15118]
        ])

        affine_matrix = cv2.estimateAffinePartial2D(
            target_face['kps'],
            template,
            method=cv2.LMEDS
        )[0]

        crop_frame = cv2.warpAffine(temp_frame, affine_matrix, (512, 512))
        return crop_frame, affine_matrix

    except Exception as e:
        logger.error(f"Error in warp_face: {str(e)}")
        return None, None

def enhance_face(target_face: Face, temp_frame, face_enhancer_model, enhancement_settings=None):
    frame_processor = face_enhancer_model
    crop_frame, affine_matrix = warp_face(target_face, temp_frame)
    crop_frame = prepare_crop_frame(crop_frame)
    
    # Default settings if none provided
    smoothness = 0.5  # Default 50%
    color_match = 0.5  # Default 50%
    sharpness = 0.5   # Default 50%
    
    # Apply custom settings if provided
    if enhancement_settings:
        smoothness = enhancement_settings.smoothness / 100.0
        color_match = enhancement_settings.colorMatch / 100.0
        sharpness = enhancement_settings.sharpness / 100.0
    
    frame_processor_inputs = {
        'input': crop_frame
    }
    
    # Apply enhancement settings
    if enhancement_settings:
        # Apply smoothness
        kernel_size = int((1 - smoothness) * 5) * 2 + 1  # Odd number from 1 to 9
        if kernel_size > 1:
            crop_frame = cv2.GaussianBlur(crop_frame, (kernel_size, kernel_size), 0)
            
        # Apply color matching
        if color_match > 0:
            target_mean = cv2.mean(temp_frame)
            current_mean = cv2.mean(crop_frame)
            crop_frame = cv2.addWeighted(
                crop_frame, 
                1 - color_match,
                np.full_like(crop_frame, target_mean), 
                color_match,
                0
            )
            
        # Apply sharpness
        if sharpness > 0.5:  # Only sharpen if value is above default
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * (sharpness - 0.5) * 2
            crop_frame = cv2.filter2D(crop_frame, -1, kernel)
    
    # Process the frame
    crop_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    crop_frame = normalize_crop_frame(crop_frame)
    paste_frame = paste_back(temp_frame, crop_frame, affine_matrix)
    temp_frame = blend_frame(temp_frame, paste_frame)
    
    return temp_frame