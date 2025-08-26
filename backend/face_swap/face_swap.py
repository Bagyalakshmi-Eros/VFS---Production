# import cv2
# import onnxruntime
# import numpy as np
# import os
# import json
# import logging
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Dict, List, Tuple
# from moviepy.editor import VideoFileClip


# from face_swap.utils.common import Face
# from face_swap.resnetface import ResnetFace 
# from face_swap.arcface_onnx import ArcFaceONNX
# from face_swap.inswapper import INSwapper
# from face_swap.face_enhancer import enhance_face
# from config import UPLOAD_FOLDER, BASE_DIR
# from config import RESNETFACE_MODEL_PATH, ARCFACE_MODEL_PATH, FACE_SWAPPER_MODEL_PATH, FACE_ENHANCER_MODEL_PATH
# from config import PROVIDERS

# from concurrent.futures import ThreadPoolExecutor
# # Get the number of available CPU cores
# cpu_cores = os.cpu_count()

# # Set the number of workers, with a minimum of 16 and max equal to the number of available CPU cores
# min_workers = 16
# max_workers = max(min_workers, cpu_cores)

# # Initialize the ThreadPoolExecutor with the dynamic max_workers
# faceswap_executor = ThreadPoolExecutor(max_workers=max_workers)

# print(f"Using {max_workers} workers for face swap execution.")

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# try:
#     retinaface_det_model = ResnetFace(RESNETFACE_MODEL_PATH, providers=PROVIDERS)
#     retinaface_det_model.prepare(ctx_id=1, input_size=(640, 640), det_thresh=0.7)
#     arcface_emedding_model = ArcFaceONNX(ARCFACE_MODEL_PATH, providers=PROVIDERS)
#     face_swapper_model = INSwapper(FACE_SWAPPER_MODEL_PATH, providers=PROVIDERS)
#     face_enhancer_model = onnxruntime.InferenceSession(FACE_ENHANCER_MODEL_PATH, providers=PROVIDERS)
#     logger.info("Models loaded successfully")
# except Exception as e:
#     logger.error(f"Error initializing models: {str(e)}")
#     raise

# def process_face(frame: np.ndarray, bbox: np.ndarray, kps: np.ndarray, det_score: float, frame_number: int) -> Face:
#     face = Face(bbox=bbox, kps=kps, det_score=det_score)
#     face['frame_number'] = frame_number
#     face['embedding'] = arcface_emedding_model.get(frame, face)
#     return face

# def crop_faces(video_path: str, uid: str):
#     try:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Failed to open video file: {video_path}")

#         directory = os.path.join(UPLOAD_FOLDER, uid, "cropped_faces")
#         os.makedirs(directory, exist_ok=True)

#         face_distance = 1.5
#         embeddings = []
#         frame_number = -1
#         all_bboxes = []
#         all_kps = []
#         all_face_info = {}
#         filename_counter = 1
#         all_faces = {0: []}
#         unique_id = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_number += 1
            
#             try:
#                 bboxes, kpss = retinaface_det_model.detect(frame, max_num=0, metric='default')
                
#                 if bboxes is None or len(bboxes) == 0:
#                     continue

#                 for i in range(bboxes.shape[0]):
#                     bbox = bboxes[i, 0:4]
#                     det_score = bboxes[i, 4]
#                     x1, y1, x2, y2 = bbox.astype(int)
                    
#                     if (x2 - x1) <= 80 or (y2 - y1) <= 80:
#                         continue

#                     kps = None if kpss is None else kpss[i]
#                     face = process_face(frame, bbox, kps, det_score, frame_number)
                    
#                     if unique_id == 0:
#                         sim_id = 0
#                         unique_id += 1
#                     else:
#                         max_sim = 0
#                         max_index = unique_id
                        
#                         for group_id, faces in all_faces.items():
#                             if not faces:
#                                 continue
                            
#                             similarities = np.array([
#                                 np.sum(np.square(face.normed_embedding - known_face.normed_embedding))
#                                 for known_face in faces
#                             ])
#                             group_sim = np.mean(similarities < face_distance)
                            
#                             if group_sim > max_sim:
#                                 max_sim = group_sim
#                                 max_index = group_id
                        
#                         sim_id = max_index if max_sim > 0.25 else unique_id
#                         if sim_id == unique_id:
#                             unique_id += 1

#                     face['group_id'] = sim_id

#                     if sim_id in all_faces:
#                         all_faces[sim_id].append(face)
#                     else:
#                         all_faces[sim_id] = [face]
                    
#                     try:
#                         face_crop = frame[y1:y2, x1:x2]
#                         if face_crop.size == 0:
#                             continue
                            
#                         sim_id_directory = os.path.join(directory, str(sim_id))
#                         os.makedirs(sim_id_directory, exist_ok=True)
                        
#                         filename = f"images_{filename_counter}.jpg"
#                         cv2.imwrite(os.path.join(sim_id_directory, filename), face_crop)
#                         filename_counter += 1
#                     except Exception as e:
#                         logger.warning(f"Error saving face crop: {str(e)}")
#                         continue

#                     if frame_number in all_face_info:
#                         if face['group_id'] in all_face_info[frame_number]:
#                             all_face_info[frame_number][face['group_id']].append(len(embeddings))
#                         else:
#                             all_face_info[frame_number][face['group_id']] = [len(embeddings)]
#                     else:
#                         all_face_info[frame_number] = {face['group_id']: [len(embeddings)]}

#                     embeddings.append(face['embedding'])
#                     all_bboxes.append(face['bbox'])
#                     all_kps.append(face['kps'])

#             except Exception as e:
#                 logger.error(f"Error processing frame {frame_number}: {str(e)}")
#                 continue

#         try:
#             np.save(os.path.join(UPLOAD_FOLDER, uid, 'face_embeddings.npy'), embeddings)
#             np.save(os.path.join(UPLOAD_FOLDER, uid, 'face_bboxes.npy'), all_bboxes)
#             np.save(os.path.join(UPLOAD_FOLDER, uid, 'face_kps.npy'), all_kps)

#             info = {'max_groups': unique_id, 'all_face_info': all_face_info}
#             with open(os.path.join(UPLOAD_FOLDER, uid, 'all_info.json'), 'w') as file:
#                 json.dump(info, file, indent=4)
#         except Exception as e:
#             logger.error(f"Error saving processed data: {str(e)}")
#             raise

#     except Exception as e:
#         logger.error(f"Error in crop_faces: {str(e)}")
#         raise
#     finally:
#         if cap is not None:
#             cap.release()

# def process_frame_batch(batch_data):
#     batch_frames, batch_indices, frame_face_info, group_ids, group_new_faces, all_embeddings, all_bboxes, all_kps = batch_data
    
#     processed_batch = []
    
#     try:
#         for i, (frame_idx, frame) in enumerate(zip(batch_indices, batch_frames)):
#             try:
#                 processed_frame = frame.copy()
                
#                 if str(frame_idx) in frame_face_info:
#                     for group_id in frame_face_info[str(frame_idx)]:
#                         if int(group_id) in group_ids:
#                             for gi in frame_face_info[str(frame_idx)][str(group_id)]:
#                                 old_face = Face(bbox=all_bboxes[gi], kps=all_kps[gi])
#                                 old_face['embedding'] = all_embeddings[gi]
                                
#                                 processed_frame = face_swapper_model.get(
#                                     processed_frame, old_face, group_new_faces[int(group_id)], paste_back=True
#                                 )
                                
#                                 processed_frame = enhance_face(old_face, processed_frame, face_enhancer_model)
                
#                 processed_batch.append((frame_idx, processed_frame))
                
#             except Exception as e:
#                 logger.error(f"Failed to process frame {frame_idx}: {str(e)}")
#                 processed_batch.append((frame_idx, frame))
                
#     except Exception as e:
#         logger.error(f"Batch processing failed: {str(e)}")
#         for i, (frame_idx, frame) in enumerate(zip(batch_indices, batch_frames)):
#             processed_batch.append((frame_idx, frame))
    
#     return processed_batch

# def process_frames_batched(uid, frames, all_face_info, group_ids, group_new_faces, all_embeddings, all_bboxes, all_kps):
#     total_frames = len(frames)
#     max_workers = min(8, os.cpu_count())
#     batch_size = max(1, total_frames // (max_workers * 4))
    
#     print(f"[INFO] Using {max_workers} workers with batch size {batch_size}")
    
#     batches = []
#     for i in range(0, total_frames, batch_size):
#         batch_frames = frames[i:i + batch_size]
#         batch_indices = list(range(i, min(i + batch_size, total_frames)))
#         batch_data = (batch_frames, batch_indices, all_face_info, group_ids, group_new_faces, all_embeddings, all_bboxes, all_kps)
#         batches.append(batch_data)
    
#     print(f"[INFO] Created {len(batches)} batches for processing")
    
#     processed_frames = [None] * total_frames
    
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_batch = {
#             executor.submit(process_frame_batch, batch_data): i 
#             for i, batch_data in enumerate(batches)
#         }
        
#         completed_batches = 0
#         for future in as_completed(future_to_batch):
#             batch_results = future.result()
            
#             for frame_idx, processed_frame in batch_results:
#                 processed_frames[frame_idx] = processed_frame
            
#             completed_batches += 1
#             progress = (completed_batches / len(batches)) * 100
#             print(f"[PROGRESS] Completed {completed_batches}/{len(batches)} batches ({progress:.1f}%)")
#             # Write progress to file
#             write_progress(uid, round(progress, 1))
    
#     print(f"[SUCCESS] All {total_frames} frames processed successfully")
#     return processed_frames

# def write_video_output_fast(processed_frames, video_info, result_file_path, input_file_path):
#     print("[INFO] Writing processed frames to video...")
    
#     temp_output = os.path.join(os.path.dirname(result_file_path), 'temp_' + os.path.basename(result_file_path))
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(
#         temp_output,
#         fourcc,
#         video_info['fps'],
#         (video_info['width'], video_info['height'])
#     )
    
#     chunk_size = 100
#     for i in range(0, len(processed_frames), chunk_size):
#         chunk = processed_frames[i:i + chunk_size]
#         for frame in chunk:
#             out.write(frame)
        
#         if i % (chunk_size * 5) == 0:
#             progress = (i / len(processed_frames)) * 100
#             # print(f"[PROGRESS] Written {i}/{len(processed_frames)} frames ({progress:.1f}%)")
    
#     out.release()
#     print(f"[SUCCESS] Video written to {temp_output}")
    
#     try:
#         print("[INFO] Adding audio to final video...")
#         original_video = VideoFileClip(input_file_path)
#         face_swapped_video = VideoFileClip(temp_output)

#         final_video = face_swapped_video.set_audio(original_video.audio)
#         final_video.write_videofile(
#             result_file_path,
#             codec='libx264',
#             audio_codec='aac',
#             remove_temp=True,
#             verbose=False,
#             logger=None,
#             threads=4
#         )

#         original_video.close()
#         face_swapped_video.close()
#         print(f"[SUCCESS] Final video saved to {result_file_path}")

#     except Exception as e:
#         print(f"[WARNING] Error processing audio: {str(e)}")
#         if os.path.exists(temp_output):
#             os.rename(temp_output, result_file_path)
#             print(f"[INFO] Video saved without audio to {result_file_path}")
#     finally:
#         if os.path.exists(temp_output):
#             try:
#                 os.remove(temp_output)
#             except Exception as e:
#                 print(f"[WARNING] Error removing temporary file: {str(e)}")

# def run_face_swap(uid, all_face_info, group_ids, all_embeddings, all_bboxes, all_kps, input_file_path, result_file_path):
#     print(f"[INFO] Starting ultra-optimized face swap for generation {uid}")
#     start_time = time.time()
    
#     try:
#         print("[INFO] Loading target faces...")
#         group_new_faces = {}
#         for gi in group_ids:
#             new_face = get_processed_face(os.path.join(UPLOAD_FOLDER, uid, 'new_faces', f'{gi}.jpg'))
#             group_new_faces[gi] = new_face
#         print(f"[INFO] Loaded {len(group_new_faces)} target faces")

#         cap = cv2.VideoCapture(input_file_path)
#         if not cap.isOpened():
#             raise ValueError("Unable to read input video")
            
#         video_info = {
#             'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#             'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#             'fps': cap.get(cv2.CAP_PROP_FPS),
#             'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         }
#         cap.release()
        
#         total_frames = video_info['total_frames']
#         estimated_memory_mb = (total_frames * video_info['width'] * video_info['height'] * 3) / (1024 * 1024)
        
#         print(f"[INFO] Video info: {total_frames} frames, {estimated_memory_mb:.1f}MB estimated")
        
#         if estimated_memory_mb > 6000 or total_frames > 3000:
#             print(f"[WARNING] Large video detected, using optimized sequential processing")
#             return run_face_swap_streaming(uid, all_face_info, group_ids, all_embeddings, all_bboxes, all_kps, input_file_path, result_file_path)
        
#         print("[INFO] Reading frames for batch processing...")
#         frames = []
#         cap = cv2.VideoCapture(input_file_path)
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frames.append(frame)
#         cap.release()
        
#         print(f"[INFO] Loaded {len(frames)} frames into memory")
        
#         # Write initial progress
#         write_progress(uid, 0.0)
#         processed_frames = process_frames_batched(
#             uid, frames, all_face_info, group_ids, group_new_faces, 
#             all_embeddings, all_bboxes, all_kps
#         )
        
#         write_video_output_fast(processed_frames, video_info, result_file_path, input_file_path)
        
#         # Write 100% progress at end
#         write_progress(uid, 100.0)
#         end_time = time.time()
#         processing_time = end_time - start_time
#         fps_processed = total_frames / processing_time
        
#         print(f"[SUCCESS] Face swap completed in {processing_time:.2f} seconds")
#         print(f"[STATS] Processed {fps_processed:.2f} frames per second")
        
#     except Exception as e:
#         print(f"[ERROR] Optimized face swap failed: {str(e)}")
#         print("[INFO] Falling back to streaming processing...")
#         return run_face_swap_streaming(uid, all_face_info, group_ids, all_embeddings, all_bboxes, all_kps, input_file_path, result_file_path)

# def run_face_swap_streaming(uid, all_face_info, group_ids, all_embeddings, all_bboxes, all_kps, input_file_path, result_file_path):
#     print("[INFO] Using optimized streaming processing")
    
#     group_new_faces = {}
#     for gi in group_ids:
#         new_face = get_processed_face(os.path.join(UPLOAD_FOLDER, uid, 'new_faces', f'{gi}.jpg'))
#         group_new_faces[gi] = new_face

#     temp_output = os.path.join(UPLOAD_FOLDER, uid, 'temp_streaming.mp4')

#     cap = cv2.VideoCapture(input_file_path)
#     if not cap.isOpened():
#         raise ValueError("Unable to read input video")

#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))

#     batch_size = 32
#     frame_batch = []
#     frame_indices = []
#     frame_number = 0
    
#     print(f"[INFO] Processing {total_frames} frames in batches of {batch_size}")
#     # Write initial progress
#     write_progress(uid, 0.0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             if frame_batch:
#                 processed_batch = process_streaming_batch(
#                     frame_batch, frame_indices, all_face_info, group_ids, 
#                     group_new_faces, all_embeddings, all_bboxes, all_kps
#                 )
#                 for processed_frame in processed_batch:
#                     out.write(processed_frame)
#             break

#         frame_batch.append(frame)
#         frame_indices.append(frame_number)
        
#         if len(frame_batch) >= batch_size:
#             processed_batch = process_streaming_batch(
#                 frame_batch, frame_indices, all_face_info, group_ids, 
#                 group_new_faces, all_embeddings, all_bboxes, all_kps
#             )
            
#             for processed_frame in processed_batch:
#                 out.write(processed_frame)
            
#             frame_batch = []
#             frame_indices = []
            
#             if frame_number % 320 == 0:
#                 progress = (frame_number / total_frames) * 100
#                 print(f"[PROGRESS] Processed {frame_number}/{total_frames} frames ({progress:.1f}%)")
#                 write_progress(uid, round(progress, 1))
        
#         frame_number += 1

#     cap.release()
#     out.release()
#     # Write 100% progress at end
#     write_progress(uid, 100.0)
#     print(f"[SUCCESS] Streaming face swap completed for {uid}")

#     try:
#         original_video = VideoFileClip(input_file_path)
#         face_swapped_video = VideoFileClip(temp_output)
#         final_video = face_swapped_video.set_audio(original_video.audio)
#         final_video.write_videofile(
#             result_file_path,
#             codec='libx264',
#             audio_codec='aac',
#             remove_temp=True,
#             verbose=False,
#             logger=None,
#             threads=2
#         )
#         original_video.close()
#         face_swapped_video.close()
#     except Exception as e:
#         print(f"Error processing audio: {str(e)}")
#         if os.path.exists(temp_output):
#             os.rename(temp_output, result_file_path)
#     finally:
#         if os.path.exists(temp_output):
#             try:
#                 os.remove(temp_output)
#             except Exception as e:
#                 print(f"Error removing temporary file: {str(e)}")

# def process_streaming_batch(frame_batch, frame_indices, all_face_info, group_ids, group_new_faces, all_embeddings, all_bboxes, all_kps):
#     processed_batch = []
    
#     for frame, frame_idx in zip(frame_batch, frame_indices):
#         processed_frame = frame.copy()
        
#         if str(frame_idx) in all_face_info:
#             for group_id in all_face_info[str(frame_idx)]:
#                 if int(group_id) in group_ids:
#                     for gi in all_face_info[str(frame_idx)][str(group_id)]:
#                         old_face = Face(bbox=all_bboxes[gi], kps=all_kps[gi])
#                         old_face['embedding'] = all_embeddings[gi]
                        
#                         processed_frame = face_swapper_model.get(processed_frame, old_face, group_new_faces[int(group_id)], paste_back=True)
#                         processed_frame = enhance_face(old_face, processed_frame, face_enhancer_model)
        
#         processed_batch.append(processed_frame)
    
#     return processed_batch

# def get_processed_face(img_path: str) -> Face:
#     try:
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"Image file not found: {img_path}")
            
#         image = cv2.imread(img_path)
#         if image is None:
#             raise ValueError(f"Failed to read image: {img_path}")

#         bboxes, kpss = retinaface_det_model.detect(image, max_num=1, metric='default')
#         if bboxes is None or len(bboxes) == 0:
#             raise ValueError("No face detected in the image")

#         bbox = bboxes[0, 0:4]
#         det_score = bboxes[0, 4]
#         kps = kpss[0]
        
#         face = Face(bbox=bbox, kps=kps, det_score=det_score)
#         face['embedding'] = arcface_emedding_model.get(image, face)
        
#         return face

#     except Exception as e:
#         logger.error(f"Error in get_processed_face: {str(e)}")
#         raise

# def get_images_from_group(uid: str, num_images: int = 5) -> Dict:
#     try:
#         base_path = os.path.join(BASE_DIR, uid, "cropped_faces")
#         if not os.path.exists(base_path):
#             raise FileNotFoundError(f"Directory not found: {base_path}")

#         groups = os.listdir(base_path)
#         group_images = {}

#         for group in groups:
#             group_path = os.path.join(base_path, group)
#             if os.path.isdir(group_path):
#                 images = sorted(os.listdir(group_path))[:num_images]
#                 images = [f"{uid}/cropped_faces/{group}/{img}" for img in images]
#                 group_images[group] = images

#         return group_images

#     except Exception as e:
#         logger.error(f"Error in get_images_from_group: {str(e)}")
#         raise

# def write_progress(uid, progress):
#     """Write progress percentage to a progress.json file for the given uid."""
#     try:
#         progress_path = os.path.join(UPLOAD_FOLDER, uid, 'progress.json')
#         with open(progress_path, 'w') as f:
#             json.dump({"progress": progress}, f)
#     except Exception as e:
#         print(f"[WARNING] Could not write progress for {uid}: {e}")





import cv2
import onnxruntime
import numpy as np
import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from moviepy.editor import VideoFileClip


from face_swap.utils.common import Face
from face_swap.resnetface import ResnetFace 
from face_swap.arcface_onnx import ArcFaceONNX
from face_swap.inswapper import INSwapper
from face_swap.face_enhancer import enhance_face
from config import UPLOAD_FOLDER, BASE_DIR
from config import RESNETFACE_MODEL_PATH, ARCFACE_MODEL_PATH, FACE_SWAPPER_MODEL_PATH, FACE_ENHANCER_MODEL_PATH
from config import PROVIDERS

from concurrent.futures import ThreadPoolExecutor

# === NEW: Video Enhancement Configuration ===
SHARPEN_BACKGROUND = True
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Check if OpenCV was built with CUDA support
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
if USE_CUDA:
    print("CUDA acceleration detected for video enhancement.")
else:
    print("CUDA not available for video enhancement. Using CPU fallback.")

def sharpen_frame(frame, kernel):
    """Apply sharpening filter to entire frame using CUDA if available."""
    if USE_CUDA:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(frame)
        sharpened_gpu = cv2.cuda.filter2D(gpu_img, -1, kernel)
        sharpened = sharpened_gpu.download()
        return sharpened
    else:
        return cv2.filter2D(frame, -1, kernel)
# === END NEW: Video Enhancement Configuration ===

# Get the number of available CPU cores
cpu_cores = os.cpu_count()

# Set the number of workers, with a minimum of 16 and max equal to the number of available CPU cores
min_workers = 16
max_workers = max(min_workers, cpu_cores)

# Initialize the ThreadPoolExecutor with the dynamic max_workers
faceswap_executor = ThreadPoolExecutor(max_workers=max_workers)

print(f"Using {max_workers} workers for face swap execution.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    retinaface_det_model = ResnetFace(RESNETFACE_MODEL_PATH, providers=PROVIDERS)
    retinaface_det_model.prepare(ctx_id=1, input_size=(640, 640), det_thresh=0.7)
    arcface_emedding_model = ArcFaceONNX(ARCFACE_MODEL_PATH, providers=PROVIDERS)
    face_swapper_model = INSwapper(FACE_SWAPPER_MODEL_PATH, providers=PROVIDERS)
    face_enhancer_model = onnxruntime.InferenceSession(FACE_ENHANCER_MODEL_PATH, providers=PROVIDERS)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    raise

def process_face(frame: np.ndarray, bbox: np.ndarray, kps: np.ndarray, det_score: float, frame_number: int) -> Face:
    face = Face(bbox=bbox, kps=kps, det_score=det_score)
    face['frame_number'] = frame_number
    face['embedding'] = arcface_emedding_model.get(frame, face)
    return face

def crop_faces(video_path: str, uid: str):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        directory = os.path.join(UPLOAD_FOLDER, uid, "cropped_faces")
        os.makedirs(directory, exist_ok=True)

        face_distance = 1.5
        embeddings = []
        frame_number = -1
        all_bboxes = []
        all_kps = []
        all_face_info = {}
        filename_counter = 1
        all_faces = {0: []}
        unique_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            
            try:
                bboxes, kpss = retinaface_det_model.detect(frame, max_num=0, metric='default')
                
                if bboxes is None or len(bboxes) == 0:
                    continue

                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i, 0:4]
                    det_score = bboxes[i, 4]
                    x1, y1, x2, y2 = bbox.astype(int)
                    
                    if (x2 - x1) <= 80 or (y2 - y1) <= 80:
                        continue

                    kps = None if kpss is None else kpss[i]
                    face = process_face(frame, bbox, kps, det_score, frame_number)
                    
                    if unique_id == 0:
                        sim_id = 0
                        unique_id += 1
                    else:
                        max_sim = 0
                        max_index = unique_id
                        
                        for group_id, faces in all_faces.items():
                            if not faces:
                                continue
                            
                            similarities = np.array([
                                np.sum(np.square(face.normed_embedding - known_face.normed_embedding))
                                for known_face in faces
                            ])
                            group_sim = np.mean(similarities < face_distance)
                            
                            if group_sim > max_sim:
                                max_sim = group_sim
                                max_index = group_id
                        
                        sim_id = max_index if max_sim > 0.25 else unique_id
                        if sim_id == unique_id:
                            unique_id += 1

                    face['group_id'] = sim_id

                    if sim_id in all_faces:
                        all_faces[sim_id].append(face)
                    else:
                        all_faces[sim_id] = [face]
                    
                    try:
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue
                            
                        sim_id_directory = os.path.join(directory, str(sim_id))
                        os.makedirs(sim_id_directory, exist_ok=True)
                        
                        filename = f"images_{filename_counter}.jpg"
                        cv2.imwrite(os.path.join(sim_id_directory, filename), face_crop)
                        filename_counter += 1
                    except Exception as e:
                        logger.warning(f"Error saving face crop: {str(e)}")
                        continue

                    if frame_number in all_face_info:
                        if face['group_id'] in all_face_info[frame_number]:
                            all_face_info[frame_number][face['group_id']].append(len(embeddings))
                        else:
                            all_face_info[frame_number][face['group_id']] = [len(embeddings)]
                    else:
                        all_face_info[frame_number] = {face['group_id']: [len(embeddings)]}

                    embeddings.append(face['embedding'])
                    all_bboxes.append(face['bbox'])
                    all_kps.append(face['kps'])

            except Exception as e:
                logger.error(f"Error processing frame {frame_number}: {str(e)}")
                continue

        try:
            np.save(os.path.join(UPLOAD_FOLDER, uid, 'face_embeddings.npy'), embeddings)
            np.save(os.path.join(UPLOAD_FOLDER, uid, 'face_bboxes.npy'), all_bboxes)
            np.save(os.path.join(UPLOAD_FOLDER, uid, 'face_kps.npy'), all_kps)

            info = {'max_groups': unique_id, 'all_face_info': all_face_info}
            with open(os.path.join(UPLOAD_FOLDER, uid, 'all_info.json'), 'w') as file:
                json.dump(info, file, indent=4)
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in crop_faces: {str(e)}")
        raise
    finally:
        if cap is not None:
            cap.release()

def process_frame_batch(batch_data):
    batch_frames, batch_indices, frame_face_info, group_ids, group_new_faces, all_embeddings, all_bboxes, all_kps = batch_data
    
    processed_batch = []
    
    try:
        for i, (frame_idx, frame) in enumerate(zip(batch_indices, batch_frames)):
            try:
                processed_frame = frame.copy()
                
                if str(frame_idx) in frame_face_info:
                    for group_id in frame_face_info[str(frame_idx)]:
                        if int(group_id) in group_ids:
                            for gi in frame_face_info[str(frame_idx)][str(group_id)]:
                                old_face = Face(bbox=all_bboxes[gi], kps=all_kps[gi])
                                old_face['embedding'] = all_embeddings[gi]
                                
                                processed_frame = face_swapper_model.get(
                                    processed_frame, old_face, group_new_faces[int(group_id)], paste_back=True
                                )
                                
                                processed_frame = enhance_face(old_face, processed_frame, face_enhancer_model)

                # === NEW: Apply frame-level sharpening enhancement after all face swaps ===
                if SHARPEN_BACKGROUND:
                    processed_frame = sharpen_frame(processed_frame, SHARPEN_KERNEL)
                # === END NEW ===
                
                processed_batch.append((frame_idx, processed_frame))
                
            except Exception as e:
                logger.error(f"Failed to process frame {frame_idx}: {str(e)}")
                processed_batch.append((frame_idx, frame))
                
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        for i, (frame_idx, frame) in enumerate(zip(batch_indices, batch_frames)):
            processed_batch.append((frame_idx, frame))
    
    return processed_batch

def process_frames_batched(uid, frames, all_face_info, group_ids, group_new_faces, all_embeddings, all_bboxes, all_kps):
    total_frames = len(frames)
    max_workers = min(8, os.cpu_count())
    batch_size = max(1, total_frames // (max_workers * 4))
    
    print(f"[INFO] Using {max_workers} workers with batch size {batch_size}")
    
    batches = []
    for i in range(0, total_frames, batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, total_frames)))
        batch_data = (batch_frames, batch_indices, all_face_info, group_ids, group_new_faces, all_embeddings, all_bboxes, all_kps)
        batches.append(batch_data)
    
    print(f"[INFO] Created {len(batches)} batches for processing")
    
    processed_frames = [None] * total_frames
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_frame_batch, batch_data): i 
            for i, batch_data in enumerate(batches)
        }
        
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch_results = future.result()
            
            for frame_idx, processed_frame in batch_results:
                processed_frames[frame_idx] = processed_frame
            
            completed_batches += 1
            progress = (completed_batches / len(batches)) * 100
            print(f"[PROGRESS] Completed {completed_batches}/{len(batches)} batches ({progress:.1f}%)")
            # Write progress to file
            write_progress(uid, round(progress, 1))
    
    print(f"[SUCCESS] All {total_frames} frames processed successfully")
    return processed_frames

def write_video_output_fast(processed_frames, video_info, result_file_path, input_file_path):
    print("[INFO] Writing processed frames to video...")
    
    temp_output = os.path.join(os.path.dirname(result_file_path), 'temp_' + os.path.basename(result_file_path))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        temp_output,
        fourcc,
        video_info['fps'],
        (video_info['width'], video_info['height'])
    )
    
    chunk_size = 100
    for i in range(0, len(processed_frames), chunk_size):
        chunk = processed_frames[i:i + chunk_size]
        for frame in chunk:
            out.write(frame)
        
        if i % (chunk_size * 5) == 0:
            progress = (i / len(processed_frames)) * 100
            # print(f"[PROGRESS] Written {i}/{len(processed_frames)} frames ({progress:.1f}%)")
    
    out.release()
    print(f"[SUCCESS] Video written to {temp_output}")
    
    try:
        print("[INFO] Adding audio to final video...")
        original_video = VideoFileClip(input_file_path)
        face_swapped_video = VideoFileClip(temp_output)

        final_video = face_swapped_video.set_audio(original_video.audio)
        final_video.write_videofile(
            result_file_path,
            codec='libx264',
            audio_codec='aac',
            remove_temp=True,
            verbose=False,
            logger=None,
            threads=4
        )

        original_video.close()
        face_swapped_video.close()
        print(f"[SUCCESS] Final video saved to {result_file_path}")

    except Exception as e:
        print(f"[WARNING] Error processing audio: {str(e)}")
        if os.path.exists(temp_output):
            os.rename(temp_output, result_file_path)
            print(f"[INFO] Video saved without audio to {result_file_path}")
    finally:
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except Exception as e:
                print(f"[WARNING] Error removing temporary file: {str(e)}")

def run_face_swap(uid, all_face_info, group_ids, all_embeddings, all_bboxes, all_kps, input_file_path, result_file_path):
    print(f"[INFO] Starting ultra-optimized face swap for generation {uid}")
    start_time = time.time()
    
    try:
        print("[INFO] Loading target faces...")
        group_new_faces = {}
        for gi in group_ids:
            new_face = get_processed_face(os.path.join(UPLOAD_FOLDER, uid, 'new_faces', f'{gi}.jpg'))
            group_new_faces[gi] = new_face
        print(f"[INFO] Loaded {len(group_new_faces)} target faces")

        cap = cv2.VideoCapture(input_file_path)
        if not cap.isOpened():
            raise ValueError("Unable to read input video")
            
        video_info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        
        total_frames = video_info['total_frames']
        estimated_memory_mb = (total_frames * video_info['width'] * video_info['height'] * 3) / (1024 * 1024)
        
        print(f"[INFO] Video info: {total_frames} frames, {estimated_memory_mb:.1f}MB estimated")
        
        if estimated_memory_mb > 6000 or total_frames > 3000:
            print(f"[WARNING] Large video detected, using optimized sequential processing")
            return run_face_swap_streaming(uid, all_face_info, group_ids, all_embeddings, all_bboxes, all_kps, input_file_path, result_file_path)
        
        print("[INFO] Reading frames for batch processing...")
        frames = []
        cap = cv2.VideoCapture(input_file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"[INFO] Loaded {len(frames)} frames into memory")
        
        # Write initial progress
        write_progress(uid, 0.0)
        processed_frames = process_frames_batched(
            uid, frames, all_face_info, group_ids, group_new_faces, 
            all_embeddings, all_bboxes, all_kps
        )
        
        write_video_output_fast(processed_frames, video_info, result_file_path, input_file_path)
        
        # Write 100% progress at end
        write_progress(uid, 100.0)
        end_time = time.time()
        processing_time = end_time - start_time
        fps_processed = total_frames / processing_time
        
        print(f"[SUCCESS] Face swap completed in {processing_time:.2f} seconds")
        print(f"[STATS] Processed {fps_processed:.2f} frames per second")
        
    except Exception as e:
        print(f"[ERROR] Optimized face swap failed: {str(e)}")
        print("[INFO] Falling back to streaming processing...")
        return run_face_swap_streaming(uid, all_face_info, group_ids, all_embeddings, all_bboxes, all_kps, input_file_path, result_file_path)

def run_face_swap_streaming(uid, all_face_info, group_ids, all_embeddings, all_bboxes, all_kps, input_file_path, result_file_path):
    print("[INFO] Using optimized streaming processing")
    
    group_new_faces = {}
    for gi in group_ids:
        new_face = get_processed_face(os.path.join(UPLOAD_FOLDER, uid, 'new_faces', f'{gi}.jpg'))
        group_new_faces[gi] = new_face

    temp_output = os.path.join(UPLOAD_FOLDER, uid, 'temp_streaming.mp4')

    cap = cv2.VideoCapture(input_file_path)
    if not cap.isOpened():
        raise ValueError("Unable to read input video")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))

    batch_size = 32
    frame_batch = []
    frame_indices = []
    frame_number = 0
    
    print(f"[INFO] Processing {total_frames} frames in batches of {batch_size}")
    # Write initial progress
    write_progress(uid, 0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_batch:
                processed_batch = process_streaming_batch(
                    frame_batch, frame_indices, all_face_info, group_ids, 
                    group_new_faces, all_embeddings, all_bboxes, all_kps
                )
                for processed_frame in processed_batch:
                    out.write(processed_frame)
            break

        frame_batch.append(frame)
        frame_indices.append(frame_number)
        
        if len(frame_batch) >= batch_size:
            processed_batch = process_streaming_batch(
                frame_batch, frame_indices, all_face_info, group_ids, 
                group_new_faces, all_embeddings, all_bboxes, all_kps
            )
            
            for processed_frame in processed_batch:
                out.write(processed_frame)
            
            frame_batch = []
            frame_indices = []
            
            if frame_number % 320 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"[PROGRESS] Processed {frame_number}/{total_frames} frames ({progress:.1f}%)")
                write_progress(uid, round(progress, 1))
        
        frame_number += 1

    cap.release()
    out.release()
    # Write 100% progress at end
    write_progress(uid, 100.0)
    print(f"[SUCCESS] Streaming face swap completed for {uid}")

    try:
        original_video = VideoFileClip(input_file_path)
        face_swapped_video = VideoFileClip(temp_output)
        final_video = face_swapped_video.set_audio(original_video.audio)
        final_video.write_videofile(
            result_file_path,
            codec='libx264',
            audio_codec='aac',
            remove_temp=True,
            verbose=False,
            logger=None,
            threads=2
        )
        original_video.close()
        face_swapped_video.close()
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        if os.path.exists(temp_output):
            os.rename(temp_output, result_file_path)
    finally:
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")

def process_streaming_batch(frame_batch, frame_indices, all_face_info, group_ids, group_new_faces, all_embeddings, all_bboxes, all_kps):
    processed_batch = []
    
    for frame, frame_idx in zip(frame_batch, frame_indices):
        processed_frame = frame.copy()
        
        if str(frame_idx) in all_face_info:
            for group_id in all_face_info[str(frame_idx)]:
                if int(group_id) in group_ids:
                    for gi in all_face_info[str(frame_idx)][str(group_id)]:
                        old_face = Face(bbox=all_bboxes[gi], kps=all_kps[gi])
                        old_face['embedding'] = all_embeddings[gi]
                        
                        processed_frame = face_swapper_model.get(processed_frame, old_face, group_new_faces[int(group_id)], paste_back=True)
                        processed_frame = enhance_face(old_face, processed_frame, face_enhancer_model)

        # === NEW: Apply frame-level sharpening enhancement after all face swaps ===
        if SHARPEN_BACKGROUND:
            processed_frame = sharpen_frame(processed_frame, SHARPEN_KERNEL)
        # === END NEW ===
        
        processed_batch.append(processed_frame)
    
    return processed_batch

def get_processed_face(img_path: str) -> Face:
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        bboxes, kpss = retinaface_det_model.detect(image, max_num=1, metric='default')
        if bboxes is None or len(bboxes) == 0:
            raise ValueError("No face detected in the image")

        bbox = bboxes[0, 0:4]
        det_score = bboxes[0, 4]
        kps = kpss[0]
        
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        face['embedding'] = arcface_emedding_model.get(image, face)
        
        return face

    except Exception as e:
        logger.error(f"Error in get_processed_face: {str(e)}")
        raise

def get_images_from_group(uid: str, num_images: int = 5) -> Dict:
    try:
        base_path = os.path.join(BASE_DIR, uid, "cropped_faces")
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Directory not found: {base_path}")

        groups = os.listdir(base_path)
        group_images = {}

        for group in groups:
            group_path = os.path.join(base_path, group)
            if os.path.isdir(group_path):
                images = sorted(os.listdir(group_path))[:num_images]
                images = [f"{uid}/cropped_faces/{group}/{img}" for img in images]
                group_images[group] = images

        return group_images

    except Exception as e:
        logger.error(f"Error in get_images_from_group: {str(e)}")
        raise

def write_progress(uid, progress):
    """Write progress percentage to a progress.json file for the given uid."""
    try:
        progress_path = os.path.join(UPLOAD_FOLDER, uid, 'progress.json')
        with open(progress_path, 'w') as f:
            json.dump({"progress": progress}, f)
    except Exception as e:
        print(f"[WARNING] Could not write progress for {uid}: {e}")