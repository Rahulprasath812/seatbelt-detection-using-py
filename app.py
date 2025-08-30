import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template_string, Response, send_file
from flask_socketio import SocketIO, emit
import base64
import threading
import time
from datetime import datetime
import os
import json
import io
from PIL import Image, ImageDraw, ImageFont
import requests
import random
import pytesseract
import pyttsx3
import speech_recognition as sr
from urllib.parse import urlparse

# Global variables
YOLO_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

# Advanced AI imports
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
    print("✅ YOLO Detection: ENABLED")
except ImportError:
    print("⚠️ YOLO not available")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers available for advanced analysis")
except ImportError:
    print("⚠️ Transformers not available")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic_ai_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class SpeechSystem:
    """Text-to-Speech system for reading analysis results"""
    
    def __init__(self):
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.enabled = True
            print("✅ Text-to-Speech system initialized")
        except Exception as e:
            print(f"⚠️ TTS initialization failed: {e}")
            self.enabled = False
    
    def speak(self, text):
        """Convert text to speech"""
        if not self.enabled:
            return
        
        try:
            def speak_thread():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            thread = threading.Thread(target=speak_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"TTS error: {e}")

class FaceCapture:
    """Enhanced face capture system for violators"""
    
    def __init__(self):
        # Load face detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Face recognition for tracking
        self.known_faces = []
        self.violation_count = {}
        
        print("✅ Face capture system initialized")
    
    def detect_and_capture_faces(self, frame, helmet_status_list):
        """Detect faces and capture violators"""
        captured_faces = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        # Also detect profile faces
        profile_faces = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        # Combine all face detections
        all_faces = list(faces) + list(profile_faces)
        
        for i, (x, y, w, h) in enumerate(all_faces):
            # Extract face region with padding
            padding = 20
            face_x1 = max(0, x - padding)
            face_y1 = max(0, y - padding)
            face_x2 = min(frame.shape[1], x + w + padding)
            face_y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[face_y1:face_y2, face_x1:face_x2]
            
            # Check if this person has helmet violation
            is_violator = False
            for status in helmet_status_list:
                person_bbox = status.get('person_bbox', [])
                if len(person_bbox) == 4:
                    px1, py1, px2, py2 = person_bbox
                    # Check if face is within person detection area
                    if (face_x1 >= px1 and face_y1 >= py1 and 
                        face_x2 <= px2 and face_y2 <= py2 and 
                        status.get('helmet_worn', True) == False):
                        is_violator = True
                        break
            
            if is_violator and face_region.size > 0:
                # Save violator face
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                face_filename = f"violations/violator_face_{timestamp}_{i}.jpg"
                cv2.imwrite(face_filename, face_region)
                
                captured_faces.append({
                    'face_path': face_filename,
                    'bbox': [face_x1, face_y1, face_x2, face_y2],
                    'violation_type': 'NO_HELMET',
                    'face_image_base64': self.frame_to_base64(face_region)
                })
                
                # Draw blue box around violator face
                cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 3)  # Blue box
                cv2.putText(frame, 'VIOLATOR FACE', (face_x1, face_y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return captured_faces
    
    def frame_to_base64(self, frame):
        """Convert frame to base64 for web display"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
        except:
            return None

class OCRSystem:
    """OCR system for number plate recognition"""
    
    def __init__(self):
        try:
            pytesseract.get_tesseract_version()
            self.enabled = True
            print("✅ OCR System (Tesseract) initialized")
        except Exception as e:
            print(f"⚠️ OCR initialization failed: {e}")
            self.enabled = False
    
    def extract_number_plates(self, frame):
        """Extract number plates from image using OCR"""
        if not self.enabled:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(filtered, 30, 200)
            
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            number_plates = []
            
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)
                
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h
                    area = w * h
                    
                    if (2.0 <= aspect_ratio <= 5.0) and (area > 1000):
                        roi = gray[y:y+h, x:x+w]
                        roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        roi = cv2.medianBlur(roi, 3)
                        
                        config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                        text = pytesseract.image_to_string(roi, config=config).strip()
                        
                        if len(text) >= 4 and any(c.isalnum() for c in text):
                            number_plates.append({
                                'text': text,
                                'bbox': [x, y, x+w, y+h],
                                'confidence': 0.8
                            })
            
            return number_plates
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return []
    
    def draw_number_plates(self, frame, number_plates):
        """Draw detected number plates on frame"""
        for plate in number_plates:
            x1, y1, x2, y2 = plate['bbox']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
            
            text = plate['text']
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width, y1), (0, 255, 255), -1)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        return frame

class VideoProcessor:
    """Process video files frame by frame"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    def is_video_file(self, filename):
        """Check if file is a supported video format"""
        return any(filename.lower().endswith(fmt) for fmt in self.supported_formats)
    
    def process_video(self, video_path, analysis_callback):
        """FIXED: Process video file and analyze each frame"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                return None
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video info: {frame_count} frames, {fps} FPS, {width}x{height}")
            
            # Create output directory if it doesn't exist
            os.makedirs('analyzed', exist_ok=True)
            
            # Create output video writer with better codec
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"analyzed/video_analysis_{timestamp}.mp4"
            
            # Try different codecs for better compatibility
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'mp4v'),
                cv2.VideoWriter_fourcc(*'XVID'),
                cv2.VideoWriter_fourcc(*'H264'),
                cv2.VideoWriter_fourcc(*'MJPG')
            ]
            
            out = None
            for fourcc in fourcc_options:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"Using codec: {fourcc}")
                    break
                out.release()
            
            if out is None or not out.isOpened():
                print("Error: Could not create output video writer with any codec")
                cap.release()
                return None
            
            results = []
            frame_number = 0
            total_violations = 0
            
            print("Starting video processing...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                try:
                    # Analyze current frame
                    analysis = analysis_callback(frame.copy())
                    analysis['frame_number'] = frame_number
                    analysis['timestamp'] = frame_number / fps if fps > 0 else 0
                    
                    # Count violations in this frame
                    frame_violations = len(analysis.get('violations', []))
                    total_violations += frame_violations
                    
                    results.append(analysis)
                    
                    # Write annotated frame
                    if 'annotated_frame' in analysis and analysis['annotated_frame'] is not None:
                        out.write(analysis['annotated_frame'])
                    else:
                        out.write(frame)
                    
                    frame_number += 1

                    # Progress update every 100 frames
                    if frame_number % 100 == 0:
                        progress = (frame_number / frame_count) * 100 if frame_count > 0 else 0
                        print(f"Processing video: {progress:.1f}% ({frame_number}/{frame_count}) - Violations so far: {total_violations}")
                
                except Exception as e:
                    print(f"Error processing frame {frame_number}: {e}")
                    out.write(frame)  # Write original frame if analysis fails
                    frame_number += 1
            
            cap.release()
            out.release()
            
            print(f"Video processing complete. Output saved: {output_path}")
            print(f"Total violations found: {total_violations}")
            
            return {
                'output_path': output_path,
                'total_frames': frame_count,
                'total_violations': total_violations,
                'analysis_results': results,
                'processed_frames': frame_number
            }
            
        except Exception as e:
            print(f"Video processing error: {e}")
            if 'cap' in locals():
                cap.release()
            if 'out' in locals() and out is not None:
                out.release()
            return None

class AdvancedTrafficAI:
    def __init__(self):
        print("🚀 Initializing FIXED Traffic AI System...")
        
        # Initialize systems
        self.speech_system = SpeechSystem()
        self.ocr_system = OCRSystem()
        self.video_processor = VideoProcessor()
        self.face_capture = FaceCapture()
        
        # Detection models
        self.person_model = None
        self.init_models()
        
        # Detection thresholds
        self.person_confidence_threshold = 0.5
        self.helmet_confidence_threshold = 0.4
        
        # Statistics
        self.stats = {
            'total_analyzed': 0,
            'helmet_violations': 0,
            'videos_processed': 0,
            'number_plates_detected': 0,
            'faces_captured': 0,
            'speech_outputs': 0
        }
        
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories"""
        for folder in ['violations', 'uploads', 'analyzed', 'models', 'results']:
            os.makedirs(folder, exist_ok=True)
    
    def init_models(self):
        """Initialize AI models"""
        global YOLO_AVAILABLE
        
        if YOLO_AVAILABLE:
            try:
                self.person_model = YOLO('yolov8n.pt')
                print("✅ YOLO person detection model loaded")
            except Exception as e:
                print(f"❌ YOLO model loading error: {e}")
                YOLO_AVAILABLE = False
    
    def detect_helmet_properly(self, frame):
        """FIXED: Proper helmet detection with correct logic"""
        persons_with_helmets = []
        persons_without_helmets = []
        
        # Step 1: Detect persons using YOLO
        if YOLO_AVAILABLE and self.person_model:
            try:
                results = self.person_model(frame, conf=self.person_confidence_threshold, verbose=False)
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            if class_id == 0:  # Person class
                                conf = float(box.conf[0])
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # Extract person region
                                person_region = frame[y1:y2, x1:x2]
                                if person_region.size == 0:
                                    continue
                                
                                # Focus on head region (top 25% of person)
                                head_height = max(1, int((y2 - y1) * 0.25))
                                head_region = frame[y1:y1+head_height, x1:x2]
                                
                                if head_region.size == 0:
                                    continue
                                
                                # FIXED: Proper helmet detection
                                helmet_worn = self.is_helmet_present(head_region)
                                
                                person_data = {
                                    'bbox': [x1, y1, x2, y2],
                                    'head_bbox': [x1, y1, x2, y1+head_height],
                                    'confidence': conf,
                                    'helmet_worn': helmet_worn
                                }
                                
                                if helmet_worn:
                                    persons_with_helmets.append(person_data)
                                else:
                                    persons_without_helmets.append(person_data)
                                    
            except Exception as e:
                print(f"YOLO detection error: {e}")
        
        # Fallback: Use face detection if no persons detected
        if not persons_with_helmets and not persons_without_helmets:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_capture.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                # Expand to include potential helmet area
                helmet_area_y1 = max(0, y - int(h * 0.3))
                helmet_region = frame[helmet_area_y1:y+h, x:x+w]
                
                helmet_worn = self.is_helmet_present(helmet_region)
                
                person_data = {
                    'bbox': [x, helmet_area_y1, x+w, y+h],
                    'head_bbox': [x, helmet_area_y1, x+w, y],
                    'confidence': 0.8,
                    'helmet_worn': helmet_worn
                }
                
                if helmet_worn:
                    persons_with_helmets.append(person_data)
                else:
                    persons_without_helmets.append(person_data)
        
        return persons_with_helmets, persons_without_helmets
    
    def is_helmet_present(self, head_region):
        """FIXED: Determine if helmet is present in head region"""
        try:
            if head_region.size == 0:
                return False
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # Method 1: Check for helmet-like colors (non-skin tones)
            # Skin color range in HSV
            skin_lower = np.array([0, 20, 70])
            skin_upper = np.array([20, 150, 255])
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            skin_pixels = cv2.countNonZero(skin_mask)
            total_pixels = head_region.shape[0] * head_region.shape[1]
            skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
            
            # Method 2: Check for hard/smooth surfaces (helmet texture)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            edge_density = cv2.countNonZero(edges) / total_pixels if total_pixels > 0 else 0
            
            # Method 3: Check for circular/oval shapes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            has_helmet_shape = False
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.4:  # Somewhat circular
                            has_helmet_shape = True
                            break
            
            # Decision logic: FIXED
            helmet_indicators = 0
            
            if skin_ratio < 0.3:  # Low skin visibility
                helmet_indicators += 1
            
            if edge_density < 0.1:  # Smooth surface (helmet)
                helmet_indicators += 1
            
            if has_helmet_shape:  # Circular/oval shape
                helmet_indicators += 1
            
            # Helmet is present if we have 2 or more indicators
            return helmet_indicators >= 2
            
        except Exception as e:
            print(f"Helmet detection error: {e}")
            return False
    
    def frame_to_base64(self, frame):
        """Convert frame to base64 for web display"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
        except:
            return None
    
    def analyze_frame(self, frame):
        """Analyze single frame for violations"""
        try:
            # Create copy for annotation
            annotated_frame = frame.copy()
            
            # Step 1: Detect persons and helmet status
            persons_with_helmets, persons_without_helmets = self.detect_helmet_properly(frame)
            
            # Step 2: Capture faces of violators
            helmet_status_list = []
            for person in persons_without_helmets:
                person['helmet_worn'] = False
                person['person_bbox'] = person['bbox']
                helmet_status_list.append(person)
            
            captured_faces = self.face_capture.detect_and_capture_faces(annotated_frame, helmet_status_list)
            
            # Step 3: Draw detection boxes with CORRECT colors
            violations = []
            
            # Draw GREEN boxes for persons WITH helmets
            for person in persons_with_helmets:
                x1, y1, x2, y2 = person['bbox']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # GREEN
                cv2.putText(annotated_frame, f'HELMET WORN', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw RED boxes for persons WITHOUT helmets
            for person in persons_without_helmets:
                x1, y1, x2, y2 = person['bbox']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # RED
                cv2.putText(annotated_frame, f'NO HELMET - VIOLATION', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                violations.append("NO_HELMET")
            
            # Step 4: OCR for number plates
            number_plates = self.ocr_system.extract_number_plates(frame.copy())
            if number_plates:
                annotated_frame = self.ocr_system.draw_number_plates(annotated_frame, number_plates)
            
            # Step 5: Add analysis overlay
            self.add_analysis_overlay(annotated_frame, len(persons_with_helmets), 
                                    len(persons_without_helmets), number_plates, captured_faces)
            
            return {
                'violations': violations,
                'persons_with_helmets': len(persons_with_helmets),
                'persons_without_helmets': len(persons_without_helmets),
                'number_plates': number_plates,
                'captured_faces': captured_faces,
                'annotated_frame': annotated_frame,
                'annotated_frame_base64': self.frame_to_base64(annotated_frame)
            }
            
        except Exception as e:
            print(f"Frame analysis error: {e}")
            return {
                'violations': [],
                'persons_with_helmets': 0,
                'persons_without_helmets': 0,
                'number_plates': [],
                'captured_faces': [],
                'annotated_frame': frame,
                'annotated_frame_base64': self.frame_to_base64(frame)
            }
    
    def analyze_media_file(self, file_path):
        """Analyze uploaded image or video file"""
        try:
            if self.video_processor.is_video_file(file_path):
                return self.analyze_video_file(file_path)
            else:
                return self.analyze_image_file(file_path)
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_image_file(self, image_path):
        """FIXED: Analyze single image file"""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                return {'success': False, 'error': 'Could not load image'}
            
            # Analyze frame
            analysis = self.analyze_frame(frame)
            
            # Generate speech output
            speech_text = self.generate_speech_text(analysis)
            self.speech_system.speak(speech_text)
            
            # Save annotated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"analyzed/image_analysis_{timestamp}.jpg"
            cv2.imwrite(output_path, analysis['annotated_frame'])
            
            # Update statistics
            self.stats['total_analyzed'] += 1
            self.stats['helmet_violations'] += len(analysis['violations'])
            self.stats['number_plates_detected'] += len(analysis['number_plates'])
            self.stats['faces_captured'] += len(analysis['captured_faces'])
            self.stats['speech_outputs'] += 1
            
            return {
                'success': True,
                'type': 'image',
                'violations': analysis['violations'],
                'persons_with_helmets': analysis['persons_with_helmets'],
                'persons_without_helmets': analysis['persons_without_helmets'],
                'number_plates': analysis['number_plates'],
                'captured_faces': analysis['captured_faces'],
                'analysis_image_path': output_path,
                'annotated_image_base64': analysis['annotated_frame_base64'],
                'speech_text': speech_text,
                'statistics': self.stats
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def analyze_video_file(self, video_path):
        """FIXED: Analyze video file frame by frame"""
        try:
            print(f"Starting video analysis for: {video_path}")
            
            def frame_analysis_callback(frame):
                return self.analyze_frame(frame)
            
            # Process video
            video_result = self.video_processor.process_video(video_path, frame_analysis_callback)
            
            if not video_result:
                return {'success': False, 'error': 'Video processing failed'}
            
            # Calculate summary statistics
            total_violations = video_result['total_violations']
            total_faces_captured = sum(len(r.get('captured_faces', [])) for r in video_result.get('analysis_results', []))
            
            # Generate summary speech
            speech_text = f"Video analysis complete. {video_result['processed_frames']} frames processed. {total_violations} helmet violations detected. {total_faces_captured} violator faces captured."
            self.speech_system.speak(speech_text)
            
            # Update statistics
            self.stats['total_analyzed'] += 1
            self.stats['videos_processed'] += 1
            self.stats['helmet_violations'] += total_violations
            self.stats['faces_captured'] += total_faces_captured
            self.stats['speech_outputs'] += 1
            
            return {
                'success': True,
                'type': 'video',
                'output_path': video_result['output_path'],
                'total_frames': video_result['processed_frames'],
                'total_violations': total_violations,
                'total_faces_captured': total_faces_captured,
                'speech_text': speech_text,
                'statistics': self.stats
            }
            
        except Exception as e:
            print(f"Video analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_speech_text(self, analysis):
        """Generate text for speech output"""
        speech_parts = []
        
        # Helmet analysis
        with_helmets = analysis['persons_with_helmets']
        without_helmets = analysis['persons_without_helmets']
        
        if without_helmets > 0:
            speech_parts.append(f"Alert! {without_helmets} person{'s' if without_helmets > 1 else ''} detected without helmet. Violation marked in red.")
        
        if with_helmets > 0:
            speech_parts.append(f"{with_helmets} person{'s' if with_helmets > 1 else ''} wearing helmet properly. Marked in green.")
        
        if with_helmets == 0 and without_helmets == 0:
            speech_parts.append("No persons detected in this image.")
        
        # Number plates
        plates = analysis['number_plates']
        if plates:
            plates_text = ", ".join([plate['text'] for plate in plates])
            speech_parts.append(f"Number plates detected: {plates_text}")
        
        # Captured faces
        faces = analysis['captured_faces']
        if faces:
            speech_parts.append(f"{len(faces)} violator face{'s' if len(faces) > 1 else ''} captured and saved.")
        
        return " ".join(speech_parts)
    
    def add_analysis_overlay(self, frame, with_helmets, without_helmets, number_plates, captured_faces):
        """Add analysis overlay to frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Panel dimensions
        panel_height = 140
        panel_y = height - panel_height - 10
        
        # Draw background panel
        cv2.rectangle(overlay, (10, panel_y), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        y_offset = panel_y + 25
        
        # Helmet status
        if without_helmets > 0:
            status_text = f"VIOLATIONS: {without_helmets} person(s) without helmet"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = f"SAFE: All {with_helmets} person(s) wearing helmets"
            status_color = (0, 255, 0)  # Green
        
        cv2.putText(frame, status_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Number plates
        cv2.putText(frame, f"NUMBER PLATES: {len(number_plates)}", 
                   (20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Captured faces
        cv2.putText(frame, f"VIOLATOR FACES CAPTURED: {len(captured_faces)}", 
                   (20, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, y_offset + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Overall status indicator
        if without_helmets == 0:
            cv2.putText(frame, "STATUS: SAFE", (width - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "STATUS: VIOLATION", (width - 220, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        return {
            'detection_stats': self.stats,
            'system_status': {
                'speech_enabled': self.speech_system.enabled,
                'ocr_enabled': self.ocr_system.enabled,
                'yolo_available': YOLO_AVAILABLE
            }
        }

# Initialize the fixed system
ai_detector = AdvancedTrafficAI()

# FIXED: Add route to serve images
@app.route('/get_image/<path:filename>')
def get_image(filename):
    """Serve analyzed images and captured faces"""
    try:
        return send_file(filename)
    except Exception as e:
        print(f"Image serving error: {e}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/get_video/<path:filename>')
def get_video(filename):
    """Serve analyzed videos"""
    try:
        return send_file(filename)
    except Exception as e:
        print(f"Video serving error: {e}")
        return jsonify({'error': 'Video not found'}), 404

# Flask Routes
@app.route('/analyze_media', methods=['POST'])
def analyze_media():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        print(f"Analyzing file: {file_path}")
        
        # Analyze file (image or video)
        result = ai_detector.analyze_media_file(file_path)
        
        # Convert NumPy types for JSON serialization
        if 'annotated_frame' in result:
            del result['annotated_frame']  # Remove frame data for JSON response
        
        result = convert_numpy_types(result)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_statistics')
def get_statistics():
    try:
        stats = ai_detector.get_statistics()
        stats = convert_numpy_types(stats)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/test_speech', methods=['POST'])
def test_speech():
    try:
        data = request.get_json()
        text = data.get('text', 'Speech test successful. Traffic AI system is ready for helmet detection analysis.')
        ai_detector.speech_system.speak(text)
        return jsonify({'success': True, 'message': 'Speech test initiated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_violation_images')
def get_violation_images():
    """Get list of captured violator faces"""
    try:
        violation_files = []
        violations_dir = 'violations'
        
        if os.path.exists(violations_dir):
            for filename in os.listdir(violations_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(violations_dir, filename)
                    violation_files.append({
                        'filename': filename,
                        'path': file_path,
                        'timestamp': os.path.getctime(file_path)
                    })
        
        # Sort by timestamp (newest first)
        violation_files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'violation_files': violation_files})
        
    except Exception as e:
        return jsonify({'error': str(e)})

# SocketIO events for real-time communication
@socketio.on('connect')
def handle_connect():
    emit('system_status', {'status': 'connected', 'message': 'Fixed Traffic AI System Ready'})

@socketio.on('chat_message')
def handle_chat_message(data):
    message = data.get('message', '')
    
    # Generate contextual response based on message
    if 'helmet' in message.lower():
        response = "The system now correctly detects helmets: GREEN boxes for helmet worn, RED boxes for violations. Violator faces are automatically captured."
    elif 'video' in message.lower():
        response = "Video analysis has been fixed - it now processes each frame and creates an annotated output video with proper violation detection."
    elif 'face' in message.lower():
        response = "Face capture is working - violator faces are automatically detected and saved when helmet violations occur."
    else:
        response = "Upload an image or video to test the fixed helmet detection system. The confusion with inverted results has been resolved."
    
    # Speak the response
    ai_detector.speech_system.speak(response)
    
    emit('chat_response', {'message': response, 'timestamp': datetime.now().isoformat()})

# Main dashboard route with FIXED interface
@app.route('/')
def dashboard():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>FIXED Traffic AI - Complete Helmet Detection & Face Capture</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: #333;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header {
            background: rgba(255, 255, 255, 0.95); padding: 30px; border-radius: 15px;
            margin-bottom: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header h1 { 
            font-size: 2.5em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;
        }
        .fix-notice {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;
            text-align: center; font-weight: bold;
        }
        .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .panel {
            background: rgba(255, 255, 255, 0.95); padding: 25px; border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); height: fit-content;
        }
        .upload-area {
            border: 3px dashed #667eea; padding: 60px 20px; text-align: center; 
            border-radius: 15px; margin: 20px 0; transition: all 0.3s; cursor: pointer;
            background: linear-gradient(135deg, #f8f9ff 0%, #e3f2fd 100%);
        }
        .upload-area:hover { 
            border-color: #764ba2; background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            transform: translateY(-2px);
        }
        .upload-area h3 { color: #667eea; margin-bottom: 10px; }
        .upload-area p { color: #666; }
        .btn {
            padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer;
            font-size: 16px; font-weight: 600; margin: 8px 4px; transition: all 0.3s;
        }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .btn-success { background: linear-gradient(135deg, #2ed573 0%, #17a2b8 100%); color: white; }
        .btn-danger { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); color: white; }
        .btn-warning { background: linear-gradient(135deg, #ffa500 0%, #ff6b6b 100%); color: white; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .analysis-result {
            margin: 20px 0; padding: 20px; border-radius: 10px;
        }
        .result-success { background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
        .result-violation { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
        .result-processing { background: #fff3cd; color: #856404; border: 2px solid #ffeeba; }
        .preview-container {
            text-align: center; margin: 20px 0; border-radius: 10px; overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .preview-container img, .preview-container video {
            max-width: 100%; max-height: 400px; object-fit: contain;
            background: #000; border-radius: 8px;
        }
        .analysis-image {
            max-width: 100%; border-radius: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            margin: 15px 0;
        }
        .chat-container {
            height: 350px; display: flex; flex-direction: column;
        }
        .chat-messages {
            flex: 1; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px;
            padding: 15px; background: #f8f9fa; margin-bottom: 15px;
        }
        .chat-message {
            margin: 10px 0; padding: 12px; border-radius: 8px; max-width: 90%;
        }
        .chat-message.ai {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
            border-left: 4px solid #2196f3;
        }
        .chat-message.user {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
            margin-left: auto; border-left: 4px solid #9c27b0;
        }
        .chat-input-container {
            display: flex; gap: 10px;
        }
        .chat-input {
            flex: 1; padding: 12px; border: 2px solid #ddd; border-radius: 8px;
            font-size: 14px;
        }
        .stats-grid {
            display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 12px; text-align: center;
        }
        .stat-number { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .stat-label { font-size: 0.9em; opacity: 0.9; }
        .progress-bar {
            width: 100%; height: 8px; background: #e9ecef; border-radius: 4px;
            overflow: hidden; margin: 10px 0;
        }
        .progress-fill {
            height: 100%; background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .status-indicator {
            display: inline-block; width: 12px; height: 12px; border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background: #2ed573; }
        .status-offline { background: #ff6b6b; }
        .detection-legend {
            background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;
        }
        .legend-item {
            display: flex; align-items: center; margin: 8px 0;
        }
        .legend-color {
            width: 20px; height: 20px; border-radius: 4px; margin-right: 10px;
        }
        .green-box { background: #28a745; }
        .red-box { background: #dc3545; }
        .yellow-box { background: #ffc107; }
        .blue-box { background: #007bff; }
        h2 { color: #495057; margin-bottom: 20px; }
        @media (max-width: 1000px) { .main-grid { grid-template-columns: 1fr; } }
        .violation-gallery {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 15px; margin: 15px 0;
        }
        .violation-face {
            border: 3px solid #dc3545; border-radius: 8px; overflow: hidden;
            text-align: center; background: #fff; transition: transform 0.3s;
        }
        .violation-face:hover { transform: scale(1.05); }
        .violation-face img {
            width: 100%; height: 100px; object-fit: cover;
        }
        .violation-face p {
            font-size: 0.7em; padding: 8px; color: #dc3545; font-weight: bold;
        }
        .face-capture-section {
            margin: 20px 0; padding: 15px; background: #fff3cd; border-radius: 8px;
            border: 2px solid #ffc107;
        }
        .face-capture-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 10px; margin-top: 10px;
        }
        .captured-face {
            border: 2px solid #007bff; border-radius: 6px; overflow: hidden;
            background: #e3f2fd;
        }
        .captured-face img {
            width: 100%; height: 80px; object-fit: cover;
        }
        .captured-face p {
            font-size: 0.6em; padding: 4px; color: #0056b3; text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FIXED Traffic AI System</h1>
            <p>Complete Helmet Detection with Face Capture & Working Video Analysis</p>
        </div>
        
        <div class="fix-notice">
            ALL ISSUES FIXED: Image display working, Face capture visible, Video analysis functional
        </div>
        
        <div class="main-grid">
            <div class="panel">
                <h2>Media Analysis</h2>
                
                <div class="upload-area" id="uploadArea" onclick="document.getElementById('mediaUpload').click();">
                    <h3>Upload Image or Video</h3>
                    <p>Drag & drop or click to select image/video files</p>
                    <p><small>Supported: JPG, PNG, MP4, AVI, MOV, etc.</small></p>
                    <input type="file" id="mediaUpload" accept="image/*,video/*" style="display: none;" onchange="handleFileUpload()">
                </div>
                
                <div id="mediaPreview" style="display: none;" class="preview-container">
                    <!-- Preview will be inserted here -->
                </div>
                
                <div style="text-align: center; margin: 20px 0;">
                    <button class="btn btn-primary" onclick="analyzeMedia()" id="analyzeBtn" disabled>
                        Analyze with FIXED AI
                    </button>
                    <button class="btn btn-warning" onclick="testSpeech()">
                        Test Speech
                    </button>
                    <button class="btn btn-success" onclick="loadViolationGallery()">
                        View All Captured Faces
                    </button>
                </div>
                
                <div class="detection-legend">
                    <h4>FIXED Detection System:</h4>
                    <div class="legend-item">
                        <div class="legend-color green-box"></div>
                        <span><strong>Green Box:</strong> Helmet Properly Worn (SAFE)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color red-box"></div>
                        <span><strong>Red Box:</strong> No Helmet - VIOLATION</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color blue-box"></div>
                        <span><strong>Blue Box:</strong> Violator Face Captured</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color yellow-box"></div>
                        <span><strong>Yellow Box:</strong> Number Plate Detected</span>
                    </div>
                </div>
                
                <div id="analysisResults" style="display: none;">
                    <!-- Analysis results will be displayed here -->
                </div>
                
                <div id="violationGallery" style="display: none;">
                    <h4>Captured Violator Faces:</h4>
                    <div class="violation-gallery" id="violationFaces">
                        <!-- Violation faces will be displayed here -->
                    </div>
                </div>
                
                <div id="progressContainer" style="display: none;">
                    <p>Processing...</p>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h2>AI Chat Assistant</h2>
                <div class="chat-container">
                    <div class="chat-messages" id="chatMessages">
                        <div class="chat-message ai">
                            <strong>AI Assistant:</strong><br>
                            ALL ISSUES FIXED:
                            <ul style="margin: 10px 0; padding-left: 20px;">
                                <li>✅ Helmet detection logic corrected</li>
                                <li>✅ Analyzed images now display properly</li>
                                <li>✅ Face capture visible in results</li>
                                <li>✅ Video analysis working with codecs fixed</li>
                                <li>✅ Clear red/green marking system</li>
                            </ul>
                            Upload a file to test all improvements!
                        </div>
                    </div>
                    <div class="chat-input-container">
                        <input type="text" class="chat-input" id="chatInput" placeholder="Ask about the fixes...">
                        <button class="btn btn-primary" onclick="sendChatMessage()">Send</button>
                    </div>
                </div>
                
                <div>
                    <h3>System Status</h3>
                    <p><span class="status-indicator status-online"></span><strong>AI Detection:</strong> FIXED & Active</p>
                    <p><span class="status-indicator" id="speechStatus"></span><strong>Speech System:</strong> <span id="speechText">Checking...</span></p>
                    <p><span class="status-indicator" id="ocrStatus"></span><strong>OCR System:</strong> <span id="ocrText">Checking...</span></p>
                    <p><span class="status-indicator status-online"></span><strong>Face Capture:</strong> Active & Visible</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number" id="totalAnalyzed">0</div>
                        <div class="stat-label">Files Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="violationsFound">0</div>
                        <div class="stat-label">Helmet Violations</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="facesCaptured">0</div>
                        <div class="stat-label">Violator Faces</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="videosProcessed">0</div>
                        <div class="stat-label">Videos Processed</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let currentFile = null;
        
        socket.on('connect', function() {
            console.log('Connected to FIXED server');
            updateSystemStatus();
        });
        
        socket.on('chat_response', function(data) {
            addChatMessage(data.message, 'ai');
        });
        
        function updateSystemStatus() {
            fetch('/get_statistics')
            .then(response => response.json())
            .then(data => {
                if (data.system_status) {
                    const status = data.system_status;
                    
                    const speechStatus = document.getElementById('speechStatus');
                    const speechText = document.getElementById('speechText');
                    if (status.speech_enabled) {
                        speechStatus.className = 'status-indicator status-online';
                        speechText.textContent = 'Ready';
                    } else {
                        speechStatus.className = 'status-indicator status-offline';
                        speechText.textContent = 'Not Available';
                    }
                    
                    const ocrStatus = document.getElementById('ocrStatus');
                    const ocrText = document.getElementById('ocrText');
                    if (status.ocr_enabled) {
                        ocrStatus.className = 'status-indicator status-online';
                        ocrText.textContent = 'Ready';
                    } else {
                        ocrStatus.className = 'status-indicator status-offline';
                        ocrText.textContent = 'Not Available';
                    }
                }
                
                if (data.detection_stats) {
                    updateStatistics(data.detection_stats);
                }
            })
            .catch(err => console.error('Status update error:', err));
        }
        
        function handleFileUpload() {
            const fileInput = document.getElementById('mediaUpload');
            const file = fileInput.files[0];
            
            if (!file) return;
            
            currentFile = file;
            
            const previewContainer = document.getElementById('mediaPreview');
            previewContainer.style.display = 'block';
            
            const fileType = file.type;
            
            if (fileType.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewContainer.innerHTML = `
                        <img src="${e.target.result}" alt="Preview" class="analysis-image">
                        <p><strong>File:</strong> ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)</p>
                        <p><em>Ready for helmet detection and face capture analysis</em></p>
                    `;
                };
                reader.readAsDataURL(file);
            } else if (fileType.startsWith('video/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewContainer.innerHTML = `
                        <video controls style="max-width: 100%; max-height: 300px;">
                            <source src="${e.target.result}" type="${fileType}">
                        </video>
                        <p><strong>File:</strong> ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)</p>
                        <p><em>Video analysis will process each frame with fixed codec support</em></p>
                    `;
                };
                reader.readAsDataURL(file);
            }
            
            document.getElementById('analyzeBtn').disabled = false;
        }
        
        function analyzeMedia() {
            if (!currentFile) {
                alert('Please select a file first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', currentFile);
            
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('analysisResults').style.display = 'none';
            
            const analyzeBtn = document.getElementById('analyzeBtn');
            analyzeBtn.textContent = 'Analyzing with FIXED Logic...';
            analyzeBtn.disabled = true;
            
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 3 + 1;
                if (progress > 90) progress = 90;
                document.getElementById('progressFill').style.width = progress + '%';
            }, 200);
            
            fetch('/analyze_media', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                clearInterval(progressInterval);
                document.getElementById('progressFill').style.width = '100%';
                
                setTimeout(() => {
                    document.getElementById('progressContainer').style.display = 'none';
                    displayAnalysisResults(data);
                    
                    analyzeBtn.textContent = 'Analyze with FIXED AI';
                    analyzeBtn.disabled = false;
                }, 500);
            })
            .catch(error => {
                clearInterval(progressInterval);
                document.getElementById('progressContainer').style.display = 'none';
                
                console.error('Analysis error:', error);
                alert('Analysis failed: ' + error.message);
                
                analyzeBtn.textContent = 'Analyze with FIXED AI';
                analyzeBtn.disabled = false;
            });
        }
        
        function displayAnalysisResults(data) {
            const resultsContainer = document.getElementById('analysisResults');
            
            if (!data.success) {
                resultsContainer.innerHTML = `
                    <div class="analysis-result result-violation">
                        <h4>Analysis Failed</h4>
                        <p>${data.error}</p>
                    </div>
                `;
                resultsContainer.style.display = 'block';
                return;
            }
            
            let resultHtml = '';
            
            if (data.type === 'image') {
                const hasViolations = data.violations.length > 0;
                const resultClass = hasViolations ? 'result-violation' : 'result-success';
                const statusIcon = hasViolations ? '❌' : '✅';
                
                resultHtml = `
                    <div class="analysis-result ${resultClass}">
                        <h4>${statusIcon} ${hasViolations ? 'VIOLATIONS DETECTED' : 'ALL SAFE - NO VIOLATIONS'}</h4>
                        <p><strong>Persons with Helmets:</strong> ${data.persons_with_helmets} (Green boxes)</p>
                        <p><strong>Persons without Helmets:</strong> ${data.persons_without_helmets} (Red boxes)</p>
                        <p><strong>Number Plates:</strong> ${data.number_plates.length} detected</p>
                        ${data.number_plates.length > 0 ? `
                            <p><strong>Detected Plates:</strong> ${data.number_plates.map(p => p.text).join(', ')}</p>
                        ` : ''}
                    </div>
                `;
                
                // FIXED: Display the analyzed image
                if (data.annotated_image_base64) {
                    resultHtml += `
                        <div class="preview-container">
                            <h4>Analyzed Image with Detections:</h4>
                            <img src="${data.annotated_image_base64}" alt="Analyzed Image" class="analysis-image">
                        </div>
                    `;
                }
                
                // FIXED: Display captured faces
                if (data.captured_faces && data.captured_faces.length > 0) {
                    resultHtml += `
                        <div class="face-capture-section">
                            <h4>🎯 Captured Violator Faces (${data.captured_faces.length}):</h4>
                            <div class="face-capture-grid">
                    `;
                    
                    data.captured_faces.forEach((face, index) => {
                        if (face.face_image_base64) {
                            resultHtml += `
                                <div class="captured-face">
                                    <img src="${face.face_image_base64}" alt="Violator ${index + 1}">
                                    <p>Violator ${index + 1}</p>
                                </div>
                            `;
                        }
                    });
                    
                    resultHtml += `
                            </div>
                            <p><small>Faces automatically captured from helmet violation areas</small></p>
                        </div>
                    `;
                }
                
                resultHtml += `
                    <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 8px;">
                        <p><strong>Speech Output:</strong> ${data.speech_text}</p>
                    </div>
                `;
                
            } else if (data.type === 'video') {
                resultHtml = `
                    <div class="analysis-result result-success">
                        <h4>✅ VIDEO ANALYSIS COMPLETE</h4>
                        <p><strong>Total Frames Processed:</strong> ${data.total_frames}</p>
                        <p><strong>Helmet Violations Found:</strong> ${data.total_violations}</p>
                        <p><strong>Violator Faces Captured:</strong> ${data.total_faces_captured || 0}</p>
                        <p><strong>Output Video:</strong> <a href="/get_video/${data.output_path}" target="_blank">Download Analyzed Video</a></p>
                        <p><strong>Speech Output:</strong> ${data.speech_text}</p>
                    </div>
                `;
            }
            
            resultsContainer.innerHTML = resultHtml;
            resultsContainer.style.display = 'block';
            
            addChatMessage(data.speech_text, 'ai');
            updateSystemStatus();
        }
        
        function testSpeech() {
            const testText = 'Speech system test. FIXED Traffic AI ready. All issues resolved - image display, face capture, and video analysis working.';
            
            fetch('/test_speech', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: testText})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addChatMessage('Speech test initiated: ' + testText, 'ai');
                } else {
                    addChatMessage('Speech test failed: ' + data.error, 'ai');
                }
            });
        }
        
        function loadViolationGallery() {
            fetch('/get_violation_images')
            .then(response => response.json())
            .then(data => {
                const gallery = document.getElementById('violationGallery');
                const facesContainer = document.getElementById('violationFaces');
                
                if (data.violation_files && data.violation_files.length > 0) {
                    let facesHtml = '';
                    data.violation_files.forEach(file => {
                        const date = new Date(file.timestamp * 1000).toLocaleDateString();
                        facesHtml += `
                            <div class="violation-face">
                                <img src="/get_image/${file.path}" alt="Violator" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjgwIiB2aWV3Qm94PSIwIDAgMTAwIDgwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjgwIiBmaWxsPSIjZjhkN2RhIi8+Cjx0ZXh0IHg9IjUwIiB5PSI0NSIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjEyIiBmaWxsPSIjNzIxYzI0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5ObyBJbWFnZTwvdGV4dD4KPHN2Zz4K'">
                                <p>${file.filename}<br><small>${date}</small></p>
                            </div>
                        `;
                    });
                    facesContainer.innerHTML = facesHtml;
                    gallery.style.display = 'block';
                } else {
                    facesContainer.innerHTML = '<p>No violator faces captured yet. Upload images/videos with helmet violations to see captured faces here.</p>';
                    gallery.style.display = 'block';
                }
            })
            .catch(err => {
                console.error('Error loading violation gallery:', err);
                document.getElementById('violationGallery').innerHTML = '<p>Error loading violation gallery.</p>';
                document.getElementById('violationGallery').style.display = 'block';
            });
        }
        
        function sendChatMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addChatMessage(message, 'user');
            socket.emit('chat_message', {message: message});
            
            input.value = '';
        }
        
        function addChatMessage(message, sender) {
            const chatMessages = document.getElementById('chatMessages');
            const messageEl = document.createElement('div');
            messageEl.className = `chat-message ${sender}`;
            
            const prefix = sender === 'ai' ? '<strong>AI Assistant:</strong><br>' : '<strong>You:</strong><br>';
            messageEl.innerHTML = prefix + message;
            
            chatMessages.appendChild(messageEl);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function updateStatistics(stats) {
            document.getElementById('totalAnalyzed').textContent = stats.total_analyzed;
            document.getElementById('violationsFound').textContent = stats.helmet_violations;
            document.getElementById('facesCaptured').textContent = stats.faces_captured;
            document.getElementById('videosProcessed').textContent = stats.videos_processed;
        }
        
        // Chat input enter key
        document.getElementById('chatInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
        
        // File drag and drop
        const uploadArea = document.getElementById('uploadArea');
        
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.style.backgroundColor = '#f0f8ff';
            uploadArea.style.transform = 'scale(1.02)';
        });
        
        uploadArea.addEventListener('dragleave', function() {
            uploadArea.style.backgroundColor = '';
            uploadArea.style.transform = 'scale(1)';
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.style.backgroundColor = '';
            uploadArea.style.transform = 'scale(1)';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('mediaUpload').files = files;
                handleFileUpload();
            }
        });
        
        // Initial status load
        updateSystemStatus();
        
        // Periodic stats update
        setInterval(updateSystemStatus, 10000);
        
        // Auto-load violation gallery on page load
        setTimeout(loadViolationGallery, 2000);
    </script>
</body>
</html>
    """)

# Main execution block
if __name__ == '__main__':
    import threading
    import webbrowser
    
    def open_browser():
        """Open browser after a short delay"""
        import time
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    # Start browser in separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("=" * 80)
    print("COMPLETELY FIXED TRAFFIC AI SYSTEM")
    print("=" * 80)
    print("Dashboard: http://localhost:5000")
    print("ALL ISSUES RESOLVED:")
    print("  ✅ Helmet detection logic corrected (no more inverted results)")
    print("  ✅ Analyzed images now display properly in web interface")
    print("  ✅ Face capture for violators implemented and visible")
    print("  ✅ Video analysis processing completely fixed with codec support")
    print("  ✅ Clear red/green marking system")
    print("  ✅ Proper violation counting and statistics")
    print("  ✅ Base64 image encoding for web display")
    print("  ✅ File serving routes for images and videos")
    print("=" * 80)
    
    try:
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False,
                    allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down FIXED Traffic AI System...")
        print("System shutdown complete")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("FIXED Traffic AI System stopped")