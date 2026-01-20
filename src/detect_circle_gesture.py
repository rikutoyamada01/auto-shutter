import cv2
import math
import mediapipe as mp
from profiler import profiler

class CircleGestureDetector:
    def __init__(self, 
                 static_image_mode=False, 
                 model_complexity=0, # 0=Lite, 1=Full, 2=Heavy. 0 is best for Pi.
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        """
        MediaPipe Pose initialized with lightweight settings for Raspberry Pi.
        model_complexity=0 is the fastest.
        """
        self.mp_pose = mp.solutions.pose # type: ignore
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils # type: ignore
        self.mp_drawing_styles = mp.solutions.drawing_styles # type: ignore

    def _get_landmarks(self, frame):
        """
        Internal method to process frame and get landmarks.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with profiler.measure("mediapipe_inference"):
            results = self.pose.process(frame_rgb)
        return results

    def detect(self, frame):
        """
        Detects the 'Circle' (Maru) gesture.
        Returns:
            draw_frame: Frame with landmarks drawn
            detected_flag: Boolean, True if gesture is detected
        """
        results = self._get_landmarks(frame)
        detected_flag = False
        draw_frame = frame.copy()
        
        if results.pose_landmarks:
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                draw_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Get key landmarks
            landmarks = results.pose_landmarks.landmark
            
            # MediaPipe Pose Landmarks Indices:
            # 11: left_shoulder, 12: right_shoulder
            # 13: left_elbow, 14: right_elbow
            # 15: left_wrist, 16: right_wrist
            
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            l_elbow = landmarks[13]
            r_elbow = landmarks[14]
            l_wrist = landmarks[15]
            r_wrist = landmarks[16]

            # Visibility check (confidence)
            if (l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5 and
                l_elbow.visibility > 0.5 and r_elbow.visibility > 0.5 and
                l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5):
                
                # Logic: Y coordinate increases downwards (0 is top)
                cond_wrists_above_elbows = (l_wrist.y < l_elbow.y) and (r_wrist.y < r_elbow.y)
                cond_elbows_above_shoulders = (l_elbow.y < l_shoulder.y) and (r_elbow.y < r_shoulder.y)
                
                # Distance calculation
                h, w, _ = frame.shape
                lw_px = (int(l_wrist.x * w), int(l_wrist.y * h))
                rw_px = (int(r_wrist.x * w), int(r_wrist.y * h))
                ls_px = (int(l_shoulder.x * w), int(l_shoulder.y * h))
                rs_px = (int(r_shoulder.x * w), int(r_shoulder.y * h))
                
                wrist_dist = math.hypot(lw_px[0] - rw_px[0], lw_px[1] - rw_px[1])
                shoulder_width = math.hypot(ls_px[0] - rs_px[0], ls_px[1] - rs_px[1])
                
                # Threshold: 1.2x shoulder width
                cond_wrists_close = wrist_dist < (shoulder_width * 1.2)

                if cond_wrists_above_elbows and cond_elbows_above_shoulders and cond_wrists_close:
                    detected_flag = True
                    cv2.putText(draw_frame, "MARU (CIRCLE) DETECTED!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.line(draw_frame, lw_px, rw_px, (0, 0, 255), 4)

        return draw_frame, detected_flag

    def detect_edge_proximity(self, frame, margin: int):
        """
        Checks if the detected person is too close to the left or right edge.
        Returns:
            draw_frame: Frame with bounding box and warning
            is_at_edge: Boolean, True if person touches the margin
        """
        results = self._get_landmarks(frame)
        draw_frame = frame.copy()
        is_at_edge = False
        
        h, w, _ = frame.shape
        
        if results.pose_landmarks:
            # Calculate bounding box of all visible landmarks
            min_x = w
            max_x = 0
            min_y = h
            max_y = 0
            has_visible_landmarks = False

            for lm in results.pose_landmarks.landmark:
                if lm.visibility > 0.5:
                    has_visible_landmarks = True
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    
                    min_x = min(min_x, px)
                    max_x = max(max_x, px)
                    min_y = min(min_y, py)
                    max_y = max(max_y, py)

            if has_visible_landmarks:
                edges = set()
                # Check margins
                if min_x < margin:
                    edges.add("LEFT")
                if max_x > (w - margin):
                    edges.add("RIGHT")
                if min_y < margin:
                    edges.add("TOP")
                
                # Check for "FAR" (Approach logic)
                # Calculate bounding box width ratio
                box_width = max_x - min_x
                if box_width > 0:
                    width_ratio = box_width / w
                    # Threshold: if person takes up less than 30% of screen width, they are too far.
                    if width_ratio < 0.3:
                         edges.add("FAR")

                if edges:
                    is_at_edge = True
                    color = (0, 0, 255) # Red
                    label = f"STATUS: {', '.join(edges)}"
                else:
                    is_at_edge = False
                    color = (0, 255, 0) # Green
                    label = "OK"

                # Draw bounding box
                cv2.rectangle(draw_frame, (min_x, min_y), (max_x, max_y), color, 2)
                cv2.putText(draw_frame, label, (min_x, min_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                 # No landmarks visible
                 edges = set()
                 is_at_edge = False

        else:
             edges = set()
             is_at_edge = False

        # Draw margin lines
        cv2.line(draw_frame, (margin, 0), (margin, h), (200, 200, 200), 1)
        cv2.line(draw_frame, (w - margin, 0), (w - margin, h), (200, 200, 200), 1)
        
        # Draw Top Margin line for visual feedback
        cv2.line(draw_frame, (0, margin), (w, margin), (200, 200, 200), 1)

        return draw_frame, edges

# --- Test Main ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = CircleGestureDetector()
    print("'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        # Test both methods alternatively or based on key
        # For simple test, let's show edge detection
        result_frame, is_edge = detector.detect_edge_proximity(frame, 50)
        
        if is_edge:
             cv2.putText(result_frame, "WARNING: EDGE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        cv2.imshow("MediaPipe Pose", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
