import cv2
import numpy as np

class UIOverlay:
    def __init__(self):
        self.header_height = 40
        self.footer_height = 60

    def draw_text_with_outline(self, frame, text, pos, scale, color, thickness=2, outline_color=(0, 0, 0)):
        """Helper to draw text with a black outline for readability without background bars."""
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, outline_color, thickness + 3)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def draw_header(self, frame, mode_text, status_text=""):
        h, w = frame.shape[:2]
        # No background bar
        
        # Mode (Left) - slightly larger, Green for active feel
        self.draw_text_with_outline(frame, mode_text, (10, 35), 1.0, (0, 255, 0), 2)
        
        # Status (Right) - Red for warnings/timeout
        if status_text:
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            x = w - text_size[0] - 10
            self.draw_text_with_outline(frame, status_text, (x, 35), 0.8, (0, 0, 255), 2)

    def draw_footer(self, frame, main_text, sub_text=None, progress=None):
        h, w = frame.shape[:2]
        # No background bar

        # Main text (Center) - White with heavy outline
        if main_text:
            text_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 45
            if sub_text:
                text_y = h - 65
            
            self.draw_text_with_outline(frame, main_text, (text_x, text_y), text_scale, (255, 255, 255), thickness)
        
        # Sub text (Below main, smaller)
        if sub_text:
             text_scale = 0.6
             thickness = 1
             text_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
             text_x = (w - text_size[0]) // 2
             self.draw_text_with_outline(frame, sub_text, (text_x, h - 30), text_scale, (200, 200, 200), thickness)

        # Progress bar (Bottom edge) - Keep this as it's non-intrusive
        if progress is not None:
             bar_height = 20
             p = max(0.0, min(1.0, progress))
             
             # Draw background track for better contrast
             cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)
             
             # Draw progress
             cv2.rectangle(frame, (0, h - bar_height), (int(w * p), h), (0, 255, 255), -1)

    def draw_center_status(self, frame, text, color=(0, 255, 0), scale=2.0, thickness=3):
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        # Text Outline (Black)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4)
        # Main Text
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def draw_qr_result(self, frame, qr_image):
         """Draws QR code in the center of the screen."""
         if qr_image is None:
             return

         h, w = frame.shape[:2]
         
         # Resize QR to fit nicely (e.g., 50% of screen height)
         qr_h, qr_w = qr_image.shape[:2]
         target_h = int(h * 0.5)
         
         if qr_h > 0:
            scale = target_h / qr_h
            resized_qr = cv2.resize(qr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
            # Center overlay
            rh, rw = resized_qr.shape[:2]
            y_offset = (h - rh) // 2
            x_offset = (w - rw) // 2
            
            # Draw white background border for QR code (for readability)
            border = 10
            cv2.rectangle(frame, (x_offset - border, y_offset - border), 
                          (x_offset + rw + border, y_offset + rh + border), (255, 255, 255), -1)
            
            # Check bounds to safely copy
            if y_offset >= 0 and x_offset >= 0 and y_offset+rh <= h and x_offset+rw <= w:
                frame[y_offset:y_offset+rh, x_offset:x_offset+rw] = resized_qr
