import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

class UIOverlay:
    def __init__(self):
        self.header_height = 40
        self.footer_height = 60
        
        # Cross-platform Japanese font detection
        self.font_path = self._find_japanese_font()
        self.default_font_size = 32
        
        # Load default font
        try:
            if self.font_path:
                self.font = ImageFont.truetype(self.font_path, self.default_font_size)
                print(f"[UI] Loaded font: {self.font_path}")
            else:
                print("[UI] No TrueType font found, using default font")
                self.font = ImageFont.load_default()
        except Exception as e:
            print(f"[UI] Font loading failed: {e}, using default font")
            self.font = ImageFont.load_default()
    
    def _find_japanese_font(self):
        """Find a suitable Japanese font for the current platform"""
        font_candidates = []
        
        if sys.platform == "win32":
            # Windows fonts
            font_candidates = [
                "C:/Windows/Fonts/meiryo.ttc",
                "C:/Windows/Fonts/msgothic.ttc",
                "C:/Windows/Fonts/msmincho.ttc",
                "C:/Windows/Fonts/YuGothM.ttc",
            ]
        else:
            # Linux/Raspberry Pi fonts
            font_candidates = [
                "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                "/usr/share/fonts/truetype/fonts-japanese-mincho.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
                "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Fallback non-Japanese
            ]
        
        # Find first existing font
        for font in font_candidates:
            if os.path.exists(font):
                return font
        
        return None

    def draw_text_jp(self, frame, text, pos, size, color, outline_color=(0, 0, 0), thickness=2):
        """
        Draws text using PIL to support Japanese characters.
        """
        if not text:
            return frame

        # Convert to PIL
        # 1. OpenCV (BGR) -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Load font with specific size
        # Note: Loading font every time might be slow, but usually UI text count is low.
        # Optimization: Cache fonts of different sizes if needed. For now, simple load.
        font = None
        if self.font_path:
            try:
                font = ImageFont.truetype(self.font_path, size)
            except Exception as e:
                print(f"[UI] Failed to load font size {size}: {e}")
        
        if font is None:
            font = ImageFont.load_default()

        x, y = pos
        
        # Draw Outline (simulate by drawing multiple times)
        if thickness > 0:
            for dx in range(-thickness, thickness + 1):
                for dy in range(-thickness, thickness + 1):
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Main text
        draw.text((x, y), text, font=font, fill=color)

        # Convert back to OpenCV
        frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def draw_header(self, frame, mode_text, status_text=""):
        h, w = frame.shape[:2]
        
        # Mode (Left)
        self.draw_text_jp(frame, mode_text, (20, 20), 24, (0, 255, 0), thickness=2)
        
        # Status (Right)
        if status_text:
            # Need to measure text width to align right. PIL has getbbox or getlength
            # Just approximate layout or use simplified fixed position if measurement is complex
            # PIL getlength is available in newer versions. getbbox is safer.
            # Let's align roughly or use fixed offset? Right alignment requires width.
            
            # Simple right alignment logic
            dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            font = None
            if self.font_path:
                try:
                    font = ImageFont.truetype(self.font_path, 20)
                except:
                    pass
            
            if font:
                try:
                    text_bbox = dummy_draw.textbbox((0, 0), status_text, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                except:
                    text_w = 200  # Fallback estimate
            else:
                text_w = 200  # Fallback estimate

            x = w - text_w - 20
            self.draw_text_jp(frame, status_text, (x, 20), 20, (0, 0, 255), thickness=2)

    def draw_footer(self, frame, main_text, sub_text=None, progress=None):
        h, w = frame.shape[:2]

        # Main text (Center)
        if main_text:
            font_size = 32
            # Center alignment
            dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
            font = None
            if self.font_path:
                try:
                    font = ImageFont.truetype(self.font_path, font_size)
                except:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
            
            try:
                text_bbox = dummy_draw.textbbox((0, 0), main_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
            except:
                text_w = len(main_text) * 16  # Rough estimate
            
            x = (w - text_w) // 2
            y = h - 80
            if sub_text:
                 y = h - 100
            
            self.draw_text_jp(frame, main_text, (x, y), font_size, (255, 255, 255), thickness=2)
        
        # Sub text
        if sub_text:
             font_size = 20
             dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
             font = None
             if self.font_path:
                 try:
                     font = ImageFont.truetype(self.font_path, font_size)
                 except:
                     font = ImageFont.load_default()
             else:
                 font = ImageFont.load_default()
             
             try:
                 text_bbox = dummy_draw.textbbox((0, 0), sub_text, font=font)
                 text_w = text_bbox[2] - text_bbox[0]
             except:
                 text_w = len(sub_text) * 10  # Rough estimate
             
             x = (w - text_w) // 2
             self.draw_text_jp(frame, sub_text, (x, h - 40), font_size, (200, 200, 200), thickness=1)

        # Progress bar (Bottom edge) - CV2 is fine for shapes
        if progress is not None:
             bar_height = 20
             p = max(0.0, min(1.0, progress))
             
             cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)
             cv2.rectangle(frame, (0, h - bar_height), (int(w * p), h), (0, 255, 255), -1)

    def draw_center_status(self, frame, text, color=(0, 255, 0), scale=None, thickness=2):
        # Scale ignored, using fixed massive font size
        font_size = 60
        h, w = frame.shape[:2]
        
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        font = None
        if self.font_path:
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()
        
        try:
            text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except:
            text_w = len(text) * 30  # Rough estimate
            text_h = font_size
        
        x = (w - text_w) // 2
        y = (h - text_h) // 2
        
        self.draw_text_jp(frame, text, (x, y), font_size, color, thickness=3)

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
