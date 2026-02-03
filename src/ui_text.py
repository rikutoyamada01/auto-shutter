# -*- coding: utf-8 -*-
"""
UI text translations for Japanese and English
"""

class UIText:
    """UI text messages in Japanese and English"""
    
    @staticmethod
    def get_text(key, use_japanese=True):
        """Get UI text in Japanese or English"""
        texts = {
            # Ready state
            "ready_gesture": {
                "ja": "「マル」を作ってスタート",
                "en": "Make a Circle to Start"
            },
            "starting": {
                "ja": "開始します！",
                "en": "Starting!"
            },
            
            # Pre-adjust state
            "lower_hands": {
                "ja": "手を下ろしてください",
                "en": "Lower Your Hands"
            },
            
            # Adjust state
            "adjusting": {
                "ja": "位置調整中...",
                "en": "Adjusting Position..."
            },
            "approaching": {
                "ja": "近づいています...",
                "en": "Approaching..."
            },
            "too_close": {
                "ja": "近すぎます",
                "en": "Too Close"
            },
            "edge_top": {
                "ja": "上",
                "en": "Top"
            },
            "edge_left": {
                "ja": "左",
                "en": "Left"
            },
            "edge_right": {
                "ja": "右",
                "en": "Right"
            },
            
            # Take picture state
            "take_picture_gesture": {
                "ja": "「マル」を作って撮影",
                "en": "Make Circle to Capture"
            },
            "pose": {
                "ja": "ポーズをとって！",
                "en": "Strike a Pose!"
            },
            
            # Cooldown state
            "photo_taken": {
                "ja": "撮影完了！",
                "en": "Photo Taken!"
            },
            
            # Result state
            "complete": {
                "ja": "完了！ お疲れ様でした",
                "en": "Complete! Thank You!"
            },
            "thank_you": {
                "ja": "ご利用ありがとうございました",
                "en": "Thank You for Using"
            },
            
            # State names
            "state_ready": {
                "ja": "待機中",
                "en": "Ready"
            },
            "state_pre_adjust": {
                "ja": "準備中",
                "en": "Preparing"
            },
            "state_adjust": {
                "ja": "移動中",
                "en": "Adjusting"
            },
            "state_take_picture": {
                "ja": "撮影",
                "en": "Capture"
            },
            "state_cooldown": {
                "ja": "保存中",
                "en": "Saving"
            },
            "state_result": {
                "ja": "完了",
                "en": "Done"
            },
            
            # Other
            "remaining": {
                "ja": "残り",
                "en": "Remaining"
            },
            "seconds": {
                "ja": "秒",
                "en": "s"
            },
            "status": {
                "ja": "状態",
                "en": "Status"
            }
        }
        
        if key not in texts:
            return key
        
        lang = "ja" if use_japanese else "en"
        return texts[key].get(lang, key)
