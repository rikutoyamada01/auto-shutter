# -*- coding: utf-8 -*-
import cv2
import numpy as np

class FixedBackgroundSubtractor:
    """
    固定された単一の背景画像と比較して差分を検出するクラス。
    """
    def __init__(self, blur_ksize=(31, 31), threshold_val=50):
        """
        :param blur_ksize: ガウシアンブラーのカーネルサイズ
        :param threshold_val: 二値化の閾値
        """
        self.blur_ksize = blur_ksize
        self.threshold_val = threshold_val
        self.background_gray = None

    def set_background(self, background_frame):
        """
        比較の基準となる背景画像を設定します。
        :param background_frame: 背景として設定するカラーフレーム
        """
        gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
        self.background_gray = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        print("固定背景を設定しました。")

    def get_foreground_mask(self, frame):
        """
        現在のフレームと固定背景を比較し、前景マスク（動きがあった部分）を取得します。
        :param frame: 現在のカラーフレーム
        :return: 前景マスク (二値化画像)
        """
        if self.background_gray is None:
            raise ValueError("背景が設定されていません。set_background()を先に呼び出してください。")

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, self.blur_ksize, 0)
        
        frame_delta = cv2.absdiff(self.background_gray, current_gray)
        thresh = cv2.threshold(frame_delta, self.threshold_val, 255, cv2.THRESH_BINARY)[1]
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        return thresh

class AdaptiveBackgroundSubtractor:
    """
    背景を少しずつ更新していくことで、照明の変化などに対応するクラス。
    """
    def __init__(self, alpha=0.02, blur_ksize=(31, 31), threshold_val=50):
        """
        :param alpha: 背景モデルの更新率 (小さいほどゆっくり更新)
        :param blur_ksize: ガウシアンブラーのカーネルサイズ
        :param threshold_val: 二値化の閾値
        """
        self.alpha = alpha
        self.blur_ksize = blur_ksize
        self.threshold_val = threshold_val
        self.background_model = None

    def initialize_background(self, initial_frame):
        """
        最初のフレームで背景モデルを初期化します。
        :param initial_frame: 最初のカラーフレーム
        """
        gray_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, self.blur_ksize, 0)
        self.background_model = gray_frame.astype("float")
        print("適応的背景モデルを初期化しました。")

    def get_foreground_mask(self, frame):
        """
        現在のフレームから前景マスク（動きがあった部分）を取得し、背景モデルを更新します。
        :param frame: 現在のカラーフレーム
        :return: 前景マスク (二値化画像)
        """
        if self.background_model is None:
            raise ValueError("背景モデルが初期化されていません。initialize_background()を先に呼び出してください。")

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, self.blur_ksize, 0)

        # 背景モデルをゆっくり更新
        cv2.accumulateWeighted(current_gray, self.background_model, self.alpha)

        # 比較のために背景モデルをuint8に変換
        background_gray = cv2.convertScaleAbs(self.background_model)

        # 差分を計算
        frame_delta = cv2.absdiff(background_gray, current_gray)

        # 二値化
        thresh = cv2.threshold(frame_delta, self.threshold_val, 255, cv2.THRESH_BINARY)[1]

        # 膨張処理
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        return thresh
