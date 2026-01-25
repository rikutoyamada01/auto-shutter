# システムフローチャート (System Flowchart)

このドキュメントは、Auto-Shutterシステムの動作ロジックをMermaid形式で視覚化したものです。

## 状態遷移図

```mermaid
flowchart TD
    %% Define States
    Init([初期化 / Initialization])
    Ready[READY: 待機状態]
    PreAdjust[PRE_ADJUST: 準備時間 2s]
    Adjust[ADJUST: 位置自動調整 5s]
    TakePicture[TAKE_PICTURE: 撮影待機]
    Countdown[カウントダウン 3s]
    Capture[写真撮影 / Capture]
    Cooldown[PICTURE_COOLDOWN: クールダウン]
    Result[RESULT: 結果表示 20s]

    %% Initialization Flow
    Init -->|カメラ・AI・モーター設定| Ready

    %% READY State
    Ready -->|「丸」ジェスチャー検知| PreAdjust
    Ready -->|検知なし| Ready

    %% PRE_ADJUST State
    PreAdjust -->|2秒経過| Adjust

    %% ADJUST State
    Adjust -->|人物の位置判定| RobotAction{移動判定}
    RobotAction -->|近すぎる / 上端| MoveBack[後退]
    RobotAction -->|左端に寄っている| TurnLeft[左旋回]
    RobotAction -->|右端に寄っている| TurnRight[右旋回]
    RobotAction -->|遠すぎる| MoveFwd[前進]
    RobotAction -->|位置OK| WaitAdjust[待機]
    
    MoveBack --> CheckTimeAdjust{5秒経過?}
    TurnLeft --> CheckTimeAdjust
    TurnRight --> CheckTimeAdjust
    MoveFwd --> CheckTimeAdjust
    WaitAdjust --> CheckTimeAdjust

    CheckTimeAdjust -->|Yes| TakePicture
    CheckTimeAdjust -->|No| Adjust

    %% TAKE_PICTURE State
    TakePicture -->|「丸」ジェスチャー検知| Countdown
    TakePicture -->|30秒タイムアウト| Ready
    
    Countdown -->|3秒経過| Capture
    Capture -->|画像を保存| Cooldown

    %% PICTURE_COOLDOWN State
    Cooldown --> CheckCount{3枚撮影した?}
    CheckCount -->|No| TakePicture
    CheckCount -->|Yes| Result

    %% RESULT State
    Result -->|HTTPサーバー起動| ShowQR[QRコード表示]
    ShowQR -->|20秒経過| StopServer[サーバー停止]
    StopServer --> Ready

    %% Styling
    classDef state fill:#f9f,stroke:#333,stroke-width:2px;
    classDef decision fill:#ff9,stroke:#333,stroke-width:2px;
    classDef action fill:#9cf,stroke:#333,stroke-width:2px;

    class Ready,PreAdjust,Adjust,TakePicture,Cooldown,Result state;
    class CheckTimeAdjust,RobotAction,CheckCount decision;
    class MoveBack,TurnLeft,TurnRight,MoveFwd,Capture,ShowQR action;
```

## 各フェーズの説明

1.  **READY**: ユーザーがカメラの前で「丸」のポーズをするのを待機します。
2.  **PRE_ADJUST**: ポーズ検知後、ユーザーが手を下ろして自然な姿勢になるための猶予時間です。
3.  **ADJUST**: MediaPipeを使用して人物のバウンディングボックスを解析し、適切な構図になるようロボットが物理的に移動します。
4.  **TAKE_PICTURE**: 再度「丸」のポーズをすることで撮影を開始します。
5.  **RESULT**: 撮影した画像を共有するためのQRコードを生成・表示します。
