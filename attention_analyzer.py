from collections import deque
import time
import threading
import winsound

GAZE_THRESHOLD    = 0.40
ATTENTION_WINDOW  = 60
BLINK_FRAMES      = 2
DROWSY_CLOSED_FRAMES = 20   # Bu kadar frame kapalı kalırsa uyuklama

# EAR eşiği kalibrasyon ile dinamik olarak ayarlanır
DEFAULT_EAR_THRESHOLD = 0.22
DEFAULT_DROWSY_EAR    = 0.17


class AttentionAnalyzer:
    def __init__(self):
        self.blink_count   = 0
        self.closed_frames = 0

        # Bireysel göz
        self.left_closed_frames  = 0
        self.right_closed_frames = 0

        self.gaze_off_count = 0
        self.total_frames   = 0

        self.ear_history      = deque(maxlen=15)
        self.attention_scores = deque(maxlen=300)

        self.start_time      = time.time()
        self.last_blink_time = time.time()
        self.blink_times     = deque(maxlen=50)

        self.alert_level   = 0
        self.alert_message = ""

        # Kalibrasyon
        self.calibrating        = True
        self.calib_frames       = 0
        self.calib_ear_samples  = []
        self.CALIB_DURATION     = 60   # frame sayısı
        self.EAR_THRESHOLD      = DEFAULT_EAR_THRESHOLD
        self.DROWSY_EAR         = DEFAULT_DROWSY_EAR

        # Ses uyarısı
        self._sound_playing = False

    # ------------------------------------------------------------------ #

    def update(self, eye_data: dict) -> dict:
        avg_ear   = eye_data["avg_ear"]
        left_ear  = eye_data["left_ear"]
        right_ear = eye_data["right_ear"]
        gaze_h    = eye_data["gaze_h"]
        gaze_v    = eye_data["gaze_v"]

        self.ear_history.append(avg_ear)
        self.total_frames += 1

        # --- Kalibrasyon ---
        if self.calibrating:
            result = self._calibrate(avg_ear, eye_data)
            return result

        thr = self.EAR_THRESHOLD

        # Engel kontrolü
        obstacle = self._check_obstacle(eye_data)

        # --- Göz kapalılık ---
        avg_closed  = avg_ear  < thr
        left_closed = left_ear < thr
        right_closed= right_ear< thr

        if avg_closed:
            self.closed_frames += 1
        else:
            if self.closed_frames >= BLINK_FRAMES:
                self.blink_count += 1
                now = time.time()
                self.blink_times.append(now)
                self.last_blink_time = now
            self.closed_frames = 0

        self.left_closed_frames  = self.left_closed_frames  + 1 if left_closed  else 0
        self.right_closed_frames = self.right_closed_frames + 1 if right_closed else 0

        # Bakış yönü
        gaze_off = abs(gaze_h) > GAZE_THRESHOLD or abs(gaze_v) > GAZE_THRESHOLD
        if gaze_off:
            self.gaze_off_count += 1

        score = self._calculate_attention(avg_ear, gaze_h, gaze_v)
        self.attention_scores.append(score)

        alert_level, alert_message = self._update_alert(
            avg_ear, left_closed, right_closed, obstacle
        )

        # Uyuklama sesi
        if alert_level == 2:
            self._trigger_sound()

        elapsed = int(time.time() - self.start_time)
        bpm     = self._blinks_per_minute()

        return {
            "blink_count":      self.blink_count,
            "blinks_per_minute": bpm,
            "attention_score":  score,
            "avg_attention":    self._avg_attention(),
            "gaze_direction":   self._gaze_label(gaze_h, gaze_v),
            "alert_level":      alert_level,
            "alert_message":    alert_message,
            "elapsed_seconds":  elapsed,
            "is_eyes_closed":   self.closed_frames >= BLINK_FRAMES,
            "left_eye_closed":  self.left_closed_frames  >= BLINK_FRAMES,
            "right_eye_closed": self.right_closed_frames >= BLINK_FRAMES,
            "has_glasses":      eye_data.get("has_glasses", False),
            "obstacle":         obstacle,
            "calibrating":      False,
            "ear_threshold":    self.EAR_THRESHOLD,
        }

    # ------------------------------------------------------------------ #

    def _calibrate(self, avg_ear, eye_data):
        self.calib_frames += 1
        # İlk 10 frame'i ısınma olarak atla
        if self.calib_frames > 10:
            self.calib_ear_samples.append(avg_ear)

        if self.calib_frames >= self.CALIB_DURATION:
            if self.calib_ear_samples:
                # Açık göz EAR ortalamasının %72'si → eşik
                mean_ear = sum(self.calib_ear_samples) / len(self.calib_ear_samples)
                self.EAR_THRESHOLD = round(mean_ear * 0.72, 3)
                self.DROWSY_EAR    = round(mean_ear * 0.58, 3)
            self.calibrating = False
            self.start_time  = time.time()
            self.last_blink_time = time.time()

        remaining = self.CALIB_DURATION - self.calib_frames
        return {
            "calibrating":      True,
            "calib_remaining":  max(0, remaining),
            "avg_ear_so_far":   avg_ear,
            "alert_level":      0,
            "alert_message":    "",
            "has_glasses":      eye_data.get("has_glasses", False),
            "obstacle":         False,
            "blink_count":      0,
            "blinks_per_minute": 0.0,
            "attention_score":  100,
            "avg_attention":    100,
            "gaze_direction":   "Kalibrasyon",
            "elapsed_seconds":  0,
            "is_eyes_closed":   False,
            "left_eye_closed":  False,
            "right_eye_closed": False,
            "ear_threshold":    self.EAR_THRESHOLD,
        }

    def _check_obstacle(self, eye_data) -> bool:
        left_ear  = eye_data["left_ear"]
        right_ear = eye_data["right_ear"]
        thr = self.EAR_THRESHOLD
        diff = abs(left_ear - right_ear)
        one_very_low = (left_ear < thr * 0.45) or (right_ear < thr * 0.45)
        other_normal = (left_ear > thr * 0.80) or (right_ear > thr * 0.80)
        return diff > 0.09 and one_very_low and other_normal

    def _calculate_attention(self, ear, gaze_h, gaze_v) -> int:
        score = 100
        if ear < self.DROWSY_EAR:
            score -= 50
        elif ear < self.EAR_THRESHOLD:
            score -= 25
        gaze_penalty = (abs(gaze_h) + abs(gaze_v)) * 30
        score -= min(int(gaze_penalty), 40)
        if self._blinks_per_minute() > 25:
            score -= 10
        return max(0, min(100, score))

    def _blinks_per_minute(self) -> float:
        now = time.time()
        return len([t for t in self.blink_times if now - t <= 60])

    def _avg_attention(self) -> int:
        if not self.attention_scores:
            return 100
        return int(sum(self.attention_scores) / len(self.attention_scores))

    def _gaze_label(self, h, v) -> str:
        if abs(h) <= GAZE_THRESHOLD and abs(v) <= GAZE_THRESHOLD:
            return "Merkez"
        parts = []
        if v < -GAZE_THRESHOLD:
            parts.append("Yukari")
        elif v > GAZE_THRESHOLD:
            parts.append("Asagi")
        if h < -GAZE_THRESHOLD:
            parts.append("Sol")
        elif h > GAZE_THRESHOLD:
            parts.append("Sag")
        return "-".join(parts)

    def _update_alert(self, _ear, left_closed, right_closed, obstacle):
        time_since_blink = time.time() - self.last_blink_time

        if obstacle:
            return 1, "UYARI: Gozlerde engel var!"

        # Uyuklama: uzun sure kapalı
        if self.closed_frames >= DROWSY_CLOSED_FRAMES:
            return 2, "!! UYUKLAMA TESPIT EDILDI !!"

        # Bireysel göz kapalı uyarısı
        if self.left_closed_frames >= 8 and not right_closed:
            return 1, "Sol goz kapali"
        if self.right_closed_frames >= 8 and not left_closed:
            return 1, "Sag goz kapali"

        if time_since_blink > 8:
            return 1, "Dikkat: Uzun suredir blink yok"

        if self._avg_attention() < 50:
            return 1, "Dikkat skoru dusuk"

        return 0, ""

    def _trigger_sound(self):
        if self._sound_playing:
            return
        self._sound_playing = True
        def beep():
            for _ in range(4):
                winsound.Beep(1200, 250)
                time.sleep(0.08)
            self._sound_playing = False
        threading.Thread(target=beep, daemon=True).start()
