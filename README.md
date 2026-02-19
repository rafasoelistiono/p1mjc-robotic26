# P2MJC Notes (MuJoCo Unitree G1) — Progress 2

Progress ini melanjutkan fondasi Progress 1, tapi fokusnya sekarang adalah **transisi berdiri** dan **jalan (0.11 → 0.23)**. Target akhir: **robot bisa berdiri dan berjalan**.

---
## 1. Interpolasi untuk Gerakan Smooth

Progress ini memperkenalkan konsep **interpolasi pose**, supaya perpindahan antar pose terasa halus dan realistis.

Gunakan fungsi berikut:

```python
def smoothstep(progress):
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress * progress * (3.0 - 2.0 * progress)

def interpolate_pose(start_pose, target_pose, progress):
    blended = {}
    for joint in JOINT_ACTS:
        start = start_pose[joint]
        end = target_pose.get(joint, start)
        blended[joint] = start + progress * (end - start)
    return blended
```

**Ide utama:**

* `smoothstep()` membuat progress berubah halus (ease-in/ease-out)
* `interpolate_pose()` mem-blend sudut joint dari `start_pose → target_pose`

---

## 2. Referensi Gerakan

Gunakan referensi ini untuk:

* **transisi berdiri** 
    - Stand transition: `stand_transition.mp4`

[<img src="stand_transition.GIF" width="960px">](stand_transition.mp4)
* **ritme langkah berjalan**
    - Walking pose : `walking_pose.mp4`

[<img src="https://oss-global-cdn.unitree.com/static/244cd5c4f823495fbfb67ef08f56aa33.GIF" width="960px">](https://oss-global-cdn.unitree.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4)
---

## 3. Implementasi yang Diharapkan (Struktur Umum)

### Transisi Berdiri (0.10 → 0.14)

**Hint :**

* secara bertahap ubah:

  * `left_hip_pitch_joint`, `right_hip_pitch_joint` 
  * cocokan gerakan dengan video `assets/stand_transition.mp4` (Sekedar referensi)


### Berjalan (0.11 → 0.23)

Contoh gaya urutan di fungsi main:

```python
goto(viewer, STAND_READY,        0.30, label="w_00_ready")
goto(viewer, SHIFT_WEIGHT_LEFT,  0.25, label="w_01_shift_L")
goto(viewer, RIGHT_FOOT_LIFT,    0.20, label="w_02_R_lift")
goto(viewer, RIGHT_SWING,        0.20, label="w_03_R_swing")
goto(viewer, RIGHT_FOOT_DOWN,    0.20, label="w_04_R_down")
goto(viewer, SHIFT_WEIGHT_RIGHT, 0.25, label="w_05_shift_R")
goto(viewer, LEFT_FOOT_LIFT,     0.20, label="w_06_L_lift")
goto(viewer, LEFT_SWING,         0.20, label="w_07_L_swing")
goto(viewer, LEFT_FOOT_DOWN,     0.20, label="w_08_L_down")
```

> Timing di atas hanya contoh baseline. Silakan sesuaikan agar total segmen terasa mirip video 0.11–0.23.

---

## 4. Target Progress (Progress 2)

### Wajib: `0.11 – 0.23`

Output minimal yang diharapkan:

* bisa melakukan **transisi berdiri** pada `0.10 – 0.14`
* robot **berdiri stabil** 
* robot **berjalan** (langkah maju kanan kiri) (2 langkah)

> Hint : output diatas tidak perlu mirip seperti pada video namun diharapkan untuk bisa melanjutkan progress sebelumnya + melakukan aktivitas bangun lalu berdiri + berjalan


---

## 5. Rubrik Penilaian (Matriks Capaian)

| Komponen | Deskripsi                                                                   | Bobot |
| -------- | --------------------------------------------------------------------------- | ----- |
| 1        | Mampu mengetahui joint dan actuator yang diperlukan                         | 25%   |
| 2        | Mampu manipulasi gerakan dan memperhitungkan sudut gerakan (radian)         | 25%   |
| 3        | Mampu membuat gerakan mulus dengan fungsi `smooth` | 25%   |
| 4        | Membuat dokumentasi yang baik                                               | 25%   |


